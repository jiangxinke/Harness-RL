from __future__ import annotations

import asyncio
import json
import sys
from types import ModuleType, SimpleNamespace

import httpx
import pytest

from slime.utils.types import Sample
from slime_plugins.loongreflect.slime_adapter import generate


class ByteTokenizer:
    def __call__(self, text, **kwargs):
        return {"input_ids": list(text.encode())}

    def decode(self, tokens, **kwargs):
        return bytes(tokens).decode()

    def apply_chat_template(self, messages, **kwargs):
        return (
            "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages)
            + "<assistant>"
        )


@pytest.mark.parametrize(
    "failure", ["search", "search_invalid", "generation", "generation_invalid", None]
)
def test_real_rollout_masks_service_outages_and_keeps_empty_results_valid(
    monkeypatch, failure
):
    module = ModuleType("slime.rollout.sglang_rollout")
    tokenizer = ByteTokenizer()
    module.GenerateState = lambda args: SimpleNamespace(tokenizer=tokenizer)
    monkeypatch.setitem(sys.modules, "slime.rollout.sglang_rollout", module)
    calls = []

    def respond(request):
        if request.url.path == "/retrieve":
            if failure == "search":
                raise httpx.ConnectError("retriever unavailable", request=request)
            if failure == "search_invalid":
                return httpx.Response(200, json={"result": {}})
            return httpx.Response(200, json={"result": [[]]})
        if failure == "generation":
            raise httpx.ConnectError("generator unavailable", request=request)
        if failure == "generation_invalid":
            return httpx.Response(200, json={"text": ""})
        calls.append(json.loads(request.content))
        output = (
            "inspect</think><search>evidence</search>"
            if len(calls) == 1
            else "ready</think><answer>ok</answer>"
        )
        return httpx.Response(
            200,
            json={
                "text": output,
                "meta_info": {
                    "finish_reason": {"type": "stop"},
                    "output_token_logprobs": [
                        [-0.2, token, None] for token in output.encode()
                    ],
                },
            },
        )

    client = httpx.Client
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kwargs: client(transport=httpx.MockTransport(respond), **kwargs),
    )
    args = SimpleNamespace(
        partial_rollout=False,
        sglang_router_ip="generator",
        sglang_router_port=80,
        rollout_temperature=1.0,
        rollout_top_p=1.0,
        rollout_max_response_len=1024,
        rollout_max_context_len=4096,
        seq_length=4096,
        reflect_max_steps=3,
        reflect_search_url="http://search/retrieve",
    )
    samples = asyncio.run(
        generate(args, Sample(index=7, group_index=2, prompt="query", label="ok"), {})
    )
    assert samples and all(sample.rollout_id == 7 for sample in samples)
    if failure:
        assert all(
            sample.remove_sample
            and not any(sample.loss_mask)
            and not any(sample.reflect_mask)
            for sample in samples
        )
        assert all(
            sample.metadata["infrastructure_failure"]["retryable"] for sample in samples
        )
        assert all(sample.reward == 0.0 for sample in samples)
    else:
        assert len(samples) == 2
        assert all(
            not sample.remove_sample and any(sample.loss_mask) for sample in samples
        )
        assert all(
            "infrastructure_failure" not in sample.metadata for sample in samples
        )
        assert all("PRIVATE REFERENCE ANSWER" not in call["text"] for call in calls)
