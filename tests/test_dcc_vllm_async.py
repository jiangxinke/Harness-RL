"""Per-request scheduling, timeout cancellation and token provenance contracts."""

import asyncio
import sys
from types import SimpleNamespace

import pytest

from slime_plugins.dcc.hf_policy import encode_messages
from slime_plugins.dcc.vllm_policy import AsyncVLLMPolicy


class Tokenizer:
    eos_token_id = 1

    def apply_chat_template(self, messages, **kwargs):
        return [ord(c) + 3 for c in messages[-1]["content"]]

    def decode(self, tokens, **kwargs):
        return "".join(chr(c - 3) for c in tokens if c > 2)


class Engine:
    def __init__(self):
        self.waiters, self.calls, self.aborts = {}, [], []
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.shutdowns = 0

    async def generate(self, prompt, params, request_id, *, lora_request):
        self.calls.append((prompt, params, request_id, lora_request))
        self.waiters[request_id] = asyncio.Event()
        self.started.set()
        if Tokenizer().decode(prompt["prompt_token_ids"]) == "slow":
            await self.release.wait()
        yield SimpleNamespace(
            prompt_token_ids=prompt["prompt_token_ids"],
            outputs=[SimpleNamespace(token_ids=[ord(c) + 3 for c in "answer"] + [1], finish_reason="stop")],
        )

    async def abort(self, requests):
        requests = [requests] if isinstance(requests, str) else requests
        self.aborts.extend((request, asyncio.get_running_loop()) for request in requests)

    def shutdown(self):
        self.shutdowns += 1


@pytest.fixture(autouse=True)
def runtime(monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(
        SamplingParams=lambda **kwargs: SimpleNamespace(**kwargs)))
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", SimpleNamespace(
        RequestOutputKind=SimpleNamespace(FINAL_ONLY="final")))


def messages(text):
    return [{"role": "user", "content": text}]


def test_short_request_finishes_while_unrelated_long_request_is_running():
    async def run():
        engine = Engine()
        policy = AsyncVLLMPolicy(engine, Tokenizer())
        slow = asyncio.create_task(policy(messages("slow"), max_new_tokens=16, temperature=0))
        await engine.started.wait()
        fast = await asyncio.wait_for(policy(messages("fast"), max_new_tokens=16, temperature=0), .2)
        assert fast.text == "answer" and not slow.done()
        engine.release.set()
        assert (await slow).text == "answer"
        assert len({call[2] for call in engine.calls}) == 2
        await policy.close()
        assert not engine.aborts and engine.shutdowns == 1
    asyncio.run(run())


def test_async_explicit_sampling_matches_sync_configuration():
    from slime_plugins.dcc.eval_sampling import EvaluationSampling

    async def run():
        engine = Engine()
        sampling = EvaluationSampling(temperature=0.7, top_p=0.8, top_k=20, frequency_penalty=0.05)
        policy = AsyncVLLMPolicy(engine, Tokenizer(), evaluation_sampling=sampling, seed=47)
        result = await policy(messages("fast"), max_new_tokens=16, temperature=0.7)
        actual = engine.calls[0][1]
        assert {key: getattr(actual, key) for key in sampling.to_dict()} == sampling.to_dict()
        assert result.prompt_ids == encode_messages(Tokenizer(), messages("fast"), 16384)
        assert actual.seed == 47 and actual.output_kind == "final"
        with pytest.raises(ValueError, match="must match"):
            await policy(messages("fast"), max_new_tokens=16, temperature=0)
        assert len(engine.calls) == 1
        await policy.close()
    asyncio.run(run())


def test_async_structured_generation_preserves_native_requests_and_cancellation(monkeypatch):
    from slime_plugins.dcc import vllm_policy

    monkeypatch.setattr(vllm_policy, "validate_web_structured_schema", lambda schema: None)
    monkeypatch.setattr(sys.modules["vllm.sampling_params"], "StructuredOutputsParams",
                        lambda **kwargs: SimpleNamespace(**kwargs), raising=False)
    schema = {"type": "object"}

    async def run():
        engine = Engine()
        policy = AsyncVLLMPolicy(engine, Tokenizer(), web_structured_final=True)
        await policy(messages("fast"), max_new_tokens=16, temperature=0)
        assert not hasattr(engine.calls[-1][1], "structured_outputs")
        result = await policy.generate_structured(messages("fast"), json_schema=schema,
                                                  max_new_tokens=16, temperature=0)
        params = engine.calls[-1][1]
        assert params.structured_outputs.json is schema and params.structured_outputs.disable_fallback
        assert params.max_tokens == 16 and params.output_kind == "final"
        assert result.prompt_ids == encode_messages(Tokenizer(), messages("fast"), 16384)
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(policy.generate_structured(
                messages("slow"), json_schema=schema, max_new_tokens=16, temperature=0), .02)
        assert engine.aborts == [(engine.calls[-1][2], asyncio.get_running_loop())]
        assert not policy._active_requests
        await policy.close()

    asyncio.run(run())


def test_episode_timeout_aborts_only_its_request_and_later_work_succeeds():
    async def run():
        engine = Engine()
        policy = AsyncVLLMPolicy(engine, Tokenizer())
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(policy(messages("slow"), max_new_tokens=16, temperature=0), .02)
        assert engine.aborts == [(engine.calls[0][2], asyncio.get_running_loop())]
        assert not policy._active_requests
        result = await policy(messages("fast"), max_new_tokens=16, temperature=0)
        assert result.prompt_ids == encode_messages(Tokenizer(), messages("fast"), 16384)
        assert result.completion_ids[-1] == 1
        assert result.finish_reason == "stop" and result.logprobs is None
        await policy.close()
        await policy.close()
        assert engine.shutdowns == 1 and len(engine.aborts) == 1
        with pytest.raises(RuntimeError, match="closed"):
            await policy(messages("fast"), max_new_tokens=16, temperature=0)
    asyncio.run(run())


def test_async_policy_keeps_lora_and_sampling_budgets_and_rejects_mismatched_output():
    async def run():
        engine, adapter = Engine(), object()
        policy = AsyncVLLMPolicy(engine, Tokenizer(), lora_request=adapter,
                                 max_prompt_tokens=20, max_model_len=30)
        await policy(messages("fast"), max_new_tokens=26, temperature=0)
        _, params, _, actual_adapter = engine.calls[0]
        assert actual_adapter is adapter and params.max_tokens == 26
        assert params.output_kind == "final" and params.temperature == 0
        with pytest.raises(ValueError, match="refusing to shorten"):
            await policy(messages("fast"), max_new_tokens=27, temperature=0)
        with pytest.raises(ValueError, match="greedy evaluation"):
            await policy(messages("fast"), max_new_tokens=4, temperature=1)
        original = engine.generate
        async def corrupt(*args, **kwargs):
            async for row in original(*args, **kwargs):
                row.prompt_token_ids = [999]
                yield row
        engine.generate = corrupt
        with pytest.raises(RuntimeError, match="provenance"):
            await policy(messages("fast"), max_new_tokens=4, temperature=0)
        assert len(engine.aborts) == 1
        await policy.close()
    asyncio.run(run())
