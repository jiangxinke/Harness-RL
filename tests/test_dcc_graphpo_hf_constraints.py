"""Exercise the real LMFE token mask through HFPolicy's summary-only queue."""

import asyncio
import json
import re
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("lmformatenforcer")

from slime_plugins.dcc.hf_policy import HFPolicy
from slime_plugins.dcc.graphpo_runtime import summary_selection_schema, summary_selection_regex


class AsciiTokenizer:
    pad_token_id, eos_token_id = 0, 1
    all_special_ids = [0, 1, 2]

    def __len__(self):
        return 131

    def encode(self, text, **kwargs):
        return [ord(character) + 3 for character in text]

    def decode(self, ids, **kwargs):
        return "".join(chr(value - 3) for value in ids if value >= 3)

    def apply_chat_template(self, messages, **kwargs):
        return self.encode(json.dumps(messages))


class AdversarialGenerationModel:
    device = torch.device("cpu")

    def __init__(self):
        self.generation_config = SimpleNamespace(eos_token_id=1)
        self.calls = []

    def eval(self):
        return self

    def generate(self, *, input_ids, max_new_tokens, **kwargs):
        self.calls.append(kwargs)
        mask = kwargs.get("prefix_allowed_tokens_fn")
        if mask is None:
            return torch.cat((input_ids, torch.tensor([[ord("x") + 3, 1]] * len(input_ids))), dim=1)
        desired = AsciiTokenizer().encode('{"fact_ids":[19],"unresolved_ids":[19],"claim_ids":[19]}') + [1]
        rows = []
        for batch_id, prompt in enumerate(input_ids):
            row = prompt.tolist()
            for index in range(max_new_tokens):
                allowed = mask(batch_id, torch.tensor(row))
                preferred = desired[index] if index < len(desired) else 1
                selected = preferred if preferred in allowed else min(allowed)
                row.append(selected)
                if selected == 1:
                    break
            rows.append(row)
        width = max(map(len, rows))
        return torch.tensor([row + [0] * (width - len(row)) for row in rows])


def test_real_token_mask_blocks_cross_domain_id_and_empty_domain_population():
    schema = summary_selection_schema({"facts": [{"id": 19}], "unresolved": [{"id": 1}], "claims": []})
    model, tokenizer = AdversarialGenerationModel(), AsciiTokenizer()
    policy = HFPolicy(model, tokenizer)
    result = policy.generate([[{"role": "user", "content": "Select excerpts."}]], 512, 0,
                             selection_schema=schema)[0]
    assert re.fullmatch(summary_selection_regex(schema), result.text)
    value = json.loads(result.text)
    assert value["claim_ids"] == []
    assert all(index == 1 for index in value["unresolved_ids"])
    assert result.finish_reason == "stop"
    assert policy.summary_constraint_usage["calls"] == 1
    assert policy.summary_constraint_usage["setup_seconds"] > 0
    policy.generate([[{"role": "user", "content": "Ordinary actor."}]], 2, 1.0)
    assert "prefix_allowed_tokens_fn" not in model.calls[-1]


def test_queue_separates_different_summary_domains_and_ordinary_actor_requests():
    async def run():
        model = AdversarialGenerationModel()
        policy = HFPolicy(model, AsciiTokenizer(), batch_size=3)
        schemas = [summary_selection_schema({"facts": [{"id": index}], "unresolved": [], "claims": []})
                   for index in (1, 19)]
        messages = [{"role": "user", "content": "Select excerpts."}]
        try:
            results = await asyncio.gather(
                *(policy.generate_summary(messages, selection_schema=schema, max_new_tokens=512, temperature=0)
                  for schema in schemas),
                policy(messages, max_new_tokens=512, temperature=0),
            )
        finally:
            await policy.close()
        assert all(re.fullmatch(summary_selection_regex(schema), result.text)
                   for schema, result in zip(schemas, results))
        assert results[2].text == "x"
        assert len(model.calls) == 3
        assert sum("prefix_allowed_tokens_fn" in call for call in model.calls) == 2
    asyncio.run(run())
