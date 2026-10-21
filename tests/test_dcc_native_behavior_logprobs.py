"""Real tiny HF generation parity and strict streaming behavior boundaries."""
from __future__ import annotations

import math

import pytest
import torch
from transformers import GenerationConfig, LogitsProcessorList, Qwen3MoeConfig, Qwen3MoeForCausalLM

from slime_plugins.dcc.hf_native_policy import _explicit_generation_kwargs
from slime_plugins.dcc.native_behavior_logprobs import StreamingRawThoughtLogprobs


def tiny_model():
    torch.manual_seed(17)
    model = Qwen3MoeForCausalLM(Qwen3MoeConfig(
        vocab_size=32, hidden_size=16, intermediate_size=24, moe_intermediate_size=8,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        head_dim=8, num_experts=4, num_experts_per_tok=2, attention_dropout=0.0)).eval()
    return model


def generate(model, *, tracker, seed=37, eos=None, max_new_tokens=7):
    ids = torch.tensor([[2, 3, 4], [5, 6, 7], [2, 2, 8]])
    attention = torch.tensor([[0, 1, 1], [1, 1, 1], [0, 0, 1]])
    cfg = GenerationConfig(do_sample=True, temperature=1., top_p=1., top_k=0,
        pad_token_id=2, eos_token_id=eos, max_new_tokens=max_new_tokens,
        num_beams=1, num_return_sequences=1, min_length=0, early_stopping=False,
        repetition_penalty=1., renormalize_logits=False,
        return_dict_in_generate=True, output_logits=True, use_cache=True)
    torch.manual_seed(seed)
    with torch.inference_mode():
        result = model.generate(input_ids=ids, attention_mask=attention,
            logits_processor=LogitsProcessorList([tracker] if tracker is not None else []),
            **_explicit_generation_kwargs(cfg))
    state = torch.get_rng_state().clone()
    rows = result.sequences[:, ids.shape[1]:].tolist()
    if eos:
        rows = [row[:next((i + 1 for i, t in enumerate(row) if t in eos), len(row))] for row in rows]
    return result, rows, state


def assert_output_logits_parity(result, rows, actual):
    for b, row in enumerate(rows):
        oracle = [result.logits[t][b].float().log_softmax(-1)[token].item()
                  for t, token in enumerate(row)]
        assert len(actual[b]) == len(row)
        torch.testing.assert_close(torch.tensor(actual[b]), torch.tensor(oracle), rtol=0, atol=1e-6)


def test_real_tiny_hf_actual_sampled_scores_match_output_logits_without_changing_rng():
    model = tiny_model()
    reference, _, rng = generate(model, tracker=None)
    tracker = StreamingRawThoughtLogprobs()
    result, rows, tracked_rng = generate(model, tracker=tracker)
    assert torch.equal(reference.sequences, result.sequences)
    assert torch.equal(rng, tracked_rng)
    actual = tracker.finish(rows)
    assert_output_logits_parity(result, rows, actual)
    assert not torch.cuda.is_initialized()


def test_real_tiny_hf_left_padding_ragged_eos_and_cap_keep_only_actual_scores():
    model = tiny_model()
    with torch.no_grad():
        model.lm_head.weight.zero_()
    tracker = StreamingRawThoughtLogprobs()
    result, rows, _ = generate(model, tracker=tracker, seed=19, eos=[0, 1], max_new_tokens=32)
    actual = tracker.finish(rows)
    assert len({len(row) for row in rows}) > 1
    assert any(row[-1] in [0, 1] for row in rows)
    assert any(len(row) < result.sequences.shape[1] - 3 for row in rows)
    assert_output_logits_parity(result, rows, actual)
    for values in actual:
        assert values == pytest.approx([-math.log(32)] * len(values), abs=1e-6)


def test_tracker_returns_same_scores_and_keeps_only_pending_vocabulary_matrix():
    tracker = StreamingRawThoughtLogprobs()
    raw = torch.tensor([[1., 2., 3.]], dtype=torch.bfloat16)
    original = raw.clone()
    prefix = torch.tensor([[0, 1]])
    rng = torch.get_rng_state().clone()
    assert tracker(prefix, raw) is raw
    assert torch.equal(raw, original) and torch.equal(rng, torch.get_rng_state())
    # Modifying the caller's tensor cannot mutate the recorded distribution.
    raw.fill_(99.)
    tracker(torch.tensor([[0, 1, 2]]), torch.zeros((1, 3)))
    assert tracker._pending.shape == (1, 3)
    assert tracker._tokens[0].numel() == 1 and tracker._logprobs[0].numel() == 1
    actual = tracker.finish([[2, 1]])
    assert actual[0] == pytest.approx([original.float().log_softmax(-1)[0, 2].item(), -math.log(3)])


@pytest.mark.parametrize("rows", [[], [[]], [[0, 1]], [[True]], [[3]], [[-1]]])
def test_tracker_rejects_malformed_actual_token_rows(rows):
    tracker = StreamingRawThoughtLogprobs()
    tracker(torch.tensor([[0]]), torch.zeros(1, 3))
    with pytest.raises(ValueError):
        tracker.finish(rows)


def test_tracker_rejects_actual_token_mismatch_and_nonfinite_selected_score():
    tracker = StreamingRawThoughtLogprobs()
    tracker(torch.tensor([[0]]), torch.zeros(1, 3))
    tracker(torch.tensor([[0, 1]]), torch.zeros(1, 3))
    with pytest.raises(ValueError, match="differ"):
        tracker.finish([[2, 0]])
    invalid = StreamingRawThoughtLogprobs()
    invalid(torch.tensor([[0]]), torch.tensor([[float("nan"), 0., 1.]]))
    with pytest.raises(FloatingPointError, match="Non-finite"):
        invalid.finish([[1]])
