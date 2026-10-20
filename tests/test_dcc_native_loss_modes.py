"""CPU objective/gradient evidence for optional native SFT losses."""
from types import SimpleNamespace
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

SOURCE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(SOURCE / "scripts"), str(SOURCE / "slime")]

import dcc_tool_sft as training
from dcc_native_loss import native_completion_json_mixture_nll
from slime_plugins.dcc.hf_policy import completion_logprobs


class ScalarPolicy(torch.nn.Module):
    def __init__(self, values):
        super().__init__()
        self.scores = torch.nn.ParameterList([
            torch.nn.Parameter(torch.tensor(row, dtype=torch.float64)) for row in values])


def example(index=0, thought=2, final=3):
    ids = [9] * (thought + final)
    ids[thought - 1] = 151668
    return training.Example(str(index), "final", "independent_web", [index], ids,
                            native_phase_lengths=(thought, final))


@pytest.mark.parametrize("mode,expected,gradient", [
    ({}, 3.2, [-.2] * 5),
    ({"native_completion_json_mixture_loss": True}, 3.36,
     [-.16, -.16, -.16 - .2 / 3, -.16 - .2 / 3, -.16 - .2 / 3]),
    ({"native_phase_normalized_loss": True}, 3.6,
     [-.1, -.1, -.8 / 3, -.8 / 3, -.8 / 3]),
])
def test_actual_backward_objectives_and_every_native_token_gradient(monkeypatch, mode, expected, gradient):
    policy = ScalarPolicy([[-1., -3., -2., -4., -6.]])
    calls = []
    def scores(model, prompt, completion):
        calls.append((list(prompt), list(completion)))
        return model.scores[prompt[0]]
    monkeypatch.setattr(training, "completion_logprobs", scores)
    item = example()
    observed = training.backward_batch(policy, [item], **mode)
    assert observed == pytest.approx(expected)
    assert policy.scores[0].grad.tolist() == pytest.approx(gradient)
    assert calls == [(item.prompt, item.completion)]
    assert len(item.completion) == 5 and item.completion[1] == 151668


def test_equal_example_weight_is_preserved_for_unequal_completion_lengths(monkeypatch):
    policy = ScalarPolicy([[-1., -3., -2., -4., -6.], [-2., -4.]])
    monkeypatch.setattr(training, "completion_logprobs", lambda model, prompt, completion: model.scores[prompt[0]])
    observed = training.backward_batch(policy, [example(), example(1, 1, 1)])
    assert observed == pytest.approx((3.2 + 3.0) / 2)
    assert policy.scores[0].grad.tolist() == pytest.approx([-.1] * 5)
    assert policy.scores[1].grad.tolist() == pytest.approx([-.25] * 2)


def test_mixture_telemetry_is_detached_and_uses_validated_polarity(monkeypatch):
    policy = ScalarPolicy([[-1., -3., -2., -4., -6.]])
    monkeypatch.setattr(training, "completion_logprobs", lambda model, prompt, completion: model.scores[prompt[0]])
    telemetry = {}
    training.backward_batch(policy, [example()], native_completion_json_mixture_loss=True,
                            native_loss_telemetry=telemetry,
                            native_telemetry_metadata={("0", "final"): {
                                "domain": "web", "polarity": "not_applicable", "stage": "final"}})
    row = telemetry["per_example"][0]
    assert (row["thought_tokens"], row["json_tokens"]) == (2, 3)
    assert [row[key] for key in ("thought_nll", "json_nll", "completion_nll", "objective_nll")] == pytest.approx([2., 4., 3.2, 3.36])
    assert telemetry["summary"]["example_count"] == 1
    json.dumps(telemetry, allow_nan=False)
    assert policy.scores[0].grad is not None


@pytest.mark.parametrize("mode", ["phase", "mixture"])
def test_loss_modes_reject_missing_or_fabricated_phase_boundary(mode):
    item = example()
    invalid = training.Example(item.task_id, item.stage, item.category, item.prompt,
                               [9] * 5, native_phase_lengths=(2, 3))
    function = training.native_phase_normalized_nll if mode == "phase" else native_completion_json_mixture_nll
    with pytest.raises(ValueError, match="actual closing ID"):
        function(torch.zeros(5), invalid)
    legacy = training.Example("old", "final", "web", [1], [2])
    with pytest.raises(ValueError, match="phase lengths"):
        function(torch.zeros(1), legacy)


def test_two_optional_loss_modes_are_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        training.backward_batch(None, [], native_phase_normalized_loss=True,
                                native_completion_json_mixture_loss=True)


class TinyTransitionPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transitions = torch.nn.Parameter(torch.arange(36, dtype=torch.float32).reshape(6, 6) / 10)
        self.device = torch.device("cpu")

    def forward(self, input_ids, *, use_cache, logits_to_keep):
        assert use_cache is False
        return SimpleNamespace(logits=self.transitions[input_ids][:, -logits_to_keep:])


def test_real_completion_logprob_helper_has_correct_shift_and_only_response_labels():
    policy = TinyTransitionPolicy()
    observed = completion_logprobs(policy, [0, 1], [2, 3, 4])
    expected = policy.transitions[[1, 2, 3]].log_softmax(-1).gather(-1, torch.tensor([[2], [3], [4]])).flatten()
    torch.testing.assert_close(observed, expected, rtol=1e-6, atol=1e-6)
    (-observed.mean()).backward()
    nonzero_rows = policy.transitions.grad.abs().sum(-1).nonzero().flatten().tolist()
    assert nonzero_rows == [1, 2, 3]


def test_native_mode_flags_require_native_validation_before_model_or_cuda_load():
    script = Path(training.__file__)
    result = subprocess.run([sys.executable, "-B", str(script), "--native-phase-normalized-loss",
                             "--model", "unused", "--records", "unused", "--output-dir", "unused"],
                            text=True, capture_output=True)
    assert result.returncode == 2
    assert "Native loss modes require --native-thinking" in result.stderr


def test_validated_real_phase_lengths_survive_loader_copy_and_strict_budgets(tmp_path, monkeypatch):
    from collections import Counter
    import dcc_native_sft as native
    from test_dcc_native_training import make_record
    tokenizer, identity, schema, bundle = make_record(tmp_path, monkeypatch)
    record = bundle["record"]
    actual = native.validate_native_record(bundle, tokenizer, expected_identity=identity, original_schema=schema)
    expected = tuple(len(phase["token_ids"]) for phase in bundle["native_trace"]["phases"])
    assert actual.native_phase_lengths == expected
    key = native._segment_key(record)
    data = native.NativeTrainingData([record], [], identity, identity.manifest(tokenizer),
        {key: actual}, {key: native._json_key(record)}, strict_training_budgets=True)
    copied = data.encode_record(record, tokenizer, max_prompt=16384, max_response=20480,
                                batch_tokens=53248, skips=Counter())
    assert copied.native_phase_lengths == expected
    assert copied.prompt == actual.prompt and copied.prompt is not actual.prompt
    assert copied.completion == actual.completion and copied.completion is not actual.completion
    with pytest.raises(ValueError):
        data.encode_record(record, tokenizer, max_prompt=1, max_response=20480,
                           batch_tokens=53248, skips=Counter())
