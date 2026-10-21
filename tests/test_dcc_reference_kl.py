"""CPU oracles for the optional frozen-base online-GRPO algorithm variant."""
from __future__ import annotations

from contextlib import contextmanager
import copy
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from slime_plugins.dcc.reference_kl import (ACTIVE_SCOPE, SOURCE, VARIANT,
    frozen_base_logprobs, thought_reference_kl_loss, validate_base_reference_model)


def trainer():
    spec = importlib.util.spec_from_file_location("dcc_reference_kl_tests",
        Path(__file__).resolve().parents[1] / "scripts/dcc_online_grpo.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ScalarAdapter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base = torch.nn.Parameter(torch.tensor(-2.), requires_grad=False)
        self.lora_weight = torch.nn.Parameter(torch.tensor(1.))
        self.disabled = False
        self.peft_config = {"default": SimpleNamespace(peft_type="LORA",
            modules_to_save=None, is_prompt_learning=False,
            target_modules={"q_proj", "v_proj", "o_proj"})}

    def get_model_status(self):
        return SimpleNamespace(enabled=not self.disabled, merged_adapters=[])

    @contextmanager
    def disable_adapter(self):
        before = self.disabled
        self.disabled = True
        try:
            yield
        finally:
            self.disabled = before


def segment(token=2, old=-1.4):
    return SimpleNamespace(prompt_ids=[1], completion_ids=[token], thought_logprobs=[old])


def test_k3_sign_gradient_and_fixed_budget_not_observed_length():
    current = torch.tensor([-2., -1.], requires_grad=True)
    reference = torch.tensor([-1., -2.])
    beta, denominator = .3, 2 * 3 * 8
    unweighted = thought_reference_kl_loss(current, reference,
        group_size=2, max_steps=3, thought_tokens=8)
    assert unweighted.item() == pytest.approx((math.expm1(1) - 1 + math.expm1(-1) + 1) / denominator)
    (beta * unweighted).backward()
    torch.testing.assert_close(current.grad,
        beta * torch.tensor([1 - math.exp(1), 1 - math.exp(-1)]) / denominator)
    assert current.grad[0] < 0 < current.grad[1]  # Descent moves each sampled logp toward base.
    assert reference.grad is None
    short = torch.tensor([-2.], requires_grad=True)
    (beta * thought_reference_kl_loss(short, reference[:1],
        group_size=2, max_steps=3, thought_tokens=8)).backward()
    assert short.grad.item() == current.grad[0].item()


def test_k3_small_drift_uses_expm1_and_overflow_is_refused():
    current, reference = torch.tensor([-3.]), torch.tensor([-3. + 1e-4])
    delta = float(reference - current)
    loss = thought_reference_kl_loss(current, reference, group_size=2, max_steps=1, thought_tokens=1)
    assert loss.item() > 0
    assert loss.item() == pytest.approx((math.expm1(delta) - delta) / 2, rel=.002)
    with pytest.raises(FloatingPointError, match="K3"):
        thought_reference_kl_loss(torch.tensor([-100.]), torch.tensor([0.]),
            group_size=2, max_steps=1, thought_tokens=1)
    with pytest.raises(ValueError, match="detached"):
        thought_reference_kl_loss(current, reference.requires_grad_(),
            group_size=2, max_steps=1, thought_tokens=1)


@pytest.mark.parametrize("mode,calls", [("off", 1), ("first", 2), ("full", 2)])
def test_off_is_exact_default_and_never_reads_adapter_or_reference(mode, calls):
    module = trainer()
    reports, gradients = [], []
    for explicit_zero in (False, True):
        class NoReference(torch.nn.Linear):
            def __getattr__(self, name):
                if name in {"disable_adapter", "peft_config", "get_model_status"}:
                    raise AssertionError("beta=0 must not inspect a base reference")
                return super().__getattr__(name)
        model = NoReference(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(-1.)
        observed = []
        def score(value, prompt, completion):
            observed.append((value.training, torch.is_grad_enabled()))
            return value.weight.reshape(1)
        extra = {"reference_kl_beta": 0} if explicit_zero else {}
        report = module.replay_group(model,
            SimpleNamespace(segments=[[segment()]], advantages=[-1.]), score,
            group_size=2, max_steps=1, thought_tokens=1, clip=.2,
            replay_diagnostic=mode, **extra)
        assert len(observed) == calls
        assert report["policy_loss"] == pytest.approx(math.exp(.4) / 2)
        assert "reference_kl" not in report
        reports.append(report)
        gradients.append(model.weight.grad.clone())
    assert reports[0] == reports[1]
    assert torch.equal(*gradients)


def test_active_scope_actual_old_denominator_and_separate_loss_report():
    module, model = trainer(), ScalarAdapter()
    ignored, active = segment(99), segment()
    group = SimpleNamespace(segments=[[ignored], [active]], advantages=[0., -1.])
    observed = []
    def score(value, prompt, completion):
        assert prompt == [1] and completion == [2]  # Only active thought IDs.
        observed.append((value.disabled, value.training, torch.is_grad_enabled()))
        return (value.base if value.disabled else value.base + value.lora_weight).reshape(1)
    report = module.replay_group(model, group, score, group_size=2, max_steps=2,
        thought_tokens=8, clip=.2, replay_diagnostic="off", reference_kl_beta=.3,
        reference_base_model="synthetic-frozen-base", reference_source_commit="1" * 40)
    denominator = 2 * 2 * 8
    policy = math.exp(.4) / denominator  # Actual sampled old=-1.4; base=-2 is NOT old.
    kl = (math.expm1(-1) + 1) / denominator
    assert report["policy_loss"] == pytest.approx(policy)
    assert report["reference_kl_unweighted_loss"] == pytest.approx(kl)
    assert report["reference_kl_loss"] == pytest.approx(.3 * kl)
    assert report["objective_loss"] == pytest.approx(policy + .3 * kl)
    assert model.lora_weight.grad.item() == pytest.approx((math.exp(.4) + .3 * (1 - math.exp(-1))) / denominator)
    assert model.base.grad is None
    assert observed == [(True, False, False), (False, True, True)]
    assert model.training and not model.disabled
    assert report["reference_kl"]["source"] == SOURCE
    assert report["reference_kl"]["base_model"] == "synthetic-frozen-base"
    assert report["reference_kl"]["source_commit"] == "1" * 40
    assert report["reference_kl"]["active_scope"] == ACTIVE_SCOPE
    assert report["reference_scored_segments"] == report["reference_scored_tokens"] == 1
    assert report["old_logprob_source"] == "actual-sampled-thought-behavior"
    assert report["algorithm_variant"] == VARIANT


def test_zero_signal_never_reads_reference_or_creates_kl_only_update():
    module = trainer()
    model = torch.nn.Linear(1, 1)
    group = SimpleNamespace(segments=[[segment()], [segment()]], advantages=[0., 0.])
    def forbidden(*args):
        raise AssertionError("No scorer call for original zero-signal segments")
    report = module.replay_group(model, group, forbidden, group_size=2, max_steps=2,
        thought_tokens=8, clip=.2, reference_kl_beta=.3)
    assert report["scored_segments"] == report["reference_scored_segments"] == 0
    assert report["policy_loss"] == report["reference_kl_loss"] == report["objective_loss"] == 0
    assert all(parameter.grad is None for parameter in model.parameters())


@pytest.mark.parametrize("state", ["partial-freeze", "trainable-base", "merged", "mixed-enabled", "bias"])
def test_unsupported_reference_states_are_rejected_before_any_forward(state):
    model = ScalarAdapter()
    if state == "partial-freeze":
        model.lora_weight.requires_grad_(False)
    elif state == "trainable-base":
        model.base.requires_grad_(True)
    elif state == "merged":
        model.get_model_status = lambda: SimpleNamespace(enabled=True, merged_adapters=["default"])
    elif state == "mixed-enabled":
        model.get_model_status = lambda: SimpleNamespace(enabled="irregular", merged_adapters=[])
    else:
        model.peft_config["default"].bias = "all"
    with pytest.raises(ValueError):
        validate_base_reference_model(model)


@pytest.mark.parametrize("training", [True, False])
def test_reference_exception_restores_adapter_and_original_training_mode(training):
    model = ScalarAdapter().train(training)
    def broken(value, *args):
        assert value.disabled and not value.training and not torch.is_grad_enabled()
        raise RuntimeError("reference failure")
    with pytest.raises(RuntimeError, match="reference failure"):
        frozen_base_logprobs(model, broken, [1], [2])
    assert model.training == training and not model.disabled
    assert all(value.grad is None for value in model.parameters())


@pytest.mark.parametrize("bad", ["-1", "nan", "inf", "-inf"])
def test_cli_rejects_nonfinite_negative_beta_before_reading_data(monkeypatch, bad):
    module = trainer()
    monkeypatch.setattr(module, "load_tasks", lambda _: pytest.fail("must validate beta before data"))
    with pytest.raises(SystemExit) as failure:
        module.main(["--model", "synthetic", "--train-data", "unused", "--output-dir", "unused",
            "--source-commit", "0" * 40, "--reference-kl-beta=" + bad])
    assert failure.value.code == 2


@pytest.mark.parametrize("beta", [None, 0, .01])
def test_cli_records_explicit_algorithm_variant_and_reference_identity(monkeypatch, beta):
    from slime_plugins.dcc.data import Task
    from slime_plugins.dcc.online_grpo import ALGORITHM
    module = trainer()
    task = Task("synthetic", "cuad", "train", "clause?", document="clause", gold=["clause"])
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setattr(module, "load_tasks", lambda _: [task])
    monkeypatch.setattr(module, "validate_source", lambda _: None)
    seen = []
    async def capture(args, tasks, plan, world_size, **kwargs):
        config = module.configuration(args, plan, tasks, world_size)
        seen.append(config)
        reference = config["reference_kl"]
        assert reference["beta"] == (beta or 0)
        assert config["objective_denominator"] == 2 * 2 * 4096
        if beta:
            assert config["algorithm"] == ALGORITHM + "+" + VARIANT
            assert config["algorithm_variant"] == VARIANT
            assert reference["base_model"] == "synthetic-base"
            assert reference["source_commit"] == "0" * 40
            assert reference["source"] == SOURCE
        else:
            assert config["algorithm"] == ALGORITHM
            assert "algorithm_variant" not in config
            assert reference["source"] == "not-read" and reference["base_model"] is None
    monkeypatch.setattr(module, "train", capture)
    extra = [] if beta is None else ["--reference-kl-beta", str(beta)]
    assert module.main(["--model", "synthetic-base", "--train-data", "unused",
        "--output-dir", "unused", "--source-commit", "0" * 40, *extra]) == 0
    assert len(seen) == 1


@pytest.mark.parametrize("compute_profile", ["legacy", "bf16-autocast-v1"])
def test_real_tiny_peft_checkpoint_reference_restore_and_current_gradients(compute_profile):
    """Random local weights only; compare checkpoint replay with a direct graph."""
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
    from slime_plugins.dcc.hf_chunked_logprobs import completion_logprobs_chunked
    from slime_plugins.dcc.online_grpo import thought_policy_loss
    from slime_plugins.dcc.hf_compute import configure_compute_profile
    torch.manual_seed(337)
    model = get_peft_model(Qwen3MoeForCausalLM(Qwen3MoeConfig(
        vocab_size=32, hidden_size=16, intermediate_size=24, moe_intermediate_size=8,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        num_experts=4, num_experts_per_tok=2, attention_dropout=0., max_position_embeddings=64)),
        LoraConfig(r=2, lora_alpha=4, lora_dropout=0.,
            target_modules=["q_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM"))
    with torch.no_grad():
        for name, value in model.named_parameters():
            if "lora_B" in name:
                value.normal_(0, .04)
    configure_compute_profile(model, compute_profile)
    direct = copy.deepcopy(model)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    validate_base_reference_model(model)
    prompt, thought, old = [2, 3, 4], [5, 6, 7], [-3.5] * 3
    item = SimpleNamespace(prompt_ids=prompt, completion_ids=thought, thought_logprobs=old)
    group = SimpleNamespace(segments=[[item]], advantages=[-.7])
    hook_states, scorer_states, references = [], [], []
    q_proj = next(value for name, value in model.named_modules() if name.endswith("self_attn.q_proj"))
    hook = q_proj.register_forward_pre_hook(lambda value, args:
        hook_states.append((value.disable_adapters, model.training, torch.is_grad_enabled())))
    def score(value, prompt_ids, completion_ids):
        disabled = value.get_model_status().enabled is False
        result = completion_logprobs_chunked(value, prompt_ids, completion_ids, head_chunk_tokens=2)
        scorer_states.append((disabled, value.training, torch.is_grad_enabled(), result.requires_grad))
        if disabled:
            references.append(result)
        return result
    try:
        report = trainer().replay_group(model, group, score, group_size=2, max_steps=2,
            thought_tokens=8, clip=.2, replay_diagnostic="off", reference_kl_beta=.1,
            reference_base_model="random-tiny-Qwen3Moe-fixture")
    finally:
        hook.remove()
    assert scorer_states == [(True, False, False, False), (False, True, True, True)]
    assert hook_states[0] == (True, False, False)
    assert len(hook_states) >= 3  # Decoder forward was recomputed for backward.
    assert all(state == (False, True, True) for state in hook_states[1:])
    assert all(not value.requires_grad and value.grad_fn is None for value in references)
    assert model.training and model.get_model_status().enabled is True
    assert model.dcc_compute_context_observation == dict(device_type="cpu",
        autocast_enabled=compute_profile != "legacy",
        autocast_dtype="torch.bfloat16" if compute_profile != "legacy" else None)
    with direct.disable_adapter(), torch.no_grad():
        reference = completion_logprobs_chunked(direct, prompt, thought, head_chunk_tokens=2)
    direct.train()
    current = completion_logprobs_chunked(direct, prompt, thought, head_chunk_tokens=2)
    expected = thought_policy_loss(current, torch.tensor(old), -.7,
        group_size=2, max_steps=2, thought_tokens=8, clip=.2)
    expected = expected + .1 * thought_reference_kl_loss(current, reference,
        group_size=2, max_steps=2, thought_tokens=8)
    expected.backward()
    assert report["objective_loss"] == pytest.approx(expected.item())
    direct_parameters = dict(direct.named_parameters())
    nonzero_adapter_gradients = 0
    for name, value in model.named_parameters():
        if "lora_" in name:
            assert value.grad is not None
            nonzero_adapter_gradients += int(bool(value.grad.abs().sum() > 0))
            torch.testing.assert_close(value.grad, direct_parameters[name].grad, rtol=2e-5, atol=1e-7)
        else:
            assert not value.requires_grad and value.grad is None
    assert nonzero_adapter_gradients > 0
    # A failed actual PEFT reference forward must restore state too.
    def broken(value, *args):
        assert value.get_model_status().enabled is False and not torch.is_grad_enabled()
        raise RuntimeError("real PEFT reference failure")
    with pytest.raises(RuntimeError, match="real PEFT"):
        frozen_base_logprobs(model, broken, prompt, thought)
    assert model.training and model.get_model_status().enabled is True
    assert not torch.cuda.is_initialized()
