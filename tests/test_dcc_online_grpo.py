"""Policy-gradient oracles, failure coverage, and real LoRA replay checks."""
from __future__ import annotations

import importlib.util
import asyncio
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from slime_plugins.dcc.episode import EpisodeResult
from slime_plugins.dcc.native_behavior_logprobs import BEHAVIOR_LOGPROB_SOURCE
from slime_plugins.dcc.online_grpo import (apply_policy_update, group_signal,
                                          thought_policy_loss, wait_for_update_ranks)


def episode(reward, *, status="completed", thought=(6, 7), json_ids=(8, 9)):
    step = SimpleNamespace(prompt_ids=[3, 4, 5],
        completion_ids=list(thought) + list(json_ids), thought_token_count=len(thought),
        thought_logprobs=[-3.5] * len(thought), thought_logprobs_source=BEHAVIOR_LOGPROB_SOURCE)
    return EpisodeResult("task", "cuad", "train", [], status, reward, {},
                         [step], [], [])


def signal(results):
    return group_signal(results, group_size=len(results), max_steps=2, thought_tokens=8)


def test_failed_model_thoughts_remain_negative_training_examples():
    success = episode(1)
    failed = episode(0, status="model_protocol_failure", thought=(10,), json_ids=())
    assert failed.trainable_steps == []  # Legacy evaluation property is unchanged.
    group = signal([success, failed])
    assert group.discarded_reason is None
    assert group.advantages[0] > 0 > group.advantages[1]
    assert group.segments[0][0].completion_ids == [6, 7]
    assert group.segments[1][0].completion_ids == [10]
    assert all(8 not in s.completion_ids and 9 not in s.completion_ids
               for segments in group.segments for s in segments)


@pytest.mark.parametrize("reward", [0.0, 0.1, float(torch.tensor(0.1)), 1.0])
def test_equal_rewards_have_exactly_zero_advantages(reward):
    group = signal([episode(reward) for _ in range(7)])
    assert group.advantages == [0.0] * 7
    assert group.reward_std == 0


@pytest.mark.parametrize("status", ["infrastructure_error", "context_budget_failure"])
def test_infrastructure_discards_whole_group_without_dropping_only_failures(status):
    group = signal([episode(1), episode(0, status=status)])
    assert group.discarded_reason
    assert group.advantages == [0, 0] and group.segments == [[], []]


def test_group_rejects_heldout_or_untyped_token_boundaries():
    heldout = episode(1)
    heldout.split = "validation"
    with pytest.raises(ValueError, match="same task"):
        signal([heldout, episode(0)])
    other = episode(0)
    other.steps[0].thought_token_count = True
    with pytest.raises(ValueError, match="boundaries"):
        signal([episode(1), other])


def test_fixed_budget_loss_matches_analytic_categorical_policy_gradient():
    logits = torch.tensor([0.2, -0.4, 0.1], dtype=torch.float64, requires_grad=True)
    logprobs = logits.log_softmax(-1)
    loss = thought_policy_loss(logprobs[[0, 1]], logprobs[[0, 1]].detach(), 0.7,
        group_size=4, max_steps=2, thought_tokens=8)
    actual = torch.autograd.grad(loss, logits, retain_graph=True)[0]
    oracle = -0.7 * (logprobs[0] + logprobs[1]) / 64
    expected = torch.autograd.grad(oracle, logits)[0]
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_response_length_does_not_change_each_tokens_weight():
    gradients = []
    for length in (1, 7):
        current = torch.zeros(length, requires_grad=True)
        thought_policy_loss(current, current.detach(), -1.0,
            group_size=2, max_steps=2, thought_tokens=8).backward()
        gradients.append(current.grad)
    assert gradients[0][0] == gradients[1][0] == 1 / 32


def test_zero_signal_batch_does_not_apply_existing_adam_momentum():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([parameter], lr=0.1)
    parameter.grad = torch.tensor([1.0])
    norm, applied = apply_policy_update([parameter], optimizer)
    assert norm > 0 and applied
    before = parameter.detach().clone()
    optimizer.zero_grad(set_to_none=True)
    norm, applied = apply_policy_update([parameter], optimizer)
    assert norm == 0 and not applied
    assert torch.equal(parameter.detach(), before)
    assert optimizer.state[parameter]["step"] == 1


def test_nonfinite_global_gradient_never_steps_optimizer():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([parameter], lr=0.1)
    parameter.grad = torch.tensor([float("nan")])
    with pytest.raises(FloatingPointError):
        apply_policy_update([parameter], optimizer)
    assert parameter.item() == 1.0 and not optimizer.state


def load_trainer():
    spec = importlib.util.spec_from_file_location("dcc_online_grpo_tests",
        Path(__file__).resolve().parents[1] / "scripts/dcc_online_grpo.py")
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    return trainer


def test_default_task_plan_matches_frozen_original_training_draws():
    trainer = load_trainer()
    tasks = [SimpleNamespace(id=f"task-{index}", benchmark="cuad", split="train",
                             gold=["span"] if index < 3 else []) for index in range(5)]
    # Captured from the original 13919869 training source, before this option.
    expected = [[3, 1, 2, 0, 4, 2], [0, 1, 3, 0, 1, 2],
                [4, 0, 2, 1, 4, 0], [2, 1, 3, 0, 1, 2]]
    assert trainer.task_plan(tasks, steps=4, world_size=6, seed=79) == expected
    assert trainer.task_plan(tasks, steps=4, world_size=6, seed=79,
                             sampling="stratified-3to1") == expected
    assert trainer.task_plan(tasks[:3], steps=3, world_size=2, seed=47) == [[1, 0], [2, 1], [2, 0]]
    with pytest.raises(ValueError, match="positive"):
        trainer.task_plan(tasks[3:], steps=1, world_size=2, seed=47)


def test_uniform_task_plan_is_label_blind_and_visits_every_index_per_deck():
    import random
    trainer = load_trainer()
    class PublicTask:
        benchmark, split = "cuad", "train"
        def __init__(self, index):
            self.id = f"public-task-{index}"
        @property
        def gold(self):
            raise AssertionError("Uniform selection must not access gold labels")
        @property
        def reward(self):
            raise AssertionError("Uniform selection must not access outcomes")
        @property
        def metadata(self):
            raise AssertionError("Uniform selection must not access private metadata")
    tasks = [PublicTask(index) for index in range(7)]
    state = random.getstate()
    plan = trainer.task_plan(tasks, steps=6, world_size=3, seed=79, sampling="uniform")
    flat = [index for row in plan for index in row]
    assert random.getstate() == state  # Only the explicit local RNG is used.
    assert len(plan) == 6 and all(len(row) == 3 for row in plan)
    assert sorted(flat[:7]) == sorted(flat[7:14]) == list(range(7))
    assert len(set(flat[14:])) == 4
    assert plan == trainer.task_plan(tasks, steps=6, world_size=3, seed=79, sampling="uniform")
    serial = trainer.task_plan(tasks, steps=18, world_size=1, seed=79, sampling="uniform")
    assert flat == [row[0] for row in serial]
    assert plan != trainer.task_plan(tasks, steps=6, world_size=3, seed=80, sampling="uniform")


@pytest.mark.parametrize("tasks", [[],
    [SimpleNamespace(id="heldout", benchmark="cuad", split="validation")],
    [SimpleNamespace(id="same", benchmark="cuad", split="train")] * 2])
def test_uniform_plan_still_rejects_invalid_or_heldout_task_identity(tasks):
    with pytest.raises(ValueError):
        load_trainer().task_plan(tasks, steps=1, world_size=1, seed=79, sampling="uniform")


@pytest.mark.parametrize("enable_web", [False, True])
def test_uniform_cli_reaches_cuad_and_mixed_plans_with_accurate_metadata(monkeypatch, enable_web):
    from slime_plugins.dcc.data import Task
    from slime_plugins.dcc import online_web
    trainer = load_trainer()
    # The default stratified planner rejects this all-negative pool. Uniform
    # must reach both entrypoint branches without requiring a positive label.
    cuad = [Task(f"negative-{index}", "cuad", "train", "Q", "contract", []) for index in range(5)]
    web_tasks = [Task(f"web-{index}", "widesearch", "train", "Q", gold=None) for index in range(2)]
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setattr(trainer, "load_tasks", lambda _: cuad)
    monkeypatch.setattr(trainer, "validate_source", lambda *args, **kwargs: None)
    monkeypatch.setattr(online_web, "WebTrainingInputs", lambda *args, **kwargs: SimpleNamespace(tasks=web_tasks))
    def forbidden(*args, **kwargs):
        raise AssertionError("Planner CLI tests must not load a model or initialize CUDA")
    monkeypatch.setattr(trainer, "load_policy_model", forbidden)
    monkeypatch.setattr(torch.cuda, "set_device", forbidden)
    observed = []
    async def capture(args, tasks, plan, world_size, *, web_inputs):
        config = trainer.configuration(args, plan, tasks, world_size)
        observed.append(config)
        flat = [index for row in plan for index in row]
        expected = trainer.task_plan(cuad, steps=5 if enable_web else 10,
                                    world_size=1, seed=args.seed, sampling="uniform")
        assert (flat[::2] if enable_web else flat) == [row[0] for row in expected]
        if enable_web:
            assert all(index >= len(cuad) for index in flat[1::2])
            assert config["online_web"]["cuad_plan"] == "uniform-label-blind-CUAD-TRAIN-index-deck"
        else:
            assert web_inputs is None
        assert tasks == (cuad + web_tasks if enable_web else cuad)
        assert config["cuad_task_selection"] == {
            "mode": "uniform", "stratification": None, "negative_stride": None,
            "positive_fallback_when_no_negatives": False, "seeded_independent_decks": False,
            "outcome_based_resampling": False, "label_dependent": False,
            "without_replacement_per_deck_cycle": True,
            "deck_scope": "all-CUAD-TRAIN-task-indices", "seeded_shuffled_deck": True}
        assert config["arguments"]["cuad_task_sampling"] == "uniform"
        assert config["planned_task_ids"] == [[tasks[index].id for index in row] for row in plan]
        assert config["objective_denominator"] == args.group_size * args.max_steps * args.thought_tokens
    monkeypatch.setattr(trainer, "train", capture)
    extra = (["--enable-online-web", "--web-train-data", "public.jsonl",
              "--web-train-labels", "labels.jsonl", "--web-binding", "binding.json"] if enable_web else [])
    assert trainer.main(["--model", "CPU-fixture", "--train-data", "unused.jsonl",
        "--output-dir", "unused-output", "--source-commit", "0" * 40,
        "--steps", "5", "--cuad-task-sampling", "uniform", *extra]) == 0
    assert len(observed) == 1 and not torch.cuda.is_initialized()


def test_uniform_training_still_requires_explicit_reward_labels(monkeypatch):
    from slime_plugins.dcc.data import Task
    trainer = load_trainer()
    monkeypatch.setattr(trainer, "load_tasks", lambda _: [Task("unlabelled", "cuad", "train", "Q", "contract", gold=None)])
    monkeypatch.setattr(trainer, "validate_source", lambda _: None)
    with pytest.raises(ValueError, match="explicit reward labels"):
        trainer.main(["--model", "CPU-fixture", "--train-data", "unused.jsonl",
            "--output-dir", "unused-output", "--source-commit", "0" * 40,
            "--cuad-task-sampling", "uniform"])
    assert not torch.cuda.is_initialized()


def test_invalid_cuad_sampling_cli_is_rejected_before_inputs_or_cuda(monkeypatch):
    trainer = load_trainer()
    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid sampling may not read inputs or initialize CUDA")
    monkeypatch.setattr(trainer, "load_tasks", forbidden)
    monkeypatch.setattr(torch.cuda, "set_device", forbidden)
    with pytest.raises(SystemExit) as error:
        trainer.main(["--model", "CPU-fixture", "--train-data", "unused.jsonl",
            "--output-dir", "unused-output", "--source-commit", "0" * 40,
            "--cuad-task-sampling", "unknown"])
    assert error.value.code == 2 and not torch.cuda.is_initialized()


@pytest.mark.parametrize("batch_args,expected,concise,follow_docs", [([], 4, False, False),
    (["--native-batch-size", "2", "--cuad-concise-reasoning"], 2, True, False),
    (["--auto-read-follow-docs"], 4, False, True)])
def test_native_batch_cli_keeps_logical_group_and_token_budgets(monkeypatch, batch_args, expected, concise, follow_docs):
    from slime_plugins.dcc.data import Task
    trainer = load_trainer()
    task = Task("synthetic-train", "cuad", "train", "Find the clause", document="clause", gold=["clause"])
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setattr(trainer, "load_tasks", lambda _: [task])
    monkeypatch.setattr(trainer, "validate_source", lambda _: None)
    def forbidden(*args, **kwargs):
        raise AssertionError("CLI parsing may not initialize CUDA or load a model")
    monkeypatch.setattr(torch.cuda, "set_device", forbidden)
    monkeypatch.setattr(trainer, "load_policy_model", forbidden)
    observed = []
    async def capture(args, tasks, plan, world_size, *, web_inputs):
        config = trainer.configuration(args, plan, tasks, world_size)
        observed.append(config)
        assert args.native_batch_size == expected and args.group_size == 4
        assert (args.thought_tokens, args.json_tokens, args.max_steps) == (8, 16, 2)
        assert web_inputs is None
        assert trainer.make_runner(args).cuad_concise_reasoning is concise
        assert trainer.make_runner(args).auto_read_follow_docs is follow_docs
        assert config["auto_read_follow_docs"]["enabled"] is follow_docs
        assert args.auto_read_sources == 2
    monkeypatch.setattr(trainer, "train", capture)
    assert trainer.main(["--model", "CPU-fixture", "--train-data", "unused.jsonl",
        "--output-dir", "unused-output", "--source-commit", "0" * 40,
        "--group-size", "4", "--thought-tokens", "8", "--json-tokens", "16",
        "--max-steps", "2", *batch_args]) == 0
    assert len(observed) == 1
    assert observed[0]["arguments"]["native_batch_size"] == expected
    assert observed[0]["cuad_task_selection"]["mode"] == "stratified-3to1"
    assert observed[0]["cuad_task_selection"]["negative_stride"] == 4
    assert observed[0]["cuad_task_selection"]["label_dependent"] is True
    assert observed[0]["objective_denominator"] == 4 * 2 * 8
    assert observed[0]["algorithm"] == "online-native-HF-thought-only-GRPO-v2"
    assert observed[0]["context_failure_handling"]["version"] == "hf-native-context-sampled-prefix-v2"
    assert observed[0]["context_failure_handling"]["initial_prompt"] == "discard-whole-group"
    assert observed[0]["cuad_concise_reasoning"]["enabled"] is concise
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("invalid", ["0", "-1", "1.5", "nan", "true"])
def test_invalid_native_batch_cli_is_rejected_before_cuda_or_inputs(monkeypatch, invalid):
    trainer = load_trainer()
    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid batching may not initialize CUDA, load inputs or train")
    for name in ("load_tasks", "validate_source", "load_policy_model", "train"):
        monkeypatch.setattr(trainer, name, forbidden)
    monkeypatch.setattr(torch.cuda, "set_device", forbidden)
    with pytest.raises(SystemExit) as error:
        trainer.main(["--model", "CPU-fixture", "--train-data", "unused.jsonl",
            "--output-dir", "unused-output", "--source-commit", "0" * 40,
            "--native-batch-size", invalid])
    assert error.value.code == 2
    assert not torch.cuda.is_initialized()


def test_native_batch_two_collects_complete_group_four_with_failure_and_fixed_loss():
    """Real native queue/episode control flow; only neural decoding is synthetic."""
    from slime_plugins.dcc import EpisodeRunner
    from slime_plugins.dcc.data import Task
    from slime_plugins.dcc.episode import ModelProtocolFailure
    from slime_plugins.dcc.hf_native_policy import END_THINK, NativeHFCompletion, NativeHFPolicy
    trainer = load_trainer()
    task = Task("synthetic-train", "cuad", "train", "Find clause", document="clause", gold=["clause"])
    batches, produced = [], []
    async def collect():
        # The decoder seam avoids any model/tokenizer/compiler or GPU load;
        # generate_structured, the queue worker, episode runner and drain are real.
        policy = object.__new__(NativeHFPolicy)
        policy.batch_size, policy.thought_tokens, policy.json_tokens = 2, 4, 4
        policy.sampling_profile, policy.mixed_schema_batching = "raw", False
        policy.queue, policy.worker = asyncio.Queue(), None
        def decode(messages, schema, task_ids):
            assert all(value == task.id for value in task_ids)
            batches.append(len(messages))
            values = []
            for _ in messages:
                index = len(produced)
                failed = index == 1
                thought = [20 + index] * 4 if failed else [20 + index, END_THINK]
                step = NativeHFCompletion(
                    text="" if failed else '{"plan":"Check","checks":"Observed","answer":[{"text":"clause","confidence":1}]}',
                    prompt_ids=[3, 4, 5], completion_ids=thought + ([] if failed else [151645]),
                    thought_token_count=len(thought), phase_lengths=(len(thought), 0 if failed else 1),
                    finish_reason="length" if failed else "stop",
                    native_status="thought_did_not_stop_on_real_close" if failed else "completed",
                    thought_logprobs=[-1.] * len(thought), thought_logprobs_source=BEHAVIOR_LOGPROB_SOURCE)
                produced.append(step)
                values.append(ModelProtocolFailure(step.native_status, completion=step,
                    protocol="synthetic-decoder", trace_id=str(index)) if failed else step)
            return values
        policy._generate_batch = decode
        runner = EpisodeRunner(max_steps=1, max_new_tokens=8, initial_chunks=0,
                               evaluation_temperature=1., cuad_structured_actions=True)
        results = await trainer.collect_group(runner, task, policy, group_size=4, rank=0)
        assert policy.worker is None and policy.queue.empty()
        return results
    results = asyncio.run(collect())
    assert batches == [2, 2] and len(produced) == len(results) == 4
    assert [result.steps[0] for result in results] == produced
    assert [result.status for result in results] == ["completed", "model_protocol_failure", "completed", "completed"]
    group = group_signal(results, group_size=4, max_steps=1, thought_tokens=4)
    assert group.discarded_reason is None and group.rewards == [1., 0., 1., 1.]
    assert len(group.advantages) == len(group.segments) == 4
    assert group.advantages[1] < 0 and all(group.advantages[i] > 0 for i in (0, 2, 3))
    assert group.segments[1][0].completion_ids == produced[1].completion_ids
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(-1.)
    def scorer(value, prompt, completion):
        return value.weight.reshape(1).expand(len(completion))
    report = trainer.replay_group(model, group, scorer, group_size=4, max_steps=1,
                                 thought_tokens=4, clip=.2, replay_diagnostic="off")
    oracle = -sum(advantage * len(segments[0].completion_ids)
                  for segments, advantage in zip(group.segments, group.advantages, strict=True)) / (4 * 1 * 4)
    assert report["scored_segments"] == 4 and report["scored_tokens"] == 10
    assert report["policy_loss"] == pytest.approx(oracle)
    assert model.weight.grad.item() == pytest.approx(oracle)
    assert not torch.cuda.is_initialized()


def test_real_qwen3_moe_lora_replay_improves_rewarded_thought_odds():
    pytest.importorskip("peft")
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
    from slime_plugins.dcc.hf_policy import completion_logprobs
    trainer = load_trainer()
    torch.manual_seed(13)
    config = Qwen3MoeConfig(vocab_size=32, hidden_size=16, intermediate_size=24,
        moe_intermediate_size=8, num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=1, head_dim=8, num_experts=4,
        num_experts_per_tok=2, attention_dropout=0.0)
    model = get_peft_model(Qwen3MoeForCausalLM(config), LoraConfig(
        r=2, lora_alpha=4, lora_dropout=0.0,
        target_modules=["q_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM"))
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=0.03, weight_decay=0)
    model.eval()
    def odds():
        with torch.no_grad():
            return (completion_logprobs(model, [3, 4, 5], [6])
                    - completion_logprobs(model, [3, 4, 5], [7])).item()
    before = odds()
    results = [episode(1, thought=(6,)),
               episode(0, status="model_protocol_failure", thought=(7,), json_ids=())]
    # This optimizer oracle uses a recorded synthetic behavior distribution;
    # native cached-generation capture is tested separately against output_logits.
    for result in results:
        step = result.steps[0]
        with torch.no_grad():
            step.thought_logprobs = completion_logprobs(model, step.prompt_ids,
                step.completion_ids[:step.thought_token_count]).tolist()
    group = signal(results)
    report = trainer.replay_group(model, group, completion_logprobs,
        group_size=2, max_steps=2, thought_tokens=8, clip=0.2)
    norm, applied = apply_policy_update(parameters, optimizer)
    model.eval()
    assert report["scored_tokens"] == 2 and report["scored_segments"] == 2
    assert report["maximum_replay_current_logprob_difference"] < 1e-5
    assert applied and norm > 0
    assert odds() > before
    assert not torch.cuda.is_initialized()


def test_unlabeled_web_data_is_rejected_before_any_training():
    from slime_plugins.dcc.data import Task
    trainer = load_trainer()
    task = Task("public-web", "widesearch", "train", "Find entities", gold=None,
                metadata={"source": "independent_curriculum"})
    with pytest.raises(ValueError, match="reward labels"):
        trainer.validate_train_tasks([task])
    labeled = Task("labeled-web", "widesearch", "train", "Find entities", gold=[{"name": "entity"}],
                   metadata={"source": "independent_curriculum"})
    with pytest.raises(ValueError, match="CUAD only"):
        trainer.validate_train_tasks([labeled])


def test_group_exception_drains_sibling_before_policy_close():
    trainer = load_trainer()
    events = []
    class Runner:
        counter = 0
        async def run(self, task, policy, **kwargs):
            self.counter += 1
            if self.counter == 1:
                await asyncio.sleep(0.01)
                raise RuntimeError("original rollout error")
            try:
                await asyncio.sleep(60)
            finally:
                events.append("sibling_cancelled")
    class Policy:
        def bind_task(self, *args):
            return self
        async def close(self):
            assert events == ["sibling_cancelled"]
            events.append("policy_drained")
    with pytest.raises(RuntimeError, match="original rollout error"):
        asyncio.run(trainer.collect_group(Runner(), SimpleNamespace(id="task"), Policy(),
                                           group_size=2, rank=0))
    assert events == ["sibling_cancelled", "policy_drained"]


def _distributed_update_worker(rank, rendezvous, output):
    import torch.distributed as distributed
    from datetime import timedelta
    distributed.init_process_group("gloo", init_method="file://" + rendezvous,
        rank=rank, world_size=2, timeout=timedelta(seconds=60))
    control = distributed.new_group(backend="gloo", timeout=timedelta(seconds=60))
    # A rank that finishes early cannot enter the gradient collective until
    # the other rank has actually finished its simulated rollout/replay.
    if rank == 1:
        import time
        time.sleep(0.25)
    def device_ready():
        Path(output + str(rank) + ".ready").write_text("ready")
    wait_for_update_ranks(world_size=2, distributed=distributed,
        control_group=control, synchronize=device_ready, timeout_seconds=30)
    assert all(Path(output + str(other) + ".ready").read_text() == "ready" for other in range(2))
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([parameter], lr=0.1, weight_decay=0)
    # One rank has no local policy signal; both must apply the global update.
    parameter.grad = torch.tensor([2.0]) if rank == 0 else None
    norm, applied = apply_policy_update([parameter], optimizer,
        world_size=2, distributed=distributed)
    assert norm == 1.0 and applied
    torch.testing.assert_close(parameter.detach(), torch.tensor([0.9]))
    before = parameter.detach().clone()
    optimizer.zero_grad(set_to_none=True)
    norm, applied = apply_policy_update([parameter], optimizer,
        world_size=2, distributed=distributed)
    assert norm == 0 and not applied and torch.equal(parameter.detach(), before)
    # Opposing local gradients also cancel before the decision to step Adam.
    parameter.grad = torch.tensor([1.0 if rank == 0 else -1.0])
    norm, applied = apply_policy_update([parameter], optimizer,
        world_size=2, distributed=distributed)
    assert norm == 0 and not applied and torch.equal(parameter.detach(), before)
    Path(output + str(rank)).write_text("passed")
    distributed.destroy_process_group(control)
    distributed.destroy_process_group()


def test_real_gloo_global_update_and_zero_signal_momentum(tmp_path):
    import torch.multiprocessing as mp
    mp.spawn(_distributed_update_worker,
        args=(str(tmp_path / "rendezvous"), str(tmp_path / "result-")),
        nprocs=2, join=True)
    assert [(tmp_path / f"result-{rank}").read_text() for rank in range(2)] == ["passed", "passed"]


def test_update_readiness_single_rank_is_noop_and_multi_rank_requires_cpu_group():
    def unexpected():
        raise AssertionError("Single-rank readiness must not touch CUDA")
    wait_for_update_ranks(synchronize=unexpected)
    with pytest.raises(ValueError, match="Gloo"):
        wait_for_update_ranks(world_size=2, synchronize=unexpected)


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf")])
def test_update_readiness_rejects_invalid_timeout(timeout):
    with pytest.raises(ValueError, match="timeout"):
        wait_for_update_ranks(timeout_seconds=timeout)


@pytest.mark.parametrize("native_batch_size", [4, 2])
def test_real_trainer_closes_checkpoints_and_distinguishes_zero_update_iteration(tmp_path, monkeypatch, native_batch_size):
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
    import slime_plugins.dcc.hf_native_policy as native
    trainer = load_trainer()
    torch.manual_seed(19)
    model = get_peft_model(Qwen3MoeForCausalLM(Qwen3MoeConfig(
        vocab_size=32, hidden_size=16, intermediate_size=24, moe_intermediate_size=8,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        head_dim=8, num_experts=4, num_experts_per_tok=2, attention_dropout=0.0)),
        LoraConfig(r=2, lora_alpha=4, lora_dropout=0.0,
                   target_modules=["q_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM"))
    class Tokenizer:
        def save_pretrained(self, path):
            (path / "tokenizer_config.json").write_text("{}")
    class Policy:
        def __init__(self, *args, **kwargs):
            assert kwargs["batch_size"] == native_batch_size
        def bind_task(self, *args):
            return self
        async def close(self):
            pass
    class Runner:
        count = 0
        async def run(self, task, policy, **kwargs):
            self.count += 1
            result = episode(1 if self.count == 1 else 0,
                status="completed" if self.count == 1 else "model_protocol_failure",
                thought=(6 if self.count % 2 else 7,), json_ids=(8, 9))
            # Production uses the NativeHFCompletion dataclass; its fields are
            # serializable by EpisodeResult.to_dict and preserve JSON IDs too.
            step = result.steps[0]
            result.steps = [native.NativeHFCompletion(
                text="", prompt_ids=step.prompt_ids, completion_ids=step.completion_ids,
                thought_token_count=1, phase_lengths=(1, 2),
                thought_logprobs=step.thought_logprobs,
                thought_logprobs_source=step.thought_logprobs_source)]
            return result
    for name, value in (("RANK", "0"), ("LOCAL_RANK", "0"), ("WORLD_SIZE", "1")):
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(torch.cuda, "set_device", lambda _: None)
    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda: torch.get_rng_state())
    monkeypatch.setattr(native, "NativeHFPolicy", Policy)
    monkeypatch.setattr(trainer, "load_policy_model", lambda *a, **k: (model, Tokenizer()))
    monkeypatch.setattr(trainer, "make_runner", lambda args: Runner())
    args = SimpleNamespace(model="CPU-fixture-only", adapter=None, source_commit="0" * 40,
        output_dir=tmp_path / "training", steps=2, seed=47, lora_rank=2,
        quantization="none", learning_rate=0.01, logprob_backend="original",
        head_chunk_tokens=None, thought_tokens=8, json_tokens=8,
        max_prompt_tokens=32, max_steps=2, group_size=2, clip=0.2,
        max_grad_norm=1.0, sync_artifacts=False, save_every=1, replay_diagnostic="off",
        native_batch_size=native_batch_size)
    task = SimpleNamespace(id="task")
    asyncio.run(trainer.train(args, [task], [[0], [0]], 1))
    complete = json.loads((args.output_dir / "training_complete.json").read_text())
    assert complete["complete"] and complete["has_policy_gradient_updates"]
    assert complete["rollout_iterations"] == 2 and complete["optimizer_updates"] == 1
    records = [json.loads(line) for line in (args.output_dir / "train_rank0.jsonl").read_text().splitlines()]
    for iteration in (1, 2):
        phases = [r["event"] for r in records if r.get("iteration") == iteration]
        ordered = ["collect_begin", "collect_end", "signal", "replay_begin", "replay_end", "apply_begin", "apply_end", "step"]
        assert [event for event in phases if event in ordered] == ordered
    saves = [r for r in records if r["event"] in {"save_begin", "save_end"}]
    assert [r["event"] for r in saves] == ["save_begin", "save_end"] * 4
    assert all(r.get("seconds", 0) >= 0 for r in records)
    steps = [r for r in records if r["event"] == "step"]
    assert all(r["replay_diagnostic_mode"] == "off" and r["diagnostic_scored_segments"] == 0
               and r["maximum_behavior_replay_logprob_difference"] is None for r in steps)
    assert [r["optimizer_step_applied"] for r in steps] == [True, False]
    assert [r["optimizer_updates"] for r in steps] == [1, 1]
    from safetensors.torch import load_file
    first = load_file(args.output_dir / "iteration_0001/adapter_model.safetensors")
    final = load_file(args.output_dir / "final/adapter_model.safetensors")
    assert all(torch.equal(first[k], final[k]) for k in first)
    state = torch.load(args.output_dir / "final/training_state.pt", weights_only=False)
    assert state["optimizer_steps"] == 1 and state["rollout_iterations"] == 2
    saved_config = json.loads((args.output_dir / "final/training_config.json").read_text())
    assert saved_config["replay_diagnostic_mode"] == saved_config["arguments"]["replay_diagnostic"] == "off"
    assert saved_config["arguments"]["native_batch_size"] == native_batch_size
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("invalid", [None, (), [-2.0], [-2.0, float("nan")],
    [-2.0, float("inf")], [-2.0, True], [-2.0, "-3"], [-2.0, .1]])
def test_group_refuses_missing_malformed_or_nonfinite_actual_behavior_scores(invalid):
    failed = episode(0, status="model_protocol_failure")
    failed.steps[0].thought_logprobs = invalid
    with pytest.raises(ValueError, match="actual sampled thought behavior"):
        signal([episode(1), failed])


def test_group_rejects_reconstructed_score_source_even_for_constant_rewards():
    one = episode(0)
    one.steps[0].thought_logprobs_source = "teacher-forced-old"
    with pytest.raises(ValueError, match="actual sampled thought behavior"):
        signal([one, episode(0)])


def test_replay_uses_actual_behavior_denominator_and_reports_replay_discrepancy():
    trainer = load_trainer()
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(-1.0)
    segment = SimpleNamespace(prompt_ids=[1], completion_ids=[2], thought_logprobs=[-1.4])
    group = SimpleNamespace(segments=[[segment]], advantages=[-1.0])
    def scorer(model, prompt, completion):
        return model.weight.reshape(1)
    report = trainer.replay_group(model, group, scorer,
        group_size=2, max_steps=1, thought_tokens=1, clip=.2)
    # Negative advantage selects the unclipped exp(.4), not a reconstructed
    # ratio of one. Its loss and derivative are both exp(.4)/2.
    expected = torch.exp(torch.tensor(.4)) / 2
    assert report["policy_loss"] == pytest.approx(expected.item())
    assert model.weight.grad.item() == pytest.approx(expected.item())
    assert report["old_logprob_source"] == "actual-sampled-thought-behavior"
    assert report["maximum_behavior_replay_logprob_difference"] == pytest.approx(.4)
    assert report["maximum_behavior_current_logprob_difference"] == pytest.approx(.4)
    assert report["maximum_replay_current_logprob_difference"] == 0
    assert report["importance_ratio_min"] == pytest.approx(2 * expected.item())
    assert report["importance_ratio_max"] == pytest.approx(2 * expected.item())
    assert report["importance_ratio_outside_clip_fraction"] == 1


@pytest.mark.parametrize("mode", ["full", "first", "off"])
def test_constant_group_retains_behavior_segments_without_inventing_replay_evidence(mode):
    trainer = load_trainer()
    group = signal([episode(0), episode(0, status="model_protocol_failure")])
    assert len(group.segments) == 2 and all(segment for segment in group.segments)
    model = torch.nn.Linear(1, 1)
    def forbidden(*args):
        raise AssertionError("An exact zero group must not replay")
    report = trainer.replay_group(model, group, forbidden,
        group_size=2, max_steps=2, thought_tokens=8, clip=.2, replay_diagnostic=mode)
    assert report["scored_tokens"] == 0
    assert report["maximum_behavior_replay_logprob_difference"] is None
    assert report["importance_ratio_min"] is None
    assert report["replay_diagnostic_mode"] == mode
    assert report["diagnostic_scored_segments"] == report["diagnostic_scored_tokens"] == 0


def test_replay_diagnostic_modes_preserve_clipped_categorical_loss_and_gradients():
    """Actual-old numerical drift clips both signs; only diagnostics get skipped."""
    trainer = load_trainer()
    logits = torch.tensor([-.3, .1, -.2, .4])
    current = logits.log_softmax(0)
    ratios = [1.4, .7, .7, 1.4]
    old = [float(current[i]) - math.log(ratio) for i, ratio in enumerate(ratios)]
    segments = [SimpleNamespace(prompt_ids=[10 + i], completion_ids=[i],
                                thought_logprobs=[old[i]]) for i in range(4)]
    group = SimpleNamespace(segments=[segments[:2], segments[2:]], advantages=[1., -1.])
    # Pos ratio1.4 and neg ratio.7 are clipped constants. The remaining two
    # derivatives are -.7*dlogp[1]/4 and +1.4*dlogp[3]/4.
    expected_gradient = -.7 * logits.softmax(0)
    expected_gradient[1] -= .7
    expected_gradient[3] += 1.4
    expected_gradient /= 4
    reports, gradients = {}, {}
    for mode, diagnostic_count in [("full", 4), ("first", 1), ("off", 0)]:
        model = torch.nn.Linear(1, 4, bias=False)
        with torch.no_grad():
            model.weight[:, 0].copy_(logits)
        calls = []
        def scorer(value, prompt, completion):
            index = completion[0]
            calls.append((value.training, torch.is_grad_enabled(), index))
            # Deliberately distinct diagnostic numerics. They must never be
            # substituted for the actual generation behavior denominator.
            delta = .04 * (index + 1) if not value.training else 0.
            return value.weight[:, 0].log_softmax(0)[[index]] + delta
        report = trainer.replay_group(model, group, scorer, group_size=2,
            max_steps=2, thought_tokens=1, clip=.2, replay_diagnostic=mode)
        assert len(calls) == 4 + diagnostic_count
        assert sum(not training and not grad for training, grad, _ in calls) == diagnostic_count
        assert [(grad, index) for training, grad, index in calls if training] == [(True, i) for i in range(4)]
        assert report["scored_segments"] == report["scored_tokens"] == 4
        assert report["diagnostic_scored_segments"] == report["diagnostic_scored_tokens"] == diagnostic_count
        assert report["replay_diagnostic_mode"] == mode
        assert report["policy_loss"] == pytest.approx(.075, abs=1e-7)
        assert report["importance_ratio_min"] == pytest.approx(.7)
        assert report["importance_ratio_max"] == pytest.approx(1.4)
        assert report["importance_ratio_outside_clip_fraction"] == 1.
        assert report["maximum_behavior_current_logprob_difference"] == pytest.approx(-math.log(.7))
        if mode == "off":
            assert report["maximum_behavior_replay_logprob_difference"] is None
            assert report["maximum_replay_current_logprob_difference"] is None
            assert report["replay_diagnostic_scope"] == "not-collected"
        else:
            assert report["maximum_replay_current_logprob_difference"] == pytest.approx(.16 if mode == "full" else .04)
            assert report["replay_diagnostic_scope"] == ("all-nonzero-advantage-segments" if mode == "full"
                else "first-nonzero-advantage-segment-per-group")
        gradients[mode] = model.weight.grad[:, 0]
        torch.testing.assert_close(gradients[mode], expected_gradient, rtol=1e-6, atol=1e-7)
        reports[mode] = report
    assert torch.equal(gradients["full"], gradients["first"]) and torch.equal(gradients["full"], gradients["off"])
    assert len({report["policy_loss"] for report in reports.values()}) == 1
    assert not torch.cuda.is_initialized()


def test_first_diagnostic_uses_first_nonzero_advantage_segment_only():
    trainer = load_trainer()
    ignored = SimpleNamespace(prompt_ids=[99], completion_ids=[99], thought_logprobs=[-1.])
    active = SimpleNamespace(prompt_ids=[1], completion_ids=[2], thought_logprobs=[-1.])
    group = SimpleNamespace(segments=[[ignored], [active]], advantages=[0., 1.])
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(-1.)
    calls = []
    def scorer(value, prompt, completion):
        assert prompt == [1] and completion == [2]
        calls.append((value.training, torch.is_grad_enabled()))
        return value.weight.reshape(1)
    report = trainer.replay_group(model, group, scorer, group_size=2,
        max_steps=1, thought_tokens=1, clip=.2, replay_diagnostic="first")
    assert calls == [(False, False), (True, True)]
    assert report["diagnostic_scored_segments"] == report["scored_segments"] == 1


@pytest.mark.parametrize("mode", ["full", "first", "off"])
@pytest.mark.parametrize("failure", ["behavior_nan", "behavior_shape", "current_nan", "ratio_inf"])
def test_every_diagnostic_mode_retains_actual_behavior_current_and_ratio_checks(mode, failure):
    trainer = load_trainer()
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(-1.)
    behavior = {"behavior_nan": [float("nan")], "behavior_shape": [], "current_nan": [-1.], "ratio_inf": [-1000.]}[failure]
    segment = SimpleNamespace(prompt_ids=[1], completion_ids=[2], thought_logprobs=behavior)
    group = SimpleNamespace(segments=[[segment]], advantages=[1.])
    def scorer(value, prompt, completion):
        result = value.weight.reshape(1)
        return result * float("nan") if failure == "current_nan" and value.training else result
    with pytest.raises(FloatingPointError):
        trainer.replay_group(model, group, scorer, group_size=2,
            max_steps=1, thought_tokens=1, clip=.2, replay_diagnostic=mode)


def test_invalid_replay_diagnostic_fails_before_any_forward():
    trainer = load_trainer()
    model = torch.nn.Linear(1, 1)
    def forbidden(*args):
        raise AssertionError("Invalid mode may not run a decoder")
    with pytest.raises(ValueError, match="full, first or off"):
        trainer.replay_group(model, signal([episode(1), episode(0)]), forbidden,
            group_size=2, max_steps=2, thought_tokens=8, clip=.2, replay_diagnostic="unknown")
