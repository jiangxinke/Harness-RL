"""CPU-only matched HF evaluator: real runner/scorer, invented model outputs."""
from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path
from dataclasses import replace

import pytest
import torch


def load_evaluator():
    path = Path(__file__).resolve().parents[1] / "scripts/dcc_online_evaluate.py"
    spec = importlib.util.spec_from_file_location("dcc_online_evaluate_test_subject", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def evaluator(monkeypatch):
    value = load_evaluator()
    from slime_plugins.dcc.episode import ModelProtocolFailure
    from slime_plugins.dcc.hf_native_policy import NativeHFCompletion
    from slime_plugins.dcc.prompt_history import PromptHistoryBudgetExceeded

    value.observed, value.loads, value.policies, value.journals = [], [], [], []
    original_runner = value.make_runner
    def make_runner(args):
        runner = original_runner(args)
        original = runner.run
        async def run(task, policy, **kwargs):
            assert task.gold is None and kwargs == {"evaluation": True, "score_train": False}
            value.observed.append((task.id, task.split))
            return await original(task, policy, **kwargs)
        runner.run = run
        return runner
    monkeypatch.setattr(value, "make_runner", make_runner)
    def load(model, **kwargs):
        value.loads.append((model, kwargs))
        fixture = torch.nn.Module()
        fixture.base_weight = torch.nn.Parameter(torch.ones(2, dtype=torch.float32))
        if kwargs["adapter"]:
            fixture.lora_A = torch.nn.Parameter(torch.ones(2, dtype=torch.float32))
        fixture.dcc_quantization_report = {"mode": "nf4", "expert_4bit_coverage": 1.0}
        return fixture, object()
    monkeypatch.setattr(value, "load_policy_model", load)
    monkeypatch.setattr(value, "validate_source", lambda source: None)
    class Journal(value.OrderedDurableJSONLWriter):
        def __init__(self, path):
            super().__init__(path)
            value.journals.append(self)
    monkeypatch.setattr(value, "OrderedDurableJSONLWriter", Journal)
    class Policy:
        def __init__(self, model, tokenizer, **kwargs):
            self.options = kwargs
            self.closed = False
            value.policies.append(self)
        def bind_task(self, task_id, rank):
            self.task_id = task_id
            return self
        async def generate_structured(self, messages, **kwargs):
            assert kwargs["max_new_tokens"] == self.options["thought_tokens"] + self.options["json_tokens"]
            assert kwargs["temperature"] == 1.0
            assert all("secret-heldout-label" not in message["content"] for message in messages)
            if self.task_id.endswith("context"):
                raise PromptHistoryBudgetExceeded("invented_CPU_context_limit",
                    diagnostics={"original_prompt_tokens": 50, "minimum_prompt_tokens": 50,
                                 "effective_prompt_limit": 40},
                    original_prompt_ids=[1, 2])
            random_value = torch.randint(0, 1000000, (1,)).item()
            self.options["trace_sink"]({"task_id": self.task_id, "random_value": random_value})
            completion = NativeHFCompletion(
                '{"plan":"Read","checks":"Direct evidence","answer":[{"text":"invented CPU source clause","confidence":0.9}]}',
                [1, 2], [3, 4, 5], thought_token_count=2, phase_lengths=(2, 1))
            if self.task_id.endswith("failure"):
                raise ModelProtocolFailure("invented CPU truncated thought", completion=completion,
                                           protocol="CPU-fixture", trace_id="CPU-fixture")
            return completion
        async def close(self):
            self.closed = True
    monkeypatch.setattr(value, "NativeHFPolicy", Policy)
    original_score = value.evaluate_results
    def evaluate(tasks, predictions):
        assert value.journals and all(journal._closed for journal in value.journals)
        assert all(policy.closed for policy in value.policies)
        assert all(task.gold is not None for task in tasks)
        return original_score(tasks, predictions)
    monkeypatch.setattr(value, "evaluate_results", evaluate)
    return value


def tasks(value):
    from slime_plugins.dcc.data import Task
    return [Task("cpu-success", "cuad", "validation", "Extract a clause.",
                 "invented CPU source clause", ["invented CPU source clause"]),
            Task("cpu-failure", "cuad", "validation", "Extract another category.",
                 "unrelated invented CPU source clause", []),
            Task("cpu-context", "cuad", "validation", "Extract a final category.",
                 "unrelated invented source", ["secret-heldout-label"])]


def arguments(value, output, **overrides):
    args = value.parser().parse_args(["--model", "invented-CPU-model", "--data", "invented-CPU-tasks.jsonl",
        "--output-dir", str(output), "--source-commit", "a" * 40,
        "--thought-tokens", "4", "--json-tokens", "256", "--max-steps", "1"])
    for name, setting in overrides.items():
        setattr(args, name, setting)
    return args


def test_real_runner_journals_and_cuad_score_keep_every_failure(evaluator, tmp_path):
    args = arguments(evaluator, tmp_path / "eval")
    all_tasks = tasks(evaluator)
    asyncio.run(evaluator.generate(args, all_tasks))
    # Prediction generation itself must never consume the real gold scorer.
    assert not (args.output_dir / "report.json").exists()
    predictions = evaluator.read_jsonl(args.output_dir / "predictions_rank0.jsonl")
    assert evaluator.observed == [(task.id, "validation") for task in all_tasks]
    assert [row["status"] for row in predictions] == ["completed", "model_protocol_failure", "context_budget_failure"]
    assert all("reward" not in row and not (evaluator.SCORE_FIELDS & row["metrics"].keys()) for row in predictions)
    assert evaluator.loads[0][1] == dict(adapter=None, trainable=False, rank=0, quantization="nf4", compute_profile="legacy")
    report = evaluator.score(args, all_tasks)
    assert report["tasks"] == report["metric_task_denominator"] == 3
    assert report["prediction_coverage"] == 1 and report["complete"] is False
    assert report["failure_zero_macro_f1"] == pytest.approx(1 / 3)
    assert report["cuad"]["micro_f1"] == pytest.approx(2 / 3)
    assert report["scored_after_journals_closed"] is True
    assert report["comparison"]["thought_sampling"] == dict(do_sample=True, temperature=0.6, top_p=0.95, top_k=20)
    assert not torch.cuda.is_initialized()


def test_sharding_preserves_task_seeds_and_base_adapter_configuration(evaluator, tmp_path):
    all_tasks = tasks(evaluator)
    base = arguments(evaluator, tmp_path / "base", profile="greedy")
    asyncio.run(evaluator.generate(base, all_tasks))
    evaluator.score(base, all_tasks)
    adapter = arguments(evaluator, tmp_path / "adapter", adapter="invented-online-final-adapter",
                        base_output_dir=base.output_dir, shards=2, profile="greedy")
    for rank in (1, 0):
        adapter.shard_index = rank
        asyncio.run(evaluator.generate(adapter, all_tasks))
    base_rows = evaluator.read_jsonl(base.output_dir / "native_traces_rank0.jsonl")
    trained_rows = [row for rank in range(2) for row in evaluator.read_jsonl(adapter.output_dir / f"native_traces_rank{rank}.jsonl")]
    assert {row["task_id"]: row["random_value"] for row in base_rows} == {row["task_id"]: row["random_value"] for row in trained_rows}
    assert evaluator.manifest(base, all_tasks)["comparison"] == evaluator.manifest(adapter, all_tasks)["comparison"]
    assert evaluator.manifest(base, all_tasks)["condition"] != evaluator.manifest(adapter, all_tasks)["condition"]
    report = evaluator.score(adapter, all_tasks)
    assert report["tasks"] == 3 and len(report["shard_receipts"]) == 2
    assert report["comparison"]["thought_sampling_profile"] == "greedy"
    assert report["matched_base"]["failure_zero_macro_f1_delta"] == 0
    assert report["matched_base"]["complete"] is False  # Context failure retained, never hidden.
    assert all(policy.options["sampling_profile"] == "greedy" for policy in evaluator.policies)


def test_merge_rejects_missing_incomplete_or_changed_shard(evaluator, tmp_path):
    all_tasks = tasks(evaluator)
    args = arguments(evaluator, tmp_path / "eval", shards=2)
    asyncio.run(evaluator.generate(args, all_tasks))
    with pytest.raises(ValueError, match="Missing or extra"):
        evaluator.score(args, all_tasks)
    args.shard_index = 1
    asyncio.run(evaluator.generate(args, all_tasks))
    marker = args.output_dir / "rank1.done.json"
    original = json.loads(marker.read_text())
    marker.write_text(json.dumps(dict(original, journals_closed=False)))
    with pytest.raises(ValueError, match="configuration, ownership or completed coverage"):
        evaluator.score(args, all_tasks)
    marker.write_text(json.dumps(original))
    args.thought_tokens += 1
    with pytest.raises(ValueError, match="configuration, ownership or completed coverage"):
        evaluator.score(args, all_tasks)


def test_cli_generate_and_score_only_do_not_load_gpu_for_scoring(evaluator, tmp_path):
    data = tmp_path / "invented-tasks.jsonl"
    data.write_text("".join(json.dumps(task.to_dict()) + "\n" for task in tasks(evaluator)))
    output = tmp_path / "eval"
    command = ["--model", "invented-CPU-model", "--data", str(data), "--output-dir", str(output),
               "--source-commit", "a" * 40, "--max-steps", "1"]
    assert evaluator.main(command) == 0
    assert len(evaluator.loads) == 1
    (output / "report.json").unlink()
    assert evaluator.main(command + ["--score-only"]) == 0
    assert len(evaluator.loads) == 1
    assert not torch.cuda.is_initialized()


def test_cli_single_shard_parallel_generation_is_explicitly_generation_only(evaluator, tmp_path, capsys):
    data = tmp_path / "invented-tasks.jsonl"
    data.write_text("".join(json.dumps(task.to_dict()) + "\n" for task in tasks(evaluator)[:2]))
    common = ["--model", "invented-CPU-model", "--data", str(data),
              "--source-commit", "a" * 40, "--max-steps", "1"]
    base_dir, adapter_dir = tmp_path / "base", tmp_path / "adapter"
    base = common + ["--output-dir", str(base_dir)]
    adapter = common + ["--output-dir", str(adapter_dir), "--adapter", "invented-online-adapter",
                        "--base-output-dir", str(base_dir)]
    # Adapter generation can finish before the base exists, with an honest
    # generation-only success; this is not a successful scoring pipeline yet.
    assert evaluator.main(adapter + ["--generate-only"]) == 0
    stage = json.loads(capsys.readouterr().out)
    assert stage["generation_complete"] is True and stage["scoring_complete"] is False
    assert (adapter_dir / "rank0.done.json").exists() and not (adapter_dir / "report.json").exists()
    assert evaluator.main(base + ["--generate-only"]) == 0
    capsys.readouterr()
    assert not (base_dir / "report.json").exists()
    assert len(evaluator.loads) == 2
    assert evaluator.main(base + ["--score-only"]) == 0
    capsys.readouterr()
    assert evaluator.main(adapter + ["--score-only"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["matched_base"]["complete"] is True
    assert len(evaluator.loads) == 2
    with pytest.raises(SystemExit):
        evaluator.main(adapter + ["--generate-only", "--score-only"])
    assert len(evaluator.loads) == 2 and not torch.cuda.is_initialized()


def test_cli_rejects_web_and_invalid_shards_before_model_loading(evaluator, tmp_path):
    data = tmp_path / "invented-tasks.jsonl"
    data.write_text(json.dumps(dict(id="web", benchmark="widesearch", split="test", prompt="Find rows", gold=[])) + "\n")
    command = ["--model", "invented-CPU-model", "--data", str(data), "--output-dir", str(tmp_path / "eval"),
               "--source-commit", "a" * 40]
    with pytest.raises(SystemExit):
        evaluator.main(command)
    with pytest.raises(SystemExit):
        evaluator.main(command + ["--shards", "2", "--shard-index", "2"])
    assert not evaluator.loads


def test_generation_is_exclusive_and_never_resumes(evaluator, tmp_path):
    args = arguments(evaluator, tmp_path / "eval")
    all_tasks = tasks(evaluator)
    asyncio.run(evaluator.generate(args, all_tasks))
    with pytest.raises(FileExistsError):
        asyncio.run(evaluator.generate(args, all_tasks))
    assert len(evaluator.loads) == 1


def test_public_identity_binds_inputs_without_using_heldout_labels(evaluator):
    original = tasks(evaluator)
    identity = evaluator.public_task_identity(original)
    relabeled = [replace(task, gold=["a different secret heldout label"]) for task in original]
    assert evaluator.public_task_identity(relabeled) == identity
    for name in ("prompt", "document"):
        changed = [replace(original[0], **{name: getattr(original[0], name) + " changed"}), *original[1:]]
        assert evaluator.public_task_identity(changed) != identity


@pytest.mark.parametrize("change", ["extra", "duplicate_row", "duplicate_shard", "precision", "packages", "missing_trace", "unknown_status"])
def test_merge_refuses_extra_duplicate_and_heterogeneous_shards(evaluator, tmp_path, change):
    all_tasks = tasks(evaluator)
    args = arguments(evaluator, tmp_path / "eval", shards=2)
    for rank in range(2):
        args.shard_index = rank
        asyncio.run(evaluator.generate(args, all_tasks))
    if change == "extra":
        (args.output_dir / "config_rank2.json").write_text("{}")
    elif change == "duplicate_row":
        path = args.output_dir / "predictions_rank1.jsonl"
        text = path.read_text()
        path.write_text(text + text)
    elif change == "duplicate_shard":
        path = args.output_dir / "config_rank1.json"
        path.write_text((args.output_dir / "config_rank0.json").read_text())
    elif change == "precision":
        path = args.output_dir / "rank1.done.json"
        row = json.loads(path.read_text())
        row["precision"]["base_parameter_dtypes"] = {"torch.bfloat16": 2}
        path.write_text(json.dumps(row))
    elif change == "packages":
        path = args.output_dir / "config_rank1.json"
        row = json.loads(path.read_text())
        row["packages"]["torch"] = "invented-incompatible-version"
        path.write_text(json.dumps(row))
    elif change == "missing_trace":
        (args.output_dir / "native_traces_rank1.jsonl").unlink()
    else:
        path = args.output_dir / "predictions_rank1.jsonl"
        row = evaluator.read_jsonl(path)[0]
        row["status"] = "invented_unknown_status"
        path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError):
        evaluator.score(args, all_tasks)
    assert not (args.output_dir / "report.json").exists()


@pytest.mark.parametrize("setting,replacement", [("seed", 999), ("thought_tokens", 5), ("profile", "greedy"),
    ("initial_chunks", 1), ("retrieval_context_chars", 100), ("model", "different-CPU-model"), ("quantization", "none")])
def test_adapter_scoring_requires_actual_matching_base(evaluator, tmp_path, setting, replacement):
    all_tasks = tasks(evaluator)[:2]
    base = arguments(evaluator, tmp_path / "base")
    asyncio.run(evaluator.generate(base, all_tasks))
    evaluator.score(base, all_tasks)
    adapter = arguments(evaluator, tmp_path / "adapter", adapter="invented-online-adapter",
                        base_output_dir=base.output_dir, **{setting: replacement})
    asyncio.run(evaluator.generate(adapter, all_tasks))
    with pytest.raises(ValueError, match="Base and adapter differ"):
        evaluator.score(adapter, all_tasks)
    assert not (adapter.output_dir / "report.json").exists()


def test_adapter_generation_can_finish_before_base_but_scoring_must_wait(evaluator, tmp_path):
    all_tasks = tasks(evaluator)[:2]
    base = arguments(evaluator, tmp_path / "base")
    adapter = arguments(evaluator, tmp_path / "adapter", adapter="invented-online-adapter",
                        base_output_dir=base.output_dir)
    asyncio.run(evaluator.generate(adapter, all_tasks))
    with pytest.raises(FileNotFoundError):
        evaluator.score(adapter, all_tasks)
    assert not (adapter.output_dir / "report.json").exists()
    asyncio.run(evaluator.generate(base, all_tasks))
    evaluator.score(base, all_tasks)
    report = evaluator.score(adapter, all_tasks)
    assert report["matched_base"]["complete"] is True


def test_adapter_score_requires_base_reference_and_identical_scoring_labels(evaluator, tmp_path):
    all_tasks = tasks(evaluator)[:2]
    base = arguments(evaluator, tmp_path / "base")
    asyncio.run(evaluator.generate(base, all_tasks))
    evaluator.score(base, all_tasks)
    adapter = arguments(evaluator, tmp_path / "adapter", adapter="invented-online-adapter")
    asyncio.run(evaluator.generate(adapter, all_tasks))
    with pytest.raises(ValueError, match="requires --base-output-dir"):
        evaluator.score(adapter, all_tasks)
    adapter.base_output_dir = base.output_dir
    changed_labels = [replace(all_tasks[0], gold=["different scoring reference"]), all_tasks[1]]
    with pytest.raises(ValueError, match="Base and adapter differ"):
        evaluator.score(adapter, changed_labels)
    assert not (adapter.output_dir / "report.json").exists()


@pytest.mark.parametrize("algorithm,updates,marked_online", [("actual-online", 2, True),
    ("actual-online", 0, False), ("offline-DPO-control", 2, False)])
def test_adapter_provenance_separates_training_and_evaluation_source(evaluator, tmp_path, algorithm, updates, marked_online):
    adapter = tmp_path / "invented-adapter"
    adapter.mkdir()
    actual_algorithm = evaluator.ALGORITHM if algorithm == "actual-online" else algorithm
    (adapter / "training_config.json").write_text(json.dumps(dict(algorithm=actual_algorithm,
        source_commit="b" * 40, rollout_iterations=4, optimizer_updates=updates,
        sft_auxiliary=False, offline_preferences=False)))
    (adapter / "checkpoint_complete.json").write_text(json.dumps(dict(complete=True, optimizer_steps=updates)))
    args = arguments(evaluator, tmp_path / "eval", adapter=str(adapter))
    config = evaluator.manifest(args, tasks(evaluator))
    assert config["comparison"]["source_commit"] == "a" * 40
    assert config["condition"]["training"]["training_source_commit"] == "b" * 40
    assert config["condition"]["training"]["algorithm"] == actual_algorithm
    assert config["condition"]["training"]["optimizer_updates"] == updates
    assert config["condition"]["online_policy_gradient_updates_recorded"] is marked_online


def test_shared_loader_cpu_preparation_preserves_nonquantized_and_lora_dtypes(monkeypatch, tmp_path):
    """Real shared loader/PEFT; tiny CPU BF16 weights, no actual NF4 matrices.

    Only model acquisition and the NF4 matrix inspector are stubbed. Actual
    prepare_model_for_kbit_training and adapter save/reload remain in use.
    This establishes nonquantized-layer/adapter precision, not GPU NF4 kernels.
    """
    pytest.importorskip("peft")
    transformers = pytest.importorskip("transformers")
    value = load_evaluator()
    from slime_plugins.dcc import hf_policy, hf_quantization
    calls = []
    def acquire(model_name, **kwargs):
        calls.append(kwargs)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(19)
            config = transformers.Qwen3MoeConfig(vocab_size=32, hidden_size=16, intermediate_size=24,
                moe_intermediate_size=8, num_hidden_layers=1, num_attention_heads=2,
                num_key_value_heads=1, head_dim=8, num_experts=4, num_experts_per_tok=2,
                attention_dropout=0.0)
            return transformers.Qwen3MoeForCausalLM(config).to(dtype=torch.bfloat16)
    class Tokenizer:
        pad_token_id = 0
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", staticmethod(acquire))
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", staticmethod(lambda name: Tokenizer()))
    monkeypatch.setattr(torch.cuda, "set_device", lambda rank: None)
    monkeypatch.setattr(hf_quantization, "nf4_load_kwargs", lambda: {})
    monkeypatch.setattr(hf_quantization, "quantization_report", lambda model: {"mode": "CPU-nonquantized-fixture"})
    trained, _ = hf_policy.load_policy_model("invented-CPU-model", trainable=True, quantization="nf4", lora_rank=2)
    with torch.no_grad():
        for name, parameter in trained.named_parameters():
            if "lora_B" in name:
                parameter.fill_(0.04)
    trained.eval()
    adapter_dir = tmp_path / "actual-tiny-CPU-adapter"
    trained.save_pretrained(adapter_dir)
    base, _ = hf_policy.load_policy_model("invented-CPU-model", trainable=False, quantization="nf4")
    reloaded, _ = hf_policy.load_policy_model("invented-CPU-model", adapter=str(adapter_dir), trainable=False, quantization="nf4")
    base.eval()
    reloaded.eval()
    train_precision = value.precision_observation(trained)
    base_precision = value.precision_observation(base)
    reload_precision = value.precision_observation(reloaded)
    assert train_precision["base_parameter_dtypes"] == base_precision["base_parameter_dtypes"] == reload_precision["base_parameter_dtypes"]
    assert set(base_precision["base_parameter_dtypes"]) == {"torch.float32"}
    assert train_precision["adapter_parameter_dtypes"] == reload_precision["adapter_parameter_dtypes"]
    assert set(reload_precision["adapter_parameter_dtypes"]) == {"torch.float32"}
    assert all(call["dtype"] == torch.bfloat16 and call["attn_implementation"] == "sdpa" for call in calls)
    ids = torch.tensor([[3, 4, 5]])
    with torch.no_grad():
        expected = trained(ids, use_cache=False).logits
        actual = reloaded(ids, use_cache=False).logits
        unchanged = base(ids, use_cache=False).logits
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    assert not torch.allclose(actual, unchanged)
    assert not torch.cuda.is_initialized()
