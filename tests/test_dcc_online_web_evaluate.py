"""CPU real-runner tests; model outputs and Web observations are invented."""
from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import uuid

import pytest
import torch

from slime_plugins.dcc.data import Task
from slime_plugins.dcc.hf_native_policy import NativeHFCompletion, END_THINK, PROTOCOL
from slime_plugins.dcc.episode import ModelProtocolFailure
from slime_plugins.dcc.prompt_history import PromptHistoryBudgetExceeded
from slime_plugins.dcc.tools import WebTools
from slime_plugins.dcc.web_hf_evaluation import load_public_tasks, public_task, task_identity

URL = "https://example.com/invented-cpu-source"
EVALUATION = {"required": ["name", "country"], "unique_columns": ["name"],
              "eval_pipeline": {"country": {"metric": ["llm_judge"]}}}


def references(ids=("success", "failure", "context"), split="validation"):
    return [Task("cpu-" + name, "widesearch", split, "List the invented public entities and countries.",
        gold=[{"name": "Alpha", "country": "AA"}],
        metadata={"evaluation": deepcopy(EVALUATION), "source": "independent_curriculum",
                  "curriculum_version": "invented-CPU-only", "teacher_query": "private-teacher-hint",
                  "private_label": "secret-heldout-label"}) for name in ids]


def write_tasks(path, tasks):
    path.write_text("".join(json.dumps(task.to_dict()) + "\n" for task in tasks))


def arguments(evaluator, output, **kwargs):
    args = evaluator.parser().parse_args(["--model", "invented-CPU-model", "--data", "unused-CPU-data.jsonl",
        "--output-dir", str(output), "--source-commit", "a" * 40,
        "--max-steps", "4", "--thought-tokens", "4", "--json-tokens", "256"])
    for key, value in kwargs.items():
        setattr(args, key, value)
    return args


@pytest.fixture
def evaluator(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts/dcc_online_web_evaluate.py"
    spec = importlib.util.spec_from_file_location("dcc_online_web_evaluate_subject", path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    value.journals, value.policies, value.loads, value.observed, value.metric_calls = [], [], [], [], []

    class Tokenizer:
        chat_template = "invented CPU template"
        pad_token_id = 2
        def __len__(self):
            return 200000
        def apply_chat_template(self, messages, **kwargs):
            text = json.dumps(messages, sort_keys=True)
            return [ord(char) + 100 for char in text] + [151667]
        def decode(self, ids, **kwargs):
            return "".join(chr(token - 100) for token in ids)

    class Web(WebTools):
        web_cached_rows = True
        async def web_search(self, query, *, count=8, **kwargs):
            return {"query": query, "results": [{"url": URL, "title": "CPU source", "snippet": "Alpha AA"}],
                    "backend": "public_web", "configured_backend": "public_web", "search_scope": "web"}
        async def fetch_url(self, url, **kwargs):
            return {"url": url, "text": "Alpha AA", "title": "Invented CPU source", "start": 0, "end": 8,
                    "total_chars": 8, "truncated": False}
        def citation_validity(self, answer):
            return 1.

    original_runner = value.make_runner
    def runner(args):
        result = original_runner(args)
        result.web_tools_factory = Web
        original = result.run
        async def run(task, policy, **kwargs):
            assert task.gold is None and set(task.metadata) == {"evaluation"}
            assert task.metadata["evaluation"] == {"required": ["name", "country"]}
            assert kwargs == {"evaluation": True, "score_train": False}
            value.observed.append((task.id, task.split))
            return await original(task, policy, **kwargs)
        result.run = run
        return result
    monkeypatch.setattr(value, "make_runner", runner)
    monkeypatch.setattr(value, "validate_source", lambda commit: None)

    def load(model, **kwargs):
        value.loads.append(kwargs)
        model = torch.nn.Module()
        model.base_weight = torch.nn.Parameter(torch.ones(2))
        if kwargs["adapter"]:
            model.lora_A = torch.nn.Parameter(torch.ones(2))
        model.config = SimpleNamespace(to_dict=lambda: {"model_type": "invented-CPU"}, _commit_hash="CPU-fixture")
        return model, Tokenizer()
    monkeypatch.setattr(value, "load_policy_model", load)

    class Journal(value.OrderedDurableJSONLWriter):
        def __init__(self, path):
            super().__init__(path)
            value.journals.append(self)
    monkeypatch.setattr(value, "OrderedDurableJSONLWriter", Journal)

    class Policy:
        def __init__(self, model, tokenizer, **kwargs):
            self.tokenizer, self.options = tokenizer, kwargs
            self.eos_ids, self.suffix_ids, self.worker = [151645], [151667], None
            self.closed = True
            value.policies.append(self)
        def bind_task(self, task_id, rank):
            self.task_id, self.turn, self.closed = task_id, 0, False
            return self
        async def generate_structured(self, messages, *, json_schema, **kwargs):
            assert "secret-heldout-label" not in json.dumps(messages)
            assert "private-teacher-hint" not in json.dumps(messages)
            if self.task_id.endswith("context"):
                raise PromptHistoryBudgetExceeded("invented-CPU-limit", diagnostics={
                    "original_prompt_tokens": 50, "minimum_prompt_tokens": 50,
                    "effective_prompt_limit": 40}, original_prompt_ids=[])
            if self.task_id.endswith("timeout"):
                await asyncio.sleep(.06)
            thought = [10 + torch.randint(0, 10, (1,)).item(), END_THINK]
            action = ({"tool": "web_search", "arguments": {"query": "invented public entities", "count": 8}}
                      if self.turn == 0 else {"answer": [{"name": "Alpha", "country": "AA", "citations": [URL]}]})
            if self.task_id.endswith("wrong") and self.turn:
                action["answer"][0]["country"] = "AB"
            text = json.dumps({"plan": "Read the requested entities", "checks": "Use actual read evidence", **action})
            final = [ord(char) + 100 for char in text] + self.eos_ids
            status = "completed"
            if self.task_id.endswith("failure"):
                thought, final, text, status = [11, 12, 13, 14], [], "", "thought_did_not_stop_on_real_close"
            trace_id = uuid.uuid4().hex
            prompt = self.tokenizer.apply_chat_template(messages)
            completion = NativeHFCompletion(text=text, prompt_ids=prompt, completion_ids=thought + final,
                thought_token_count=len(thought), phase_lengths=(len(thought), len(final)), native_status=status,
                finish_reason="stop" if final else "length", trace_id=trace_id)
            self.options["trace_sink"](dict(protocol=PROTOCOL, task_id=self.task_id, trace_id=trace_id,
                prompt_ids=prompt, thought_ids=thought, json_ids=final, thought_token_count=len(thought),
                status=status, final_text=text, json_public_schema=deepcopy(json_schema),
                thought_sampling_profile=self.options["sampling_profile"], thought_logprobs=None,
                thought_logprobs_source=None))
            self.turn += 1
            if status != "completed":
                raise ModelProtocolFailure(status, completion=completion, protocol=PROTOCOL, trace_id=trace_id)
            return completion
        async def close(self):
            self.closed, self.worker = True, None
    monkeypatch.setattr(value, "NativeHFPolicy", Policy)

    metric = value.evaluate_widesearch
    def score(*args, **kwargs):
        assert value.journals and all(writer._closed for writer in value.journals)
        assert all(policy.closed for policy in value.policies)
        assert kwargs == {"judge": None}
        value.metric_calls.append(args)
        return metric(*args, **kwargs)
    monkeypatch.setattr(value, "evaluate_widesearch", score)
    return value


def generate(evaluator, args, tasks, tmp_path):
    data = tmp_path / "prepared-tasks.jsonl"
    write_tasks(data, tasks)
    public = load_public_tasks(data)
    asyncio.run(evaluator.generate(args, public))
    return public


def make_adapter(path):
    path.mkdir()
    (path / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "invented-CPU-model"}))
    (path / "adapter_model.safetensors").write_bytes(b"invented CPU bytes; not real tensors")
    return str(path)


def test_public_loader_discards_labels_private_hints_and_scoring_rules(tmp_path):
    data = tmp_path / "tasks.jsonl"
    tasks = references()
    write_tasks(data, tasks)
    public = load_public_tasks(data)
    assert all(task.gold is None for task in public)
    assert all(set(task.metadata) == {"evaluation", "source", "curriculum_version"} for task in public)
    assert all(task.metadata["evaluation"] == {"required": ["name", "country"]} for task in public)
    assert "secret-heldout-label" not in repr(public) and "private-teacher-hint" not in repr(public)
    assert task_identity(public) == task_identity(tasks)
    changed = [replace(public[0], metadata={"evaluation": {"required": ["country", "name"]}}), *public[1:]]
    assert task_identity(changed) != task_identity(public)


def test_real_runner_owned_journals_zero_all_failures_before_proxy_score(evaluator, tmp_path):
    tasks = references()
    args = arguments(evaluator, tmp_path / "run")
    generate(evaluator, args, tasks, tmp_path)
    assert not evaluator.metric_calls and not (args.output_dir / "report.json").exists()
    rows = evaluator.read_jsonl(args.output_dir / "predictions_rank0.jsonl")
    assert [row["status"] for row in rows] == ["completed", "model_protocol_failure", "context_budget_failure"]
    assert all("reward" not in row for row in rows)
    report = evaluator.score(args, tasks)
    assert report["widesearch"]["f1_by_row"] == pytest.approx(1 / 3)
    assert report["widesearch"]["f1_by_item"] == pytest.approx(1 / 3)
    assert report["metric_task_denominator"] == report["widesearch"]["metric_denominator"] == 3
    assert report["complete"] is False and report["scored_after_journals_closed"] is True
    assert report["widesearch"]["judge_model"] is None
    assert report["widesearch"]["official_gpt41_semantic_score"] is None
    receipt = report["shard_receipts"][0]
    assert receipt["native_ownership"]["all_native_traces_owned"] is True
    assert receipt["native_ownership"]["native_traces"] == 3
    harness = report["comparison"]["harness"]
    assert all(harness[key]["enabled"] for key in ("web_structured_actions", "web_structured_final",
        "web_explicit_review", "web_navigation_ledger", "web_research_checkpoints",
        "web_data_discovery_guidance", "web_multi_citation", "web_cached_rows"))
    assert "official" not in report["comparison"]["dataset"]["declared_metadata"][0]["source"]
    assert not torch.cuda.is_initialized()


def test_late_timeout_completion_is_drained_owned_and_scores_zero(evaluator, tmp_path):
    tasks = references(("timeout",))
    args = arguments(evaluator, tmp_path / "timeout", episode_timeout=.01)
    generate(evaluator, args, tasks, tmp_path)
    row = evaluator.read_jsonl(args.output_dir / "predictions_rank0.jsonl")[0]
    assert row["status"] == "infrastructure_error" and row["steps"] == []
    assert "operation=policy" in row["detail"]
    owner = evaluator.read_jsonl(args.output_dir / "request_owners_rank0.jsonl")[0]
    assert owner["request_status"] == "native" and owner["delivery"] == "cancelled"
    report = evaluator.score(args, tasks)
    assert report["widesearch"]["f1_by_row"] == 0 and not evaluator.metric_calls
    assert report["shard_receipts"][0]["native_ownership"]["cancelled_native_completions"] == 1


def test_base_adapter_match_across_shards_and_actual_strict_proxy_semantics(evaluator, tmp_path):
    tasks = references(("success", "wrong"))
    base = arguments(evaluator, tmp_path / "base", profile="raw")
    public = generate(evaluator, base, tasks, tmp_path)
    baseline = evaluator.score(base, tasks)
    # llm_judge rules become exact matching without a semantic judge: one
    # wrong non-key cell gets row F1 zero and item F1 one half.
    assert baseline["widesearch"]["f1_by_row"] == .5
    assert baseline["widesearch"]["f1_by_item"] == .75
    adapter = arguments(evaluator, tmp_path / "trained", profile="raw", shards=2,
        adapter=make_adapter(tmp_path / "adapter"), base_output_dir=base.output_dir)
    for rank in (1, 0):
        adapter.shard_index = rank
        asyncio.run(evaluator.generate(adapter, public))
    report = evaluator.score(adapter, tasks)
    assert report["matched_base"]["f1_by_row_delta"] == report["matched_base"]["f1_by_item_delta"] == 0
    assert report["matched_base"]["benchmark_improvement_claimed"] is False
    base_traces = evaluator.read_jsonl(base.output_dir / "native_traces_rank0.jsonl")
    trained_traces = [row for rank in range(2) for row in evaluator.read_jsonl(adapter.output_dir / f"native_traces_rank{rank}.jsonl")]
    assert {(row["task_id"], tuple(row["thought_ids"])) for row in base_traces} == {(row["task_id"], tuple(row["thought_ids"])) for row in trained_traces}
    assert len(evaluator.loads) == 3 and not torch.cuda.is_initialized()


@pytest.mark.parametrize("mutation", ["missing_done", "open_journals", "orphan_trace", "duplicate_trace",
    "step_trace", "owner_schema", "owner_token", "answer", "task_seed", "scored_rollout"])
def test_corrupt_or_unclosed_native_evidence_rejected_before_scoring(evaluator, tmp_path, mutation):
    tasks = references(("success",))
    args = arguments(evaluator, tmp_path / "run")
    generate(evaluator, args, tasks, tmp_path)
    output = args.output_dir
    if mutation == "missing_done":
        (output / "rank0.done.json").unlink()
    elif mutation == "open_journals":
        path = output / "rank0.done.json"
        row = json.loads(path.read_text());row["journals_closed"] = False;path.write_text(json.dumps(row))
    else:
        journal = "native_traces" if mutation in {"orphan_trace", "duplicate_trace"} else "request_owners" if mutation in {"owner_schema", "owner_token"} else "predictions"
        path = output / f"{journal}_rank0.jsonl"
        rows = evaluator.read_jsonl(path)
        if mutation == "orphan_trace":
            rows.append({**rows[0], "trace_id": "orphan"})
        elif mutation == "duplicate_trace":
            rows[1]["trace_id"] = rows[0]["trace_id"]
        elif mutation == "step_trace":
            rows[0]["steps"][0]["trace_id"] = "foreign"
        elif mutation == "owner_schema":
            rows[0]["json_schema"] = {}
        elif mutation == "owner_token":
            rows[0]["completion"]["completion_ids"][0] += 1
        elif mutation == "answer":
            rows[0]["answer"][0]["name"] = "Forged"
        elif mutation == "task_seed":
            rows[0]["task_seed"] += 1
        elif mutation == "scored_rollout":
            rows[0]["metrics"]["f1_by_row"] = 1.
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError):
        evaluator.score(args, tasks)
    assert not evaluator.metric_calls


@pytest.mark.parametrize("difference", ["seed", "max_steps", "profile", "public_prompt", "references", "reference_rules", "source_metadata", "search_backend", "web_timeout", "search_endpoint", "proxy", "search_locale", "search_cache"])
def test_base_adapter_comparison_rejects_mismatches(evaluator, tmp_path, monkeypatch, difference):
    tasks = references(("success",))
    base = arguments(evaluator, tmp_path / "base")
    public = generate(evaluator, base, tasks, tmp_path)
    evaluator.score(base, tasks)
    adapter = arguments(evaluator, tmp_path / "trained", adapter=make_adapter(tmp_path / "adapter"), base_output_dir=base.output_dir)
    if difference in {"seed", "max_steps"}:
        setattr(adapter, difference, getattr(adapter, difference) + 1)
    elif difference == "profile":
        adapter.profile = "greedy"
    elif difference == "public_prompt":
        tasks = [replace(tasks[0], prompt="A different public question")]
        public = [replace(public[0], prompt=tasks[0].prompt)]
    elif difference == "references":
        tasks = [replace(tasks[0], gold=[{"name": "Other", "country": "ZZ"}])]
    elif difference == "reference_rules":
        meta = deepcopy(tasks[0].metadata);meta["evaluation"]["eval_pipeline"]["country"] = {"metric": ["exact_match"], "preprocess": ["lower"]}
        tasks = [replace(tasks[0], metadata=meta)]
    elif difference == "source_metadata":
        tasks = [replace(tasks[0], metadata={**tasks[0].metadata, "source_revision": "changed-source"})]
        public = [replace(public[0], metadata={**public[0].metadata, "source_revision": "changed-source"})]
    elif difference == "search_backend":
        monkeypatch.setenv("DCC_SEARCH_BACKEND", "wikipedia")
    elif difference == "web_timeout":
        monkeypatch.setenv("DCC_WEB_TIMEOUT", "15")
    elif difference == "search_endpoint":
        monkeypatch.setenv("DCC_SEARCH_URL", "https://example.com/invented-search")
    elif difference == "proxy":
        monkeypatch.setattr(evaluator.urllib.request, "getproxies", lambda: {"https": "https://example.com/invented-proxy"})
    elif difference == "search_locale":
        from slime_plugins.dcc import search_provider
        monkeypatch.setattr(search_provider, "_BROWSER_HEADERS", {**search_provider._BROWSER_HEADERS, "Accept-Language": "zh"})
    elif difference == "search_cache":
        from slime_plugins.dcc import search_provider
        monkeypatch.setattr(search_provider, "_CACHE_SECONDS", 1)
    asyncio.run(evaluator.generate(adapter, public))
    with pytest.raises(ValueError):
        evaluator.score(adapter, tasks)


def test_actual_tool_configuration_binds_limits_without_recording_secrets(evaluator, tmp_path, monkeypatch):
    monkeypatch.setenv("DCC_SEARCH_BACKEND", "public_web")
    monkeypatch.setenv("DCC_WEB_TIMEOUT", "15")
    monkeypatch.setenv("DCC_SEARCH_URL", "https://private-user:private-password@example.com/search")
    monkeypatch.setenv("SERPER_API_KEY", "private-serper-key")
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "private-brave-key")
    monkeypatch.setattr(evaluator.urllib.request, "getproxies", lambda: {"https": "https://proxy-user:proxy-password@example.com"})
    config = evaluator.runner_configuration(arguments(evaluator, tmp_path / "run"))["web_tools"]
    assert config["search_backend"] == "public_web" and config["web_timeout_seconds"] == 15
    assert config["search_count_default"] == 8 and config["search_count_bounds"] == [1, 10]
    assert config["max_page_chars"] == 14000 and config["public_download_max_bytes"] == 5_000_000
    assert config["search_headers"]["Accept-Language"] == "en-US,en;q=0.9"
    assert len(config["implementation_identities"]) == 3 and config["api_key_configured"] is None
    serialized = json.dumps(config)
    assert all(secret not in serialized for secret in ("private-user", "private-password", "private-serper-key", "private-brave-key", "proxy-user", "proxy-password"))


def test_cli_generation_then_score_only_does_not_reload_model(evaluator, tmp_path, capsys):
    data = tmp_path / "prepared.jsonl"
    write_tasks(data, references(("success",)))
    command = ["--model", "invented-CPU-model", "--data", str(data), "--output-dir", str(tmp_path / "run"),
               "--source-commit", "a" * 40, "--max-steps", "4"]
    assert evaluator.main(command + ["--generate-only"]) == 0
    assert json.loads(capsys.readouterr().out)["scoring_complete"] is False
    assert not evaluator.metric_calls
    assert evaluator.main(command + ["--score-only"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["widesearch"]["f1_by_row"] == 1 and len(evaluator.loads) == 1
    assert not torch.cuda.is_initialized()


def test_train_inputs_and_labeled_generate_are_rejected(evaluator, tmp_path):
    with pytest.raises(ValueError, match="Held-out"):
        public_task(references(("success",), split="train")[0])
    with pytest.raises(ValueError, match="label-free"):
        asyncio.run(evaluator.generate(arguments(evaluator, tmp_path / "run"), references(("success",))))
    assert not evaluator.loads


def test_missing_shard_and_missing_base_closure_prevent_any_reference_score(evaluator, tmp_path):
    tasks = references(("success", "wrong"))
    args = arguments(evaluator, tmp_path / "partial", shards=2)
    public = generate(evaluator, args, tasks, tmp_path)
    with pytest.raises(ValueError, match="Missing or extra"):
        evaluator.score(args, tasks)
    assert not evaluator.metric_calls
    base = arguments(evaluator, tmp_path / "base")
    asyncio.run(evaluator.generate(base, public))
    adapter = arguments(evaluator, tmp_path / "trained", adapter=make_adapter(tmp_path / "adapter"),
                        base_output_dir=base.output_dir)
    asyncio.run(evaluator.generate(adapter, public))
    # Even with every current shard closed, a missing baseline report must
    # not allow a partial adapter scoring pass.
    with pytest.raises(FileNotFoundError):
        evaluator.score(adapter, tasks)
    assert not evaluator.metric_calls


def test_exclusive_generation_refuses_retries_and_changed_configuration(evaluator, tmp_path):
    tasks = references(("success",))
    args = arguments(evaluator, tmp_path / "run")
    public = generate(evaluator, args, tasks, tmp_path)
    with pytest.raises(FileExistsError):
        asyncio.run(evaluator.generate(args, public))
    assert len(evaluator.loads) == 1
    args.thought_tokens += 1
    with pytest.raises(ValueError, match="configuration"):
        evaluator.score(args, tasks)
    assert not evaluator.metric_calls


def test_base_adapter_actual_storage_precision_must_match(evaluator, tmp_path, monkeypatch):
    tasks = references(("success",))
    base = arguments(evaluator, tmp_path / "base")
    public = generate(evaluator, base, tasks, tmp_path)
    evaluator.score(base, tasks)
    actual_loader = evaluator.load_policy_model
    def changed(model, **kwargs):
        loaded, tokenizer = actual_loader(model, **kwargs)
        if kwargs["adapter"]:
            loaded.base_weight = torch.nn.Parameter(torch.ones(2, dtype=torch.bfloat16))
        return loaded, tokenizer
    monkeypatch.setattr(evaluator, "load_policy_model", changed)
    adapter = arguments(evaluator, tmp_path / "trained", adapter=make_adapter(tmp_path / "adapter"),
                        base_output_dir=base.output_dir)
    asyncio.run(evaluator.generate(adapter, public))
    evaluator.metric_calls.clear()
    with pytest.raises(ValueError, match="Base and adapter"):
        evaluator.score(adapter, tasks)
    assert not evaluator.metric_calls


def test_adapter_content_change_after_generation_is_rejected(evaluator, tmp_path):
    tasks = references(("success",))
    adapter_path = tmp_path / "adapter"
    args = arguments(evaluator, tmp_path / "trained", adapter=make_adapter(adapter_path))
    generate(evaluator, args, tasks, tmp_path)
    (adapter_path / "adapter_model.safetensors").write_bytes(b"different CPU fixture bytes")
    with pytest.raises(ValueError, match="configuration"):
        evaluator.score(args, tasks)
    assert not evaluator.metric_calls
