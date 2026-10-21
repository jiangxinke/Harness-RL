"""Synthetic CPU controls for real closed-journal Web runtime paths.

No model, network, held-out data, GPU admission or training is represented here.
"""
from __future__ import annotations

import asyncio
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from slime_plugins.dcc import online_web as web
from slime_plugins.dcc.data import Task
from slime_plugins.dcc.episode import EpisodeRunner, PolicyCompletion, ModelProtocolFailure
from slime_plugins.dcc.tools import InfrastructureError
from slime_plugins.dcc.prompt_history import PromptHistoryBudgetExceeded
from slime_plugins.dcc.hf_native_policy import NativeHFCompletion, NativeHFPolicy
from slime_plugins.dcc.hf_policy import _chat_tokens
from slime_plugins.dcc.native_hf_context import restore_verified_native_context
from slime_plugins.dcc.online_grpo import group_signal

URL, EOS = "https://example.com/invented", 2
COMMIT = "a" * 40


class ToyTokenizer:
    def encode(self, text):
        return [1000 + ord(char) for char in text]
    def decode(self, ids, *, skip_special_tokens=False):
        return "".join(chr(token - 1000) for token in ids)
    def apply_chat_template(self, messages, **kwargs):
        return self.encode(json.dumps(messages))


class FixtureCompletion(NativeHFCompletion):
    pass


def public_row():
    return dict(source="independent_curriculum", id="invented-web-train",
        benchmark="widesearch", split="train", prompt="List invented source rows.",
        metadata={"evaluation": {"required": ["id", "value"]}})


def label_for(task):
    return dict(task_id=task.id, benchmark="widesearch", split="train",
        gold=[{"id": "a", "value": "8"}], evaluation=task.metadata["evaluation"],
        label_protocol=web.LABEL_PROTOCOL, train_reward=web.configuration())


class FixtureTools:
    async def web_search(self, query, count=8):
        return {"results": [{"url": URL, "title": "Invented source"}]}
    async def fetch_url(self, url, start=0):
        return {"url": URL, "text": "id a value 8"}
    def citation_validity(self, answer):
        return 1.0


class FixturePolicy:
    def __init__(self, modes):
        self.modes, self.bound, self.serial = modes, 0, 0
        self.tokenizer, self.eos_ids, self.worker = ToyTokenizer(), [EOS], None
        self.trace_sink = None
    def bind_task(self, task_id, rank=0):
        sample, parent = self.bound, self
        self.bound += 1
        mode = self.modes[sample]
        class Bound:
            turn = 0
            async def generate_structured(self, messages, *, json_schema, **kwargs):
                if mode == "infra":
                    raise InfrastructureError("Invented unavailable environment")
                if mode == "context":
                    raise PromptHistoryBudgetExceeded("invented_too_long", diagnostics={"original_prompt_tokens": 10, "minimum_prompt_tokens": 10, "effective_prompt_limit": 1}, original_prompt_ids=[])
                if mode in {"hf-context", "late-hf-context", "slow-hf-context"}:
                    if mode == "slow-hf-context":
                        await asyncio.sleep(0.03)
                    if self.turn == 0:
                        self.prompt_limit = len(_chat_tokens(parent.tokenizer, messages)) + (
                            16 if mode == "late-hf-context" else -1)
                    if mode != "late-hf-context" or self.turn:
                        # Actual NativeHFPolicy full-request preflight, no model
                        # or encoded-generation replacement for the failed call.
                        native = object.__new__(NativeHFPolicy)
                        native.tokenizer, native.max_prompt_tokens = parent.tokenizer, self.prompt_limit
                        native.thought_tokens, native.json_tokens = 4, 256
                        native.allow_truncation, native.mixed_schema_batching = False, False
                        native.sampling_profile, native.capture_thought_logprobs = "raw", True
                        native.trace_sink = parent.trace_sink
                        error = native._generate_batch([messages], json_schema, [task_id])[0]
                        assert isinstance(error, PromptHistoryBudgetExceeded)
                        raise error
                if mode == "slow":
                    await asyncio.sleep(0.03)
                turn = self.turn
                self.turn += 1
                parent.serial += 1
                trace_id = f"invented-trace-{parent.serial}"
                action = ({"tool": "web_search", "arguments": {"query": "invented source", "count": 8}}
                          if turn == 0 else {"tool": "fetch_url", "arguments": {"url": URL, "start": 0}}
                          if turn == 1 else {"answer": [{"id": "a", "value": "8", "citations": [URL]}]})
                action = {"plan": "Read source", "checks": "Exact public columns", **action}
                failed = mode == "model-failure"
                text = "" if failed else json.dumps(action)
                thought = [77] if failed else [77, web.END_THINK]
                final = [] if failed else parent.tokenizer.encode(text) + [EOS]
                ids = parent.tokenizer.apply_chat_template(messages)
                status = "thought_did_not_stop_on_real_close" if failed else "completed"
                completion = FixtureCompletion(text, ids, thought + final,
                    phase_lengths=(len(thought), len(final)),
                    thought_token_count=len(thought), native_status=status, trace_id=trace_id,
                    thought_logprobs=[-0.5] * len(thought), thought_logprobs_source=web.BEHAVIOR_SCORES)
                trace = dict(protocol=web.HF_PROTOCOL, trace_id=trace_id, task_id=task_id,
                    status=status, thought_sampling_profile="raw", prompt_ids=ids, thought_ids=thought, json_ids=final,
                    thought_token_count=len(thought), final_text=text, json_public_schema=deepcopy(json_schema),
                    thought_logprobs=list(completion.thought_logprobs), thought_logprobs_source=web.BEHAVIOR_SCORES)
                if mode == "bad-uuid":
                    trace["trace_id"] = "foreign-" + trace_id
                if mode == "bad-score":
                    trace["thought_logprobs"][0] = -9.0
                if mode == "bad-source":
                    trace["thought_logprobs_source"] = "teacher-forced-replay"
                if mode == "bad-profile":
                    trace["thought_sampling_profile"] = "thinking"
                if parent.trace_sink is not None:
                    parent.trace_sink(trace)
                if failed:
                    raise ModelProtocolFailure(status, completion=completion,
                                               protocol=web.HF_PROTOCOL, trace_id=trace_id)
                return completion
        return Bound()
    async def close(self):
        self.worker = None


def runner(timeout=2):
    return EpisodeRunner(max_steps=3, timeout_seconds=timeout, web_tools_factory=FixtureTools,
                         auto_read_sources=0, web_structured_actions=True,
                         web_structured_final=True, web_explicit_review=True,
                         web_multi_citation=True, web_cached_rows=True)


def inputs(tmp_path):
    task = web.public_web_task(public_row())
    public, labels, binding = (tmp_path / name for name in ("public.jsonl", "labels.jsonl", "binding.json"))
    public.write_text(json.dumps(public_row()) + "\n")
    labels.write_text(json.dumps(label_for(task)) + "\n")
    record = dict(version=web.BINDING, root_accepted=True, training_reward_admission_approved=True,
        review_record="synthetic-test-only", source_commit=COMMIT, protocols=web.binding_protocols(),
        public_train=web.file_identity(public), private_train_labels=web.file_identity(labels),
        public_task_ids=[task.id], source_evidence=dict(review_record="synthetic-only", reviewed=True,
            receipt_pairs=1, lineage_cells=2, inventory={"synthetic": "not-real-admission"}))
    binding.write_text(json.dumps(record))
    return web.WebTrainingInputs(public, labels, binding, source_commit=COMMIT), binding


def collect(tmp_path, modes, *, timeout=2):
    policy = FixturePolicy(modes)
    task = web.public_web_task(public_row())
    group = asyncio.run(web.collect_public_group(runner(timeout), task, policy,
        group_size=len(modes), rank=0, iteration=1, directory=tmp_path / "group", source_commit=COMMIT))
    return group, policy


def test_real_group_journals_close_before_first_label_deserialization(tmp_path, monkeypatch):
    data, _ = inputs(tmp_path)
    original = web._read_jsonl
    parses = []
    def observed(path, **kwargs):
        if Path(path) == data.labels_path:
            assert (tmp_path / "group/generation_closed.json").is_file()
            parses.append("label-after-close")
        return original(path, **kwargs)
    monkeypatch.setattr(web, "_read_jsonl", observed)
    assert data.label_reads == 0
    group, policy = collect(tmp_path, ["ok", "model-failure"])
    assert not parses and data.label_reads == 0
    results, report = web.score_closed_group(group, inputs=data, tokenizer=policy.tokenizer)
    assert parses == ["label-after-close"]
    assert [result.reward for result in results] == [1.0, 0.0]
    assert [result.status for result in results] == ["completed", "model_protocol_failure"]
    assert report["model_failures_retained"] and not report["gradient_group_discarded"]
    assert group.closure["closed_journals"] == dict(native_traces=4, request_owners=4, public_outcomes=2)
    assert len(web._read_jsonl(tmp_path / "group/public_outcomes.jsonl")) == 2
    assert all(row["reward"] == 0 for row in web._read_jsonl(tmp_path / "group/public_outcomes.jsonl"))


@pytest.mark.parametrize("failure", ["infra", "context", "hf-context", "slow", "slow-hf-context"])
def test_whole_unreliable_group_retained_and_zero(tmp_path, failure):
    data, _ = inputs(tmp_path)
    group, policy = collect(tmp_path, ["ok", failure], timeout=0.01 if failure.startswith("slow") else 2)
    results, report = web.score_closed_group(group, inputs=data, tokenizer=policy.tokenizer)
    assert len(results) == 2 and [result.reward for result in results] == [0, 0]
    assert report["gradient_group_discarded"]
    assert results[1].status in {"infrastructure_error", "context_budget_failure"}
    if failure == "slow":
        rows = web._read_jsonl(tmp_path / "group/request_owners.jsonl")
        assert any(row.get("delivery") == "cancelled" and row.get("trace_id") for row in rows)
    elif failure == "slow-hf-context":
        assert results[1].status == "infrastructure_error"
        rows = web._read_jsonl(tmp_path / "group/request_owners.jsonl")
        assert any(row.get("delivery") == "cancelled" and row.get("native_hf_preflight") for row in rows)


def test_post_sampling_HF_overflow_keeps_sibling_reward_and_replays_closed_journals(tmp_path):
    from slime_plugins.dcc.episode import EpisodeResult
    data, _ = inputs(tmp_path)
    group, policy = collect(tmp_path, ["ok", "late-hf-context"])
    results, report = web.score_closed_group(group, inputs=data, tokenizer=policy.tokenizer)
    assert [result.reward for result in results] == [1., 0.]
    assert [len(result.steps) for result in results] == [3, 1]
    assert results[1].status == "context_budget_failure"
    assert not report["gradient_group_discarded"]
    assert report["receipts"][0]["train_metric"]["evaluated"]
    assert not report["receipts"][1]["train_metric"]["evaluated"]
    signal = group_signal(results, group_size=2, max_steps=3, thought_tokens=4)
    assert signal.discarded_reason is None and signal.advantages[0] > 0 > signal.advantages[1]
    assert signal.segments[1][0].completion_ids == [77, web.END_THINK]
    saved = web._read_jsonl(tmp_path / "group/public_outcomes.jsonl")
    traces = web._read_jsonl(tmp_path / "group/native_traces.jsonl")
    owners = web._read_jsonl(tmp_path / "group/request_owners.jsonl")
    restored = []
    for row in saved:
        row.pop("sample")
        row["steps"] = [NativeHFCompletion(**step) for step in row["steps"]]
        result = EpisodeResult(**row)
        if result.status == "context_budget_failure":
            result = restore_verified_native_context(result, traces=traces, tokenizer=policy.tokenizer)
        restored.append(result)
    # Re-run the genuine owner/schema/token verifier from closed JSONL bytes.
    prompts = web.verify_owned_native_group(group.task, restored, owners, traces,
        eos_ids=[EOS], tokenizer=policy.tokenizer, iteration=1, rank=0)
    rebound = web.ClosedPublicGroup(group.task, tuple(restored), group.closure, COMMIT, prompts)
    rescored, _ = web.score_closed_group(rebound, inputs=data, tokenizer=policy.tokenizer)
    assert group_signal(rescored, group_size=2, max_steps=3, thought_tokens=4) == signal


def test_uuid_mismatch_preserves_failed_journals_without_close_marker(tmp_path):
    with pytest.raises(ValueError, match="trace ownership"):
        collect(tmp_path, ["ok", "bad-uuid"])
    assert not (tmp_path / "group/generation_closed.json").exists()
    assert len(web._read_jsonl(tmp_path / "group/public_outcomes.jsonl")) == 2


@pytest.mark.parametrize("field", ["source_commit", "protocols", "public_train", "private_train_labels", "root_accepted"])
def test_reject_mismatched_binding_before_label_parse(tmp_path, field):
    data, binding = inputs(tmp_path)
    record = json.loads(binding.read_text())
    record[field] = False
    binding.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        web.WebTrainingInputs(tmp_path / "public.jsonl", data.labels_path, binding, source_commit=COMMIT)


def test_reject_pending_old_manifest_and_changed_private_labels(tmp_path):
    data, _ = inputs(tmp_path)
    group, policy = collect(tmp_path, ["ok", "ok"])
    data.labels_path.write_text(data.labels_path.read_text() + "\n")
    with pytest.raises(ValueError, match="changed"):
        web.score_closed_group(group, inputs=data, tokenizer=policy.tokenizer)


def test_public_input_mechanically_rejects_labels_and_private_metadata():
    for field, value in (("gold", []), ("private", "hidden"), ("metadata", {"evaluation": {"required": ["id"]}, "source_url": URL})):
        row = public_row()
        row[field] = value
        with pytest.raises(ValueError):
            web.public_web_task(row)


def test_mixed_plan_uses_original_cuad_strata_and_public_web_deck():
    spec = importlib.util.spec_from_file_location("web_test_trainer", Path(__file__).parents[1] / "scripts/dcc_online_grpo.py")
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    cuad = [Task("positive", "cuad", "train", "Q", "contract", ["span"]),
            Task("negative", "cuad", "train", "Q", "contract", [])]
    tasks = [web.public_web_task({**public_row(), "id": f"web-{index}"}) for index in range(3)]
    plan = web.mixed_task_plan(cuad, tasks, steps=20, world_size=2, seed=47, cuad_plan=trainer.task_plan)
    flat = [value for row in plan for value in row]
    assert flat[::2] == [row[0] for row in trainer.task_plan(cuad, steps=20, world_size=1, seed=47)]
    web_values = flat[1::2]
    assert all(len(set(web_values[index:index + 3])) == 3 for index in range(0, 18, 3))
    with pytest.raises(ValueError, match="CUAD only"):
        trainer.validate_train_tasks([Task("web", "widesearch", "train", "Q", gold=[])])


def test_cell_gates_reject_gold_on_unrelated_fetched_text_or_missing_prompt(tmp_path):
    data, _ = inputs(tmp_path)
    group, policy = collect(tmp_path, ["ok", "ok"])
    assert web.unchanged_native_cell_grounding(group.task, group.results[0], "unrelated prompt")
    group.results[0].tool_events[1]["observation"]["text"] = "unrelated source"
    assert web.unchanged_native_cell_grounding(group.task, group.results[0], "a 8")

def test_each_group_deserializes_only_after_its_own_close_without_label_cache(tmp_path, monkeypatch):
    data, _ = inputs(tmp_path)
    group, policy = collect(tmp_path, ["ok", "ok"])
    original, count = web._read_jsonl, []
    def observed(path, **kwargs):
        if Path(path) == data.labels_path:
            count.append(1)
        return original(path, **kwargs)
    monkeypatch.setattr(web, "_read_jsonl", observed)
    web.score_closed_group(group, inputs=data, tokenizer=policy.tokenizer)
    web.score_closed_group(group, inputs=data, tokenizer=policy.tokenizer)
    assert len(count) == 2 and data.label_reads == 2


def test_unexpected_episode_exception_drains_sibling_and_keeps_no_close_marker(tmp_path):
    calls = []
    class BrokenRunner:
        count = 0
        async def run(self, task, owner, **kwargs):
            self.count += 1
            if self.count == 1:
                await asyncio.sleep(0.01)
                raise RuntimeError("original failure")
            try:
                await asyncio.sleep(30)
            finally:
                calls.append("sibling-drained")
    class Policy(FixturePolicy):
        async def close(self):
            assert calls == ["sibling-drained"]
            calls.append("policy-drained")
            raise RuntimeError("secondary drain failure")
    policy = Policy(["ok", "ok"])
    with pytest.raises(RuntimeError, match="original failure") as caught:
        asyncio.run(web.collect_public_group(BrokenRunner(), web.public_web_task(public_row()), policy,
            group_size=2, rank=0, iteration=1, directory=tmp_path / "failed", source_commit=COMMIT))
    assert "secondary drain failure" in " ".join(caught.value.__notes__)
    assert calls == ["sibling-drained", "policy-drained"]
    assert not (tmp_path / "failed/generation_closed.json").exists()
    for name in ("native_traces", "request_owners", "public_outcomes"):
        assert (tmp_path / "failed" / f"{name}.jsonl").is_file()


def test_missing_web_switch_rejects_paths_before_runtime(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("web_cli_test", Path(__file__).parents[1] / "scripts/dcc_online_grpo.py")
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    monkeypatch.setattr(trainer, "validate_source", lambda _: None)
    monkeypatch.setattr(trainer, "load_tasks", lambda _: [Task("positive", "cuad", "train", "Q", "contract", ["span"])])
    with pytest.raises(SystemExit) as caught:
        trainer.main(["--model", "not-loaded", "--train-data", "train.jsonl", "--source-commit", COMMIT,
                      "--output-dir", str(tmp_path / "out"), "--web-train-data", "public.jsonl"])
    assert caught.value.code == 2


@pytest.mark.parametrize("mode", ["bad-score", "bad-source", "bad-profile"])
def test_actual_streamed_behavior_score_trace_binding_required_before_label_access(tmp_path, mode):
    data, _ = inputs(tmp_path)
    with pytest.raises(ValueError, match="binding differs"):
        collect(tmp_path, ["ok", mode])
    assert data.label_reads == 0
    assert not (tmp_path / "group/generation_closed.json").exists()
