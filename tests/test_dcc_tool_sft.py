"""CPU checks for leakage, evidence, balancing and completion-only tool SFT."""

from __future__ import annotations

import dataclasses
import asyncio
import importlib.util
import json
import math
import sys
from collections import Counter
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

_SPEC = importlib.util.spec_from_file_location("dcc_tool_sft_tests", Path(__file__).resolve().parents[1] / "scripts/dcc_tool_sft.py")
_TRAIN = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _TRAIN
_SPEC.loader.exec_module(_TRAIN)

URL = "https://example.org/registry.csv"


@pytest.mark.parametrize("positive,stages", [(True, ["final"]), (False, ["final"]),
                                            (True, ["search", "final"])])
def test_recorded_native_cuad_samples_only_actual_stages(positive, stages):
    records = [{"task_id": "recorded-task", "stage": stage,
                "metadata": {"positive": positive, "category": "Recorded category"}}
               for stage in stages]

    def recorded_encoder(record, tokenizer, **kwargs):
        return _TRAIN.Example(record["task_id"], "cuad_tool_" + record["stage"],
                              record["metadata"]["category"], [1], [2])

    kwargs = dict(rank=0, world=1, seed=42, web_fraction=0, negative_fraction=0.5,
                  max_prompt=10, max_response=10, batch_tokens=20,
                  cuad_records=records, record_encoder=recorded_encoder)
    with pytest.raises(ValueError, match="lost a full stage"):
        _TRAIN.MixedSampler([], [], None, None, **kwargs)
    sampler = _TRAIN.MixedSampler([], [], None, None, recorded_cuad_stages=True, **kwargs)
    assert {sampler.sample().stage for _ in range(2 * len(stages))} == {
        "cuad_tool_" + stage for stage in stages}
    assert sampler.cuad_missing_stages[0]["missing_stages"] == [
        stage for stage in _TRAIN.CUAD_STAGES if stage not in stages]


class Tokenizer:
    eos_token_id = 1

    def encode(self, text, *, add_special_tokens=False):
        return [ord(character) + 3 for character in text]

    def decode(self, tokens, *, skip_special_tokens=False):
        return "".join(chr(token - 3) for token in tokens if token >= 3)

    def apply_chat_template(self, messages, **kwargs):
        return self.encode("".join(message["content"] for message in messages))


def observation(value):
    return {"role": "user", "content": "Tool observation (source material):\n" + json.dumps(value) + "\nRemaining turns: 4."}


def web_records(task_id="independent-example"):
    messages = [{"role": "system", "content": "Use actual sources."}, {"role": "user", "content": "Find registered entries."}]
    search = {"tool": "web_search", "arguments": {"query": "official registry"}}
    fetch = {"tool": "fetch_url", "arguments": {"url": URL, "start": 0}}
    records = []
    for stage, action in (("search", search), ("fetch", fetch), ("final", {"answer": [{"number": "200", "name": "OK", "citations": [URL]}]})):
        records.append({"task_id": task_id, "source": "independent_curriculum", "split": "train", "stage": stage,
                        "messages": deepcopy(messages), "completion": json.dumps(action), "source_urls": [URL]})
        messages.append({"role": "assistant", "content": json.dumps(action)})
        if stage == "search":
            messages.append(observation({"results": [{"url": URL}]}))
        elif stage == "fetch":
            messages.append(observation({"url": URL, "text": "Value,Description\n200,OK\n"}))
    return records


@pytest.mark.parametrize("stage", range(3))
def test_valid_web_teacher_history_accepts_real_evidence(stage):
    record = web_records()[stage]
    assert _TRAIN.validate_record(record) is record


@pytest.mark.parametrize("field,value", [("split", "validation"), ("split", "test"), ("source", "official_widesearch")])
def test_web_teacher_rejects_any_nontraining_or_benchmark_source(field, value):
    record = web_records()[0]
    record[field] = value
    with pytest.raises(ValueError, match="independent_curriculum train"):
        _TRAIN.validate_record(record)


def test_final_does_not_trust_source_urls_or_fabricated_citations():
    record = web_records()[2]
    record["messages"] = record["messages"][:2]
    with pytest.raises(ValueError, match="successful search and fetch"):
        _TRAIN.validate_record(record)
    record = web_records()[2]
    target = json.loads(record["completion"])
    target["answer"][0]["citations"] = ["https://invented.example/source"]
    record["completion"] = json.dumps(target)
    with pytest.raises(ValueError, match="observed successful source"):
        _TRAIN.validate_record(record)


@pytest.mark.parametrize("bad_observation", [{"url": URL, "text": ""}, {"url": URL, "text": "200 OK", "error": "fetch failed"}])
def test_final_requires_nonempty_successful_fetch(bad_observation):
    record = web_records()[2]
    record["messages"][-1] = observation(bad_observation)
    with pytest.raises(ValueError, match="successful search and fetch"):
        _TRAIN.validate_record(record)


def test_final_target_and_declared_grounding_must_match_fetched_text():
    record = web_records()[2]
    record["completion"] = record["completion"].replace('"OK"', '"Invented answer"')
    with pytest.raises(ValueError, match="unsupported by fetched"):
        _TRAIN.validate_record(record)
    record = web_records()[2]
    record["grounding_fields"] = ["a misleading declared field"]
    with pytest.raises(ValueError, match="do not match"):
        _TRAIN.validate_record(record)


def test_teacher_must_not_include_current_target_in_history():
    record = web_records()[2]
    record["messages"].append({"role": "assistant", "content": record["completion"]})
    with pytest.raises(ValueError, match="before the target"):
        _TRAIN.validate_record(record)
    record["messages"].append({"role": "user", "content": "Already answered"})
    with pytest.raises(ValueError, match="final target"):
        _TRAIN.validate_record(record)


def test_record_loader_rejects_duplicate_segments_and_validation(tmp_path):
    path = tmp_path / "records.jsonl"
    rows = web_records()
    path.write_text("\n".join(json.dumps(record) for record in rows))
    assert len(_TRAIN.load_records(path)) == 3
    path.write_text("\n".join(json.dumps(record) for record in [*rows, rows[0]]))
    with pytest.raises(ValueError, match="Duplicate"):
        _TRAIN.load_records(path)


def encode(record, **kwargs):
    return _TRAIN.encode_record(record, Tokenizer(), max_prompt=kwargs.get("max_prompt", 8192),
                                max_response=kwargs.get("max_response", 2048), batch_tokens=kwargs.get("batch_tokens", 12288), skips=kwargs["skips"])


def test_web_never_truncates_prompt_or_target_and_reports_exact_skip_reasons():
    record = web_records()[2]
    skipped = Counter()
    assert encode(record, skips=skipped, max_prompt=10) is None
    assert skipped == {"web_final_prompt_too_long": 1}
    assert encode(record, skips=skipped, max_response=3) is None
    assert skipped["web_final_target_too_long"] == 1
    assert encode(record, skips=skipped, batch_tokens=10) is None
    assert skipped["web_final_batch_budget"] == 1
    example = encode(record, skips=skipped)
    assert example.completion == Tokenizer().encode(record["completion"]) + [1]
    assert "Tool observation" in Tokenizer().decode(example.prompt)
    assert "Tool observation" not in Tokenizer().decode(example.completion)


def sampler(records=None, tasks=None, rank=0, world=1, **kwargs):
    return _TRAIN.MixedSampler(records or web_records(), tasks or [], Tokenizer(), None,
                               rank=rank, world=world, seed=17, web_fraction=kwargs.get("web_fraction", 1),
                               negative_fraction=kwargs.get("negative_fraction", 1 / 3), max_prompt=8192,
                               max_response=2048, batch_tokens=12288)


def test_web_stages_are_balanced_and_workers_use_disjoint_tasks():
    records = [record for index in range(8) for record in web_records(f"independent-{index}")]
    first, second = sampler(records, rank=0, world=2), sampler(records, rank=1, world=2)
    left, right = [first.sample() for _ in range(60)], [second.sample() for _ in range(60)]
    assert Counter(example.stage for example in left) == {"search": 20, "fetch": 20, "final": 20}
    assert {example.task_id for example in left}.isdisjoint(example.task_id for example in right)


def test_cuad_sampling_balances_polarity_and_categories_instead_of_dataset_sizes(monkeypatch):
    from slime_plugins.dcc.data import Task

    tasks = [Task(f"positive-large-{i}", "cuad", "train", "Q", document="Actual span", gold=["Actual span"], metadata={"category": "large"}) for i in range(100)]
    tasks += [dataclasses.replace(tasks[0], id="positive-small", metadata={"category": "small"})]
    tasks += [dataclasses.replace(tasks[0], id="negative", gold=[])]
    monkeypatch.setattr(_TRAIN, "supervised_example", lambda *args: ([1, 2], [3, 4]))
    selected = sampler(tasks=tasks, web_fraction=0)
    examples = [selected.sample() for _ in range(3000)]
    stages = Counter(example.stage for example in examples)
    assert 0.29 < stages["cuad_negative"] / len(examples) < 0.38
    positive = Counter(example.category for example in examples if example.stage == "cuad_positive")
    assert 0.45 < positive["small"] / sum(positive.values()) < 0.55
    with pytest.raises(ValueError, match="CUAD train"):
        sampler(tasks=[dataclasses.replace(tasks[0], split="validation")])


def test_token_batch_keeps_long_pending_final_instead_of_dropping_its_stage():
    selected = sampler()
    selected.batch_tokens = 1000
    first = _TRAIN.Example("search", "search", "web", [1] * 10, [2] * 10)
    final = _TRAIN.Example("final", "final", "web", [1] * 600, [2] * 300)
    queue = [first, final]
    selected.sample = lambda: queue.pop(0) if selected.pending is None else _pop_pending(selected)
    selected.batch_tokens = 910
    assert _TRAIN.sample_batch(selected, 2) == [first]
    assert selected.pending is final
    assert _TRAIN.sample_batch(selected, 1) == [final]


def _pop_pending(sampler):
    value, sampler.pending = sampler.pending, None
    return value


def test_backward_uses_only_completion_positions_and_averages_examples():
    class Model(torch.nn.Module):
        device = "cpu"

        def __init__(self):
            super().__init__()
            self.scores = torch.nn.Parameter(torch.zeros(16, 8))

        def forward(self, *, input_ids, use_cache, logits_to_keep):
            return SimpleNamespace(logits=self.scores[:input_ids.shape[1]][-logits_to_keep:].unsqueeze(0))

    example = _TRAIN.Example("task", "fetch", "web", [1, 2, 3, 4], [5, 6])
    model = Model()
    loss = _TRAIN.backward_batch(model, [example, example])
    assert loss == pytest.approx(math.log(8))
    assert model.scores.grad.abs().sum(dim=-1).nonzero().flatten().tolist() == [3, 4]
    reference = Model()
    _TRAIN.backward_batch(reference, [example])
    torch.testing.assert_close(model.scores.grad, reference.scores.grad)


def _unequal_batch_worker(rank, rendezvous):
    from torch import distributed

    class Model(torch.nn.Module):
        device = "cpu"

        def __init__(self):
            super().__init__()
            self.scores = torch.nn.Parameter(torch.zeros(16, 8))
            self.unused = torch.nn.Parameter(torch.ones(2))

        def forward(self, *, input_ids, use_cache, logits_to_keep):
            return SimpleNamespace(logits=self.scores[:input_ids.shape[1]][-logits_to_keep:].unsqueeze(0))

    distributed.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2,
                                   timeout=timedelta(seconds=30))
    try:
        examples = [_TRAIN.Example(str(i), "final", "web", [1, 2, 3, 4], target)
                    for i, target in enumerate(([5, 6], [6, 7], [2, 2, 3]))]
        local = examples[:1] if rank == 0 else examples[1:]
        model, reference = Model(), Model()
        _TRAIN.backward_batch(model, local)
        global_examples = _TRAIN.average_example_gradients(list(model.parameters()), len(local), 2)
        assert global_examples == 3
        _TRAIN.backward_batch(reference, examples)
        torch.testing.assert_close(model.scores.grad, reference.scores.grad)
        torch.testing.assert_close(model.unused.grad, torch.zeros_like(model.unused))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
        optimizer.step()
        reference_optimizer.step()
        torch.testing.assert_close(model.scores, reference.scores)
        torch.testing.assert_close(model.unused, reference.unused)
    finally:
        distributed.destroy_process_group()


def test_two_cpu_ranks_weight_unequal_batches_like_all_global_examples(tmp_path):
    if not torch.distributed.is_available() or not torch.distributed.is_gloo_available():
        pytest.skip("CPU distributed Gloo is unavailable")
    rendezvous = (tmp_path / "distributed-example-average").as_uri()
    torch.multiprocessing.spawn(_unequal_batch_worker, args=(rendezvous,), nprocs=2, join=True)


SPAN = "Delaware law governs this agreement."


def cuad_records(task_id="cuad-train-example"):
    messages = [{"role": "system", "content": "Find verbatim spans."}, {"role": "user", "content": "Which governing law applies?"}]
    messages.append({"role": "user", "content": "Initial contract retrieval (source material):\n" + json.dumps({"document_length": 1000, "hits": [{"start": 0, "end": 10, "text": "Distractor"}]})})
    search = {"tool": "search_document", "arguments": {"query": "governing jurisdiction"}}
    read = {"tool": "read_document", "arguments": {"start": 100, "end": 100 + len(SPAN)}}
    records = []
    for stage, action in (("search", search), ("read", read), ("final", {"answer": [{"text": SPAN, "confidence": 0.9}]})):
        records.append({"task_id": task_id, "source": "cuad_train", "benchmark": "cuad", "split": "train", "stage": stage,
                        "messages": deepcopy(messages), "completion": json.dumps(action), "grounding_fields": [SPAN],
                        "metadata": {"category": "Governing Law", "positive": True, "plan_id": 1}})
        messages.append({"role": "assistant", "content": json.dumps(action)})
        if stage == "search":
            messages.append(observation({"document_length": 1000, "hits": [{"start": 100, "end": 100 + len(SPAN), "text": SPAN}]}))
        elif stage == "read":
            messages.append(observation({"document_length": 1000, "start": 100, "end": 100 + len(SPAN), "text": SPAN}))
    return records


@pytest.mark.parametrize("index", range(3))
def test_cuad_tool_records_allow_initial_retrieval_and_grounded_history(index):
    record = cuad_records()[index]
    assert _TRAIN.validate_cuad_record(record) is record
    example = encode(record, skips=Counter())
    assert example.stage == "cuad_tool_" + record["stage"]


@pytest.mark.parametrize("arguments", [{"start": 100}, {"start": 100, "end": 12100}, {"start": 100, "end": 100 + len(SPAN)},
                                      {"start": 0, "end": 100 + len(SPAN) + 800},
                                      {"start": 0, "end": 100 + len(SPAN) + 1600}, {"start": 0, "end": 8100}])
def test_cuad_read_can_use_hit_boundary_or_public_reader_windows(arguments):
    record = cuad_records()[1]
    record["completion"] = json.dumps({"tool": "read_document", "arguments": arguments})
    assert _TRAIN.validate_cuad_record(record) is record


@pytest.mark.parametrize("arguments", [{"start": 600, "end": 700}, {"start": 100, "end": 250}, {"start": 100, "end": 12101},
                                      {"start": 1, "end": 8100}, {"start": 0, "end": 100 + len(SPAN) + 801},
                                      {"start": False, "end": 8100}])
def test_cuad_read_rejects_unexposed_starts_and_arbitrary_gold_selected_ends(arguments):
    record = cuad_records()[1]
    record["completion"] = json.dumps({"tool": "read_document", "arguments": arguments})
    with pytest.raises(ValueError, match="CUAD read"):
        _TRAIN.validate_cuad_record(record)


def test_cuad_review_fields_preserve_grounding_and_completion_only_encoding():
    record = cuad_records()[2]
    target = json.loads(record["completion"])
    target = {"plan": "Check the public question.", "checks": "The source supports the span.", **target}
    record["completion"] = json.dumps(target)
    assert _TRAIN.validate_cuad_record(record) is record
    example = encode(record, skips=Counter())
    assert example.completion == Tokenizer().encode(record["completion"]) + [1]
    target["answer"][0]["text"] = "Invented clause absent from source"
    record["completion"] = json.dumps(target)
    with pytest.raises(ValueError, match="differs from declared"):
        _TRAIN.validate_cuad_record(record)


def test_cuad_final_rejects_source_and_grounding_mismatch():
    record = cuad_records()[2]
    record["split"] = "validation"
    with pytest.raises(ValueError, match="split=train"):
        _TRAIN.validate_cuad_record(record)
    record = cuad_records()[2]
    record["grounding_fields"] = ["An invented span"]
    with pytest.raises(ValueError, match="differs from declared"):
        _TRAIN.validate_cuad_record(record)
    record = cuad_records()[2]
    # Neither separate source contains this full target; joining observations
    # must not fabricate evidence across unrelated offsets or repeated pages.
    record["grounding_fields"] = ["Distractor Delaware"]
    record["completion"] = json.dumps({"answer": [{"text": "Distractor Delaware"}]})
    with pytest.raises(ValueError, match="absent from actual"):
        _TRAIN.validate_cuad_record(record)


def test_cuad_tool_loader_and_stage_balancing_preserve_negative_direct_examples(tmp_path, monkeypatch):
    from slime_plugins.dcc.data import Task

    path = tmp_path / "cuad-tools.jsonl"
    path.write_text("\n".join(json.dumps(record) for record in cuad_records()))
    records = _TRAIN.load_cuad_records(path)
    task = Task("negative", "cuad", "train", "Which law?", document="Unrelated contract", gold=[], metadata={"category": "Governing Law"})
    monkeypatch.setattr(_TRAIN, "supervised_example", lambda *args: ([1, 2], [3, 4]))
    selected = _TRAIN.MixedSampler(web_records(), [task], Tokenizer(), None, rank=0, world=1, seed=42,
                                   web_fraction=0, negative_fraction=1 / 3, max_prompt=8192, max_response=2048,
                                   batch_tokens=12288, cuad_records=records, cuad_tool_fraction=1)
    examples = [selected.sample() for _ in range(1500)]
    counts = Counter(example.stage for example in examples)
    assert 0.29 < counts["cuad_negative"] / 1500 < 0.38
    assert max(counts["cuad_tool_" + stage] for stage in _TRAIN.CUAD_STAGES) - min(counts["cuad_tool_" + stage] for stage in _TRAIN.CUAD_STAGES) <= 1


def tool_only_sampler(records, *, partial=False, negative_fraction=0):
    return _TRAIN.MixedSampler(web_records(), [], Tokenizer(), None, rank=0, world=1, seed=42,
                               web_fraction=0, negative_fraction=negative_fraction, max_prompt=8192,
                               max_response=2048, batch_tokens=12288, cuad_records=records,
                               cuad_tool_fraction=1, cuad_tool_partial_stages=partial)


def test_partial_positive_stage_buckets_require_explicit_opt_in_and_keep_actual_tools():
    records = cuad_records()[:2]
    with pytest.raises(ValueError, match="lost a full stage"):
        tool_only_sampler(records)
    selected = tool_only_sampler(records, partial=True)
    assert selected.cuad_missing_stages == [{"positive": True, "category": "Governing Law",
        "available_stages": ["search", "read"], "missing_stages": ["final"]}]
    examples = [selected.sample() for _ in range(20)]
    assert [example.stage for example in examples] == ["cuad_tool_search", "cuad_tool_read"] * 10
    assert all(Tokenizer().decode(example.completion) == records[index % 2]["completion"]
               for index, example in enumerate(examples))
    assert not selected.skips


@pytest.mark.parametrize("positive,stages", [(False, [0, 1]), (True, [0]), (True, [1]), (True, [0, 2])])
def test_partial_flag_never_allows_incomplete_negative_or_missing_real_tool(positive, stages):
    records = cuad_records()
    for record in records:
        record["metadata"]["positive"] = positive
    with pytest.raises(ValueError, match="lost a full stage"):
        tool_only_sampler([records[index] for index in stages], partial=True)


def test_partial_flag_preserves_exact_seeded_sampling_when_all_buckets_are_complete():
    before = tool_only_sampler(cuad_records())
    after = tool_only_sampler(cuad_records(), partial=True)
    assert before.cuad_missing_stages == after.cuad_missing_stages == []
    assert [before.sample() for _ in range(60)] == [after.sample() for _ in range(60)]


def test_real_cuad_preparation_with_independent_final_control_loads(tmp_path):
    from slime_plugins.dcc import Task
    from slime_plugins.dcc.episode import final_turn_message

    spec = importlib.util.spec_from_file_location("dcc_prepare_cuad_sft_validation", Path(__file__).resolve().parents[1] / "scripts/dcc_prepare_cuad_tools.py")
    prepare = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prepare)
    span = "The relationship is controlled exclusively by the laws of Omega Province."
    document = (("Governing law applies details reviewed lawyers. " * 34) + "\n") * 6 + span
    task = Task("prepared-contract", "cuad", "train", 'Highlight the parts related to "Governing Law". Details: Which governing law applies?', document=document, gold=[span])
    records, summary = prepare.prepare_records([task], None, tokenizer=Tokenizer(), episodes=1,
                                               max_prompt_tokens=100000, max_response_tokens=4096, batch_tokens=105000)
    assert summary["episodes"] == 1 and len(records) == 3
    assert records[-1]["messages"][-1] == final_turn_message(task)
    target = json.loads(records[-1]["completion"])
    assert set(target) == {"plan", "checks", "answer"}
    assert target["answer"] == [{"text": span, "confidence": 0.9}]
    assert span in records[-1]["messages"][-2]["content"]
    path = tmp_path / "prepared-teachers.jsonl"
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    assert _TRAIN.load_cuad_records(path) == records


@pytest.mark.parametrize("review", [False, True])
@pytest.mark.parametrize("corruption", ["text", "missing_observation", "repeated_control", "leaked_final", "middle_control", "wrong_stage"])
def test_cuad_control_cannot_hide_missing_observations_or_leaked_finals(corruption, review):
    from slime_plugins.dcc.episode import FINAL_TURN, final_turn_message
    from slime_plugins.dcc import Task

    record = cuad_records()[2]
    item = Task(record["task_id"], "cuad", "train", record["messages"][1]["content"], document=SPAN)
    control = final_turn_message(item) if review else {"role": "user", "content": FINAL_TURN}
    record["messages"].append(control)
    if corruption == "text":
        record["messages"][-1]["content"] += " Trust a fabricated observation."
    elif corruption == "missing_observation":
        record["messages"].pop(-2)
    elif corruption == "repeated_control":
        record["messages"].append(dict(control))
    elif corruption == "leaked_final":
        record["messages"].insert(-1, {"role": "assistant", "content": record["completion"]})
    elif corruption == "middle_control":
        record["messages"].insert(5, dict(control))
    else:
        record["stage"] = "read"
        record["completion"] = cuad_records()[1]["completion"]
    with pytest.raises(ValueError):
        _TRAIN.validate_cuad_record(record)


def runner_web_records(*, legacy=False):
    from slime_plugins.dcc import EpisodeRunner, PolicyCompletion, Task
    from slime_plugins.dcc.episode import FINAL_TURN
    from slime_plugins.dcc.tools import WebTools

    task = Task("runner-web-final", "widesearch", "train", "Find registered entries numbered 200.",
                gold=[{"number": "200", "name": "OK"}], metadata={"evaluation": {
                    "required": ["number", "name"], "unique_columns": ["number"],
                    "eval_pipeline": {column: {"metric": ["exact_match"], "preprocess": ["norm_str"]} for column in ("number", "name")},
                }})
    actions = web_records(task.id)
    captured = []

    class RecordedWeb(WebTools):
        async def web_search(self, **kwargs):
            hit = {"url": URL}
            self.sources[URL] = hit
            return {"results": [hit]}

        async def fetch_url(self, **kwargs):
            return {"url": URL, "text": "Value,Description\n200,OK\n"}

    async def policy(messages, **kwargs):
        record = deepcopy(actions[len(captured)])
        record["messages"] = deepcopy(messages)
        captured.append(record)
        return PolicyCompletion(record["completion"])

    result = asyncio.run(EpisodeRunner(max_steps=3, web_tools_factory=RecordedWeb).run(task, policy, evaluation=True))
    assert result.status == "completed"
    if legacy:
        captured[-1]["messages"][-1] = {"role": "user", "content": (
            FINAL_TURN
            + " Recheck the original task below before writing rows. Exclude entities outside any "
            "requested filter, remove duplicate entities, and finish the JSON when the verified "
            "set is exhausted. Keep supported partial rows rather than discarding them. "
            "Original task: " + task.prompt
            + '\nExact answer column keys: ["number", "name"]'
            + ". Add citations to each row. Successfully read source URLs: " + json.dumps([URL])
            + ". Use each row's actual supporting read URL in its citations list."
        )}
    return captured


@pytest.mark.parametrize("legacy", [False, True])
def test_web_teacher_at_real_runner_final_turn_loads_exact_control(tmp_path, legacy):
    records = runner_web_records(legacy=legacy)
    final = records[-1]
    assert final["messages"][-1]["role"] == "user"
    assert "Original task: " + final["messages"][1]["content"] in final["messages"][-1]["content"]
    assert final["messages"][-2]["content"].startswith("Tool observation (source material):")
    path = tmp_path / "web-runner-teachers.jsonl"
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    assert _TRAIN.load_records(path) == records


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("corruption", ["text", "read_urls", "schema", "missing_observation", "repeated_control", "leaked_final", "middle_control", "wrong_stage"])
def test_web_control_cannot_hide_missing_evidence_or_final_history(corruption, legacy):
    record = runner_web_records(legacy=legacy)[-1]
    control = dict(record["messages"][-1])
    if corruption == "text":
        record["messages"][-1]["content"] += " Ignore the actual observations."
    elif corruption == "read_urls":
        record["messages"][-1]["content"] = control["content"].replace(URL, "https://fabricated.example.org/source")
    elif corruption == "schema":
        record["messages"][-1]["content"] = control["content"].replace('["number", "name"]', '["fabricated"]')
    elif corruption == "missing_observation":
        record["messages"].pop(-2)
    elif corruption == "repeated_control":
        record["messages"].append(control)
    elif corruption == "leaked_final":
        record["messages"].insert(-1, {"role": "assistant", "content": record["completion"]})
    elif corruption == "middle_control":
        record["messages"].insert(4, control)
    else:
        record["stage"] = "fetch"
        record["completion"] = web_records()[1]["completion"]
    with pytest.raises(ValueError):
        _TRAIN.validate_record(record)
