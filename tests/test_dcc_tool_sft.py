"""CPU checks for leakage, evidence, balancing and completion-only tool SFT."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import dataclasses
import importlib.util
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

_SPEC = importlib.util.spec_from_file_location("dcc_tool_sft_tests", Path(__file__).resolve().parents[1] / "scripts/dcc_tool_sft.py")
_TRAIN = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _TRAIN
_SPEC.loader.exec_module(_TRAIN)

URL = "https://example.org/registry.csv"


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


@pytest.mark.parametrize("arguments", [{"start": 100}, {"start": 100, "end": 12100}, {"start": 100, "end": 100 + len(SPAN)}])
def test_cuad_read_can_use_hit_boundary_or_public_reader_windows(arguments):
    record = cuad_records()[1]
    record["completion"] = json.dumps({"tool": "read_document", "arguments": arguments})
    assert _TRAIN.validate_cuad_record(record) is record


@pytest.mark.parametrize("arguments", [{"start": 600, "end": 700}, {"start": 100, "end": 250}, {"start": 100, "end": 12101}])
def test_cuad_read_rejects_unexposed_starts_and_arbitrary_gold_selected_ends(arguments):
    record = cuad_records()[1]
    record["completion"] = json.dumps({"tool": "read_document", "arguments": arguments})
    with pytest.raises(ValueError, match="CUAD read"):
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
