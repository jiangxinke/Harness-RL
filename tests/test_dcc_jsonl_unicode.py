"""Literal Unicode separators in JSON strings must survive LF JSONL records.

Invented CPU-only fixtures; no real benchmark, model, network, or GPU access.
"""
import asyncio
import json
from pathlib import Path
import runpy
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "slime")]
from slime_plugins.dcc import PolicyCompletion, Task
from slime_plugins.dcc.episode import _terminal_widesearch_rows

SEPARATORS = ("\x85", "\u2028", "\u2029")


def dump_jsonl(path, rows, *, ending="\n"):
    path.write_bytes(ending.join(json.dumps(row, ensure_ascii=False) for row in rows).encode("utf-8")
                     + ending.encode("utf-8"))


@pytest.mark.parametrize("separator", SEPARATORS)
@pytest.mark.parametrize("ending", ("\n", "\r\n"))
def test_prediction_readers_and_scoring_preserve_unicode_text(tmp_path, separator, ending):
    value = "Alpha" + separator + "Beta\nLiteral escaped newline"
    rows = [dict(task_id="one", status="completed", answer=[dict(text=value, confidence=1)]),
            dict(task_id="two", status="completed", answer=[])]
    path = tmp_path / "predictions_rank0.jsonl"
    dump_jsonl(path, rows, ending=ending)
    simple = runpy.run_path(str(ROOT / "scripts/dcc_evaluate.py"))
    assert simple["read_predictions"](path) == rows
    tasks = [Task(key, "cuad", "validation", "Find clause", document=value, gold=[])
             for key in ("one", "two")]
    scoring = runpy.run_path(str(ROOT / "scripts/dcc_model_evaluate.py"))
    saved, manifest, runtime = scoring["_score_predictions"](tmp_path, tasks)
    assert saved == {row["task_id"]: row for row in rows}
    assert manifest == {} and runtime is None


def test_completed_evaluation_resume_preserves_unicode_and_does_not_regenerate(tmp_path, monkeypatch):
    fixture = runpy.run_path(str(ROOT / "tests/test_dcc_model_scoring.py"))
    value = "Alpha" + "".join(SEPARATORS) + "Beta"
    calls = []
    class Policy:
        provenance = {"backend": "vllm", "model": "example/model", "adapter": None}
        async def __call__(self, messages, **kwargs):
            calls.append("generate")
            text = json.dumps({"answer": [{"text": value, "confidence": 1}]}, ensure_ascii=False)
            return PolicyCompletion(text, [1], [2])
        async def close(self):
            calls.append("close")
    tasks = [Task("synthetic-contract", "cuad", "validation", "Find clause", document=value, gold=[value])]
    evaluate, args = fixture["stub_evaluation"](monkeypatch, tmp_path, Policy(), tasks)
    asyncio.run(evaluate(args))
    before = (tmp_path / "predictions_rank0.jsonl").read_bytes()
    assert all(separator.encode() in before for separator in SEPARATORS)
    asyncio.run(evaluate(args))
    assert calls == ["generate", "close"]
    assert (tmp_path / "predictions_rank0.jsonl").read_bytes() == before
    saved, _, _ = fixture["cli"]()["_score_predictions"](tmp_path, tasks)
    assert saved[tasks[0].id]["answer"] == [{"text": value, "confidence": 1}]


@pytest.mark.parametrize("separator", SEPARATORS)
def test_native_trace_unicode_reaches_original_ownership_guard(tmp_path, separator):
    fixture = runpy.run_path(str(ROOT / "tests/test_dcc_native_thinking.py"))
    from slime_plugins.dcc.native_thinking_eval import validate_native_sidecars
    manifest, runtime = fixture["declarations"](3)
    trace = dict(trace_id="trace", task_id="foreign-task", rank=0,
                 native_thinking=manifest["native_thinking"],
                 source_commit=manifest["native_thinking"]["source_receipt"]["source_commit"],
                 protocol=manifest["native_thinking"]["protocol"],
                 raw_source_text="Alpha" + separator + "Beta")
    dump_jsonl(tmp_path / "native_thinking_traces_rank0.jsonl", [trace])
    # A Unicode string is parsed intact, then the unchanged semantic guard
    # rejects the foreign trace; no real engine or pretrained tokenizer exists.
    with pytest.raises(ValueError, match="Native trace ownership/source/protocol mismatch"):
        validate_native_sidecars(tmp_path, {}, manifest, runtime, {}, [], tokenizer=fixture["FakeTokenizer"]())


@pytest.mark.parametrize("separator", SEPARATORS)
def test_grounded_reviewed_rows_preserve_exact_unicode_gold_target_and_ids(tmp_path, separator):
    fixture = runpy.run_path(str(ROOT / "tests/test_grounded_final.py"))
    builder, trainer = fixture["B"], fixture["T"]
    dataset = tmp_path / "dataset"
    tokenizer, rows, manifest, _ = fixture["synthetic_dataset"](dataset)
    span = "Alpha" + separator + "Beta"
    rows[0].update(gold=[span], span_mappings=[dict(target_text=span)],
                   **builder.encode_target(tokenizer, [span], final_review=False, eos_ids=[151643, 151645]))
    rows[0]["total_tokens"] = len(rows[0]["prompt_ids"]) + len(rows[0]["target_ids"])
    payload = builder.rows_bytes(rows)
    assert separator.encode() in payload
    manifest.update(rows_bytes=len(payload), rows_blake2b_256=builder.digest(payload))
    (dataset / "rows.jsonl").write_bytes(payload)
    (dataset / "manifest.json").write_text(builder.canonical(manifest))
    approval = trainer.training_binding(manifest, training_source_commit="e"*40, derivative_id="synthetic-derivative")
    loaded, _, binding = trainer.load_reviewed_rows(dataset, approval, tokenizer,
        training_source_commit="e"*40, derivative_id="synthetic-derivative")
    assert loaded == rows and binding == approval
    assert loaded[0]["gold"] == [span] and separator in loaded[0]["target_text"]
    # Keep the existing rejection of an internal blank JSONL record.
    bad = payload.replace(b"\n", b"\n\n", 1)
    manifest.update(rows_bytes=len(bad), rows_blake2b_256=builder.digest(bad))
    (dataset / "rows.jsonl").write_bytes(bad)
    (dataset / "manifest.json").write_text(builder.canonical(manifest))
    approval = trainer.training_binding(manifest, training_source_commit="e"*40, derivative_id="synthetic-derivative")
    with pytest.raises(json.JSONDecodeError):
        trainer.load_reviewed_rows(dataset, approval, tokenizer,
            training_source_commit="e"*40, derivative_id="synthetic-derivative")


@pytest.mark.parametrize("separator", SEPARATORS)
def test_widesearch_terminal_json_rows_preserve_unicode_values(separator):
    rows = [{"Name": "Alpha" + separator + "Beta", "citations": ["https://example.org/a"]},
            {"Name": "Gamma", "citations": ["https://example.org/b"]}]
    text = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    assert _terminal_widesearch_rows(text) == rows
    assert _terminal_widesearch_rows("```json\n" + text + "\n```") == rows


def test_cached_reference_reader_preserves_unicode_and_orphan_rejection(tmp_path, monkeypatch):
    import dcc_train_cached_rows as cached
    text = "Alpha" + "".join(SEPARATORS) + "Beta"
    path = tmp_path / "native_web_train.jsonl"
    dump_jsonl(path, [dict(collector={"public_source_text": text}, record={"task_id": "synthetic"})])
    seen = []
    def source_receipt(collector, registry, *, task_id):
        assert collector == {"public_source_text": text} and task_id == "synthetic"
        seen.append(task_id)
        return [{"source_id": "source"}]
    monkeypatch.setattr(cached, "receipt_sources", source_receipt)
    cached.validate_references(path, {"source": {}})
    assert seen == ["synthetic"]
    with pytest.raises(ValueError, match="missing or orphan"):
        cached.validate_references(path, {"source": {}, "orphan": {}})


def test_widesearch_download_inventory_and_preparation_use_lf_jsonl(tmp_path, monkeypatch):
    prepare = runpy.run_path(str(ROOT / "scripts/dcc_prepare_data.py"))
    source = tmp_path / "raw"
    (source / "cuad").mkdir(parents=True)
    for name in ("CUADv1.json", "test.json"):
        (source / "cuad" / name).write_text("{}")
    wide = source / "widesearch"
    (wide / "widesearch_gold").mkdir(parents=True)
    config = {"required": ["name"], "unique_columns": ["name"],
              "eval_pipeline": {"name": {"metric": ["exact_match"]}}}
    query = "List Alpha" + "".join(SEPARATORS) + "Beta"
    rows = [dict(instance_id=f"synthetic_{i}", query=query, language="en", evaluation=json.dumps(config))
            for i in range(2)]
    dump_jsonl(wide / "widesearch.jsonl", rows)
    for row in rows:
        (wide / "widesearch_gold" / (row["instance_id"] + ".csv")).write_text("Name\nAlpha\n")
    def no_network(*args, **kwargs):
        raise AssertionError("Synthetic local inventory must never download")
    monkeypatch.setitem(prepare["download_sources"].__globals__, "_get", no_network)
    prepare["download_sources"](source)
    out = tmp_path / "prepared"
    summary = prepare["prepare_widesearch"](wide, out)
    assert summary["tasks"] == {"train": 0, "validation": 0, "test": 2}
    with (out / "test.jsonl").open() as handle:
        prepared = [json.loads(line) for line in handle]
    assert [row["prompt"] for row in prepared] == [query, query]
