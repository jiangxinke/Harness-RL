from __future__ import annotations

import dataclasses
import asyncio
import importlib.util
import json
from pathlib import Path
import runpy
from types import SimpleNamespace

import pytest
from slime_plugins.dcc.data import Task
from slime_plugins.dcc.retrieval import (
    document_read_windows,
    expand_document_context,
    fit_retrieval_profile,
    load_retrieval_profile,
    search_document,
)
from slime_plugins.dcc.tools import DocumentTools

QUESTION = 'Highlight the parts related to "Governing Law". Details: Which governing law applies?'


def examples():
    return [
        Task(str(i), "cuad", "train", QUESTION,
             document="Delaware jurisdiction applies.", gold=["Delaware jurisdiction applies."],
             metadata={"category": "This field is ignored"})
        for i in range(3)
    ]


def profile(tmp_path):
    fitted = fit_retrieval_profile(examples())
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(fitted))
    return fitted, path, load_retrieval_profile(path)


@pytest.mark.parametrize("split,benchmark", [("validation", "cuad"), ("test", "cuad"), ("train", "widesearch")])
def test_fitting_rejects_nontraining_rows_even_without_answers(split, benchmark):
    with pytest.raises(ValueError, match="CUAD train"):
        fit_retrieval_profile([*examples(), dataclasses.replace(examples()[0], id="forbidden", split=split, benchmark=benchmark, gold=[])])


def test_profile_uses_question_text_and_training_spans_not_arbitrary_metadata(tmp_path):
    fitted, _, loaded = profile(tmp_path)
    assert fitted["training"]["tasks"] == 3
    assert fitted["training"]["positive_tasks"] == 3
    assert fitted["questions"][QUESTION]["category"] == "Governing Law"
    assert "delaware" in loaded.questions[QUESTION]
    assert "This field is ignored" not in json.dumps(fitted)
    assert "Delaware jurisdiction applies." not in json.dumps(fitted)
    changed = [dataclasses.replace(row, metadata={"category": "different", "gold": "not used"}) for row in examples()]
    assert fit_retrieval_profile(changed) == fitted
    assert fit_retrieval_profile(reversed(examples())) == fitted


def test_profile_disabled_unknown_question_and_unknown_category_preserve_baseline(tmp_path):
    _, _, loaded = profile(tmp_path)
    tools = DocumentTools("Delaware jurisdiction applies. " * 50)
    assert load_retrieval_profile(None) is None
    for question, selected in [(QUESTION, None), ("An unseen question", loaded), (QUESTION.replace("Governing Law", "Unknown"), loaded)]:
        assert search_document(tools, question, profile=selected) == tools.search_document(question)
    assert search_document(tools, "  An unseen question  ", profile=loaded) == tools.search_document("  An unseen question  ")


def test_insufficient_training_support_falls_back_without_using_eval_labels(tmp_path):
    path = tmp_path / "sparse.json"
    path.write_text(json.dumps(fit_retrieval_profile(examples(), min_examples=99)))
    loaded = load_retrieval_profile(path)
    tools = DocumentTools("Delaware jurisdiction applies. " * 50)
    assert loaded.questions[QUESTION] == ()
    assert search_document(tools, QUESTION, profile=loaded) == tools.search_document(QUESTION)


def test_expansion_improves_reach_without_adding_chunks_or_changing_source_offsets(tmp_path):
    _, _, loaded = profile(tmp_path)
    distractor = ("governing law applies parts details which " * 7)[:280] + "\n"
    source = distractor * 5 + ("Delaware jurisdiction applies. " * 9) + "\n"
    tools = DocumentTools(source, chunk_chars=300, overlap_chars=0)
    original = tools.search_document(QUESTION, top_k=4)
    enhanced = search_document(tools, QUESTION, top_k=4, profile=loaded)
    assert not any("Delaware" in hit["text"] for hit in original["hits"])
    assert any("Delaware" in hit["text"] for hit in enhanced["hits"])
    assert len(enhanced["hits"]) == len(original["hits"]) == 4
    assert len({(hit["start"], hit["end"]) for hit in enhanced["hits"]}) == 4
    assert all(hit["text"] == source[hit["start"]:hit["end"]] for hit in enhanced["hits"])
    assert search_document(tools, QUESTION, profile=loaded) == enhanced
    assert search_document(tools, "  " + QUESTION + "  ", profile=loaded) == enhanced
    assert len(search_document(tools, QUESTION, top_k=100, profile=loaded)["hits"]) <= 8


def test_context_expansion_repairs_a_clause_crossing_a_chunk_boundary():
    source = "x" * 150 + "The governing law is Delaware." + "x" * 170
    tools = DocumentTools(source, chunk_chars=160, overlap_chars=0)
    observation = {"document_length": len(source), "hits": [
        {"start": 160, "end": 320, "text": source[160:320], "score": 2.0},
        {"start": 0, "end": 160, "text": source[:160], "score": 1.0},
    ]}
    before = json.loads(json.dumps(observation))
    expanded = expand_document_context(tools, observation, context_chars=30, max_chars=400)
    assert "The governing law is Delaware." not in observation["hits"][0]["text"]
    assert len(expanded["hits"]) == 1
    assert "The governing law is Delaware." in expanded["hits"][0]["text"]
    assert expanded["hits"][0]["score"] == 2.0
    assert all(hit["text"] == source[hit["start"]:hit["end"]] for hit in expanded["hits"])
    assert observation == before


def test_context_expansion_is_opt_in_and_bounded_without_reading_labels():
    tools = DocumentTools("word " * 2000)
    baseline = tools.search_document("word", top_k=8)
    assert search_document(tools, "word") == tools.search_document("word")
    assert expand_document_context(tools, baseline, context_chars=0, max_chars=1) is baseline
    expanded = search_document(tools, "word", top_k=8, context_chars=800, max_context_chars=5000)
    assert 0 < sum(len(hit["text"]) for hit in expanded["hits"]) <= 5000
    assert expanded["document_length"] == len(tools.document)
    assert len({(hit["start"], hit["end"]) for hit in expanded["hits"]}) == len(expanded["hits"])


@pytest.mark.parametrize("kwargs", [{"context_chars": -1}, {"context_chars": True}, {"max_chars": 0}, {"max_chars": 1.5}])
def test_context_expansion_rejects_invalid_budgets(kwargs):
    tools = DocumentTools("source")
    with pytest.raises(ValueError):
        expand_document_context(tools, tools.search_document("source"), **kwargs)


def test_context_expansion_rejects_fabricated_source_offsets():
    tools = DocumentTools("source")
    with pytest.raises(ValueError, match="exact source"):
        expand_document_context(tools, {"hits": [{"start": 0, "end": 6, "text": "invent", "score": 1}]})


def test_public_read_windows_are_derived_from_hits_and_reach_backwards():
    windows = document_read_windows({"start": 5000, "end": 6500})
    assert {"start": 4200, "end": 7300} in windows
    assert {"start": 1000, "end": 13000} in windows
    assert len(windows) == len({tuple(window.items()) for window in windows})
    assert all(window["start"] >= 0 for window in document_read_windows({"start": 20, "end": 100}))


def test_explicit_missing_or_malformed_profile_fails_fast(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_retrieval_profile(tmp_path / "missing.json")
    path = tmp_path / "invalid.json"
    path.write_text("not JSON")
    with pytest.raises(ValueError, match="malformed"):
        load_retrieval_profile(path)


@pytest.mark.parametrize("change", [
    lambda p: p.update(version=999),
    lambda p: p.update(version=True),
    lambda p: p["training"].update(split="validation"),
    lambda p: p["training"].update(positive_tasks=99),
    lambda p: p["parameters"].update(rrf_constant=float("nan")),
    lambda p: p["parameters"].update(min_examples=0),
    lambda p: p["questions"][QUESTION].update(category="Metadata must not choose the category"),
    lambda p: p["questions"][QUESTION].update(terms=["Delaware jurisdiction applies."]),
])
def test_loader_rejects_invalid_configuration(tmp_path, change):
    fitted, path, _ = profile(tmp_path)
    change(fitted)
    path.write_text(json.dumps(fitted))
    with pytest.raises(ValueError):
        load_retrieval_profile(path)


def test_fitting_rejects_duplicates_bad_labels_and_unusable_corpus():
    with pytest.raises(ValueError, match="duplicate"):
        fit_retrieval_profile([examples()[0], examples()[0]])
    with pytest.raises(ValueError, match="gold"):
        fit_retrieval_profile([dataclasses.replace(examples()[0], gold=[""])])
    with pytest.raises(ValueError, match="positive answers"):
        fit_retrieval_profile([dataclasses.replace(examples()[0], gold=[])])


def test_fitter_cli_writes_external_profile_and_rejects_validation(tmp_path):
    spec = importlib.util.spec_from_file_location("dcc_fit_retrieval_cli", Path(__file__).resolve().parents[1] / "scripts/dcc_fit_retrieval.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    source, output = tmp_path / "train.jsonl", tmp_path / "artifacts/profile.json"
    source.write_text("\n".join(json.dumps(dataclasses.asdict(row)) for row in examples()))
    assert cli.main(["--data", str(source), "--output", str(output)]) == 0
    assert load_retrieval_profile(output) is not None
    with pytest.raises(FileExistsError):
        cli.main(["--data", str(source), "--output", str(output)])
    source.write_text(json.dumps(dataclasses.asdict(dataclasses.replace(examples()[0], split="validation"))))
    with pytest.raises(ValueError, match="CUAD train"):
        cli.main(["--data", str(source), "--output", str(tmp_path / "bad.json")])
    assert not (tmp_path / "bad.json").exists()


def test_evaluator_resumes_unchanged_profile_and_rejects_changed_contents(monkeypatch, tmp_path):
    fitted, profile_path, _ = profile(tmp_path)
    evaluate = runpy.run_path(str(Path(__file__).parents[1] / "scripts/dcc_model_evaluate.py"))["evaluate"]
    task = dataclasses.replace(examples()[0], split="validation")
    monkeypatch.setitem(evaluate.__globals__, "load_tasks", lambda path: [task])

    def unexpected(*args, **kwargs):
        pytest.fail("Resuming completed predictions must not load a GPU model")

    monkeypatch.setitem(evaluate.__globals__, "load_policy_model", unexpected)
    for name in ("RANK", "LOCAL_RANK"):
        monkeypatch.setenv(name, "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    args = SimpleNamespace(model="example/model", adapter=None, data=["tasks.jsonl"],
                           max_steps=3, max_new_tokens=1024, max_prompt_tokens=8192,
                           initial_chunks=4, seed=42, limit=0, output_dir=str(tmp_path),
                           retrieval_profile=str(profile_path), retry_infrastructure=False)
    (tmp_path / "predictions_rank0.jsonl").write_text(
        json.dumps({"task_id": task.id, "status": "completed"}) + "\n"
    )
    asyncio.run(evaluate(args))
    asyncio.run(evaluate(args))
    fitted["questions"][QUESTION]["terms"] = ["replacement"]
    profile_path.write_text(json.dumps(fitted))
    with pytest.raises(ValueError, match="configuration changed"):
        asyncio.run(evaluate(args))
