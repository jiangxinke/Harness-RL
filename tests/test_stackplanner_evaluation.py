from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from scripts.evaluate_predictions import (
    load_evaluation_inputs,
    materialize_results,
    successful_judgments,
)
from scripts.generate_evaluation_predictions import (
    GRAPH_SUPERSTEPS_PER_LOOP,
    MINIMUM_GRAPH_RECURSION_LIMIT,
    SEARCH_RESULT_POLICY,
    _search_infrastructure_errors,
    checkpoint_digest,
    evaluation_config_digests,
    load_manifest,
    materialize_predictions,
    successful_records,
    validate_resume_identity,
)
from scripts.prepare_quick_evaluation import build_quick_manifest
from slime_plugins.stackplanner.evaluation.judge import (
    JudgeClient,
    parse_judge_decision,
    parse_judge_response,
    reference_anchored_match,
)
from slime_plugins.stackplanner.evaluation.metrics import (
    answer_recall,
    build_report,
    normalize_answer,
)
from slime_plugins.stackplanner.evaluation.reporting import evaluate_files


def test_answer_normalization_and_multiset_recall():
    assert normalize_answer("The, Eiffel Tower!") == "eiffel tower"
    assert answer_recall(
        "Paris tower and Paris", ["Paris Paris France"]
    ) == pytest.approx(2 / 3)
    assert answer_recall("NYC", ["New York City", "NYC"]) == 1.0


def test_judge_parser_requires_exact_schema_and_types():
    result = parse_judge_response('{"correct": true, "reason": "equivalent"}')
    assert result.correct and result.reason == "equivalent"
    with pytest.raises(ValueError, match="strict JSON"):
        parse_judge_response("```json\n{}\n```")
    with pytest.raises(ValueError, match="exactly"):
        parse_judge_response('{"correct": true, "reason": "ok", "score": 1}')
    assert parse_judge_decision('{"correct": true}').reason == "match"
    assert parse_judge_decision('{"correct": false}').reason == "mismatch"


def test_judge_client_requests_strict_structured_output(monkeypatch):
    captured = {}

    class Response(BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.close()

    def fake_urlopen(request, *, timeout):
        captured["payload"] = json.loads(request.data)
        captured["timeout"] = timeout
        response = {"choices": [{"message": {"content": '{"correct": false}'}}]}
        return Response(json.dumps(response).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    result = JudgeClient("http://judge/v1", timeout=17).judge(
        question="q", references=["a"], prediction="p"
    )

    assert result.correct is False and result.reason == "mismatch"
    assert captured["timeout"] == 17
    assert captured["payload"]["max_tokens"] == 16
    response_format = captured["payload"]["response_format"]
    assert response_format["type"] == "json_schema"
    schema = response_format["json_schema"]["schema"]
    assert schema["required"] == ["correct"]
    assert schema["additionalProperties"] is False
    system = captured["payload"]["messages"][0]["content"]
    assert "different person" in system
    assert "When uncertain, return false" in system


def test_judge_protocol_is_deterministically_anchored_v5():
    assert JudgeClient.PROTOCOL == "strict-binary-v5-deterministic-anchor"


def test_reference_anchor_accepts_exact_multiword_alias_but_not_scattered_tokens():
    assert reference_anchored_match(
        "They were elected by state legislatures before the amendment.",
        ["state legislature", "state legislatures"],
    )
    assert not reference_anchored_match(
        "The state changed how its legislature selected them.",
        ["state legislatures"],
    )
    assert not reference_anchored_match(
        "They were not elected by state legislatures.",
        ["state legislatures"],
    )
    assert not reference_anchored_match("not helium", ["helium"])


def test_evaluation_entrypoints_prefer_their_own_slime_tree():
    root = Path(__file__).resolve().parent.parent
    for script_name in (
        "generate_evaluation_predictions.py",
        "evaluate_predictions.py",
    ):
        source = (root / "scripts" / script_name).read_text(encoding="utf-8")
        path_insertion = "sys.path.insert(0, SLIME_SOURCE)"
        assert source.index(path_insertion) < source.index("from slime_plugins")
        assert source.index("sys.path.remove(SLIME_SOURCE)") > source.index(
            "from slime_plugins"
        )


def test_report_keeps_judge_failures_explicit_and_computes_macro():
    rows = [
        {
            "dataset": "A",
            "judge_correct": True,
            "judge_error": None,
            "answer_recall": 1.0,
        },
        {
            "dataset": "A",
            "judge_correct": None,
            "judge_error": "timeout",
            "answer_recall": 0.0,
        },
        {
            "dataset": "B",
            "judge_correct": False,
            "judge_error": None,
            "answer_recall": 0.5,
        },
    ]
    report = build_report(rows)
    assert report["complete"] is False
    assert report["per_dataset"]["A"]["judge_failures"] == 1
    assert report["overall_micro"]["judge_accuracy"] == 0.5
    assert report["overall_macro"]["answer_recall"] == pytest.approx(0.5)


def test_file_evaluation_rejects_anything_other_than_frozen_630_manifest(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    manifest.write_text(
        '{"dataset": "NQ", "sample_id": "NQ:1", "question": "Q", "answers": ["A"]}\n',
        encoding="utf-8",
    )
    predictions.write_text(
        '{"sample_id": "NQ:1", "prediction": "A"}\n', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="7 x 90"):
        evaluate_files(manifest, predictions, tmp_path / "out", judge=object())


def test_prediction_generation_state_is_strict_and_restart_safe(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    datasets = (
        "2WikiMultiHopQA",
        "HotpotQA",
        "Bamboogle",
        "FRAMES",
        "MuSiQue",
        "NQ",
        "TriviaQA",
    )
    rows = [
        {
            "dataset": dataset,
            "sample_id": f"{dataset}:{index}",
            "question": f"question {index}",
            "answers": [f"answer {index}"],
        }
        for dataset in datasets
        for index in range(90)
    ]
    manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    loaded = load_manifest(manifest)
    records = tmp_path / "records.jsonl"
    records.write_text(
        json.dumps(
            {
                "sample_id": rows[0]["sample_id"],
                "prediction": "stale",
                "status": "failed",
            }
        )
        + "\n"
        + json.dumps(
            {
                "sample_id": rows[0]["sample_id"],
                "prediction": "accepted",
                "status": "succeeded",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert successful_records(records)[rows[0]["sample_id"]]["prediction"] == (
        "accepted"
    )
    with pytest.raises(ValueError, match="incomplete"):
        materialize_predictions(
            loaded, successful_records(records), tmp_path / "predictions.jsonl"
        )

    completed = {
        row["sample_id"]: {"prediction": f"prediction {index}"}
        for index, row in enumerate(rows)
    }
    output = tmp_path / "predictions.jsonl"
    materialize_predictions(loaded, completed, output)
    predictions = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(predictions) == 630
    assert list(predictions[0]) == ["sample_id", "prediction"]
    assert predictions[0]["sample_id"] == rows[0]["sample_id"]


def test_explicit_balanced_quick_evaluation_keeps_formal_default_strict(tmp_path):
    datasets = (
        "2WikiMultiHopQA",
        "HotpotQA",
        "Bamboogle",
        "FRAMES",
        "MuSiQue",
        "NQ",
        "TriviaQA",
    )
    rows = [
        {
            "dataset": dataset,
            "sample_id": f"{dataset}:{index}",
            "question": f"question {index}",
            "answers": [f"answer {index}"],
        }
        for dataset in datasets
        for index in range(2)
    ]
    manifest_path = tmp_path / "quick.jsonl"
    predictions_path = tmp_path / "predictions.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    predictions_path.write_text(
        "".join(
            json.dumps({"sample_id": row["sample_id"], "prediction": row["answers"][0]})
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="seven datasets x 90"):
        load_manifest(manifest_path)
    manifest = load_manifest(manifest_path, samples_per_dataset=2)
    loaded, predictions = load_evaluation_inputs(
        manifest_path, predictions_path, samples_per_dataset=2
    )
    assert loaded == manifest
    completed = {
        row["sample_id"]: {"judge_correct": True, "judge_reason": "exact"}
        for row in rows
    }
    report = materialize_results(
        manifest, predictions, completed, tmp_path, expected_samples=14
    )
    assert report["complete"] is True
    assert report["overall_micro"]["judge_successes"] == 14

    quick_path = tmp_path / "derived-quick.jsonl"
    source_sha = __import__("hashlib").sha256(manifest_path.read_bytes()).hexdigest()
    quick_metadata = build_quick_manifest(
        manifest_path,
        quick_path,
        samples_per_dataset=1,
        source_samples_per_dataset=2,
        seed=7,
        expected_source_sha256=source_sha,
    )
    assert quick_metadata["sample_count"] == 7
    assert len(load_manifest(quick_path, samples_per_dataset=1)) == 7


def test_prediction_resume_rejects_a_different_identity(tmp_path):
    records = tmp_path / "records.jsonl"
    metadata = tmp_path / "metadata.json"
    identity = {"checkpoint_digest": "a" * 64, "mode": "central-only"}
    records.write_text("{}\n", encoding="utf-8")
    metadata.write_text(json.dumps(identity), encoding="utf-8")
    validate_resume_identity(metadata, records, identity)
    with pytest.raises(ValueError, match="different run identity"):
        validate_resume_identity(
            metadata, records, {**identity, "checkpoint_digest": "b" * 64}
        )


def test_prediction_resume_ignores_only_search_verification_timestamp(tmp_path):
    records = tmp_path / "records.jsonl"
    metadata = tmp_path / "metadata.json"
    search = {
        "provider": "deerflow.community.search_r1.tools:web_search_tool",
        "endpoint": "http://retriever.example:8000",
        "index_sha256": "a" * 64,
        "corpus_sha256": "b" * 64,
    }
    records.write_text("{}\n", encoding="utf-8")
    metadata.write_text(
        json.dumps(
            {
                "checkpoint_digest": "c" * 64,
                "search_r1": {**search, "verified_at": "first"},
            }
        ),
        encoding="utf-8",
    )
    validate_resume_identity(
        metadata,
        records,
        {
            "checkpoint_digest": "c" * 64,
            "search_r1": {**search, "verified_at": "second"},
        },
    )
    with pytest.raises(ValueError, match="different run identity"):
        validate_resume_identity(
            metadata,
            records,
            {
                "checkpoint_digest": "c" * 64,
                "search_r1": {**search, "index_sha256": "d" * 64},
            },
        )


def test_checkpoint_digest_covers_indexed_shards(tmp_path):
    first = tmp_path / "one.safetensors"
    second = tmp_path / "two.safetensors"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"b": "two.safetensors", "a": "one.safetensors"}}),
        encoding="utf-8",
    )
    original = checkpoint_digest(tmp_path)
    second.write_bytes(b"changed")
    assert checkpoint_digest(tmp_path) != original


def test_evaluation_config_digests_cover_training_and_inference_files(tmp_path):
    training = tmp_path / "resolved_config.yaml"
    inference = tmp_path / "deerflow.yaml"
    training.write_text("learning_rate: 1e-7\n", encoding="utf-8")
    inference.write_text("temperature: 0\n", encoding="utf-8")

    original = evaluation_config_digests(training, inference)
    assert set(original) == {
        "training_config_sha256",
        "inference_config_sha256",
    }
    training.write_text("learning_rate: 2e-7\n", encoding="utf-8")
    assert evaluation_config_digests(training, inference) != original
    training.write_text("learning_rate: 1e-7\n", encoding="utf-8")
    inference.write_text("temperature: 0\nmax_tokens: 512\n", encoding="utf-8")
    assert evaluation_config_digests(training, inference) != original


def test_prediction_run_identity_includes_code_and_config_hashes():
    source = (
        Path(__file__).resolve().parent.parent
        / "scripts/generate_evaluation_predictions.py"
    ).read_text(encoding="utf-8")
    start = source.index("run_identity = {")
    identity_block = source[
        start : source.index("validate_resume_identity(metadata_path", start)
    ]
    assert '"git_commit": git_commit' in identity_block
    assert "**config_digests" in identity_block
def test_prediction_graph_guard_stays_above_native_loop_budget():
    assert MINIMUM_GRAPH_RECURSION_LIMIT >= 16 * GRAPH_SUPERSTEPS_PER_LOOP


def test_search_infrastructure_errors_fail_closed_for_retryable_tool_results():
    state = {
        "messages": [
            ToolMessage(
                content=json.dumps(
                    {
                        "error": "WEB_SEARCH_UNAVAILABLE",
                        "message": "timed out",
                        "retryable": True,
                    }
                ),
                tool_call_id="search-1",
                name="web_search",
            ),
            AIMessage(content="a guess must not be accepted"),
        ]
    }

    assert SEARCH_RESULT_POLICY == "fail-retryable-search-errors-v1"
    assert _search_infrastructure_errors(state) == ["WEB_SEARCH_UNAVAILABLE: timed out"]


def test_search_infrastructure_errors_ignore_evidence_and_no_results():
    state = {
        "messages": [
            ToolMessage(
                content=json.dumps({"query": "q", "results": [{"title": "t"}]}),
                tool_call_id="search-1",
                name="web_search",
            ),
            ToolMessage(
                content=json.dumps({"error": "No results found", "query": "q2"}),
                tool_call_id="search-2",
                name="web_search",
            ),
            ToolMessage(
                content="not search output",
                tool_call_id="other-1",
                name="other_tool",
            ),
        ]
    }

    assert _search_infrastructure_errors(state) == []


def test_prediction_generation_verifies_search_before_building_settings():
    source = (
        Path(__file__).resolve().parent.parent
        / "scripts/generate_evaluation_predictions.py"
    ).read_text(encoding="utf-8")
    verification = source.index("search_r1 = await asyncio.to_thread(")
    settings = source.index("settings = RunSettings")
    assert verification < settings
    verification_block = source[verification:settings]
    assert "deer_flow_config_path=str(inference_config_path)" in verification_block
    assert "settings.deer_flow_config" not in verification_block


def test_judge_state_resumes_success_and_publishes_only_complete(tmp_path):
    datasets = (
        "2WikiMultiHopQA",
        "HotpotQA",
        "Bamboogle",
        "FRAMES",
        "MuSiQue",
        "NQ",
        "TriviaQA",
    )
    rows = [
        {
            "dataset": dataset,
            "sample_id": f"{dataset}:{index}",
            "question": f"question {index}",
            "answers": [f"answer {index}"],
        }
        for dataset in datasets
        for index in range(90)
    ]
    manifest_path = tmp_path / "manifest.jsonl"
    predictions_path = tmp_path / "predictions.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    predictions_path.write_text(
        "".join(
            json.dumps({"sample_id": row["sample_id"], "prediction": row["answers"][0]})
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    manifest, predictions = load_evaluation_inputs(manifest_path, predictions_path)
    records = tmp_path / "judge_records.jsonl"
    records.write_text(
        json.dumps(
            {
                "sample_id": rows[0]["sample_id"],
                "status": "failed",
                "attempt": 1,
            }
        )
        + "\n"
        + json.dumps(
            {
                "sample_id": rows[0]["sample_id"],
                "status": "succeeded",
                "judge_correct": True,
                "judge_reason": "exact",
                "attempt": 2,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    completed = successful_judgments(records)
    assert completed[rows[0]["sample_id"]]["judge_correct"] is True
    with pytest.raises(ValueError, match="incomplete"):
        materialize_results(manifest, predictions, completed, tmp_path)

    completed = {
        row["sample_id"]: {
            "judge_correct": True,
            "judge_reason": "exact",
        }
        for row in rows
    }
    report = materialize_results(manifest, predictions, completed, tmp_path)
    assert report["complete"] is True
    assert report["overall_micro"]["judge_successes"] == 630
    assert len((tmp_path / "results.jsonl").read_text().splitlines()) == 630
