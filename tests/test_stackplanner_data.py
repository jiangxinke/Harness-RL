from __future__ import annotations

import json
from pathlib import Path

from slime_plugins.stackplanner.data.asearcher import (
    convert_rows,
    convert_training_file,
)
from slime_plugins.stackplanner.data.sampling import (
    EVALUATION_SOURCES,
    normalize_rows,
    prepare_evaluation_manifest,
    stable_sample,
)


def test_training_conversion_preserves_question_answer_and_provenance(tmp_path: Path):
    source = tmp_path / "source.jsonl"
    output = tmp_path / "output.jsonl"
    source.write_text(
        json.dumps({"question": "  Who?  ", "answer": "Alice", "id": "q-1"}) + "\n",
        encoding="utf-8",
    )
    assert convert_training_file(source, output) == 1
    row = json.loads(output.read_text(encoding="utf-8"))
    assert row["prompt"] == "Who?"
    assert row["label"] == "Alice"
    assert row["metadata"]["source_id"] == "q-1"
    assert len(list(convert_rows([{"question": "Q", "answer": ["A"]}]))) == 1


def test_evaluation_normalization_supports_jsonl_and_frames_fields():
    regular = normalize_rows("NQ", [{"question": "Q?", "answer": ["A", "Alias"]}])
    frames = normalize_rows("FRAMES", [{"Question": "F?", "answers": [{"text": "B"}]}])
    assert regular[0]["answers"] == ["A", "Alias"]
    assert frames[0]["answers"] == ["B"]


def test_stable_sample_is_order_independent_and_seeded():
    rows = [{"sample_id": f"NQ:{index}"} for index in range(20)]
    first = stable_sample(rows, dataset="NQ", count=7, seed=42)
    second = stable_sample(reversed(rows), dataset="NQ", count=7, seed=42)
    third = stable_sample(rows, dataset="NQ", count=7, seed=43)
    assert first == second
    assert first != third


def test_prepare_evaluation_manifest_selects_exactly_seven_times_ninety(tmp_path: Path):
    source_root = tmp_path / "raw"
    for dataset, relative_path in EVALUATION_SOURCES.items():
        path = source_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if dataset == "FRAMES":
            rows = [
                {
                    "id": f"{dataset}-{index}",
                    "Question": f"Question {index}",
                    "answers": [f"Answer {index}"],
                }
                for index in range(100)
            ]
            path.write_text(json.dumps(rows), encoding="utf-8")
        else:
            rows = [
                {
                    "id": f"{dataset}-{index}",
                    "question": f"Question {index}",
                    "answer": f"Answer {index}",
                }
                for index in range(100)
            ]
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
    first = tmp_path / "evaluation.jsonl"
    second = tmp_path / "evaluation-again.jsonl"
    manifest = prepare_evaluation_manifest(source_root, first, download=False)
    repeated = prepare_evaluation_manifest(source_root, second, download=False)
    rows = [json.loads(line) for line in first.read_text(encoding="utf-8").splitlines()]
    assert manifest["sample_count"] == 630
    assert all(source["selected_rows"] == 90 for source in manifest["sources"].values())
    assert {
        dataset: sum(row["dataset"] == dataset for row in rows)
        for dataset in EVALUATION_SOURCES
    } == {dataset: 90 for dataset in EVALUATION_SOURCES}
    assert first.read_bytes() == second.read_bytes()
    assert first.read_bytes().count(b"\r\n") == 630
    assert first.read_bytes().replace(b"\r\n", b"").count(b"\n") == 0
    assert manifest["output_sha256"] == repeated["output_sha256"]
