import importlib.util
import json
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location(
    "dcc_prepare_data",
    Path(__file__).resolve().parents[1] / "scripts" / "dcc_prepare_data.py",
)
prepare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare)


def contract(title):
    return {
        "title": title,
        "paragraphs": [
            {
                "context": "source " + title,
                "qas": [
                    {
                        "id": title + "__Parties",
                        "question": "Who signed?",
                        "answers": [{"text": title, "answer_start": 7}],
                    }
                ],
            }
        ],
    }


def test_cuad_keeps_official_test_and_splits_whole_contracts(tmp_path):
    source = tmp_path / "raw"
    source.mkdir()
    contracts = [contract(name) for name in ["a", "b", "c", "test"]]
    (source / "CUADv1.json").write_text(json.dumps({"data": contracts}))
    (source / "test.json").write_text(json.dumps({"data": [contracts[-1]]}))
    output = tmp_path / "prepared"
    summary = prepare.prepare_cuad(source, output, validation_contracts=1)
    assert summary["contracts"] == {"train": 2, "validation": 1, "test": 1}
    documents = json.loads((output / "documents.json").read_text())
    titles = {}
    for split in ["train", "validation", "test"]:
        rows = [
            json.loads(line)
            for line in (output / (split + ".jsonl")).read_text().splitlines()
        ]
        titles[split] = {row["metadata"]["contract_id"] for row in rows}
        for row in rows:
            assert not row["document"]
            assert row["metadata"]["document_file"] == "documents.json"
            assert (
                documents[row["metadata"]["document_key"]]
                == "source " + row["metadata"]["contract_id"]
            )
    assert titles["test"] == {"test"}
    assert not titles["train"] & titles["validation"]


def test_widesearch_full_split_is_evaluation_only(tmp_path):
    source = tmp_path / "raw"
    gold = source / "widesearch_gold"
    gold.mkdir(parents=True)
    config = {
        "required": ["name"],
        "unique_columns": ["name"],
        "eval_pipeline": {"name": {"metric": ["exact_match"]}},
    }
    task = {
        "instance_id": "ws_en_001",
        "query": "List the names",
        "language": "en",
        "evaluation": json.dumps(config),
    }
    (source / "widesearch.jsonl").write_text(json.dumps(task) + "\n")
    (gold / "ws_en_001.csv").write_text("Name\nAlpha\n")
    output = tmp_path / "prepared"
    summary = prepare.prepare_widesearch(source, output)
    assert summary["tasks"] == {"train": 0, "validation": 0, "test": 1}
    assert (output / "train.jsonl").read_text() == ""
    row = json.loads((output / "test.jsonl").read_text())
    assert row["gold"] == [{"name": "Alpha"}]
    assert row["metadata"]["evaluation"] == config
    assert row["split"] == "test"


def test_cuad_validation_cannot_consume_all_training_contracts(tmp_path):
    (tmp_path / "CUADv1.json").write_text(
        json.dumps({"data": [contract("a"), contract("test")]})
    )
    (tmp_path / "test.json").write_text(json.dumps({"data": [contract("test")]}))
    with pytest.raises(ValueError, match="leave at least one"):
        prepare.prepare_cuad(tmp_path, tmp_path / "out", validation_contracts=1)
