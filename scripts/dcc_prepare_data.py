#!/usr/bin/env python3
"""Download the public benchmarks and prepare leakage-free DCC task files.

CUAD retains the official 102-contract test split. Validation is selected from
the 408 training contracts. WideSearch's entire 200-task full split is test only.
No WideSearch gold is used to construct training examples.
"""

from __future__ import annotations

import argparse
import io
import json
import random
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen

CUAD_REVISION = "67faa0e6023b04fcaae6cc09497ab00e5d63a2a2"
WIDESEARCH_REVISION = "6531a7e5b497d44c8912407e0cb3dc95bd98cc09"
WIDESEARCH_CODE_REVISION = "9825ba7b140b71d81b364793f86dabe4cfed6749"
CUAD_SOURCE = "https://github.com/The-Atticus-Project/cuad"
WIDESEARCH_SOURCE = "https://huggingface.co/datasets/ByteDance-Seed/WideSearch"


def _get(url: str) -> bytes:
    with urlopen(url, timeout=90) as response:
        return response.read()


def download_sources(directory: Path) -> None:
    cuad_dir, wide_dir = directory / "cuad", directory / "widesearch"
    cuad_dir.mkdir(parents=True, exist_ok=True)
    wide_dir.mkdir(parents=True, exist_ok=True)
    if not all((cuad_dir / name).exists() for name in ("CUADv1.json", "test.json")):
        archive = zipfile.ZipFile(
            io.BytesIO(
                _get(
                    f"https://raw.githubusercontent.com/The-Atticus-Project/cuad/{CUAD_REVISION}/data.zip"
                )
            )
        )
        for name in archive.namelist():
            if name.endswith(".json"):
                (cuad_dir / Path(name).name).write_bytes(archive.read(name))
    base = f"https://huggingface.co/datasets/ByteDance-Seed/WideSearch/resolve/{WIDESEARCH_REVISION}/"
    task_file = wide_dir / "widesearch.jsonl"
    if not task_file.exists():
        task_file.write_bytes(_get(base + "widesearch.jsonl"))
    tasks = [
        json.loads(line)
        for line in task_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    gold_dir = wide_dir / "widesearch_gold"
    gold_dir.mkdir(exist_ok=True)
    missing = [
        f"widesearch_gold/{task['instance_id']}.csv"
        for task in tasks
        if not (gold_dir / f"{task['instance_id']}.csv").exists()
    ]
    with ThreadPoolExecutor(max_workers=8) as executor:
        for name, content in zip(
            missing, executor.map(lambda name: _get(base + name), missing), strict=True
        ):
            (wide_dir / name).write_bytes(content)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def prepare_cuad(
    source: Path,
    output: Path,
    *,
    validation_contracts: int = 20,
    seed: int = 42,
    inline_documents: bool = False,
) -> dict:
    full = json.loads((source / "CUADv1.json").read_text(encoding="utf-8"))["data"]
    official_test = json.loads((source / "test.json").read_text(encoding="utf-8"))[
        "data"
    ]
    test_titles = {item["title"] for item in official_test}
    training = [item for item in full if item["title"] not in test_titles]
    if len({item["title"] for item in full}) != len(full):
        raise ValueError("Duplicate CUAD contract titles")
    if not 0 <= validation_contracts < len(training):
        raise ValueError(
            "validation_contracts must leave at least one training contract"
        )
    validation = set(
        random.Random(seed).sample(
            sorted(item["title"] for item in training), validation_contracts
        )
    )
    output.mkdir(parents=True, exist_ok=True)
    documents: dict[str, str] = {}
    splits: dict[str, list[dict]] = {"train": [], "validation": [], "test": []}
    contract_splits: dict[str, str] = {}
    # Reuse the official test's annotations directly rather than reconstructing it.
    contracts = sorted(training + official_test, key=lambda item: item["title"])
    for number, contract in enumerate(contracts):
        title = contract["title"]
        split = (
            "test"
            if title in test_titles
            else "validation"
            if title in validation
            else "train"
        )
        contract_splits[title] = split
        for paragraph_index, paragraph in enumerate(contract["paragraphs"]):
            context = paragraph["context"]
            document_key = f"contract_{number:04d}_{paragraph_index}"
            if not inline_documents:
                documents[document_key] = context
            for qa in paragraph["qas"]:
                metadata = {
                    "contract_id": title,
                    "category": qa["id"].split("__", 1)[-1],
                    "source": CUAD_SOURCE,
                    "source_revision": CUAD_REVISION,
                    "original_split": "test" if split == "test" else "train",
                    "is_impossible": qa.get("is_impossible", not qa["answers"]),
                }
                if not inline_documents:
                    metadata["document_file"] = "documents.json"
                    metadata["document_key"] = document_key
                splits[split].append(
                    {
                        "id": qa["id"],
                        "benchmark": "cuad",
                        "split": split,
                        "prompt": qa["question"],
                        "document": context if inline_documents else "",
                        "gold": list(
                            dict.fromkeys(answer["text"] for answer in qa["answers"])
                        ),
                        "metadata": metadata,
                    }
                )
    if not inline_documents:
        (output / "documents.json").write_text(
            json.dumps(documents, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    for split, records in splits.items():
        _write_jsonl(output / f"{split}.jsonl", records)
    return {
        "tasks": {split: len(rows) for split, rows in splits.items()},
        "contracts": {
            split: sum(value == split for value in contract_splits.values())
            for split in splits
        },
        "test_split": "official",
        "validation_seed": seed,
        "source": CUAD_SOURCE,
        "source_revision": CUAD_REVISION,
        "license": "CC-BY-4.0 (Atticus dataset card)",
    }


def prepare_widesearch(source: Path, output: Path) -> dict:
    import pandas as pd

    records = []
    tasks = [
        json.loads(line)
        for line in (source / "widesearch.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    for task in tasks:
        evaluation = task["evaluation"]
        if isinstance(evaluation, str):
            evaluation = json.loads(evaluation)
        gold = pd.read_csv(source / "widesearch_gold" / f"{task['instance_id']}.csv")
        gold.columns = [
            str(column).strip().lower().replace(" ", "") for column in gold.columns
        ]
        missing = set(evaluation["required"]) - set(gold.columns)
        if missing:
            raise ValueError(
                f"Missing gold columns for {task['instance_id']}: {sorted(missing)}"
            )
        # Preserve pandas numeric inference; its missing CSV cells compare as 'nan'
        # in the reference evaluator, but JSON artifacts must remain standards compliant.
        gold = (
            gold[evaluation["required"]]
            .astype(object)
            .where(gold[evaluation["required"]].notna(), "nan")
        )
        records.append(
            {
                "id": task["instance_id"],
                "benchmark": "widesearch",
                "split": "test",
                "prompt": task["query"],
                "document": "",
                "gold": gold.to_dict(orient="records"),
                "metadata": {
                    "evaluation": evaluation,
                    "answer_columns": evaluation["required"],
                    "language": task["language"],
                    "source": WIDESEARCH_SOURCE,
                    "source_revision": WIDESEARCH_REVISION,
                    "original_split": "full",
                },
            }
        )
    _write_jsonl(output / "test.jsonl", records)
    # Empty files make the absence of benchmark training/validation data explicit.
    _write_jsonl(output / "train.jsonl", [])
    _write_jsonl(output / "validation.jsonl", [])
    return {
        "tasks": {"train": 0, "validation": 0, "test": len(records)},
        "test_split": "entire official full split; never used for training",
        "source": WIDESEARCH_SOURCE,
        "source_revision": WIDESEARCH_REVISION,
        "evaluator_source": "https://github.com/ByteDance-Seed/WideSearch",
        "evaluator_revision": WIDESEARCH_CODE_REVISION,
        "license": "Dataset card: other; evaluation code: MIT",
        "default_official_judge": "gpt-4.1-2025-04-14",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--source-dir",
        type=Path,
        help="Existing raw download directory (cuad/ and widesearch/)",
    )
    parser.add_argument("--validation-contracts", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--inline-documents",
        action="store_true",
        help="Repeat full CUAD text in each QA instead of document references",
    )
    parser.add_argument(
        "--download", action="store_true", help="Download missing raw source files"
    )
    args = parser.parse_args()
    source = args.source_dir or args.output_dir / "raw"
    if args.download or args.source_dir is None:
        download_sources(source)
    summary = {
        "cuad": prepare_cuad(
            source / "cuad",
            args.output_dir / "cuad",
            validation_contracts=args.validation_contracts,
            seed=args.seed,
            inline_documents=args.inline_documents,
        ),
        "widesearch": prepare_widesearch(
            source / "widesearch", args.output_dir / "widesearch"
        ),
        "training_policy": "Only CUAD official training contracts; WideSearch is evaluation only. Add independent web curricula separately.",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
