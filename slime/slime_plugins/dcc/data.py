"""Validated benchmark tasks; labels are deliberately separate from policy input."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any


@dataclasses.dataclass(frozen=True, slots=True)
class Task:
    id: str
    benchmark: str
    split: str
    prompt: str
    document: str = ""
    gold: Any = None
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.benchmark not in {"cuad", "widesearch"}:
            raise ValueError(f"unsupported benchmark: {self.benchmark}")
        if self.split not in {"train", "validation", "test"}:
            raise ValueError(f"unsupported split: {self.split}")
        if not self.id or not self.prompt.strip():
            raise ValueError("task requires a nonempty id and prompt")
        if self.benchmark == "cuad" and not self.document:
            raise ValueError("CUAD task requires the source contract")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Task:
        return cls(**{field.name: value[field.name] for field in dataclasses.fields(cls) if field.name in value})

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def load_tasks(path: str | Path, *, split: str | None = None) -> list[Task]:
    tasks: list[Task] = []
    seen: set[str] = set()
    documents: dict[Path, str | dict[str, str]] = {}
    path = Path(path)
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                reference = value.get("metadata", {}).get("document_file")
                if not value.get("document") and reference:
                    document_path = (path.parent / reference).resolve()
                    if document_path not in documents:
                        content = document_path.read_text(encoding="utf-8")
                        documents[document_path] = json.loads(content) if value.get("metadata", {}).get("document_key") else content
                    document = documents[document_path]
                    document_key = value.get("metadata", {}).get("document_key")
                    value["document"] = document[document_key] if document_key else document
                    if not isinstance(value["document"], str):
                        raise ValueError("document reference must resolve to text")
                task = Task.from_dict(value)
            except (TypeError, ValueError, KeyError) as exc:
                raise ValueError(f"invalid task at {path}:{number}: {exc}") from exc
            if task.id in seen:
                raise ValueError(f"duplicate task id: {task.id}")
            seen.add(task.id)
            if split is None or task.split == split:
                tasks.append(task)
    return tasks


def validate_disjoint(*collections: Iterable[Task]) -> None:
    """Reject ID leakage and CUAD contracts appearing in different splits."""

    ids: dict[tuple[str, str], str] = {}
    documents: dict[str, str] = {}
    for collection in collections:
        for task in collection:
            key = (task.benchmark, task.id)
            if key in ids and ids[key] != task.split:
                raise ValueError(f"task {task.id} appears in {ids[key]} and {task.split}")
            ids[key] = task.split
            if task.benchmark == "cuad":
                # Exact source text also catches renamed contract IDs without
                # hashing large files or relying on an external manifest.
                key_text = task.document
                if key_text in documents and documents[key_text] != task.split:
                    raise ValueError(f"CUAD contract leaks across splits: {task.id}")
                documents[key_text] = task.split
