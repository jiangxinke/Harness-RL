from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Chunk:
    source: str
    text: str


class LocalRetriever:
    """Tiny dependency-free lexical retriever for local text-like files."""

    SUFFIXES = {".txt", ".md", ".json", ".jsonl", ".csv"}

    def __init__(self, chunks: list[Chunk], *, top_k: int = 3) -> None:
        if not chunks:
            raise ValueError("the local corpus contains no readable text")
        self.chunks = chunks
        self.top_k = top_k

    @classmethod
    def from_path(cls, path: str | Path, *, chunk_chars: int = 1_500, top_k: int = 3) -> LocalRetriever:
        path = Path(path)
        files = [path] if path.is_file() else sorted(item for item in path.rglob("*") if item.is_file() and item.suffix.lower() in cls.SUFFIXES)
        chunks: list[Chunk] = []
        for file in files:
            text = file.read_text(encoding="utf-8", errors="replace")
            for start in range(0, len(text), chunk_chars):
                body = text[start : start + chunk_chars].strip()
                if body:
                    chunks.append(Chunk(f"{file}:{start}", body))
        return cls(chunks, top_k=top_k)

    def search(self, question: str) -> str:
        query_terms = _terms(question)
        ranked = sorted(
            ((len(query_terms & _terms(chunk.text)), -index, chunk) for index, chunk in enumerate(self.chunks)),
            reverse=True,
        )
        matches = [chunk for score, _, chunk in ranked[: self.top_k] if score > 0]
        if not matches:
            return "No relevant local evidence found."
        return "\n\n".join(f"[{chunk.source}]\n{chunk.text}" for chunk in matches)


def _terms(text: str) -> set[str]:
    lowered = text.lower()
    words = set(re.findall(r"[a-z0-9_]+", lowered))
    han = re.findall(r"[\u3400-\u9fff]", lowered)
    return words | {"".join(han[index : index + 2]) for index in range(len(han) - 1)}
