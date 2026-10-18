from __future__ import annotations

import os
import threading
from typing import Any

import httpx


def _search_concurrency_limit() -> int:
    raw = os.environ.get("REFLECT_SEARCH_CONCURRENCY", "8")
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"REFLECT_SEARCH_CONCURRENCY must be a positive integer, got {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"REFLECT_SEARCH_CONCURRENCY must be a positive integer, got {raw!r}")
    return value


# Every Reflect tree runs in its own worker thread, while the configured dense
# retriever is a single GPU-backed FastAPI process. Bound concurrent requests
# across all HttpSearch instances in this process so a large rollout batch does
# not enqueue dozens of simultaneous forwards against one model.
_SEARCH_SEMAPHORE = threading.BoundedSemaphore(_search_concurrency_limit())


class HttpSearch:
    """Search adapter for a JSON POST endpoint."""

    def __init__(
        self,
        url: str = "http://localhost:8002/retrieve",
        *,
        top_k: int = 3,
        timeout: float = 60.0,
    ) -> None:
        self.url = url
        self.top_k = top_k
        # trust_env=False is the httpx equivalent of no_proxy="*".
        self.client = httpx.Client(
            timeout=timeout,
            trust_env=False,
            limits=httpx.Limits(max_connections=1, max_keepalive_connections=0),
            headers={"Connection": "close"},
        )

    def search(self, question: str) -> str:
        with _SEARCH_SEMAPHORE:
            with self.client.stream(
                "POST",
                self.url,
                json={"queries": [question], "topk": self.top_k},
            ) as response:
                response.raise_for_status()
                payload = response.read()
        return self.format_results(self._first_query_results(httpx.Response(200, content=payload).json()))

    def close(self) -> None:
        self.client.close()

    @staticmethod
    def _first_query_results(payload: Any) -> list[Any]:
        if not isinstance(payload, dict) or not isinstance(payload.get("result"), list):
            raise ValueError("search response must contain a result list")
        result = payload["result"]
        if not result or not isinstance(result[0], list):
            raise ValueError("search response contains no result set for the query")
        return result[0]

    @staticmethod
    def format_results(results: list[Any]) -> str:
        documents: list[str] = []
        for rank, item in enumerate(results, 1):
            if isinstance(item, dict):
                document = item.get("document", item)
                if not isinstance(document, dict):
                    continue
                document_id = document.get("id", rank)
                contents = document.get("contents") or document.get("content") or document.get("text")
            else:
                document_id = rank
                contents = str(item)
            if contents:
                documents.append(f"[{rank}] id={document_id}\n{contents}")
        return "\n\n".join(documents) or "No search results found."
