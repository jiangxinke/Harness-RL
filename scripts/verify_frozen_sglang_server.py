#!/usr/bin/env python3
"""Wait for and exercise the immutable OpenAI-compatible endpoint."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request


def request(url: str, *, payload: dict[str, object] | None = None) -> object:
    body = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=body,
        headers={"content-type": "application/json"},
        method="GET" if body is None else "POST",
    )
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.load(response)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:18002/v1")
    parser.add_argument("--model", default="Qwen2.5-3B-Instruct")
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    args = parser.parse_args()
    base_url = args.base_url.rstrip("/")
    deadline = time.monotonic() + args.timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            models = request(f"{base_url}/models")
            break
        except (OSError, urllib.error.URLError, ValueError) as exc:
            last_error = exc
            time.sleep(5)
    else:
        raise SystemExit(f"endpoint did not become ready: {last_error}")

    assert isinstance(models, dict) and models.get("data"), models
    result = request(
        f"{base_url}/chat/completions",
        payload={
            "model": args.model,
            "messages": [{"role": "user", "content": "Reply with READY only."}],
            "temperature": 0,
            "max_tokens": 8,
        },
    )
    assert isinstance(result, dict) and result.get("choices"), result
    content = result["choices"][0]["message"]["content"]
    assert isinstance(content, str) and content.strip(), result
    print(json.dumps({"models": models, "response": content}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
