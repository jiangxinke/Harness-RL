"""Tests for the safety-bounded direct web fetch provider."""

from __future__ import annotations

import httpx
import pytest

from deerflow.community.direct_fetch import tools


@pytest.mark.asyncio
async def test_fetch_public_url_returns_pretty_json(monkeypatch):
    monkeypatch.setattr(
        tools,
        "validate_public_http_url",
        lambda *_args, **_kwargs: None,
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://api.example.test/data"
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            json={"value": 12.5, "country": "Brazil"},
        )

    result = await tools.fetch_public_url(
        "https://api.example.test/data",
        transport=httpx.MockTransport(handler),
    )

    assert '"value": 12.5' in result
    assert '"country": "Brazil"' in result


@pytest.mark.asyncio
async def test_fetch_public_url_rejects_private_destination():
    result = await tools.fetch_public_url(
        "http://127.0.0.1/private",
        transport=httpx.MockTransport(lambda _request: httpx.Response(200, text="must not run")),
    )

    assert result == ("Error: Refusing to fetch a private, loopback, or metadata address")


@pytest.mark.asyncio
async def test_fetch_public_url_revalidates_redirect_target(monkeypatch):
    def validate(url: str, **_kwargs):
        if "127.0.0.1" in url:
            return "Error: blocked redirect target"
        return None

    monkeypatch.setattr(tools, "validate_public_http_url", validate)

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "public.example.test":
            return httpx.Response(
                302,
                headers={"location": "http://127.0.0.1/private"},
            )
        return httpx.Response(200, text="must not run")

    result = await tools.fetch_public_url(
        "https://public.example.test/start",
        transport=httpx.MockTransport(handler),
    )

    assert result == "Error: blocked redirect target"


@pytest.mark.asyncio
async def test_fetch_public_url_enforces_response_byte_limit(monkeypatch):
    monkeypatch.setattr(
        tools,
        "validate_public_http_url",
        lambda *_args, **_kwargs: None,
    )

    result = await tools.fetch_public_url(
        "https://public.example.test/large",
        max_bytes=4,
        transport=httpx.MockTransport(lambda _request: httpx.Response(200, content=b"12345")),
    )

    assert result == "Error: Response exceeded the configured 4-byte limit"
