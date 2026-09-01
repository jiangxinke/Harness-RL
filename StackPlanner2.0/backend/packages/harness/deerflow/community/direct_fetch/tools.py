"""Safety-bounded direct HTTP fetch tool for public pages and APIs."""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import urljoin

import httpx
from langchain.tools import tool

from deerflow.community.url_safety import validate_public_http_url
from deerflow.config import get_app_config

DEFAULT_TIMEOUT_SECONDS = 20.0
DEFAULT_MAX_BYTES = 1_048_576
DEFAULT_MAX_CHARS = 20_000
MAX_REDIRECTS = 5


def _coerce_positive_int(value: object, default: int) -> int:
    if isinstance(value, bool):
        return default
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _coerce_positive_float(value: object, default: float) -> float:
    if isinstance(value, bool):
        return default
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _coerce_bool(value: object, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return default


def _tool_options() -> dict[str, Any]:
    config = get_app_config().get_tool_config("web_fetch")
    extra = dict(config.model_extra or {}) if config is not None else {}
    return {
        "timeout": _coerce_positive_float(
            extra.get("timeout"),
            DEFAULT_TIMEOUT_SECONDS,
        ),
        "max_bytes": _coerce_positive_int(
            extra.get("max_bytes"),
            DEFAULT_MAX_BYTES,
        ),
        "max_chars": _coerce_positive_int(
            extra.get("max_chars"),
            DEFAULT_MAX_CHARS,
        ),
        "allow_private_addresses": _coerce_bool(
            extra.get("allow_private_addresses"),
            False,
        ),
        # Direct fetch deliberately ignores ambient operator proxies by
        # default. A deployment may explicitly opt back in.
        "trust_env": _coerce_bool(extra.get("trust_env"), False),
    }


def _decode_body(response: httpx.Response, body: bytes, *, max_chars: int) -> str:
    encoding = response.encoding or "utf-8"
    text = body.decode(encoding, errors="replace")
    content_type = response.headers.get("content-type", "").lower()
    if "json" in content_type:
        try:
            text = json.dumps(
                json.loads(text),
                ensure_ascii=False,
                indent=2,
            )
        except json.JSONDecodeError:
            pass
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}\n...<truncated>"


async def fetch_public_url(
    url: str,
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    max_bytes: int = DEFAULT_MAX_BYTES,
    max_chars: int = DEFAULT_MAX_CHARS,
    allow_private_addresses: bool = False,
    trust_env: bool = False,
    transport: httpx.AsyncBaseTransport | None = None,
) -> str:
    """Fetch one public URL while bounding redirects and response size."""

    current_url = url
    async with httpx.AsyncClient(
        timeout=timeout,
        follow_redirects=False,
        trust_env=trust_env,
        transport=transport,
        headers={
            "Accept": "application/json,text/plain,text/html;q=0.9,*/*;q=0.5",
            "User-Agent": "StackPlanner-WebFetch/1.0",
        },
    ) as client:
        for redirect_count in range(MAX_REDIRECTS + 1):
            url_error = validate_public_http_url(
                current_url,
                allow_private_addresses=allow_private_addresses,
                action="fetch",
            )
            if url_error:
                return url_error

            try:
                async with client.stream("GET", current_url) as response:
                    if response.is_redirect:
                        if redirect_count >= MAX_REDIRECTS:
                            return f"Error: Too many redirects (>{MAX_REDIRECTS})"
                        location = response.headers.get("location")
                        if not location:
                            return "Error: Redirect response did not include a Location header"
                        current_url = urljoin(str(response.url), location)
                        continue

                    response.raise_for_status()
                    body = bytearray()
                    async for chunk in response.aiter_bytes():
                        body.extend(chunk)
                        if len(body) > max_bytes:
                            return f"Error: Response exceeded the configured {max_bytes}-byte limit"
                    return _decode_body(
                        response,
                        bytes(body),
                        max_chars=max_chars,
                    )
            except httpx.HTTPStatusError as exc:
                return f"Error: HTTP {exc.response.status_code} while fetching {current_url}"
            except httpx.TimeoutException:
                return f"Error: Fetch timed out after {timeout:g}s"
            except httpx.RequestError as exc:
                return f"Error: Fetch failed: {type(exc).__name__}: {exc}"

    return f"Error: Too many redirects (>{MAX_REDIRECTS})"


@tool("web_fetch", parse_docstring=True)
async def web_fetch_tool(url: str) -> str:
    """Fetch an exact public HTTP(S) page or API URL.

    Use an exact URL supplied by the user, found in prior search/fetch results,
    or constructed from an explicitly named official public API and its
    supplied parameters. For multi-entity APIs, prefer bounded single-entity
    requests when the batch delimiter is not verified. This tool performs only
    GET requests, blocks private/loopback destinations by default, limits
    redirects and response size, and cannot access authenticated content.

    Args:
        url: Exact public http:// or https:// URL to fetch.
    """

    return await fetch_public_url(url, **_tool_options())
