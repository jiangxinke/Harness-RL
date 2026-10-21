"""Optional one-hop documentation discovery from actually read Web pages."""
from __future__ import annotations

import re
from urllib.parse import urldefrag, urlsplit

AUTO_READ_FOLLOW_DOCS_VERSION = "observed-same-origin-documentation-hop-v1"
_DOCUMENTATION = re.compile(r"(?:^|[\W_])(?:help|api|docs?|documentation)(?:$|[\W_])", re.I)


def auto_read_follow_docs_configuration(enabled=False):
    if type(enabled) is not bool:
        raise ValueError("auto_read_follow_docs must be a boolean")
    return {"enabled": enabled,
            "version": AUTO_READ_FOLLOW_DOCS_VERSION if enabled else None,
            "max_followups_per_search": 1 if enabled else None,
            "candidate_source": "successful-read.outgoing_links" if enabled else None,
            "same_origin_only": True if enabled else None,
            "additional_attempts": 0, "additional_turns": 0}


def _origin(url):
    if (not isinstance(url, str) or len(url) > 2048
            or any(ord(char) <= 32 or ord(char) == 127 for char in url)
            or "\\" in url):
        return None
    try:
        parsed = urlsplit(url)
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname
                or parsed.username is not None or parsed.password is not None):
            return None
        port = parsed.port
        return parsed.scheme, parsed.hostname.lower(), port if port is not None else (443 if parsed.scheme == "https" else 80)
    except ValueError:
        return None


def observed_documentation_link(page, *, excluded):
    """Return at most one literal observed link, without fetching or inventing URLs.

    The normal WebTools.fetch_url path retains all public-host and redirect
    checks. Same-origin filtering here only limits automatic navigation.
    """
    if not isinstance(page, dict) or page.get("error"):
        return None
    origin = _origin(page.get("url"))
    links = page.get("outgoing_links")
    if origin is None or not isinstance(links, list):
        return None
    source = urldefrag(page["url"])[0]
    for link in links:
        if not isinstance(link, dict):
            continue
        url, title = link.get("url"), link.get("title", "")
        if _origin(url) != origin or not isinstance(title, str):
            continue
        url = urldefrag(url)[0]
        if url == source or url in excluded or not _DOCUMENTATION.search(url + " " + title):
            continue
        return url
    return None
