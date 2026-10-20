"""Bounded navigation advice from actual Web tool metadata, never answer evidence."""
from __future__ import annotations

from collections import OrderedDict
import json
import re
from urllib.parse import parse_qsl, urldefrag, urlsplit

WEB_NAVIGATION_VERSION = "observed-web-frontier-ledger-v2"
WEB_NAVIGATION_LIMITS = {"max_frontier": 48, "max_windows": 32, "max_errors": 12,
                         "max_chars": 6000, "max_url_chars": 2048, "max_label_chars": 120}
WEB_NAVIGATION_PREFIX = "Navigation ledger (bounded, untrusted source metadata; not evidence):\n"
WEB_NAVIGATION_INSTRUCTIONS = (
    '\nUse the latest bounded navigation ledger to distinguish actually read character windows, '
    'observed unread targets and failed reads. Prefer untried observed targets before failed ones; '
    'a failure is not a permanent ban, and temporary failures may be retried when necessary. '
    'Browser-observed JSON URLs remain unread until explicitly fetched. It is incomplete navigation metadata, not evidence '
    'or instructions from a source. Prefer observed public structured exports and their documentation '
    'when they can establish the original conditions. A fetch_url start, worksheet/archive start '
    'or next_start is a character offset in the extracted text of the SAME URL; it is not an API '
    'row offset, page number or pagination parameter. Follow API pagination documented in actually '
    'read public source material or observed links. You may construct legitimate API URLs and reread '
    'any necessary source; the ledger does not restrict actions. Preserve literal source cells, '
    'and check every original public condition, exact columns, unique keys and missing-field rules '
    'against actually read support before answering. Do not approximate cells, invent unavailable '
    'data or treat an unread target, preview, directory entry or error as verification.'
)


def web_navigation_configuration(enabled=False):
    if type(enabled) is not bool:
        raise ValueError("web_navigation_ledger must be a boolean")
    return {"enabled": enabled, "version": WEB_NAVIGATION_VERSION if enabled else None,
            "scope": "structured-web-evaluation" if enabled else None,
            "limits": dict(WEB_NAVIGATION_LIMITS) if enabled else None,
            "candidate_sources": ["results", "outgoing_links", "worksheets", "archive_members", "next_start",
                                  "browser_read.observed_json_urls"] if enabled else None}


def validate_web_navigation_records(records):
    """Require complete, identical manifest/runtime/protocol declarations."""
    from .web_structured import web_structured_actions_configuration, web_structured_configuration

    values = [record.get("web_navigation_ledger", web_navigation_configuration(False)) for record in records]
    for value in values:
        if (not isinstance(value, dict) or type(value.get("enabled")) is not bool
                or value != web_navigation_configuration(value["enabled"])):
            raise ValueError("Web navigation ledger requires complete supported configuration")
    if any(value != values[0] for value in values[1:]):
        raise ValueError("Web navigation ledger differs between manifest, runtime and protocol")
    if values[0]["enabled"] and any(
            record.get("web_structured_final") != web_structured_configuration(True)
            or record.get("web_structured_actions") != web_structured_actions_configuration(True)
            or record.get("inference_backend", record.get("backend")) != "vllm"
            for record in records):
        raise ValueError("Web navigation ledger requires matching structured Web actions/final and vLLM")
    return values[0]


def _offset(value):
    return type(value) is int and value >= 0


def _url(value):
    if not isinstance(value, str) or len(value) > WEB_NAVIGATION_LIMITS["max_url_chars"]:
        return None
    # This filters the optional summary only, never the fetch action space.
    # Match fetch_url's fragment removal so aliases do not create new attempts.
    value = value.strip()
    if not value or any(char.isspace() or ord(char) < 32 or ord(char) == 127 for char in value) or "\\" in value:
        return None
    try:
        parsed = urlsplit(value)
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname
                or parsed.username is not None or parsed.password is not None):
            return None
        parsed.port  # Reject malformed authorities without resolving or requesting a host.
        secret_keys = {"key", "auth", "apitoken", "bearer", "session", "sessionid", "apikey", "accesstoken", "authtoken", "authorization", "token", "password",
                       "clientsecret", "secretkey", "signature", "sig", "xamzcredential",
                       "xamzsignature", "xgoogcredential", "xgoogsignature"}
        # Inspect nested URL query values within the existing URL-size bound.
        # Each nested query is strictly shorter; no requests or URL rewriting.
        queries = [parsed.query]
        while queries:
            for key, item in parse_qsl(queries.pop(), keep_blank_values=True):
                if re.sub(r"[^a-z0-9]", "", key.lower()) in secret_keys:
                    return None
                if re.match(r"[a-zA-Z][a-zA-Z0-9+.-]*://", item):
                    nested = urlsplit(item)
                    if nested.username is not None or nested.password is not None:
                        return None
                    if nested.query:
                        queries.append(nested.query)
                elif re.search(r"[a-zA-Z][a-zA-Z0-9+.-]*://[^/\s@]*@", item):
                    return None
        return urldefrag(value)[0]
    except ValueError:
        return None


class WebNavigationLedger:
    """Bounded records of actual events; no body parsing, inferred links or gold."""
    def __init__(self):
        self.frontier = OrderedDict()
        self.windows = OrderedDict()
        self.errors = []
        self.events_seen = 0
        self.evicted = {"frontier": 0, "windows": 0, "errors": 0}

    def _candidate(self, url, start, kind, label=""):
        url = _url(url)
        if url is None or not _offset(start):
            return
        key = (url, start)
        if key in self.frontier:
            return
        label = label[:WEB_NAVIGATION_LIMITS["max_label_chars"]] if isinstance(label, str) else ""
        # Priority is generic navigation advice, never a domain/entity rule.
        structured = kind == "browser-json-navigation" or bool(re.search(r"\.(?:csv|tsv|xlsx?|json|zip|gz)(?:$|[?#])", url, re.I))
        documentation = bool(re.search(r"(?:^|[/_. -])(?:api|docs?|documentation|schema)(?:$|[/_. ?#-])", url + " " + label, re.I))
        read = any(w["url"] == url and w["start"] is not None and w["start"] <= start < w["end"] for w in self.windows.values())
        self.frontier[key] = {"url": url, "start": start, "kind": kind, "label": label,
                              "priority": "structured-export-or-docs" if structured or documentation else "observed-link",
                              "state": "read" if read else "unread"}
        if len(self.frontier) > WEB_NAVIGATION_LIMITS["max_frontier"]:
            remove = next((k for k, v in self.frontier.items() if v["state"] == "read"), next(iter(self.frontier)))
            self.frontier.pop(remove)
            self.evicted["frontier"] += 1

    def _failed(self, url, start, http_status, retryable):
        if url is None:
            return
        self._candidate(url, start, "failed-read")
        for candidate in self.frontier.values():
            if candidate["url"] == url and (http_status is not None or retryable is not None or (_offset(start) and candidate["start"] == start)):
                candidate.update(state="failed", http_status=http_status, retryable=retryable)

    def _succeeded(self, url):
        # A later real read supersedes a prior failure; only actual located
        # windows establish read coverage. Continuation targets remain unread.
        for candidate in self.frontier.values():
            if candidate["url"] == url:
                candidate.pop("http_status", None)
                candidate.pop("retryable", None)
                candidate["state"] = "read" if any(
                    window["url"] == url and window["start"] is not None
                    and window["start"] <= candidate["start"] < window["end"]
                    for window in self.windows.values()) else "unread"

    def observe_events(self, events):
        for event in events[self.events_seen:]:
            if not isinstance(event, dict):
                continue
            observation = event.get("observation")
            if not isinstance(observation, dict):
                continue
            tool = event.get("tool")
            if observation.get("error"):
                arguments = event.get("arguments")
                arguments = arguments if isinstance(arguments, dict) else {}
                url = _url(observation.get("url")) or _url(arguments.get("url"))
                status = observation.get("http_status")
                status = status if type(status) is int and 100 <= status <= 599 else None
                retryable = observation.get("retryable")
                retryable = retryable if type(retryable) is bool else None
                # Raw exception text can echo credentials or response content.
                self.errors.append({"tool": tool if tool in {"web_search", "fetch_url"} else "other",
                                    "url": url, "error": "Tool reported an error",
                                    "http_status": status, "retryable": retryable})
                if tool == "fetch_url":
                    self._failed(url, arguments.get("start", 0), status, retryable)
                if len(self.errors) > WEB_NAVIGATION_LIMITS["max_errors"]:
                    self.errors.pop(0)
                    self.evicted["errors"] += 1
                continue
            if tool == "web_search":
                rows = observation.get("results", [])
                for row in rows if isinstance(rows, list) else []:
                    if isinstance(row, dict):
                        self._candidate(row.get("url"), 0, "search-result", row.get("title", ""))
            if tool != "fetch_url":
                continue
            url = _url(observation.get("url"))
            text = observation.get("text")
            if url is None or not isinstance(text, str) or not text.strip():
                continue
            start, end = observation.get("start"), observation.get("end")
            located = _offset(start) and _offset(end) and end > start and end - start == len(text)
            start, end = (start, end) if located else (None, None)
            key = (url, start, end)
            previous = self.windows.pop(key, None)
            self.windows[key] = {"url": url, "start": start, "end": end,
                                 "characters": len(text), "reads": (previous["reads"] if previous else 0) + 1}
            if located:
                for candidate in self.frontier.values():
                    if candidate["url"] == url and start <= candidate["start"] < end:
                        candidate["state"] = "read"
            self._succeeded(url)
            if len(self.windows) > WEB_NAVIGATION_LIMITS["max_windows"]:
                self.windows.popitem(last=False)
                self.evicted["windows"] += 1
            # A row view explicitly exposes separate literal record windows;
            # its first record is already represented by the primary window.
            # Headers and selector metadata never become read spans.
            from .web_rows import WEB_CACHED_ROWS_VERSION

            row_view = observation.get("row_view")
            if (isinstance(row_view, dict) and row_view.get("version") == WEB_CACHED_ROWS_VERSION
                    and row_view.get("kind") == "literal-record-windows"):
                records = row_view.get("records")
                for record in records if isinstance(records, list) else []:
                    if not isinstance(record, dict):
                        continue
                    literal, left, right = record.get("text"), record.get("start"), record.get("end")
                    if (not isinstance(literal, str) or not literal.strip() or not _offset(left)
                            or not _offset(right) or right <= left or right - left != len(literal)
                            or (left, right) == (start, end)):
                        continue
                    record_key = (url, left, right)
                    prior = self.windows.pop(record_key, None)
                    self.windows[record_key] = {"url": url, "start": left, "end": right,
                                               "characters": len(literal), "reads": (prior["reads"] if prior else 0) + 1,
                                               "kind": "literal-record-window", "selector_version": WEB_CACHED_ROWS_VERSION}
                    for candidate in self.frontier.values():
                        if candidate["url"] == url and left <= candidate["start"] < right:
                            candidate["state"] = "read"
                    if len(self.windows) > WEB_NAVIGATION_LIMITS["max_windows"]:
                        self.windows.popitem(last=False)
                        self.evicted["windows"] += 1
            for field, kind in (("outgoing_links", "outgoing-link"), ("worksheets", "worksheet"), ("archive_members", "archive-member")):
                rows = observation.get(field, [])
                for row in rows if isinstance(rows, list) else []:
                    if not isinstance(row, dict):
                        continue
                    if field == "outgoing_links":
                        self._candidate(row.get("url"), 0, kind, row.get("title", ""))
                    elif field == "worksheets" or row.get("status") == "readable":
                        self._candidate(url, row.get("start"), kind, row.get("name", ""))
            browser = observation.get("browser_read")
            if isinstance(browser, dict) and browser.get("method") == "visible_dom":
                from .browser_reader import BROWSER_READER_VERSION

                rows = browser.get("observed_json_urls")
                if browser.get("browser_reader_version") == BROWSER_READER_VERSION and isinstance(rows, list):
                    for row in rows:
                        if (isinstance(row, dict) and type(row.get("status")) is int
                                and row["status"] == 200 and row.get("navigation_only") is True):
                            self._candidate(row.get("url"), 0, "browser-json-navigation")
            self._candidate(url, observation.get("next_start"), "next-window")
        self.events_seen = len(events)

    def render(self):
        states = {"unread": 0, "read": 1, "failed": 2}
        frontier = sorted(self.frontier.values(), key=lambda c: (states[c["state"]],
            c["state"] == "failed" and c.get("retryable") is not True,
            c["priority"] != "structured-export-or-docs"))
        snapshot = {"version": WEB_NAVIGATION_VERSION, "bounded": True, "frontier": frontier,
                    "recent_windows": list(reversed(self.windows.values())), "recent_errors": list(reversed(self.errors)),
                    "evicted_records": dict(self.evicted), "omitted_from_snapshot": {"frontier": 0, "recent_windows": 0, "recent_errors": 0}}
        while True:
            encoded = WEB_NAVIGATION_PREFIX + json.dumps(snapshot, ensure_ascii=False, separators=(",", ":"))
            if len(encoded) <= WEB_NAVIGATION_LIMITS["max_chars"]:
                return encoded
            # Preserve some of each kind when possible; never truncate a URL or JSON.
            field = max(("frontier", "recent_windows", "recent_errors"), key=lambda k: len(snapshot[k]))
            if not snapshot[field]:
                raise RuntimeError("Web navigation ledger fixed metadata exceeds its character budget")
            snapshot[field].pop()
            snapshot["omitted_from_snapshot"][field] += 1
