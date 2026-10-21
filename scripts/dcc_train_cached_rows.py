"""Explicit independent Web TRAIN v3 cache receipts and offline literal replay.

Cache payloads are published once per episode/URL; requests contain ordered IDs.
No network, tokenizer, labels, or target-derived schema is used here.
Source digests are computed once at publication and once at sidecar loading.
The ordinary WebTools cache is immutable: replacement/eviction fails closed.
"""
from __future__ import annotations
import copy
import json
import hashlib
from pathlib import Path
import urllib.parse

VERSION = "dcc-web-train-cache-receipt-v3"
SOURCE_VERSION = "dcc-web-train-cache-source-v3"
MAX_PAGE_CHARS = 14000
MAX_SOURCE_CHARS = 5_000_000


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def configuration():
    from slime_plugins.dcc.web_rows import WEB_CACHED_ROWS_VERSION, web_cached_rows_configuration
    require(WEB_CACHED_ROWS_VERSION == "cached-literal-record-windows-v3", "TRAIN cache requires actual v3 implementation")
    return web_cached_rows_configuration(True)


def equal(a, b):
    # Source strings normally retain object identity across cache snapshots.
    # No per-request reserialization/hash of the complete downloaded text.
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal(a[key], b[key]) for key in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def source_digest(source):
    # Once when the source is published, once when its sidecar is loaded.
    payload = json.dumps(source, sort_keys=True, ensure_ascii=False, allow_nan=False,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def canonical(url):
    require(isinstance(url, str), "Cache URL must be a string")
    url = urllib.parse.urldefrag(url.strip())[0]
    require(url.startswith(("https://", "http://")), "Cache URL must be HTTP(S)")
    return url


def source_id(episode_id, index):
    return episode_id + ":cached-source:" + str(index)


class CachePublisher:
    def __init__(self, journal, *, task_id, episode_id, source_receipt, profile_version):
        self.journal, self.task_id, self.episode_id = journal, task_id, episode_id
        self.source_receipt, self.profile_version = copy.deepcopy(source_receipt), profile_version
        self.tools = None
        self.sources = {}

    def attach(self, tools):
        from slime_plugins.dcc.tools import WebTools
        require(self.tools is None and isinstance(tools, WebTools), "TRAIN cached rows require original episode-local WebTools")
        require(not tools.pages, "TRAIN downloaded cache must start empty")
        require(type(tools.max_page_chars) is int and tools.max_page_chars == MAX_PAGE_CHARS,
                "TRAIN cached rows retain the original 14000-character observation budget")
        self.tools = tools
        return tools

    def snapshot(self, turn):
        require(self.tools is not None, "Missing actual episode cache")
        require(set(self.sources) <= set(self.tools.pages), "TRAIN downloaded cache was cleared/evicted")
        for url, text in self.tools.pages.items():
            require(canonical(url) == url and isinstance(text, str) and text.strip()
                    and len(text) <= MAX_SOURCE_CHARS, "Invalid or oversized actual downloaded cache source")
            value = {"url": url, "text": text}
            for attribute, key in (("page_worksheets", "worksheets"), ("page_archives", "archives"),
                                   ("page_links", "links"), ("page_browser", "browser")):
                mapping = getattr(self.tools, attribute)
                if url in mapping:
                    value[key] = copy.deepcopy(mapping[url])
            if url in self.sources:
                require(equal(value, self.sources[url]["source"]), "TRAIN downloaded cache source was overwritten")
            else:
                require(type(turn) is int and 0 < turn < 20, "Cache source cannot precede a public fetch")
                index = len(self.sources)
                record = {"version": SOURCE_VERSION, "profile_version": self.profile_version,
                          "configuration": configuration(), "source_receipt": copy.deepcopy(self.source_receipt),
                          "task_id": self.task_id, "episode_id": self.episode_id,
                          "source_id": source_id(self.episode_id, index), "source_index": index,
                          "available_turn": turn, "source": value, "source_sha256": source_digest(value)}
                self.journal.write(record)
                self.sources[url] = record
        return {"version": VERSION, "configuration": configuration(), "max_page_chars": MAX_PAGE_CHARS,
                "source_ids": [record["source_id"] for record in self.sources.values()],
                "source_sha256": [record["source_sha256"] for record in self.sources.values()]}


def load_sources(path, *, source_receipt, profile_version):
    from dcc_native_sft import _strict_json
    result, episodes = {}, {}
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            require(line.strip() and line.endswith("\n"), "Incomplete cache source journal")
            row = _strict_json(line)
            keys = {"version", "profile_version", "configuration", "source_receipt", "task_id", "episode_id",
                    "source_id", "source_index", "available_turn", "source", "source_sha256"}
            require(isinstance(row, dict) and set(row) == keys and row["version"] == SOURCE_VERSION
                    and row["profile_version"] == profile_version and equal(row["configuration"], configuration())
                    and equal(row["source_receipt"], source_receipt), "Cache source identity differs from trusted source/profile/configuration")
            require(all(isinstance(row[key], str) and row[key].strip() for key in ("task_id", "episode_id", "source_id")),
                    "Cache source episode/task ownership required")
            previous = episodes.setdefault(row["episode_id"], [])
            require(type(row["source_index"]) is int and row["source_index"] == len(previous)
                    and row["source_id"] == source_id(row["episode_id"], len(previous))
                    and row["source_id"] not in result, "Cache source duplicate or sequence differs")
            require(type(row["available_turn"]) is int and 0 < row["available_turn"] < 20
                    and (not previous or row["available_turn"] > previous[-1]["available_turn"])
                    and (not previous or row["task_id"] == previous[-1]["task_id"]), "Cache source publication order/owner differs")
            source = row["source"]
            require(isinstance(source, dict) and {"url", "text"} <= set(source)
                    and set(source) <= {"url", "text", "worksheets", "archives", "links", "browser"}
                    and canonical(source["url"]) == source["url"] and isinstance(source["text"], str)
                    and bool(source["text"].strip()) and len(source["text"]) <= MAX_SOURCE_CHARS
                    and all(source["url"] != older["source"]["url"] for older in previous),
                    "Cache source URL/content is invalid, duplicated or overwritten")
            require(all(key not in source or isinstance(source[key], list) for key in ("worksheets", "archives", "links"))
                    and ("browser" not in source or isinstance(source["browser"], dict)), "Invalid cache source navigation metadata")
            require(type(row["source_sha256"]) is str and row["source_sha256"] == source_digest(source),
                    "Cache source content/metadata digest differs from its original publication")
            previous.append(row); result[row["source_id"]] = row
    return result


def receipt_sources(collector, registry, *, task_id):
    require(isinstance(collector, dict) and isinstance(registry, dict), "Explicit cached TRAIN source registry required")
    receipt = collector.get("web_cached_rows")
    require(isinstance(receipt, dict) and set(receipt) == {"version", "configuration", "max_page_chars", "source_ids", "source_sha256"}
            and receipt["version"] == VERSION and equal(receipt["configuration"], configuration())
            and type(receipt["max_page_chars"]) is int and receipt["max_page_chars"] == MAX_PAGE_CHARS,
            "Complete cached TRAIN request receipt required")
    ids = receipt["source_ids"]
    require(isinstance(ids, list) and all(isinstance(key, str) for key in ids) and len(ids) == len(set(ids)),
            "Duplicate/invalid cached TRAIN source references")
    episode, turn = collector.get("episode_id"), collector.get("turn")
    require(isinstance(episode, str) and type(turn) is int and 0 <= turn < 20, "Cache receipt episode/turn required")
    expected = [row for row in registry.values() if row["episode_id"] == episode and row["available_turn"] <= turn]
    require(ids == [row["source_id"] for row in expected]
            and equal(receipt["source_sha256"], [row["source_sha256"] for row in expected])
            and all(key in registry and registry[key]["task_id"] == task_id for key in ids),
            "Cached TRAIN references differ from actual active episode/source sequence")
    return expected


class CachedReplay:
    def __init__(self, collector, registry, *, task_id):
        from slime_plugins.dcc.tools import WebTools
        self.sources = receipt_sources(collector, registry, task_id=task_id)
        self.tools = WebTools(web_cached_rows=True, max_page_chars=MAX_PAGE_CHARS)
        self.seen = []

    def observe(self, action, observation, turn):
        pending = [row for row in self.sources if row["available_turn"] == turn + 1]
        if action["tool"] != "fetch_url":
            require(not pending, "Cache source publication lacks an actual preceding fetch")
            return
        args = action["arguments"]
        url = canonical(args["url"])
        require(len(pending) <= 1, "A single fetch cannot publish multiple cache sources")
        if pending:
            require("select_rows" not in args and pending[0]["source"]["url"] == url,
                    "Cache source publication differs from actual normal fetch URL")
            row = pending[0]; source = row["source"]
            self.tools.pages[url] = source["text"]
            for attribute, key in (("page_worksheets", "worksheets"), ("page_archives", "archives"),
                                   ("page_links", "links"), ("page_browser", "browser")):
                if key in source:
                    getattr(self.tools, attribute)[url] = copy.deepcopy(source[key])
            self.seen.append(row["source_id"])
        if url not in self.tools.pages and "select_rows" not in args:
            require(isinstance(observation, dict) and observation.get("error"),
                    "Successful normal fetch lacks its original downloaded cache source")
            return
        # A cached normal fetch or select_rows completes without an await. If an
        # implementation ever tries I/O, close immediately instead of running it.
        call = self.tools.fetch_url(**args)
        try:
            call.send(None)
        except StopIteration as result:
            require(equal(observation, result.value), "Cached TRAIN observation differs from original offline WebTools replay")
        else:
            raise ValueError("Offline cache replay unexpectedly attempted asynchronous I/O")
        finally:
            call.close()

    def finish(self, turn):
        require(self.seen == [row["source_id"] for row in self.sources],
                "Cached TRAIN active sources are missing their original historical fetch")


def validate_references(path, registry):
    """Reject unused/foreign sidecar records before any sampler-visible output."""
    from dcc_native_sft import _strict_json
    used = set()
    for line in Path(path).read_text().splitlines():
        if line.strip():
            bundle = _strict_json(line)
            rows = receipt_sources(bundle.get("collector"), registry, task_id=bundle["record"]["task_id"])
            used.update(row["source_id"] for row in rows)
    require(used == set(registry), "Native cached TRAIN sidecar contains missing or orphan source records")
