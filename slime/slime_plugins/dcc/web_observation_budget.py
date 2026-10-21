"""Default-off token budgeting of complete Web tool observations.

This operates before observations enter episode evidence or messages. Source
text remains a literal character window in the downloaded page, never a
summary. Whole-request prompt limits are still enforced independently.
"""
from __future__ import annotations

from copy import deepcopy
import json


VERSION = "web-observation-token-windows-v1"
MIN_TOKENS = 512


def web_observation_budget_configuration(tokens=None):
    if tokens is not None and (type(tokens) is not int or tokens < MIN_TOKENS):
        raise ValueError(f"Web observation token budget must be at least {MIN_TOKENS}")
    return dict(version=VERSION, max_tokens=tokens,
        scope="complete-tool-observation-JSON-including-auto-reads-and-metadata",
        source_text="literal-prefix-window; character-offset-continuation",
        evidence="only-packed-observations", metadata="explicitly-omitted-when-over-budget",
        prompt_limit="original-strict-check-still-required")


class WebObservationBudget:
    def __init__(self, tokenizer, tokens):
        web_observation_budget_configuration(tokens)
        if tokens is None or not callable(getattr(tokenizer, "encode", None)):
            raise ValueError("An actual tokenizer and explicit Web observation budget are required")
        self.tokenizer, self.tokens = tokenizer, tokens

    def count(self, value):
        return len(self.tokenizer.encode(json.dumps(value, ensure_ascii=False), add_special_tokens=False))

    def pack(self, observation):
        if not isinstance(observation, dict):
            raise ValueError("Web tool observations must be JSON objects")
        if self.count(observation) <= self.tokens:
            return deepcopy(observation)

        # Only retain whole identity/control fields. Arbitrary source-wide
        # metadata can be much larger than the ordinary text character cap.
        # Its omission is explicit, and never makes an unread source read.
        payloads = []
        omitted_metadata_fields = 0
        def project(original):
            nonlocal omitted_metadata_fields
            value = {key: deepcopy(original[key]) for key in (
                "url", "start", "end", "total_characters", "total_chars", "truncated",
                "next_start", "source_text_exposed", "http_status", "retryable", "cached",
            ) if key in original}
            if original.get("error"):
                value["error"] = "Tool reported an error; details were shortened by the observation budget."
                value["error_details_truncated"] = True
                if isinstance(original["error"], str):
                    payloads.append((value, original, "error", original["error"]))
            elif isinstance(original.get("text"), str):
                text = original["text"]
                start, end = original.get("start"), original.get("end")
                if type(start) is int and start >= 0 and type(end) is int and end - start == len(text):
                    value["text"] = ""
                    value.update(end=start, next_start=start, truncated=True, source_text_exposed=False)
                    payloads.append((value, original, "text", text))
                else:
                    value["error"] = "Source window has no aligned character offsets; request a regular fetch_url window."
                    value["retryable"] = True
                    value["source_text_exposed"] = False
            rows = original.get("results")
            if isinstance(rows, list):
                value["results"] = [{"url": row["url"]} for row in rows
                    if isinstance(row, dict) and isinstance(row.get("url"), str)]
                omitted_metadata_fields += sum(len(row) - 1 for row in rows
                    if isinstance(row, dict) and isinstance(row.get("url"), str))
            sources = original.get("auto_read_sources")
            if isinstance(sources, list):
                value["auto_read_sources"] = [project(source) for source in sources if isinstance(source, dict)]
            if type(original.get("auto_read_attempts_remaining")) is int:
                value["auto_read_attempts_remaining"] = original["auto_read_attempts_remaining"]
            omitted_metadata_fields += len(set(original) - set(value))
            return value

        result = project(observation)
        notice = dict(version=VERSION, max_tokens=self.tokens, truncated=True,
            metadata_truncated=omitted_metadata_fields > 0, omitted_metadata_fields=omitted_metadata_fields,
            omitted_search_results=0, omitted_auto_read_sources=0, omitted_auto_read_errors=0,
            recovery="Only returned text windows were exposed. Continue fetch_url at next_start; omitted metadata is not evidence.")
        result["observation_budget"] = notice

        def successful(source):
            return (not source.get("error") and source.get("source_text_exposed") is not False
                and isinstance(source.get("text"), str) and bool(source["text"].strip()))

        def finish():
            if "auto_read_sources" in result:
                result["auto_read_successful_sources"] = sum(successful(source) for source in result["auto_read_sources"])

        finish()
        # Retain whole URLs, never abbreviated identities. If identity-only
        # navigation is itself too big, omit complete entries explicitly.
        entries = (("results", "omitted_search_results"),
                   ("auto_read_sources", "omitted_auto_read_sources"))
        while self.count(result) > self.tokens:
            tails = [(field, counter) for field, counter in entries if result.get(field)]
            if not tails:
                break
            # A huge automatic-source identity must not erase all otherwise
            # usable search results before being omitted itself. Removing only
            # tails also preserves the real automatic-event/source alignment.
            field, counter = max(tails, key=lambda entry: self.count(result[entry[0]][-1]))
            removed = result[field].pop()
            notice[counter] += 1
            if field == "auto_read_sources" and removed.get("error"):
                notice["omitted_auto_read_errors"] += 1
            finish()
        if self.count(result) > self.tokens:
            # A single enormous URL/control field cannot be sliced safely.
            # Preserve failure, and do not expose/cite a rewritten identity.
            result = {"error": "Web identity exceeds observation budget; use a shorter URL or increase budget.",
                "retryable": True, "source_text_exposed": False,
                "observation_budget": {key: notice[key] for key in
                    ("version", "max_tokens", "truncated", "omitted_search_results",
                     "omitted_auto_read_sources", "omitted_auto_read_errors")}}
            result["observation_budget"].update(identity_omitted=True, metadata_truncated=True)
            if self.count(result) > self.tokens:
                raise ValueError("Tokenizer cannot fit even the explicit Web observation budget error")
            return result

        # Search over character prefixes, measuring the ENTIRE JSON every
        # time. BPE token counts need not be monotone: retain a measured fit,
        # rather than assuming the final binary-search boundary must fit.
        def render(scale):
            for value, original, field, text in payloads:
                count = len(text) * scale // 1_000_000
                if field == "error":
                    value[field] = text[:count] or "Tool reported an error (details exceed the observation budget)."
                    value["error_details_truncated"] = count < len(text)
                    continue
                value["text"] = text[:count]
                truncated = count < len(text) or bool(original.get("truncated"))
                value.update(end=original["start"] + count,
                    truncated=truncated,
                    next_start=original["start"] + count if truncated else None)
                exposed = bool(value["text"].strip()) and original.get("source_text_exposed") is not False
                if isinstance(original.get("archive_members"), list):
                    exposed = exposed and any(isinstance(member, dict) and member.get("status") == "readable"
                        and type(member.get("content_start")) is int and type(member.get("content_end")) is int
                        and bool(value["text"][max(0, member["content_start"] - original["start"]):
                            max(0, min(count, member["content_end"] - original["start"]))].strip())
                        for member in original["archive_members"])
                value["source_text_exposed"] = exposed
            finish()

        best = deepcopy(result)
        low, high = 0, 1_000_000
        while low <= high:
            scale = (low + high) // 2
            render(scale)
            if self.count(result) <= self.tokens:
                best = deepcopy(result)
                low = scale + 1
            else:
                high = scale - 1
        return best
