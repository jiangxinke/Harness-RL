"""Synthetic CPU controls for explicitly verified empty CSV/TSV padding."""
import asyncio
from copy import deepcopy
import csv
import io
import json
from pathlib import Path
import random
import runpy

import jsonschema
import pytest

from slime_plugins.dcc import EpisodeRunner
from slime_plugins.dcc import web_rows as rows
from slime_plugins.dcc.episode import _WebEvidence
from slime_plugins.dcc.tools import WebTools
from slime_plugins.dcc.web_navigation import WebNavigationLedger, WEB_NAVIGATION_PREFIX
from slime_plugins.dcc.web_structured import (
    web_action_schema, web_structured_actions_configuration, web_structured_configuration,
)

ROOT = Path(__file__).parents[1]
URL = "https://example.org/synthetic-padding.csv"
POLICY = "verified_empty_padding"
CSV = 'id,name,number,,\r\n0001,"Alpha, one",001.2300e+02,,\r\n0002,"中\r\n文",-0.00,,\r\n'


def request(**kwargs):
    return {"format": "csv", "columns": ["id", "name", "number"],
            "csv_trailing_empty_columns": POLICY, **kwargs}


def select(source=CSV, **kwargs):
    return rows.select_cached_rows(source, request(**kwargs))


def web(source=CSV, *, budget=14000, enabled=True):
    result = WebTools(web_cached_rows=enabled, max_page_chars=budget)
    result.pages[URL] = source
    return result


def fetch(instance, req=None):
    return asyncio.run(instance.fetch_url(URL, select_rows=request() if req is None else req))


def assert_spans(source, view):
    for record in view["records"]:
        assert source[record["start"]:record["end"]] == record["text"]
        assert record["end"] - record["start"] == len(record["text"])
    header = view["header"]
    assert source[header["start"]:header["end"]] == header["text"]


@pytest.mark.parametrize("kind,delimiter", [("csv", ","), ("tsv", "\t")])
@pytest.mark.parametrize("padding", [1, 2, 8])
def test_explicit_padding_retains_exact_original_header_rows_and_cells(kind, delimiter, padding):
    output = io.StringIO(newline="")
    writer = csv.writer(output, delimiter=delimiter, lineterminator="\r\n")
    writer.writerow(["id", "name", "number"] + [""] * padding)
    writer.writerow(["00000017", " Name \r\n中 ", "12345678901234567890.0000100"] + [""] * padding)
    source = "\ufeff" + output.getvalue()
    view = select(source, format=kind)
    assert view["version"] == "cached-literal-record-windows-v3"
    assert view["csv_trailing_empty_columns"] == POLICY
    assert_spans(source, view)
    record = view["records"][0]
    assert set(record["cells"]) == {"id", "name", "number"}
    assert [record["cells"][c]["value"] for c in ["id", "name", "number"]] == [
        "00000017", " Name \r\n中 ", "12345678901234567890.0000100"]
    assert view["header"]["text"].endswith(delimiter * padding + "\r\n")
    assert record["text"].endswith(delimiter * padding + "\r\n")


def test_quoted_empty_padding_and_explicit_nonzero_header_record():
    source = 'Synthetic title\r\nid,name,number,"",""\r\nA," spaced ",1.00,"",""\r\n'
    view = select(source, header_row=1)
    assert_spans(source, view)
    assert view["records"][0]["cells"]["name"]["value"] == " spaced "
    assert view["records"][0]["cells"]["number"]["value"] == "1.00"


@pytest.mark.parametrize("policy", [None, True, False, 0, [], {}, "trim", "verify", "", "VERIFIED_EMPTY_PADDING"])
def test_unknown_or_wrong_type_policy_never_guesses(policy):
    with pytest.raises(ValueError, match="csv_trailing_empty_columns"):
        select(csv_trailing_empty_columns=policy)


@pytest.mark.parametrize("policy", ["reject", POLICY])
def test_json_explicitly_carrying_delimited_padding_option_is_rejected(policy):
    with pytest.raises(ValueError, match="JSON selection does not accept"):
        rows.select_cached_rows('[{"id":"A"}]', {
            "format": "json", "columns": ["/id"], "csv_trailing_empty_columns": policy})


def test_omitted_and_explicit_reject_preserve_strict_failure_and_default_success():
    plain = {"format": "csv", "columns": ["id", "name", "number"]}
    for req in [plain, {**plain, "csv_trailing_empty_columns": "reject"}]:
        with pytest.raises(ValueError, match="distinct nonempty exact"):
            rows.select_cached_rows(CSV, req)
    source = "id,name,number\nA,Alpha,001.00\n"
    default = rows.select_cached_rows(source, plain)
    assert default == rows.select_cached_rows(source, {**plain, "csv_trailing_empty_columns": "reject"})
    assert "csv_trailing_empty_columns" not in default
    opted = rows.select_cached_rows(source, {**plain, "csv_trailing_empty_columns": POLICY})
    opted.pop("csv_trailing_empty_columns")
    assert opted == default


@pytest.mark.parametrize("header", [",", ",,", "id,,name,", ",id,name,", "id,name,id,,", "id,id,,", '"",id,'])
def test_inner_empty_duplicate_and_all_empty_headers_remain_invalid(header):
    with pytest.raises(ValueError, match="distinct nonempty exact"):
        rows.select_cached_rows(header + "\n", request(columns=["id"]))


@pytest.mark.parametrize("value", [" ", "\t", "null", "None", "0", "-0", "false", "NA", "\n", "中"])
@pytest.mark.parametrize("position", [0, 1])
def test_every_nonempty_tail_cell_rejects_even_if_late_unselected_or_before_cursor(value, position):
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(["id", "name", "number", "", ""])
    writer.writerow(["A", "Alpha", "1", "", ""])
    tail = ["", ""]
    tail[position] = value
    writer.writerow(["B", "Beta", "2", *tail])
    source = output.getvalue()
    for kwargs in [{"limit": 1}, {"after": 2}, {"where": [{"column": "id", "op": "eq", "value": "A"}]}]:
        with pytest.raises(ValueError, match="nonempty data cell"):
            select(source, **kwargs)


@pytest.mark.parametrize("late", ["B,Beta,2,\n", "B,Beta,2,,,\n", "\n", 'B,"unterminated', 'B,"bad"x,2,,\n'])
def test_late_width_or_parser_errors_reject_entire_selection(late):
    source = "id,name,number,,\nA,Alpha,1,,\n" + late
    instance = web(source)
    before = deepcopy(instance._read_windows)
    result = fetch(instance, request(limit=1, where=[{"column": "id", "op": "eq", "value": "A"}]))
    assert "error" in result
    assert not any(key in result for key in ["text", "row_view", "start", "end"])
    assert instance._read_windows == before and not instance.sources
    assert instance.pages[URL] == source


def test_full_padding_validation_precedes_numeric_predicates_and_after_cursor():
    # If predicate evaluation occurred first, row A would raise a decimal error.
    source = "id,name,number,,\nA,Alpha,not-a-number,,\nB,Beta,2,,0\n"
    with pytest.raises(ValueError, match="nonempty data cell"):
        select(source, after=2, where=[{"column": "number", "op": "num_ge", "value": "1"}])
    source = source.removesuffix("0\n") + "\n"
    with pytest.raises(ValueError, match="finite JSON decimal"):
        select(source, after=2, where=[{"column": "number", "op": "num_ge", "value": "1"}])


def test_padding_is_unaddressable_and_content_header_names_stay_exact():
    for column in ["", "ID", " id", "number "]:
        with pytest.raises(ValueError, match="absent"):
            select(columns=[column])
    with pytest.raises(ValueError, match="absent"):
        select(where=[{"column": "", "op": "eq", "value": ""}])
    source = "id, name ,number,,\nA, Alpha ,1.00,,\n"
    view = select(source, columns=[" name "])
    assert view["records"][0]["cells"][" name "]["value"] == " Alpha "
    with pytest.raises(ValueError, match="absent"):
        select(source, columns=["name"])


def test_existing_numeric_source_policy_composes_without_implicit_conversion():
    source = "id,name,number,,\nA,Alpha,10-20,,\nB,Beta,001.00,,\nC,Gamma,1.00,,\n"
    req = request(where=[{"column": "number", "op": "num_ge", "value": "0"}],
                  numeric_source_policy="exclude_uncomparable")
    view = rows.select_cached_rows(source, req)
    assert [r["cells"]["id"]["value"] for r in view["records"]] == ["C"]
    assert view["uncomparable_records_after"] == 2
    assert view["records"][0]["cells"]["number"]["value"] == "1.00"
    assert_spans(source, view)


def test_no_network_new_evidence_or_read_progress_for_zero_match_or_failed_validation(monkeypatch):
    import slime_plugins.dcc.tools as tools
    monkeypatch.setattr(tools, "_download_public", lambda *a, **kw: pytest.fail("network forbidden"))
    instance = web()
    zero = fetch(instance, request(where=[{"column": "id", "op": "eq", "value": "not-present"}]))
    assert zero["row_view"]["records"] == [] and "text" not in zero
    assert not instance._read_windows and not instance.sources
    guard = _WebEvidence()
    guard.observe("fetch_url", zero)
    assert guard.fetches == 0 and not guard.fetched_sources
    ordinary = asyncio.run(instance.fetch_url(URL))
    before = deepcopy(instance._read_windows)
    instance.pages[URL] += "broken,row\n"
    assert "error" in fetch(instance)
    assert instance._read_windows == before
    assert ordinary["text"]
    missing = WebTools(web_cached_rows=True)
    assert "cache" in fetch(missing)["error"] and not missing.pages


def test_original_observation_budget_paginates_only_actual_literal_windows():
    source = "id,name,number,,\n" + "".join(f"{i},{'x' * 350},{i},,\n" for i in range(30))
    instance = web(source, budget=3500)
    after, seen = 0, []
    while True:
        result = fetch(instance, request(after=after, limit=64))
        assert "error" not in result and len(json.dumps(result, ensure_ascii=False)) <= 3500
        view = result["row_view"]
        assert_spans(source, view)
        assert view["records"] and all(r["read_progress"]["new_text"] for r in view["records"])
        seen.extend(r["record_index"] for r in view["records"])
        assert len(instance._read_windows) == len(seen)
        first = view["records"][0]
        assert all(result[key] == first[key] for key in ["text", "start", "end"])
        if view["next_after"] is None:
            break
        assert view["next_after"] > after
        after = view["next_after"]
    assert seen == list(range(30))


@pytest.mark.parametrize("budget,source", [(200, CSV), (14000, "id,name,number,,\nA," + "x" * 16000 + ",1,,\n")])
def test_overbudget_error_does_not_publish_or_advance_reads(budget, source):
    instance = web(source, budget=budget)
    result = fetch(instance)
    assert "error" in result and "text" not in result and "row_view" not in result
    assert len(json.dumps(result, ensure_ascii=False)) <= budget
    assert not instance._read_windows and not instance.sources


def test_repeated_read_provenance_and_navigation_use_v3_original_spans():
    instance = web()
    first, second = fetch(instance), fetch(instance)
    assert_spans(CSV, first["row_view"])
    assert all(r["read_progress"]["new_text"] for r in first["row_view"]["records"])
    assert all(not r["read_progress"]["new_text"] and r["read_progress"]["previous_identical_reads"] == 1
               for r in second["row_view"]["records"])
    ledger = WebNavigationLedger()
    ledger.observe_events([{"tool": "fetch_url", "observation": first}])
    navigation = json.loads(ledger.render().removeprefix(WEB_NAVIGATION_PREFIX))
    assert {(r["start"], r["end"]) for r in first["row_view"]["records"]} == {
        (w["start"], w["end"]) for w in navigation["recent_windows"]}
    assert all(w["selector_version"] == rows.WEB_CACHED_ROWS_VERSION for w in navigation["recent_windows"]
               if w.get("kind") == "literal-record-window")
    assert first["row_view"]["header"]["start"] not in {w["start"] for w in navigation["recent_windows"]}


def test_worksheet_and_archive_explicit_boundaries_keep_absolute_padding_spans():
    source = 'Workbook\n\nWorksheet: "Exact Sheet"\n' + CSV
    start = source.index("\nWorksheet:")
    worksheet = {"name": "Exact Sheet", "start": start, "end": len(source)}
    view = rows.select_cached_rows(source, request(worksheet="Exact Sheet"), worksheets=[worksheet])
    assert_spans(source, view)
    source = "Archive\nMember navigation\n" + CSV
    archive = {"name": "data.csv", "status": "readable", "format": "csv",
               "content_start": source.index("id,name"), "content_end": len(source)}
    view = rows.select_cached_rows(source, request(archive_member="data.csv"), archives=[archive])
    assert_spans(source, view)
    assert view["records"][0]["text"].endswith(",,\r\n")


def test_original_source_record_header_and_parser_limits_still_apply(monkeypatch):
    with monkeypatch.context() as scoped:
        scoped.setitem(rows.WEB_CACHED_ROWS_LIMITS, "source_chars", len(CSV) - 1)
        with pytest.raises(ValueError, match="character limit"):
            select()
    with monkeypatch.context() as scoped:
        scoped.setitem(rows.WEB_CACHED_ROWS_LIMITS, "records", 1)
        with pytest.raises(ValueError, match="data-record limit"):
            select()
    with pytest.raises(ValueError, match="distinct nonempty"):
        select("id" + "," * 4096 + "\n", columns=["id"])
    maximum_width = "id" + "," * 4095 + "\n" + "A" + "," * 4095 + "\n"
    view = select(maximum_width, columns=["id"])
    assert len(view["records"]) == 1 and view["records"][0]["cells"]["id"]["value"] == "A"
    assert_spans(maximum_width, view)
    limit = csv.field_size_limit()
    try:
        csv.field_size_limit(16)
        with pytest.raises(ValueError, match="parser's field limit"):
            select("id,name,number,,\nA," + "x" * 17 + ",1,,\n")
    finally:
        csv.field_size_limit(limit)


def test_schema_config_manifest_runtime_protocol_and_off_identity_are_consistent():
    assert rows.select_rows_schema()["properties"]["csv_trailing_empty_columns"] == {
        "type": "string", "enum": ["reject", POLICY]}
    jsonschema.validate(request(), rows.select_rows_schema())
    action = {"tool": "fetch_url", "arguments": {"url": URL, "select_rows": request()}}
    jsonschema.validate(action, web_action_schema(["name"], [URL], successful_search=True, cached_rows=True))
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(action, web_action_schema(["name"], [URL], successful_search=True, cached_rows=False))
    config = rows.web_cached_rows_configuration(True)
    assert config["version"] == rows.WEB_CACHED_ROWS_VERSION == "cached-literal-record-windows-v3"
    assert config["csv_trailing_empty_columns"]["default"] == "reject"
    assert config["csv_trailing_empty_columns"]["formats"] == ["csv", "tsv"]
    declarations = [{"web_cached_rows": deepcopy(config), "web_structured_final": web_structured_configuration(True),
                     "web_structured_actions": web_structured_actions_configuration(True), "backend": "vllm"}
                    for _ in range(3)]
    assert rows.validate_web_cached_rows_records(declarations) == config
    for index in range(3):
        for mutation in ["missing-policy", "old-version", "wrong-default", "partial", "wrong-limit-type"]:
            broken = deepcopy(declarations)
            declaration = broken[index]["web_cached_rows"]
            if mutation == "missing-policy": declaration.pop("csv_trailing_empty_columns")
            elif mutation == "old-version": declaration["version"] = "cached-literal-record-windows-v2"
            elif mutation == "wrong-default": declaration["csv_trailing_empty_columns"]["default"] = POLICY
            elif mutation == "partial": declaration["csv_trailing_empty_columns"].pop("validation")
            elif mutation == "wrong-limit-type": declaration["limits"]["depth"] = 32.0
            with pytest.raises(ValueError):
                rows.validate_web_cached_rows_records(broken)
    off = {"enabled": False, "version": None, "scope": None, "cache": None,
           "output": None, "budget": None, "limits": None}
    assert json.dumps(rows.web_cached_rows_configuration(False)) == json.dumps(off)
    assert "csv_trailing_empty_columns=verified_empty_padding" in rows.WEB_CACHED_ROWS_INSTRUCTIONS
    assert "CSV/TSV" in rows.WEB_CACHED_ROWS_INSTRUCTIONS


@pytest.mark.parametrize("review,bounded", [(False, False), (True, False), (False, True), (True, True)])
def test_actual_pinned_decoder_accepts_explicit_csv_padding_option(review, bounded):
    xgrammar = pytest.importorskip("xgrammar")
    from xgrammar.testing import _is_grammar_accept_string
    from slime_plugins.dcc.web_review import web_review_schema
    from slime_plugins.dcc.web_review_decoder import web_review_grammar
    from slime_plugins.dcc.web_whitespace import web_bounded_whitespace_grammar
    schema = web_action_schema(["name"], [URL], successful_search=True, cached_rows=True)
    action = {"tool": "fetch_url", "arguments": {"url": URL, "select_rows": request()}}
    if review:
        schema = web_review_schema(schema)
        helpers = runpy.run_path(str(ROOT / "tests/test_dcc_web_explicit_review.py"))
        action = helpers["reviewed"](action)
    if bounded:
        grammar = xgrammar.Grammar.from_ebnf(web_bounded_whitespace_grammar(schema, explicit_review=review))
    elif review:
        grammar = xgrammar.Grammar.from_ebnf(web_review_grammar(schema))
    else:
        grammar = xgrammar.Grammar.from_json_schema(json.dumps(schema))
    assert _is_grammar_accept_string(grammar, json.dumps(action).replace("/", "\\/"))


def test_actual_synthetic_episode_model_explicitly_selects_padding_without_new_turns():
    helpers = runpy.run_path(str(ROOT / "tests/test_dcc_web_explicit_review.py"))
    class Cached(WebTools):
        async def web_search(self, query, count=8):
            return {"results": [{"url": URL, "title": "Synthetic padding table", "snippet": "navigation only"}]}
        async def fetch_url(self, url, start=0, select_rows=None):
            if select_rows is None:
                self.pages[url] = CSV
            return await super().fetch_url(url, start, select_rows)
    actions = [{"tool": "web_search", "arguments": {"query": "synthetic only"}},
               {"tool": "fetch_url", "arguments": {"url": URL}},
               {"tool": "fetch_url", "arguments": {"url": URL, "select_rows": request(
                   where=[{"column": "id", "op": "eq", "value": "0001"}])}},
               {"answer": [{"name": "Alpha, one", "citations": [URL]}]}]
    policy = helpers["Policy"](actions)
    result = asyncio.run(EpisodeRunner(max_steps=4, max_new_tokens=97, web_tools_factory=Cached,
        web_cached_rows=True, web_structured_actions=True, web_structured_final=True).run(
            helpers["task"](), policy, evaluation=True))
    assert result.status == "completed" and len(result.steps) == 4 and len(result.tool_events) == 3
    assert result.metrics["evidence_guard"]["accepted"] is True
    assert all(call[2]["max_new_tokens"] == 97 for call in policy.calls)
    assert_spans(CSV, result.tool_events[2]["observation"]["row_view"])
    assert POLICY in json.dumps(policy.calls[2][2]["json_schema"])


def test_seeded_synthetic_quoted_tables_preserve_every_original_cell_and_span():
    rng = random.Random(891)
    alphabet = ["x", "中", "0", " ", ",", "\t", '"', "\n", "é"]
    for index in range(100):
        kind, delimiter = ("csv", ",") if index % 2 else ("tsv", "\t")
        padding = 1 + index % 5
        content = [[str(index).zfill(15), "".join(rng.choice(alphabet) for _ in range(30)), "-0.000100"]]
        output = io.StringIO(newline="")
        writer = csv.writer(output, delimiter=delimiter, lineterminator="\r\n")
        writer.writerow(["id", "name", "number"] + [""] * padding)
        for record in content:
            writer.writerow(record + [""] * padding)
        source = output.getvalue()
        view = select(source, format=kind)
        assert_spans(source, view)
        actual = [view["records"][0]["cells"][column]["value"] for column in ["id", "name", "number"]]
        assert actual == content[0]
