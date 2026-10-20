"""Synthetic CPU contracts for opt-in numeric filtering of cached source rows."""
import asyncio
from copy import deepcopy
import json

import jsonschema
import pytest

from slime_plugins.dcc import web_rows as new
from slime_plugins.dcc import tools, web_navigation as navigation, web_structured as structured

# Frozen compatibility fixtures from the prior protocol, using only synthetic
# source strings. These contain no model output, benchmark or environment data.
LEGACY_DEFAULT_VIEW = {'version': 'cached-literal-record-windows-v1',
 'kind': 'literal-record-windows',
 'format': 'csv',
 'source_start': 0,
 'source_end': 18,
 'columns': ['id', 'number'],
 'after': 0,
 'candidate_records': 2,
 'matching_records_after': 2,
 'header': {'start': 0,
            'end': 10,
            'text': 'id,number\n',
            'role': 'column-navigation-only;not-a-data-record'},
 'records': [{'record_index': 0,
              'start': 10,
              'end': 14,
              'text': 'A,1\n',
              'cells': {'id': {'present': True, 'kind': 'string', 'value': 'A'},
                        'number': {'present': True, 'kind': 'string', 'value': '1'}}},
             {'record_index': 1,
              'start': 14,
              'end': 18,
              'text': 'B,2\n',
              'cells': {'id': {'present': True, 'kind': 'string', 'value': 'B'},
                        'number': {'present': True, 'kind': 'string', 'value': '2'}}}],
 'note': 'Each record is a separate actually observed literal source window. Cells are exact '
         'decoded source scalars; numeric values retain their JSON lexeme. Absent fields are not '
         'filled. Selection completeness applies only to this cached source, never to a remote '
         'dataset or undocumented API pagination.'}

LEGACY_OFF_CONFIG = {'enabled': False,
 'version': None,
 'scope': None,
 'cache': None,
 'output': None,
 'budget': None,
 'limits': None}

LEGACY_ON_CONFIG = {'enabled': True,
 'version': 'cached-literal-record-windows-v1',
 'scope': 'structured-web-evaluation',
 'cache': 'episode-downloaded-extracted-text-only',
 'output': 'literal-record-windows-with-cell-provenance',
 'budget': 'original-max-page-chars-entire-json-observation',
 'limits': {'source_chars': 5000000,
            'records': 100000,
            'nodes': 300000,
            'depth': 32,
            'columns': 32,
            'predicates': 8,
            'returned_rows': 64}}

LEGACY_OFF_SCHEMA = {'anyOf': [{'type': 'object',
            'properties': {'tool': {'const': 'web_search'},
                           'arguments': {'type': 'object',
                                         'properties': {'query': {'type': 'string'},
                                                        'count': {'type': 'integer',
                                                                  'minimum': 1,
                                                                  'maximum': 10}},
                                         'required': ['query'],
                                         'additionalProperties': False}},
            'required': ['tool', 'arguments'],
            'additionalProperties': False},
           {'type': 'object',
            'properties': {'tool': {'const': 'fetch_url'},
                           'arguments': {'type': 'object',
                                         'properties': {'url': {'type': 'string'},
                                                        'start': {'type': 'integer', 'minimum': 0}},
                                         'required': ['url'],
                                         'additionalProperties': False}},
            'required': ['tool', 'arguments'],
            'additionalProperties': False},
           {'type': 'object',
            'properties': {'answer': {'type': 'array',
                                      'items': {'anyOf': [{'type': 'object',
                                                           'properties': {'id': {'anyOf': [{'type': 'string',
                                                                                            'minLength': 1},
                                                                                           {'type': ['number',
                                                                                                     'boolean']}]},
                                                                          'citations': {'type': 'array',
                                                                                        'minItems': 1,
                                                                                        'maxItems': 1,
                                                                                        'items': {'type': 'string',
                                                                                                  'enum': ['https://fixture.example.org/records']}}},
                                                           'required': ['id', 'citations'],
                                                           'additionalProperties': False}]}}},
            'required': ['answer'],
            'additionalProperties': False}]}

LEGACY_DISABLED_OBSERVATION = {'url': 'https://fixture.example.org/records', 'error': 'Cached row selection is disabled'}

URL = "https://fixture.example.org/records"
CSV = "id,number\r\nA,1\r\nB,10-20\r\nC,Unassigned\r\nD,12.00\r\nE,-2\r\n"


def request(**changes):
    return {"format": "csv", "columns": ["id", "number"],
            "where": [{"column": "number", "op": "num_ge", "value": "0"}], **changes}


def opt_request(**changes):
    return request(numeric_source_policy="exclude_uncomparable", **changes)


def instance(source=CSV, budget=14000, enabled=True):
    result = tools.WebTools(web_cached_rows=enabled, max_page_chars=budget)
    result.pages[URL] = source
    return result


def fetch(web, req):
    return asyncio.run(web.fetch_url(URL, select_rows=req))


def ids(result):
    return [row["cells"]["id"]["value"] for row in result["records"]]


def assert_spans(source, result):
    for row in result["row_view"]["records"]:
        assert source[row["start"]:row["end"]] == row["text"]
    if result["row_view"]["records"]:
        first = result["row_view"]["records"][0]
        assert all(result[key] == first[key] for key in ("start", "end", "text"))


def test_mixed_column_default_and_explicit_error_retain_original_failure():
    for req in (request(), request(numeric_source_policy="error")):
        with pytest.raises(ValueError, match="finite JSON decimal"):
            new.select_cached_rows(CSV, req)


@pytest.mark.parametrize("op,value,expected", [("num_ge", "0", ["A", "D"]),
                                                ("num_le", "2", ["A", "E"]),
                                                ("num_eq", "12", ["D"])])
def test_explicit_policy_selects_only_comparable_literal_rows(op, value, expected):
    req = opt_request(where=[{"column": "number", "op": op, "value": value}])
    original = deepcopy(req)
    result = new.select_cached_rows(CSV, req)
    assert req == original and ids(result) == expected
    assert result["uncomparable_records_after"] == 2
    assert result["predicate_diagnostics"]["items"] == [
        {"predicate_index": 0, "match_count": len(expected), "uncomparable_count": 2}]
    for row in result["records"]:
        assert CSV[row["start"]:row["end"]] == row["text"]
    if "D" in expected:
        assert next(r for r in result["records"] if r["cells"]["id"]["value"] == "D")["cells"]["number"]["value"] == "12.00"


def test_json_scalar_types_missing_and_exact_numeric_lexemes():
    source = '[{"n":1.00},{"n":"2"},{"n":"3-8"},{"n":null},{"n":true},{}]'
    req = {"format": "json", "columns": ["/n"], "numeric_source_policy": "exclude_uncomparable",
           "where": [{"column": "/n", "op": "num_ge", "value": "0"}]}
    result = new.select_cached_rows(source, req)
    assert [r["cells"]["/n"]["value"] for r in result["records"]] == ["1.00", "2"]
    assert result["uncomparable_records_after"] == 3
    assert result["predicate_diagnostics"]["items"][0]["uncomparable_count"] == 3
    for row in result["records"]:
        cell = row["cells"]["/n"]
        assert source[row["start"]:row["end"]] == row["text"]
        assert source[cell["start"]:cell["end"]] in {"1.00", '"2"'}
    # Structurally invalid scalar operands are not rescued by the numeric policy.
    with pytest.raises(ValueError, match="must be scalar"):
        new.select_cached_rows('[{"n":[1]}]', req)


@pytest.mark.parametrize("value", ["10-20", "Unassigned", "NaN", "1e10001", "", True, None])
def test_invalid_selector_numeric_argument_is_strict_even_for_empty_source(value):
    req = opt_request(where=[{"column": "number", "op": "num_ge", "value": value}])
    with pytest.raises(ValueError):
        new.select_cached_rows("id,number\n", req)


def test_unknown_policy_and_malformed_source_are_still_errors():
    for value in ("guess", None, [], True):
        with pytest.raises(ValueError, match="numeric_source_policy"):
            new.select_cached_rows(CSV, request(numeric_source_policy=value))
    for source in ("id,number\nA,1\nB\n", "id,id\nA,1\n"):
        with pytest.raises(ValueError):
            new.select_cached_rows(source, opt_request())
    with pytest.raises(ValueError, match="absent"):
        new.select_cached_rows("other\n1\n", opt_request())


def test_independent_counts_show_and_intersection_without_claiming_evidence():
    req = opt_request(where=[{"column": "number", "op": "num_ge", "value": "10"},
                             {"column": "number", "op": "num_le", "value": "2"}])
    web = instance()
    result = fetch(web, req)
    view = result["row_view"]
    assert view["records"] == [] and view["matching_records_after"] == 0
    assert view["predicate_diagnostics"]["semantics"] == "AND"
    assert view["predicate_diagnostics"]["items"] == [
        {"predicate_index": 0, "match_count": 1, "uncomparable_count": 2},
        {"predicate_index": 1, "match_count": 2, "uncomparable_count": 2}]
    assert view["uncomparable_records_after"] == 2  # Records, not duplicated predicate evaluations.
    assert "text" not in result and not web._read_windows and not web.sources


def test_counts_follow_after_cursor_and_ne_retains_exact_exclusion_semantics():
    result = new.select_cached_rows(CSV, opt_request(after=3))
    assert ids(result) == ["D"] and result["uncomparable_records_after"] == 0
    assert result["predicate_diagnostics"]["scope"] == "cached-records-at-or-after-cursor"
    req = opt_request(where=[{"column": "id", "op": "ne", "value": "A"},
                             {"column": "number", "op": "num_ge", "value": "0"}])
    assert ids(new.select_cached_rows(CSV, req)) == ["D"]
    # An invalid earlier cell still fails strict mode even when before the cursor.
    with pytest.raises(ValueError):
        new.select_cached_rows(CSV, request(after=3))


def test_literal_spans_repeat_telemetry_and_v2_ledger_all_use_original_tools():
    web = instance()
    first, second = fetch(web, opt_request()), fetch(web, opt_request())
    assert_spans(CSV, first)
    assert_spans(CSV, second)
    assert all(r["read_progress"]["new_text"] for r in first["row_view"]["records"])
    assert all(not r["read_progress"]["new_text"] and r["read_progress"]["previous_identical_reads"] == 1
               for r in second["row_view"]["records"])
    assert set(web._read_windows) == {row["text"] for row in first["row_view"]["records"]}
    ledger = navigation.WebNavigationLedger()
    ledger.observe_events([{"tool": "fetch_url", "observation": first}])
    snapshot = json.loads(ledger.render().removeprefix(navigation.WEB_NAVIGATION_PREFIX))
    assert {(w["start"], w["end"]) for w in snapshot["recent_windows"]} == {
        (r["start"], r["end"]) for r in first["row_view"]["records"]}
    extra = [w for w in snapshot["recent_windows"] if w.get("kind") == "literal-record-window"]
    assert extra and all(w["selector_version"] == new.WEB_CACHED_ROWS_VERSION for w in extra)
    assert first["row_view"]["version"] == new.WEB_CACHED_ROWS_VERSION == "cached-literal-record-windows-v3"


def test_original_entire_observation_14k_budget_paginates_without_marking_hidden_rows():
    source = "id,number,payload\n" + "".join(f"{i},{i},{'x' * 400}\n" for i in range(40))
    source += "range,5-8,omitted\n"
    web, after, seen = instance(source), 0, []
    while True:
        req = opt_request(columns=["id", "number", "payload"], after=after, limit=64)
        result = fetch(web, req)
        assert "error" not in result and len(json.dumps(result, ensure_ascii=False)) <= 14000
        assert_spans(source, result)
        view = result["row_view"]
        seen.extend(r["record_index"] for r in view["records"])
        assert "range,5-8,omitted\n" not in web._read_windows
        if view["next_after"] is None:
            break
        assert view["next_after"] > after
        assert len(web._read_windows) == len(seen)
        after = view["next_after"]
    assert seen == list(range(40)) and len(web._read_windows) == 40


def test_oversize_metadata_or_row_is_atomic_and_does_not_advance_reads():
    for source, budget in (("id,number\nA,1\n", 500),
                           ("id,number\n" + "a" * 16000 + ",1\n", 14000)):
        web = instance(source, budget=budget)
        result = fetch(web, opt_request())
        assert "error" in result and "text" not in result and "row_view" not in result
        assert len(json.dumps(result, ensure_ascii=False)) <= budget
        assert not web._read_windows and not web.sources


def test_default_success_unchanged_except_declared_v2_and_feature_off_bytes_equal():
    source = "id,number\nA,1\nB,2\n"
    before = deepcopy(LEGACY_DEFAULT_VIEW)
    after = new.select_cached_rows(source, request())
    after["version"] = before["version"]
    assert json.dumps(before, ensure_ascii=False) == json.dumps(after, ensure_ascii=False)
    assert new.select_cached_rows(source, request()) == new.select_cached_rows(
        source, request(numeric_source_policy="error"))
    assert json.dumps(LEGACY_OFF_CONFIG) == json.dumps(new.web_cached_rows_configuration(False))
    assert LEGACY_OFF_SCHEMA == \
        structured.web_action_schema(["id"], [URL], successful_search=True, cached_rows=False)
    old_off = LEGACY_DISABLED_OBSERVATION
    new_off = fetch(instance(source, enabled=False), opt_request())
    assert json.dumps(old_off) == json.dumps(new_off)


def test_schema_prompt_and_configuration_version_are_plumbed_generically():
    jsonschema.validate(opt_request(), new.select_rows_schema())
    schema = structured.web_action_schema(["id"], [URL], successful_search=True, cached_rows=True)
    jsonschema.validate({"tool": "fetch_url", "arguments": {"url": URL, "select_rows": opt_request()}}, schema)
    config = new.web_cached_rows_configuration(True)
    record = {"web_cached_rows": config, "web_structured_final": structured.web_structured_configuration(True),
              "web_structured_actions": structured.web_structured_actions_configuration(True), "backend": "vllm"}
    assert new.validate_web_cached_rows_records([record, deepcopy(record)]) == config
    mismatched = {**record, "web_cached_rows": deepcopy(LEGACY_ON_CONFIG)}
    with pytest.raises(ValueError, match="complete supported configuration"):
        new.validate_web_cached_rows_records([record, mismatched])
    assert all(text in new.WEB_CACHED_ROWS_INSTRUCTIONS for text in
               ("exclude_uncomparable", "does not prove", "inspect those source records", "ANDed", "Use ne"))
    assert "IANA" not in new.WEB_CACHED_ROWS_INSTRUCTIONS
