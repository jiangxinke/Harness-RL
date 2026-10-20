from __future__ import annotations

import asyncio
import dataclasses
import json
import runpy
from pathlib import Path

import pytest

from slime_plugins.dcc import tools
from slime_plugins.dcc.web_curriculum import (
    PUBLIC_TABLES, REGISTRIES, PublicTable, Registry, public_table_rows,
    public_table_tasks, registry_tasks, verify_curriculum_rows,
)


def spec(split="train"):
    return Registry("fixture", "Independent Registry", "https://example.org/data.csv",
                    (("Value", "number"), ("Description", "name")), ((100, 110),), split)


CSV = "Value,Description,Reference\n100,Continue,rfc\n101,Switching Protocols,rfc\n"


def test_sources_do_not_cross_training_and_validation():
    train = {source.url for source in REGISTRIES if source.split == "train"}
    validation = {source.url for source in REGISTRIES if source.split == "validation"}
    assert train and validation and train.isdisjoint(validation)
    assert all("widesearch" not in source.url.lower() for source in REGISTRIES)


def test_registry_filters_ranges_and_unknown_entries_without_inventing_fields():
    text = CSV + "102,Unassigned,\n103,Reserved,\n104,(Unused),\n105,,\n106-109,Range,\n200,Outside,rfc\n"
    tasks = registry_tasks(spec(), text)
    assert len(tasks) == 2
    assert {task.metadata["language"] for task in tasks} == {"en", "zh"}
    assert all(task.gold == [{"number": "100", "name": "Continue"}, {"number": "101", "name": "Switching Protocols"}] for task in tasks)
    assert all(task.metadata["source"] == "independent_curriculum" for task in tasks)
    assert all(task.gold[0]["name"] not in task.prompt for task in tasks)


@pytest.mark.parametrize("text", ["Value,Wrong\n100,Continue\n", CSV + "100,Duplicate,rfc\n"])
def test_bad_csv_fails_before_creating_training_labels(text):
    with pytest.raises(ValueError):
        registry_tasks(spec(), text)


def test_validation_has_one_language_per_range():
    train = registry_tasks(spec(), CSV)
    validation = registry_tasks(spec("validation"), CSV)
    assert len(train) == 2 and len(validation) == 1
    # The shipped split uses different source registries, checked above.
    assert validation[0].split == "validation"


def test_broader_public_sources_have_disjoint_families_and_multiple_domains():
    train = {source.family for source in PUBLIC_TABLES if source.split == "train"}
    validation = {source.family for source in PUBLIC_TABLES if source.split == "validation"}
    assert len(train) >= 3 and len(validation) >= 2 and train.isdisjoint(validation)
    assert len({source.url.split("/")[2] for source in PUBLIC_TABLES}) >= 5
    assert all("widesearch" not in source.url.lower() for source in PUBLIC_TABLES)


def table_source(parser):
    return next(source for source in PUBLIC_TABLES if source.parser == parser)


PUBCHEM = json.dumps({"Table": {"Columns": {"Column": ["AtomicNumber", "Symbol", "Name", "AtomicMass", "ElectronConfiguration", "GroupBlock", "StandardState", "YearDiscovered"]},
                              "Row": [{"Cell": ["1", "H", "Hydrogen", "1.0080", "1s1", "Nonmetal", "Gas", "1766"]},
                                      {"Cell": ["2", "He", "Helium", "4.00260", "1s2", "Noble gas", "Gas", "1868"]}]}})


def test_public_table_tasks_preserve_cells_and_keep_translations_in_training():
    tasks = public_table_tasks(table_source("pubchem"), PUBCHEM)
    assert len(tasks) == 2 and {task.metadata["language"] for task in tasks} == {"en", "zh"}
    assert tasks[0].gold == tasks[1].gold == [
        {"number": "1", "symbol": "H", "name": "Hydrogen", "atomic_mass": "1.0080", "electron_configuration": "1s1", "group": "Nonmetal", "state": "Gas", "discovered": "1766"},
        {"number": "2", "symbol": "He", "name": "Helium", "atomic_mass": "4.00260", "electron_configuration": "1s2", "group": "Noble gas", "state": "Gas", "discovered": "1868"},
    ]
    assert all("Hydrogen" not in task.prompt and "Helium" not in task.prompt for task in tasks)
    assert all(task.metadata["evaluation"]["unique_columns"] == ["number"] for task in tasks)
    verify_curriculum_rows(tasks[0], PUBCHEM)


def test_compact_official_csv_preserves_the_same_rich_pubchem_labels():
    import csv
    import io

    value = json.loads(PUBCHEM)["Table"]
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(value["Columns"]["Column"])
    writer.writerows(row["Cell"] for row in value["Row"])
    assert public_table_rows(table_source("pubchem"), stream.getvalue()) == public_table_rows(table_source("pubchem"), PUBCHEM)
    assert len(table_source("pubchem").fields) >= 8


def test_new_validation_includes_rich_schemas_without_train_family_overlap():
    validation = [source for source in PUBLIC_TABLES if source.split == "validation"]
    assert all(len(source.fields) >= 8 for source in validation)


def test_teacher_source_validation_rejects_mixed_rows_even_when_every_cell_occurs():
    task = public_table_tasks(table_source("pubchem"), PUBCHEM)[0]
    mixed = [dict(row) for row in task.gold]
    mixed[0]["name"], mixed[1]["name"] = mixed[1]["name"], mixed[0]["name"]
    with pytest.raises(ValueError, match="supported"):
        verify_curriculum_rows(dataclasses.replace(task, gold=mixed), PUBCHEM)


@pytest.mark.parametrize("parser,text,expected", [
    ("china_cities", '[{"code":"1301","name":"石家庄市","provinceCode":"13"}]',
     [{"code": "1301", "name": "石家庄市", "province_code": "13"}]),
    ("nobel", '{"meta":{"count":1,"offset":0},"nobelPrizes":[{"awardYear":"1901","category":{"en":"Physics"},"laureates":[{"id":"42","portion":"1/2","motivation":{"en":"An independent scientific contribution"},"knownName":{"en":"A Person"}}]}]}',
     [{"year": "1901", "category": "Physics", "name": "A Person", "laureate_id": "42", "share": "1/2", "motivation": "An independent scientific contribution"}]),
    ("worldbank", '[{"page":1,"pages":1,"total":2},[{"id":"ABC","name":"Alpha","capitalCity":"Alpha City","region":{"id":"LCN","value":"Latin America "},"incomeLevel":{"value":"High income"},"lendingType":{"value":"Not classified"},"longitude":"-1.5","latitude":"2.5"},{"id":"AGG","name":"Aggregate","capitalCity":"","region":{"id":"NA"}}]]',
     [{"code": "ABC", "name": "Alpha", "capital": "Alpha City", "region": "Latin America", "income_group": "High income", "lending_type": "Not classified", "longitude": "-1.5", "latitude": "2.5"}]),
    ("gbif", '{"offset":0,"endOfRecords":true,"count":2,"results":[{"key":1,"canonicalName":"AvesOrder","scientificName":"AvesOrder Author","rank":"ORDER","taxonomicStatus":"ACCEPTED","kingdom":"Animalia","phylum":"Chordata","class":"Aves","parent":"Aves"},{"key":2,"rank":"ORDER","taxonomicStatus":"SYNONYM"}]}',
     [{"taxon_id": "1", "name": "AvesOrder", "scientific_name": "AvesOrder Author", "rank": "ORDER", "kingdom": "Animalia", "phylum": "Chordata", "class": "Aves", "parent": "Aves", "status": "ACCEPTED"}]),
])
def test_public_table_parsers_keep_supported_records_only(parser, text, expected):
    assert public_table_rows(table_source(parser), text) == expected


@pytest.mark.parametrize("parser,text", [
    ("nobel", '{"meta":{"count":2},"nobelPrizes":[]}'),
    ("worldbank", '[{"page":1,"pages":2,"total":2},[]]'),
    ("gbif", '{"offset":0,"endOfRecords":false,"count":1,"results":[]}'),
])
def test_incomplete_api_pages_never_become_full_enumeration_labels(parser, text):
    with pytest.raises(ValueError, match="incomplete"):
        public_table_rows(table_source(parser), text)


def test_public_sources_reject_duplicate_keys_and_malformed_cells():
    duplicated = json.loads(PUBCHEM)
    duplicated["Table"]["Row"].append(duplicated["Table"]["Row"][0])
    with pytest.raises(ValueError, match="Duplicate"):
        public_table_rows(table_source("pubchem"), json.dumps(duplicated))
    duplicated["Table"]["Row"][0]["Cell"].pop()
    with pytest.raises(ValueError, match="malformed"):
        public_table_rows(table_source("pubchem"), json.dumps(duplicated))


def test_teacher_limit_balances_sources_languages_and_excludes_validation():
    path = Path(__file__).resolve().parents[1] / "scripts/dcc_prepare_web_curriculum.py"
    balanced = runpy.run_path(str(path))["balanced_training_tasks"]
    rows = registry_tasks(spec(), CSV) + public_table_tasks(table_source("pubchem"), PUBCHEM)
    rows.append(dataclasses.replace(rows[0], id="held-out", split="validation"))
    selected = balanced(rows, limit=4)
    assert {task.metadata["source_family"] for task in selected} == {"iana-fixture", "pubchem"}
    assert {task.metadata["language"] for task in selected} == {"en", "zh"}
    assert all(task.split == "train" for task in selected)
    assert selected == balanced(list(reversed(rows)), limit=4)


@pytest.fixture
def teacher_transport(monkeypatch):
    monkeypatch.setenv("DCC_SEARCH_BACKEND", "duckduckgo_lite")
    calls = []

    def search(*args, **kwargs):
        calls.append("search")
        return '<a class="result-link" href="https://example.org/data.csv">Independent Registry</a>'

    def download(url):
        calls.append("fetch")
        return CSV.encode(), "text/csv", "utf-8"

    monkeypatch.setattr(tools, "_request", search)
    monkeypatch.setattr(tools, "_download_public", download)
    return calls


def teacher_function():
    path = Path(__file__).resolve().parents[1] / "scripts/dcc_prepare_web_curriculum.py"
    return runpy.run_path(str(path))["teacher"]


def test_teacher_runs_real_episode_tools_guard_and_metrics(teacher_transport):
    records, episode = asyncio.run(teacher_function()(registry_tasks(spec(), CSV)[0]))
    assert teacher_transport == ["search", "fetch"]
    assert episode["status"] == "completed" and episode["reward"] == 1.0
    assert [record["stage"] for record in records] == ["search", "fetch", "final"]
    assert [len(record["messages"]) for record in records] == [2, 4, 6]
    assert all(record["completion"] not in [message["content"] for message in record["messages"]] for record in records)
    assert "Switching Protocols" in records[-1]["messages"][-1]["content"]
    assert records[-1]["grounding_fields"] == ["100", "Continue", "101", "Switching Protocols"]


def test_teacher_rejects_validation_before_network(teacher_transport):
    task = registry_tasks(spec("validation"), CSV)[0]
    with pytest.raises(ValueError, match="training"):
        asyncio.run(teacher_function()(task))
    assert teacher_transport == []


def test_teacher_does_not_train_stale_source_labels(teacher_transport):
    task = registry_tasks(spec(), CSV)[0]
    task = dataclasses.replace(task, gold=[{"number": "100", "name": "Invented"}])
    with pytest.raises(ValueError, match="supported"):
        asyncio.run(teacher_function()(task))


def test_teacher_retries_empty_search_before_reading(monkeypatch, teacher_transport):
    attempts = []

    def search(*args, **kwargs):
        attempts.append(True)
        if len(attempts) == 1:
            return "No results"
        return '<a class="result-link" href="https://example.org/data.csv">Independent Registry</a>'

    monkeypatch.setattr(tools, "_request", search)
    records, episode = asyncio.run(teacher_function()(registry_tasks(spec(), CSV)[0]))
    assert [record["stage"] for record in records] == ["search", "search", "fetch", "final"]
    assert episode["reward"] == 1.0


def test_teacher_never_fabricates_a_successful_search(monkeypatch, teacher_transport):
    monkeypatch.setattr(tools, "_request", lambda *args, **kwargs: "No results")
    with pytest.raises(RuntimeError, match="no observed source"):
        asyncio.run(teacher_function()(registry_tasks(spec(), CSV)[0]))
    assert teacher_transport == []


def test_teacher_pages_through_complete_source(monkeypatch, teacher_transport):
    long_source = CSV + "#" * 15000
    monkeypatch.setattr(tools, "_download_public", lambda _: (long_source.encode(), "text/csv", "utf-8"))
    records, episode = asyncio.run(teacher_function()(registry_tasks(spec(), CSV)[0]))
    assert [record["stage"] for record in records] == ["search", "fetch", "fetch", "final"]
    assert len(episode["tool_events"]) == 3 and episode["reward"] == 1.0


def test_teacher_page_overlap_preserves_cells_across_source_boundary(monkeypatch, teacher_transport):
    prefix = "Value,Description,Reference\n99,"
    suffix = ",rfc\n100,Continue,rfc\n101,"
    name = "Boundary Verified Source Name"
    padding = 14000 - 10 - len(prefix) - len(suffix)
    source = prefix + "x" * padding + suffix + name + ",rfc\n"
    assert source.index(name) == 13990
    monkeypatch.setattr(tools, "_download_public", lambda _: (source.encode(), "text/csv", "utf-8"))
    records, episode = asyncio.run(teacher_function()(registry_tasks(spec(), source)[0]))
    fetches = [event for event in episode["tool_events"] if event["tool"] == "fetch_url"]
    assert [event["arguments"]["start"] for event in fetches] == [0, 13000]
    assert name not in fetches[0]["observation"]["text"]
    assert name in fetches[1]["observation"]["text"]
    path = Path(__file__).resolve().parents[1] / "scripts/dcc_tool_sft.py"
    validate = runpy.run_path(str(path))["validate_record"]
    assert validate(records[-1]) is records[-1]
