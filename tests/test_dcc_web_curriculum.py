from __future__ import annotations

import asyncio
import dataclasses
import runpy
from pathlib import Path

import pytest

from slime_plugins.dcc import tools
from slime_plugins.dcc.web_curriculum import REGISTRIES, Registry, registry_tasks


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
