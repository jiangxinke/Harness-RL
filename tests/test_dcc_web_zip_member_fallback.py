"""Real synthetic ZIP bytes: character-only fallback, evidence and CLI wiring."""
from __future__ import annotations

import asyncio
import importlib.util
import io
import json
from pathlib import Path
import stat
import struct
import zipfile

import pytest

from slime_plugins.dcc import tools, web_archive
from slime_plugins.dcc.episode import _WebEvidence

FIXTURES = Path(__file__).parent / "fixtures" / "dcc_zip_member_fallback"
URL = "https://example.org/synthetic-export.zip"
SMALL = "code,name\nX1,Example\n"


def zipped(*members):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, content in members:
            archive.writestr(name, content)
    return stream.getvalue()


def extract(data, *, enabled=True, sections=None):
    return tools.extract_page(data, "application/zip", archive_sections=sections,
                              web_zip_member_fallback=enabled)


def test_default_atomic_failure_and_opt_in_whole_member_recovery(monkeypatch):
    monkeypatch.setattr(web_archive, "_MAX_TEXT_CHARS", 512)
    data = (FIXTURES / "oversized-metadata-and-small-table.zip").read_bytes()
    original = []
    with pytest.raises(ValueError, match="metadata.csv.*character limit") as error:
        extract(data, enabled=False, sections=original)
    assert type(error.value) is ValueError
    assert type(error.value.__cause__) is ValueError
    assert original == []
    sections = []
    text = extract(data, sections=sections)
    skipped, table = sections
    assert skipped["name"] == "metadata.csv"
    assert skipped["status"] == "unexposed_text_limit"
    assert skipped["reason"] == "member_text_character_limit"
    assert skipped["text_exposed"] is False
    assert not {"start", "end", "content_start", "content_end", "preview"}.intersection(skipped)
    assert table["status"] == "readable"
    assert text[table["content_start"]:table["content_end"]] == SMALL
    assert "not an evidence row" not in text and len(text) <= 512


def test_original_one_million_character_cap_without_lowering_limits():
    data = zipped(("metadata.csv", "x" * 1_000_001), ("rows.csv", SMALL))
    sections = []
    text = extract(data, sections=sections)
    assert SMALL in text and sections[0]["reason"] == "member_text_character_limit"
    with pytest.raises(ValueError, match="character limit"):
        extract(data, enabled=False)


def test_total_budget_is_checked_before_member_write_and_later_small_member_fits(monkeypatch):
    monkeypatch.setattr(web_archive, "_MAX_TEXT_CHARS", 512)
    data = (FIXTURES / "cumulative-text-cap.zip").read_bytes()
    with pytest.raises(ValueError, match="extracted text exceeds"):
        extract(data, enabled=False)
    sections = []
    text = extract(data, sections=sections)
    assert [s["status"] for s in sections] == ["readable", "unexposed_text_limit", "readable"]
    assert sections[1]["reason"] == "archive_text_character_limit"
    assert "Second row" not in text and "second.csv" not in text
    assert len(text) <= 512
    for section in (sections[0], sections[2]):
        content = text[section["content_start"]:section["content_end"]]
        assert content.endswith("\n") and content.startswith("code,name\n")
        assert text[section["content_end"]:section["end"]] == "\n"


def test_all_members_skipped_is_failed_read_without_navigation(monkeypatch):
    monkeypatch.setattr(web_archive, "_MAX_TEXT_CHARS", 128)
    sections = []
    with pytest.raises(ValueError, match="no supported readable text"):
        extract(zipped(("metadata.csv", "x" * 129)), sections=sections)
    assert sections == []


@pytest.mark.parametrize("payload", [b"", b" \n\t"])
def test_empty_retained_member_cannot_rescue_all_skipped_export(monkeypatch, payload):
    monkeypatch.setattr(web_archive, "_MAX_TEXT_CHARS", 256)
    with pytest.raises(ValueError, match="no supported readable text"):
        extract(zipped(("metadata.csv", "x" * 257), ("empty.csv", payload)))


@pytest.mark.parametrize("attribute,value,match", [
    ("_MAX_DOWNLOAD_BYTES", 20, "download.*byte limit"),
    ("_MAX_MEMBERS", 1, "member limit"),
    ("_MAX_MEMBER_BYTES", 4, "member byte limit"),
    ("_MAX_TOTAL_BYTES", 20, "total expanded byte limit"),
    ("_MAX_COMPRESSION_RATIO", 0, "compression-ratio"),
])
def test_fallback_does_not_relax_archive_size_limits(monkeypatch, attribute, value, match):
    monkeypatch.setattr(web_archive, attribute, value)
    sections = []
    with pytest.raises(ValueError, match=match):
        extract(zipped(("good.csv", SMALL), ("other.csv", SMALL)), sections=sections)
    assert sections == []


@pytest.mark.parametrize("name,payload,match", [
    ("../unsafe.csv", b"code,name\nA,B\n", "unsafe member name"),
    ("nested.zip", b"literal", "nested ZIP/archive"),
    ("nested.csv", b"PK\x03\x04not-an-archive", "nested ZIP/archive"),
    ("bad.csv", b"bad\x00binary", "binary control"),
    ("bad.csv", b"name\n\xff\n", "text encoding"),
    ("bad.json", b"[{broken]", "not well-formed"),
    ("bad.xml", b"<broken>", "not well-formed"),
    ("entity.xml", b'<!DOCTYPE a [<!ENTITY b "c">]><a>&b;</a>', "entity"),
])
def test_non_character_failures_are_still_atomic_after_skipped_member(monkeypatch, name, payload, match):
    monkeypatch.setattr(web_archive, "_MAX_TEXT_CHARS", 256)
    sections = []
    with pytest.raises(ValueError, match=match):
        extract(zipped(("large.csv", "x" * 257), ("good.csv", SMALL), (name, payload)), sections=sections)
    assert sections == []


def test_crc_failure_in_oversized_member_cannot_be_skipped(monkeypatch):
    monkeypatch.setattr(web_archive, "_MAX_TEXT_CHARS", 128)
    data = bytearray(zipped(("large.csv", "x" * 129), ("good.csv", SMALL)))
    data[data.index(b"x" * 129)] = ord("y")
    with pytest.raises(ValueError, match="large.csv") as error:
        extract(bytes(data))
    assert isinstance(error.value.__cause__, zipfile.BadZipFile)


def test_encrypted_member_cannot_be_skipped(monkeypatch):
    monkeypatch.setattr(web_archive, "_MAX_TEXT_CHARS", 128)
    data = bytearray(zipped(("large.csv", "x" * 129), ("good.csv", SMALL)))
    for marker, offset in [(b"PK\x03\x04", 6), (b"PK\x01\x02", 8)]:
        struct.pack_into("<H", data, data.index(marker) + offset, 1)
    with pytest.raises(ValueError, match="encrypted"):
        extract(bytes(data))


def test_special_member_and_invalid_directory_are_not_recovered():
    member = zipfile.ZipInfo("rows.csv")
    member.create_system = 3
    member.external_attr = (stat.S_IFLNK | 0o777) << 16
    with pytest.raises(ValueError, match="symbolic link or special"):
        extract(zipped((member, SMALL)))
    data = bytearray(zipped(("rows.csv", SMALL)))
    end = data.rfind(b"PK\x05\x06")
    struct.pack_into("<H", data, end + 8, 0)
    struct.pack_into("<H", data, end + 10, 0)
    with pytest.raises(ValueError, match="member count"):
        extract(bytes(data))


def test_bad_json_after_total_budget_is_not_hidden_by_budget_skip(monkeypatch):
    monkeypatch.setattr(web_archive, "_MAX_TEXT_CHARS", 128)
    with pytest.raises(ValueError, match="not well-formed"):
        extract(zipped(("first.csv", "code,name\nX1," + "a" * 30 + "\n"),
                       ("bad.json", "[" + "a" * 90)))


@pytest.mark.parametrize("name,payload,match", [
    ("bad.json", "[" + "a" * 256, "JSON member is not well-formed"),
    ("bad.xml", "<a>" + "a" * 256, "XML member is not well-formed"),
    ("entity.xml", '<!DOCTYPE a [<!ENTITY b "c">]><a>' + "a" * 256 + "</a>", "entity"),
])
def test_per_member_character_cap_cannot_mask_structured_validation(monkeypatch, name, payload, match):
    monkeypatch.setattr(web_archive, "_MAX_TEXT_CHARS", 128)
    data = zipped((name, payload), ("rows.csv", SMALL))
    sections = []
    with pytest.raises(ValueError, match=match):
        extract(data, sections=sections)
    assert sections == []
    # Default rejection order and original cause type remain unchanged.
    with pytest.raises(ValueError, match="character limit") as error:
        extract(data, enabled=False)
    assert type(error.value.__cause__) is ValueError


@pytest.mark.parametrize("value", [0, 1, None, "true", [], {}])
def test_python_flag_requires_plain_boolean(value):
    with pytest.raises(ValueError, match="must be a boolean"):
        tools.WebTools(web_zip_member_fallback=value)


def fake_download(monkeypatch, data):
    from slime_plugins.dcc import search_provider
    monkeypatch.setattr(search_provider, "wikipedia_page", lambda url: None)
    calls = []
    def download(url):
        calls.append(url)
        return data, "application/zip", "utf-8"
    monkeypatch.setattr(tools, "_download_public", download)
    return calls


def test_navigation_only_does_not_ground_and_selectors_cannot_read_skipped_member(monkeypatch):
    monkeypatch.setattr(web_archive, "_MAX_TEXT_CHARS", 512)
    calls = fake_download(monkeypatch, (FIXTURES / "oversized-metadata-and-small-table.zip").read_bytes())
    async def run():
        web = tools.WebTools(web_zip_member_fallback=True, web_cached_rows=True, max_text_window_chars=20)
        evidence = _WebEvidence()
        first = await web.fetch_url(URL)
        assert first["source_text_exposed"] is False
        assert "archive_members" in first and not web.sources
        evidence.observe("fetch_url", first)
        assert evidence.fetches == 0 and not evidence.fetched_sources
        error = await web.fetch_url(URL, select_rows={"format": "csv", "archive_member": "metadata.csv", "columns": ["description"]})
        assert "error" in error and "text" not in error and "row_view" not in error
        evidence.observe("fetch_url", error)
        assert evidence.fetches == 0
        section = first["archive_members"][1]
        actual = await web.fetch_url(URL, start=section["content_start"])
        assert actual["source_text_exposed"] is True and actual["text"] == SMALL[:20]
        evidence.observe("fetch_url", actual)
        assert evidence.fetches == 1 and evidence.fetched_sources
        selected = await web.fetch_url(URL, select_rows={"format": "csv", "archive_member": "rows.csv", "columns": ["code", "name"]})
        assert selected["row_view"]["returned_records"] == 1
        row = selected["row_view"]["records"][0]
        assert web.pages[URL][row["start"]:row["end"]] == row["text"] == "X1,Example\n"
        assert calls == [URL]
    asyncio.run(run())


def test_all_skipped_web_read_creates_no_cached_page_or_grounding(monkeypatch):
    monkeypatch.setattr(web_archive, "_MAX_TEXT_CHARS", 128)
    fake_download(monkeypatch, zipped(("metadata.csv", "x" * 129)))
    web = tools.WebTools(web_zip_member_fallback=True, web_cached_rows=True)
    result = asyncio.run(web.fetch_url(URL))
    assert "no supported readable text" in result["error"]
    assert not web.pages and not web.page_archives and not web.sources
    evidence = _WebEvidence()
    evidence.observe("fetch_url", result)
    assert evidence.fetches == 0 and not evidence.fetched_sources


def test_healthy_zip_default_output_and_navigation_unchanged():
    data = zipped(("rows.csv", SMALL), ("notes.txt", "literal notes"))
    before, after = [], []
    assert extract(data, enabled=False, sections=before) == extract(data, sections=after)
    assert before == after


@pytest.fixture(scope="module")
def scripts():
    result = {}
    for name in ("dcc_online_grpo", "dcc_online_evaluate", "dcc_online_web_evaluate"):
        spec = importlib.util.spec_from_file_location(name + "_zip_fallback_tests",
            Path(__file__).parents[1] / "scripts" / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        result[name] = module
    return result


def training_arguments(trainer, monkeypatch, *, enabled=False):
    """Stop immediately after real CLI parsing, before data/model access."""
    command = ["--model", "synthetic-CPU-model", "--train-data", "unused-synthetic.jsonl",
               "--output-dir", "unused-synthetic-output", "--source-commit", "0" * 40]
    if enabled:
        command.append("--web-zip-member-fallback")
    original = trainer.argparse.ArgumentParser.parse_args
    captured = []
    class Parsed(Exception):
        pass
    def parse(parser, arguments=None, namespace=None):
        captured.append(original(parser, arguments, namespace))
        raise Parsed
    with monkeypatch.context() as scoped:
        scoped.setattr(trainer.argparse.ArgumentParser, "parse_args", parse)
        with pytest.raises(Parsed):
            trainer.main(command)
    return captured[0]


@pytest.mark.parametrize("enabled", [False, True])
def test_training_and_both_evaluation_parsers_record_effective_option(scripts, monkeypatch, enabled):
    import torch
    from slime_plugins.dcc.data import Task
    command = ["--model", "synthetic-CPU-model", "--data", "unused-synthetic.jsonl",
               "--output-dir", "unused-synthetic-output", "--source-commit", "0" * 40]
    if enabled:
        command += ["--web-zip-member-fallback"]
    for name in ("dcc_online_evaluate", "dcc_online_web_evaluate"):
        script = scripts[name]
        args = script.parser().parse_args(command)
        assert args.web_zip_member_fallback is enabled
    trainer, evaluator = scripts["dcc_online_grpo"], scripts["dcc_online_web_evaluate"]
    args = training_arguments(trainer, monkeypatch, enabled=enabled)
    assert args.web_zip_member_fallback is enabled
    args.enable_online_web = True
    task = Task("synthetic-train", "widesearch", "train", "Read public source", gold=None)
    train_config = trainer.configuration(args, [[0]], [task], 1)
    actual = trainer.make_runner(args).web_tools_factory()
    web_config = evaluator.runner_configuration(evaluator.parser().parse_args(command))["web_tools"]
    expected = web_archive.web_zip_member_fallback_configuration(enabled)
    assert actual.web_zip_member_fallback is enabled
    if enabled:
        assert train_config["web_zip_member_fallback"] == web_config["web_zip_member_fallback"] == expected
    else:
        assert "web_zip_member_fallback" not in train_config
        assert "web_zip_member_fallback" not in train_config["arguments"]
        assert "web_zip_member_fallback" not in web_config
    del args.web_zip_member_fallback
    assert trainer.make_runner(args).web_tools_factory().web_zip_member_fallback is False
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("online_web", [False, True])
@pytest.mark.parametrize("reference_kl_beta", [0, .02])
def test_default_configuration_identical_to_namespace_without_new_flag(
        scripts, monkeypatch, online_web, reference_kl_beta):
    trainer = scripts["dcc_online_grpo"]
    args = training_arguments(trainer, monkeypatch)
    args.enable_online_web = online_web
    args.reference_kl_beta = reference_kl_beta
    actual = trainer.configuration(args, [], [], 1)
    del args.web_zip_member_fallback
    original = trainer.configuration(args, [], [], 1)
    assert json.dumps(actual) == json.dumps(original)
    assert "web_zip_member_fallback" not in actual
    assert "web_zip_member_fallback" not in actual["arguments"]


def test_cuad_only_training_has_no_zip_fallback_configuration(scripts, monkeypatch):
    trainer = scripts["dcc_online_grpo"]
    args = training_arguments(trainer, monkeypatch, enabled=True)
    assert args.enable_online_web is False
    config = trainer.configuration(args, [], [], 1)
    assert "web_zip_member_fallback" not in config
