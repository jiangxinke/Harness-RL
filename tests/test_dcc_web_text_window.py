"""Body-window limits preserve literal cache, selector budgets and identities."""
import asyncio
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from slime_plugins.dcc.data import Task
from slime_plugins.dcc.tools import WebTools


URL = "https://example.org/synthetic-train-table"


@pytest.mark.parametrize("budget", [4, 14000])
def test_default_window_inherits_original_page_budget(budget):
    web = WebTools(max_page_chars=budget)
    web.pages[URL] = "a" * (budget + 1)
    result = asyncio.run(web.fetch_url(URL))
    assert web.max_text_window_chars == web.max_page_chars == budget
    assert result["text"] == web.pages[URL][:budget]
    assert result["next_start"] == budget


def test_body_window_preserves_full_source_offsets_and_navigation():
    web = WebTools(max_text_window_chars=6000)
    source = "a" * 5999 + "中文" + "b" * 6000
    web.pages[URL] = source
    web.page_links[URL] = [{"url": "https://example.org/next", "text": "next source"}]
    web.page_worksheets[URL] = [{"name": "synthetic", "start": 0, "end": len(source)}]
    web.page_archives[URL] = [{"name": "synthetic.csv", "status": "readable"}]
    first = asyncio.run(web.fetch_url(URL))
    second = asyncio.run(web.fetch_url(URL, start=first["next_start"]))
    last = asyncio.run(web.fetch_url(URL, start=second["next_start"]))
    assert web.pages[URL] == source
    assert "".join(result["text"] for result in (first, second, last)) == source
    for result in (first, second, last):
        assert result["text"] == source[result["start"]:result["end"]]
        assert len(result["text"]) <= 6000
        assert result["total_characters"] == len(source)
        assert result["outgoing_links"] == web.page_links[URL]
        assert result["worksheets"] == web.page_worksheets[URL]
        assert result["archive_members"] == web.page_archives[URL]
    assert first["text"].endswith("中") and second["text"].startswith("文")
    assert first["start"] == 0 and first["end"] == second["start"] == 6000
    assert last["next_start"] is None and last["truncated"] is False
    repeated = asyncio.run(web.fetch_url(URL))
    assert repeated["text"] == first["text"]
    assert repeated["read_progress"]["first_read"] == {"url": URL, "start": 0, "end": 6000}


def test_cached_selector_retains_original_json_budget_and_unseen_source_offsets():
    source = "id,name\n" + "".join(f"{index},{character * 3000}\n"
                                     for index, character in enumerate("xyz"))
    request = {"format": "csv", "columns": ["id", "name"],
               "where": [{"column": "id", "op": "eq", "value": "2"}]}
    default = WebTools(web_cached_rows=True)
    limited = WebTools(web_cached_rows=True, max_text_window_chars=6000)
    for web in (default, limited):
        web.pages[URL] = source
    original = asyncio.run(default.fetch_url(URL, select_rows=request))
    selected = asyncio.run(limited.fetch_url(URL, select_rows=request))
    assert selected == original and "error" not in selected
    assert 6000 < len(json.dumps(selected, ensure_ascii=False)) <= 14000
    row = selected["row_view"]["records"][0]
    assert row["start"] > 6000
    assert source[row["start"]:row["end"]] == row["text"] == "2," + "z" * 3000 + "\n"
    assert limited.pages[URL] == source
    assert limited.max_page_chars == 14000 and limited.max_text_window_chars == 6000


@pytest.mark.parametrize("value", [True, False, 0, -1, 1.5, "6000"])
def test_text_window_requires_a_positive_plain_integer(value):
    with pytest.raises(ValueError, match="positive integer"):
        WebTools(max_text_window_chars=value)


@pytest.fixture(scope="module")
def scripts():
    modules = {}
    for name in ("dcc_online_grpo", "dcc_online_evaluate", "dcc_online_web_evaluate"):
        spec = importlib.util.spec_from_file_location(name + "_text_window_tests",
            Path(__file__).parents[1] / "scripts" / (name + ".py"))
        value = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(value)
        modules[name] = value
    return modules


@pytest.mark.parametrize("requested,effective", [(None, 14000), (6000, 6000)])
def test_shared_training_and_native_web_runner_record_actual_window(scripts, requested, effective):
    evaluator, trainer = scripts["dcc_online_web_evaluate"], scripts["dcc_online_grpo"]
    command = ["--model", "synthetic-CPU-model", "--data", "unused-synthetic.jsonl",
               "--output-dir", "unused-synthetic-output", "--source-commit", "0" * 40]
    if requested is not None:
        command += ["--web-text-window-chars", str(requested)]
    args = evaluator.parser().parse_args(command)
    args.group_size = 4
    task = Task("synthetic-train", "widesearch", "train", "Read public source", gold=None)
    train_config = trainer.configuration(args, [[0]], [task], 1)
    runner = trainer.make_runner(args)
    actual = runner.web_tools_factory()
    web_config = evaluator.runner_configuration(args)["web_tools"]
    assert actual.max_text_window_chars == web_config["max_text_window_chars"] == effective
    assert actual.max_page_chars == web_config["max_page_chars"] == 14000
    assert train_config["web_tool_limits"] == {
        "max_page_chars": 14000, "max_text_window_chars": effective}
    assert args.auto_read_sources == runner.auto_read_sources == 2
    del args.web_text_window_chars
    assert trainer.make_runner(args).web_tools_factory().max_text_window_chars == 14000
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("name", ["dcc_online_grpo", "dcc_online_evaluate", "dcc_online_web_evaluate"])
@pytest.mark.parametrize("invalid", ["0", "-1", "1.5", "true"])
def test_cli_rejects_invalid_windows_before_data_or_model(scripts, monkeypatch, name, invalid):
    script = scripts[name]
    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid body-window limits must fail before source/data/model work")
    for attribute in ("validate_source", "load_tasks", "load_public_tasks", "load_policy_model"):
        if hasattr(script, attribute):
            monkeypatch.setattr(script, attribute, forbidden)
    data_arg = "--train-data" if name == "dcc_online_grpo" else "--data"
    with pytest.raises(SystemExit) as result:
        script.main(["--model", "synthetic-CPU-model", data_arg, "unused-synthetic.jsonl",
                     "--output-dir", "unused-synthetic-output", "--source-commit", "0" * 40,
                     "--web-text-window-chars", invalid])
    assert result.value.code == 2 and not torch.cuda.is_initialized()
