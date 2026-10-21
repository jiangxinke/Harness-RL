"""CPU-only contracts for the optional Web reasoning prompt."""
import asyncio
import importlib.util
import json
from pathlib import Path
import runpy

import pytest

from slime_plugins.dcc import EpisodeRunner, Task
from slime_plugins.dcc.web_reasoning import (
    WEB_CONCISE_REASONING_INSTRUCTIONS, web_concise_reasoning_configuration,
)

ROOT = Path(__file__).parents[1]
REVIEW = runpy.run_path(str(ROOT / "tests/test_dcc_web_explicit_review.py"))
NAV = runpy.run_path(str(ROOT / "tests/test_dcc_web_navigation.py"))
Policy, Web, task = REVIEW["Policy"], NAV["Web"], REVIEW["task"]
URL = NAV["URL"]


@pytest.mark.parametrize("invalid", [0, 1, None, "true", [], {}])
def test_switch_requires_boolean(invalid):
    with pytest.raises(ValueError, match="web_concise_reasoning must be a boolean"):
        EpisodeRunner(web_concise_reasoning=invalid)


def test_default_off_preserves_real_runner_calls_and_on_only_appends_web_prompt():
    actions = [{"tool": "web_search", "arguments": {"query": "invented public records"}},
        {"tool": "fetch_url", "arguments": {"url": URL}},
        {"answer": [{"name": "Alpha", "citations": [URL]}]}]
    policies, results = [], []
    for enabled in [None, False, True]:
        policy = Policy(actions)
        options = {} if enabled is None else {"web_concise_reasoning": enabled}
        runner = EpisodeRunner(max_steps=3, max_new_tokens=73, web_tools_factory=Web,
            web_structured_final=True, web_structured_actions=True, **options)
        result = asyncio.run(runner.run(task(), policy, evaluation=True, score_train=False))
        assert result.status == "completed"
        policies.append(policy); results.append(result)
    assert policies[0].calls == policies[1].calls
    assert results[0].to_dict() == results[1].to_dict()
    assert results[0].tool_events == results[2].tool_events
    assert results[0].answer == results[2].answer
    for before, after in zip(policies[0].calls, policies[2].calls, strict=True):
        assert before[2] == after[2]  # Exact original schema, sampling and token budget.
        assert after[1][0]["content"] == before[1][0]["content"] + WEB_CONCISE_REASONING_INSTRUCTIONS
        assert after[1][1:] == before[1][1:]
        assert after[1][0]["content"].count(WEB_CONCISE_REASONING_INSTRUCTIONS) == 1


@pytest.mark.parametrize("cuad_concise", [False, True])
def test_web_flag_never_changes_cuad_messages(cuad_concise):
    public_task = Task("invented-contract", "cuad", "train", "Extract clauses", document="A clause.")
    options = dict(initial_chunks=1, cuad_concise_reasoning=cuad_concise)
    assert EpisodeRunner(**options, web_concise_reasoning=True).initial_messages(public_task) == (
        EpisodeRunner(**options).initial_messages(public_task))


def test_source_failure_and_repeated_empty_answers_remain_failures():
    class ErrorWeb(Web):
        async def fetch_url(self, url, start=0):
            return {"url": url, "error": "Invented source request rejected"}
    actions = [{"tool": "web_search", "arguments": {"query": "invented records"}},
        {"tool": "fetch_url", "arguments": {"url": URL}}, {"answer": []}, {"answer": []}]
    results = []
    for enabled in (False, True):
        result = asyncio.run(EpisodeRunner(max_steps=4, web_tools_factory=ErrorWeb,
            web_structured_final=True, web_structured_actions=True,
            web_concise_reasoning=enabled).run(task(), Policy(actions), evaluation=True))
        assert result.status == "evidence_failure" and "empty_answer" in result.detail
        assert len(result.steps) == 4 and len(result.tool_events) == 2
        results.append(result)
    assert results[0].tool_events == results[1].tool_events
    assert results[0].metrics == results[1].metrics


@pytest.fixture(scope="module")
def scripts():
    result = {}
    for name in ("dcc_online_grpo", "dcc_online_evaluate", "dcc_online_web_evaluate"):
        spec = importlib.util.spec_from_file_location(name + "_web_reasoning_tests",
            ROOT / "scripts" / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        result[name] = module
    return result


def training_arguments(trainer, monkeypatch, *, enabled):
    command = ["--model", "synthetic-CPU-model", "--train-data", "unused.jsonl",
               "--output-dir", "unused-output", "--source-commit", "0" * 40]
    if enabled:
        command.append("--web-concise-reasoning")
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
@pytest.mark.parametrize("online_web", [False, True])
def test_cli_runner_and_versioned_configuration_are_consistent(scripts, monkeypatch, enabled, online_web):
    import torch
    trainer, evaluator = scripts["dcc_online_grpo"], scripts["dcc_online_web_evaluate"]
    args = training_arguments(trainer, monkeypatch, enabled=enabled)
    args.enable_online_web = online_web
    public_task = Task("synthetic-train", "cuad", "train", "Find a clause", document="A clause.")
    config = trainer.configuration(args, [[0]], [public_task], 1)
    assert trainer.make_runner(args).web_concise_reasoning is enabled
    expected = web_concise_reasoning_configuration(True)
    if enabled and online_web:
        assert config["web_concise_reasoning"] == expected
    else:
        assert "web_concise_reasoning" not in config
    if enabled:
        assert config["arguments"]["web_concise_reasoning"] is True
    else:
        assert "web_concise_reasoning" not in config["arguments"]
        del args.web_concise_reasoning
        assert json.dumps(config) == json.dumps(trainer.configuration(args, [[0]], [public_task], 1))
        assert trainer.make_runner(args).web_concise_reasoning is False
    command = ["--model", "synthetic-CPU-model", "--data", "unused.jsonl",
               "--output-dir", "unused-output", "--source-commit", "0" * 40]
    if enabled:
        command += ["--web-concise-reasoning"]
    for name in ("dcc_online_evaluate", "dcc_online_web_evaluate"):
        evaluation_args = scripts[name].parser().parse_args(command)
        assert evaluation_args.web_concise_reasoning is enabled
        if name == "dcc_online_web_evaluate":
            harness = evaluator.runner_configuration(evaluation_args)
            if enabled:
                assert harness["web_concise_reasoning"] == expected
            else:
                assert "web_concise_reasoning" not in harness
                del evaluation_args.web_concise_reasoning
                assert json.dumps(harness) == json.dumps(evaluator.runner_configuration(evaluation_args))
    assert not torch.cuda.is_initialized()
