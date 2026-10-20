"""CPU doubles verify opt-in evaluation lifecycle without browser or model loads."""
import asyncio
from copy import deepcopy
import json
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

import pytest

from slime_plugins.dcc import PolicyCompletion, Task
from slime_plugins.dcc.browser_reader import BROWSER_READER_VERSION, browser_reader_configuration


@pytest.fixture
def evaluation(monkeypatch, tmp_path):
    evaluate = runpy.run_path(str(Path(__file__).parents[1] / "scripts/dcc_model_evaluate.py"))["evaluate"]
    events, readers, tools = [], [], []
    failure = RuntimeError("synthetic evaluation failure")
    state = SimpleNamespace(fail_at=None, cleanup_error=None)
    runtime = {"version": BROWSER_READER_VERSION, "browser_version": "142.0.7444.23",
               "playwright_version": "1.56.0", "cpu_only": True,
               "configuration": browser_reader_configuration(True)}

    class Reader:
        def __init__(self, *, enabled):
            assert enabled is True
            self.runtime = None
            readers.append(self)

        async def start(self):
            events.append("browser_start")
            assert not list(tmp_path.glob("manifest*.json"))
            if state.fail_at == "start":
                raise failure
            self.runtime = deepcopy(runtime)
            return self.runtime

        async def close(self):
            events.append("browser_close")
            assert not (tmp_path / "rank0.done").exists()
            if state.fail_at == "close":
                raise failure
            if state.cleanup_error is not None:
                raise state.cleanup_error

    class Tools:
        backend = "synthetic"

        def __init__(self, *, browser_reader=None):
            if browser_reader is not None:
                tools.append(browser_reader)

    class Policy:
        provenance = {"backend": "vllm", "model": "example/model"}

        async def __call__(self, *args, **kwargs):
            events.append("episode")
            if state.fail_at == "episode":
                raise failure
            return PolicyCompletion('{"answer":[]}', [1], [2])

        async def close(self):
            events.append("policy_close")

    def load(*args, **kwargs):
        events.append("policy_load")
        assert (tmp_path / "manifest_rank0.json").is_file()
        if readers:
            assert events.index("browser_start") < events.index("policy_load")
        if state.fail_at == "load":
            raise failure
        return Policy()

    tasks = [Task(f"web{index}", "widesearch", "validation", "Find names", gold=[],
                  metadata={"evaluation": {"required": ["name"]}}) for index in range(2)]
    monkeypatch.setitem(evaluate.__globals__, "BrowserReader", Reader)
    monkeypatch.setitem(evaluate.__globals__, "WebTools", Tools)
    monkeypatch.setitem(evaluate.__globals__, "load_tasks", lambda path: tasks)
    monkeypatch.setitem(sys.modules, "slime_plugins.dcc.vllm_policy", SimpleNamespace(load_vllm_policy=load))
    for name, value in (("RANK", "0"), ("LOCAL_RANK", "0"), ("WORLD_SIZE", "1")):
        monkeypatch.setenv(name, value)
    args = SimpleNamespace(model="example/model", adapter=None, data=["synthetic.jsonl"],
                           max_steps=1, max_new_tokens=100, max_prompt_tokens=1024,
                           initial_chunks=4, seed=42, limit=0, output_dir=str(tmp_path),
                           retry_infrastructure=False, inference_backend="vllm", quantization="none",
                           vllm_tensor_parallel_size=1, vllm_memory_utilization=.88,
                           vllm_batched_tokens=8192, vllm_enforce_eager=True,
                           batch_size=1, timeout=30, save_tokens=False, web_browser_reader=True)
    return SimpleNamespace(evaluate=evaluate, args=args, state=state, events=events,
                           readers=readers, tools=tools, runtime=runtime, failure=failure, output=tmp_path)


def test_enabled_reader_preflights_once_shares_across_episodes_and_records_runtime(evaluation):
    e = evaluation
    asyncio.run(e.evaluate(e.args))
    assert len(e.readers) == 1 and e.tools == [e.readers[0], e.readers[0]]
    assert e.events[-2:] == ["policy_close", "browser_close"]
    for name in ("manifest_rank0.json", "inference_runtime.json"):
        assert json.loads((e.output / name).read_text())["web_browser_reader"] == e.runtime
    assert json.loads((e.output / "rank0.done").read_text()) == {"tasks": 2, "world_size": 1}


def test_disabled_reader_keeps_legacy_metadata_and_does_not_construct_browser(evaluation):
    e = evaluation
    e.args.web_browser_reader = False
    asyncio.run(e.evaluate(e.args))
    assert e.readers == [] and e.tools == []
    for name in ("manifest_rank0.json", "inference_runtime.json"):
        assert "web_browser_reader" not in json.loads((e.output / name).read_text())


@pytest.mark.parametrize("phase", ["start", "load", "episode", "close"])
def test_reader_closes_after_failures_without_certifying_completion(evaluation, phase):
    e = evaluation
    e.state.fail_at = phase
    with pytest.raises(RuntimeError) as caught:
        asyncio.run(e.evaluate(e.args))
    assert caught.value is e.failure
    assert e.events.count("browser_close") == 1
    assert not (e.output / "rank0.done").exists()
    if phase == "start":
        assert not list(e.output.glob("manifest*.json"))
        assert "policy_load" not in e.events
    if phase == "episode":
        assert e.events[-2:] == ["policy_close", "browser_close"]


def test_reader_cleanup_preserves_primary_preflight_failure(evaluation):
    e = evaluation
    e.state.fail_at = "start"
    e.state.cleanup_error = OSError("synthetic cleanup failure")
    with pytest.raises(RuntimeError) as caught:
        asyncio.run(e.evaluate(e.args))
    assert caught.value is e.failure
    assert any("Browser cleanup also failed: OSError" in note for note in e.failure.__notes__)


@pytest.mark.parametrize("args,message", [
    ({"web_browser_reader": True}, "requires --inference-backend vllm"),
    ({"web_browser_reader": 1}, "must be a boolean"),
])
def test_browser_option_rejects_unsupported_configuration_before_start(evaluation, args, message):
    with pytest.raises(ValueError, match=message):
        asyncio.run(evaluation.evaluate(SimpleNamespace(**args)))
    assert evaluation.readers == []
