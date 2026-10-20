"""CPU checks for atomic completion publication and evaluation cleanup."""
import asyncio
import json
from pathlib import Path
import runpy
from types import SimpleNamespace

import pytest


@pytest.fixture
def publication():
    namespace = runpy.run_path(str(Path(__file__).parents[1] / "scripts/dcc_model_evaluate.py"))
    return namespace["_publish_completion_marker"], namespace["evaluate"]


def test_complete_marker_becomes_visible_only_at_atomic_replace(publication, tmp_path, monkeypatch):
    publish, _ = publication
    marker = tmp_path / "rank0.done"
    previous = b'{"tasks":1,"world_size":1}\n'
    marker.write_bytes(previous)
    completion = {"tasks": 2, "world_size": 1}
    events = []
    real_sync, real_replace = publish.__globals__["os"].fsync, Path.replace

    def sync(descriptor):
        events.append("sync")
        return real_sync(descriptor)

    def replace(temporary, destination):
        assert events == ["sync"]
        assert destination == marker
        assert marker.read_bytes() == previous
        assert json.loads(temporary.read_bytes()) == completion
        assert temporary.parent == marker.parent
        events.append("replace")
        return real_replace(temporary, destination)

    monkeypatch.setattr(publish.__globals__["os"], "fsync", sync)
    monkeypatch.setattr(Path, "replace", replace)
    publish(marker, completion)
    assert events == ["sync", "replace"]
    assert json.loads(marker.read_bytes()) == completion
    assert marker.read_bytes().endswith(b"\n")
    assert list(tmp_path.glob(".*.tmp")) == []


def test_sync_failure_propagates_and_retains_unpublished_temporary(publication, tmp_path, monkeypatch):
    publish, _ = publication
    marker = tmp_path / "rank0.done"
    failure = OSError("synthetic sync failure")

    def fail_sync(descriptor):
        raise failure

    monkeypatch.setattr(publish.__globals__["os"], "fsync", fail_sync)
    with pytest.raises(OSError) as caught:
        publish(marker, {"tasks": 2, "world_size": 1})
    assert caught.value is failure
    assert not marker.exists()
    temporary, = tmp_path.glob(".*.tmp")
    assert json.loads(temporary.read_bytes()) == {"tasks": 2, "world_size": 1}


def test_replace_failure_preserves_existing_complete_marker(publication, tmp_path, monkeypatch):
    publish, _ = publication
    marker = tmp_path / "rank0.done"
    previous = b'{"tasks":1,"world_size":1}\n'
    marker.write_bytes(previous)
    failure = OSError("synthetic replace failure")

    def fail_replace(temporary, destination):
        raise failure

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError) as caught:
        publish(marker, {"tasks": 2, "world_size": 1})
    assert caught.value is failure
    assert marker.read_bytes() == previous
    temporary, = tmp_path.glob(".*.tmp")
    assert json.loads(temporary.read_bytes()) == {"tasks": 2, "world_size": 1}


def test_temporary_and_published_readback_mismatches_propagate(publication, tmp_path, monkeypatch):
    publish, _ = publication
    real_read = Path.read_bytes
    for phase in ("temporary", "published"):
        directory = tmp_path / phase
        directory.mkdir()
        marker = directory / "rank0.done"

        def mismatched(path):
            if (phase == "published" and path == marker) or (phase == "temporary" and path != marker):
                return b""
            return real_read(path)

        with monkeypatch.context() as patch:
            patch.setattr(Path, "read_bytes", mismatched)
            with pytest.raises(OSError, match="readback differs"):
                publish(marker, {"tasks": 2, "world_size": 1})
        assert marker.exists() is (phase == "published")
        if phase == "temporary":
            assert len(list(directory.glob(".*.tmp"))) == 1


def _evaluation_doubles(monkeypatch, evaluate, marker):
    events = []
    state = SimpleNamespace(evaluation_error=None, cleanup_error=None)

    class Reader:
        def __init__(self, *, enabled):
            assert enabled is True

        async def start(self):
            events.append("start")
            return {}

        async def close(self):
            events.append("close")
            assert not marker.exists()
            if state.cleanup_error is not None:
                raise state.cleanup_error

    async def episodes(args, *, browser_reader):
        events.append("evaluate")
        if state.evaluation_error is not None:
            raise state.evaluation_error
        return marker, {"tasks": 2, "world_size": 1}

    monkeypatch.setitem(evaluate.__globals__, "BrowserReader", Reader)
    monkeypatch.setitem(evaluate.__globals__, "validate_browser_reader_runtime", lambda runtime: None)
    monkeypatch.setitem(evaluate.__globals__, "_evaluate", episodes)
    args = SimpleNamespace(web_browser_reader=True, inference_backend="vllm")
    return args, events, state


def test_evaluation_cleans_up_before_publication_and_propagates_publication_error(publication, tmp_path, monkeypatch):
    publish, evaluate = publication
    marker = tmp_path / "rank0.done"
    args, events, _ = _evaluation_doubles(monkeypatch, evaluate, marker)

    def checked_publish(path, completion):
        assert events == ["start", "evaluate", "close"]
        events.append("publish")
        return publish(path, completion)

    monkeypatch.setitem(evaluate.__globals__, "_publish_completion_marker", checked_publish)
    asyncio.run(evaluate(args))
    assert events == ["start", "evaluate", "close", "publish"]
    assert json.loads(marker.read_bytes()) == {"tasks": 2, "world_size": 1}
    marker.unlink()
    events.clear()
    failure = OSError("synthetic publication failure")

    def failed_publish(path, completion):
        assert events == ["start", "evaluate", "close"]
        raise failure

    monkeypatch.setitem(evaluate.__globals__, "_publish_completion_marker", failed_publish)
    with pytest.raises(OSError) as caught:
        asyncio.run(evaluate(args))
    assert caught.value is failure
    assert not marker.exists()


def test_evaluation_cleanup_failure_blocks_publication_and_preserves_primary_error(publication, tmp_path, monkeypatch):
    _, evaluate = publication
    marker = tmp_path / "rank0.done"
    args, events, state = _evaluation_doubles(monkeypatch, evaluate, marker)
    state.cleanup_error = OSError("synthetic cleanup failure")

    def unexpected_publish(path, completion):
        pytest.fail("Completion published despite evaluation cleanup failure")

    monkeypatch.setitem(evaluate.__globals__, "_publish_completion_marker", unexpected_publish)
    with pytest.raises(OSError) as caught:
        asyncio.run(evaluate(args))
    assert caught.value is state.cleanup_error
    assert events == ["start", "evaluate", "close"]
    assert not marker.exists()
    events.clear()
    state.evaluation_error = RuntimeError("synthetic evaluation failure")
    with pytest.raises(RuntimeError) as caught:
        asyncio.run(evaluate(args))
    assert caught.value is state.evaluation_error
    assert any("Browser cleanup also failed: OSError" in note for note in caught.value.__notes__)
    assert events == ["start", "evaluate", "close"]
    assert not marker.exists()
