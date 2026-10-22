"""CPU async cancellation controls with invented phase tokens, no model/network."""
from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import asdict
import json
from threading import Event
from types import SimpleNamespace

import pytest

from test_dcc_hf_native_policy import ScriptedModel, final_ids, messages, tokenizer
from test_dcc_online_web import COMMIT, FixtureTools, inputs, public_row
from slime_plugins.dcc import online_web as web
from slime_plugins.dcc import web_train_thought_budget_close as train_close
from slime_plugins.dcc.data import Task
from slime_plugins.dcc.episode import EpisodeResult, EpisodeRunner
from slime_plugins.dcc.hf_native_policy import (
    CANCELLED_BEFORE_JSON, END_THINK, NativeHFPolicy, NativeHFRequestControl,
    NativeHFRequestExpiredBeforeSubmission,
)
from slime_plugins.dcc.online_grpo import group_signal
from slime_plugins.dcc.web_hf_evaluation import (
    OwnedEvaluationPolicy, prediction_row, verify_native_ownership,
)
from slime_plugins.dcc.web_structured import web_final_schema


THOUGHT = [10065, END_THINK]
SCORES = [-0.25, -0.5]


def native_fixture(tokenizer, *, batch_size=2, block="thought", capture=True, close_train=False):
    """Use the real queue, encoding, phase assembly, trace and owner paths."""
    model = ScriptedModel([], [])
    model.config = SimpleNamespace(vocab_size=151670, max_position_embeddings=65536)
    traces, controls, calls = [], [], []
    value = NativeHFPolicy(model, tokenizer, thought_tokens=4, json_tokens=256,
        max_prompt_tokens=32768, batch_size=batch_size, mixed_schema_batching=True,
        capture_thought_logprobs=capture, trace_sink=traces.append,
        web_train_thought_budget_close=close_train)
    started, release = Event(), Event()
    thought = [10065, 10065, 10065, END_THINK] if close_train else THOUGHT
    scores = [-.25, -.5, -.75, 0.] if close_train else SCORES

    def factory():
        control = NativeHFRequestControl()
        controls.append(control)
        return control

    def phase(prompts, *, phase, grammar=None, train_budget_close_rows=None):
        calls.append((phase, len(prompts)))
        if phase == block:
            started.set()
            assert release.wait(5), "CPU fixture phase was not released"
        if phase == "thought":
            value._last_thought_logprobs = [list(scores) for _ in prompts] if capture else None
            if close_train:
                assert train_budget_close_rows == [True] * len(prompts)
                value._last_thought_forced_positions = [[3] for _ in prompts]
            return [list(thought) for _ in prompts]
        return [final_ids(tokenizer, text='{"answer":[]}') for _ in prompts]

    value.new_request_control = factory
    value._phase = phase
    return SimpleNamespace(policy=value, traces=traces, controls=controls, calls=calls,
                           started=started, release=release, thought=thought, scores=scores)


def owned(fixture, kind, rows, *, sample=0):
    bound = fixture.policy.bind_task("invented-expiry", 0)
    if kind == "train":
        return web.OwnedEpisodePolicy(bound, task_id="invented-expiry", iteration=1,
                                     rank=0, sample=sample, sink=rows.append)
    return OwnedEvaluationPolicy(bound, task_id="invented-expiry", rank=0, sink=rows.append)


def request(owner, column="name"):
    return owner.generate_structured(messages(column), json_schema=web_final_schema([column], []),
                                     max_new_tokens=260, temperature=1.)


async def wait_until(predicate):
    async def wait():
        while not predicate():
            await asyncio.sleep(0)
    await asyncio.wait_for(wait(), 3)


def verify_cancelled(fixture, kind, rows):
    # Ownership is verified from closed JSONL, whose tuples are JSON arrays.
    rows = json.loads(json.dumps(rows))
    traces = json.loads(json.dumps(fixture.traces))
    task = Task("invented-expiry", "widesearch", "train" if kind == "train" else "validation",
                "Invented CPU request", metadata={"evaluation": {"required": ["name"]}})
    result = EpisodeResult(task.id, task.benchmark, task.split, [], "infrastructure_error", 0.,
                           {}, [], [], [])
    if kind == "train":
        results = [result, deepcopy(result)]
        web.verify_owned_native_group(task, results, rows, traces,
            eos_ids=fixture.policy.eos_ids, tokenizer=fixture.policy.tokenizer, iteration=1, rank=0,
            **({train_close.FIELD: train_close.configuration(4)}
               if fixture.policy.web_train_thought_budget_close else {}))
    else:
        verify_native_ownership([task], [prediction_row(result, seed=1)], rows, traces,
            rank=0, profile="raw", eos_ids=fixture.policy.eos_ids, tokenizer=fixture.policy.tokenizer)


@pytest.mark.parametrize("kind", ["train", "evaluation"])
def test_expired_queued_owned_request_never_submits_or_invents_completion(tokenizer, kind):
    fixture = native_fixture(tokenizer, batch_size=1, capture=kind == "train")
    first_rows, expired_rows = [], []

    async def run():
        first = asyncio.create_task(request(owned(fixture, kind, first_rows)))
        second = None
        try:
            assert await asyncio.to_thread(fixture.started.wait, 3)
            second = asyncio.create_task(request(owned(fixture, kind, expired_rows)))
            await wait_until(lambda: len(fixture.controls) == 2)
            assert not fixture.controls[1].started
            second.cancel()
            await wait_until(lambda: fixture.controls[1].expired)
            fixture.release.set()
            return await asyncio.gather(first, second, return_exceptions=True)
        finally:
            fixture.release.set()
            await fixture.policy.close()

    results = asyncio.run(run())
    assert results[0].native_status == "completed"
    assert isinstance(results[1], asyncio.CancelledError)
    assert fixture.calls == [("thought", 1), ("json", 1)]
    assert not fixture.controls[1].started
    assert len(fixture.traces) == 1
    assert len(expired_rows) == 1 and expired_rows[0]["request_status"] == "error"
    assert expired_rows[0]["error_type"] == NativeHFRequestExpiredBeforeSubmission.__name__
    assert expired_rows[0]["delivery"] == "cancelled"
    assert not {"trace_id", "completion", "completion_ids", "thought_logprobs"}.intersection(expired_rows[0])
    # This request has no native trace; retain its explicit infrastructure owner.
    fixture.traces = []
    verify_cancelled(fixture, kind, expired_rows)
    for field, value in (("trace_id", "invented"), ("completion_ids", [77]), ("thought_logprobs", [-.5]),
                         (train_close.FIELD, train_close.configuration(4)), (train_close.POSITIONS, [3])):
        forged = deepcopy(expired_rows)
        forged[0][field] = value
        with pytest.raises(ValueError, match="Unsubmitted expired request"):
            verify_cancelled(fixture, kind, forged)


@pytest.mark.parametrize("kind, close_train", [("train", False), ("evaluation", False), ("train", True)])
def test_cancelled_started_thought_drains_real_prefix_without_starting_json(tokenizer, kind, close_train):
    fixture = native_fixture(tokenizer, capture=kind == "train", close_train=close_train)
    rows = []

    async def run():
        pending = asyncio.create_task(request(owned(fixture, kind, rows)))
        try:
            assert await asyncio.to_thread(fixture.started.wait, 3)
            assert fixture.controls[0].started
            pending.cancel()
            await wait_until(lambda: fixture.controls[0].expired)
            # A sibling-error cancellation during deadline drain must not lose
            # the submitted phase or cancel its future.
            pending.cancel()
            await asyncio.sleep(0)
            assert not pending.done()
            fixture.release.set()
            return await asyncio.gather(pending, return_exceptions=True)
        finally:
            fixture.release.set()
            await fixture.policy.close()

    assert isinstance(asyncio.run(run())[0], asyncio.CancelledError)
    assert fixture.calls == [("thought", 1)]
    trace = fixture.traces[0]
    assert trace["status"] == CANCELLED_BEFORE_JSON
    assert trace["thought_ids"] == fixture.thought and trace["json_ids"] == []
    assert trace["thought_logprobs"] == (fixture.scores if kind == "train" else None)
    if close_train:
        assert trace["protocol"] == train_close.PROTOCOL
        assert trace[train_close.FIELD] == rows[0][train_close.FIELD] == train_close.configuration(4)
        assert trace[train_close.POSITIONS] == rows[0][train_close.POSITIONS] == [3]
    assert trace["final_text"] == "" and rows[0]["delivery"] == "cancelled"
    verify_cancelled(fixture, kind, rows)
    bad = deepcopy(rows)
    bad[0]["delivery"] = "delivered"
    with pytest.raises(ValueError):
        verify_cancelled(fixture, kind, bad)


@pytest.mark.parametrize("kind", ["train", "evaluation"])
def test_inflight_json_is_drained_with_its_actual_completion_and_eos(tokenizer, kind):
    fixture = native_fixture(tokenizer, block="json", capture=kind == "train")
    rows = []

    async def run():
        pending = asyncio.create_task(request(owned(fixture, kind, rows)))
        try:
            assert await asyncio.to_thread(fixture.started.wait, 3)
            pending.cancel()
            await wait_until(lambda: fixture.controls[0].expired)
            fixture.release.set()
            return await asyncio.gather(pending, return_exceptions=True)
        finally:
            fixture.release.set()
            await fixture.policy.close()

    assert isinstance(asyncio.run(run())[0], asyncio.CancelledError)
    assert fixture.calls == [("thought", 1), ("json", 1)]
    trace = fixture.traces[0]
    assert trace["status"] == "completed"
    assert trace["thought_ids"] == THOUGHT
    assert trace["json_ids"] == final_ids(tokenizer, text='{"answer":[]}')
    assert rows[0]["delivery"] == "cancelled"
    verify_cancelled(fixture, kind, rows)


@pytest.mark.parametrize("kind", ["train", "evaluation"])
def test_expired_mixed_batch_row_does_not_cancel_unexpired_sibling(tokenizer, kind):
    fixture = native_fixture(tokenizer, capture=kind == "train")
    rows = [[], []]

    async def run():
        pending = [asyncio.create_task(request(owned(fixture, kind, rows[i], sample=i), column))
                   for i, column in enumerate(("name", "other"))]
        try:
            assert await asyncio.to_thread(fixture.started.wait, 3)
            pending[0].cancel()
            await wait_until(lambda: fixture.controls[0].expired)
            fixture.release.set()
            return await asyncio.gather(*pending, return_exceptions=True)
        finally:
            fixture.release.set()
            await fixture.policy.close()

    results = asyncio.run(run())
    assert isinstance(results[0], asyncio.CancelledError)
    assert results[1].native_status == "completed" and json.loads(results[1].text) == {"answer": []}
    assert fixture.calls == [("thought", 2), ("json", 1)]
    assert [trace["status"] for trace in fixture.traces] == [CANCELLED_BEFORE_JSON, "completed"]
    assert [row[0]["delivery"] for row in rows] == ["cancelled", "delivered"]
    assert not fixture.controls[1].expired


def test_normal_controlled_request_keeps_exact_completion_and_trace_identity(tokenizer):
    fixture = native_fixture(tokenizer, block=None)
    schema = web_final_schema(["name"], [])
    original = fixture.policy._generate_batch([messages()], schema, ["normal"])[0]
    control = NativeHFRequestControl()
    assert control.try_start()
    controlled = fixture.policy._generate_batch([messages()], schema, ["normal"],
                                                request_controls=[control])[0]
    first, second = asdict(original), asdict(controlled)
    first.pop("trace_id")
    second.pop("trace_id")
    assert first == second
    traces = deepcopy(fixture.traces)
    for trace in traces:
        trace.pop("trace_id")
        trace.pop("batch_phase_elapsed_seconds")
    assert traces[0] == traces[1]
    assert fixture.calls == [("thought", 1), ("json", 1)] * 2


def test_real_episode_deadline_retains_closed_group_and_infrastructure_discard(tokenizer, tmp_path):
    fixture = native_fixture(tokenizer)
    data, _ = inputs(tmp_path)
    task = web.public_web_task(public_row())
    runner = EpisodeRunner(max_steps=1, max_new_tokens=260, timeout_seconds=.05,
        temperature=1., evaluation_temperature=1.,
        web_tools_factory=FixtureTools, web_structured_final=True, web_structured_actions=True,
        web_explicit_review=True)

    async def run():
        pending = asyncio.create_task(web.collect_public_group(runner, task, fixture.policy,
            group_size=2, rank=0, iteration=1, directory=tmp_path / "group", source_commit=COMMIT))
        try:
            assert await asyncio.to_thread(fixture.started.wait, 3)
            await wait_until(lambda: len(fixture.controls) == 2 and all(c.expired for c in fixture.controls))
            fixture.release.set()
            return await pending
        finally:
            fixture.release.set()
            if not pending.done():
                pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)

    group = asyncio.run(run())
    assert len(group.results) == 2 and all(r.status == "infrastructure_error" for r in group.results)
    assert all(r.steps == [] for r in group.results)
    assert fixture.calls == [("thought", 2)]
    assert group.closure["closed_journals"] == dict(native_traces=2, request_owners=2, public_outcomes=2)
    assert (tmp_path / "group/generation_closed.json").is_file()
    assert data.label_reads == 0
    results, report = web.score_closed_group(group, inputs=data, tokenizer=tokenizer)
    assert data.label_reads == 1 and report["gradient_group_discarded"]
    assert [r.reward for r in results] == [0., 0.]
    signal = group_signal(results, group_size=2, max_steps=1, thought_tokens=4)
    assert signal.discarded_reason == "unreliable_environment_or_context_group"
    assert signal.advantages == [0., 0.] and signal.segments == [[], []]
