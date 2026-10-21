"""Independent CPU fixtures: synthetic tasks, no datasets/models/network/GPU."""

import asyncio
import ast
from copy import deepcopy
from enum import IntEnum
import json
import io
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from slime_plugins.dcc import EpisodeRunner, PolicyCompletion, Task
from slime_plugins.dcc.episode import EpisodeResult, ModelProtocolFailure
from slime_plugins.dcc.episode_diagnostics import terminal_summary, token_accounting


def result(*, status="completed", steps=None, metrics=None, events=None, detail=""):
    return EpisodeResult("synthetic", "cuad", "validation", [], status, 0.0,
                         metrics or {}, steps or [], [], events or [], detail)


def reference(trace_id, *, status="completed", prompt=220, thought=20, final=5):
    stages = {"thought": thought}
    if final is not None:
        stages["json"] = final
    return {"trace_id": trace_id, "task_id": "synthetic", "rank": 0, "turn_index": 0,
            "status": status, "failure_reason": None, "physical_prompt_tokens": prompt,
            "completion_tokens_by_stage": stages}


class TerminalSummaryTests(unittest.IsolatedAsyncioTestCase):
    def task(self, benchmark="cuad"):
        return Task("synthetic", benchmark, "validation", "Find synthetic evidence", "Synthetic contract")

    async def test_empty_dependency_timeout_has_a_reason_without_raw_detail(self):
        async def policy(messages, **kwargs):
            raise TimeoutError()

        value = await EpisodeRunner(initial_chunks=0, max_steps=2).run(self.task(), policy)
        summary = terminal_summary(value, max_steps=2)
        self.assertEqual(value.status, "infrastructure_error")
        self.assertTrue(value.detail)
        self.assertEqual(summary["reason"], "dependency_timeout")
        self.assertEqual(summary["infrastructure"], {"operation": "policy", "turn_index": 0})
        self.assertFalse(summary["model_turn_budget_exhausted"])

    async def test_episode_timeout_and_turn_exhaustion_are_distinct(self):
        async def blocking(messages, **kwargs):
            await asyncio.Event().wait()

        async def invalid(messages, **kwargs):
            return PolicyCompletion("synthetic invalid action", prompt_ids=[1], completion_ids=[2])

        timeout = await EpisodeRunner(initial_chunks=0, max_steps=2, timeout_seconds=.01).run(self.task(), blocking)
        exhausted = await EpisodeRunner(initial_chunks=0, max_steps=2).run(self.task(), invalid)
        self.assertEqual(terminal_summary(timeout, max_steps=2)["reason"], "episode_deadline_exceeded")
        self.assertEqual(terminal_summary(exhausted, max_steps=2)["reason"], "model_turn_budget_exhausted")
        self.assertEqual(exhausted.status, "max_steps")
        self.assertEqual(exhausted.reward, 0)

    async def test_web_turn_exhaustion_is_visible_after_existing_status_rewrite(self):
        async def policy(messages, **kwargs):
            return PolicyCompletion("synthetic invalid action")

        value = await EpisodeRunner(initial_chunks=0, max_steps=2).run(self.task("widesearch"), policy)
        summary = terminal_summary(value, max_steps=2)
        self.assertEqual(value.status, "evidence_failure")
        self.assertEqual(summary["reason"], "model_turn_budget_exhausted")
        self.assertEqual(summary["evidence"]["rejection_count"], 0)

    async def test_evidence_schema_and_generation_budget_remain_different(self):
        async def empty(messages, **kwargs):
            return PolicyCompletion('{"answer":[]}', completion_ids=[2])

        value = await EpisodeRunner(initial_chunks=0, max_steps=1).run(self.task("widesearch"), empty)
        evidence = terminal_summary(value, max_steps=1)
        self.assertEqual(evidence["reason"], "evidence_rejected")
        self.assertEqual(evidence["evidence"]["rejection_counts"], {
            "empty_answer": 1, "missing_successful_fetch": 1, "missing_successful_search": 1})
        self.assertNotIn("model_protocol", evidence)
        for reason, finish, category in [
            ("json_not_complete_or_schema_valid", "stop", "schema_validation"),
            ("thought_did_not_stop_on_real_close", "length", "generation_budget"),
            ("thought_did_not_stop_on_real_close", "stop", "protocol"),
            ("json_did_not_finish_with_real_eos", "length", "generation_budget"),
        ]:
            async def failed(messages, **kwargs):
                raise ModelProtocolFailure(reason, completion=PolicyCompletion("", completion_ids=[3], finish_reason=finish),
                                           protocol="synthetic-protocol", trace_id="synthetic-trace")

            failed_result = await EpisodeRunner(initial_chunks=0, max_steps=1).run(self.task(), failed)
            summary = terminal_summary(failed_result, max_steps=1)
            self.assertEqual(summary["reason"], "model_protocol_failure")
            self.assertEqual(summary["model_protocol"]["reason"], reason)
            self.assertEqual(summary["model_protocol"]["category"], category)
            self.assertEqual(failed_result.status, "model_protocol_failure")
            self.assertEqual(failed_result.trainable_steps, [])

    async def test_actual_tool_errors_count_empty_detail_and_final_turn_denial(self):
        class Tools:
            async def web_search(self, **kwargs):
                return {"error": ""}

        async def policy(messages, **kwargs):
            return PolicyCompletion('{"tool":"web_search","arguments":{"query":"synthetic"}}')

        value = await EpisodeRunner(initial_chunks=0, max_steps=2, web_tools_factory=Tools).run(self.task("widesearch"), policy)
        summary = terminal_summary(value, max_steps=2)
        self.assertEqual(summary["tools"]["recorded_event_count"], 2)
        self.assertEqual(summary["tools"]["observation_error_count"], 2)
        self.assertEqual(summary["tools"]["observation_error_counts_by_tool"], {"web_search": 2})

    def test_compact_summary_has_only_labels_and_counts_and_does_not_mutate_result(self):
        secret = "PRIVATE_PAYLOAD https://private.example/credential?token=secret"
        value = result(status="model_protocol_failure", steps=[PolicyCompletion(secret)],
            metrics={"model_protocol_failure": {"reason": secret}, "evidence_guard": {
                "successful_searches": 2, "successful_fetches": 1, "observed_source_count": 3,
                "rejections": [{"turn": 0, "reasons": [secret, "empty_answer", "empty_answer"]}]}},
            events=[{"tool": secret, "arguments": {"url": secret}, "observation": {"error": secret}},
                    {"tool": "fetch_url", "observation": {"error": ""}},
                    {"tool": "web_search", "observation": {"results": [{"url": secret}]}}], detail=secret)
        value.messages = [{"role": "user", "content": secret}]
        value.answer = [{"secret": secret}]
        before = deepcopy(value.to_dict())
        summary = terminal_summary(value, max_steps=4)
        self.assertNotIn(secret, json.dumps(summary))
        self.assertNotIn("private.example", json.dumps(summary))
        self.assertEqual(summary["tools"]["observation_error_count"], 2)
        self.assertEqual(summary["evidence"]["rejection_counts"], {"empty_answer": 1, "other_evidence_rejection": 1})
        self.assertEqual(value.to_dict(), before)


class TokenAccountingTests(unittest.TestCase):
    def test_two_stage_prompt_counts_do_not_change_logical_usage(self):
        value = result(steps=[PolicyCompletion("", prompt_ids=list(range(100)), completion_ids=list(range(25)))])
        ref = reference("first", prompt=100 + 120, thought=20, final=5)
        before = deepcopy(value.to_dict())
        accounting = token_accounting(value, native_references=[ref])
        self.assertEqual(accounting["logical"], {"prompt": 100, "completion": 25})
        self.assertEqual(accounting["native"]["physical_prompt_tokens_requested"], 220)
        self.assertEqual(accounting["native"]["completion_tokens_observed"], 25)
        self.assertEqual(accounting["native"]["completion_tokens_by_stage_observed"], {"thought": 20, "json": 5})
        self.assertEqual(accounting["native"]["completion_tokens_total"], 25)
        self.assertEqual(value.to_dict(), before)

    def test_trailing_interrupted_call_adds_observed_tokens_without_estimation(self):
        value = result(status="infrastructure_error", steps=[PolicyCompletion("", prompt_ids=list(range(100)), completion_ids=list(range(25)))])
        refs = [reference("first"), reference("cancelled", status="interrupted", prompt=50, thought=7, final=None)]
        accounting = token_accounting(value, native_references=refs)
        self.assertEqual(accounting["logical"], {"prompt": 100, "completion": 25})
        native = accounting["native"]
        self.assertEqual(native["physical_prompt_tokens_requested"], 270)
        self.assertEqual(native["completion_tokens_observed"], 32)
        self.assertIsNone(native["completion_tokens_total"])
        self.assertIsNone(native["completion_tokens_unobserved"])
        self.assertFalse(native["reference_coverage_mismatch"])

    def test_cancelled_call_with_no_engine_response_reports_zero_observed_and_unknown_total(self):
        value = result(status="infrastructure_error")
        native = token_accounting(value, native_references=[reference("cancelled", status="interrupted", prompt=50, thought=0, final=None)])["native"]
        self.assertEqual(native["completion_tokens_observed"], 0)
        self.assertIsNone(native["completion_tokens_total"])
        self.assertIsNone(native["completion_tokens_unobserved"])

    def test_duplicate_trace_does_not_double_count_and_conflict_is_visible(self):
        value = result(steps=[PolicyCompletion("")])
        ref = reference("first")
        native = token_accounting(value, native_references=[ref, deepcopy(ref)])["native"]
        self.assertEqual(native["reference_count"], 1)
        self.assertEqual(native["duplicate_reference_count"], 1)
        self.assertEqual(native["physical_prompt_tokens_requested"], 220)
        conflicting = {**ref, "physical_prompt_tokens": 9999}
        native = token_accounting(value, native_references=[ref, conflicting])["native"]
        self.assertEqual(native["conflicting_reference_count"], 1)
        self.assertEqual(native["physical_prompt_tokens_requested"], 220)
        self.assertIsNone(native["completion_tokens_total"])

    def test_missing_or_malformed_reference_is_unknown_without_fabricated_counts(self):
        value = result(steps=[PolicyCompletion("", completion_ids=[1, 2])])
        for refs in [[], [None], [{"trace_id": "first", "status": "completed"}]]:
            accounting = token_accounting(value, native_references=refs)
            self.assertEqual(accounting["logical"]["completion"], 2)
            self.assertEqual(accounting["native"]["completion_tokens_observed"], 0)
            self.assertIsNone(accounting["native"]["completion_tokens_total"])

    def test_failed_protocol_preserves_observed_logical_count_but_total_is_conservative(self):
        value = result(status="model_protocol_failure", steps=[PolicyCompletion("", prompt_ids=[1], completion_ids=[2, 3], finish_reason="length")])
        accounting = token_accounting(value, native_references=[reference("failed", status="failed", prompt=1, thought=2, final=None)])
        self.assertEqual(accounting["logical"], {"prompt": 1, "completion": 2})
        self.assertEqual(accounting["native"]["completion_tokens_observed"], 2)
        self.assertIsNone(accounting["native"]["completion_tokens_total"])

    def test_non_native_is_explicit_and_does_not_invent_physical_accounting(self):
        value = result(steps=[PolicyCompletion("", prompt_ids=[1, 2], completion_ids=[3])])
        self.assertEqual(token_accounting(value)["logical"], {"prompt": 2, "completion": 1})
        self.assertIsNone(token_accounting(value)["native"])


class NativeCancellationObservationTests(unittest.IsolatedAsyncioTestCase):
    async def fixture(self, *, emit_snapshot):
        # Use the real native _phase and task-local reference-finally path with
        # a tiny synthetic engine. No actual vLLM/jsonschema dependency is used.
        from slime_plugins.dcc.native_thinking_eval import NativeEpisodePolicy, NativeThinkingPolicy

        class OutputKind(IntEnum):
            FINAL_ONLY = 2

        module = ModuleType("vllm.sampling_params")
        module.RequestOutputKind = OutputKind
        package = ModuleType("vllm")
        package.sampling_params = module
        trace_id_ready = asyncio.Event()

        class Engine:
            def __init__(self):
                self.aborted = []
                self.calls = 0

            async def generate(self, prompt, params, request_id, **kwargs):
                self.calls += 1
                if emit_snapshot is None:
                    yield SimpleNamespace(prompt_token_ids=prompt["prompt_token_ids"], finished=True,
                        outputs=[SimpleNamespace(token_ids=list(range(20 if self.calls == 1 else 5)),
                            text="synthetic", finish_reason="stop", stop_reason=99)])
                    return
                if emit_snapshot:
                    yield SimpleNamespace(prompt_token_ids=prompt["prompt_token_ids"], finished=False,
                        outputs=[SimpleNamespace(token_ids=list(range(7)), text="synthetic", finish_reason=None, stop_reason=None)])
                trace_id_ready.set()
                await asyncio.Event().wait()

            async def abort(self, request_id):
                self.aborted.append(request_id)

        engine = Engine()
        sampling = SimpleNamespace(to_dict=lambda: {})
        native = object.__new__(NativeThinkingPolicy)
        native.phase_timing = False
        native.budget = SimpleNamespace(legacy=True, thought_sampling=sampling)
        native.base = SimpleNamespace(model=engine, lora_request=None, evaluation_sampling=sampling,
            _active_requests=set(), tokenizer=SimpleNamespace(decode=lambda *args, **kwargs: "synthetic"))
        native.trace_index = {}
        params = SimpleNamespace(max_tokens=50, n=1, seed=0, stop_token_ids=[99], ignore_eos=False,
            include_stop_str_in_output=True, skip_special_tokens=False, output_kind=2, structured_outputs=None)

        class Parent:
            trace_index = native.trace_index

            async def generate_structured(self, messages, *, trace_metadata, **kwargs):
                prompt = list(range(100 if emit_snapshot is None else 50))
                trace = {**trace_metadata, "status": "running", "failure_reason": None,
                         "phases": [], "original_prompt_ids": prompt}
                self.trace_index[trace_metadata["trace_id"]] = trace
                try:
                    thought = await native._phase(trace, "thought", prompt, params)
                    if emit_snapshot is None:
                        await native._phase(trace, "json", prompt + thought["token_ids"], params)
                        trace["status"] = "completed"
                        return native._completion(trace, text="synthetic")
                except asyncio.CancelledError:
                    trace["status"] = "interrupted"
                    raise

        bound = NativeEpisodePolicy(Parent(), "synthetic", 0)
        with patch.dict(sys.modules, {"vllm": package, "vllm.sampling_params": module}):
            task = asyncio.create_task(bound.generate_structured([]))
            if emit_snapshot is None:
                completion = await task
            else:
                async with asyncio.timeout(5):
                    await trace_id_ready.wait()
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
        self.assertEqual(len(engine.aborted), 0 if emit_snapshot is None else 1)
        self.assertEqual(native.base._active_requests, set())
        self.assertEqual(len(bound.references), 1)
        specimen = result(steps=[completion]) if emit_snapshot is None else result(status="infrastructure_error")
        accounting = token_accounting(specimen, native_references=bound.references)
        if emit_snapshot is None:
            self.assertEqual(accounting["logical"], {"prompt": 100, "completion": 25})
        return accounting["native"]

    async def test_real_two_phase_requests_retain_both_prompts_and_logical_original_prompt(self):
        native = await self.fixture(emit_snapshot=None)
        self.assertEqual(native["physical_prompt_tokens_requested"], 220)
        self.assertEqual(native["completion_tokens_by_stage_observed"], {"thought": 20, "json": 5})
        self.assertEqual(native["completion_tokens_total"], 25)

    async def test_real_phase_retains_observed_snapshot_when_cancelled(self):
        native = await self.fixture(emit_snapshot=True)
        self.assertEqual(native["physical_prompt_tokens_requested"], 50)
        self.assertEqual(native["completion_tokens_observed"], 7)
        self.assertIsNone(native["completion_tokens_total"])

    async def test_real_phase_does_not_fabricate_tokens_without_snapshot(self):
        native = await self.fixture(emit_snapshot=False)
        self.assertEqual(native["physical_prompt_tokens_requested"], 50)
        self.assertEqual(native["completion_tokens_observed"], 0)
        self.assertIsNone(native["completion_tokens_total"])


class EvaluationWriterCompatibilityTests(unittest.IsolatedAsyncioTestCase):
    async def run_writer(self, script, specimen, *, native_enabled, save_tokens):
        tree = ast.parse(script.read_text())
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.AsyncFunctionDef) and node.name == "one"]
        self.assertEqual(len(functions), 1)
        writer = ast.Module(body=[functions[0]], type_ignores=[])
        ast.fix_missing_locations(writer)
        stream = io.StringIO()
        console = []
        refs = [reference("first"), reference("cancelled", status="interrupted", prompt=50, thought=7, final=None)]
        policy = SimpleNamespace(bind_task=lambda task_id, rank: SimpleNamespace(references=refs))

        async def run(task, episode_policy, *, evaluation):
            self.assertTrue(evaluation)
            return deepcopy(specimen)

        clock = iter([10.0, 11.5, 12.0])
        namespace = {
            "semaphore": asyncio.Semaphore(1), "time": SimpleNamespace(monotonic=lambda: next(clock)),
            "policy": policy, "native": {"enabled": True} if native_enabled else None,
            "rank": 0, "runner": SimpleNamespace(run=run),
            "args": SimpleNamespace(save_tokens=save_tokens, max_steps=2),
            "stream": stream, "start": 0, "json": json,
            "terminal_summary": terminal_summary, "token_accounting": token_accounting,
            "print": lambda value, **kwargs: console.append(json.loads(value)),
        }
        exec(compile(writer, str(script), "exec"), namespace)
        await namespace["one"](SimpleNamespace(id="synthetic"))
        self.assertEqual(len(console), 1)
        return json.loads(stream.getvalue()), console[0]

    async def test_actual_writer_keeps_legacy_contract_with_and_without_native_or_saved_tokens(self):
        candidate = Path(__file__).resolve().parents[1] / "scripts/dcc_model_evaluate.py"
        for native_enabled in (False, True):
            for save_tokens in (False, True):
                with self.subTest(native=native_enabled, save_tokens=save_tokens):
                    specimen = result(status="infrastructure_error" if native_enabled else "completed",
                        steps=[PolicyCompletion("synthetic", prompt_ids=list(range(100)), completion_ids=list(range(25)))])
                    before = deepcopy(specimen.to_dict())
                    old_saved = specimen.to_dict(include_tokens=save_tokens)
                    if native_enabled:
                        old_saved["native_thinking_traces"] = [reference("first"), reference("cancelled", status="interrupted", prompt=50, thought=7, final=None)]
                    old_saved.update(episode_latency_seconds=1.5, token_usage={"prompt": 100, "completion": 25})
                    old_console = {"rank": 0, "task": "synthetic", "status": specimen.status, "turns": 1,
                                   "token_usage": {"prompt": 100, "completion": 25},
                                   "episode_latency_seconds": 1.5, "elapsed": 12.0}
                    new_saved, new_console = await self.run_writer(candidate, specimen, native_enabled=native_enabled, save_tokens=save_tokens)
                    for output, legacy in [(new_saved, old_saved), (new_console, old_console)]:
                        self.assertEqual({key: value for key, value in output.items()
                                          if key not in {"token_accounting", "terminal_summary"}}, legacy)
                        self.assertEqual(output["token_usage"], {"prompt": 100, "completion": 25})
                        self.assertEqual(output["token_accounting"]["logical"], output["token_usage"])
                    self.assertEqual(new_saved["terminal_summary"], new_console["terminal_summary"])
                    if native_enabled:
                        self.assertEqual(new_saved["token_accounting"]["native"]["completion_tokens_observed"], 32)
                    else:
                        self.assertIsNone(new_saved["token_accounting"]["native"])
                    self.assertEqual(specimen.to_dict(), before)


if __name__ == "__main__":
    unittest.main()
