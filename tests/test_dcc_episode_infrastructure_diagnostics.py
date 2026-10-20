"""CPU-only coverage for episode deadline and dependency failure diagnostics."""

import asyncio
import unittest

from slime_plugins.dcc.data import Task
from slime_plugins.dcc.episode import EpisodeRunner, ModelProtocolFailure, PolicyCompletion
from slime_plugins.dcc.tools import InfrastructureError


def cuad_task():
    return Task("synthetic-cuad", "cuad", "validation", "Find the clause", "Synthetic contract")


def web_task():
    return Task("synthetic-web", "widesearch", "validation", "Find public entities")


class StubWebTools:
    def __init__(self, failure=None, *, block_search=False, block_fetch=False):
        self.failure = failure
        self.block_search, self.block_fetch = block_search, block_fetch
        self.calls = []

    async def web_search(self, query, count=8):
        self.calls.append(("web_search", query))
        if self.block_search:
            await asyncio.Event().wait()
        if self.failure is not None:
            raise self.failure
        return {"results": [{"url": "https://example.org/source", "title": "Source"}]}

    async def fetch_url(self, url, start=0):
        self.calls.append(("fetch_url", url))
        if self.block_fetch:
            await asyncio.Event().wait()
        if self.failure is not None:
            raise self.failure
        return {"url": url, "text": "Synthetic source", "start": 0, "end": 16}


async def search_policy(messages, **kwargs):
    return PolicyCompletion('{"tool":"web_search","arguments":{"query":"synthetic"}}', completion_ids=[1])


class EpisodeInfrastructureTests(unittest.IsolatedAsyncioTestCase):
    def runner(self, **kwargs):
        return EpisodeRunner(initial_chunks=0, max_steps=2, timeout_seconds=0.03, **kwargs)

    def diagnostic(self, result, reason, operation):
        self.assertEqual(result.status, "infrastructure_error")
        self.assertEqual(result.answer, [])
        self.assertEqual(result.reward, 0.0)
        self.assertEqual(result.trainable_steps, [])
        self.assertTrue(result.detail)
        diagnostic = result.metrics["infrastructure_error"]
        self.assertEqual(diagnostic["version"], "episode-infrastructure-diagnostics-v1")
        self.assertEqual(diagnostic["reason"], reason)
        self.assertEqual(diagnostic["operation"], operation)
        self.assertEqual(diagnostic["episode_deadline_exceeded"], reason == "episode_deadline_exceeded")
        return diagnostic

    async def test_episode_deadline_retains_cancellation_and_reports_policy(self):
        cancelled = asyncio.Event()

        async def policy(messages, **kwargs):
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        result = await self.runner().run(cuad_task(), policy, evaluation=True)
        diagnostic = self.diagnostic(result, "episode_deadline_exceeded", "policy")
        self.assertTrue(cancelled.is_set())
        self.assertEqual(diagnostic["exception_type"], "TimeoutError")
        self.assertEqual(diagnostic["turn_index"], 0)
        self.assertEqual(diagnostic["episode_timeout_seconds"], 0.03)
        self.assertIn("0.03 seconds", result.detail)

    async def test_deadline_preserves_completed_steps_and_inflight_turn(self):
        calls = 0

        async def policy(messages, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                return PolicyCompletion("invalid action", completion_ids=[7, 8])
            await asyncio.Event().wait()

        result = await self.runner().run(cuad_task(), policy, evaluation=True)
        diagnostic = self.diagnostic(result, "episode_deadline_exceeded", "policy")
        self.assertEqual(diagnostic["turn_index"], 1)
        self.assertEqual(result.metrics["model_calls"], 1)
        self.assertEqual(result.steps[0].completion_ids, [7, 8])
        self.assertEqual(calls, 2)

    async def test_empty_dependency_timeout_is_not_episode_deadline(self):
        async def policy(messages, **kwargs):
            raise TimeoutError()

        result = await self.runner().run(cuad_task(), policy)
        diagnostic = self.diagnostic(result, "dependency_timeout", "policy")
        self.assertEqual(diagnostic["exception_type"], "TimeoutError")
        self.assertIn("TimeoutError", result.detail)

    async def test_dependency_timeout_message_is_preserved(self):
        async def policy(messages, **kwargs):
            raise TimeoutError("dependency deadline")

        result = await self.runner().run(cuad_task(), policy)
        self.diagnostic(result, "dependency_timeout", "policy")
        self.assertEqual(result.detail, "dependency deadline")

    async def test_nested_dependency_timeout_is_not_episode_deadline(self):
        async def policy(messages, **kwargs):
            async with asyncio.timeout(0):
                await asyncio.Event().wait()

        result = await self.runner().run(cuad_task(), policy)
        self.diagnostic(result, "dependency_timeout", "policy")

    async def test_connection_error_is_distinguished(self):
        async def policy(messages, **kwargs):
            raise ConnectionError("connection closed")

        result = await self.runner().run(cuad_task(), policy)
        self.diagnostic(result, "dependency_connection_error", "policy")
        self.assertEqual(result.detail, "connection closed")

    async def test_search_backend_failure_is_distinguished(self):
        tools = StubWebTools(InfrastructureError("search unavailable"))
        result = await self.runner(web_tools_factory=lambda: tools).run(web_task(), search_policy)
        diagnostic = self.diagnostic(result, "dependency_infrastructure_error", "tool:web_search")
        self.assertEqual(diagnostic["exception_type"], "InfrastructureError")
        self.assertEqual(result.detail, "search unavailable")
        self.assertEqual(len(tools.calls), 1)

    async def test_tool_timeout_is_not_episode_deadline(self):
        tools = StubWebTools(TimeoutError())
        result = await self.runner(web_tools_factory=lambda: tools).run(web_task(), search_policy)
        self.diagnostic(result, "dependency_timeout", "tool:web_search")
        self.assertEqual(len(tools.calls), 1)

    async def test_episode_deadline_during_search_is_recorded(self):
        tools = StubWebTools(block_search=True)
        result = await self.runner(web_tools_factory=lambda: tools).run(web_task(), search_policy)
        self.diagnostic(result, "episode_deadline_exceeded", "tool:web_search")
        self.assertEqual(len(tools.calls), 1)

    async def test_episode_deadline_during_auto_read_is_recorded(self):
        tools = StubWebTools(block_fetch=True)
        result = await self.runner(web_tools_factory=lambda: tools, auto_read_sources=1).run(web_task(), search_policy)
        self.diagnostic(result, "episode_deadline_exceeded", "auto_read_sources")
        self.assertEqual([call[0] for call in tools.calls], ["web_search", "fetch_url"])

    async def test_external_cancellation_propagates(self):
        started = asyncio.Event()

        async def policy(messages, **kwargs):
            started.set()
            await asyncio.Event().wait()

        task = asyncio.create_task(self.runner().run(cuad_task(), policy))
        await started.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

    async def test_success_has_no_failure_diagnostics(self):
        async def policy(messages, **kwargs):
            return PolicyCompletion('{"answer":[]}', completion_ids=[9])

        result = await self.runner().run(cuad_task(), policy)
        self.assertEqual(result.status, "completed")
        self.assertNotIn("infrastructure_error", result.metrics)
        self.assertEqual(result.trainable_steps[0].completion_ids, [9])

    async def test_model_protocol_failure_keeps_original_classification(self):
        async def policy(messages, **kwargs):
            raise ModelProtocolFailure("bad EOS", completion=PolicyCompletion("raw", completion_ids=[5]),
                                       protocol="synthetic-protocol", trace_id="synthetic-trace")

        result = await self.runner().run(cuad_task(), policy)
        self.assertEqual(result.status, "model_protocol_failure")
        self.assertEqual(result.detail, "bad EOS")
        self.assertNotIn("infrastructure_error", result.metrics)
        self.assertEqual(result.metrics["model_protocol_failure"]["trace_id"], "synthetic-trace")


if __name__ == "__main__":
    unittest.main()
