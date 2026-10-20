"""CPU caller tests. Fake tokenizer/receipts/backend; no model or engine created."""
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT
spec = importlib.util.spec_from_file_location("collector_eager", ROOT / "scripts/dcc_collect_native_train.py")
collector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector)


class CollectorEagerCallerTests(unittest.TestCase):
    def run_actual_main(self, enabled, *, phase_timing=False):
        tokenizer = object()
        profile = {"identity": {"eos_token_ids": [11]}}
        budget, schemas = object(), object()
        receipt = {"fake_cpu_receipt": True}
        identity = SimpleNamespace(max_prompt_tokens=16384, max_model_len=36864,
                                   generation_seed=58, budget=budget,
                                   eos_token_ids=(11,),
                                   manifest=lambda actual: profile["identity"] if actual is tokenizer else None)
        base = SimpleNamespace(eos_token_ids=[11], tokenizer=tokenizer, allow_truncation=True)
        native = object()
        tasks = [object()]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "fresh"
            argv = ["dcc_collect_native_train.py", "generate", "--source-root", str(PARENT),
                    "--expected-source", "76b57e227b943fb34a446226fa67cc1420f40e52",
                    "--profile", str(Path(directory) / "fake-profile.json"), "--model", "CPU_FAKE_MODEL",
                    "--output-dir", str(output), "--world-size", "32", "--rank", "3",
                    "--public-train-data", str(Path(directory) / "fake-public.jsonl"),
                    "--collection-id", "cpu-caller-only", "--max-tasks-per-benchmark", "512",
                    "--concurrency", "4", "--gpu-memory-utilization", ".88", "--timeout-seconds", "1800"]
            if enabled:
                argv.append("--vllm-enforce-eager")
            if phase_timing:
                argv.append("--native-phase-timing")
            with patch.object(sys, "argv", argv), \
                 patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer), \
                 patch.object(collector, "load_approved", return_value=(profile, identity, schemas, receipt)), \
                 patch.object(collector, "load_public_train", return_value=tasks), \
                 patch.object(collector, "fixed_tasks", return_value=tasks) as fixed, \
                 patch("slime_plugins.dcc.vllm_policy.load_vllm_policy", return_value=base) as backend, \
                 patch.object(collector, "NativeTraceJournal", return_value="cpu-fake-journal") as journal, \
                 patch.object(collector, "NativeThinkingPolicy", return_value=native) as wrapper, \
                 patch.object(collector, "generate", new=AsyncMock()) as generate:
                collector.main()
            backend.assert_called_once()
            args, options = backend.call_args
            self.assertEqual(args, ("CPU_FAKE_MODEL",))
            self.assertEqual(options, {
                "adapter": None, "asynchronous": True, "max_prompt_tokens": 16384,
                "max_model_len": 36864, "tensor_parallel_size": 1,
                "gpu_memory_utilization": .88, "seed": 58,
                "evaluation_sampling": collector.EvaluationSampling(),
                "enforce_eager": enabled, "web_structured_final": True,
                "web_structured_actions": True,
            })
            self.assertFalse(base.allow_truncation)
            fixed.assert_called_once_with(tasks, max_tasks_per_benchmark=512, rank=3, world_size=32)
            journal.assert_called_once_with(output / "native_thinking_traces_rank3.jsonl")
            wrapper.assert_called_once_with(base, source_receipt=receipt, budget=budget,
                                            prompt_history_config=collector.PromptHistoryConfig(),
                                            phase_timing=phase_timing, trace_sink="cpu-fake-journal")
            generate.assert_awaited_once_with(tasks, native, output,
                collection_id="cpu-caller-only", profile=profile, schema_profile=schemas,
                rank=3, world_size=32, concurrency=4, timeout_seconds=1800)

    def test_actual_cli_default_off_reaches_existing_backend_with_false(self):
        self.run_actual_main(False)

    def test_actual_cli_opt_in_reaches_existing_backend_with_true_only(self):
        self.run_actual_main(True)

    def test_native_phase_timing_opt_in_only_changes_native_wrapper(self):
        self.run_actual_main(False, phase_timing=True)



if __name__ == "__main__":
    unittest.main()
