"""Opt-in observations of native requests; no engine calls or synchronization."""
from __future__ import annotations

from collections.abc import Mapping
import math
from time import monotonic


VERSION = "native-phase-monotonic-timing-v1"
# Preserve vLLM's own names and units. Some versions expose no request metrics.
ENGINE_FIELDS = (
    "arrival_time", "first_scheduled_time", "first_token_time", "last_token_time",
    "finished_time", "time_in_queue", "scheduler_time", "model_forward_time",
    "model_execute_time", "time_in_prefill", "time_in_decode", "prefill_time", "decode_time",
)


def configuration(enabled):
    if type(enabled) is not bool:
        raise ValueError("native_phase_timing must be a boolean")
    if not enabled:
        return None
    return {"enabled": True, "version": VERSION,
            "clock": "time.monotonic; seconds; local process only",
            "engine_metrics": "finite scalar RequestOutput.metrics fields; original names and units",
            "extra_engine_calls": False}


def add_arguments(parser):
    parser.add_argument("--native-phase-timing", action="store_true",
                        help="Record native phase monotonic timings and available actual vLLM request metrics")


class NativePhaseTimer:
    """Store observations in the existing phase journal, including cancellation."""

    def __init__(self, phase):
        self.started = monotonic()
        self.record = phase["timing"] = {
            "version": VERSION, "clock": "time.monotonic",
            "dispatch_monotonic_seconds": self.started,
        }

    def mark(self, event):
        now = monotonic()
        self.record[event + "_monotonic_seconds"] = now
        self.record[event + "_elapsed_seconds"] = now - self.started
        return now

    def observe(self, item):
        if "first_response_monotonic_seconds" not in self.record:
            self.mark("first_response")
        self.mark("last_response")
        # Optional telemetry must not replace an otherwise valid response with
        # a metrics accessor/serialization error on a different vLLM version.
        try:
            metrics = getattr(item, "metrics", None)
            if metrics is None:
                return
            values = {}
            for name in ENGINE_FIELDS:
                value = metrics.get(name) if isinstance(metrics, Mapping) else getattr(metrics, name, None)
                if type(value) in (int, float) and math.isfinite(value):
                    values[name] = value
            if values:
                self.record["engine_metrics"] = values
                self.record["engine_metrics_source"] = "vllm.RequestOutput.metrics"
        except Exception as error:
            self.record["engine_metrics_read_error_type"] = type(error).__name__

    def end(self):
        self.mark("end")
        self.record["elapsed_seconds"] = self.record["end_elapsed_seconds"]
        if "abort_start_monotonic_seconds" in self.record:
            self.record["abort_elapsed_seconds"] = (
                self.record["abort_end_monotonic_seconds"] - self.record["abort_start_monotonic_seconds"])
