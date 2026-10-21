"""Sampling overrides must fail before the built-in hook requests any tokens."""
from __future__ import annotations

import argparse
import asyncio
import importlib
from types import SimpleNamespace

import pytest

from slime_plugins.stackplanner.cli import build_slime_command
from slime_plugins.stackplanner.config import TrainingConfig

generator = importlib.import_module("slime_plugins.stackplanner.generate")


def final_native_args(extra):
    config = TrainingConfig.from_mapping({"run_id": "sampling-override", "extra_slime_args": extra})
    command = build_slime_command(config)
    # Slime declares exactly this float option. parse_known_args preserves its
    # real last-option-wins behavior without importing a Megatron/GPU launcher.
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--rollout-top-p", type=float, default=1.0)
    args, _ = parser.parse_known_args(command[2:])
    return args


@pytest.mark.parametrize("extra", [
    ["--rollout-top-p", ".9"],
    ["--rollout-top-p=.9"],
    ["--rollout-top-p", "1", "--rollout-top-p=.9"],
])
def test_effective_extra_sampling_fails_before_config_or_adapter(extra, monkeypatch):
    args = final_native_args(extra)
    assert args.rollout_top_p == .9
    def no_setup():
        pytest.fail("Configuration/adapter setup reached before unsupported sampling was rejected")
    monkeypatch.setattr(generator, "_load_training_config", no_setup)
    with pytest.raises(ValueError, match="built-in StackPlanner rollout currently requires top_p=1.0"):
        asyncio.run(generator.generate(args, SimpleNamespace(), {"top_p": args.rollout_top_p}))


@pytest.mark.parametrize("extra", [
    [], ["--rollout-top-p", "1"], ["--rollout-top-p=1"],
    ["--rollout-top-p", ".9", "--rollout-top-p=1"],
])
def test_effective_one_is_not_accidentally_rejected(extra, monkeypatch):
    args = final_native_args(extra)
    assert args.rollout_top_p == 1.0
    class SetupReached(Exception):
        pass
    def stop_before_setup():
        raise SetupReached
    monkeypatch.setattr(generator, "_load_training_config", stop_before_setup)
    with pytest.raises(SetupReached):
        asyncio.run(generator.generate(args, SimpleNamespace(), {"top_p": args.rollout_top_p}))


@pytest.mark.parametrize("args_top_p,sample_top_p", [(1.0, .9), (.9, 1.0)])
def test_native_converter_and_per_sample_sampling_are_both_guarded(args_top_p, sample_top_p, monkeypatch):
    def no_setup():
        pytest.fail("Unexpected adapter setup")
    monkeypatch.setattr(generator, "_load_training_config", no_setup)
    with pytest.raises(ValueError, match="top_p=1.0"):
        asyncio.run(generator.generate(SimpleNamespace(rollout_top_p=args_top_p), SimpleNamespace(),
                                       {"top_p": sample_top_p}))


def test_evaluation_and_existing_missing_sampling_default_keep_their_path(monkeypatch):
    class SetupReached(Exception):
        pass
    def stop_before_setup():
        raise SetupReached
    monkeypatch.setattr(generator, "_load_training_config", stop_before_setup)
    for args, sampling, evaluation in [(SimpleNamespace(rollout_top_p=.9), {"top_p": .9}, True),
                                       (SimpleNamespace(), {}, False)]:
        with pytest.raises(SetupReached):
            asyncio.run(generator.generate(args, SimpleNamespace(), sampling, evaluation=evaluation))


def test_custom_generator_and_train_only_replay_config_stay_compatible():
    custom = TrainingConfig.from_mapping({"run_id": "custom-top-p", "top_p": .9,
        "custom_generate_function": "custom_rollout.generate", "extra_slime_args": ["--rollout-top-p=.9"]})
    assert custom.custom_generate_function == "custom_rollout.generate"
    replay = TrainingConfig.from_mapping({"run_id": "replay-top-p", "top_p": .9,
        "train_only": True, "replay_path": "replay/{rollout_id}.pt",
        "extra_slime_args": ["--rollout-top-p=.9"]})
    assert replay.train_only
