from __future__ import annotations

import json
from pathlib import Path

import pytest

from slime.ray.utils import add_default_ray_env_vars
from slime_plugins.stackplanner.cli import (
    _checkpoint_next_rollout_id,
    _discover_policy_roles,
    _port_check,
    _read_mapping,
    _redacted_command,
    _search_r1_identity,
    build_parser,
    build_plan,
    execute,
    resolve_config,
)
from slime_plugins.stackplanner.config import TrainingMode

ROOT = Path(__file__).resolve().parent.parent


def test_ephemeral_adapter_port_is_valid_for_parallel_workers() -> None:
    check = _port_check("adapter port", "127.0.0.1", 0)
    assert check.ok
    assert "ephemeral" in check.detail


def test_help_covers_required_parameter_categories(
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = build_parser()
    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["--help"])
    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    for option in (
        "--mode",
        "--stackplanner-path",
        "--megatron-path",
        "--hf-checkpoint",
        "--data-path",
        "--train-steps",
        "--samples-per-prompt",
        "--max-parallel-delegates",
        "--adapter-port",
        "--sglang-engine-port-base",
        "--gpus-per-node",
        "--reward-function",
        "--rollout-only",
        "--train-only",
    ):
        assert option in output


def test_json_subset_yaml_needs_no_pyyaml() -> None:
    config = _read_mapping(ROOT / "configs/experiment_a_central.yaml")
    assert config["mode"] == "central-only"
    assert config["train_steps"] == 100


def test_training_search_config_accepts_a_portable_search_r1_endpoint():
    raw = _read_mapping(ROOT / "configs/experiment_c_capability_decoupled.yaml")
    config, _, _ = resolve_config(
        ["--config", str(ROOT / "configs/experiment_a_central.yaml")]
    )
    identity = _search_r1_identity(config)
    assert raw["deer_flow_config_path"] == "configs/deerflow_search_r1_only.yaml"
    assert identity["provider"] == (
        "deerflow.community.search_r1.tools:web_search_tool"
    )
    assert identity["endpoint"] == "http://127.0.0.1:8000"
    assert identity["index_sha256"] is None
    assert identity["corpus_sha256"] is None


def test_cli_wins_over_config_and_unknown_args_forward() -> None:
    config, forwarded, _ = resolve_config(
        [
            "--config",
            str(ROOT / "configs/experiment_b_all_agents.yaml"),
            "--experiment",
            "A",
            "--mode",
            "central-only",
            "--no-capability-decoupling",
            "--decoupling-scope",
            "none",
            "--train-steps",
            "3",
            "--sglang-mem-fraction-static",
            "0.65",
        ]
    )
    assert config.mode is TrainingMode.CENTRAL_ONLY
    assert config.train_steps == 3
    assert forwarded == ["--sglang-mem-fraction-static", "0.65"]
    assert config.extra_slime_args[-2:] == forwarded


def test_data_window_uses_slime_generalized_path_slice() -> None:
    config, _, _ = resolve_config(
        [
            "--config",
            str(ROOT / "configs/experiment_a_central.yaml"),
            "--data-path",
            str(ROOT / "fixtures/smoke_tasks.jsonl"),
            "--data-start",
            "1",
            "--data-limit",
            "2",
        ]
    )
    command = list(build_plan(config).command)
    prompt_data = command[command.index("--prompt-data") + 1]
    assert prompt_data == f"{ROOT / 'fixtures/smoke_tasks.jsonl'}@[1:3]"


def test_open_ended_data_window_is_supported() -> None:
    config, _, _ = resolve_config(
        [
            "--config",
            str(ROOT / "configs/experiment_a_central.yaml"),
            "--data-path",
            str(ROOT / "fixtures/smoke_tasks.jsonl"),
            "--data-start",
            "2",
        ]
    )
    command = list(build_plan(config).command)
    prompt_data = command[command.index("--prompt-data") + 1]
    assert prompt_data == f"{ROOT / 'fixtures/smoke_tasks.jsonl'}@[2:]"


def test_policy_roles_are_discovered_from_runtime_registry() -> None:
    roles = _discover_policy_roles(ROOT / "StackPlanner2.0")
    assert {
        "researcher",
        "coder",
        "reporter",
        "outline",
        "perception",
        "memory_recaller",
    } == set(roles)


def test_redaction_hides_credentials_but_not_schema_or_token_limit_options() -> None:
    command = _redacted_command(
        [
            "train.py",
            "--input-key",
            "prompt",
            "--max-tokens-per-gpu",
            "8192",
            "--api-key",
            "sensitive-value",
        ]
    )
    assert "--input-key prompt" in command
    assert "--max-tokens-per-gpu 8192" in command
    assert "sensitive-value" not in command


def test_preflight_does_not_create_run_directory(tmp_path: Path) -> None:
    raw = json.loads((ROOT / "configs/experiment_a_central.yaml").read_text())
    raw.update(
        {
            "run_id": "no-side-effects",
            "output_root": str(tmp_path),
            "data_path": str(ROOT / "fixtures/smoke_tasks.jsonl"),
        }
    )
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(raw), encoding="utf-8")
    config, _, _ = resolve_config(["--config", str(config_file)])
    plan = build_plan(config)
    assert not config.run_dir.exists()
    assert any(
        not check.ok for check in plan.checks
    )  # placeholder model paths fail closed


def test_formal_presets_share_base_checkpoint_and_data() -> None:
    central = _read_mapping(ROOT / "configs/experiment_a_central.yaml")
    shared = _read_mapping(ROOT / "configs/experiment_b_all_agents.yaml")
    assert central["hf_checkpoint"] == shared["hf_checkpoint"]
    assert central["reference_checkpoint"] == shared["reference_checkpoint"]
    assert central["data_path"] == shared["data_path"]
    assert central["train_steps"] == shared["train_steps"] == 100
    assert central["eval_interval"] == shared["eval_interval"] == 0
    assert central["reward_config"] == shared["reward_config"] == {"metric": "token_f1"}
    assert central["run_id"] != shared["run_id"]


def test_invalid_parallel_topology_fails_preflight(tmp_path: Path) -> None:
    raw = _read_mapping(ROOT / "configs/experiment_a_central.yaml")
    raw.update(
        {
            "run_id": "bad-topology",
            "output_root": str(tmp_path),
            "tensor_parallel_size": 3,
            "data_parallel_size": 8,
        }
    )
    config_file = tmp_path / "bad-topology.json"
    config_file.write_text(json.dumps(raw), encoding="utf-8")
    config, _, _ = resolve_config(["--config", str(config_file)])
    plan = build_plan(config)
    topology = next(check for check in plan.checks if check.name == "parallel topology")
    assert not topology.ok


def test_missing_production_custom_generate_hook_fails_closed(tmp_path: Path) -> None:
    raw = _read_mapping(ROOT / "configs/experiment_a_central.yaml")
    raw.update(
        {
            "run_id": "bad-rollout",
            "output_root": str(tmp_path),
            "custom_generate_function": "missing_package.missing_generate",
        }
    )
    config_file = tmp_path / "bad-rollout.json"
    config_file.write_text(json.dumps(raw), encoding="utf-8")
    config, _, _ = resolve_config(["--config", str(config_file)])
    plan = build_plan(config)
    generate = next(
        check
        for check in plan.checks
        if check.name == "StackPlanner custom-generate hook"
    )
    assert not generate.ok
    assert "--custom-generate-function-path" in plan.command
    assert "--rollout-function-path" not in plan.command


def test_launcher_uses_slime_native_stackplanner_hooks() -> None:
    config, _, _ = resolve_config(
        ["--config", str(ROOT / "configs/experiment_a_central.yaml")]
    )
    plan = build_plan(config)
    command = list(plan.command)
    assert command[command.index("--custom-generate-function-path") + 1] == (
        "slime_plugins.stackplanner.generate.generate"
    )
    assert command[command.index("--custom-reward-post-process-path") + 1] == (
        "slime_plugins.stackplanner.rewards.normalize_by_episode"
    )
    assert command[command.index("--global-batch-size") + 1] == str(
        config.rollout_batch_size * config.samples_per_prompt
    )
    assert command[command.index("--load") + 1] == str(
        Path(config.reference_checkpoint).resolve()
    )
    assert command[command.index("--start-rollout-id") + 1] == "0"
    assert "--use-tensorboard" in command
    assert command[command.index("--sglang-engine-port-base") + 1] == str(
        config.sglang_engine_port_base
    )
    assert plan.environment["SP_METRICS_JSONL"] == str(
        config.run_dir / "metrics/metrics.jsonl"
    )
    assert plan.environment["TENSORBOARD_DIR"] == str(
        config.run_dir / "metrics/tensorboard"
    )
    assert "targets/x86_64-linux/lib" in plan.environment["LIBRARY_PATH"]
    python_paths = plan.environment["PYTHONPATH"].split(":")
    assert python_paths[0] == str((ROOT / "third_party/Megatron-LM").resolve())


def test_router_cannot_overlap_reserved_sglang_engine_port_block(
    tmp_path: Path,
) -> None:
    raw = _read_mapping(ROOT / "configs/experiment_a_central.yaml")
    raw.update(
        {
            "run_id": "overlapping-engine-router-ports",
            "output_root": str(tmp_path),
            "sglang_engine_port_base": 30000,
            "sglang_router_port": 30001,
        }
    )
    config_file = tmp_path / "overlapping-ports.json"
    config_file.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="must not overlap"):
        resolve_config(["--config", str(config_file)])


def test_missing_megatron_checkout_fails_preflight(tmp_path: Path) -> None:
    raw = _read_mapping(ROOT / "configs/experiment_a_central.yaml")
    raw.update(
        {
            "run_id": "missing-megatron",
            "output_root": str(tmp_path),
            "megatron_path": str(tmp_path / "missing"),
        }
    )
    config_file = tmp_path / "missing-megatron.json"
    config_file.write_text(json.dumps(raw), encoding="utf-8")
    config, _, _ = resolve_config(["--config", str(config_file)])
    plan = build_plan(config)
    checkout = next(
        check for check in plan.checks if check.name == "Megatron repository"
    )
    assert not checkout.ok


def test_central_only_preflight_requires_independent_frozen_endpoint(
    tmp_path: Path,
) -> None:
    raw = _read_mapping(ROOT / "configs/experiment_a_central.yaml")
    raw.update(
        {
            "run_id": "missing-frozen-endpoint",
            "output_root": str(tmp_path),
            "frozen_base_url": "",
        }
    )
    config_file = tmp_path / "missing-frozen.json"
    config_file.write_text(json.dumps(raw), encoding="utf-8")
    config, _, _ = resolve_config(["--config", str(config_file)])

    isolation = next(
        check
        for check in build_plan(config).checks
        if check.name == "Central-only subagent isolation"
    )

    assert not isolation.ok
    assert "frozen backend configured=False" in isolation.detail


def test_ray_actor_runtime_env_inherits_training_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYTHONPATH", "/opt/Megatron-LM:/workspace/slime")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/opt/cuda/lib")
    monkeypatch.setenv("LIBRARY_PATH", "/opt/cuda/lib:/opt/cuda/lib/stubs")
    monkeypatch.setenv("DEER_FLOW_CONFIG_PATH", "/workspace/config.yaml")
    monkeypatch.setenv("SP_TRAINING_RUN_ID", "run-1")
    monkeypatch.setenv("UNRELATED_SECRET", "must-not-propagate")

    inherited = add_default_ray_env_vars({"PYTHONPATH": "explicit"})

    assert inherited["PYTHONPATH"] == "explicit"
    assert inherited["LD_LIBRARY_PATH"] == "/opt/cuda/lib"
    assert inherited["LIBRARY_PATH"] == "/opt/cuda/lib:/opt/cuda/lib/stubs"
    assert inherited["DEER_FLOW_CONFIG_PATH"] == "/workspace/config.yaml"
    assert inherited["SP_TRAINING_RUN_ID"] == "run-1"
    assert "UNRELATED_SECRET" not in inherited


def test_resume_checkpoint_replaces_base_actor_load(tmp_path: Path) -> None:
    raw = _read_mapping(ROOT / "configs/experiment_a_central.yaml")
    resume = tmp_path / "resumed-torch-dist"
    resume.mkdir()
    (resume / "latest_checkpointed_iteration.txt").write_text("19\n")
    raw.update(
        {
            "run_id": "resume-command",
            "output_root": str(tmp_path),
            "resume_from": str(resume),
        }
    )
    config_file = tmp_path / "resume.json"
    config_file.write_text(json.dumps(raw), encoding="utf-8")
    config, _, _ = resolve_config(["--config", str(config_file)])
    command = list(build_plan(config).command)
    assert command.count("--load") == 1
    assert command[command.index("--load") + 1] == str(resume)
    assert command[command.index("--start-rollout-id") + 1] == "20"
    assert command[command.index("--num-rollout") + 1] == "120"
    assert command.count("--use-checkpoint-opt-param-scheduler") == 1


def test_resume_tracker_rejects_release_and_invalid_values(tmp_path: Path) -> None:
    tracker = tmp_path / "latest_checkpointed_iteration.txt"
    tracker.write_text("release\n")
    with pytest.raises(ValueError, match="release checkpoint"):
        _checkpoint_next_rollout_id(tmp_path)
    tracker.write_text("not-an-iteration\n")
    with pytest.raises(ValueError, match="invalid resume"):
        _checkpoint_next_rollout_id(tmp_path)


def test_base_checkpoint_digest_is_exported_for_layer_map_identity(
    tmp_path: Path,
) -> None:
    raw = _read_mapping(ROOT / "configs/experiment_a_central.yaml")
    raw["base_checkpoint_sha256"] = "a" * 64
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(raw), encoding="utf-8")
    config, _, _ = resolve_config(["--config", str(config_file)])
    plan = build_plan(config)
    assert len(config.base_checkpoint_sha256) == 64
    assert plan.environment["SP_BASE_CHECKPOINT_SHA256"] == (
        config.base_checkpoint_sha256
    )


def test_successful_process_without_update_evidence_is_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    raw = _read_mapping(ROOT / "configs/experiment_a_central.yaml")
    raw.update(
        {
            "run_id": "audit-missing",
            "output_root": str(tmp_path),
            "data_path": str(ROOT / "fixtures/smoke_tasks.jsonl"),
            "dry_run": False,
        }
    )
    config_file = tmp_path / "audit.json"
    config_file.write_text(json.dumps(raw), encoding="utf-8")
    config, _, _ = resolve_config(["--config", str(config_file)])
    plan = build_plan(config)
    plan = type(plan)(
        plan.config,
        plan.policy_roles,
        plan.route_table,
        ("true",),
        plan.environment,
        (),
    )
    monkeypatch.setattr(
        "slime_plugins.stackplanner.cli._gpu_monitor",
        lambda _path, _stop: None,
    )
    monkeypatch.setattr(
        "slime_plugins.stackplanner.cli._search_r1_service_identity",
        lambda _config: {
            "endpoint": "http://search.test",
            "verified_at": "test",
        },
    )
    assert execute(plan) == 3
    summary = json.loads(
        (config.run_dir / "metrics/summary.json").read_text(encoding="utf-8")
    )
    assert summary["completion_verified"] is False
    assert "completion_audit_error" in summary
