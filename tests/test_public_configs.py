from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
CONFIG_NAMES = (
    "experiment_a_central.yaml",
    "experiment_b_all_agents.yaml",
    "experiment_c_capability_decoupled.yaml",
)


def _load(name: str) -> dict:
    return json.loads((ROOT / "configs" / name).read_text(encoding="utf-8"))


def test_public_training_configs_cover_supported_modes() -> None:
    configs = [_load(name) for name in CONFIG_NAMES]

    assert [config["experiment"] for config in configs] == ["A", "B", "C"]
    assert [config["mode"] for config in configs] == [
        "central-only",
        "shared-all-agents",
        "central-only",
    ]
    assert [config["capability_decoupling"] for config in configs] == [
        False,
        True,
        True,
    ]


def test_public_training_configs_use_portable_paths() -> None:
    path_keys = (
        "stackplanner_path",
        "slime_path",
        "megatron_path",
        "deer_flow_config_path",
        "data_path",
        "output_root",
        "hf_checkpoint",
        "reference_checkpoint",
        "tokenizer",
    )

    for name in CONFIG_NAMES:
        config = _load(name)
        assert config["dry_run"] is True
        for key in path_keys:
            assert not Path(config[key]).is_absolute(), (name, key, config[key])


def test_search_configuration_uses_a_portable_loopback_default() -> None:
    import yaml

    config = yaml.safe_load(
        (ROOT / "configs/deerflow_search_r1_only.yaml").read_text(encoding="utf-8")
    )
    assert [tool["name"] for tool in config["tools"]] == ["web_search"]
    assert config["tools"][0]["base_url"] == "http://127.0.0.1:8000"
    assert config["sandbox"]["allow_host_bash"] is False
