from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def test_public_setup_uses_configurable_cache_locations() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    setup = (ROOT / "scripts/setup_training_environment.sh").read_text(
        encoding="utf-8"
    )
    search_setup = (ROOT / "scripts/setup_search_r1_environment.sh").read_text(
        encoding="utf-8"
    )

    assert "$(HOME)/.cache/harness-rl" in makefile
    assert "${HOME}/.cache/harness-rl" in setup
    assert "${HOME}/.cache/harness-rl" in search_setup
    assert "SP_ENV_PREFIX" in setup
    assert "SP_SOURCE_ROOT" in setup
    assert "SP_SEARCH_R1_ROOT" in search_setup


def test_public_makefile_exposes_only_reproducible_workflows() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")

    for target in (
        "doctor:",
        "setup:",
        "verify:",
        "prepare-data:",
        "prepare-eval:",
        "search-setup:",
        "preflight:",
        "train:",
        "test:",
        "lint:",
    ):
        assert target in makefile


def test_service_launchers_are_configurable_and_repo_relative() -> None:
    configure = (ROOT / "scripts/configure_stackplanner_runtime.sh").read_text(
        encoding="utf-8"
    )
    frozen = (ROOT / "scripts/start_frozen_sglang_server.sh").read_text(
        encoding="utf-8"
    )
    search = (ROOT / "scripts/start_search_r1_retriever.sh").read_text(
        encoding="utf-8"
    )
    ray = (ROOT / "scripts/start_local_training_ray.sh").read_text(encoding="utf-8")

    assert "SP_SERVICE_HOST" in configure
    assert "SP_FROZEN_MODEL_PATH" in frozen
    assert "SP_FROZEN_CUDA_DEVICES" in frozen
    assert "SP_SEARCH_R1_CUDA_DEVICES" in search
    assert "SP_RAY_NUM_GPUS" in ray
    for script in (configure, frozen, search, ray):
        assert "repo_root=" in script
