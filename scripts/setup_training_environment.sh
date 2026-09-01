#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
SP_ENV_PREFIX=${SP_ENV_PREFIX:-${HOME}/.cache/harness-rl/training-env}
SP_SOURCE_ROOT=${SP_SOURCE_ROOT:-${HOME}/.cache/harness-rl/sources}
SP_BUILD_CACHE=${SP_BUILD_CACHE:-${HOME}/.cache/harness-rl/build}
SP_BUILD_LOG=${SP_BUILD_LOG:-$SP_BUILD_CACHE/setup.log}
SP_RESUME_BUILD=${SP_RESUME_BUILD:-0}
mkdir -p "${SP_BUILD_CACHE}" "${SP_SOURCE_ROOT}" "$(dirname "${SP_BUILD_LOG}")"

if [[ -e ${SP_ENV_PREFIX} && ${SP_RESUME_BUILD} != 1 ]]; then
  echo "environment exists: ${SP_ENV_PREFIX}" >&2
  echo "use 'make resume' to continue it or choose SP_ENV_PREFIX for a clean build" >&2
  exit 2
fi

{
  echo "started_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "repo=${repo_root}"
  echo "commit=$(git -C "${repo_root}" rev-parse HEAD)"
  echo "environment=${SP_ENV_PREFIX}"
  echo "sources=${SP_SOURCE_ROOT}"
  echo "cache=${SP_BUILD_CACHE}"
  export BASE_DIR=${SP_SOURCE_ROOT}
  export SLIME_DIR=${repo_root}/slime
  export MAMBA_ENV_PREFIX=${SP_ENV_PREFIX}
  export SP_BUILD_CACHE SP_RESUME_BUILD
  bash "${repo_root}/slime/build_conda.sh"
  SP_ENV_PREFIX=${SP_ENV_PREFIX} SP_SOURCE_ROOT=${SP_SOURCE_ROOT} \
    bash "${repo_root}/scripts/install_stackplanner_training_runtime.sh"
  echo "finished_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} 2>&1 | tee -a "${SP_BUILD_LOG}"

SP_ENV_PREFIX=${SP_ENV_PREFIX} SP_SOURCE_ROOT=${SP_SOURCE_ROOT} \
  bash "${repo_root}/scripts/verify_training_environment.sh"
