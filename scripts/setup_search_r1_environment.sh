#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
SP_SEARCH_R1_ROOT=${SP_SEARCH_R1_ROOT:-${HOME}/.cache/harness-rl/search-r1}
SP_SEARCH_R1_ENV=${SP_SEARCH_R1_ENV:-${SP_SEARCH_R1_ROOT}/venv}
SP_SEARCH_R1_STATE_ROOT=${SP_SEARCH_R1_STATE_ROOT:-${SP_SEARCH_R1_ROOT}/state}
SP_SEARCH_R1_RESUME=${SP_SEARCH_R1_RESUME:-0}
SP_SEARCH_R1_REQUIREMENTS=${SP_SEARCH_R1_REQUIREMENTS:-${repo_root}/requirements/search-r1-retriever.txt}
SP_SEARCH_R1_BUILD_CACHE=${SP_SEARCH_R1_BUILD_CACHE:-${HOME}/.cache/harness-rl/search-r1-build}
MICROMAMBA_EXE=${MICROMAMBA_EXE:-${HOME}/.local/bin/micromamba}

if [[ ! -f ${SP_SEARCH_R1_REQUIREMENTS} ]]; then
  echo "Search-R1 requirements are missing: ${SP_SEARCH_R1_REQUIREMENTS}" >&2
  exit 2
fi
if [[ -e ${SP_SEARCH_R1_ENV} && ${SP_SEARCH_R1_RESUME} != 1 ]]; then
  echo "Search-R1 environment exists; set SP_SEARCH_R1_RESUME=1 to continue it" >&2
  exit 2
fi
if [[ ! -e ${SP_SEARCH_R1_ENV} ]]; then
  if [[ ! -x ${MICROMAMBA_EXE} ]]; then
    echo "micromamba is missing: ${MICROMAMBA_EXE}; build the trainer environment first" >&2
    exit 2
  fi
  MAMBA_ROOT_PREFIX="${SP_SEARCH_R1_BUILD_CACHE}/micromamba" \
    "${MICROMAMBA_EXE}" create -p "${SP_SEARCH_R1_ENV}" \
    python=3.12 pip -c conda-forge -y
elif [[ ! -x ${SP_SEARCH_R1_ENV}/bin/python ]]; then
  echo "partial Search-R1 environment has no Python: ${SP_SEARCH_R1_ENV}" >&2
  exit 2
fi
python_bin="${SP_SEARCH_R1_ENV}/bin/python"
mkdir -p "${SP_SEARCH_R1_STATE_ROOT}" "${SP_SEARCH_R1_BUILD_CACHE}/pip"
export PIP_CACHE_DIR="${SP_SEARCH_R1_BUILD_CACHE}/pip"

"${python_bin}" -m pip install --upgrade "pip==26.0.1"
"${python_bin}" -m pip install \
  "torch==2.8.0" "torchvision==0.23.0" "torchaudio==2.8.0" \
  --index-url https://download.pytorch.org/whl/cu126
"${python_bin}" -m pip install "faiss-gpu-cu12==1.11.0"
"${python_bin}" -m pip install -r "${SP_SEARCH_R1_REQUIREMENTS}"

SP_SEARCH_R1_ENV=${SP_SEARCH_R1_ENV} \
  bash "${repo_root}/scripts/verify_search_r1_environment.sh"
"${python_bin}" -m pip freeze >"${SP_SEARCH_R1_STATE_ROOT}/requirements.freeze.txt"
{
  printf 'created_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'repo_commit=%s\n' "$(git -C "${repo_root}" rev-parse HEAD)"
  printf 'environment=%s\n' "${SP_SEARCH_R1_ENV}"
  printf 'requirements_sha256=%s\n' "$(sha256sum "${SP_SEARCH_R1_REQUIREMENTS}" | awk '{print $1}')"
} >"${SP_SEARCH_R1_STATE_ROOT}/environment-manifest.env"
echo "Search-R1 environment passed: ${SP_SEARCH_R1_ENV}"
