#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
SP_ENV_PREFIX=${SP_ENV_PREFIX:-${HOME}/.cache/harness-rl/training-env}
SP_SOURCE_ROOT=${SP_SOURCE_ROOT:-${HOME}/.cache/harness-rl/sources}

printf 'repo=%s\n' "${repo_root}"
printf 'repo_commit=%s\n' "$(git -C "${repo_root}" rev-parse HEAD)"
printf 'repo_dirty=%s\n' "$([[ -n $(git -C "${repo_root}" status --short) ]] && printf true || printf false)"
printf 'environment=%s\n' "${SP_ENV_PREFIX}"
if [[ -x ${SP_ENV_PREFIX}/bin/python ]]; then
  "${SP_ENV_PREFIX}/bin/python" - <<'PY'
from importlib.metadata import PackageNotFoundError, version
for package in ("torch", "ray", "sglang", "transformer-engine", "deerflow-harness"):
    try:
        print(f"{package}={version(package)}")
    except PackageNotFoundError:
        print(f"{package}=MISSING")
PY
else
  echo 'environment_state=MISSING'
fi
for source_name in sglang Megatron-LM; do
  source_dir="${SP_SOURCE_ROOT}/${source_name}"
  if git -C "${source_dir}" rev-parse --git-dir >/dev/null 2>&1; then
    printf '%s_commit=%s\n' "${source_name}" "$(git -C "${source_dir}" rev-parse HEAD)"
    printf '%s_dirty=%s\n' "${source_name}" "$([[ -n $(git -C "${source_dir}" status --short) ]] && printf true || printf false)"
  else
    printf '%s=MISSING\n' "${source_name}"
  fi
done
