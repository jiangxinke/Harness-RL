#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
SP_ENV_PREFIX=${SP_ENV_PREFIX:-${HOME}/.cache/harness-rl/training-env}
SP_SOURCE_ROOT=${SP_SOURCE_ROOT:-${HOME}/.cache/harness-rl/sources}
constraints="${repo_root}/requirements/training-runtime-constraints.txt"
runtime_lock="${repo_root}/requirements/training-runtime.lock"
source "${repo_root}/scripts/training_env.sh"
sglang_patch="${repo_root}/slime/docker/patch/v0.5.15.post1/sglang-openai.patch"
sglang_source="${SP_SOURCE_ROOT}/sglang"
if ! git -C "${sglang_source}" rev-parse --git-dir >/dev/null 2>&1; then
  echo "SGLang source is missing: ${sglang_source}" >&2
  exit 2
fi
if git -C "${sglang_source}" apply --check "${sglang_patch}"; then
  git -C "${sglang_source}" apply "${sglang_patch}"
elif git -C "${sglang_source}" apply --reverse --check "${sglang_patch}"; then
  echo "sglang-openai.patch already applied, skipping"
else
  echo "sglang-openai.patch does not apply cleanly" >&2
  exit 1
fi
# Refresh editable metadata without resolving SGLang's CUDA dependencies.
python -m pip install --editable "${sglang_source}/python" \
  --no-deps --no-build-isolation

# Install only the embedded graph runtime.  The full DeerFlow HTTP server uses
# the harness `server` extra in StackPlanner2.0/backend and has a gRPC contract
# that cannot coexist with SGLang in this process.  The complete transitive
# closure is hashed so a later PyPI release cannot silently change a rebuild.
python -m pip install --require-hashes --requirement "${runtime_lock}"
python -m pip install --no-deps --editable \
  "${repo_root}/StackPlanner2.0/backend/packages/harness"
python -m pip uninstall -y \
  langgraph-api \
  langgraph-cli \
  langgraph-runtime-inmem \
  grpcio-tools \
  flash-qla \
  || true

# flashinfer-python 0.6.12 unconditionally declares cuda-tile[tileiras].  That
# brings CUDA 13 compiler-only tools even for its cu12 extra.  Keep those tools
# because removing them makes the installed FlashInfer metadata inconsistent;
# their runtime dependencies remain deliberately absent and the training
# contract keeps all CUDA runtime/library lookup rooted in the conda CUDA 12.9
# prefix.  The exact upstream metadata defects are audited by
# verify_training_dependency_consistency.py.

python - <<'PY'
from importlib.metadata import PackageNotFoundError, version

expected = {
    "grpcio": "1.83.0",
    "grpcio-health-checking": "1.83.0",
    "grpcio-reflection": "1.83.0",
    "numpy": "1.26.4",
    "openai": "2.54.0",
    "pillow": "11.3.0",
    "protobuf": "7.35.1",
    "setuptools": "80.9.0",
}
for package, wanted in expected.items():
    actual = version(package)
    if actual != wanted:
        raise SystemExit(f"{package} mismatch: expected {wanted}, got {actual}")
for forbidden in (
    "langgraph-api",
    "langgraph-cli",
    "langgraph-runtime-inmem",
    "grpcio-tools",
):
    try:
        actual = version(forbidden)
    except PackageNotFoundError:
        continue
    raise SystemExit(f"training prefix contains server-only package {forbidden} {actual}")

import grpc
import langchain_openai
import langgraph
import openai
import sglang

print(
    "StackPlanner training runtime passed",
    grpc.__version__,
    openai.__version__,
    langchain_openai.__file__,
    langgraph.__path__,
    sglang.__file__,
)
PY
python "${repo_root}/scripts/verify_training_dependency_consistency.py"
