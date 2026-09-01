#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
SP_ENV_PREFIX=${SP_ENV_PREFIX:-${HOME}/.cache/harness-rl/training-env}
SP_SOURCE_ROOT=${SP_SOURCE_ROOT:-${HOME}/.cache/harness-rl/sources}
python_bin="${SP_ENV_PREFIX}/bin/python"

if [[ ! -x ${python_bin} ]]; then
  echo "environment Python is missing: ${python_bin}" >&2
  exit 2
fi
source "${repo_root}/scripts/training_env.sh"
cd "${repo_root}"

SP_EXPECTED_REPO_ROOT=${repo_root} python - <<'PY'
import os
from pathlib import Path

import amp_C
import fake_int4_quant_cuda
import fused_layer_norm_cuda
import ray
import sglang
import slime_plugins
import torch
import transformer_engine.pytorch
from apex.optimizers import FusedAdam

expected_slime = (Path(os.environ["SP_EXPECTED_REPO_ROOT"]) / "slime").resolve()
actual_slime = Path(slime_plugins.__file__).resolve()
if not actual_slime.is_relative_to(expected_slime):
    raise SystemExit(
        f"slime editable install points at another checkout: {actual_slime}; "
        f"expected below {expected_slime}"
    )
assert torch.__version__ == "2.11.0+cu129", torch.__version__
assert torch.version.cuda == "12.9", torch.version.cuda
assert torch.cuda.is_available()
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
print(amp_C.__file__)
print(fused_layer_norm_cuda.__file__)
print(fake_int4_quant_cuda.__file__)
print(ray.__version__, sglang.__file__, FusedAdam)
print("slime checkout", actual_slime)
PY
python "${repo_root}/scripts/verify_training_runtime_dependencies.py"
python "${repo_root}/scripts/verify_training_dependency_consistency.py"

sglang_commit=0b3bb0cbe31873994c9f989fddfe2f87ca839fdd
megatron_commit=1dcf0dafa884ad52ffb243625717a3471643e087
[[ $(git -C "${SP_SOURCE_ROOT}/sglang" rev-parse HEAD) == "${sglang_commit}" ]]
[[ $(git -C "${SP_SOURCE_ROOT}/Megatron-LM" rev-parse HEAD) == "${megatron_commit}" ]]
# --full-index makes this fingerprint independent of each checkout's
# core.abbrev setting; the default abbreviated object IDs are not portable.
sglang_diff=$(git -C "${SP_SOURCE_ROOT}/sglang" diff --binary --full-index HEAD | sha256sum | awk '{print $1}')
[[ ${sglang_diff} == d31648a7092d6a218b76c4ee1f30d2bb5867ed1152301c56a8cb58253e969b99 ]]
sglang_local_checkpoint=$(sha256sum \
  "${SP_SOURCE_ROOT}/sglang/python/sglang/srt/weight_sync/local_checkpoint.py" | awk '{print $1}')
[[ ${sglang_local_checkpoint} == 0b8f184d4180e29d225458405b0091f91f35f8573a64147ec5ad65c15b316c54 ]]
sglang_bundle=$(
  cd "${SP_SOURCE_ROOT}/sglang"
  { git diff --name-only HEAD; git ls-files --others --exclude-standard; } \
    | LC_ALL=C sort -u \
    | while IFS= read -r source_file; do sha256sum "${source_file}"; done \
    | sha256sum \
    | awk '{print $1}'
)
[[ ${sglang_bundle} == c9010e89f2692fe6c67611b1d14319982d7971e630db234db63a50d60631f612 ]]
megatron_diff=$(git -C "${SP_SOURCE_ROOT}/Megatron-LM" diff --binary --full-index HEAD | sha256sum | awk '{print $1}')
[[ ${megatron_diff} == 620e5f7af8a7751f740c76d56e883623acdd1aac6b3a44316f799a0d1a676c78 ]]
megatron_bundle=$(
  cd "${SP_SOURCE_ROOT}/Megatron-LM"
  git diff --name-only HEAD \
    | LC_ALL=C sort -u \
    | while IFS= read -r source_file; do sha256sum "${source_file}"; done \
    | sha256sum \
    | awk '{print $1}'
)
[[ ${megatron_bundle} == b950d5ea6573bad52f9577bd9ca6c7c4a938e2586cb409a782043875b33be223 ]]

expected_patch_hashes=$(mktemp)
actual_patch_hashes=$(mktemp)
trap 'rm -f "${expected_patch_hashes}" "${actual_patch_hashes}"' EXIT
printf '%s\n' \
  '4c1b47f52f7a4fd025739851ecc38f9b5bd437054aab006365e9c455eb8739bf  sglang.patch' \
  'fdb070bcf92849167bea1afb821cbc9974d6fcaf6a2c353e47ac68091c5c16b1  sglang-top_p.patch' \
  'e70a3dc140806b0a985c4d856e741600ae8cf89ee664e4dac6460318e22b1a7d  sglang-release_hicache.patch' \
  'bca438816b4a91778fad2c34b08c9fabd15bc571645c74b457d26cf6f7b2f4c3  sglang-pull_weights.patch' \
  'ecd92d0d6168440cb65d74ab369fa86f5b16feea0847d404f333862a2d14c17b  sglang-openai.patch' \
  '976fd8bbe1f4e386be97e35353efdbfb592dacaea9797c5a81ea324d3c2c5b77  megatron.patch' \
  >"${expected_patch_hashes}"
(
  cd "${repo_root}/slime/docker/patch/v0.5.15.post1"
  sha256sum \
    sglang.patch \
    sglang-top_p.patch \
    sglang-release_hicache.patch \
    sglang-pull_weights.patch \
    sglang-openai.patch \
    megatron.patch
) >"${actual_patch_hashes}"
diff -u "${expected_patch_hashes}" "${actual_patch_hashes}"

env -u PYTHONPATH python -m pytest -o addopts='' "${repo_root}/tests" -q
python -m ruff check \
  "${repo_root}/train_stackplanner.py" \
  "${repo_root}/scripts" \
  "${repo_root}/slime/slime_plugins/stackplanner" \
  "${repo_root}/tests"
python -m ruff format --check \
  "${repo_root}/train_stackplanner.py" \
  "${repo_root}/scripts" \
  "${repo_root}/slime/slime_plugins/stackplanner" \
  "${repo_root}/tests"
python -m compileall -q \
  "${repo_root}/train_stackplanner.py" \
  "${repo_root}/scripts" \
  "${repo_root}/slime/slime_plugins/stackplanner"
python "${repo_root}/train_stackplanner.py" --help >/dev/null
echo "training environment verification passed"
