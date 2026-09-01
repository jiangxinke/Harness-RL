#!/usr/bin/env bash

set -euxo pipefail

export SLIME_DIR="${SLIME_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"

# create conda
export MAMBA_ENV_NAME="${MAMBA_ENV_NAME:-slime}"
export MICROMAMBA_EXE="${MICROMAMBA_EXE:-$HOME/.local/bin/micromamba}"
export SP_RESUME_BUILD="${SP_RESUME_BUILD:-0}"
export SP_BUILD_CACHE="${SP_BUILD_CACHE:-$HOME/.cache/stackplanner-rl}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$SP_BUILD_CACHE/pip}"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$SP_BUILD_CACHE/micromamba}"
export CARGO_HOME="${CARGO_HOME:-$SP_BUILD_CACHE/cargo}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$SP_BUILD_CACHE/torch-extensions}"
export HF_HOME="${HF_HOME:-$SP_BUILD_CACHE/huggingface}"
# The conda cuda-nvcc activation hook checks whether these variables exist but
# then expands them unconditionally. Define empty defaults so strict nounset
# mode remains usable with an otherwise clean shell.
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"
export NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:-}"
mkdir -p \
  "$PIP_CACHE_DIR" \
  "$MAMBA_ROOT_PREFIX" \
  "$CARGO_HOME" \
  "$TORCH_EXTENSIONS_DIR" \
  "$HF_HOME"
if [ ! -x "$MICROMAMBA_EXE" ]; then
  yes '' | "${SHELL}" <(curl -L micro.mamba.pm/install.sh)
fi
export PS1=tmp
touch "$CARGO_HOME/env"
eval "$("$MICROMAMBA_EXE" shell hook --shell bash)"

# The micromamba installer writes `nodefaults` into ~/.condarc as a channel
# entry, which newer micromamba versions try to fetch as a real anaconda.org
# repo (it isn't — it's a meta-tag) and time out on. Strip it.
if [ -f ~/.condarc ]; then
  sed -i '/^\s*-\s*nodefaults\s*$/d' ~/.condarc
fi

if [ -n "${MAMBA_ENV_PREFIX:-}" ]; then
  env_target=(-p "$MAMBA_ENV_PREFIX")
  env_activation_target="$MAMBA_ENV_PREFIX"
else
  env_target=(-n "$MAMBA_ENV_NAME")
  env_activation_target="$MAMBA_ENV_NAME"
fi
if [ -e "${MAMBA_ENV_PREFIX:-$MAMBA_ROOT_PREFIX/envs/$MAMBA_ENV_NAME}" ]; then
  if [ "$SP_RESUME_BUILD" != 1 ]; then
    echo "environment already exists; set SP_RESUME_BUILD=1 to continue it" >&2
    exit 2
  fi
  if [ ! -x "${MAMBA_ENV_PREFIX:-$MAMBA_ROOT_PREFIX/envs/$MAMBA_ENV_NAME}/bin/python" ]; then
    echo "existing environment is incomplete and has no Python; choose a new prefix" >&2
    exit 2
  fi
else
  "$MICROMAMBA_EXE" create "${env_target[@]}" python=3.12 pip -c conda-forge -y
fi
micromamba activate "$env_activation_target"
export CUDA_HOME="$CONDA_PREFIX"

# Keep these in sync with docker/Dockerfile:
#   - SGLANG_IMAGE_TAG (ARG)            -> SGLANG_VERSION below
#   - MEGATRON_COMMIT (ARG)             -> MEGATRON_COMMIT below
#   - PATCH_VERSION (ARG, default "latest") -> PATCH_VERSION below
#   - TMS_COMMIT (ARG)                   -> TMS_COMMIT below
export SGLANG_VERSION="v0.5.15.post1"
export SGLANG_COMMIT="0b3bb0cbe31873994c9f989fddfe2f87ca839fdd"
export MEGATRON_COMMIT="1dcf0dafa884ad52ffb243625717a3471643e087"
export PATCH_VERSION="v0.5.15.post1"
export TMS_COMMIT="8d30c59ca12a68d9deccbc9c6599076a1218cbc5"
export FLASH_QLA_COMMIT="821fd9d37ede18fdc2a4e707fefe3770bfc32e58"

export BASE_DIR=${BASE_DIR:-"$HOME"}
mkdir -p "$BASE_DIR"
cd "$BASE_DIR"

# install cuda 12.9 as it's the default cuda version for torch
"$MICROMAMBA_EXE" install -p "$CONDA_PREFIX" \
  cuda=12.9.1 \
  cuda-nvtx=12.9.79 \
  cuda-nvtx-dev=12.9.79 \
  nccl \
  -c nvidia/label/cuda-12.9.1 \
  -c nvidia \
  -c conda-forge \
  -y
# Do not install conda-forge cuDNN here.  The PyTorch cu129 wheel pins and
# supplies its matching pip runtime.  A second conda cuDNN under
# $CONDA_PREFIX/lib takes precedence through Transformer Engine's RPATH and can
# mix incompatible component libraries in one process.
# sglang's editable install builds a Rust extension (sglang-grpc via
# setuptools-rust). It needs rustc/cargo plus the protoc binary at wheel-build
# time; the Python protobuf package alone does not provide protoc.
"$MICROMAMBA_EXE" install -p "$CONDA_PREFIX" -c conda-forge rust "libprotobuf=3.20.3" -y
if [ ! -x "$CONDA_PREFIX/bin/protoc" ]; then
  echo "conda libprotobuf installation did not provide the required protoc binary" >&2
  exit 1
fi
"$CONDA_PREFIX/bin/protoc" --version

pip install cuda-python==12.9.7

# install sglang. The Dockerfile starts FROM slimerl/sglang:v0.5.15.post1-cu129
# which already has sglang installed with cu129-built native kernels; we have
# to install it ourselves here. Two follow-up steps clean up the cu13 spill:
#   1. force-reinstall torch / sglang-kernel / sgl-deep-gemm to their +cu129
#      wheels (pypi defaults are cu13);
#   2. uninstall the cu13 nvidia-* runtime libs sglang dragged in, then
#      reinstall the cu12 equivalents to repair the `site-packages/nvidia/*`
#      shared dirs (pip uninstall stomps libs co-owned across cu12/cu13).
if [ ! -d "$BASE_DIR/sglang" ]; then
  cd $BASE_DIR
  git clone https://github.com/sgl-project/sglang.git
fi
cd $BASE_DIR/sglang
git checkout ${SGLANG_COMMIT}
pip install -e "python[all]" --extra-index-url https://download.pytorch.org/whl/cu129
pip install --force-reinstall --no-deps \
  torch==2.11.0+cu129 torchvision==0.26.0+cu129 torchaudio==2.11.0+cu129 \
  --index-url https://download.pytorch.org/whl/cu129
pip install --force-reinstall --no-deps \
  sglang-kernel==0.4.4 sgl-deep-gemm==0.1.4 \
  --index-url https://docs.sglang.ai/whl/cu129/
pip uninstall -y \
  nvidia-cublas \
  nvidia-cuda-cupti \
  nvidia-cuda-nvrtc \
  nvidia-cuda-runtime \
  nvidia-cudnn-cu13 \
  nvidia-cufft \
  nvidia-cufile \
  nvidia-curand \
  nvidia-cusolver \
  nvidia-cusparse \
  nvidia-cusparselt-cu13 \
  nvidia-nccl-cu13 \
  nvidia-nvjitlink \
  nvidia-nvshmem-cu13 \
  nvidia-nvtx \
  nvidia-cutlass-dsl-libs-cu13 \
  || true
pip install --force-reinstall --no-deps \
  nvidia-cublas-cu12==12.9.2.10 \
  nvidia-cuda-cupti-cu12 \
  nvidia-cuda-nvrtc-cu12 \
  nvidia-cuda-runtime-cu12 \
  nvidia-cudnn-cu12==9.17.1.4 \
  nvidia-cufft-cu12 \
  nvidia-cufile-cu12 \
  nvidia-curand-cu12 \
  nvidia-cusolver-cu12 \
  nvidia-cusparse-cu12 \
  nvidia-cusparselt-cu12 \
  nvidia-nccl-cu12 \
  nvidia-nvjitlink-cu12 \
  nvidia-nvshmem-cu12 \
  nvidia-nvtx-cu12 \
  --index-url https://download.pytorch.org/whl/cu129 \
  --extra-index-url https://pypi.org/simple


pip install cmake ninja

# The SGLang dependency set includes FA4, while the validated TE 2.16 CP stack
# uses FA2 2.7.4 through 2.8.3. This wheel targets Python 3.12, PyTorch 2.11,
# CUDA 12, and the CXX11 ABI used by the conda environment. Upstream does not
# publish that combination, so pin the community build linked from upstream
# issue #2425 by its SHA256 digest.
pip uninstall -y flash-attn-4 flash_attn_4 || true
pip install --no-deps \
  "https://github.com/lesj0610/flash-attention/releases/download/v2.8.3-cu12-torch2.11/flash_attn-2.8.3%2Bcu12torch2.11cxx11abiTRUE-cp312-cp312-linux_x86_64.whl#sha256=3d0c8e60f820321eedd7166e79c33cb816263d8be6e35c3f5ba8fe2df6fea697"

pip install --no-deps fla-core==0.4.2 flash-linear-attention==0.4.2
# FlashQLA requires SM90+.  The production A100 (SM80) training fleet cannot
# import it, and installing it also downgrades TileLang below SGLang's lock.
# Keep the explicit override for portable builds on supported Hopper hardware.
SP_INSTALL_FLASH_QLA="${SP_INSTALL_FLASH_QLA:-$(python - <<'PY'
import torch

print("1" if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 9 else "0")
PY
)}"
if [ "$SP_INSTALL_FLASH_QLA" = 1 ]; then
  pip install --no-deps \
    git+https://github.com/QwenLM/FlashQLA.git@${FLASH_QLA_COMMIT} \
    --no-build-isolation
fi
# tilelang (matches Dockerfile)
pip install --no-deps tilelang -f https://tile-ai.github.io/whl/nightly/cu128/

# Transformer Engine 2.16.1 has no prebuilt wheel for the exact Python 3.12 /
# torch 2.11.0+cu129 tag and falls back to compiling its PyTorch extension. The
# torch-pinned pip cuDNN runtime owns headers below site-packages/nvidia/cudnn,
# not under the conda CUDA prefix. Expose both headers and libraries before TE
# compiles; this keeps the single torch-pinned cuDNN runtime while avoiding a
# second conda cuDNN installation.
PYTHON_SITE=$(python -c 'import site; print(site.getsitepackages()[0])')
export CPATH="$PYTHON_SITE/nvidia/cudnn/include:$CUDA_HOME/targets/x86_64-linux/include${CPATH:+:$CPATH}"
export CPLUS_INCLUDE_PATH="$PYTHON_SITE/nvidia/cudnn/include:$CUDA_HOME/targets/x86_64-linux/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
export CFLAGS="-I$PYTHON_SITE/nvidia/cudnn/include -I$CUDA_HOME/targets/x86_64-linux/include ${CFLAGS:-}"
export CXXFLAGS="-I$PYTHON_SITE/nvidia/cudnn/include -I$CUDA_HOME/targets/x86_64-linux/include ${CXXFLAGS:-}"
export LDFLAGS="-L$PYTHON_SITE/nvidia/cudnn/lib -L$CUDA_HOME/targets/x86_64-linux/lib ${LDFLAGS:-}"
pip install --no-build-isolation "transformer_engine[pytorch]==2.16.1"

# PyTorch's extension helper forwards CUDA_HOME to nvcc, but Apex also has
# host-C++ translation units that include CUDA headers directly.  The conda
# toolkit keeps those headers below targets/x86_64-linux, so expose that path
# to both compilers explicitly.
# Retain the TE header/library paths above for Apex's host-C++ translation
# units as well.
NVCC_APPEND_FLAGS="--threads 4" \
  pip -v install --disable-pip-version-check --no-cache-dir \
  --no-build-isolation \
  --config-settings "--build-option=--cpp_ext --cuda_ext --parallel 8" git+https://github.com/NVIDIA/apex.git@10417aceddd7d5d05d7cbf7b0fc2daad1105f8b4

TMS_CUDA_MAJOR="${TMS_CUDA_MAJOR:-$(python -c 'import torch; print(torch.version.cuda.split(".")[0])')}"
export TMS_CUDA_MAJOR
# --no-build-isolation: TMS's setup.py needs to find nvcc + headers + the
# installed torch to build its cu${TMS_CUDA_MAJOR} native hook; pip's default
# PEP 517 build venv hides them, so the wheel comes out python-only (~46KB)
# and sglang trips `Only hook_mode=preload supports pauseable CUDA Graph`
# because the preload .so was never compiled in.
pip install -v git+https://github.com/zhuzilin/torch_memory_saver.git@${TMS_COMMIT} \
  --no-cache-dir --force-reinstall --no-build-isolation
pip install "nvidia-modelopt[torch]>=0.37.0" --no-build-isolation
pip install https://github.com/zhuzilin/sgl-router/releases/download/v0.3.2-9daabcd/sglang_router-0.3.2-cp38-abi3-manylinux_2_28_x86_64.whl --force-reinstall
python -c "import sglang_router; assert 'slime' in sglang_router.__version__"

# megatron
cd $BASE_DIR
if [ ! -d "$BASE_DIR/Megatron-LM" ]; then
  git clone https://github.com/NVIDIA/Megatron-LM.git --recursive
fi
# pre-install Megatron's build deps explicitly since we use --no-build-isolation
pip install "setuptools<80.0.0" pybind11 "packaging>=24.2"
# --no-build-isolation: setup.py builds a C++ extension (megatron.core.datasets.helpers_cpp)
# that subprocess-shells `python3 -m pybind11`; without isolation pip uses the
# current env's python which already has pybind11 installed. Otherwise the ext
# is marked optional and silently skipped, which breaks GPT dataset loading.
cd $BASE_DIR/Megatron-LM && git checkout ${MEGATRON_COMMIT} && pip install -e . --no-build-isolation

# install slime and apply patches

cd $SLIME_DIR
# Install slime's pure-python runtime deps first (wandb, ray, accelerate,
# transformers, etc.) from its requirements.txt, then install slime itself
# with --no-deps so pip doesn't re-resolve and stomp our pinned native libs
# (torch+cu129, sglang-kernel+cu129, ...). The Dockerfile does the same thing
# in two RUN layers (line ~71 + line ~124).
pip install -r requirements.txt
pip install -e . --no-deps

# requirements.txt may let pip replace the commit-pinned TMS build above with
# an older PyPI wheel.  That implementation rejects nested region() calls when
# preload mode starts with TMS_INIT_ENABLE=1, while the Megatron patch relies on
# nested regions to mark param/grad buffers as no-CPU-backup allocations.
# Reassert the pinned source after dependency resolution so Python and the
# native preload hook come from the same validated revision.
pip install -v git+https://github.com/zhuzilin/torch_memory_saver.git@${TMS_COMMIT} \
  --no-deps --no-cache-dir --force-reinstall --no-build-isolation

# int4_qat kernel (matches Dockerfile)
cd $SLIME_DIR/slime/backends/megatron_utils/kernels/int4_qat
pip install . --no-build-isolation

# Reassert the native runtime without dependency resolution after the Python
# requirements layer.  torch 2.11.0+cu129 declares cuDNN 9.17.1.4 exactly; an
# older 9.16 runtime is rejected before training starts.
pip install --force-reinstall --no-deps \
  nvidia-cublas-cu12==12.9.2.10 \
  nvidia-cudnn-cu12==9.17.1.4
pip install "numpy==1.26.4" "scipy==1.17.1"
# kernels 0.15.x trips a ValueError("Either a revision or a version must be
# specified") on `transformers.integrations.hub_kernels` import; pin to <0.15
# so `import sglang` works at runtime.
pip install "kernels<0.15.0"

# apply patches in the same order as Dockerfile
patch_dir="$SLIME_DIR/docker/patch/${PATCH_VERSION}"
if [ ! -d "$patch_dir" ]; then
  echo "Patch directory does not exist: $patch_dir" >&2
  exit 1
fi

cd $BASE_DIR/sglang
for patch_name in sglang.patch sglang-top_p.patch sglang-release_hicache.patch sglang-pull_weights.patch sglang-openai.patch; do
  patch_path="$patch_dir/${patch_name}"
  if [ ! -f "$patch_path" ]; then
    continue
  fi
  if git apply --check "$patch_path"; then
    git apply "$patch_path"
  elif git apply --reverse --check "$patch_path"; then
    echo "$patch_name already applied, skipping"
  else
    echo "$patch_name does not apply cleanly" >&2
    exit 1
  fi
done
cd $BASE_DIR/Megatron-LM
megatron_patch="$patch_dir/megatron.patch"
if [ ! -f "$megatron_patch" ]; then
  echo "Megatron patch does not exist: $megatron_patch" >&2
  exit 1
fi
if git apply --reverse --check "$megatron_patch"; then
  echo "megatron.patch already applied, skipping"
else
  git update-index --refresh || true
  if ! git apply "$megatron_patch" --3way; then
    echo "megatron.patch does not apply cleanly" >&2
    exit 1
  fi
  if git grep -n '^<<<<<<< ' -- .; then
    echo "megatron patch failed to apply cleanly. Please resolve conflicts." >&2
    exit 1
  fi
fi

python - <<'PY'
import inspect
import sglang
import torch
import torchaudio
import torchvision
from torch_memory_saver.entrypoint import _TorchMemorySaverImpl

assert torch.__version__ == "2.11.0+cu129"
assert torchaudio.__version__ == "2.11.0+cu129"
assert torchvision.__version__ == "0.26.0+cu129"
assert hasattr(torch.ops.torchvision, "nms")
assert "orig_interesting_region" in inspect.getsource(
    _TorchMemorySaverImpl._with_region_config
), "torch_memory_saver lacks nested preload-region support"
PY
