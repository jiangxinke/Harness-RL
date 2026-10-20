# Reproducing the optional native LoongReflect runtime

This describes the Linux x86-64, Python 3.12 CUDA 12.9 runtime for the production
Megatron trainer and SGLang rollout engine. The versions below have passed actual
isolated installation and imports, including Transformer Engine's compiled
PyTorch extension, the custom router, Megatron training, and the production
LoongReflect actor. **That establishes runtime preparation, not a completed GPU
training run or a benchmark score.** Production end-to-end acceptance is a
separate check described below.

Use [loongreflect.md](loongreflect.md) for the algorithm, loss modes, data format,
and launcher. This runtime retains full-parameter Megatron training. LoRA in
other training paths does not imply LoRA support in this optional integration.

## Isolated environment and constraints

Create a fresh virtual environment without `--system-site-packages`. Keep the
repository, downloaded wheels, cloned dependency source, and durable run outputs
in directories selected for your own installation:

```bash
python3.12 -m venv .venv-loongreflect
source .venv-loongreflect/bin/activate
export REPO_ROOT="$PWD"
export RUNTIME_DIR="$PWD/.loongreflect-runtime"
export ARTIFACT_ROOT="/path/to/persistent/loongreflect-runs"
export PIP_CACHE_DIR="${TMPDIR:-/tmp}/loongreflect-pip-cache"
export BUILD_TMP="${TMPDIR:-/tmp}/loongreflect-build"
mkdir -p "$RUNTIME_DIR" "$ARTIFACT_ROOT" "$PIP_CACHE_DIR" "$BUILD_TMP"
export TMPDIR="$BUILD_TMP"
export PYTHONNOUSERSITE=1
export PIP_CONSTRAINT="$REPO_ROOT/constraints/loongreflect-native-cu129.txt"
```

Keep temporary compiler files and download caches on a fast local filesystem.
Write training logs, model checkpoints, EMA state, and acceptance receipts under
`ARTIFACT_ROOT`. Do not commit these local directories or machine-specific paths.
Run installation with `CUDA_VISIBLE_DEVICES=''`; allocate GPUs explicitly for
subsequent training. Choose an installed CUDA toolkit and architecture flags
appropriate for the GPU on which the compiled extension will run.

The constraints protect the Torch/CUDA ABI and NumPy 1.x. In particular,
Transformer Engine's transitive ONNX dependencies otherwise can upgrade NumPy to
2.x. The compatible ONNX family is pinned as well. Use the constraints for every
later dependency-resolving installation, including native builds.

Ray's placement-group serialization imports PyArrow even when the training job
does not use Ray Data. An unconstrained resolution selected PyArrow 26.0.0,
whose native module requires NumPy 2 and failed with the NumPy 1.26.4 runtime.
`pyarrow==19.0.1` passed actual NumPy/Arrow conversion, Ray
`SerializationContext.serialize`, Arrow roundtrip, and SGLang server-argument
imports in this environment. Keep this pin under the same constraints. These
CPU checks do not establish production GPU training acceptance.

The default SGLang rollout imports the reward hub before dispatching the
custom LoongReflect reward. Its math utility imports `pylatexenc`, which is
already declared in `slime/requirements.txt`; include `pylatexenc==2.10` in the
manual native setup below. Verify the actual rollout and reward-hub modules,
as well as the actor. A successful actor import alone does not cover this path.

```bash
CUDA_VISIBLE_DEVICES='' python -m pip install \
  torch==2.11.0+cu129 torchvision==0.26.0+cu129 torchaudio==2.11.0+cu129 \
  --index-url https://download.pytorch.org/whl/cu129 \
  --extra-index-url https://pypi.org/simple
CUDA_VISIBLE_DEVICES='' python -m pip install \
  numpy==1.26.4 scipy==1.17.1 \
  onnx==1.19.1 onnxscript==0.5.6 onnx-ir==0.1.12 ml-dtypes==0.5.3
```

Build `transformer_engine[pytorch]==2.16.1` against this environment with
`--no-build-isolation`. Set `CUDA_HOME` to the chosen toolkit. Set
`NVTE_CUDA_ARCHS` for the target GPU, for example `80` for an Ampere SM80 GPU,
and bound `MAX_JOBS` to available CPU/memory. Install the build prerequisites
required by Transformer Engine before its PyTorch extension; do not resolve a
second Torch installation inside a separate build environment.

```bash
CUDA_VISIBLE_DEVICES='' python -m pip install setuptools wheel ninja pybind11
CUDA_VISIBLE_DEVICES='' NVTE_FRAMEWORK=pytorch \
  python -m pip install --no-build-isolation 'transformer_engine[pytorch]==2.16.1'
```

## Native wheels and the required TMS build variable

Install the CUDA 12.9 kernels and original SGLang wheel **without resolving its
original dependencies yet**:

```bash
CUDA_VISIBLE_DEVICES='' python -m pip install --no-deps \
  sglang-kernel==0.4.4 sgl-deep-gemm==0.1.4 \
  --index-url https://docs.sglang.ai/whl/cu129/
CUDA_VISIBLE_DEVICES='' python -m pip download --no-deps \
  sglang==0.5.15.post1 --dest "$RUNTIME_DIR/wheels"
CUDA_VISIBLE_DEVICES='' python -m pip install --no-deps \
  "$RUNTIME_DIR"/wheels/sglang-0.5.15.post1-*.whl
CUDA_VISIBLE_DEVICES='' python -m pip install --no-deps \
  'https://github.com/lesj0610/flash-attention/releases/download/v2.8.3-cu12-torch2.11/flash_attn-2.8.3%2Bcu12torch2.11cxx11abiTRUE-cp312-cp312-linux_x86_64.whl'
CUDA_VISIBLE_DEVICES='' TMS_CUDA_MAJOR=12 python -m pip install \
  --no-build-isolation \
  'git+https://github.com/zhuzilin/torch_memory_saver.git@8d30c59ca12a68d9deccbc9c6599076a1218cbc5'
CUDA_VISIBLE_DEVICES='' python -m pip install --no-deps \
  'https://github.com/zhuzilin/sgl-router/releases/download/v0.3.2-9daabcd/sglang_router-0.3.2-cp38-abi3-manylinux_2_28_x86_64.whl'
```

The pinned TMS source explicitly requires `TMS_CUDA_MAJOR=12` at metadata/build
time. Omitting it produces a real `metadata-generation-failed` error before
compilation. Its version is `0.0.9.post1+slime`; the CUDA-major suffix selects the
matching compiled library. The router is the custom Slime build, whose runtime
version string contains `slime`. A different PyPI router with the same public
version is insufficient.

## Apply the checked-in patches to real dependency source

Use a private clone of the exact Megatron revision; the complete source root
must remain on `PYTHONPATH` because package installation alone may omit
`megatron.training`:

```bash
git clone https://github.com/NVIDIA/Megatron-LM.git "$RUNTIME_DIR/Megatron-LM"
git -C "$RUNTIME_DIR/Megatron-LM" checkout 1dcf0dafa884ad52ffb243625717a3471643e087
export PATCH_DIR="$REPO_ROOT/slime/docker/patch/v0.5.15.post1"
git -C "$RUNTIME_DIR/Megatron-LM" apply --check "$PATCH_DIR/megatron.patch"
git -C "$RUNTIME_DIR/Megatron-LM" apply "$PATCH_DIR/megatron.patch"
```

For SGLang, stage a copy of the fresh environment's installed package in a
source-like `python/sglang` layout. Apply these four runtime patches **in order**
from the directory containing `python/`:

1. `sglang.patch`
2. `sglang-top_p.patch`
3. `sglang-release_hicache.patch`
4. `sglang-pull_weights.patch`

Run `git apply --check` immediately before each `git apply`. The four patches
have genuinely applied to the released 0.5.15.post1 wheel:

```bash
export SITE_PACKAGES="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
mkdir -p "$RUNTIME_DIR/sglang-patch/python"
cp -a "$SITE_PACKAGES/sglang" "$RUNTIME_DIR/sglang-patch/python/"
for patch in sglang.patch sglang-top_p.patch sglang-release_hicache.patch sglang-pull_weights.patch; do
  git -C "$RUNTIME_DIR/sglang-patch" apply --check "$PATCH_DIR/$patch"
  git -C "$RUNTIME_DIR/sglang-patch" apply "$PATCH_DIR/$patch"
done
```

Publish only after all four checks pass. Keep the original source clone and
other Python environments untouched. Do not apply these patches twice.

The fifth patch, `sglang-openai.patch`, changes only upstream
`python/pyproject.toml`. A wheel contains no such file. Applying its source
hunks to a wheel fails; instead perform exactly these equivalent
`Requires-Dist` replacements in the fresh wheel's `METADATA`:

| Original declaration | CUDA 12 declaration |
| --- | --- |
| `cuda-python>=13.0` | `cuda-python>=12.9,<13` |
| `flash-attn-4==4.0.0b15` | `flash-attn==2.8.3` |
| `flashinfer_python[cu13]==0.6.12` | `flashinfer_python[cu12]==0.6.12` |
| `nvidia-cutlass-dsl[cu13]==4.5.2` | `nvidia-cutlass-dsl==4.5.2` |
| `openai==2.6.1` | `openai>=2.26,<3` |

Require exactly one occurrence of every original declaration before changing
anything. Retain all other base requirements and their platform markers.
Generate a requirements file from the rewritten `Requires-Dist` entries,
excluding entries guarded by `extra == ...`, then install it under the same
constraints. The following publishes the validated package into the fresh
virtual environment and records the packaging equivalence. It refuses a
second ambiguous publication or inherited global site-packages:

```bash
CUDA_VISIBLE_DEVICES='' python - <<'PY'
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import shutil
import sys
import sysconfig

assert sys.prefix != sys.base_prefix
site = Path(sysconfig.get_paths()['purelib']).resolve()
assert site.is_relative_to(Path(sys.prefix).resolve())
assert metadata.version('sglang') == '0.5.15.post1'
runtime = Path(os.environ['RUNTIME_DIR']).resolve()
staged = runtime / 'sglang-patch/python/sglang'
installed = site / 'sglang'
backup = runtime / 'unpatched-sglang'
assert staged.is_dir() and installed.is_dir() and not backup.exists()
metadata_file = site / 'sglang-0.5.15.post1.dist-info/METADATA'
original = metadata_file.read_text()
rewrites = {
    'cuda-python>=13.0': 'cuda-python>=12.9,<13',
    'flash-attn-4==4.0.0b15': 'flash-attn==2.8.3',
    'flashinfer_python[cu13]==0.6.12': 'flashinfer_python[cu12]==0.6.12',
    'nvidia-cutlass-dsl[cu13]==4.5.2': 'nvidia-cutlass-dsl==4.5.2',
    'openai==2.6.1': 'openai>=2.26,<3',
}
patched = original
for old, new in rewrites.items():
    declaration = 'Requires-Dist: ' + old + '\n'
    assert patched.count(declaration) == 1, old
    patched = patched.replace(declaration, 'Requires-Dist: ' + new + '\n')
requirements = [line.removeprefix('Requires-Dist: ') for line in patched.splitlines()
                if line.startswith('Requires-Dist: ') and 'extra ==' not in line]
shutil.copytree(installed, backup)
(runtime / 'unpatched-sglang-METADATA').write_text(original)
shutil.rmtree(installed)
shutil.copytree(staged, installed)
metadata_file.write_text(patched)
(runtime / 'sglang-cu12-base.requirements').write_text('\n'.join(requirements) + '\n')
(runtime / 'packaging-equivalence.json').write_text(json.dumps({
    'source_patch': 'sglang-openai.patch',
    'source_file': 'python/pyproject.toml',
    'wheel_handling': 'Exact equivalent Requires-Dist replacements; wheel has no pyproject.toml',
    'replacements': rewrites,
}, indent=2) + '\n')
PY
```

Record this as a packaging-equivalent wheel metadata change, not an assertion
that absent `pyproject.toml` hunks were applied.

The actual wheel pins **Transformers 5.12.1**, Torch 2.11.0, TorchAO 0.17.0 and
TorchCodec 0.11.1. Historical Transformers 5.6.0 pins from other source revisions
do not describe this wheel. Do not install the unmodified wheel normally and
let it choose CUDA 13 or replace the chosen Torch ABI.

Install the remaining runtime utilities under the constraints:

```bash
CUDA_VISIBLE_DEVICES='' python -m pip install \
  -r "$RUNTIME_DIR/sglang-cu12-base.requirements" \
  'ray[default]==2.59.0' pyarrow==19.0.1 pylatexenc==2.10 \
  wandb tensorboard PyYAML httpx \
  blake3 xxhash zstandard omegaconf accelerate
export PYTHONPATH="$REPO_ROOT/slime:$RUNTIME_DIR/Megatron-LM${PYTHONPATH:+:$PYTHONPATH}"
```

## Verify imports before reserving GPUs

With GPU visibility empty, verify exact installed versions and actual imports:

```bash
CUDA_VISIBLE_DEVICES='' python - <<'PY'
import importlib
import importlib.metadata as metadata
import sys
import torch
assert sys.prefix != sys.base_prefix
for name, expected in {
    'torch': '2.11.0+cu129', 'numpy': '1.26.4', 'scipy': '1.17.1',
    'transformer-engine': '2.16.1', 'sglang': '0.5.15.post1',
    'pyarrow': '19.0.1', 'pylatexenc': '2.10',
}.items():
    assert metadata.version(name) == expected, (name, metadata.version(name))
assert torch.version.cuda == '12.9'
for name in [
    'transformer_engine.pytorch', 'torch_memory_saver',
    'sglang.srt.server_args', 'sglang_router.launch_router',
    'megatron.core.optimizer', 'megatron.training.training',
    'slime.backends.megatron_utils.actor', 'slime_plugins.loongreflect.runtime',
    'slime.rollout.sglang_rollout', 'slime.rollout.rm_hub', 'slime.ray.rollout',
    'slime.backends.sglang_utils.sglang_engine',
    'slime_plugins.loongreflect.slime_adapter', 'slime_plugins.loongreflect.self_distill',
]:
    module = importlib.import_module(name)
    print(name, module.__file__)
import sglang_router
assert 'slime' in sglang_router.__version__
import numpy as np
import pyarrow as pa
import ray
from ray._private.serialization import SerializationContext
from ray._private.worker import global_worker
array = np.array([1, 2, 3], dtype=np.int64)
table = pa.table({'values': pa.array(array)})
serialized = SerializationContext(global_worker).serialize({'table': table, 'array': array})
assert serialized.total_bytes > 0
assert ray.cloudpickle.loads(ray.cloudpickle.dumps(table)).equals(table)
print('Native imports passed; production GPU training remains a separate check.')
PY
```

The Qwen2 native HF loader can initialize the actor/reference directly from a
real pretrained HF directory when it has no Megatron checkpoint tracker. For a
small execution check, `Qwen/Qwen2.5-0.5B-Instruct` with revision
`7ae557604adf67be50417f59c2c2f167def9a775` and the checked-in
`slime/scripts/models/qwen2.5-0.5B.sh` preset avoid an unnecessary conversion.
Use the same existing HF directory for `--hf-checkpoint`,
`--megatron-checkpoint`, and `--reference-checkpoint` in a fresh run. An actual
Megatron resume directory must include the matching EMA checkpoint.

## Production acceptance

Reserve two eligible GPUs before running one training GPU and one SGLang rollout
GPU. Use an independent Ray instance, private available ports, a unique run ID,
and a new persistent output directory. Put the short Ray socket directory on a
real local filesystem with Unix socket support. A local symlink into a FUSE
mount still uses the target filesystem and can fail with `bind: Function not
implemented`; it does not solve IPC compatibility. Keep model/checkpoint and
training logs in persistent storage, and copy Ray diagnostic logs there before
cleaning up the local runtime directory. Clean up only that run's recorded
descendants or unique run ID, checking PID
creation times. Do not use global Ray/process shutdown commands.

For the production code to launch its own SGLang router, leave
`sglang_router_ip` unset (`null` in YAML) and select an available
`sglang_router_port`. A non-null router IP means an external router already
exists: `_start_router` returns that address without launching a service.
Setting a loopback IP alone therefore does not start a router. Retain the
custom Slime router and verify that its real service starts before rollout.

Use actual training-split inputs and a real searchable corpus. The pretrained
model must sample the search and reflection actions itself. Do not inject
reflection tokens or replace the model, retrieval, teacher, or optimizer with
stubs. The private reference answer belongs only in the teacher/validator and
in legitimate searchable evidence, never in the student's task prompt.

A complete Joint execution check requires actual process exit 0, ranked
retrieval evidence and search tree nodes, nonzero sampled reflection masks,
nonzero dense tree credit, finite nonzero gradients, a successful optimizer
update, effective reflection distillation weight, finite EMA-teacher scores,
saved full optimizer/scheduler/EMA state, scheduler advancement, changed real
model parameters, and the saved EMA matching production dtype arithmetic.
Fast-Slow additionally requires three actual distillation-only optimizer
updates followed by one policy update on the same rollout. One-step BF16 EMA
rounding may leave some tensors numerically unchanged; verify the exact
`dst.mul_(tau).add_(src, alpha=1-tau)` arithmetic and report that honestly.

Bind a future completion claim to the actual command, terminal status,
acceptance receipt and checkpoint artifacts. No production GPU acceptance or
held-out benchmark score is asserted by this installation document.
