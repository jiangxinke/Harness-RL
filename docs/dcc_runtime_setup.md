# DCC runtime setup

Create separate environments for BF16/grouped MoE and full-expert NF4. Install
the complete requirements file, including its nested base requirements:

```bash
# In the BF16 environment, after selecting a CUDA-compatible PyTorch build:
python -m pip install -r configs/dcc/requirements-training.txt
python -m pip check
python scripts/dcc_preflight.py --runtime bf16

# In a different environment for NF4:
python -m pip install -r configs/dcc/requirements-quantized.txt
python -m pip check
python scripts/dcc_preflight.py --runtime nf4
```

Both files include `requirements.txt` using pip's relative `-r` directive.
Installing only Transformers, PEFT and bitsandbytes leaves out benchmark and
web dependencies. Pandas is needed when the scoring module is imported;
dateparser is needed by date-aware WideSearch columns; pypdf and trafilatura
support PDF and HTML evidence extraction. The supported pandas range is below
version 3. Do not change dependency versions in an environment with running
jobs; prepare a separate environment or coordinate with the run owner.

Preflight always hides GPUs in its own process. It checks the selected declared
dependency versions, imports the runtime, and runs tiny in-memory HTML/PDF and
CUAD/WideSearch date-scoring examples. It performs no network requests or model
downloads and writes no artifacts. `--runtime harness` checks only the base
harness dependencies. `--include-tests` also checks pytest and pytest-asyncio.
Version checking uses `packaging`, which the model runtime already requires;
install it explicitly when checking a minimal harness-only environment.

For a deeper CPU check of the quantized runtime, use:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=slime python -m pytest -q \
  tests/test_dcc_core.py tests/test_dcc_search_provider.py \
  tests/test_dcc_widesearch_metrics.py tests/test_dcc_evaluation.py \
  tests/test_dcc_evidence_guard.py tests/test_dcc_quantization.py
```

The quantization tests load a tiny synthetic Qwen3 MoE model, verify expert
coverage and adapter gradients, and round-trip adapter weights. A preflight
pass does not establish real-model CUDA compatibility, memory capacity,
search-provider availability, or benchmark quality. Follow it with the
explicit GPU smoke and actual evaluation in [dcc_training.md](dcc_training.md),
using a configured external artifact root for all logs and checkpoints.
