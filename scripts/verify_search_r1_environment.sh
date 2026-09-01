#!/usr/bin/env bash
set -euo pipefail

SP_SEARCH_R1_ROOT=${SP_SEARCH_R1_ROOT:-${HOME}/.cache/harness-rl/search-r1}
SP_SEARCH_R1_ENV=${SP_SEARCH_R1_ENV:-${SP_SEARCH_R1_ROOT}/venv}
python_bin="${SP_SEARCH_R1_ENV}/bin/python"
if [[ ! -x ${python_bin} ]]; then
  echo "Search-R1 environment Python is missing: ${python_bin}" >&2
  exit 2
fi
"${python_bin}" - <<'PY'
import datasets
import faiss
import fastapi
import numpy
import pydantic
import torch
import transformers
import uvicorn

assert torch.__version__ == "2.8.0+cu126", torch.__version__
assert torch.version.cuda == "12.6", torch.version.cuda
assert torch.cuda.is_available()
assert hasattr(faiss, "StandardGpuResources")
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
print(faiss.__version__, datasets.__version__, transformers.__version__)
print(fastapi.__version__, pydantic.__version__, numpy.__version__, uvicorn.__version__)
PY
