#!/usr/bin/env bash

# Source this file. It prints no secrets and derives all CUDA paths from the
# selected prefix so the repository may live anywhere.
SP_ENV_PREFIX=${SP_ENV_PREFIX:-${HOME}/.cache/harness-rl/training-env}
if [[ ! -x ${SP_ENV_PREFIX}/bin/python ]]; then
  echo "training environment is missing: ${SP_ENV_PREFIX}" >&2
  return 2 2>/dev/null || exit 2
fi
export SP_ENV=${SP_ENV_PREFIX}
export PATH="${SP_ENV}/bin:${PATH}"
export CUDA_HOME=${SP_ENV}
python_site="${SP_ENV}/lib/python3.12/site-packages"
cuda_libs="${python_site}/torch/lib:${python_site}/nvidia/cudnn/lib:${python_site}/nvidia/cublas/lib:${SP_ENV}/lib:${SP_ENV}/targets/x86_64-linux/lib"
export LD_LIBRARY_PATH="${cuda_libs}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export LIBRARY_PATH="${SP_ENV}/lib:${SP_ENV}/targets/x86_64-linux/lib:${SP_ENV}/targets/x86_64-linux/lib/stubs${LIBRARY_PATH:+:${LIBRARY_PATH}}"
export CPATH="${SP_ENV}/targets/x86_64-linux/include${CPATH:+:${CPATH}}"
export CPLUS_INCLUDE_PATH="${SP_ENV}/targets/x86_64-linux/include${CPLUS_INCLUDE_PATH:+:${CPLUS_INCLUDE_PATH}}"
# Megatron requires this value whenever tensor or context parallelism exceeds
# one.  Keep it in the shared launcher environment so 7B presets behave the
# same from every documented entrypoint and on a freshly restored node.
export CUDA_DEVICE_MAX_CONNECTIONS=1
unset PYTHONPATH
