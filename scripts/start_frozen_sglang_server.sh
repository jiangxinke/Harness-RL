#!/usr/bin/env bash
set -euo pipefail

# Start the immutable specialist endpoint used by central-only training. It is
# outside the trainable actor lifecycle, so slime weight updates cannot reach it.
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
SP_ENV_PREFIX=${SP_ENV_PREFIX:-${HOME}/.cache/harness-rl/training-env}
SP_FROZEN_MODEL_PATH=${SP_FROZEN_MODEL_PATH:-}
SP_FROZEN_MODEL_NAME=${SP_FROZEN_MODEL_NAME:-Qwen2.5-3B-Instruct}
SP_FROZEN_CUDA_DEVICES=${SP_FROZEN_CUDA_DEVICES:-0}
SP_FROZEN_DP_SIZE=${SP_FROZEN_DP_SIZE:-1}
SP_FROZEN_HOST=${SP_FROZEN_HOST:-0.0.0.0}
SP_FROZEN_PORT=${SP_FROZEN_PORT:-18002}
SP_FROZEN_MEM_FRACTION=${SP_FROZEN_MEM_FRACTION:-0.70}
SP_FROZEN_CONTEXT_LENGTH=${SP_FROZEN_CONTEXT_LENGTH:-16384}
SP_SERVICE_STATE_ROOT=${SP_SERVICE_STATE_ROOT:-${HOME}/.cache/harness-rl/services/frozen}
log_path=${SP_SERVICE_STATE_ROOT}/frozen-sglang.log
pid_path=${SP_SERVICE_STATE_ROOT}/frozen-sglang.pid

if [[ -z ${SP_FROZEN_MODEL_PATH} ]]; then
  echo "SP_FROZEN_MODEL_PATH is required" >&2
  exit 2
fi
if [[ ! ${SP_FROZEN_CUDA_DEVICES} =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "invalid SP_FROZEN_CUDA_DEVICES: ${SP_FROZEN_CUDA_DEVICES}" >&2
  exit 2
fi
IFS=, read -r -a frozen_devices <<<"${SP_FROZEN_CUDA_DEVICES}"
if (( ${#frozen_devices[@]} != SP_FROZEN_DP_SIZE )); then
  echo "SP_FROZEN_DP_SIZE must equal the visible GPU count" >&2
  exit 2
fi
if [[ ! ${SP_FROZEN_PORT} =~ ^[0-9]+$ ]] || (( SP_FROZEN_PORT < 1 || SP_FROZEN_PORT > 65535 )); then
  echo "invalid SP_FROZEN_PORT: ${SP_FROZEN_PORT}" >&2
  exit 2
fi
for required in "${SP_ENV_PREFIX}/bin/python" "${SP_FROZEN_MODEL_PATH}/config.json"; do
  if [[ ! -e ${required} ]]; then
    echo "missing frozen endpoint asset: ${required}" >&2
    exit 2
  fi
done

mkdir -p "${SP_SERVICE_STATE_ROOT}"
if [[ -s ${pid_path} ]]; then
  existing_pid=$(<"${pid_path}")
  if [[ ${existing_pid} =~ ^[0-9]+$ ]] && kill -0 "${existing_pid}" 2>/dev/null; then
    echo "frozen SGLang endpoint already running as PID ${existing_pid}"
    exit 0
  fi
fi
if "${SP_ENV_PREFIX}/bin/python" - "${SP_FROZEN_HOST}" "${SP_FROZEN_PORT}" <<'PY'
import socket
import sys

host = sys.argv[1]
if host == "0.0.0.0":
    host = "127.0.0.1"
with socket.socket() as sock:
    sys.exit(0 if sock.connect_ex((host, int(sys.argv[2]))) == 0 else 1)
PY
then
  echo "frozen endpoint port is already occupied: ${SP_FROZEN_HOST}:${SP_FROZEN_PORT}" >&2
  exit 2
fi

source "${repo_root}/scripts/training_env.sh"
export CUDA_VISIBLE_DEVICES=${SP_FROZEN_CUDA_DEVICES}
nohup python -m sglang.launch_server \
  --model-path "${SP_FROZEN_MODEL_PATH}" \
  --served-model-name "${SP_FROZEN_MODEL_NAME}" \
  --host "${SP_FROZEN_HOST}" --port "${SP_FROZEN_PORT}" \
  --tp-size 1 --dp-size "${SP_FROZEN_DP_SIZE}" \
  --context-length "${SP_FROZEN_CONTEXT_LENGTH}" \
  --mem-fraction-static "${SP_FROZEN_MEM_FRACTION}" \
  --tool-call-parser qwen --trust-remote-code \
  >"${log_path}" 2>&1 </dev/null &
server_pid=$!
printf '%s\n' "${server_pid}" >"${pid_path}"
printf 'started_frozen_sglang_pid=%s log=%s\n' "${server_pid}" "${log_path}"
