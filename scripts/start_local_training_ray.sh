#!/usr/bin/env bash
set -euo pipefail

# Start one independent single-node Ray head for a formal training arm. Resolve
# the address on every invocation so stale pre-restart IPs are never reused.
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
SP_ENV_PREFIX=${SP_ENV_PREFIX:-${HOME}/.cache/harness-rl/training-env}
SP_RAY_NODE_IP=${SP_RAY_NODE_IP:-}
SP_RAY_NUM_GPUS=${SP_RAY_NUM_GPUS:-}
SP_RAY_PORT=${SP_RAY_PORT:-6379}
SP_RAY_STATE_ROOT=${SP_RAY_STATE_ROOT:-${HOME}/.cache/harness-rl/ray}

source "${repo_root}/scripts/training_env.sh"
if ray status >/dev/null 2>&1; then
  ray status
  echo "local Ray runtime is already available"
  exit 0
fi
if [[ -z ${SP_RAY_NODE_IP} ]]; then
  SP_RAY_NODE_IP=$(python - <<'PY'
import socket

with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
    sock.connect(("8.8.8.8", 80))
    print(sock.getsockname()[0])
PY
)
fi
if [[ -z ${SP_RAY_NUM_GPUS} ]]; then
  SP_RAY_NUM_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
fi
if [[ ! ${SP_RAY_NODE_IP} =~ ^[0-9]+(\.[0-9]+){3}$ ]]; then
  echo "invalid SP_RAY_NODE_IP: ${SP_RAY_NODE_IP}" >&2
  exit 2
fi
if [[ ! ${SP_RAY_NUM_GPUS} =~ ^[1-9][0-9]*$ ]]; then
  echo "invalid SP_RAY_NUM_GPUS: ${SP_RAY_NUM_GPUS}" >&2
  exit 2
fi

mkdir -p "${SP_RAY_STATE_ROOT}"
ray start --head \
  --node-ip-address "${SP_RAY_NODE_IP}" \
  --port "${SP_RAY_PORT}" \
  --num-gpus "${SP_RAY_NUM_GPUS}" \
  --disable-usage-stats \
  --dashboard-host 127.0.0.1 \
  >"${SP_RAY_STATE_ROOT}/start.log"
ray status
{
  printf 'started_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'node=%s\n' "$(hostname)"
  printf 'node_ip=%s\n' "${SP_RAY_NODE_IP}"
  printf 'address=%s:%s\n' "${SP_RAY_NODE_IP}" "${SP_RAY_PORT}"
  printf 'num_gpus=%s\n' "${SP_RAY_NUM_GPUS}"
} >"${SP_RAY_STATE_ROOT}/status.env"
printf 'started_local_ray=%s:%s\n' "${SP_RAY_NODE_IP}" "${SP_RAY_PORT}"
