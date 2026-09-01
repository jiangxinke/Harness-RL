#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
SP_SEARCH_R1_ROOT=${SP_SEARCH_R1_ROOT:-${HOME}/.cache/harness-rl/search-r1}
SP_SEARCH_R1_STATE_ROOT=${SP_SEARCH_R1_STATE_ROOT:-${SP_SEARCH_R1_ROOT}/state}
SP_SEARCH_R1_SOURCE=${SP_SEARCH_R1_SOURCE:-${repo_root}/slime/examples/search-r1/local_dense_retriever/retrieval_server.py}
SP_SEARCH_R1_CUDA_DEVICES=${SP_SEARCH_R1_CUDA_DEVICES:-0}
SP_SEARCH_R1_TOPK=${SP_SEARCH_R1_TOPK:-5}

python_bin="${SP_SEARCH_R1_ROOT}/venv/bin/python"
index_path="${SP_SEARCH_R1_ROOT}/index/e5_Flat.index"
corpus_path="${SP_SEARCH_R1_ROOT}/index/wiki-18.jsonl"
model_path="${SP_SEARCH_R1_ROOT}/model/intfloat-e5-base-v2"
log_path="${SP_SEARCH_R1_STATE_ROOT}/retriever.log"
pid_path="${SP_SEARCH_R1_STATE_ROOT}/retriever.pid"

for required in "${python_bin}" "${index_path}" "${corpus_path}" "${model_path}/config.json" "${SP_SEARCH_R1_SOURCE}"; do
  if [[ ! -e ${required} ]]; then
    echo "missing Search-R1 runtime asset: ${required}" >&2
    exit 2
  fi
done

mkdir -p "${SP_SEARCH_R1_STATE_ROOT}"
if [[ -s ${pid_path} ]]; then
  existing_pid=$(<"${pid_path}")
  if [[ ${existing_pid} =~ ^[0-9]+$ ]] && kill -0 "${existing_pid}" 2>/dev/null; then
    echo "Search-R1 retriever already running as PID ${existing_pid}"
    exit 0
  fi
fi

export CUDA_VISIBLE_DEVICES=${SP_SEARCH_R1_CUDA_DEVICES}
nohup "${python_bin}" "${SP_SEARCH_R1_SOURCE}" \
  --index_path "${index_path}" \
  --corpus_path "${corpus_path}" \
  --topk "${SP_SEARCH_R1_TOPK}" \
  --retriever_name e5 \
  --retriever_model "${model_path}" \
  --faiss_gpu >"${log_path}" 2>&1 < /dev/null &
retriever_pid=$!
echo "${retriever_pid}" >"${pid_path}"
echo "started Search-R1 retriever as PID ${retriever_pid}; log=${log_path}"
