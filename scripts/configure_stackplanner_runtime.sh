#!/usr/bin/env bash
set -euo pipefail

# Materialize the committed, key-free service configuration for StackPlanner.
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
SP_STACKPLANNER_CONFIG_SOURCE=${SP_STACKPLANNER_CONFIG_SOURCE:-${repo_root}/configs/deerflow_search_r1_only.yaml}
SP_STACKPLANNER_CONFIG_TARGET=${SP_STACKPLANNER_CONFIG_TARGET:-${repo_root}/StackPlanner2.0/config.yaml}
SP_SERVICE_HOST=${SP_SERVICE_HOST:-127.0.0.1}
SP_OVERWRITE_RUNTIME_CONFIG=${SP_OVERWRITE_RUNTIME_CONFIG:-0}

if [[ ! ${SP_SERVICE_HOST} =~ ^[A-Za-z0-9][A-Za-z0-9.-]*$ ]]; then
  echo "invalid SP_SERVICE_HOST: ${SP_SERVICE_HOST}" >&2
  exit 2
fi
if [[ ! -f ${SP_STACKPLANNER_CONFIG_SOURCE} ]]; then
  echo "runtime config template is missing: ${SP_STACKPLANNER_CONFIG_SOURCE}" >&2
  exit 2
fi
if [[ -e ${SP_STACKPLANNER_CONFIG_TARGET} && ${SP_OVERWRITE_RUNTIME_CONFIG} != 1 ]]; then
  echo "runtime config already exists; refusing to overwrite: ${SP_STACKPLANNER_CONFIG_TARGET}" >&2
  exit 2
fi

mkdir -p "$(dirname "${SP_STACKPLANNER_CONFIG_TARGET}")"
tmp_config="${SP_STACKPLANNER_CONFIG_TARGET}.tmp.$$"
sed "s#http://127.0.0.1:#http://${SP_SERVICE_HOST}:#g" \
  "${SP_STACKPLANNER_CONFIG_SOURCE}" >"${tmp_config}"
chmod 600 "${tmp_config}"
mv "${tmp_config}" "${SP_STACKPLANNER_CONFIG_TARGET}"
printf 'configured_stackplanner_runtime=%s\n' "${SP_STACKPLANNER_CONFIG_TARGET}"
