#!/usr/bin/env bash
set -euo pipefail

SP_ENV_PREFIX=${SP_ENV_PREFIX:-${HOME}/.cache/harness-rl/training-env}
SP_SOURCE_ROOT=${SP_SOURCE_ROOT:-${HOME}/.cache/harness-rl/sources}
SP_BUILD_CACHE=${SP_BUILD_CACHE:-${HOME}/.cache/harness-rl/build}
minimum_free_gib=${SP_MINIMUM_FREE_GIB:-120}
failures=0

pass() { printf 'PASS  %s\n' "$*"; }
warn() { printf 'WARN  %s\n' "$*"; }
fail() { printf 'FAIL  %s\n' "$*" >&2; failures=$((failures + 1)); }

if [[ $(uname -s) == Linux && $(uname -m) == x86_64 ]]; then
  pass "platform Linux x86_64"
else
  fail "validated conda build requires Linux x86_64; got $(uname -s) $(uname -m)"
fi
for command_name in bash curl git make patch sed sha256sum tar zstd; do
  if command -v "${command_name}" >/dev/null; then
    pass "command ${command_name}"
  else
    fail "missing command ${command_name}"
  fi
done
if command -v protoc >/dev/null; then
  pass "host protoc available (the isolated environment installs its own locked build tool)"
else
  warn "host protoc absent; setup will install it inside the isolated environment"
fi
if command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1; then
  pass "NVIDIA driver and $(nvidia-smi -L | wc -l) visible GPU(s)"
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
else
  fail "nvidia-smi cannot enumerate a GPU"
fi

mkdir -p "$(dirname "${SP_ENV_PREFIX}")" "${SP_SOURCE_ROOT}" "${SP_BUILD_CACHE}"
for target in "$(dirname "${SP_ENV_PREFIX}")" "${SP_SOURCE_ROOT}" "${SP_BUILD_CACHE}"; do
  if [[ -w ${target} ]]; then
    pass "writable ${target}"
  else
    fail "not writable: ${target}"
  fi
done
available_kib=$(df -Pk "${SP_SOURCE_ROOT}" | awk 'NR==2 {print $4}')
required_kib=$((minimum_free_gib * 1024 * 1024))
if (( available_kib >= required_kib )); then
  pass "at least ${minimum_free_gib} GiB free for sources/environment"
else
  fail "less than ${minimum_free_gib} GiB free below ${SP_SOURCE_ROOT}"
fi

if command -v docker >/dev/null; then
  warn "Docker is available; the validated A100 path still uses make setup unless an image digest is deliberately selected"
else
  pass "Docker absent; conda/micromamba path selected"
fi
printf 'INFO  environment=%s\n' "${SP_ENV_PREFIX}"
printf 'INFO  sources=%s\n' "${SP_SOURCE_ROOT}"
printf 'INFO  cache=%s\n' "${SP_BUILD_CACHE}"
if (( failures != 0 )); then
  printf 'Host doctor found %d blocking problem(s).\n' "${failures}" >&2
  exit 1
fi
