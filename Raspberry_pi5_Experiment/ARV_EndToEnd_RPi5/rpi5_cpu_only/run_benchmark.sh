#!/usr/bin/env bash
set -euo pipefail

ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_ROOT="$(cd "${ENV_DIR}/.." && pwd)"
COMMON_BENCH="${BUNDLE_ROOT}/common/scripts/benchmark_arv_e2e_latency.py"
IMAGE="${1:-${BUNDLE_ROOT}/assets/benchmark_input.png}"
if [[ $# -ge 1 ]]; then
  shift
fi

mkdir -p "${ENV_DIR}/results" "${ENV_DIR}/logs"

STAMP="$(date +%Y%m%d)"
CPU_PYTHON="${CPU_PYTHON:-${ENV_DIR}/.venv/bin/python}"
if [[ ! -x "${CPU_PYTHON}" ]]; then
  CPU_PYTHON="$(command -v python3)"
fi

exec "${CPU_PYTHON}" "${COMMON_BENCH}" \
  --mode cpu \
  --image "${IMAGE}" \
  --output-json "${ENV_DIR}/results/${STAMP}_rpi5_cpu_arv_e2e_latency.json" \
  --output-log "${ENV_DIR}/logs/${STAMP}_rpi5_cpu_arv_e2e_latency.log" \
  "$@"
