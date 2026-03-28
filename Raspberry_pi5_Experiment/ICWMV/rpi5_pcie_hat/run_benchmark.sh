#!/usr/bin/env bash
set -euo pipefail

ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${ENV_DIR}/../../.." && pwd)"
COMMON_BENCH="${ENV_DIR}/../common/scripts/benchmark_rpi5_latency.py"

usage() {
  cat <<'EOF'
사용법:
  bash run_benchmark.sh /path/to/image.jpg [추가옵션...]
EOF
}

IMAGE="${1:-}"
if [[ -z "${IMAGE}" || "${IMAGE}" == "-h" || "${IMAGE}" == "--help" ]]; then
  usage
  exit 1
fi
shift || true

mkdir -p "${ENV_DIR}/results" "${ENV_DIR}/logs"

STAMP="$(date +%Y%m%d)"
HOST_PYTHON="${HOST_PYTHON:-python3}"
CORAL_PYTHON="${CORAL_PYTHON:-${REPO_ROOT}/.venv-coral39/bin/python}"
if [[ ! -x "${CORAL_PYTHON}" ]]; then
  CORAL_PYTHON="$(command -v python3)"
fi

exec "${HOST_PYTHON}" "${COMMON_BENCH}" \
  --mode pcie-hat \
  --image "${IMAGE}" \
  --runner-python "${CORAL_PYTHON}" \
  --output-json "${ENV_DIR}/results/${STAMP}_rpi5_pcie_hat_latency.json" \
  --output-log "${ENV_DIR}/logs/${STAMP}_rpi5_pcie_hat_latency.log" \
  "$@"
