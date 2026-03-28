#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-}"
IMAGE="${2:-}"

if [[ $# -ge 2 ]]; then
  shift 2
elif [[ $# -ge 1 ]]; then
  shift 1
fi

DEFAULT_IMAGE="${ROOT_DIR}/assets/benchmark_input.png"
if [[ -z "${IMAGE}" ]]; then
  IMAGE="${DEFAULT_IMAGE}"
fi

usage() {
  cat <<'EOF'
사용법:
  bash run_arv_e2e_benchmark.sh <cpu|coral|pcie-hat|all> [image_path] [추가옵션...]

예시:
  bash run_arv_e2e_benchmark.sh cpu
  bash run_arv_e2e_benchmark.sh coral /home/pi/test.jpg
  bash run_arv_e2e_benchmark.sh all
  bash run_arv_e2e_benchmark.sh all "" --protocol extended
EOF
}

if [[ -z "${MODE}" || "${MODE}" == "-h" || "${MODE}" == "--help" ]]; then
  usage
  exit 1
fi

run_env() {
  local env_dir="$1"
  bash "${ROOT_DIR}/${env_dir}/run_benchmark.sh" "${IMAGE}" "$@"
}

case "${MODE}" in
  cpu)
    run_env "rpi5_cpu_only" "$@"
    ;;
  coral)
    run_env "rpi5_coral_usb" "$@"
    ;;
  pcie-hat)
    run_env "rpi5_pcie_hat" "$@"
    ;;
  all)
    run_env "rpi5_cpu_only" "$@"
    run_env "rpi5_coral_usb" "$@"
    run_env "rpi5_pcie_hat" "$@"
    ;;
  *)
    echo "알 수 없는 mode: ${MODE}" >&2
    usage
    exit 1
    ;;
esac
