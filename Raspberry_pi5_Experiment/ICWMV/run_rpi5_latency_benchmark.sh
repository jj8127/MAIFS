#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-}"
IMAGE="${2:-}"
shift $(( $# >= 2 ? 2 : $# ))

usage() {
  cat <<'EOF'
사용법:
  bash run_rpi5_latency_benchmark.sh <cpu|coral|pcie-hat|all> /path/to/image.jpg [추가옵션...]

예시:
  bash run_rpi5_latency_benchmark.sh cpu /home/pi/test.jpg
  bash run_rpi5_latency_benchmark.sh coral /home/pi/test.jpg
  bash run_rpi5_latency_benchmark.sh pcie-hat /home/pi/test.jpg
  bash run_rpi5_latency_benchmark.sh all /home/pi/test.jpg
EOF
}

if [[ -z "${MODE}" || -z "${IMAGE}" || "${MODE}" == "-h" || "${MODE}" == "--help" ]]; then
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
