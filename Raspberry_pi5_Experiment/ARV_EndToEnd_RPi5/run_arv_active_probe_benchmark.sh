#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-}"

if [[ $# -ge 1 ]]; then
  shift
fi

PROBE_DIR="${ROOT_DIR}/assets/active_probe_set"

usage() {
  cat <<'EOF'
사용법:
  bash run_arv_active_probe_benchmark.sh <cpu|coral|pcie-hat|all> [추가옵션...]

설명:
  번들 안에 포함된 active_probe_set 이미지만 사용해
  1) 실제 keep/revert 후보 탐색
  2) 실제 ARV-active 종단간 반복 측정
  을 한 번에 수행한다.

예시:
  bash run_arv_active_probe_benchmark.sh cpu
  bash run_arv_active_probe_benchmark.sh all
  bash run_arv_active_probe_benchmark.sh cpu --protocol extended
EOF
}

if [[ -z "${MODE}" || "${MODE}" == "-h" || "${MODE}" == "--help" ]]; then
  usage
  exit 1
fi

if [[ ! -d "${PROBE_DIR}" ]]; then
  echo "probe 이미지 디렉토리를 찾을 수 없습니다: ${PROBE_DIR}" >&2
  exit 1
fi

exec bash "${ROOT_DIR}/run_arv_active_workflow.sh" all "${MODE}" "${PROBE_DIR}" "$@"
