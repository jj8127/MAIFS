#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACTION="${1:-}"
MODE="${2:-}"
TARGET="${3:-}"

if [[ $# -ge 3 ]]; then
  shift 3
elif [[ $# -ge 2 ]]; then
  shift 2
elif [[ $# -ge 1 ]]; then
  shift 1
fi

usage() {
  cat <<'EOF'
사용법:
  bash run_arv_active_workflow.sh discover <cpu|coral|pcie-hat|all> <image_dir> [추가옵션...]
  bash run_arv_active_workflow.sh benchmark <cpu|coral|pcie-hat> <manifest_json> [추가옵션...]
  bash run_arv_active_workflow.sh all <cpu|coral|pcie-hat|all> <image_dir> [추가옵션...]

설명:
  discover  : 실제 keep/revert가 일어나는 후보 이미지를 찾는다.
  benchmark : discovery manifest를 바탕으로 ARV-active 종단간 시간을 반복 측정한다.
  all       : discover 후 후보가 있으면 바로 benchmark까지 이어서 수행한다.

예시:
  bash run_arv_active_workflow.sh discover cpu /home/pi/arv_eval_images --infer-sub-type-from-path
  bash run_arv_active_workflow.sh benchmark cpu /home/pi/arv_manifest.json
  bash run_arv_active_workflow.sh all coral /home/pi/arv_eval_images --infer-sub-type-from-path
EOF
}

if [[ -z "${ACTION}" || -z "${MODE}" || -z "${TARGET}" || "${ACTION}" == "-h" || "${ACTION}" == "--help" ]]; then
  usage
  exit 1
fi

pick_env_dir() {
  case "$1" in
    cpu) echo "rpi5_cpu_only" ;;
    coral) echo "rpi5_coral_usb" ;;
    pcie-hat) echo "rpi5_pcie_hat" ;;
    *)
      echo "알 수 없는 mode: $1" >&2
      exit 1
      ;;
  esac
}

pick_python() {
  local mode="$1"
  local env_dir="$2"
  if [[ "${mode}" == "cpu" ]]; then
    local py="${CPU_PYTHON:-${ROOT_DIR}/${env_dir}/.venv/bin/python}"
    [[ -x "${py}" ]] && echo "${py}" || command -v python3
  else
    local py="${CORAL_PYTHON:-${ROOT_DIR}/${env_dir}/.venv/bin/python}"
    [[ -x "${py}" ]] && echo "${py}" || command -v python3
  fi
}

run_discover_one() {
  local mode="$1"
  local image_dir="$2"
  shift 2
  local env_dir
  env_dir="$(pick_env_dir "${mode}")"
  local py
  py="$(pick_python "${mode}" "${env_dir}")"
  mkdir -p "${ROOT_DIR}/${env_dir}/results" "${ROOT_DIR}/${env_dir}/logs"
  local stamp
  stamp="$(date +%Y%m%d)"
  local suffix
  suffix="$(tr '-' '_' <<< "${mode}")"
  local out_json="${ROOT_DIR}/${env_dir}/results/${stamp}_${suffix}_arv_active_discovery.json"
  local out_log="${ROOT_DIR}/${env_dir}/logs/${stamp}_${suffix}_arv_active_discovery.log"

  "${py}" "${ROOT_DIR}/common/scripts/discover_arv_active_cases.py" \
    --mode "${mode}" \
    --image-dir "${image_dir}" \
    --output-json "${out_json}" \
    --output-log "${out_log}" \
    "$@"
  echo "${out_json}"
}

run_benchmark_one() {
  local mode="$1"
  local manifest="$2"
  shift 2
  local env_dir
  env_dir="$(pick_env_dir "${mode}")"
  local py
  py="$(pick_python "${mode}" "${env_dir}")"
  mkdir -p "${ROOT_DIR}/${env_dir}/results" "${ROOT_DIR}/${env_dir}/logs"
  local stamp
  stamp="$(date +%Y%m%d)"
  local suffix
  suffix="$(tr '-' '_' <<< "${mode}")"
  local out_json="${ROOT_DIR}/${env_dir}/results/${stamp}_${suffix}_arv_active_latency.json"
  local out_log="${ROOT_DIR}/${env_dir}/logs/${stamp}_${suffix}_arv_active_latency.log"

  "${py}" "${ROOT_DIR}/common/scripts/benchmark_arv_active_cases.py" \
    --mode "${mode}" \
    --manifest "${manifest}" \
    --output-json "${out_json}" \
    --output-log "${out_log}" \
    "$@"
  echo "${out_json}"
}

run_action_for_mode() {
  local action="$1"
  local mode="$2"
  local target="$3"
  shift 3
  case "${action}" in
    discover)
      run_discover_one "${mode}" "${target}" "$@"
      ;;
    benchmark)
      run_benchmark_one "${mode}" "${target}" "$@"
      ;;
    all)
      local manifest
      local count
      manifest="$(run_discover_one "${mode}" "${target}" "$@")"
      count="$(python3 - <<'PY' "${manifest}"
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
print(len(payload.get("cases", [])))
PY
)"
      if [[ "${count}" == "0" ]]; then
        echo "active case가 없어 benchmark를 건너뜁니다: ${manifest}" >&2
        return 2
      fi
      run_benchmark_one "${mode}" "${manifest}" "$@"
      ;;
    *)
      echo "알 수 없는 action: ${action}" >&2
      usage
      exit 1
      ;;
  esac
}

case "${MODE}" in
  cpu|coral|pcie-hat)
    run_action_for_mode "${ACTION}" "${MODE}" "${TARGET}" "$@"
    ;;
  all)
    if [[ "${ACTION}" == "benchmark" ]]; then
      echo "benchmark action에서는 mode=all을 쓸 수 없습니다. manifest는 backend별 결과여야 합니다." >&2
      exit 1
    fi
    run_action_for_mode "${ACTION}" cpu "${TARGET}" "$@" || [[ $? -eq 2 ]]
    run_action_for_mode "${ACTION}" coral "${TARGET}" "$@" || [[ $? -eq 2 ]]
    run_action_for_mode "${ACTION}" pcie-hat "${TARGET}" "$@" || [[ $? -eq 2 ]]
    ;;
  *)
    echo "알 수 없는 mode: ${MODE}" >&2
    usage
    exit 1
    ;;
esac
