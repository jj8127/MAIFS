#!/usr/bin/env bash
set -euo pipefail

ARV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAIFS_ROOT="$(cd "${ARV_DIR}/../.." && pwd)"
DATA_DIR="${ARV_DIR}/data"

mkdir -p "${DATA_DIR}/experiments/results/backbone_eval"
mkdir -p "${DATA_DIR}/experiments/results/specm_eval"
mkdir -p "${DATA_DIR}/experiments/results/specm_complementary_eval"
mkdir -p "${DATA_DIR}/experiments/results/hema_icwmv_veto"
mkdir -p "${DATA_DIR}/experiments/results/generalist_arv"
mkdir -p "${DATA_DIR}/experiments/results/arv_backbone_transfer"

copy_if_exists() {
  local src="$1"
  local dst_dir="$2"
  if [[ -f "${src}" ]]; then
    cp -f "${src}" "${dst_dir}/"
    echo "[OK] $(basename "${src}")"
  else
    echo "[MISS] ${src}"
  fi
}

echo "=== ARV 서버 재현 데이터 동기화 ==="
echo "MAIFS 루트: ${MAIFS_ROOT}"
echo "대상 경로 : ${DATA_DIR}"

cp -f "${MAIFS_ROOT}/experiments/run_hema_icwmv_veto_loo_cd.py" \
  "${DATA_DIR}/experiments/"
cp -f "${MAIFS_ROOT}/experiments/run_comp_nots_richer_veto.py" \
  "${DATA_DIR}/experiments/"
cp -f "${MAIFS_ROOT}/experiments/run_icwmv_backbone_transfer.py" \
  "${DATA_DIR}/experiments/"

find "${MAIFS_ROOT}/experiments/results/backbone_eval" \
  -maxdepth 1 -type f -name 'mobilenetv2_dualstream_*.jsonl' -exec cp -f {} "${DATA_DIR}/experiments/results/backbone_eval/" \;

find "${MAIFS_ROOT}/experiments/results/specm_eval" \
  -maxdepth 1 -type f -name 'specm_comp_noTS_*.jsonl' -exec cp -f {} "${DATA_DIR}/experiments/results/specm_eval/" \;

find "${MAIFS_ROOT}/experiments/results/specm_complementary_eval" \
  -maxdepth 1 -type f -exec cp -f {} "${DATA_DIR}/experiments/results/specm_complementary_eval/" \;

copy_if_exists \
  "${MAIFS_ROOT}/experiments/results/hema_icwmv_veto/hema_icwmv_veto_loo_cd_20260323_114321.json" \
  "${DATA_DIR}/experiments/results/hema_icwmv_veto"
copy_if_exists \
  "${MAIFS_ROOT}/experiments/results/hema_icwmv_veto/hema_icwmv_veto_meta_warmstart_20260325_082654.json" \
  "${DATA_DIR}/experiments/results/hema_icwmv_veto"

echo
echo "완료. 현재 data/experiments/results 파일 수:"
find "${DATA_DIR}/experiments/results" -type f | wc -l
echo
echo "참고:"
echo "  - MobileCLIP 계열 generalist/backbone transfer 전체 재현에는 추가 JSONL과 datasets 자산이 필요하다."
echo "  - Raspberry Pi 5 시간 측정은 Server_Reproduction/ARV가 아니라 Raspberry_pi5_Experiment 아래에서 수행한다."
