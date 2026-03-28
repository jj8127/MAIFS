#!/usr/bin/env bash
set -euo pipefail

ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="${1:-${ENV_DIR}/.venv}"
REQ_FILE="${ENV_DIR}/requirements.txt"

python3 -m venv "${VENV_PATH}"
"${VENV_PATH}/bin/python" -m pip install --upgrade pip setuptools wheel
"${VENV_PATH}/bin/python" -m pip install -r "${REQ_FILE}"

cat <<EOF

완료되었습니다.

중요:
  1. libedgetpu 런타임이 설치되어 있어야 합니다.
  2. PCIe HAT 장치가 lspci 와 /dev/apex* 에서 확인되어야 합니다.

다음 실행:
  cd ${ENV_DIR}/..
  CORAL_PYTHON=${VENV_PATH}/bin/python bash run_arv_e2e_benchmark.sh pcie-hat

EOF
