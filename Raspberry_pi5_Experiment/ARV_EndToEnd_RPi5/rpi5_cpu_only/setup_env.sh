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

다음 실행:
  ${VENV_PATH}/bin/python --version
  cd ${ENV_DIR}/..
  bash run_arv_e2e_benchmark.sh cpu

EOF
