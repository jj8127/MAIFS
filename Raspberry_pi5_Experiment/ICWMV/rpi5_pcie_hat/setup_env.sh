#!/usr/bin/env bash
set -euo pipefail

ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${ENV_DIR}/../../.." && pwd)"
VENV_PATH="${1:-${REPO_ROOT}/.venv-coral39}"
REQ_FILE="${ENV_DIR}/requirements.txt"
SYNC_SCRIPT="${ENV_DIR}/../common/sync_models.sh"
INFER_SCRIPT="${ENV_DIR}/../common/scripts/rpi5_infer_coral.py"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv가 필요합니다. https://docs.astral.sh/uv/ 를 참고해 먼저 설치하세요." >&2
  exit 1
fi

echo "[1/4] Python 3.9 설치 확인"
uv python install 3.9

echo "[2/4] 가상환경 생성: ${VENV_PATH}"
uv venv --python 3.9 --seed "${VENV_PATH}"

echo "[3/4] pip 업그레이드"
"${VENV_PATH}/bin/python" -m pip install --upgrade pip setuptools wheel

echo "[4/4] Coral 런타임 의존성 설치"
"${VENV_PATH}/bin/python" -m pip install -r "${REQ_FILE}"

cat <<EOF

완료되었습니다.

다음 단계:
  1. libedgetpu 런타임이 RPi5에 설치되어 있는지 확인
  2. 공통 모델을 다시 동기화하려면
     bash ${SYNC_SCRIPT}
  3. Coral 추론 실행
     ${VENV_PATH}/bin/python ${INFER_SCRIPT} image.jpg --json

EOF
