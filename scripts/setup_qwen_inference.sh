#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv-qwen}"

MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-32B-Instruct}"
QWEN_LOCAL_DIR="${QWEN_LOCAL_DIR:-$HOME/models/qwen2.5-32b-instruct}"
DOWNLOAD_MODEL="${DOWNLOAD_MODEL:-1}"
INSTALL_DEPS="${INSTALL_DEPS:-1}"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Usage:
  ./scripts/setup_qwen_inference.sh

Environment variables:
  PYTHON_BIN        Python executable (default: python3)
  VENV_DIR          Virtual env path (default: ./MAIFS/.venv-qwen)
  MODEL_ID          HF model id (default: Qwen/Qwen2.5-32B-Instruct)
  QWEN_LOCAL_DIR    Local model dir (default: ~/models/qwen2.5-32b-instruct)
  INSTALL_DEPS      1/0 install dependencies (default: 1)
  DOWNLOAD_MODEL    1/0 download model (default: 1)
EOF
  exit 0
fi

echo "============================================"
echo "MAIFS Qwen Inference Setup"
echo "============================================"
echo "ROOT_DIR: $ROOT_DIR"
echo "PYTHON_BIN: $PYTHON_BIN"
echo "VENV_DIR: $VENV_DIR"
echo "MODEL_ID: $MODEL_ID"
echo "QWEN_LOCAL_DIR: $QWEN_LOCAL_DIR"
echo "DOWNLOAD_MODEL: $DOWNLOAD_MODEL"
echo "INSTALL_DEPS: $INSTALL_DEPS"
echo "============================================"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "[error] python not found: $PYTHON_BIN" >&2
  exit 1
fi

if [ -d "$VENV_DIR" ] && [ ! -f "$VENV_DIR/bin/activate" ]; then
  rm -rf "$VENV_DIR"
fi

if [ ! -d "$VENV_DIR" ] || [ ! -f "$VENV_DIR/bin/activate" ]; then
  if "$PYTHON_BIN" -c "import ensurepip" >/dev/null 2>&1; then
    if ! "$PYTHON_BIN" -m venv "$VENV_DIR"; then
      rm -rf "$VENV_DIR"
      if command -v uv >/dev/null 2>&1; then
        uv venv --seed --clear "$VENV_DIR"
      else
        echo "[error] failed to create virtual environment with $PYTHON_BIN -m venv" >&2
        echo "[hint] install uv or python-venv package, then rerun." >&2
        exit 1
      fi
    fi
  else
    if command -v uv >/dev/null 2>&1; then
      uv venv --seed --clear "$VENV_DIR"
    else
      echo "[error] python ensurepip is unavailable and uv is not installed" >&2
      echo "[hint] install uv or python-venv package, then rerun." >&2
      exit 1
    fi
  fi
fi

# shellcheck source=/dev/null
if [ ! -f "$VENV_DIR/bin/activate" ]; then
  echo "[error] virtual environment is missing activate script: $VENV_DIR/bin/activate" >&2
  exit 1
fi
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel

if [ "$INSTALL_DEPS" = "1" ]; then
  python -m pip install -r "$ROOT_DIR/envs/requirements-qwen-vllm.txt"
fi

if [ "$DOWNLOAD_MODEL" = "1" ]; then
  python "$ROOT_DIR/scripts/download_qwen_model.py" \
    --model-id "$MODEL_ID" \
    --local-dir "$QWEN_LOCAL_DIR"
fi

cat <<EOF

[done] setup complete

Next steps:
1) Start vLLM server
   source "$VENV_DIR/bin/activate"
   MODEL_NAME="$QWEN_LOCAL_DIR" ./scripts/start_vllm_server.sh

2) Run inference smoke test
   source "$VENV_DIR/bin/activate"
   python scripts/test_vllm_inference.py --base-url http://localhost:8000

3) Run MAIFS Qwen demo
   source "$VENV_DIR/bin/activate"
   python scripts/example_qwen_analysis.py --demo --url http://localhost:8000 --model "$QWEN_LOCAL_DIR"
EOF
