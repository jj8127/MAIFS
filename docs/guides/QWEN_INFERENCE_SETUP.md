# Qwen 다운로드 및 Inference 셋업

MAIFS 문서에서 가정한 Qwen(vLLM) 경로를 실제로 재현하기 위한 실행 가이드입니다.

## 1) 원클릭 셋업

프로젝트 루트에서 실행:

```bash
cd /home/dsu/Desktop/MAIFS
./scripts/setup_qwen_inference.sh
```

기본 동작:
- `.venv-qwen` 가상환경 생성
- `envs/requirements-qwen-vllm.txt` 설치
- `Qwen/Qwen2.5-32B-Instruct` 다운로드 (`~/models/qwen2.5-32b-instruct`)

환경변수로 동작 제어:

```bash
# 모델만 바꿔서 설치/다운로드
MODEL_ID="Qwen/Qwen2.5-14B-Instruct" ./scripts/setup_qwen_inference.sh

# 다운로드는 건너뛰고 의존성만 설치
DOWNLOAD_MODEL=0 ./scripts/setup_qwen_inference.sh

# 의존성 설치는 건너뛰고 모델만 다운로드
INSTALL_DEPS=0 ./scripts/setup_qwen_inference.sh
```

## 2) 수동 다운로드 (선택)

```bash
cd /home/dsu/Desktop/MAIFS
source .venv-qwen/bin/activate
python scripts/download_qwen_model.py \
  --model-id Qwen/Qwen2.5-32B-Instruct \
  --local-dir ~/models/qwen2.5-32b-instruct
```

## 3) vLLM 서버 실행

```bash
cd /home/dsu/Desktop/MAIFS
source .venv-qwen/bin/activate
MODEL_NAME="$HOME/models/qwen2.5-32b-instruct" ./scripts/start_vllm_server.sh
```

필요 시 GPU 지정:

```bash
CUDA_VISIBLE_DEVICES=0,1 MODEL_NAME="$HOME/models/qwen2.5-32b-instruct" ./scripts/start_vllm_server.sh
```

## 4) Inference 스모크 테스트

```bash
cd /home/dsu/Desktop/MAIFS
source .venv-qwen/bin/activate
python scripts/test_vllm_inference.py --base-url http://localhost:8000
```

## 5) MAIFS 연동 추론 확인

```bash
cd /home/dsu/Desktop/MAIFS
source .venv-qwen/bin/activate
python scripts/example_qwen_analysis.py \
  --demo \
  --url http://localhost:8000 \
  --model "$HOME/models/qwen2.5-32b-instruct"
```

## 6) 추가 참고

- vLLM 서버 시작 스크립트: `scripts/start_vllm_server.sh`
- 모델 다운로드 스크립트: `scripts/download_qwen_model.py`
- 추론 테스트 스크립트: `scripts/test_vllm_inference.py`
