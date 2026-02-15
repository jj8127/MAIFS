# MAIFS

Multi-Agent Image Forensic System

MAIFS는 4개 포렌식 에이전트의 결과를 합의(Consensus) 및 토론(Debate)으로 통합해 이미지 진위를 판정하는 프로젝트입니다.

## 1. 현재 기준 아키텍처 (2026-02-13)

| 슬롯 | Agent/Tool | 기본 백엔드 | 역할 |
|------|------------|-------------|------|
| Frequency (Compression) | `FrequencyAgent` / `CATNetAnalysisTool` | CAT-Net v2 | JPEG 압축/이중 압축 아티팩트 기반 조작 탐지 |
| Noise | `NoiseAgent` / `NoiseAnalysisTool` | PRNU/SRM (권장: MVSS) | 센서 노이즈 및 조작 노이즈 불일치 탐지 |
| Semantic | `FatFormerAgent` / `FatFormerTool` | FatFormer (CLIP+DWT) | AI 생성 이미지 탐지 |
| Spatial | `SpatialAgent` / `SpatialAnalysisTool` | Mesorch (default) | 픽셀 단위 조작 영역 분할 |

추가로 DAAC 실험 파이프라인(`src/meta/`, `experiments/run_phase1.py`)에서 메타 분류기를 학습해 COBRA 대비 성능 향상을 검증합니다.

## 2. 빠른 설치 (처음 사용자 권장)

### 2.1 요구 사항

1. OS: Linux(Ubuntu 권장)
2. Python: 3.10 이상 (현재 로컬 기준 3.12에서 동작 확인)
3. GPU(권장): NVIDIA + CUDA 사용 가능 드라이버
4. 디스크: 체크포인트/데이터셋 포함 시 수십 GB 이상

### 2.2 저장소 클론

```bash
git clone https://github.com/jj8127/MAIFS.git
cd MAIFS
```

### 2.3 가상환경 생성 (venv 권장)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

GPU를 쓸 경우 PyTorch를 먼저 설치한 뒤 나머지 의존성을 설치하세요.

```bash
# 예시: CUDA 12.4 wheel
pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision
pip install -r requirements.txt
```

CPU만 사용할 경우:

```bash
pip install -r requirements.txt
```

### 2.4 Conda 대안 경로

```bash
conda env create -f envs/conda-maifs-main.yml
conda activate maifs
```

Qwen/vLLM 추가 기능이 필요하면:

```bash
conda env update -n maifs -f envs/conda-maifs-qwen-addon.yml
```

### 2.5 설치 확인

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("cuda_device_count:", torch.cuda.device_count())
PY
python main.py version
```

## 3. 체크포인트 준비 (필수)

이 저장소에는 대용량 체크포인트가 기본 포함되지 않습니다. 아래 파일을 지정 경로에 배치해야 권장 성능이 나옵니다.

### 3.1 필수 체크포인트

| 구성 | 파일명 | 배치 경로 |
|------|--------|-----------|
| CAT-Net (full) | `CAT_full_v2.pth.tar` | `CAT-Net-main/output/splicing_dataset/CAT_full/CAT_full_v2.pth.tar` |
| CAT-Net pretrained (DCT) | `DCT_djpeg.pth.tar` | `CAT-Net-main/pretrained_models/DCT_djpeg.pth.tar` |
| CAT-Net pretrained (RGB) | `hrnetv2_w48_imagenet_pretrained.pth` | `CAT-Net-main/pretrained_models/hrnetv2_w48_imagenet_pretrained.pth` |
| Mesorch | `mesorch-98.pth` | `Mesorch-main/mesorch/mesorch-98.pth` |
| MVSS | `mvssnet_casia.pt` | `MVSS-Net-master/ckpt/mvssnet_casia.pt` |
| FatFormer CLIP | `ViT-L-14.pt` | `Integrated Submodules/FatFormer/pretrained/ViT-L-14.pt` |
| FatFormer finetuned | `fatformer.pth` | `Integrated Submodules/FatFormer/checkpoint/fatformer.pth` |

### 3.2 선택 체크포인트 (A/B 비교 또는 구형 백엔드 실험용)

| 구성 | 파일명 | 배치 경로 |
|------|--------|-----------|
| OmniGuard | `model_checkpoint_01500.pt` | `OmniGuard-main/checkpoint/model_checkpoint_01500.pt` |
| OmniGuard ViT | `iml_vit.pth` | `OmniGuard-main/checkpoint/iml_vit.pth` |
| TruFor | `trufor.pth.tar` | `TruFor-main/TruFor_train_test/pretrained_models/trufor.pth.tar` |

### 3.3 공식 다운로드 출처

1. CAT-Net weights: https://drive.google.com/drive/folders/1hBEfnFtGG6q_srBHVEmbF3fTq0IhP8jq?usp=sharing
2. Mesorch weights: https://drive.google.com/drive/folders/1jwYv-S3HAZqzz0YxM9bJynBiPv-O9-6x?usp=sharing
3. MVSS weights: https://drive.google.com/drive/folders/1CztGkd91xF1QqEXuc2n8rVDTBJ7X695U?usp=sharing
4. OmniGuard checkpoint zip: https://drive.google.com/file/d/1khdBDUDIRIhPIKlV0ictcbTdWLh-WFY_/view?usp=sharing
5. FatFormer 가중치: 프로젝트 운영자가 배포한 파일을 위 경로에 배치

### 3.4 체크포인트 준비 상태 점검 명령

```bash
find \
  CAT-Net-main/output/splicing_dataset/CAT_full/CAT_full_v2.pth.tar \
  CAT-Net-main/pretrained_models/DCT_djpeg.pth.tar \
  CAT-Net-main/pretrained_models/hrnetv2_w48_imagenet_pretrained.pth \
  Mesorch-main/mesorch/mesorch-98.pth \
  MVSS-Net-master/ckpt/mvssnet_casia.pt \
  "Integrated Submodules/FatFormer/pretrained/ViT-L-14.pt" \
  "Integrated Submodules/FatFormer/checkpoint/fatformer.pth" \
  -maxdepth 0 -type f
```

## 4. 데이터셋 준비

평가 스크립트 기준으로 아래 디렉터리 구조가 필요합니다.

```text
datasets/
├── CASIA2_subset/
│   ├── Tp/
│   ├── Au/
│   └── GT/
├── IMD2020_subset/
│   └── IMD2020_Generative_Image_Inpainting_yu2018_01/
│       ├── images/
│       └── masks/
└── GenImage_subset/
    └── BigGAN/val/
        ├── ai/
        └── nature/
```

소스 참고:

1. CASIA v2.0: Kaggle `divg07/casia-20-image-tampering-detection-dataset`
2. IMD2020: https://staff.utia.cas.cz/novozada/db/
3. GenImage: https://github.com/GenImage-Dataset/GenImage

## 5. 기본 실행

### 5.1 CLI 분석

```bash
export MAIFS_NOISE_BACKEND=mvss
export MAIFS_SPATIAL_BACKEND=mesorch
python main.py analyze /path/to/image.jpg --algorithm drwa --device cuda
```

결과를 저장하려면:

```bash
python main.py analyze /path/to/image.jpg -o outputs/report.json
```

### 5.2 Web UI 실행

```bash
export MAIFS_NOISE_BACKEND=mvss
export MAIFS_SPATIAL_BACKEND=mesorch
python main.py server --host 0.0.0.0 --port 7860
```

## 6. 도구 단위 성능 평가 (권장)

최신 재평가 스크립트:

```bash
python scripts/evaluate_tools.py \
  --max-samples 100 \
  --noise-backend mvss \
  --spatial-backend-a omniguard \
  --spatial-backend-b mesorch \
  --out outputs/tool_reeval_spatial_ab_mesorch_100.json
```

빠른 스모크 테스트:

```bash
python scripts/evaluate_tools.py --max-samples 20 --out outputs/tool_reeval_smoke_20.json
```

## 7. DAAC 메타모델 재학습 (GPU)

```bash
export MAIFS_META_USE_GPU=1
python experiments/run_phase1.py experiments/configs/phase1_mesorch_retrain.yaml
```

결과는 기본적으로 `experiments/results/phase1_mesorch_retrain/` 아래에 저장됩니다.

최신 예시 결과:

1. `experiments/results/phase1_mesorch_retrain/phase1_results_20260213_161739.json`
2. `experiments/results/phase1_mesorch_retrain/ablation_report.txt`

## 8. 자주 쓰는 환경변수

| 변수 | 설명 | 기본값 |
|------|------|--------|
| `MAIFS_DEVICE` | 전역 디바이스 강제 (`cuda`/`cpu`) | 자동 감지 |
| `MAIFS_SPATIAL_BACKEND` | Spatial 백엔드 (`mesorch`, `omniguard`, `trufor`) | `mesorch` |
| `MAIFS_MESORCH_CHECKPOINT` | Mesorch 체크포인트 경로 오버라이드 | `Mesorch-main/mesorch/mesorch-98.pth` |
| `MAIFS_MVSS_CHECKPOINT` | MVSS 체크포인트 경로 오버라이드 | `MVSS-Net-master/ckpt/mvssnet_casia.pt` |
| `MAIFS_NOISE_BACKEND` | Noise 백엔드 (`mvss`, `prnu`) | `prnu` |
| `MAIFS_CATNET_CHECKPOINT` | CAT-Net full 체크포인트 경로 오버라이드 | 코드 기본 경로 |
| `MAIFS_CATNET_CONFIG` | CAT-Net yaml 경로 오버라이드 | 코드 기본 경로 |
| `MAIFS_META_USE_GPU` | Phase1 메타학습 GPU 사용 여부 (`1/0`) | `1` |

## 9. 트러블슈팅

### 9.1 GPU가 안 쓰이는 경우

1. `nvidia-smi`로 드라이버/GPU 인식 확인
2. `python -c "import torch; print(torch.cuda.is_available())"` 확인
3. CUDA 미지원 torch 설치 시 GPU 비활성화되므로 torch 재설치
4. Phase1 학습은 로그에 `[xgboost/cuda]`, `[torch/cuda]`가 보이면 정상

### 9.2 CAT-Net fallback이 뜨는 경우

`CAT_full_v2.pth.tar`, `DCT_djpeg.pth.tar`, `hrnetv2_w48_imagenet_pretrained.pth` 3개가 모두 정확한 위치에 있는지 확인하세요.

### 9.3 Spatial 성능이 낮게 나오는 경우

1. `MAIFS_SPATIAL_BACKEND=mesorch`로 고정
2. `Mesorch-main/mesorch/mesorch-98.pth` 파일 확인
3. `scripts/evaluate_tools.py` A/B 결과에서 backend별 F1 비교

## 10. 주요 경로

| 경로 | 설명 |
|------|------|
| `main.py` | CLI 엔트리포인트 |
| `app.py` | Gradio UI |
| `src/tools/` | CAT-Net/MVSS/FatFormer/Spatial 구현 |
| `src/agents/` | 4개 전문가 에이전트 |
| `src/meta/` | DAAC 메타 특징/학습/평가 |
| `experiments/run_phase1.py` | Phase1 실험 파이프라인 |
| `scripts/evaluate_tools.py` | 도구 단위 재평가 스크립트 |
| `CLAUDE.md` | 프로젝트 SSOT 운영 문서 |
| `docs/research/DAAC_RESEARCH_PLAN.md` | 연구 계획/실험 결과 문서 |

## 11. 참고 문서

1. 프로젝트 가이드: `CLAUDE.md`
2. 연구 계획: `docs/research/DAAC_RESEARCH_PLAN.md`
3. 기술 이론 백서: `docs/research/MAIFS_TECHNICAL_THEORY.md`
4. 실험 결과: `experiments/results/`
5. 도구 평가 결과: `outputs/`
