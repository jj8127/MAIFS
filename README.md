# MAIFS

**Multi-Agent Image Forensic System** — 4개 포렌식 에이전트가 합의(Consensus)·토론(Debate)으로 협력하여 이미지의 진위(authentic / manipulated / AI-generated)를 판별하는 시스템.

---

## 연구 트랙 개요

이 저장소는 두 개의 독립적인 연구 트랙을 포함합니다.

| 트랙 | 제목 | 상태 | 문서 |
|------|------|------|------|
| **Track 1** | **DAAC** — Disagreement-Aware Adaptive Consensus | 실험 완료, 논문 작성 중 (KIPS 2026) | [DAAC_RESEARCH_PLAN.md](docs/research/DAAC_RESEARCH_PLAN.md) |
| **Track 2** | **SHIELD** — Shapley-based Hardware-aware Interaction-preserving Ensemble Lightweighting for on-Device forensics | 진행 중 (2차 논문) | [SHIELD_RESEARCH_PLAN.md](docs/research/SHIELD_RESEARCH_PLAN.md) |

---

## Track 1: DAAC (1차 논문)

> **핵심 가설**: 에이전트 간 불일치 패턴(disagreement) 자체가 조작 유형을 암시하는 탐지 신호다.

### 시스템 구성 (서버 기준)

| Agent | Backend | 크기 | 추론 | 역할 | 맹점 |
|-------|---------|------|------|------|------|
| FrequencyAgent | CAT-Net v2 (HRNet-W48) | ~150MB | ~80ms | JPEG 이중압축 아티팩트 탐지 | AI-generated 탐지 불가 |
| NoiseAgent | MVSS-Net | ~120MB | ~61ms | 픽셀 수준 조작 마스크 예측 | AI-generated 탐지 불가 |
| FatFormerAgent | CLIP ViT-L/14 + FAA | ~890MB | ~57ms | AI 생성 이미지 탐지 | Manipulated 탐지 불가 |
| SpatialAgent | Mesorch (ViT) | ~100MB | ~97ms | 부분 조작 영역 분할 | AI-generated 탐지 불가 |
| **합계** | | **~1.26GB** | **~313ms** | | |

> **에이전트 맹점 구조**가 DAAC의 핵심 학습 신호. 특히 `disagree_frequency_fatformer`가 GBM 특징 중요도 1위(56.5%).

### DAAC 합의 계층

- 입력: 4개 에이전트 출력 → **43-dim 메타 특징** 추출
- 분류기: GBM (Gradient Boosting Machine)
- 추론 비용: 0.069ms (전체의 **0.02%** 미만)

### 최종 실험 결과

**Protocol-P** (1,500장 실데이터, 10 seeds):

| 방법 | Macro-F1 | Cohen's κ |
|------|----------|----------|
| COBRA (기존 합의) | 0.266 | 0.068 |
| **DAAC-GBM** | **0.861 ± 0.016** | **0.796** |
| p-value (Wilcoxon) | **0.00195** | |

**Protocol-M** (6개 데이터 조합, 60 runs): sign 60/0, p = 1.63e-11, Δ = +0.493

**DAAC 재학습 (경량 백본 기준, Phase 3.4)**:

| 설정 | avg macro-F1 |
|------|-------------|
| DAAC-GBM (25-dim, MNV2+CLIP only) | base=99.01% / 4-DS 평균=96.25% |
| ICWMV 4-model (SpecM-v2 기준) | 4-DS 평균=**96.48%** |

### 관련 실험 스크립트

```bash
# DAAC 재학습 (경량 25-dim 기준)
.venv-qwen/bin/python experiments/run_daac_retrain_lightweight.py

# ICWMV 4-model 합의 평가
.venv-qwen/bin/python experiments/run_icwmv_consensus.py

# CKA 다양성 분석
.venv-qwen/bin/python experiments/run_cka_diversity_4model.py
```

결과: `experiments/results/daac_retrain/`, `experiments/results/icwmv/`, `experiments/results/cka_diversity/`

---

## Track 2: SHIELD (2차 논문, 진행 중)

> **핵심 목표**: DAAC 성능을 유지하면서 Raspberry Pi 5에 배포 가능한 경량 포렌식 아키텍처 설계.

### 스펙 목표

| 지표 | 현재 (서버) | 목표 (RPi5) |
|------|-----------|------------|
| 모델 총 크기 | ~1.26GB | <500MB (목표 ~180MB) |
| 추론 시간 | ~313ms | <1초 |
| Macro-F1 | 0.861 | >0.80 |
| 타겟 디바이스 | GPU 서버 | Raspberry Pi 5 (8GB) |

### 5대 기여 (Contributions)

| # | 기여 | 방법론 | 상태 |
|---|------|--------|------|
| C1 | 에이전트 가치·상호작용 정량화 | Model Shapley (N=4) + STII | `DONE` |
| C2 | 고유/중복/시너지 정보 분해 | PID (Partial Information Decomposition) | `DONE` |
| C3 | 포렌식 특화 경량화 | QAT + mixed-precision | `NOT_STARTED` |
| C4 | 백본 교체 + 어댑터 전이 | FatFormer → MobileCLIP-S2 (~890→50MB) | `IN_PROGRESS` |
| C5 | Confidence-gated cascade | Tier 1→2→3 조건부 추론 | `NOT_STARTED` |

### Phase별 진행 현황

**Phase 1: Agent Valuation** ✅

- Model Shapley: freq=0.2690 > fat=0.1216 > noise=0.0886 > spatial=0.0547
- Spatial 제거: 4/4 데이터셋 확정 (Unique ≈ 0)
- 핵심 발견: freq↔fatformer 상호작용 음수(-0.1823, 대체재 관계) — greedy 선택 부적합

**Phase 2: 경량 백본 수급 및 단독 평가** ✅

| 백본 | 역할 | 크기 | 4-DS 평균 acc |
|------|------|------|--------------|
| MobileNetV2 dual-stream | 3-class generalist | 5.77M / 22.5MiB | 95.8% |
| MobileCLIP-S2 finetuned | 3-class generalist | 99.4M / 380.3MiB | 93.2% |
| Specialist-M v2 | binary: auth vs manip | 7.66M / 29.3MiB | manip F1=0.827 |
| Specialist-G | binary: auth vs aigen | 35.91M (0.10M trainable) | aigen F1=0.981 |
| Tiny-LaDeDa | cascade Tier-1 스크리너 | — | ai_gen recall 73-86% |

**Phase 3: Track 비교 및 조합 선정** ✅

3-Track 비교 결과:
- **RPi5 권고**: MNV2-only (22.5MiB / CPU 35.9ms)
- **GPU/서버 권고**: CLIP+MNV2 4-model (402.8MiB)
- **ForMa 전면 제거**: CPU 1613ms 병목 + Shapley ≈ 0

**Phase 4: Edge Deployment** (진행 중)

| 모델 | ONNX 크기 | 서버 1-thread | RPi5 추정 | 양자화 결과 |
|------|----------|-------------|---------|-----------|
| MNV2 | 22.5MB | 14ms | ~57ms | Dynamic ×1.04, Static ×1.05 |
| SpecM | 30MB | 20.6ms | ~82ms | **Static ×1.25** (cos=1.00 ✓) |
| SpecG | 141.5MB | 200ms | ~800ms (서버 전용) | 붕괴(cos<0.2) ✗ |
| CLIP | 141.3MB | 197.7ms | ~791ms (서버 전용) | 붕괴(cos<0.2) ✗ |

> **RPi5 배포 Scenario A** (추천): MNV2-FP32 + SpecM-INT8-Static → 합계 예상 ~500ms
> SpecG/CLIP은 FastViT attention 양자화 불가 → **서버 하이브리드 아키텍처**로 전환

```bash
# ONNX 변환 + 서버 벤치마크
.venv-qwen/bin/python experiments/run_onnx_export.py

# INT8 양자화 (Dynamic + Static PTQ)
.venv-qwen/bin/python experiments/run_quantization.py

# 양자화 모델 정확도 평가 (macro-F1)
.venv-qwen/bin/python experiments/run_quant_accuracy_eval.py
```

---

## 설치

### 요구 사항

- OS: Linux (Ubuntu 권장)
- Python: 3.10+
- GPU (권장): NVIDIA + CUDA
- 디스크: 체크포인트 포함 시 수십 GB+

### 환경 구성

```bash
git clone https://github.com/jj8127/MAIFS.git
cd MAIFS

# venv (기본)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# CUDA 환경
pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision
pip install -r requirements.txt

# 선택 의존성 (CAT-Net/FatFormer/Mesorch)
pip install -r requirements-optional-tools.txt
```

---

## 체크포인트 준비

저장소에 대용량 체크포인트는 포함되지 않습니다. 아래 파일을 지정 경로에 배치해야 합니다.

### 필수 체크포인트

| 모델 | 파일 | 경로 |
|------|------|------|
| CAT-Net full | `CAT_full_v2.pth.tar` | `CAT-Net-main/output/splicing_dataset/CAT_full/` |
| CAT-Net DCT | `DCT_djpeg.pth.tar` | `CAT-Net-main/pretrained_models/` |
| CAT-Net RGB | `hrnetv2_w48_imagenet_pretrained.pth` | `CAT-Net-main/pretrained_models/` |
| Mesorch | `mesorch-98.pth` | `Mesorch-main/mesorch/` |
| MVSS-Net | `mvssnet_casia.pt` | `MVSS-Net-master/ckpt/` |
| FatFormer CLIP | `ViT-L-14.pt` | `Integrated Submodules/FatFormer/pretrained/` |
| FatFormer finetuned | `fatformer.pth` | `Integrated Submodules/FatFormer/checkpoint/` |

**다운로드 출처:**
1. CAT-Net: https://drive.google.com/drive/folders/1hBEfnFtGG6q_srBHVEmbF3fTq0IhP8jq
2. Mesorch: https://drive.google.com/drive/folders/1jwYv-S3HAZqzz0YxM9bJynBiPv-O9-6x
3. MVSS-Net: https://drive.google.com/drive/folders/1CztGkd91xF1QqEXuc2n8rVDTBJ7X695U
4. FatFormer: 프로젝트 운영자 배포 파일 사용

### 상태 점검

```bash
find CAT-Net-main/output/splicing_dataset/CAT_full/CAT_full_v2.pth.tar \
  CAT-Net-main/pretrained_models/DCT_djpeg.pth.tar \
  Mesorch-main/mesorch/mesorch-98.pth \
  MVSS-Net-master/ckpt/mvssnet_casia.pt \
  "Integrated Submodules/FatFormer/checkpoint/fatformer.pth" \
  -maxdepth 0 -type f
```

---

## 데이터셋 준비

```
datasets/
├── CASIA2_subset/
│   ├── Tp/  Au/  GT/
├── IMD2020_subset/
│   └── IMD2020_Generative_Image_Inpainting_yu2018_01/
│       ├── images/  masks/
└── GenImage_subset/
    └── BigGAN/val/
        ├── ai/  nature/
```

소스: CASIA v2.0 (Kaggle) · IMD2020 (staff.utia.cas.cz) · GenImage (github.com/GenImage-Dataset/GenImage)

---

## 기본 실행

```bash
# CLI 분석
export MAIFS_NOISE_BACKEND=mvss
export MAIFS_SPATIAL_BACKEND=mesorch
python main.py analyze /path/to/image.jpg --algorithm drwa --device cuda

# Web UI
python main.py server --host 0.0.0.0 --port 7860

# 도구 단위 성능 평가
python scripts/evaluate_tools.py --max-samples 100 \
  --noise-backend mvss --spatial-backend-a mesorch \
  --out outputs/tool_eval.json
```

---

## 환경변수

| 변수 | 설명 | 기본값 |
|------|------|--------|
| `MAIFS_DEVICE` | 전역 디바이스 강제 (`cuda`/`cpu`) | 자동 감지 |
| `MAIFS_SPATIAL_BACKEND` | Spatial 백엔드 (`mesorch`, `omniguard`) | `mesorch` |
| `MAIFS_NOISE_BACKEND` | Noise 백엔드 (`mvss`, `prnu`) | `mvss` |
| `MAIFS_META_USE_GPU` | 메타학습 GPU 사용 (`1/0`) | `1` |
| `MAIFS_CATNET_CHECKPOINT` | CAT-Net 체크포인트 경로 오버라이드 | 코드 기본값 |
| `MAIFS_MESORCH_CHECKPOINT` | Mesorch 체크포인트 경로 오버라이드 | `Mesorch-main/mesorch/mesorch-98.pth` |
| `MAIFS_MVSS_CHECKPOINT` | MVSS 체크포인트 경로 오버라이드 | `MVSS-Net-master/ckpt/mvssnet_casia.pt` |

---

## 트러블슈팅

**GPU 미인식**: `nvidia-smi` 및 `python -c "import torch; print(torch.cuda.is_available())"` 확인 후 CUDA wheel 재설치.

**CAT-Net fallback**: `CAT_full_v2.pth.tar`, `DCT_djpeg.pth.tar`, `hrnetv2_w48_imagenet_pretrained.pth` 3개 경로 확인. `requirements-optional-tools.txt` 기준으로 `jpegio`, `torch-dct` 설치 필요.

**Spatial 성능 저하**: `MAIFS_SPATIAL_BACKEND=mesorch` 고정 후 `Mesorch-main/mesorch/mesorch-98.pth` 파일 확인.

---

## 주요 경로

| 경로 | 설명 |
|------|------|
| `AGENTS.md` | 프로젝트 상태·로드맵 SSOT |
| `CLAUDE.md` | 아키텍처·코딩 규칙 가이드 |
| `docs/research/DAAC_RESEARCH_PLAN.md` | DAAC 연구 계획·실험 결과 |
| `docs/research/SHIELD_RESEARCH_PLAN.md` | SHIELD 후속 연구 상세 계획 |
| `docs/research/MAIFS_TECHNICAL_THEORY.md` | 기술 이론 백서 |
| `docs/research/papers/` | 논문 파일 (DAAC_KIPS_Paper.docx 등) |
| `docs/research/OPERATIONS_RISK_PRIORITIES.md` | 운영 리스크 이력 |
| `src/tools/` | CAT-Net / MVSS / FatFormer / Spatial 구현 |
| `src/agents/` | 4개 전문가 에이전트 |
| `src/meta/` | DAAC 메타 특징·학습·평가 모듈 |
| `experiments/` | 실험 스크립트 (run_*.py, train_*.py) |
| `experiments/results/` | 모든 실험 결과 JSON/JSONL |
