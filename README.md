# MAIFS

**Multi-Agent Image Forensic System** — 4개 포렌식 에이전트가 합의(Consensus)·토론(Debate)으로 협력하여 이미지의 진위(authentic / manipulated / AI-generated)를 판별하는 시스템.

---

## 연구 트랙 개요

| 트랙 | 제목 | 상태 |
|------|------|------|
| **Track 1** | **DAAC** — Disagreement-Aware Adaptive Consensus | 실험 완료, 논문 작성 중 (KIPS 2026) |
| **Track 2** | **SHIELD** — Shapley-based Hardware-aware Interaction-preserving Ensemble Lightweighting for on-Device forensics | 진행 중 (2차 논문) |

---

## Track 1: DAAC (1차 논문)

> **핵심 가설**: 에이전트 간 불일치 패턴(disagreement) 자체가 조작 유형을 암시하는 탐지 신호다.

### 시스템 구성 (서버 기준)

| Agent | Backend | 크기 | 추론 | 맹점 |
|-------|---------|------|------|------|
| FrequencyAgent | CAT-Net v2 (HRNet-W48) | ~150MB | ~80ms | AI-generated 탐지 불가 |
| NoiseAgent | MVSS-Net | ~120MB | ~61ms | AI-generated 탐지 불가 |
| FatFormerAgent | CLIP ViT-L/14 + FAA | ~890MB | ~57ms | Manipulated 탐지 불가 |
| SpatialAgent | Mesorch (ViT) | ~100MB | ~97ms | AI-generated 탐지 불가 |
| **합계** | | **~1.26GB** | **~313ms** | |

> **에이전트 맹점 구조**가 DAAC의 핵심 학습 신호. `disagree_frequency_fatformer` GBM 특징 중요도 1위 (56.5%).

### 최종 실험 결과

**Protocol-P** (1,500장 실데이터, 10 seeds):

| 방법 | Macro-F1 | Cohen's κ |
|------|----------|----------|
| COBRA (기존 합의) | 0.266 | 0.068 |
| **DAAC-GBM** | **0.861 ± 0.016** | **0.796** |

**Protocol-M** (6개 데이터 조합, 60 runs): sign 60/0, p = 1.63e-11, Δ = +0.493

**DAAC 재학습 (경량 백본 기준, Phase 3.4)**:

| 설정 | avg macro-F1 |
|------|-------------|
| DAAC-GBM (25-dim, MNV2+CLIP) | 4-DS avg = 96.25% |
| ICWMV 4-model (SpecM-v4 기준) | 4-DS avg = **96.58%** |

---

## Track 2: SHIELD (2차 논문, 진행 중)

> **핵심 목표**: DAAC 성능을 유지하면서 Raspberry Pi 5에 배포 가능한 경량 포렌식 아키텍처 설계.

### Phase별 진행 현황

**Phase 1: Agent Valuation** ✅

- Model Shapley: freq=0.2690 > fat=0.1216 > noise=0.0886 > spatial=0.0547
- PID: spatial Unique ≈ 0 → 4/4 데이터셋에서 제거 가능
- CKA: 모든 에이전트 쌍 < 0.1 (특징 독립적)

**Phase 2: 경량 백본 수급 및 단독 평가** ✅

| 백본 | 역할 | 크기 | 4-DS avg |
|------|------|------|---------|
| MobileNetV2 dual-stream | 3-class generalist | 5.77M / 22.5MB | 95.8% |
| MobileCLIP-S2 finetuned (ft4) | 3-class generalist | 99.4M / 380MB | 93.2% |
| Specialist-M v4 | binary: auth vs manip | 7.66M / 29MB | val manip_f1=0.7792 |
| Specialist-G | binary: auth vs aigen | 35.91M | aigen_f1=0.981 |

**Phase 3: 앙상블 조합 선정** ✅

| 조합 | 4-DS avg macro-F1 | RPi5 추정 |
|------|-----------------|----------|
| MNV2 단독 | 95.81% | ~57ms |
| **ICWMV: MNV2 + SpecM-v4** | **96.58%** | **~140ms** ✓ |
| ICWMV: 4-model (서버) | 96.48% | — |

> RPi5 2-model ICWMV(96.58%)가 서버 4-model(96.48%)을 **+0.10%p 초과 달성**

**Phase 4: Edge Deployment** ✅ (서버 준비 완료, RPi5 실측 대기)

| 모델 | ONNX 크기 | Dynamic INT8 Δ | 서버 1T | RPi5 추정 |
|------|----------|---------------|--------|---------|
| MNV2 | 22.5MB | **+0.00%p** ✓ | 14ms | ~56ms |
| SpecM-v4 | 29.2MB | **cosine=0.99999** ✓ | 21ms | ~84ms |
| SpecG | 141.5MB | +0.17%p ✓ | 200ms | ~800ms (서버 전용) |
| CLIP | 141.3MB | +0.06%p ✓ | 197.7ms | ~791ms (서버 전용) |

> Static INT8: MNV2 -13%p, SpecG/CLIP 붕괴 → **Dynamic INT8 확정**

### SpecM 학습 이력

| 버전 | 변경 | val manip_f1 | openSDI auth_recall | ICWMV avg |
|------|------|-------------|---------------------|-----------|
| v1 | CASIA2 only | 0.764 | 7% | — |
| v2 | +IMD2020 +JPEG aug | 0.827 | 11% | 96.48% (4-model) |
| v3 | +GenImage_nature +RandomErasing(p=0.3) | 0.7832 | 62% | 96.43% |
| **v4** | **v3 resume + LR=3e-5 + RandomErasing(value=random)** | 0.7792 | **70.3%** | **96.58%** |

---

## RPi5 배포 (즉시 실행 가능)

### 필요 파일 (총 ~45MB)

```
weights/onnx_quant/mnv2_int8_dynamic.onnx     (19MB)
weights/onnx_quant/specm_v4_int8_dynamic.onnx (26MB)
inference_rpi5.py
```

### 서버 → RPi5 파일 전송

```bash
scp weights/onnx_quant/mnv2_int8_dynamic.onnx \
    weights/onnx_quant/specm_v4_int8_dynamic.onnx \
    inference_rpi5.py \
    pi@<RPi5_IP>:~/maifs/
```

### RPi5에서 실행

```bash
# 의존성 (경량, 추가 ML 라이브러리 불필요)
pip3 install onnxruntime pillow numpy

# 추론
python3 inference_rpi5.py image.jpg
python3 inference_rpi5.py image.jpg --json          # JSON 출력
python3 inference_rpi5.py image.jpg --threads 4     # 4-core 활용 (~100ms)
python3 inference_rpi5.py image.jpg --w-spec 1.0    # ICWMV 가중치
```

### Coral USB Accelerator (추가 가속)

Coral Edge TPU는 TFLite Static INT8 모델이 필요합니다. 서버에서 변환 후 배포:

```bash
# 서버: 변환 도구 설치
pip install tensorflow-cpu onnx2tf

# Edge TPU 컴파일러 (Ubuntu x86_64)
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" \
  | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
sudo apt update && sudo apt install edgetpu-compiler

# RPi5: pycoral
pip3 install pycoral
```

예상 가속: MNV2 on Edge TPU ~5ms (현재 ~56ms → 약 11×)

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
python3 -m venv .venv-qwen
source .venv-qwen/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

---

## 체크포인트 준비

저장소에 대용량 체크포인트는 포함되지 않습니다.

### 서버 체크포인트 (DAAC)

| 모델 | 파일 | 경로 |
|------|------|------|
| CAT-Net full | `CAT_full_v2.pth.tar` | `CAT-Net-main/output/splicing_dataset/CAT_full/` |
| CAT-Net DCT | `DCT_djpeg.pth.tar` | `CAT-Net-main/pretrained_models/` |
| CAT-Net RGB | `hrnetv2_w48_imagenet_pretrained.pth` | `CAT-Net-main/pretrained_models/` |
| Mesorch | `mesorch-98.pth` | `Mesorch-main/mesorch/` |
| MVSS-Net | `mvssnet_casia.pt` | `MVSS-Net-master/ckpt/` |
| FatFormer CLIP | `ViT-L-14.pt` | `Integrated Submodules/FatFormer/pretrained/` |
| FatFormer finetuned | `fatformer.pth` | `Integrated Submodules/FatFormer/checkpoint/` |

### RPi5 배포 모델 (SHIELD Phase 4)

| 파일 | 크기 | 설명 |
|------|------|------|
| `weights/onnx_quant/mnv2_int8_dynamic.onnx` | 19MB | MNV2 Dynamic INT8 |
| `weights/onnx_quant/specm_v4_int8_dynamic.onnx` | 26MB | SpecM-v4 Dynamic INT8 |

---

## 데이터셋 준비

```
datasets/
├── CASIA2_subset/        Tp/  Au/  GT/
├── IMD2020_subset/       IMD2020_Generative_Image_Inpainting_yu2018_01/images/
├── GenImage_subset/      BigGAN/val/{ai,nature}/
└── OpenSDID_subset/      authentic/  manipulated/  ai_generated/
```

소스: CASIA v2.0 · IMD2020 · GenImage · OpenSDI-Dataset

---

## 주요 스크립트

| 스크립트 | 설명 |
|---------|------|
| `inference_rpi5.py` | **RPi5 배포용** ICWMV 추론 (onnxruntime only) |
| `experiments/train_specialist_m_v4.py` | SpecM-v4 fine-tuning (v3 resume + LR=3e-5) |
| `experiments/train_specialist_m_v3.py` | SpecM-v3 학습 (GenImage+RandomErasing) |
| `experiments/run_onnx_export.py` | PyTorch → ONNX 변환 + CPU 벤치마크 |
| `experiments/run_quantization.py` | Dynamic / Static INT8 PTQ |
| `experiments/run_quant_accuracy_eval.py` | 양자화 모델 macro-F1 평가 |
| `experiments/run_icwmv_consensus.py` | ICWMV 4-model 합의 평가 |
| `main.py` | CLI 진입점 (서버용 전체 파이프라인) |
| `app.py` | Gradio Web UI |

---

## 환경변수

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `MAIFS_DEVICE` | 자동 | `cuda` / `cpu` 강제 |
| `MAIFS_SPATIAL_BACKEND` | `mesorch` | `mesorch` / `omniguard` |
| `MAIFS_NOISE_BACKEND` | `mvss` | `mvss` / `prnu` |
| `MAIFS_META_USE_GPU` | `1` | 메타학습 GPU 사용 |
| `MAIFS_CATNET_CHECKPOINT` | 코드 기본값 | CAT-Net 체크포인트 경로 오버라이드 |

---

## 주요 경로

| 경로 | 설명 |
|------|------|
| `AGENTS.md` | 프로젝트 상태·로드맵 SSOT |
| `CLAUDE.md` | 아키텍처·코딩 규칙 가이드 |
| `docs/research/SHIELD_RESEARCH_PLAN.md` | SHIELD 후속 연구 상세 계획 |
| `docs/research/DAAC_RESEARCH_PLAN.md` | DAAC 1차 연구 계획 (완료) |
| `docs/SHIELD_Phase4_PTQ_Report_20260320.pdf` | Phase 4 PTQ 시행착오 전체 보고서 |
| `weights/onnx_quant/` | RPi5 배포용 INT8 ONNX 모델 |
| `experiments/results/` | 모든 실험 결과 JSON/JSONL |
