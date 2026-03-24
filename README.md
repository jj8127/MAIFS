# MAIFS

**Multi-Agent Image Forensic System** 연구 저장소.

2026-03-24 기준으로 이 저장소는 두 축으로 정리됩니다.

- **DAAC**: 서버 환경의 4-agent 합의 시스템
- **SHIELD / ICWMV**: DAAC 아이디어를 경량 백본과 엣지 환경으로 옮긴 후속 연구

현재 논문 작성의 중심은 **MobileNetV2의 confident-but-wrong 오류를 어떻게 교정할 것인가**, 그리고 이를 위해 제안한 **역신뢰도 가중 다수결(ICWMV)** 입니다.

---

## 현재 상태 요약

| 트랙 | 초점 | 상태 |
|------|------|------|
| **DAAC** | 4-agent disagreement-aware server consensus | 실험 완료, 논문화 진행 |
| **SHIELD** | 경량 백본 탐색 + on-device 포렌식 | 핵심 실험 완료 |
| **ICWMV 논문선** | MNV2 + SpecM-v4 + fixed-rule consensus | 초안 작성 중 |
| **EWCT 실험선** | complementary specialist 학습 | supporting evidence 정리 완료 |

---

## 현재까지 확정된 핵심 결과

### 1. DAAC

- 4개 전문가 에이전트와 메타 분류기를 사용하는 서버형 합의 시스템
- Protocol-P 기준 `DAAC-GBM = 0.861 ± 0.016`, `COBRA = 0.266`
- 다중 전문가의 **불일치(disagreement)** 자체가 강한 탐지 신호라는 점을 확인

### 2. 경량 백본 선택

DAAC 스택은 엣지 장치에 너무 무거워서, 여러 경량 백본을 비교한 뒤 `MNV2`가 주 generalist로 선택됐습니다.

| 백본 | 역할 | 4-DS 결과 | 비고 |
|------|------|-----------|------|
| `MobileNetV2 dual-stream` | 3-class generalist | **0.9581** | 현재 주 backbone |
| `MobileCLIP-S2 ft4` | 3-class generalist | 0.9532 | 전이 실험용 strong backbone |
| `MobileCLIP zero-shot` | 3-class generalist | 0.3005 | weak operating point stress test |
| `SpecM-v4` | auth/manip specialist | ICWMV용 기준 전문가 | 현재 주 specialist |

### 3. ICWMV

ICWMV는 **Inverse-Confidence Weighted Majority Vote**, 한국어로는 **역신뢰도 가중 다수결**입니다.

- 동기: `MNV2`의 이진(auth/manip) 오분류 중 `74.3%`가 `confidence > 0.6`
- 따라서 confidence threshold 기반 cascade는 오분류 대부분에 개입하지 못함
- ICWMV는 모든 샘플에 specialist 신호를 반영하는 fixed-rule consensus

강한 MNV2 기준 4-DS LOO-CD:

| 방법 | avg macro-F1 | 교정률 |
|------|-------------|--------|
| `MNV2 only` | 0.9581 | 0.0% |
| `Cascade + SpecM-v4` | 0.9612 | 19.3% |
| **`ICWMV + SpecM-v4`** | **0.9652** | **35.4%** |

### 4. 백본 전이 / 강도 실험

ICWMV는 특정 체크포인트 하나에만 맞춘 규칙이 아니라는 점을 확인했습니다.

| Generalist | 설정 | Backbone F1 | ICWMV F1 | ΔF1 | 교정률 |
|------------|------|-------------|----------|-----|--------|
| MNV2 | strong | 0.9581 | 0.9652 | +0.71%p | 35.4% |
| MNV2 | weak | 0.8414 | 0.8637 | +2.23%p | 49.8% |
| MobileCLIP | fine-tuned | 0.9532 | 0.9569 | +0.37%p | 19.7% |
| MobileCLIP | zero-shot | 0.3005 | 0.4320 | +13.14%p | 77.1% |
| MNV2 | no-finetuning | 0.3658 | 0.5556 | +18.98%p | 80.8% |

핵심 해석은 **backbone family보다 backbone strength가 약할수록 ICWMV 이득이 커진다**는 점입니다.

### 5. EWCT

EWCT는 **Error-Weighted Complementary Training**입니다.

- 목적: `MNV2`가 어려워하거나 틀릴 가능성이 큰 샘플에 `SpecM`이 더 집중하도록 학습
- clean한 결과:
  - `w_max`는 이론적으로 불필요하며 실험으로도 제거 가능
  - `confident-but-wrong` 분석이 cascade 실패를 정량적으로 설명
- 현재 결론:
  - EWCT는 **교정률 확장용 operating point**로는 유효
  - learned scalar fusion(HEMA/action-gate/veto)은 아직 ICWMV를 안정적으로 넘지 못함

`ICWMV + EWCT-noTS`:

- avg macro-F1: `0.9575`
- 교정률: `49.2%`

즉, **F1 우선 운영점은 `ICWMV + v4`**, **교정률 우선 운영점은 `ICWMV + EWCT-noTS`** 입니다.

### 6. RPi5 / Coral 배포

실험용 배포 경로도 정리돼 있습니다.

| 경로 | avg macro-F1 | RPi5 평균 레이턴시 | 비고 |
|------|-------------|-------------------|------|
| ONNX INT8 CPU | 0.9535 | 88.3ms | 현재 가장 안정적 |
| Coral USB Edge TPU | 0.9308 | 63.4ms | 속도 우위, 정확도 gap 존재 |

Coral 경로는 compile 자체는 성공했고 실제 RPi5 실측도 완료됐지만, 현재 연구 판단상 **정확도 기준 주력선은 여전히 ONNX** 입니다.

---

## 현재 논문 포지션

현 시점에서 가장 defensible한 논문 뼈대는 다음 세 가지입니다.

1. `MNV2`의 **confident-but-wrong** 패턴 정량화
2. 이를 교정하기 위한 **ICWMV(역신뢰도 가중 다수결)** 제안
3. `EWCT`에서의 **w_max 제거**와 learned fusion의 한계 분석

반대로 `HEMA`는 현재까지의 공정 재평가 기준으로는 **핵심 주장보다는 한계 사례 / exploratory line**에 가깝습니다.

---

## 문서 가이드

| 문서 | 역할 |
|------|------|
| [AGENTS.md](AGENTS.md) | 프로젝트 SSOT, 연구 진행상황, ledger |
| [docs/PROGRESS_REPORT.md](docs/PROGRESS_REPORT.md) | 현재 연구 진행상황 요약 |
| [docs/research/PAPER_DRAFT_v1.md](docs/research/PAPER_DRAFT_v1.md) | KIPS 2026 논문 초안 |
| [docs/research/PAPER_DRAFT_NOTES.md](docs/research/PAPER_DRAFT_NOTES.md) | 논문용 수치/테이블/서술 메모 |
| [docs/research/EWCT_EXPERIMENT_REPORT.md](docs/research/EWCT_EXPERIMENT_REPORT.md) | EWCT/HEMA/ICWMV 비교 보고서 |
| [deploy/README.md](deploy/README.md) | RPi5 / Coral 배포 가이드 |
| [docs/research/RPi5_EXPERIMENT_GUIDE.md](docs/research/RPi5_EXPERIMENT_GUIDE.md) | 실측 절차와 기록 가이드 |

---

## 주요 스크립트

| 스크립트 | 설명 |
|---------|------|
| `inference_rpi5.py` | RPi5용 통합 추론 엔진 (`onnx` / `tflite` / `edgetpu`) |
| `experiments/run_icwmv_backbone_transfer.py` | strong/weak/no-ft backbone transfer 실험 |
| `experiments/run_icwmv_specm_compare.py` | SpecM variant별 ICWMV 비교 |
| `experiments/run_fuser_loo_cd.py` | fixed-rule / learned fusion LOO-CD 비교 |
| `experiments/run_hema_action_gate_loo_cd.py` | action-gate learned fusion 복구 시도 |
| `experiments/run_hema_icwmv_veto_loo_cd.py` | ICWMV selective veto 실험 |
| `experiments/run_edgetpu_export.py` | ONNX -> full INT8 TFLite -> Edge TPU compile |
| `experiments/run_rpi5_model_export.py` | ONNX 양자화 export 및 검증 |

---

## 빠른 시작

```bash
git clone https://github.com/jj8127/MAIFS.git
cd MAIFS
python3 -m venv .venv-qwen
source .venv-qwen/bin/activate
pip install -r requirements.txt
```

배포/추론은 아래 문서를 바로 보는 편이 빠릅니다.

- [deploy/README.md](deploy/README.md)
- [docs/research/RPi5_EXPERIMENT_GUIDE.md](docs/research/RPi5_EXPERIMENT_GUIDE.md)

---

## 체크포인트와 결과물

대용량 체크포인트는 저장소에 포함되지 않습니다. 실험 결과 JSON/JSONL과 양자화/Coral 산출물은 `experiments/results/`, `weights/onnx_quant/`, `weights/tflite/`, `weights/tflite_edgetpu/` 아래에 정리됩니다.

---

## 현재 우선순위

1. `PAPER_DRAFT_v1.md`와 참고문헌/표 캡션 마감
2. `ICWMV` 중심 논문 서사 정교화
3. 필요 시 Coral 결과는 본문보다 부록/후속연구로 분리

