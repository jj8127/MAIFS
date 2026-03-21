# SHIELD Phase 4: ONNX 경량화, RPi5 배포, 벤치마크 결과

## 1. ONNX 변환: PyTorch → Edge 배포

### 변환 목표

Raspberry Pi 5에서 동작하려면 PyTorch 모델을 **ONNX(Open Neural Network Exchange)** 형식으로 변환해야 한다. ONNX Runtime은 CPU에서 효율적으로 동작하는 최적화된 추론 엔진이다.

변환한 모델과 크기:

| 모델 | ONNX 크기 |
|------|---------|
| MNV2 | 22.5MB |
| SpecM-v4 | 29.2MB (ONNX), 26.4MB (INT8) |
| SpecG | 141.5MB |
| MobileCLIP-ft4 | 141.3MB |

RPi5 예산 기준 (200ms 이내): **MNV2와 SpecM만 OK**. SpecG와 MobileCLIP은 CPU에서 너무 느리다.

---

## 2. PTQ(Post-Training Quantization): Dynamic INT8

### 포렌식 신호와 양자화의 딜레마

딥리서치 결과 포렌식 신호(PRNU, DCT, SRM)는 low-magnitude, high-frequency 특성을 가진다. INT8 양자화 노이즈가 이 신호보다 커지면 탐지 성능이 손상된다. 이론적으로 포렌식 모델은 양자화에 취약하다.

### 실험 결과: Dynamic INT8은 무손실

**Dynamic INT8** (런타임에 activation 범위 동적 계산): 전 모델 **무손실**

| 모델 | FP32 accuracy | Dynamic INT8 | 차이 |
|------|-------------|--------------|------|
| MNV2 | 95.37% | 95.37% | +0.00%p |
| SpecM-v4 | 65.11% | 65.14% | +0.03%p |
| SpecG | 87.67% | 87.84% | +0.17%p |
| MobileCLIP | 95.32% | 95.38% | +0.06%p |

포렌식 신호가 dynamic quantization에는 안전하게 유지됐다.

**Static INT8** (calibration 데이터로 범위 사전 계산): FastViT 계열 붕괴

| 모델 | 정확도 하락 |
|------|-----------|
| MNV2 | -13.42%p (심각) |
| SpecG | -47.23%p (붕괴) |
| MobileCLIP | -63.25%p (붕괴) |
| SpecM-v4 | -0.85%p (허용 범위) |

FastViT 기반 모델들(SpecG, CLIP)은 Static PTQ에서 무너진다. 이론적으로 예측했던 포렌식 신호 손실이 여기서 나타났다. **Static INT8 사용 금지, Dynamic INT8로 통일** 결정.

---

## 3. ONNX 정확도 검증

ONNX 변환 후 원래 PyTorch 모델과의 출력 일치 여부를 cosine similarity로 검증:
- SpecM-v4 ONNX cosine similarity: **0.99999** (사실상 동일)

ONNX 변환이 모델 출력을 변경하지 않는 것을 확인했다.

---

## 4. RPi5 end-to-end 벤치마크 (Phase 4.4)

### 환경

- 기기: Raspberry Pi 5 (8GB)
- OS: Debian GNU/Linux 13 (Trixie), kernel 6.12.75
- Python: 3.13.5
- onnxruntime: 1.24.4
- 모델: MNV2-Dynamic INT8 + SpecM-v4-Dynamic INT8

### 추론 레이턴시 (threads=4, warmup 제외 10회)

측정값: 108.5 / 95.2 / 118.4 / 135.4 / 106.3 / 112.0 / 90.3 / 100.8 / 127.4 / 125.6 ms

- **평균: 112.0ms**
- 최소: 90.3ms
- 최대: 135.4ms

**예상(140ms) 대비 -20% 빠르다.**

### 스레드별 레이턴시 비교

| 스레드 수 | 평균 레이턴시 |
|---------|------------|
| 1 thread | 192.8ms |
| 2 threads | 128.6ms |
| **4 threads** | **114.3ms** |

Cortex-A76 4코어를 모두 활용하는 4 threads가 최적. 1→4 스레드 감소율은 40%로 메모리 대역폭 한계로 인해 이론적 4배보다 적다.

### 메모리 사용량

- Maximum resident set size: **156.3MB** (두 모델 동시 로드)
- RPi5 8GB RAM의 약 2% → 매우 여유 있음

### 콜드스타트 vs 웜스타트

- 모델 로드 시간 (5회 평균): 188.7ms
- 추론 시간 (5회 평균): 113.8ms
- 전체 콜드스타트: 약 302ms (처음 실행)
- 웜스타트 (모델 로드 후): 112ms

실제 사용에서는 모델을 메모리에 유지하면 112ms 추론이 가능하다.

### ICWMV 정상 동작 확인

실제 이미지로 테스트한 JSON 출력:
```json
{
  "verdict": "ai_generated",
  "confidence": 0.4518,
  "scores": {"authentic": 0.3115, "manipulated": 0.2367, "ai_generated": 0.4518},
  "mnv2_scores": {"authentic": 0.0808, "manipulated": 0.3356, "ai_generated": 0.5836},
  "specm_scores": {"authentic": 0.7240, "manipulated": 0.2760}
}
```

MNV2가 ai_generated 58%로 판정하는 동안 SpecM은 authentic 72%로 반박하는 충돌 케이스. ICWMV가 최종적으로 ai_generated(45% 신뢰도)로 판정 — 불확실 케이스를 낮은 신뢰도로 표현하는 것이 의도한 동작이다.

---

## 5. 사용 방법 (inference_rpi5.py)

RPi5에서 실행하는 방법:

```bash
# 기본 실행
python inference_rpi5.py image.jpg --threads 4

# JSON 출력
python inference_rpi5.py image.jpg --threads 4 --json

# 모델 경로 직접 지정
python inference_rpi5.py image.jpg --mnv2 /path/to/mnv2.onnx --specm /path/to/specm.onnx
```

의존성: `pip install onnxruntime pillow numpy` (약 50MB)

---

## 6. 현재까지 전체 성능 비교

### 서버 vs RPi5 성능 비교

| 시스템 | 모델 | 크기 | 추론 시간 | Macro-F1 |
|--------|------|------|---------|---------|
| 서버 원본 MAIFS | 4 agents | ~1.26GB | ~313ms | 0.8613 (DAAC) |
| 서버 4-model ICWMV | MNV2+CLIP+SpecM+SpecG | ~580MB | ~53ms (GPU) | **96.48%** |
| **RPi5 2-model ICWMV** | **MNV2+SpecM-v4 INT8** | **~46MB** | **~112ms** | **96.58%** |

RPi5 2-model이 서버 4-model보다 **+0.10%p 높다**. 크기는 80배 작고, GPU 추론 대비 2배 느리지만 CPU에서 동작한다.

### 4개 데이터셋별 성능 (RPi5 2-model ICWMV v4 기준)

| 데이터셋 | macro-F1 | 비고 |
|---------|---------|------|
| base (CASIA2+BigGAN) | 95.86% | 기본 in-distribution |
| dsC (CASIA2+IMD2020+BigGAN) | 98.44% | 더 다양한 조작 포함 |
| opensdi (OpenSDI 소셜미디어) | 94.68% | OOD 테스트 |
| aigenproxy (AI-GenBench) | 97.33% | 다양한 생성 모델 |
| **평균** | **96.58%** | |

---

## 7. Phase 4.5: WildRF 벤치마크 (진행 예정)

WildRF는 소셜미디어(Reddit, Twitter, Facebook)에서 수집한 실제 딥페이크 이미지 데이터셋이다. "Locally Aware Deepfake Detection Algorithm"(LaDeDa, arXiv:2406.09398) 논문에서 제안했다.

WildRF 평가 계획:
- 우리 ICWMV 시스템을 WildRF test set (reddit/twitter/facebook 3개 플랫폼)에서 평가
- 3-class → binary 매핑: authentic=real, (manipulated+aigen)=fake
- 비교 기준: Tiny-LaDeDa (WildRF-trained) mAP=93.7%

우리 시스템은 WildRF로 학습하지 않았기 때문에 이것은 순수한 OOD 일반화 테스트다.

---

## 8. 전체 연구 요약 (Phase 1~4)

| Phase | 핵심 내용 | 주요 결과 |
|-------|---------|---------|
| Phase 1 | 4개 에이전트 Shapley/PID/CKA/STII 분석 | SpatialAgent 제거 확정. Freq>Fat>Noise 순위 (데이터셋 의존) |
| Phase 2 | 경량 백본 수급 및 평가 | MNV2(22MB)+MobileCLIP(380MB) 선정. ForMa 제외(CPU 병목) |
| Phase 3 | 3-Track 비교 + Binary Specialist | RPi5: MNV2 단독. 서버: MNV2+CLIP. SpecM 계열(v1→v5b) 개발 |
| Phase 3.5 | ICWMV 4-model 앙상블 | SpecM-v4 + ICWMV → 서버 초과 96.58% |
| Phase 4.1 | ONNX 변환 | MNV2=22MB/14ms, SpecM=30MB/21ms (GPU) |
| Phase 4.2 | PTQ Dynamic INT8 | 전 모델 무손실(Δ≤+0.17%p). RPi5 확정 ~140ms 예상 |
| Phase 4.4 | **RPi5 실측** | **112ms avg, 156MB, 4 threads 최적** |
| Phase 4.5 | WildRF 벤치마크 | 데이터셋 다운로드 후 평가 예정 |
| Phase 4.6 | Embedding CKA 분석 | SpecM RGB/SRM 중복 원인 규명. MobileCLIP CKA=0.028★ |
| Phase 4.7 | SpecM-v5b (MobileCLIP) | 4-DS avg 0.8347 (v4 대비 +1.8%p). opensdi OOD 약점으로 배포는 v4 유지 |

---

## 9. 남은 과제와 미래 방향

### 단기 과제
1. **WildRF 벤치마크 완료**: 실제 소셜미디어 환경에서의 일반화 성능 확인
2. **ForensicHub 평가**: 23개 데이터셋 표준 벤치마크 비교
3. **SpecM-v5b opensdi 개선**: MobileCLIP backbone의 OOD 약점 해결

### 중기 과제
4. **Hailo-8L NPU 경로**: RPi5의 AI 가속 칩 활용 (13 TOPS INT8)
5. **QAT(Quantization-Aware Training)**: Static INT8 성능 복구 시도

### 장기 과제 (SHIELD 논문)
6. **SHIELD 논문 작성**: 5개 Contribution 완성
   - C1: Model Shapley + STII 에이전트 가치 정량화
   - C2: PID 정보 분해
   - C3: QAT + mixed-precision 포렌식 특화 경량화
   - C4: MobileCLIP backbone 교체 + FAA adapter 재학습
   - C5: Confidence-gated cascade (Tier 1→2→3)

현재까지 C1(완료), C4의 일부(SpecM-v5b MobileCLIP backbone, 완료)가 달성됐다.

---

## 10. 참조 핵심 논문

| 논문 | 관련성 |
|------|--------|
| FatFormer (CVPR 2024) | FAA backbone-agnostic 근거 |
| MobileCLIP (CVPR 2024) | S2 variant, DataCompDR 학습 |
| LaDeDa / Tiny-LaDeDa (arXiv 2406.09398) | WildRF 평가 기준, 93.7% mAP |
| Data Shapley (ICML 2019) | Model Shapley 이론 기반 |
| STII (Sundararajan et al. 2020) | Pairwise interaction index |
| Meyen et al. (2021) | Binary specialist 이론적 근거 |
| Nguyen et al. (NeurIPS 2021) | Unbiased Linear CKA (대각 제거) |
| ForensicHub (NeurIPS 2025) | 23 datasets, 42 models 통합 벤치마크 |
