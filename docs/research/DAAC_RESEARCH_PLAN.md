# DAAC: Disagreement-Aware Adaptive Consensus

> Disagreement-Aware Adaptive Consensus for Multi-Agent Image Forensics

## 1. 문제 정의

### 1.1 현재 MAIFS의 한계

현재 MAIFS의 최종 의사결정은 본질적으로 agent output aggregation 중심이다:

```
v_final = Σ w_i · v_i    (가중 투표)
```

- trust score가 하드코딩 (`frequency: 0.85, noise: 0.80, fatformer: 0.85, spatial: 0.85`)
- 입력 이미지 특성(JPEG 여부, 해상도, 생성 모델 종류)에 따른 에이전트별 성능 차이를 반영하지 못함
- 토론/LLM 해석/규칙 기반 분기 등 비선형 경로가 있어 완전한 선형 앙상블은 아니지만, 핵심 의사결정 구조는 가중 집계

### 1.2 학술적 기여 부족

리뷰어 관점에서의 핵심 비판:

> "기존 모델(FFT, PRNU, FatFormer, ViT)을 파이프라인으로 묶고 가중 투표한 것만으로는 학술적 기여로 부족하다."

- 개별 Tool: 기존 기법의 재구현
- 합의 메커니즘: 단순 가중 평균 (COBRA)
- 토론: 판정 정확도 개선 효과 미검증

### 1.3 핵심 가설

**에이전트 간 불일치 패턴 자체가 조작 유형을 암시하는 강력한 탐지 신호이다.**

관찰 기반 근거:
- 실제 이미지: 4개 에이전트 대체로 일치 (AUTHENTIC)
- GAN 이미지: Frequency↑, FatFormer↑, Noise△, Spatial△ → 특정 불일치 패턴
- Diffusion 이미지: Frequency△, FatFormer↑, Noise↑, Spatial△ → 다른 불일치 패턴
- Inpainting: Frequency△, FatFormer△, Noise△, Spatial↑ → 또 다른 패턴

## 2. 제안: DAAC (Disagreement-Aware Adaptive Consensus)

### 2.1 개요

DAAC는 두 가지 메커니즘을 결합한다:

1. **Adaptive Routing** (방향 2): 입력 이미지 특성에 따라 에이전트 가중치를 동적으로 조정
2. **Disagreement Feature Exploitation** (방향 3): 에이전트 간 불일치 패턴 자체를 메타 특징으로 활용

### 2.2 논문 스토리

> "다중 포렌식 에이전트의 개별 판정보다, 에이전트 간 일치/불일치 패턴이 더 강력한 탐지 신호임을 발견했다. 이를 기반으로, 입력 이미지 특성에 따라 에이전트 가중치를 동적으로 조정하고, 불일치 패턴 자체를 메타 특징으로 활용하는 Disagreement-Aware Adaptive Consensus (DAAC)를 제안한다."

### 2.3 기존 COBRA → DAAC 업그레이드

```
COBRA (현재):
  Image → [Agent₁~₄] → verdict/confidence → 고정 가중 투표 → 최종

DAAC (제안):
  Image → [Agent₁~₄] → verdict/confidence ─┐
                                             ├→ 43-dim 메타 특징 → 메타 분류기 → 최종
                                             │
  (pairwise disagreement + aggregate stats) ─┘
```

## 3. Phase 1: Disagreement Pattern Analysis

### 3.1 목표

불일치 패턴이 최종 판정 및 조작 유형 예측에 유의미한지 검증한다.

### 3.2 메타 특징 설계 (43-dim)

#### Per-agent features (20 dim)

| 특징 | 차원 | 설명 |
|------|------|------|
| verdict one-hot | 16 | 4 categories (AUTH/MANIP/AI_GEN/UNCERTAIN) × 4 agents |
| confidence | 4 | 각 에이전트의 confidence score |

#### Pairwise disagreement features (18 dim, 6 pairs)

| 특징 | 차원 | 설명 |
|------|------|------|
| binary disagreement | 6 | 같은 verdict인지 (0/1) |
| confidence difference | 6 | \|c_i - c_j\| |
| conflict strength | 6 | 불일치 시 \|c_i - c_j\|, 일치 시 0 |

**conflict strength의 핵심 역할:**

단순 disagreement(같다/다르다)보다 "얼마나 확신하면서 서로 다른가"가 더 정보량이 높다.

- Frequency=AI_GENERATED(0.9) vs FatFormer=AUTHENTIC(0.85) → conflict 1.75 (강한 충돌)
- Frequency=UNCERTAIN(0.4) vs Noise=UNCERTAIN(0.5) → conflict 0 (약한 불일치)

#### Aggregate features (5 dim)

| 특징 | 차원 | 설명 |
|------|------|------|
| confidence 분산 | 1 | var(c₁, c₂, c₃, c₄) |
| verdict 엔트로피 | 1 | -Σ p(v) log p(v) |
| max-min gap | 1 | max(c) - min(c) |
| 고유 verdict 수 | 1 | \|{v₁, v₂, v₃, v₄}\| |
| majority 비율 | 1 | 다수파 에이전트 비율 (0.25~1.0) |

### 3.3 실험 경로

**1차: 경로 B (시뮬레이션)**
- 에이전트별 confusion matrix + confidence 분포를 논문/벤치마크에서 추출
- 에이전트 간 오류 상관성 모델링 포함
- 즉시 43-dim feature/ablation/통계검정 파이프라인 검증
- 목표: "disagreement 단독 신호가 유의미한가" 빠른 판정

**2차: 경로 A (실데이터)**
- FatFormer/Spatial 체크포인트 확보 후 실제 에이전트 출력으로 재실행
- B→A 성능 갭은 논문에서 "simulation-to-real gap"으로 명시
- 동일 코드(collector만 교체)로 실행

### 3.4 시뮬레이터 설계 원칙

각 에이전트의 예상 성능 분포를 다음 기반으로 설정:

| Agent | 강점 | 약점 | 참조 |
|-------|------|------|------|
| Frequency (FFT) | GAN upsampling 아티팩트 | Diffusion, JPEG 혼동 | Durall et al. 2020 |
| Noise (PRNU/SRM) | 카메라 진위, 센서 노이즈 부재 탐지 | 고압축, 리사이즈에 취약 | Chen et al. 2008, Fridrich & Kodovsky 2012 |
| FatFormer (CLIP+DWT) | Diffusion 탐지, 교차 일반화 | 부분 조작 미탐지 | Liu et al. CVPR 2024 |
| Spatial (ViT) | 부분 조작 영역 탐지 | AI 전체 생성 약함 | Guillaro et al. 2023 |

에이전트 간 상관성 모델링:
- Frequency ↔ FatFormer: 중간 상관 (둘 다 주파수 경로 사용)
- Noise ↔ 나머지: 낮은 상관 (독립적 신호 원천)
- Spatial ↔ 나머지: 낮은 상관 (픽셀 수준 분석)

### 3.5 베이스라인

| 베이스라인 | 설명 |
|-----------|------|
| Majority Vote | 단순 다수결 (동률 시 UNCERTAIN) |
| COBRA (현재) | RoT/DRWA/AVGA 고정 가중 합의 |
| Logistic Regression | 43-dim → verdict (linear meta-classifier) |
| XGBoost | 비선형 메타 분류기 |
| MLP (소형) | 43 → 32 → 16 → 4 (비선형, 과적합 모니터링) |

### 3.6 Ablation 설계

| ID | Feature Set | 차원 | 검증 대상 |
|----|------------|------|----------|
| A1 | confidence만 | 4 | confidence 단독 정보량 |
| A2 | verdict만 | 16 | verdict 패턴 정보량 |
| A3 | disagreement만 | 23 | **핵심: 불일치 단독 신호** |
| A4 | verdict + confidence | 20 | 기존 앙상블 수준 |
| A5 | Full (V+C+D) | 43 | DAAC 완전체 |
| A6 | 에이전트 제거 (−1) | 각 ~33 | 각 에이전트의 기여도 |

**핵심 ablation: A3 (disagreement만)**
- A3가 random baseline보다 유의하게 높아야 "불일치 자체가 신호"라는 가설 성립
- A3가 marginal contribution만 있으면 방향 3의 기여가 사라지고, 논문 스토리가 "adaptive routing이 좋다"로 축소됨

### 3.7 데이터 분할 (누수 방지)

1. **생성기 교차 split**: train 생성기 ≠ test 생성기
   - Train: SD 1.5, Midjourney, BigGAN
   - Test: DALL-E 3, SD XL, StyleGAN3
2. **데이터셋 교차 split**: GenImage train → 타 데이터셋 test
3. **동일 이미지/근접 변형 중복 제거**

### 3.8 평가 지표

| 지표 | 용도 |
|------|------|
| Macro-F1 | 클래스 불균형 보정 주요 지표 |
| Balanced Accuracy | 보조 지표 |
| AUROC (OvR) | 확률 기반 평가 |
| ECE (Expected Calibration Error) | confidence calibration |
| Brier Score | calibration 보조 |
| Class-wise Confusion Matrix | 에이전트/조작유형별 분석 |

### 3.9 통계 검정

| 검정 | 용도 |
|------|------|
| McNemar Test | 분류 성능 비교 (쌍대) |
| Bootstrap CI (95%) | 신뢰 구간 추정 (1000회 리샘플링) |

### 3.10 Go/No-Go 기준

Phase 1 → Phase 2 진행 조건:

1. **Full 모델(A5)이 COBRA 대비 Macro-F1 유의 개선** (McNemar p < 0.05)
2. **교차 데이터셋에서도 성능 유지** (Macro-F1 하락 < 5%p)
3. **Disagreement 단독(A3)이 random baseline보다 유의하게 높음** (가설 검증)

3개 조건 모두 충족 시 Go → Phase 2 (Adaptive Routing) 착수.

## 4. Phase 2: Adaptive Routing (조건부)

### 4.1 개요

Phase 1에서 Go 판정 시, 경량 Meta-Router Network를 도입하여 이미지 특성에 따라 에이전트 가중치를 동적으로 생성한다.

```python
image_features = lightweight_encoder(image)    # e.g., ResNet-18
agent_weights = softmax(MLP(image_features))   # [w_freq, w_noise, w_fat, w_spatial]
final = meta_classifier(agent_results, agent_weights, disagreement_features)
```

### 4.2 Phase 1과의 결합

Phase 1의 메타 분류기에 router-generated weights를 추가 입력으로 제공:

```
43-dim (Phase 1) + 4-dim (router weights) = 47-dim → meta classifier
```

## 5. Phase 3: 벤치마크 및 논문 작성 (조건부)

### 5.1 필수 실험

- SOTA 대비 정량 비교 (GenImage, DiffusionForensics)
- 단일 모델 vs. 다중 에이전트 앙상블 성능/강건성 비교
- 토론 메커니즘 실효성 ablation
- Latency/throughput 시스템 벤치마크

### 5.2 추가 실험 (선택)

- 적대적 공격 시나리오 (FGSM/PGD) 하 강건성 평가
- JPEG 재압축, 리사이즈 등 후처리 강건성
- 에이전트 수 스케일링 실험 (2~6개)

## 6. 구현 코드 구조

```
src/meta/
├── __init__.py
├── collector.py         # 경로 A: 실제 에이전트 실행 → agent_outputs.jsonl
├── simulator.py         # 경로 B: confusion matrix 기반 synthetic output 생성
├── features.py          # 43-dim 메타 특징 추출
├── baselines.py         # majority vote, COBRA wrapper
├── trainer.py           # LogReg, XGBoost, MLP 학습/평가
├── evaluate.py          # Macro-F1, AUROC, ECE, McNemar, bootstrap CI
└── ablation.py          # 6가지 feature set ablation runner

experiments/
├── run_phase1.py        # Phase 1 전체 파이프라인 실행 스크립트
└── configs/
    └── phase1.yaml      # 데이터셋 경로, 하이퍼파라미터, split 설정
```

## 7. 리뷰어별 대응 전략

| 리뷰어 관점 | 핵심 비판 | DAAC로 대응 |
|------------|----------|------------|
| R1 (Vision) | 기존 모델 조합, 벤치마크 부재 | 메타 분류기가 새 알고리즘, GenImage 벤치마크 |
| R2 (MAS) | 피상적 정보 교환, 에이전트 상관성 | 불일치 패턴이 심층 정보 교환, 상관성 활용 |
| R3 (Systems) | 성능 프로파일링 부재 | Phase 3에서 latency 벤치마크 |
| R4 (Security) | 적대적 강건성 미검증 | 다중 에이전트 동시 공격 어려움 실험적 증명 |
| R5 (Industry) | 실제 데이터 검증 없음 | 경로 A 실데이터 + 교차 데이터셋 검증 |

## 8. Phase 1-B 실험 결과 (2026-02-12)

### 8.1 실험 환경

- 시뮬레이션 데이터: Gaussian copula 기반 합성 에이전트 출력
- Train: 8000, Val: 1000, Test: 1000 (균등 레이블 분포)
- 생성기 교차 split 적용 (train/test 서브타입 분포 상이)

### 8.2 베이스라인 vs 메타 분류기

| 방법 | Macro-F1 | Balanced Acc | AUROC | ECE |
|------|----------|-------------|-------|-----|
| Majority Vote | 0.9467 | 0.9465 | — | — |
| COBRA (DRWA) | 0.9787 | 0.9791 | 0.9937 | 0.0649 |
| LogReg (43-dim) | 0.9595 | 0.9595 | 0.9940 | 0.0181 |
| **GBM (43-dim)** | **0.9949** | **0.9950** | **1.0000** | **0.0036** |
| MLP (43-dim) | 0.9908 | 0.9909 | 0.9996 | 0.0046 |

### 8.3 Ablation 결과 (best model 기준)

| ID | Feature Set | Dim | Best F1 | Best Model |
|----|------------|-----|---------|------------|
| A1 | confidence only | 4 | 0.6521 | GBM |
| A2 | verdict only | 16 | 0.9343 | LogReg |
| A3 | **disagreement only** | 23 | **0.6407** | GBM |
| A4 | verdict + confidence | 20 | 0.9959 | GBM |
| A5 | Full (V+C+D) | 43 | 0.9949 | GBM |

에이전트 제거 ablation (A6, GBM 기준):
| 제거 에이전트 | Macro-F1 | 하락폭 |
|-------------|----------|--------|
| − FatFormer | 0.9938 | −0.0011 |
| − Frequency | 0.9899 | −0.0050 |
| − Noise | 0.9877 | −0.0072 |
| − Spatial | 0.9879 | −0.0070 |

### 8.4 Go/No-Go 판정: **GO**

| 조건 | 결과 | 상세 |
|------|------|------|
| C1: A5 > COBRA | **PASS** | F1 +0.0162, McNemar p=0.0004 |
| C2: 교차 데이터셋 | **PASS** | F1 drop 0.51%p < 5%p |
| C3: A3 > random | **PASS** | F1 0.641 >> 0.383 |

### 8.5 주요 발견 및 해석

1. **A4 ≈ A5**: verdict+confidence(20-dim)만으로 full(43-dim)과 동등한 성능. 시뮬레이션의 한계로, 에이전트 오류 상관 구조가 단순하여 disagreement 특징의 추가 정보량이 제한적. 실데이터(Path A)에서 재검증 필요.

2. **A3 유의미성**: disagreement만으로 random 대비 +30.7%p 개선. 불일치 패턴 자체가 탐지 신호임을 확인. 다만 verdict 정보 없이는 한계 존재.

3. **특징 중요도**: GBM 기준 `fatformer_verdict_ai_generated`(0.185), `spatial_verdict_manipulated`(0.154), `fatformer_confidence`(0.144) — 각 에이전트의 전문성이 반영됨.

4. **에이전트 기여도**: Noise와 Spatial 제거 시 가장 큰 성능 하락 → 독립적 신호 원천의 가치 확인.

### 8.6 다음 단계

- **Phase 1-A**: 실제 에이전트 출력으로 재실행 (체크포인트 확보 시)
- **Phase 2**: Adaptive Routing — 이미지 특성 기반 동적 가중치 생성

## 9. 타임라인

| 단계 | 내용 | 상태 |
|------|------|------|
| Phase 1-B | 시뮬레이션 기반 가설 검증 | ✅ 완료 (GO) |
| Phase 1-A | 실데이터 검증 | 대기 (체크포인트 필요) |
| Phase 2 | Adaptive Routing | 착수 가능 |
| Phase 3 | 벤치마크 + 논문 | Phase 2 완료 후 |

## 10. 데이터셋 (Phase 1-A용)

| 데이터셋 | 용도 | 소스 |
|----------|------|------|
| CASIA v2.0 | Manipulation (Tp/Au/GT) | Kaggle: `divg07/casia-20-image-tampering-detection-dataset` |
| DIV2K | Authentic (HR) | Kaggle: `soumikrakshit/div2k-high-resolution-images` |
| IMD2020 | Inpainting | `https://staff.utia.cas.cz/novozada/db/` |
| GenImage (BigGAN) | AI Generated | `https://github.com/GenImage-Dataset/GenImage` |

모든 데이터셋은 `datasets/` 디렉토리에 위치 (git 미추적).

## 11. 참고 문헌

- Liu et al., "Forgery-aware Adaptive Transformer for Generalizable Synthetic Image Detection," CVPR 2024 (FatFormer)
- Durall et al., "Watch your Up-Convolution: CNN Based Generative Deep Neural Networks are Failing to Reproduce Spectral Distributions," CVPR 2020
- Chen et al., "Determining image origin and integrity using sensor noise," IEEE TIFS 2008
- Fridrich & Kodovsky, "Rich Models for Steganalysis of Digital Images," IEEE TIFS 2012
- Guillaro et al., "TruFor: Leveraging all-round clues for trustworthy image forgery detection and localization," CVPR 2023
- Zhu et al., "GenImage: A Million-Scale Benchmark for Detecting AI-Generated Image," NeurIPS 2023
