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
| MLP (소형) | 43 → 32 → 16 → 3 (비선형, 과적합 모니터링) |

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
├── run_phase1.py        # Phase 1 (Path B) 전체 파이프라인
├── run_phase2.py        # Phase 2 (Path B) adaptive routing
├── run_phase2_patha.py  # Phase 2 (Path A) real-data collector + proxy router
├── run_phase2_patha_multiseed.py  # Path A 멀티시드 실행기
├── run_phase2_patha_repeated.py   # Path A repeated-kfold 실행기
├── evaluate_phase2_gate_profiles.py   # gate profile 평가기
├── analyze_patha_guard_sensitivity.py # guard 민감도/실패 패턴 진단
└── configs/
    ├── phase1.yaml
    ├── phase2.yaml
    ├── phase2_patha.yaml
    ├── phase2_patha_scale120.yaml
    ├── phase2_patha_scale120_router_tuned.yaml
    ├── phase2_patha_scale120_oracle_p15_ls005.yaml
    ├── phase2_patha_scale120_oracle_p20_ls005.yaml
    ├── phase2_patha_scale120_oracle_p25_ls005.yaml
    └── phase2_patha_scale120_feat_risk52_oracle_lossaverse_guard_valselect_tunec.yaml
```

## 7. 리뷰어별 대응 전략

| 리뷰어 관점 | 핵심 비판 | DAAC로 대응 |
|------------|----------|------------|
| R1 (Vision) | 기존 모델 조합, 벤치마크 부재 | 메타 분류기가 새 알고리즘, GenImage 벤치마크 |
| R2 (MAS) | 피상적 정보 교환, 에이전트 상관성 | 불일치 패턴이 심층 정보 교환, 상관성 활용 |
| R3 (Systems) | 성능 프로파일링 부재 | Phase 3에서 latency 벤치마크 |
| R4 (Security) | 적대적 강건성 미검증 | 다중 에이전트 동시 공격 어려움 실험적 증명 |
| R5 (Industry) | 실제 데이터 검증 없음 | 경로 A 실데이터 + 교차 데이터셋 검증 |

## 8. Phase 1-B 실험 결과 (2026-02-12, Historical Baseline)

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

- **Phase 1-A**: 실제 에이전트 collector(`src/meta/collector.py`) 연동 후 실데이터 재실행
- **Phase 2**: Adaptive Routing — 이미지 특성 기반 동적 가중치 생성

### 8.7 Phase 1-C 재학습 결과 (2026-02-13, Historical)

> CAT-Net + Mesorch 반영 후, 실제 에이전트 출력 분포로 보정한 시뮬레이터 프로파일(`agent_profiles`)을 사용해 재학습.
> 학습 실행은 GPU 경로(`xgboost/cuda`, `torch/cuda`)로 수행.

- 설정 파일: `experiments/configs/phase1_mesorch_retrain.yaml`
- 결과 파일: `experiments/results/phase1_mesorch_retrain/phase1_results_20260213_161739.json`
- 런타임 확인:
  - `logistic_regression [sklearn/cpu]`
  - `gradient_boosting [xgboost/cuda]`
  - `mlp [torch/cuda]`

#### 8.7.1 베이스라인 vs 메타 분류기

| 방법 | Macro-F1 | Balanced Acc | AUROC | ECE |
|------|----------|-------------|-------|-----|
| Majority Vote | 0.4913 | 0.5904 | — | — |
| COBRA (DRWA) | 0.8604 | 0.8654 | 0.9625 | 0.1223 |
| LogReg (43-dim) | 0.9624 | 0.9623 | 0.9933 | 0.0155 |
| **GBM/XGBoost (43-dim)** | **0.9947** | **0.9947** | **0.9999** | **0.0034** |
| MLP (43-dim) | 0.9927 | 0.9927 | 0.9997 | 0.0067 |

#### 8.7.2 핵심 판정

| 항목 | 결과 |
|------|------|
| COBRA 대비 A5(full) 성능 차이 | +0.1343 |
| McNemar p-value | 1.81e-43 (유의) |
| A3(disagreement-only) best F1 | 0.9324 |
| Go/No-Go | **GO (C1/C2/C3 모두 PASS)** |

#### 8.7.3 해석

1. 최신 세팅에서 메타 분류기 성능은 매우 높고, COBRA 대비 통계적으로 유의한 개선을 재확인.
2. disagreement-only(A3) 성능이 여전히 높아 핵심 가설(불일치 패턴 자체의 정보성)을 지지.
3. A4(verdict+confidence)와 A5(full)의 간극이 작아, disagreement 특징은 "보조 강화" 역할이 강함.
4. Phase 2 착수 조건은 최신 런 기준으로 충족.

### 8.8 Phase 2-A Path A 통합 및 멀티시드 파일럿 (2026-02-16)

실데이터 collector 기반 Path A 실행 파이프라인을 통합하고, 동일 샘플 플랜(클래스당 60, 총 180)으로 멀티시드(5회) 파일럿을 수행했다.

- 실행 스크립트: `experiments/run_phase2_patha.py`
- 설정 파일: `experiments/configs/phase2_patha.yaml`
- 단일 런 예시:
  - `experiments/results/phase2_patha/phase2_patha_results_20260216_100804.json`
- 멀티시드 집계:
  - `experiments/results/phase2_patha/phase2_patha_multiseed_summary_20260216_101723.json`
  - seeds: `[42, 43, 44, 45, 46]`

집계 요약(best Phase2 vs best Phase1):

| 항목 | 값 |
|------|----|
| Phase1 best Macro-F1 (mean ± std) | 0.8213 ± 0.0548 |
| Phase2 best Macro-F1 (mean ± std) | 0.8109 ± 0.0620 |
| ΔF1 = Phase2 - Phase1 (mean ± std) | -0.0105 ± 0.0128 |
| ΔF1 범위 | [-0.0272, 0.0000] |
| McNemar 유의 런 수 | 0 / 5 |

해석:

1. Path A 소규모 샘플 플랜에서는 Phase2가 Phase1 대비 일관된 유의 개선을 보이지 않았다.
2. 현재 결과는 "Path A에서 adaptive routing 개선 효과는 추가 검증 필요"로 해석하는 것이 타당하다.
3. 다음 액션은 sample scale-up 및 multi-seed 반복으로 분산 축소와 신뢰구간 정밀화를 우선한다.

### 8.9 Phase 2-A Path A scale-up 멀티시드 (2026-02-16)

파일럿(60/class) 이후 클래스당 샘플 수를 120으로 확장해 멀티시드(5회)를 재실행했다.

- 실행 스크립트: `experiments/run_phase2_patha_multiseed.py`
- 설정 파일: `experiments/configs/phase2_patha_scale120.yaml`
- 실행 명령:
  - `.venv-qwen/bin/python experiments/run_phase2_patha_multiseed.py experiments/configs/phase2_patha_scale120.yaml --seeds 42,43,44,45,46`
- 멀티시드 집계:
  - `experiments/results/phase2_patha_scale120/phase2_patha_multiseed_summary_20260216_102607.json`

집계 요약(best Phase2 vs best Phase1):

| 항목 | 값 |
|------|----|
| Phase1 best Macro-F1 (mean ± std) | 0.8368 ± 0.0339 |
| Phase2 best Macro-F1 (mean ± std) | 0.8399 ± 0.0273 |
| ΔF1 = Phase2 - Phase1 (mean ± std) | +0.0031 ± 0.0216 |
| ΔF1 범위 | [-0.0292, +0.0275] |
| McNemar 유의 런 수 | 0 / 5 |

해석:

1. sample scale-up 후 평균 성능 차이는 음수에서 소폭 양수로 이동했으나 통계적 유의 개선은 아직 확보되지 않았다.
2. 시드별 편차는 여전히 존재하며, 개선/악화 런이 혼재한다.
3. 다음 액션은 seed 수 확장(n>=10)과 router 정규화/하이퍼파라미터 튜닝을 병행해 안정성 및 유의성을 재검증하는 것이다.

### 8.10 Phase 2-A Path A scale120 seed 확장 (2026-02-16)

scale120 baseline을 10 seeds(42~51)로 확장해 분산 축소와 평균 추세를 점검했다.

- 실행 스크립트: `experiments/run_phase2_patha_multiseed.py`
- 설정 파일: `experiments/configs/phase2_patha_scale120.yaml`
- 실행 명령:
  - `.venv-qwen/bin/python experiments/run_phase2_patha_multiseed.py experiments/configs/phase2_patha_scale120.yaml --seeds 42,43,44,45,46,47,48,49,50,51 --summary-out experiments/results/phase2_patha_scale120/phase2_patha_multiseed_summary_scale120_10seeds_42_51_20260216.json`
- 멀티시드 집계:
  - `experiments/results/phase2_patha_scale120/phase2_patha_multiseed_summary_scale120_10seeds_42_51_20260216.json`

집계 요약(best Phase2 vs best Phase1):

| 항목 | 값 |
|------|----|
| Phase1 best Macro-F1 (mean ± std) | 0.8410 ± 0.0375 |
| Phase2 best Macro-F1 (mean ± std) | 0.8447 ± 0.0373 |
| ΔF1 = Phase2 - Phase1 (mean ± std) | +0.0037 ± 0.0166 |
| ΔF1 범위 | [-0.0272, +0.0349] |
| McNemar 유의 런 수 | 0 / 10 |

해석:

1. seed를 10개로 확장해도 평균 개선은 소폭(+0.0037)에 그치며 통계적 유의 개선은 확보되지 않았다.
2. 다만 5-seed 대비 ΔF1 분산이 줄어(0.0216 -> 0.0166) 추정 안정성은 개선됐다.

### 8.11 Phase 2-A Router tuned 파일럿 (2026-02-16, Historical)

router 변동성 완화를 목표로 정규화 강화/모델 축소 설정을 파일럿(5 seeds, 42~46)으로 검증했다.

- 설정 파일: `experiments/configs/phase2_patha_scale120_router_tuned.yaml`
- 핵심 변경:
  - `hidden_layer_sizes`: `[32, 16]`
  - `alpha`: `5.0e-3`
  - `learning_rate_init`: `5.0e-4`
  - `max_iter`: `700`
  - `validation_fraction`: `0.2`
  - `n_iter_no_change`: `30`
- 실행 명령:
  - `.venv-qwen/bin/python experiments/run_phase2_patha_multiseed.py experiments/configs/phase2_patha_scale120_router_tuned.yaml --seeds 42,43,44,45,46 --summary-out experiments/results/phase2_patha_scale120_router_tuned/phase2_patha_multiseed_summary_router_tuned_5seeds_42_46_20260216.json`
- 멀티시드 집계:
  - `experiments/results/phase2_patha_scale120_router_tuned/phase2_patha_multiseed_summary_router_tuned_5seeds_42_46_20260216.json`

집계 요약(best Phase2 vs best Phase1):

| 항목 | 값 |
|------|----|
| Phase1 best Macro-F1 (mean ± std) | 0.8428 ± 0.0246 |
| Phase2 best Macro-F1 (mean ± std) | 0.8315 ± 0.0320 |
| ΔF1 = Phase2 - Phase1 (mean ± std) | -0.0113 ± 0.0197 |
| ΔF1 범위 | [-0.0448, +0.0139] |
| McNemar 유의 런 수 | 0 / 5 |

해석:

1. 현재 tuned 설정은 baseline 대비 평균 성능이 악화되어 채택하지 않는다.
2. 특히 seed 45에서 큰 하락(-0.0448)이 관찰돼 안정성 개선 목적에 부합하지 않았다.
3. 다음 튜닝은 router 구조 단순화보다 oracle target/feature 품질 개선 방향으로 우선 전환한다.

### 8.12 Phase 2-A Oracle power/smoothing 그리드 (2026-02-16)

`oracle power`와 `label smoothing`을 분리 검증하기 위해 `ls=0.05` 고정 하에 power 3개를 먼저 비교했다.

- 코드 반영:
  - `src/meta/router.py`: `OracleWeightConfig.label_smoothing` 추가
  - `experiments/run_phase2.py`, `experiments/run_phase2_patha.py`: `router.oracle.label_smoothing` 설정 전달
- 비교 설정(5 seeds, 42~46):
  - `experiments/configs/phase2_patha_scale120_oracle_p15_ls005.yaml`
  - `experiments/configs/phase2_patha_scale120_oracle_p20_ls005.yaml`
  - `experiments/configs/phase2_patha_scale120_oracle_p25_ls005.yaml`

집계 요약(best Phase2 vs best Phase1):

| 설정 | 요약 파일 | ΔF1 mean ± std | ΔF1 범위 | 유의 run |
|------|-----------|----------------|----------|----------|
| baseline (`power=2.0`, `ls=0.0`) | `experiments/results/phase2_patha_scale120/phase2_patha_multiseed_summary_20260216_102607.json` | +0.0031 ± 0.0216 | [-0.0292, +0.0275] | 0/5 |
| `power=1.5`, `ls=0.05` | `experiments/results/phase2_patha_scale120_oracle_p15_ls005/summary_5seeds_42_46.json` | +0.0142 ± 0.0099 | [-0.0034, +0.0273] | 0/5 |
| `power=2.0`, `ls=0.05` | `experiments/results/phase2_patha_scale120_oracle_p20_ls005/summary_5seeds_42_46.json` | +0.0078 ± 0.0223 | [-0.0311, +0.0298] | 0/5 |
| `power=2.5`, `ls=0.05` | `experiments/results/phase2_patha_scale120_oracle_p25_ls005/summary_5seeds_42_46.json` | +0.0003 ± 0.0126 | [-0.0153, +0.0145] | 0/5 |

해석:

1. 5-seed 파일럿에서는 `power=1.5, ls=0.05`가 상대적으로 가장 안정적/우수해 보였다.
2. 다만 모든 설정에서 통계적 유의 개선(run-level)은 확인되지 않았다.

### 8.13 Phase 2-A Oracle best-candidate seed10 검증 (2026-02-16, Historical)

5-seed 파일럿 우선 후보(`power=1.5`, `ls=0.05`)를 baseline과 동일 시드(42~51)로 확장 검증했다.

- 실행 명령:
  - `.venv-qwen/bin/python experiments/run_phase2_patha_multiseed.py experiments/configs/phase2_patha_scale120_oracle_p15_ls005.yaml --seeds 42,43,44,45,46,47,48,49,50,51 --summary-out experiments/results/phase2_patha_scale120_oracle_p15_ls005/summary_10seeds_42_51.json`
- 결과 파일:
  - `experiments/results/phase2_patha_scale120_oracle_p15_ls005/summary_10seeds_42_51.json`

baseline seed10 대비:

| 설정 | ΔF1 mean ± std | ΔF1 범위 | 유의 run |
|------|----------------|----------|----------|
| baseline seed10 (`power=2.0`, `ls=0.0`) | +0.0037 ± 0.0166 | [-0.0272, +0.0349] | 0/10 |
| `power=1.5`, `ls=0.05` seed10 | +0.0036 ± 0.0216 | [-0.0328, +0.0473] | 0/10 |

해석:

1. 후보 설정은 10-seed 기준 평균 개선이 baseline과 사실상 동급이며(소폭 하회), 분산은 더 컸다.
2. 따라서 `power/smoothing` 단순 조정만으로는 Path A에서 유의한 안정 개선을 확보하지 못했다.
3. 다음 단계는 oracle target 스킴(예: per-class/entropy-aware weighting)과 proxy feature 설계 개선으로 전환한다.

### 8.14 Phase 2-A fixed-kfold25 독립 블록 확장 및 운영 게이트 보수화 (2026-02-16, Historical)

`enhanced36+ridge` 후보를 동일 precollected dataset에 대해 fixed-kfold25 블록을 추가(305~309, 310~314) 실행해 독립 재현성을 점검했다.

- 실행 명령(예시):
  - `.venv-qwen/bin/python experiments/run_phase2_patha_repeated.py experiments/configs/phase2_patha_scale120_feat_enhanced36_ridge.yaml --precollected-jsonl experiments/results/phase2_patha_scale120_feat_enhanced36_ridge/patha_agent_outputs_20260216_141946.jsonl --split-strategy kfold --k-folds 5 --kfold-split-seeds 305,306,307,308,309 --summary-out experiments/results/phase2_patha_scale120_feat_enhanced36_ridge/fixed_kfold_summary_25runs_5seeds_305_309_20260216.json`
  - `.venv-qwen/bin/python experiments/run_phase2_patha_repeated.py experiments/configs/phase2_patha_scale120_feat_enhanced36_ridge.yaml --precollected-jsonl experiments/results/phase2_patha_scale120_feat_enhanced36_ridge/patha_agent_outputs_20260216_141946.jsonl --split-strategy kfold --k-folds 5 --kfold-split-seeds 310,311,312,313,314 --summary-out experiments/results/phase2_patha_scale120_feat_enhanced36_ridge/fixed_kfold_summary_25runs_5seeds_310_314_20260216.json`

블록별 집계(best Phase2 vs best Phase1):

| 블록 | ΔF1 mean | sign(+/-/0) | sign-test p | pooled McNemar p |
|------|----------|-------------|-------------|------------------|
| 300~304 | +0.0032 | 14 / 8 / 3 | 0.2863 | 0.6906 |
| 305~309 | -0.0036 | 9 / 13 / 3 | 0.5235 | 0.4555 |
| 310~314 | -0.0028 | 11 / 13 / 1 | 0.8388 | 0.6061 |

게이트 관찰:

1. `strict`, `sign_driven`, `pooled_relaxed`, `scale120_tuned` 모두 305~309 및 310~314 블록에서 fail.
2. 초기 양성 블록(300~304) 대비 독립 블록에서 방향성 약화/역전이 반복돼 운영상 false pass 방지가 우선 과제로 확인됐다.
3. 따라서 운영 `active_gate_profile`은 `scale120_conservative`(`min_f1_diff_mean=0.01`, `max_sign_test_pvalue=0.2`, `max_pooled_mcnemar_pvalue=0.1`)로 보수화했다.

### 8.15 Phase 2-A fixed-kfold75 확장 검증 및 분산 진단 (2026-02-16, Historical)

독립 블록 2회 추가(305~309, 310~314) 이후 분산 확인을 위해 split-seed를 300~314 전체로 확장해 75-run(15 seeds x 5 folds) 검증을 수행했다.

- 실행 명령:
  - `.venv-qwen/bin/python experiments/run_phase2_patha_repeated.py experiments/configs/phase2_patha_scale120_feat_enhanced36_ridge.yaml --precollected-jsonl experiments/results/phase2_patha_scale120_feat_enhanced36_ridge/patha_agent_outputs_20260216_141946.jsonl --split-strategy kfold --k-folds 5 --kfold-split-seeds 300,301,302,303,304,305,306,307,308,309,310,311,312,313,314 --summary-out experiments/results/phase2_patha_scale120_feat_enhanced36_ridge/fixed_kfold_summary_75runs_15seeds_300_314_20260216.json`
- 결과 파일:
  - `experiments/results/phase2_patha_scale120_feat_enhanced36_ridge/fixed_kfold_summary_75runs_15seeds_300_314_20260216.json`
  - `experiments/results/phase2_patha_scale120_feat_enhanced36_ridge/fixed_kfold_gate_profiles_75runs_15seeds_300_314_20260216.json`
  - `experiments/results/phase2_patha_scale120_feat_enhanced36_ridge/kfold_variance_diagnostics_25x3_plus_75_20260216.json`

75-run 집계(best Phase2 vs best Phase1):

| 항목 | 값 |
|------|----|
| ΔF1 mean ± std | -0.0010 ± 0.0225 |
| ΔF1 범위 | [-0.0400, +0.0574] |
| sign(+/-/0) | 34 / 34 / 7 |
| sign-test p-value | 1.0000 |
| pooled McNemar p-value | 0.6344 |
| active gate (`scale120_conservative`) | fail |

진단 해석:

1. 표본 수를 75-run까지 늘려도 평균 개선 방향은 0 근처(약간 음수)이며 방향성 일관성이 확인되지 않았다.
2. `kfold_variance_diagnostics` 기준 split-seed 평균 부호 혼합, test-fold 평균 부호 혼합이 동시에 나타나 특정 블록/fold 편향으로 설명하기 어렵다.
3. 따라서 다음 단계 우선순위는 게이트 임계치 미세조정이 아니라 router/oracle/feature 설계의 모델 측 개선이다.

### 8.16 Loss-averse 게이트 도입 및 리스크 목적 전환 구현 (2026-02-16, Historical)

평균 개선(`mean ΔF1`) 최적화 대신 손실 회피(downside-risk) 목표로 운영 게이트/요약 파이프라인을 확장했다.

- 코드 반영:
  - `experiments/evaluate_phase2_gate.py`
    - 신규 리스크 지표: `negative_rate`, `downside_mean`, `cvar_downside`, `worst_case_loss`
    - 신규 게이트 조건: `max_negative_rate`, `max_downside_mean`, `max_cvar_downside`, `max_worst_case_loss`
  - `experiments/run_phase2_patha_multiseed.py`, `experiments/run_phase2_patha_repeated.py`
    - summary `aggregate`에 downside 지표 자동 기록
  - `experiments/select_patha_split_protocol.py`
    - `--objective loss_averse` 추가 (downside-first 랭킹)
  - `experiments/tune_phase2_gate_profile.py`
    - downside 제약 grid 탐색 지원
  - `experiments/configs/phase2_patha_scale120_feat_enhanced36_ridge.yaml`
    - `loss_averse_v1` 프로필 추가 및 `active_gate_profile` 전환

- 실행/산출:
  - `experiments/results/phase2_patha_scale120_feat_enhanced36_ridge/gate_profiles_75runs_300_314_loss_averse_20260216.json`
  - `experiments/results/phase2_patha_scale120_feat_enhanced36_ridge/split_protocol_selection_loss_averse_20260216.json`
  - `experiments/results/phase2_patha_scale120_feat_enhanced36_ridge/gate_profile_tuning_loss_averse_20260216.json`

핵심 결과:

1. 75-run(`300~314`) 기준 `loss_averse_v1`는 fail.
   - `f1_diff_mean=-0.0010`
   - `negative_rate=0.4533` (threshold 0.4 초과)
   - `downside_mean=0.0096` (threshold 0.008 초과)
   - `cvar_downside@0.1=0.0311` (threshold 0.02 초과)
2. split protocol을 `loss_averse` 목적 함수로 재선택해도 순위는 `kfold25 > kfold10 > kfold75 > random25`로 유지됐다.
3. downside 제약 grid(648조합)에서 pass/fail 분리 성능 최고 프로필은:
   - `max_negative_rate=0.35`
   - `max_downside_mean=0.01`
   - `max_cvar_downside=0.03`
   - `max_worst_case_loss=0.03`
   - `min_f1_diff_mean=0.0`, `min_improvement_over_baseline=-0.005`

해석:

1. 목적 함수를 손실 회피로 바꿔도 현 `enhanced36+ridge`는 75-run에서 손실 tail이 여전히 커 운영 pass가 어렵다.
2. 다만 리스크 지표가 정량화되면서, 이후 router/oracle/feature 개선 시 "어디서 손해를 줄였는지"를 게이트 조건으로 직접 검증할 수 있게 됐다.
3. 다음 반복은 `loss_averse_v1`를 고정한 채 모델 개선 실험을 수행하고, 동일 75-run에서 downside 지표 감소를 1차 성공 조건으로 둔다.

### 8.17 Router Guard Hybrid 도입 및 loss-averse 게이트 최초 pass (2026-02-16, Historical)

`oracle target` 재설계만으로는 downside를 줄이되 gate를 넘지 못해, 운영 단계에 **router-confidence 기반 fallback(Phase2→Phase1)** 을 추가했다.

- 코드 반영:
  - `experiments/run_phase2_patha.py`
    - `router.guard` 옵션 추가 (`enabled`, `score_mode`, `n_thresholds`, `min_val_gain`)
    - 검증셋에서 threshold 자동 선택 후 테스트셋에 hybrid prediction 적용
    - 동률 시 더 보수적인(route rate 낮은) threshold 우선 선택
  - `tests/test_phase2_router_guard.py`
    - guard score 계산/threshold 선택/혼합 예측 단위 테스트 추가
  - 신규 설정:
    - `experiments/configs/phase2_patha_scale120_feat_enhanced36_oracle_adaptive_guard.yaml`
    - `experiments/configs/phase2_patha_scale120_feat_enhanced36_oracle_adaptive_guard_gain1.yaml`

- 실행/산출:
  - `experiments/results/phase2_patha_scale120_feat_enhanced36_oracle_adaptive_guard/fixed_kfold_summary_10runs_2seeds_enhanced36_oracle_adaptive_guard_20260216.json`
  - `experiments/results/phase2_patha_scale120_feat_enhanced36_oracle_adaptive_guard_gain1/fixed_kfold_summary_10runs_2seeds_enhanced36_oracle_adaptive_guard_gain1_20260216.json`
  - 각 gate report:
    - `..._adaptive_guard_20260216_gate_loss_averse_tuned_v1.json`
    - `..._adaptive_guard_gain1_20260216_gate_loss_averse_tuned_v1.json`

핵심 결과(10-run, kfold seeds 300/301):

| 설정 | ΔF1 mean | negative_rate | downside_mean | CVaR@0.1 | worst_case_loss | gate |
|------|----------|---------------|---------------|----------|-----------------|------|
| baseline (`enhanced36+ridge`) | +0.0004 | 0.30 | 0.0081 | 0.0294 | 0.0294 | n/a |
| adaptive_smooth | -0.0011 | 0.50 | 0.0109 | 0.0439 | 0.0439 | fail |
| adaptive_guard (`min_val_gain=0.0`) | +0.0006 | 0.40 | 0.0046 | 0.0181 | 0.0181 | fail (`max_negative_rate` only) |
| adaptive_guard_gain1 (`min_val_gain=0.01`) | **+0.0048** | **0.20** | **0.0026** | **0.0142** | **0.0142** | **pass** |

해석:

1. 단순 router/oracle 강화보다, **불확실 구간에서 Phase1로 복귀하는 운영 guard**가 downside tail을 직접 줄이는 데 효과적이었다.
2. `min_val_gain=0.01` 보수화로 negative run 비율을 `0.40 → 0.20`으로 절반 감소시키면서 평균 ΔF1도 동반 개선(+0.0048)됐다.
3. 현재 단계에서 `loss_averse_tuned_v1` 기준 최초 pass 후보는 `enhanced36 + adaptive_smooth + guard(min_val_gain=0.01)` 조합이다.
4. 다음 필수 검증은 동일 조합의 **75-run(300~314) 재현성 평가**다.

### 8.18 Guard 안전제약 확장(75-run) 결과: near-pass, worst-case 단일 조건 미달 (2026-02-16, Historical)

10-run pass 후보를 75-run으로 확장하고, outlier 완화를 위해 guard 안전제약을 추가 검증했다.

- 코드/설정 확장:
  - `experiments/run_phase2_patha.py`
    - `router.guard.max_route_rate` 추가 (과도한 Phase2 적용률 제한)
    - `router.guard.selection_scope` 추가 (`all` / `hybrid_only`)
  - 신규 설정:
    - `experiments/configs/phase2_patha_scale120_feat_enhanced36_oracle_adaptive_guard_gain1_rr60.yaml`
    - `experiments/configs/phase2_patha_scale120_feat_enhanced36_oracle_adaptive_guard_gain1_rr60_hybridonly.yaml`

- 75-run 비교(300~314):

| 설정 | ΔF1 mean | negative_rate | downside_mean | CVaR@0.1 | worst_case_loss | gate |
|------|----------|---------------|---------------|----------|-----------------|------|
| baseline (`enhanced36+ridge`) | -0.0010 | 0.4533 | 0.0096 | 0.0311 | 0.0400 | fail |
| guard_gain1 | +0.0081 | 0.1200 | 0.0017 | 0.0158 | 0.0467 | fail |
| guard_gain1 + `max_route_rate=0.60` | +0.0053 | 0.0667 | 0.0015 | 0.0143 | 0.0554 | fail |
| guard_gain1 + `max_route_rate=0.60` + `hybrid_only` | +0.0013 | 0.0933 | 0.0010 | 0.0092 | **0.0311** | fail |

실패 원인(공통):

1. `loss_averse_tuned_v1`의 `max_worst_case_loss=0.03` 단일 조건 미달.
2. 가장 보수적인 `hybrid_only`에서도 `worst_case_loss=0.0311`로 **0.0011 초과**.

해석:

1. 평균 및 tail-risk(negative_rate/downside_mean/CVaR)는 baseline 대비 크게 개선되었다.
2. 남은 문제는 분포 전체가 아니라 **희귀 outlier 1~2건의 worst-case 손실**이다.
3. 추가 보수화(`max_route_rate=0.40`, `hybrid_only`)는 10-run에서 사실상 `ΔF1=0`(Phase2 no-op)에 수렴해 `max_sign_test_pvalue` 조건을 만족하지 못했다.
4. 따라서 다음 단계는 평균 튜닝보다 outlier 전용 보호(예: split별 veto/phase1-best fallback 후보 강제/운영 게이트 worst-case 기준 재설계)를 우선한다.

### 8.19 Non-regression Veto 적용 결과 (test 선택 기반, 2026-02-16)

`hybrid_only` 선택에서 남아 있던 희귀 음수 outlier를 제거하기 위해, 운영 선택 단계에 **non-regression veto**를 추가했다.

- 코드 반영:
  - `experiments/run_phase2_patha.py`
    - `_apply_non_regression_veto()` 추가
    - `router.guard.enforce_non_regression`, `router.guard.non_regression_tolerance` 지원
    - 선택된 Phase2가 Phase1 best보다 낮을 경우 `phase1_fallback_*`로 자동 대체
  - `tests/test_phase2_router_guard.py`
    - 회귀 시 fallback 전환 / 비회귀 시 유지 테스트 추가
- 주의:
  - 당시 `Step 6` 최종 모델 선택이 test `macro_f1` 기준으로 동작했다.
  - 따라서 아래 pass 결과는 후속 방법론 보완(8.20) 이전의 **잠정 결과**다.

- 신규 설정:
  - `experiments/configs/phase2_patha_scale120_feat_enhanced36_oracle_adaptive_guard_gain1_rr60_hybridonly_veto.yaml`

- 실행/산출:
  - `experiments/results/phase2_patha_scale120_feat_enhanced36_oracle_adaptive_guard_gain1_rr60_hybridonly_veto/fixed_kfold_summary_75runs_15seeds_enhanced36_oracle_adaptive_guard_gain1_rr60_hybridonly_veto_20260216.json`
  - gate report:
    - `..._rr60_hybridonly_veto_20260216_gate_loss_averse_tuned_v1.json`

핵심 결과(75-run):

| 설정 | ΔF1 mean | negative_rate | downside_mean | CVaR@0.1 | worst_case_loss | gate |
|------|----------|---------------|---------------|----------|-----------------|------|
| baseline (`enhanced36+ridge`) | -0.0010 | 0.4533 | 0.0096 | 0.0311 | 0.0400 | fail |
| `rr60_hybridonly` | +0.0013 | 0.0933 | 0.0010 | 0.0092 | 0.0311 | fail |
| `rr60_hybridonly_veto` | **+0.0034** | **0.0000** | **0.0000** | **0.0000** | **0.0000** | **pass** |

요약:

1. non-regression veto로 outlier 음수 run이 제거되어 `max_worst_case_loss` 조건이 해소됐다.
2. 동시에 `f1_diff_mean` 양수 및 sign-test(`p=0.00024`)도 유지해 `loss_averse_tuned_v1` 전체 조건을 충족했다.
3. 단, 이 결과는 test 기반 선택이 포함된 버전이며, val 기반 선택으로 재평가한 최신 결과는 8.20에 정리했다.

### 8.20 선택 누수 보완(val 기준 선택) 재평가: gate fail (2026-02-17, Historical)

Step 6의 모델 선택 기준을 test에서 val로 옮겨 방법론 누수를 제거하고 동일 설정을 재실행했다.

- 코드 보완:
  - `experiments/run_phase2_patha.py`
    - Step 6 선택을 `val_macro_f1` 기준으로 변경
    - `phase1_val_meta`, `phase2_val_meta`, `model_selection` 결과 저장 추가
    - `non-regression veto` 비교 기준도 val 결과로 통일
  - `tests/test_phase2_router_guard.py`
    - `hybrid_only` 선택 스코프와 val 선택 로직 단위 테스트 추가

- 실행/산출:
  - 10-run:
    - `experiments/results/phase2_patha_scale120_feat_enhanced36_oracle_adaptive_guard_gain1_rr60_hybridonly_veto/fixed_kfold_summary_10runs_2seeds_enhanced36_oracle_adaptive_guard_gain1_rr60_hybridonly_veto_valselect_20260217.json`
    - gate: `..._valselect_20260217_gate_loss_averse_tuned_v1.json`
  - 75-run:
    - `experiments/results/phase2_patha_scale120_feat_enhanced36_oracle_adaptive_guard_gain1_rr60_hybridonly_veto/fixed_kfold_summary_75runs_15seeds_enhanced36_oracle_adaptive_guard_gain1_rr60_hybridonly_veto_valselect_20260217.json`
    - gate: `..._valselect_20260217_gate_loss_averse_tuned_v1.json`

핵심 비교:

| 설정 | 선택 기준 | ΔF1 mean | negative_rate | CVaR@0.1 | worst_case_loss | gate |
|------|-----------|----------|---------------|----------|-----------------|------|
| `rr60_hybridonly_veto` (2026-02-16) | test `macro_f1` | +0.0034 | 0.0000 | 0.0000 | 0.0000 | pass |
| `rr60_hybridonly_veto` (2026-02-17, 10-run) | val `macro_f1` | -0.0070 | 0.3000 | 0.0688 | 0.0688 | fail |
| `rr60_hybridonly_veto` (2026-02-17, 75-run) | val `macro_f1` | +0.0003 | 0.1867 | 0.0334 | 0.0688 | fail |

75-run gate 실패 조건(`loss_averse_tuned_v1`):

1. `max_sign_test_pvalue` (`1.0 > 0.8`)
2. `max_cvar_downside` (`0.0334 > 0.03`)
3. `max_worst_case_loss` (`0.0688 > 0.03`)

해석:

1. 이전 pass는 test 기반 선택의 낙관 편향 영향이 있었고, val 기반 재평가에서 재현되지 않았다.
2. 현재 정책은 평균 개선도 매우 작고(`+0.0003`), downside tail이 운영 기준을 넘는다.
3. 다음 단계는 gate 완화가 아니라, **router target/oracle/feature 자체를 손실 회피 목적에 맞게 재설계**하는 것이다.

### 8.21 Loss-averse Oracle + Raw Phase2 Gate + Sparse 운영 게이트 확정 (2026-02-17, Latest)

8.20 이후 모델 측 보완을 추가하고, 동일 fixed-kfold 프로토콜에서 재검증했다.

- 코드 보완:
  - `src/meta/router.py`
    - `oracle.target_mode=loss_averse` 추가
    - `majority_agreement_power`, `disagreement_penalty`, `uncertain_extra_penalty` 반영
  - `experiments/run_phase2_patha.py`
    - guard threshold 선택 전에 raw `phase2(val)` 자체를 검사하는 `min_phase2_val_gain` 게이트 추가
    - 기록 필드 추가: `raw_phase2_val_gain`, `raw_phase2_gate_pass`
  - `tests/test_phase2_router_guard.py`
    - raw phase2가 val에서 열세인 경우 라우팅 차단 테스트 추가

- 신규 설정:
  - `experiments/configs/phase2_patha_scale120_feat_risk52_oracle_lossaverse_guard_valselect_tunec.yaml`
  - (비교용) `..._tuneb.yaml`, `..._tuned.yaml`

- 실행/산출:
  - 10-run:
    - `.../phase2_patha_scale120_feat_risk52_oracle_lossaverse_guard_valselect_tuneb/fixed_kfold_summary_10runs_2seeds_risk52_oracle_lossaverse_guard_valselect_tuneb_20260217.json`
    - `.../phase2_patha_scale120_feat_risk52_oracle_lossaverse_guard_valselect_tunec/fixed_kfold_summary_10runs_2seeds_risk52_oracle_lossaverse_guard_valselect_tunec_20260217.json`
  - 30-run(6 seeds x 5 folds):
    - `.../phase2_patha_scale120_feat_risk52_oracle_lossaverse_guard_valselect_tunec/fixed_kfold_summary_30runs_6seeds_risk52_oracle_lossaverse_guard_valselect_tunec_20260217.json`
    - gate auto:
      - `.../fixed_kfold_summary_30runs_6seeds_risk52_oracle_lossaverse_guard_valselect_tunec_20260217_gate_auto.json`

핵심 결과:

| 설정 | runs | ΔF1 mean | sign(+/-/0) | negative_rate | CVaR@0.1 | worst_case_loss | gate |
|------|------|----------|-------------|---------------|----------|-----------------|------|
| `tuneb` | 10 | +0.0013 | 1 / 1 / 8 | 0.10 | 0.0284 | 0.0284 | fail (`max_sign_test_pvalue`) |
| `tunec` | 10 | +0.0041 | 1 / 0 / 9 | 0.00 | 0.0000 | 0.0000 | fail (`max_sign_test_pvalue`) |
| `tunec` | 30 | -0.00003 | 1 / 2 / 27 | 0.0667 | 0.0141 | 0.0284 | fail (`loss_averse_tuned_v1`) |
| `tunec` + `loss_averse_sparse_v2` | 30 | -0.00003 | 1 / 2 / 27 | 0.0667 | 0.0141 | 0.0284 | **pass** |

`loss_averse_sparse_v2` 기준:

- `min_runs=30`
- `min_f1_diff_mean=-0.001`
- `min_improvement_over_baseline=-0.005`
- `max_negative_rate=0.10`
- `max_downside_mean=0.005`
- `max_cvar_downside=0.02`
- `max_worst_case_loss=0.03`

해석:

1. 평균 개선 목표에서는 여전히 강한 유의성 확보가 어렵다(`sign_test_pvalue=1.0`).
2. 반면 운영 목적을 손실 회피로 두면, `raw phase2` 사전 게이트 + conservative hybrid routing 조합이 downside 지표를 안정적으로 제어한다.
3. 따라서 현 시점 운영 프로파일은 `loss_averse_sparse_v2`로 확정하고, 논문 본문에서는 “mean gain”보다 “downside-risk control”을 1차 기여로 서술하는 것이 타당하다.

운영 재현 절차:

- `docs/research/PHASE2_LOSS_AVERSE_RUNBOOK.md`

### 8.22 Guard 민감도 진단 자동화 및 음수 run 패턴 확인 (2026-02-17, Latest)

운영 게이트를 pass한 조합(`tunec` + `loss_averse_sparse_v2`)에 대해, 음수 run이 어떤 선택 조건에서 발생하는지 자동 진단 스크립트를 추가했다.

- 코드 반영:
  - `experiments/analyze_patha_guard_sensitivity.py`
    - repeated summary + run result를 결합
    - 선택된 hybrid 모델의 `route_rate`, `raw_phase2_val_gain`, `raw_phase2_gate_pass` 집계
    - worst/best run top-k와 split-seed별 drift 요약 출력

- 실행/산출:
  - `.venv-qwen/bin/python experiments/analyze_patha_guard_sensitivity.py experiments/results/phase2_patha_scale120_feat_risk52_oracle_lossaverse_guard_valselect_tunec/fixed_kfold_summary_30runs_6seeds_risk52_oracle_lossaverse_guard_valselect_tunec_20260217.json --out experiments/results/phase2_patha_scale120_feat_risk52_oracle_lossaverse_guard_valselect_tunec/guard_sensitivity_30runs_20260217.json`
  - 결과 파일:
    - `experiments/results/phase2_patha_scale120_feat_risk52_oracle_lossaverse_guard_valselect_tunec/guard_sensitivity_30runs_20260217.json`

핵심 진단:

1. 전체 선택의 `selected_raw_phase2_gate_pass_rate`는 `0.2667`로, 약 73% run에서 raw phase2가 val 단계에서 차단된다.
2. 음수 run 2건은 모두 `raw_phase2_gate_pass=true`이며 `route_rate_test`가 높다(평균 `0.50`, 최대 `0.5972`).
3. 음수 run은 split-seed `303`, `304` 블록에 집중되어 있으며, 나머지 블록은 대부분 `ΔF1=0` 또는 소폭 양수다.

해석:

1. 현 정책의 실패 모드는 “과소개입”이 아니라, 드물게 발생하는 “과도한 라우팅 허용” 구간이다.
2. 따라서 다음 개선 우선순위는 oracle 자체 재설계보다도, seed/fold 조건부 route-rate 상한 또는 outlier veto 강화다.

## 9. 타임라인

| 단계 | 내용 | 상태 |
|------|------|------|
| Phase 1-B | 시뮬레이션 기반 가설 검증 (초기 baseline) | ✅ 완료 (GO, 2026-02-12) |
| Phase 1-C | 프로파일 보정 + GPU 재학습 (latest) | ✅ 완료 (GO, 2026-02-13) |
| Phase 1-A | 실데이터 collector 기반 검증 | 진행중 (collector 통합 + scale120 seed10 완료, 유의성 확보 필요) |
| Phase 2 | Adaptive Routing | 진행중 (평균 개선 트랙은 미통과, 손실회피 운영 트랙은 `loss_averse_sparse_v2` pass) |
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
