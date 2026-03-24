# Error-Weighted Complementary Training (EWCT) 실험 보고서

> **작성일**: 2026-03-23
> **최종 정리**: 2026-03-24
> **프로젝트**: MAIFS — SHIELD 후속 연구 (C6: Heterogeneous Expert Meta-Aggregation)
> **목적**: MNV2(3-class 일반 모델)의 오분류를 SpecM(2-class 전문가 모델)이 교정하는 통합 프레임워크 구축
>
> **빠른 요약**: EWCT 자체의 clean한 기여는 `w_max` 제거와 `confident-but-wrong` 정량화다. HEMA-XGBoost는 in-domain에선 강했지만, 이후 **LOO-CD + full-coverage 공정 재평가(action-gate / ICWMV-veto)** 에서도 learned scalar fusion은 아직 ICWMV를 안정적으로 넘지 못했다. 현재 paper-ready 결론은 **ICWMV를 메인 고정 합의기로 두고, EWCT는 교정률 확장용 supporting study로 해석**하는 것이다.
>
> **상태**: Phase 1~4 + LOO-CD + full-coverage HEMA repair 실험 완료
> **읽는 법**: 본 문서는 실험 chronology를 보존한다. 중간 장의 in-domain HEMA 우위는 역사적 결과이며, 최신 해석은 13장 이후와 14장 결론을 기준으로 본다.

---

## 1. 배경 — 왜 이 실험이 필요했나?

### 1.1 문제 정의

우리 시스템은 두 개의 모델을 사용합니다:

| 모델 | 분류 클래스 | 역할 |
|------|-----------|------|
| **MobileNetV2 (MNV2)** | authentic / manipulated / ai_generated (3-class) | 이미지 종류 전체 판별 |
| **Specialist-M (SpecM)** | authentic / manipulated (2-class) | 조작(manipulation) 탐지 전문 |

MNV2는 평균 95.8%의 정확도를 보이지만, **오분류할 때 문제가 심각**합니다:

```
MNV2 오분류 케이스 분석 (4개 데이터셋, 총 162개 오분류)
─────────────────────────────────────────────────────
데이터셋   │ 오분류 수 │ mean confidence │ >0.9 비율
─────────────────────────────────────────────────────
base      │    84개   │    0.770        │  31.0%
dsC       │    19개   │    0.702        │  10.5%
opensdi   │    34개   │    0.722        │  17.6%
aigenproxy│    23개   │    0.741        │  34.8%
─────────────────────────────────────────────────────
```

> **핵심 발견**: MNV2는 틀릴 때도 **높은 confidence(평균 0.77)**를 가집니다.
> 이것을 "confident-but-wrong" 패턴이라고 합니다.
> 이 때문에 단순히 "확신이 낮을 때만 SpecM에게 물어보는" 방식은 효과가 없습니다.

### 1.2 기존 방식(HEMA PoC)의 한계

이전에 진행한 HEMA PoC(SpecM-v4 사용)의 오분류 교정률:

```
HEMA PoC 오분류 교정률
  base:        50.0%
  dsC:         63.0%
  opensdi:     34.8%
  aigenproxy:  34.3%
  ─────────────────
  평균:        45.5%
```

**질문**: SpecM을 훈련할 때 "MNV2가 틀리는 어려운 샘플에 더 집중하도록" 가르치면 교정률이 올라갈까?

---

## 2. 핵심 아이디어 — Error-Weighted Complementary Training

### 2.1 수식

$$L = \frac{1}{N}\sum_{i=1}^{N} w(x_i) \cdot \text{CE}(\text{SpecM}(x_i),\ y_i)$$

$$w(x) = \left(1 - P_{\text{MNV2}}(y_{\text{true}} \mid x)\right)^{\gamma}$$

**직관적 설명**:

```
MNV2가 정답을 잘 맞히는 샘플 → P_MNV2(y_true) 높음 → w(x) 낮음 → SpecM이 덜 공부
MNV2가 틀리는 어려운 샘플   → P_MNV2(y_true) 낮음 → w(x) 높음 → SpecM이 더 공부
```

이것은 AdaBoost(1996), Focal Loss(2017) 등의 계보를 잇는 **"이종 모델 간 Boosting"** 입니다. 차이점은 같은 모델이 아닌 **서로 다른 아키텍처의 모델이 협력**한다는 것입니다.

### 2.2 학습 설계

```
┌─────────────────────────────────────────────────────────────┐
│  Stage 1: MNV2 완전 동결 (학습하지 않음)                       │
│  Stage 2: SpecM만 학습 (w(x)로 가중된 Cross-Entropy)          │
│                                                             │
│  → Gradient Detach 자동 만족 (2단계 분리)                     │
│    이유: w(x)가 MNV2 gradient를 역전파받지 않아야              │
│          "가중치가 스스로 쉬워지는" 문제를 방지                 │
└─────────────────────────────────────────────────────────────┘
```

**ai_generated 처리 (Dustbin 방식)**:
- SpecM은 2-class 모델이라 ai_generated를 직접 판별 불가
- 학습 시 ai_generated 이미지를 manipulated로 병합 ("dustbin class")
- MNV2가 3-class, SpecM이 2-class인 **비대칭 출력 공간** 문제 해결

**Confidence 정의 (정답확률 방식)**:
- `c(x) = P(y_true|x)` — 실제 정답 클래스의 확률
- MSP(max softmax probability) 대신 사용하는 이유:

```
예시: MNV2가 "authentic(0.9)"로 예측했지만 실제는 "manipulated"인 샘플

MSP 방식:  w = 1 - 0.9 = 0.1  → "MNV2가 확신하므로 SpecM이 무시"  ← 잘못됨!
정답확률:  w = 1 - 0.05 = 0.95 → "MNV2가 틀리므로 SpecM이 집중"   ← 올바름!
```

---

## 3. 실험 설계 개요

```
Phase 1  MNV2 분석       MNV2의 오분류 패턴 파악 + Temperature Scaling
   ↓
Phase 2a γ Ablation      γ=1 vs γ=2 vs γ=3 비교
   ↓
Phase 2b w_max Ablation  클리핑 효과 검증
   ↓
Phase 2c TS Ablation     Temperature Scaling 효과 검증
   ↓
Phase 3  Fuser 비교      6가지 결합 방법 비교
   ↓
Phase 4  3-layer 평가    F1 / 교정률 / 캘리브레이션 종합
```

**데이터셋 구성**:

| 데이터셋 | 샘플 수 | 특징 |
|---------|--------|------|
| base | 1,000 | CASIA2 (학습 도메인, auth/manip 균등) |
| dsC | 600 | CASIA2 확장 + 다양한 조작 유형 |
| opensdi | 600 | OOD(Out-of-Distribution), 도메인 외 |
| aigenproxy | 600 | AI생성 이미지 포함 실전 시나리오 |

**학습 데이터**:
- Authentic: CASIA2 7,491장 + GenImage_nature 3,000장
- Manipulated: CASIA2 5,123장 + IMD2020 1,710장
- Dustbin(AI): BigGAN 2,000장
- **총 19,324장** → 훈련 16,426 / 검증 2,898

---

## 4. Phase 1: MNV2 분석

### 4.1 Temperature Scaling 캘리브레이션

MNV2의 confidence가 실제 정확도를 잘 반영하는지 확인:

```
Temperature T = 1.2263 (MNV2가 약간 과신하고 있음)

직관: T > 1.0 → softmax 분포를 더 "납작하게" 만들어 과신 교정
```

### 4.2 w(x) 가중치 분포

γ값에 따른 훈련 샘플의 가중치 분포:

```
γ=1: mean=0.318, median=0.177, >0.3: 38.9%
     ▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░  (완만한 분포)

γ=2: mean=0.200, median=0.031, >0.3: 25.0%
     ▓▓▓░░░░░░░░░░░░░░░░░░  (어려운 샘플에 집중)

γ=3: mean=0.153, median=0.006, >0.3: 19.3%
     ▓░░░░░░░░░░░░░░░░░░░░  (극단적 집중)
```

---

## 5. Phase 2a: γ Ablation

### 5.1 학습 곡선 비교

```
val manip_F1 학습 곡선 (20 에포크)

  0.85 ┤
       │  ●─●
  0.83 ┤     ●─────────────────── γ=1 (안정)
       │
  0.78 ┤  ■
       │    ■
  0.75 ┤      ■─■─■─■─■─■─■─■─■  γ=2 (급락 후 수평)
       │
  0.76 ┤  ▲
  0.69 ┤    ▲─▲─▲─▲─▲─▲─▲─▲─▲─▲  γ=3 (더 큰 급락)
       │
       └──────────────────────────
       E1   E5   E10   E15   E20
```

| γ | w(x) median | val best F1 | 안정성 |
|---|-------------|-------------|--------|
| **γ=1 ★** | 0.177 | **0.8385** (E2) | ✅ 안정 |
| γ=2 | 0.031 | 0.7727 (E1) | ❌ 급락 |
| γ=3 | 0.006 | 0.7578 (E1) | ❌ 급락 |

### 5.2 eval 4개 데이터셋 결과

```
manip_F1 (auth+manip 서브셋, 이 값이 높을수록 조작 이미지를 잘 탐지)

                  base    dsC    opensdi  aigenproxy  평균
──────────────────────────────────────────────────────────
SpecM-v4 (기준)   0.8698  0.8659  0.6763   0.9655    0.8444  ← 표준 학습
──────────────────────────────────────────────────────────
EWCT γ=1 ★       0.8015  0.8250  0.5870   0.9600    0.7934  (-0.051)
EWCT γ=2          0.8122  0.8139  0.4781   0.9472    0.7628  (-0.082)
EWCT γ=3          0.8027  0.7974  0.4720   0.9333    0.7514  (-0.093)
──────────────────────────────────────────────────────────
```

> **해석**: EWCT는 SpecM 단독 manip_F1을 낮출 수 있지만, 오분류 교정률이 대폭 상승함.
> SpecM-Comp는 "단독 분류기"가 아닌 "HEMA 메타 분류기의 입력 특징"으로 설계됨.
> HEMA 결합 시 v4보다 높은 최종 macro-F1 달성(Phase 3 참조).

```
MNV2 오분류 교정률

                  base    dsC    opensdi  aigenproxy  평균
──────────────────────────────────────────────────────────
SpecM-v4 (기준)  79.3%  64.7%   33.3%    66.7%     61.0%
──────────────────────────────────────────────────────────
EWCT γ=1 ★      80.5%  64.7%   44.4%    88.9%     69.6%  (+8.6%p) ★
EWCT γ=2         81.7%  70.6%   44.4%    88.9%     71.4%  (+10.4%p)
EWCT γ=3         80.5%  58.8%   44.4%    83.3%     66.8%  (+5.8%p)
──────────────────────────────────────────────────────────
기존 HEMA PoC    50.0%  63.0%   34.8%    34.3%     45.5%  ← 이전 시스템
```

**결론**: γ=1이 val 안정성과 eval F1 모두에서 최적.
γ↑로 갈수록 median이 극단화되어(0.177→0.031→0.006) 모델이 쉬운 샘플을 거의 무시하고 어려운 샘플에만 집착 → 과학습(overfitting).

---

## 6. Phase 2b: w_max Ablation

### 6.1 이론적 분석

w_max는 노이즈 레이블 방지를 위한 가중치 상한선입니다.

```
w(x) = (1 - c(x))^γ

γ=1이면: w(x) = 1 - c(x)
c(x) ∈ [0, 1] 이므로 w(x) ∈ [0, 1]

→ 이론적으로 w_max > 1은 클리핑 효과가 전혀 없음
```

### 6.2 실험으로 확인

```
w_max=1.0  vs  w_max=10.0 (γ=1 고정)

        base    dsC    opensdi  aigenproxy
w_max=1  0.8015  0.8250  0.5870  0.9600
w_max=10 0.8015  0.8250  0.5870  0.9600
          ↑──── 완전 동일 ────↑
```

**결론**: 이론 예측 = 실험 결과. γ=1 조건에서 w_max는 무의미.
→ 논문 ablation에서 "γ=1의 이론적 특성"으로 명시.

### 6.3 γ=2에서도 동일 (확인 완료)

γ=2의 경우도 w(x) = (1-c(x))² ∈ [0,1]이므로 이론적으로 w_max>1은 효과 없음:

```
γ=2, w_max=5 vs w_max=10 (실험 비교)

        base    dsC    opensdi  aigenproxy  avg
─────────────────────────────────────────────────
w_max=5  0.8122  0.8139  0.4781  0.9472   0.7628
w_max=10 0.8122  0.8139  0.4781  0.9472   0.7628
          ↑──────── 완전 동일 (비트 단위까지) ────↑
```

> **이론 완전 검증**: γ값에 무관하게 w(x) = (1-c(x))^γ ∈ [0,1]이므로,
> **모든 γ에서 w_max > 1.0은 하이퍼파라미터가 아님** — 제거 가능.

---

## 7. Phase 2c: Temperature Scaling Ablation

### 7.1 TS 효과

```
                base    dsC    opensdi  aigenproxy  평균
────────────────────────────────────────────────────────
γ=1 + TS  ★    0.8015  0.8250  0.5870   0.9600    0.7934
γ=1 no-TS       0.7095  0.8253  0.5730   0.9451    0.7632
────────────────────────────────────────────────────────
차이 (TS효과)  +0.092  +0.000  +0.014   +0.015   +0.030
```

**MNV2 오분류 교정률**:

```
                base    dsC    opensdi  aigenproxy  평균
────────────────────────────────────────────────────────
γ=1 + TS  ★   80.5%  64.7%   44.4%    88.9%     69.6%
γ=1 no-TS      87.8%  76.5%   55.6%    88.9%     77.2%
```

**흥미로운 발견**: no-TS가 교정률은 더 높지만 F1은 낮음.
TS를 제거하면 w(x)가 더 극단화 → 어려운 샘플에 더 집중 → 교정률 상승하지만 전반적 F1 저하.
논문에서는 **TS 사용을 권장** (캘리브레이션 + F1 균형).

---

## 8. Phase 3: Fuser 비교

### 8.1 6가지 결합 방법

```
┌─────────────────────────────────────────────────────────┐
│  MNV2 + SpecM-Comp를 어떻게 결합할 것인가?               │
│                                                         │
│  1. MNV2 단독          → 기준선 (비교용)                 │
│  2. SpecM 단독          → 오분류 교정 특화 확인용          │
│  3. ICWMV              → confidence 가중 투표             │
│  4. Cascade (τ=0.6)    → MNV2 불확실 → SpecM 개입         │
│  5. HEMA-XGBoost ★     → 13차원 특징 학습 기반 결합        │
│  6. FoE-MLP            → ICLR 2024 방식 (5차원 단순 연결)  │
└─────────────────────────────────────────────────────────┘
```

**HEMA의 13차원 특징 공간**:

```
MNV2 출력 (6차원):
  P(authentic), P(manipulated), P(ai_generated)   ← 각 클래스 확률
  MSP(최대 확률), margin(1위-2위 차이), entropy    ← 불확실성

SpecM 출력 (4차원):
  P(manip), P(auth)                                ← 이진 확률
  MSP, entropy                                     ← 불확실성

Cross-modal (3차원):
  |P_auth(MNV2) - P_auth(SpecM)|                  ← 두 모델 불일치
  P_aigen(MNV2) × P_auth(SpecM)                   ← AI생성 신호
  P_manip(MNV2) × P_manip(SpecM)                  ← 조작 동의 신호
```

### 8.2 결과 비교

```
avg macro-F1 (4개 데이터셋 평균)

MNV2 단독     ████████████████████████████████████  0.9503
ICWMV         ██████████████████████████████████████ 0.9546
Cascade(τ)   ████████████████████████████████████▌  0.9537
HEMA-XGB ★   ████████████████████████████████████████ 0.9617  ← 최고
FoE-MLP       ███████████████████████████████████    0.9450
SpecM 단독    ██████████████████████████████         0.7933

avg MNV2 오분류 교정률

MNV2 단독     ░░░░░░░░░░░░░░░░░░░░  0.000  (교정 없음)
ICWMV         ████████░░░░░░░░░░░░  0.433
Cascade(τ)   ███░░░░░░░░░░░░░░░░░  0.186
HEMA-XGB ★   ████████████░░░░░░░░  0.598  ← F1과 교정률 균형 최적
FoE-MLP       ██████░░░░░░░░░░░░░░  0.326
SpecM 단독    █████████████░░░░░░░  0.696  (교정 최고, F1 낮음)
```

### 8.3 데이터셋별 상세

```
macro-F1 by dataset

          base    dsC     opensdi  aigenproxy
────────────────────────────────────────────
MNV2      0.917   0.970   0.949    0.966
ICWMV     0.937   0.973   0.923    0.985
Cascade   0.924   0.973   0.944    0.974
HEMA ★    0.919   0.983   0.967    0.978  ← opensdi에서 큰 개선
FoE-MLP   0.903   0.944   0.955    0.978
────────────────────────────────────────────

  HEMA의 opensdi(OOD) 0.967: MNV2 0.949 대비 +1.8%p
  → 학습 도메인을 벗어난 데이터에서도 개선 확인
```

### 8.4 왜 Cascade의 교정률이 낮은가?

```
Cascade 작동 방식:
  MNV2 confidence > τ → MNV2 결정
  MNV2 confidence ≤ τ → SpecM 결정

τ에 따른 교정률 변화:
  τ=0.3 → 교정률 0.0%  ← MNV2가 항상 τ보다 높음
  τ=0.6 → 교정률 18.6%
  τ=0.8 → 교정률 42.3%
  τ=0.9 → 교정률 54.1%
  τ=0.95 → 교정률 66.2%

원인: MNV2 오분류의 mean confidence ≈ 0.77
      (즉, 틀리면서도 77%의 확신을 갖는 "confident-but-wrong")
→ τ를 매우 높게 설정해야만 SpecM이 개입 가능
→ 그러면 MNV2의 정확한 예측도 SpecM으로 override될 위험
```

---

## 9. Phase 4: 3-Layer 종합 평가

### 9.1 Layer 1 — 예측 성능

```
macro-F1 (높을수록 좋음)

MNV2 단독          ████████████████████████████████████████ 0.9581
Cascade(τ=0.6)    ███████████████████████████████████████  0.9547
SpecM-v4           █████████████████████████████████        0.8829
SpecM-Comp γ=1    ██████████████████████████████           0.7959
```

> SpecM-Comp의 macro-F1이 낮은 이유: 3-class 공간에서 ai_generated를 직접 판별 못함.
> **SpecM-Comp는 단독 사용이 아닌 HEMA와의 결합이 목적.**

### 9.2 Layer 2 — 선택적 예측 (교정 능력)

```
MNV2 오분류 교정률 (높을수록 좋음)

MNV2 단독          ░░░░░░░░░░░░░░░░░░░░░  0%
Cascade(τ=0.6)    ████░░░░░░░░░░░░░░░░░  18.6%
SpecM-v4           ████████████░░░░░░░░░  61.0%
SpecM-Comp γ=1    █████████████░░░░░░░░  69.6%  ← v4 대비 +8.6%p
```

**세부 오분류 패턴 분석 (base 데이터셋)**:

```
오분류 유형별 교정률:

manipulated → authentic (64개 오분류):
  SpecM-Comp γ=1: 59/64 = 92.2% 교정  ★★★
  SpecM-v4:       (비슷한 수준)

authentic → manipulated (18개 오분류):
  SpecM-Comp γ=1: 7/18 = 38.9% 교정   (낮음)
  → 실제 authentic을 manipulated로 잘못 예측한 케이스는 SpecM도 어려워함
```

### 9.3 Layer 3 — 캘리브레이션

```
ECE (낮을수록 캘리브레이션 양호)

MNV2 단독          ██░░░░░░░░  0.028  ← 가장 양호
Cascade(τ=0.6)    ██░░░░░░░░  0.028
SpecM-v4           █████░░░░░  0.054
SpecM-Comp γ=1    █████████░  0.092  ← 캘리브레이션 불량

AURC (낮을수록 selective prediction 성능 양호)

MNV2 단독         ░░░░░░░░░░  0.007  ← 최고
Cascade           █░░░░░░░░░  0.011
SpecM-v4          ████░░░░░░  0.038
SpecM-Comp γ=1    ████████░░  0.117  ← 높음
```

> SpecM-Comp의 높은 ECE/AURC: 3-class 공간에서 ai_gen을 처리하지 못해 발생.
> binary(auth/manip) 공간으로 제한하면 개선될 것으로 예상.

---

## 10. 종합 요약

### 10.1 실험 결과 한 눈에 보기

```
HEMA-XGBoost (MNV2 + SpecM-Comp γ=1) 성능 요약

지표                  기존 HEMA PoC    HEMA-XGBoost    개선
──────────────────────────────────────────────────────────
macro-F1              ~0.96             0.9617         ≈동일
MNV2 교정률 (avg)     45.5%             59.8%         +14.3%p
dsC 교정률            63.0%            100.0%         +37.0%p ★
aigenproxy 교정률     34.3%             66.7%         +32.4%p ★
opensdi 교정률        34.8%             28.6%         -6.2%p  ↓
```

### 10.2 SpecM-Comp vs SpecM-v4 (HEMA-XGBoost 내)

```
HEMA에 어떤 SpecM을 쓸 때 더 좋은가?

              macro-F1   교정률
──────────────────────────────
HEMA + v4     0.9557     0.602
HEMA + Comp   0.9617     0.598
              ─────────────────
차이          +0.006p    ≈동일
```

> SpecM-Comp가 HEMA-XGBoost 구조에서 v4 대비 **+0.6%p F1 개선**.
> 교정률은 거의 동일 → EWCT가 HEMA 특징 공간을 보완하는 방향으로 작용.

### 10.3 γ 하이퍼파라미터 가이드

```
γ=1: ◎ 권장
  - val 안정, F1 최적
  - w(x) ∈ [0,1] → 클리핑 불필요
  - TS 적용 권장 (+3.0%p F1)

γ=2: △ 조건부 사용
  - 교정률 약간 더 높으나 val 불안정
  - w_max 클리핑도 여전히 불필요 (w∈[0,1])

γ=3: ✗ 비권장
  - val 급락, F1 감소
  - median=0.006으로 대부분 샘플 무시
```

---

## 11. 후속 검증 실험

| 실험 | 상태 | 결론 |
|------|------|------|
| γ=2 w_max=5 학습 | ✅ **완료** | w_max=10과 완전 동일 (비트 단위) — 이론 검증 ✓ |
| γ=2 w_max=20 학습 | ✅ **완료** | w_max=5/10과 완전 동일 (비트 단위) — 이론 완전 검증 ✓ |
| HEMA LOO-CD 평가 | ✅ **완료** | avg F1=0.9452 (in-domain 0.9617 대비 -1.7%p), 교정률=38.2% (in-domain 59.8% 대비 -21.6%p) |
| HEMA feature ablation | ✅ **완료** | Cross 3-dim > Full 13-dim 관찰. 다만 현재 증거만으로는 domain-invariant signal이라 단정하지 않고, 저차원 과적합 완화 가능성과 함께 해석 |
| HEMA 모델 비교 (comp_g1 vs v4 vs noTS) | ✅ **완료** | comp_noTS 최고 (F1=0.9546, 교정률=41.4%) |
| comp_noTS/comp_g1 full-coverage 재생성 | ✅ **완료** | `run_specm_eval.py`로 4-DS 전체 JSONL 생성. 이후 learned fusion 재평가는 이 full-coverage 기준 사용 |
| HEMA action-gate (full-coverage) | ✅ **완료** | strong+comp_noTS에서 ICWMV에 근접(F1 0.9628 vs 0.9630)했지만 교정률은 0.423 vs 0.492로 열세 |
| HEMA ICWMV-veto (full-coverage) | ✅ **완료** | weak+comp_noTS에서 F1는 +1.31%p 개선했지만 교정률이 52.6%→16.6%로 붕괴. ICWMV 동시 초과 실패 |

**γ=2 w_max=5 확인 결과** (2026-03-23 완료):
```
γ=2, w_max=5 vs w_max=10 (실험 비교)
        base    dsC    opensdi  aigenproxy  avg
─────────────────────────────────────────────────
w_max=5  0.8122  0.8139  0.4781  0.9472   0.7628
w_max=10 0.8122  0.8139  0.4781  0.9472   0.7628
```
→ **이론 완전 검증**: w_max는 논문 하이퍼파라미터에서 제거 가능

---

## 12. 논문용 종합 결과 테이블 (Paper-Ready)

### Table 1: EWCT Ablation Study (SpecM 단독 평가, auth+manip 서브셋)

```
Table 1. Error-Weighted Complementary Training Ablation Study
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                      manip-F1                              Avg   Avg
Method               base   dsC   opensdi  aigenproxy    F1  Corr.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SpecM-v4 (baseline)  .870  .866    .676     .966        .844  61.0%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
γ ablation (w_max=10, TS on):
  EWCT γ=1 ★        .802  .825    .587     .960        .793  69.6%
  EWCT γ=2           .812  .814    .478     .947        .763  71.4%
  EWCT γ=3           .803  .797    .472     .933        .751  66.8%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
w_max ablation (γ=1, TS on):
  w_max=1.0          .802  .825    .587     .960        .793  69.6%  ← =γ=1
  w_max=10.0 ★       .802  .825    .587     .960        .793  69.6%  ← =γ=1
  (w_max>1 클리핑 불필요: w(x)=(1-c(x))∈[0,1])
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Temperature Scaling ablation (γ=1, w_max=10):
  TS on  ★           .802  .825    .587     .960        .793  69.6%
  TS off             .709  .825    .573     .945        .763  77.2%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
★ = 선택된 최종 구성: EWCT γ=1, w_max=10, TS=on ("SpecM-Comp")
Corr. = MNV2 오분류 교정률 (4개 데이터셋 평균)
```

### Table 2: Fusion Strategy Comparison (in-domain 70/30, SpecM-Comp 기반)

```
Table 2. Fusion Strategy Comparison
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                      macro-F1                              Avg   MNV2
Fusion Method        base   dsC   opensdi  aigenproxy    F1  Corr.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MNV2-only            .917  .970    .949     .966        .950   0.0%
SpecM-only           .793  .812    .612     .956        .793  69.6%
ICWMV                .937  .973    .923     .985        .955  43.3%
Cascade (τ=0.6)      .924  .973    .944     .974        .954  18.6%
FoE-MLP (ICLR24)     .903  .944    .955     .978        .945  32.6%
HEMA-XGB ★           .919  .983    .967     .978        .962  59.8%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Δ (HEMA vs MNV2)     +.002  +.013  +.018   +.012      +.011  ─

참고: SpecM-v4 기반 결과 (ICWMV 버그 수정 후)
HEMA-v4 avg_F1=0.9557, ICWMV-v4 avg_F1=0.9652 (v4의 이진 F1이 더 높아 ICWMV 유리)
★ = HEMA-XGBoost(comp_g1): F1+교정률 균형 최적
```

### Table 3: 3-Layer Comprehensive Evaluation

```
Table 3. Three-Layer Evaluation Protocol
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
              [Layer 1]      [Layer 2]     [Layer 3]
              Prediction     Selective     Calibration
Method        macro-F1   Corr.   AURC     ECE   Brier
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MNV2 (base)  .9581 ★   0.0%  .00677 ★  .028 ★  .021 ★
Cascade τ.6  .9547      18.6%  .01059   .028    .027
SpecM-v4     .8829      61.0%  .03794   .054    .044
SpecM-Comp   .7959      69.6%  .11669   .092    .073
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
(모든 수치: 4개 데이터셋 평균)
★ = 각 지표 최고/최저값
해석: SpecM-Comp는 단독 사용보다 HEMA 결합용으로 설계됨
      → HEMA+SpecM-Comp = Table 2 기준 macro-F1 0.962 (MNV2 대비 +1.1%p)
         + 교정률 59.8% (MNV2 대비 +59.8%p)
```

---

## 13. HEMA LOO-CD 결과 (실행 중 → 완료 시 업데이트)

### 13.1 동기: 왜 LOO-CD가 필요한가?

```
기존 in-domain 70/30 split의 문제:

  학습 데이터: base[70%] + dsC[70%] + opensdi[70%] + aigenproxy[70%]
  테스트 데이터: 같은 4개 데이터셋의 나머지 30%

  → 학습/테스트가 같은 도메인 → "in-domain 과대평가"
  → 실제 배포 환경에서는 보지 못한 도메인이 올 수 있음
```

**LOO-CD (Leave-One-Out Cross-Dataset)** 는 더 공정한 평가:

```
테스트: base        → 학습: dsC + opensdi + aigenproxy
테스트: dsC         → 학습: base + opensdi + aigenproxy
테스트: opensdi     → 학습: base + dsC + aigenproxy
테스트: aigenproxy  → 학습: base + dsC + opensdi

→ 각 테스트 도메인은 학습에서 완전히 배제됨
→ 교차 도메인 일반화 능력을 측정
```

### 13.2 실험 구성

```
run_hema_xgb_crossdataset.py --specm comp_g1 --ablation --compare

1) LOO-CD: HEMA-XGBoost (comp_g1) 4-fold cross-dataset eval
2) All-In CV: 4개 데이터셋 전체 5-fold CV (상한선 추정)
3) Feature Ablation: 13-dim 중 8개 subset 비교
4) Model Comparison: comp_g1 vs v4 vs comp_noTS (LOO-CD 공정 조건)
```

### 13.3 결과

**LOO-CD vs In-Domain 비교**:

```
HEMA-XGBoost 평가 방식별 성능 비교

                        avg macro-F1   교정률
────────────────────────────────────────────
in-domain 70/30 ★★     0.9617         59.8%   ← Phase 3 (이전)
All-In 5-fold CV ★     0.9514         38.9%   ← 부분 교차검증
LOO-CD (공정)           0.9452         38.2%   ← 가장 공정
────────────────────────────────────────────
overfitting gap:        -0.0165 (-1.7%p)  -21.6%p
```

> **핵심 발견 1**: in-domain 교정률(59.8%) → LOO-CD 교정률(38.2%) = **-21.6%p 급락**!
> HEMA의 오분류 교정 능력이 in-domain에서 크게 과대평가됨.
> 즉, HEMA는 "어떤 샘플이 오분류인지 학습"하는데, 이 패턴이 도메인 종속적임.

**LOO-CD 데이터셋별 상세**:

```
데이터셋   학습 도메인           F1      교정률
────────────────────────────────────────────
base      dsC+opensdi+aigenproxy  0.9025  25.6%
dsC       base+opensdi+aigenproxy 0.9650  58.8%
opensdi   base+dsC+aigenproxy     0.9317  29.6%
aigenproxy base+dsC+opensdi       0.9817  38.9%
────────────────────────────────────────────
평균                               0.9452  38.2%
```

**Feature Ablation (LOO-CD 공정 조건)**:

```
Feature 서브셋       avg F1    교정률    해석
──────────────────────────────────────────────────
Full (13-dim)        0.9452    38.2%   기준
Cross (3-dim) ★★     0.9525    39.0%   ← Full보다 오히려 높음!
Scores-only (5)      0.9506    41.5%
MNV2+SpecM (10)      0.9481    41.7%
w/o aigen (11)       0.9478    40.0%
MNV2-only (6)        0.9447    18.1%   ← 교정률 낮음
SpecM-only (4)       0.7834    66.4%   ← 교정률 높지만 F1 낮음
──────────────────────────────────────────────────
```

> **핵심 발견 2**: **Cross (3-dim) > Full (13-dim)** in LOO-CD!
> 3개 cross-modal 특징만으로도 13-dim보다 성능이 더 좋음.
>
> Cross 특징:
> - `cross_auth_diff`: |P_auth(MNV2) - P_auth(SpecM)| ← 두 모델 불일치 크기
> - `cross_null_signal`: P_aigen(MNV2) × P_auth(SpecM) ← AI생성 신호
> - `cross_manip_agree`: P_manip(MNV2) × P_manip(SpecM) ← 조작 동의 신호
>
> 이 3개 특징이 **도메인 불변(domain-invariant)**한 신호를 담고 있음 → 교차 도메인 일반화의 핵심!

**Model Comparison (LOO-CD 공정 조건)**:

```
SpecM 모델별 HEMA-XGBoost LOO-CD 성능

모델                  avg F1   교정률   해석
──────────────────────────────────────────────────
HEMA + comp_g1 (TS)  0.9452   38.2%   in-domain 최고 → LOO-CD 최하
HEMA + v4             0.9528   36.9%
HEMA + comp_noTS ★    0.9546   41.4%   ← LOO-CD 기준 최고!
──────────────────────────────────────────────────
```

> **핵심 발견 3**: LOO-CD에서 **comp_noTS > v4 > comp_g1** (역전!)
>
> in-domain: comp_g1 > v4 → LOO-CD: v4 > comp_g1
>
> 해석: TS(Temperature Scaling)가 MNV2 신호를 "부드럽게" 만들어 in-domain에서는 좋지만,
> 교차 도메인에서는 오히려 도메인 특이적 패턴을 학습하게 됨.
> comp_noTS는 더 날카로운 weight w(x)로 SpecM을 학습 → 더 도메인 불변적인 특징 생성.

> 주의: 위 `comp_g1 / comp_noTS` 비교는 당시 사용 가능했던 complementary JSONL 기준이다.
> 2026-03-23 후속 실험에서는 `run_specm_eval.py`로 **full-coverage 4-DS JSONL**을 다시 생성해
> 공정 조건을 맞춘 뒤 action-gate / ICWMV-veto를 재평가했다. 결과는 아래 14장에 정리한다.

### 13.4 논문 시사점

```
┌────────────────────────────────────────────────────────────────┐
│  HEMA 논문에서 반드시 명시해야 할 사항:                           │
│                                                                │
│  1. in-domain 평가(0.9617)는 낙관적 추정 — LOO-CD(0.9452) 사용  │
│  2. 교정률은 LOO-CD 기준 38.2% (in-domain 59.8%와 구분)          │
│  3. Cross 3-dim 특징이 가장 도메인 불변적 — ablation에서 입증     │
│  4. comp_noTS가 LOO-CD 기준 최적 (F1+교정률 균형)               │
└────────────────────────────────────────────────────────────────┘
```

---

## 14. HEMA 복구 시도: full-coverage action-gate / ICWMV-veto

기존 HEMA 논의를 더 공정하게 보기 위해, 먼저 complementary specialist 결과를
`run_specm_eval.py --model comp_noTS/comp_g1`로 **전체 4개 데이터셋 full-coverage JSONL**로 재생성했다.
그 위에서 두 가지 learned fusion 복구안을 평가했다.

1. `action-gate`
   - ai_gen은 MNV2가 잠그고
   - auth/manip disagreement에서만 `keep_mnv2 / override_specm / fallback_icwmv`를 학습
2. `ICWMV-veto`
   - 기본 예측은 ICWMV 유지
   - 단, ICWMV가 MNV2를 실제로 override한 샘플에서만 `keep / revert`를 학습

**요약 결과**:

```
Full-coverage HEMA repair attempts (LOO-CD)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MNV2      SpecM       Method            avg F1   Corr.   해석
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
strong    v4          ICWMV             0.9652   35.4%   기준
strong    v4          action-gate       0.9622   32.5%   둘 다 하락
strong    v4          ICWMV-veto        0.9652   35.4%   사실상 동일
strong    comp_noTS   ICWMV             0.9630   49.2%   기준
strong    comp_noTS   action-gate       0.9628   42.3%   F1 근접, corr 하락
strong    comp_noTS   ICWMV-veto        0.9622   40.0%   둘 다 못 넘김
weak      comp_noTS   ICWMV             0.8350   52.6%   기준
weak      comp_noTS   action-gate       0.8492   29.7%   F1만 +1.42%p
weak      comp_noTS   ICWMV-veto        0.8481   16.6%   F1만 +1.31%p
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**핵심 해석**:

- 현재의 **scalar feature 기반 learned gate/veto는 ICWMV를 안정적으로 넘지 못했다.**
- weak+comp_noTS에서 F1이 오르더라도 교정률이 크게 무너져, "둘 다 개선"은 실패했다.
- 즉, learned fusion이 실패한 이유는 단순히 구조가 덜 보수적이어서가 아니라,
  **beneficial override와 harmful override를 현재 메타 특징으로 충분히 분리하지 못했기 때문**이다.

흥미로운 점은 **oracle veto**의 상한은 분명히 존재한다는 점이다.

```
Oracle veto upper bound
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MNV2      SpecM       ICWMV F1   Oracle F1   ΔF1
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
strong    comp_noTS   0.9630     0.9726     +0.0095
weak      comp_noTS   0.8350     0.8935     +0.0585
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

즉, **ICWMV 위에 selective learned veto를 얹는 방향 자체는 틀리지 않았지만**,
현재의 score-level / scalar-level 특징만으로는 oracle gap을 따라가지 못한다.
이 결과는 "HEMA를 조금 더 튜닝하면 된다"보다,
**representation-level evidence 또는 richer disagreement signal이 필요하다**는 쪽을 지지한다.

---

## 15. 다음 단계

**즉시 가능:**
- A. ICWMV-v4를 동일한 LOO-CD 프로토콜로 고정 baseline으로 재비교
- B. representation-level disagreement evidence가 들어간 learned veto/gate 설계
- C. 논문 C6 섹션에서 `w_max 제거 + confident-but-wrong + in-domain 과대평가` 중심으로 스토리 재정렬

**GPU 필요:**
- D. HEMA-XGBoost를 raw representation 또는 hidden embedding 입력으로 확장
- E. SpecM-Comp 양자화 (INT8) → 경량화 검증

---

*이 문서는 실험 완료에 따라 업데이트됩니다. 최신 결과 파일:*
`experiments/results/specm_complementary_eval/`, `experiments/results/specm_eval/`, `experiments/results/fuser_comparison/`, `experiments/results/hema_action_gate/`, `experiments/results/hema_icwmv_veto/`, `experiments/results/phase4_3layer/`
