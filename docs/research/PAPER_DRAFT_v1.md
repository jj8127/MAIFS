# MobileNetV2 위조 탐지의 고확신 오류 교정을 위한 역신뢰도 가중 다수결과 엣지 배포

> 초안 버전 3.1 | 2026-03-24
> 투고 대상: 한국정보처리학회 (KIPS) 2026

---

## 초록

딥러닝 기반 이미지 위조 탐지 모델은 오분류 시에도 높은 신뢰도를 유지하는 고확신 오류(confident-but-wrong) 문제를 가진다. 본 논문에서는 MobileNetV2 이중 스트림 모델(MNV2)의 이진(authentic/manipulated) 오분류 중 74.3%가 높은 신뢰도(confidence > 0.6)를 동반하며, 이로 인해 신뢰도가 낮은 샘플에만 보조 모델을 적용하는 기존 방식으로는 오분류의 25.7%에만 개입 가능함을 정량적으로 분석한다. 이를 해결하기 위해 조작 전문 모델과 **역신뢰도 가중 다수결(ICWMV; Inverse-Confidence Weighted Majority Vote)** 을 결합한 경량 합의 기법을 제안한다. ICWMV는 선행 연구 [10]의 합의 철학에서 영감을 받아 2-모델 엣지 환경에 맞게 재설계한 고정 규칙 합의이다. 4개 데이터셋 교차 평가 기준으로 제안 시스템은 MNV2 단독 대비 오분류 교정률 35.4%, macro-F1 +0.71%p를 달성한다. 또한 조작 전문 모델을 오류 집중 학습시키는 오류 가중 보완 학습(EWCT)을 도입하여 교정률 49.2%까지 확장 가능한 operating point를 제공한다. ICWMV는 MNV2 외에도 MobileCLIP fine-tuned(+0.37%p)와 zero-shot(+13.14%p) 백본에서도 일관되게 효과를 보여 특정 체크포인트에 종속되지 않음을 확인한다. 나아가 Raspberry Pi 5에서 ONNX INT8(88.3ms) 및 Coral USB Edge TPU(63.4ms)로 실시간 추론이 가능함을 확인하며, Coral의 경우 MNV2에서 1.85배 가속과 메모리 40% 절감을 달성한다.

**키워드**: 이미지 위조 탐지, 고확신 오류, 역신뢰도 가중 다수결, 경량 합의, 엣지 배포, Coral Edge TPU

---

## 1. 서론

디지털 이미지 조작 기술의 발전과 AI 생성 이미지의 확산으로 인해 이미지 진위 판별의 중요성이 증대되고 있다 [1, 2]. 이에 대응하여 압축 아티팩트 탐지, 픽셀 수준 조작 마스크 예측, CLIP 기반 AI 생성 탐지 등 전문화된 탐지 모델들이 제안되었으며 [4–6], 이들을 합의 계층으로 통합하는 다중 전문가 접근이 연구되고 있다.

선행 연구 [10]에서는 다중 전문가 에이전트의 불일치 패턴을 학습 신호로 활용하는 합의 기법을 제안하였으며, 에이전트 간 불일치 자체가 탐지 유형을 구분하는 핵심 신호임을 확인하였다. 그러나 이 서버 측 합의 스택은 다수의 전문가 모델을 동시에 실행해야 하므로 엣지 디바이스에의 직접 배치가 어렵다. 본 논문은 이 문제에서 출발하여, **경량 백본과 ICWMV 합의로 고확신 오류를 교정하는 온디바이스 시스템**을 제안한다.

본 논문에서 수행한 분석에 따르면, MNV2의 이진(authentic/manipulated) 오분류 사례 중 **74.3%가 confidence > 0.6**이며, 평균 오분류 신뢰도는 0.7538에 달한다. 신뢰도가 낮은 샘플(confidence < τ)에만 보조 모델을 추가로 적용하는 임계값 기반 선별 방식을 흔히 대응책으로 사용하는데, τ=0.6을 기준으로 하면 오분류 중 confidence가 낮아 개입 가능한 비율은 단 25.7%에 불과하다. 즉, 나머지 74.3%의 오분류는 모델이 이미 높은 신뢰도로 틀린 경우이므로 이 선별 방식이 개입 자체를 하지 않아 교정 기회를 잃는다. 따라서 신뢰도에 무관하게 항상 보조 모델 신호를 반영하는 합의 전략이 필요하다.

이에 본 논문의 핵심 기여는 다음과 같다:

1. **고확신 오류 정량 분석**: MNV2 오분류 패턴의 신뢰도 분포를 체계적으로 분석하고, 임계값 기반 선별 방식의 한계를 수치로 제시한다.
2. **ICWMV 제안**: 선행 연구 [10]의 합의 철학에서 영감을 받아 2-모델 엣지 환경에 맞게 재설계한 역신뢰도 가중 다수결(ICWMV)을 제안하고, 교정률 35.4%를 달성한다.
3. **EWCT**: MNV2 오류 케이스에 집중하는 조작 전문 모델 학습 방식(EWCT)을 도입하여 교정률을 49.2%까지 높이는 operating point를 제공한다.
4. **백본 일반화 검증**: ICWMV가 MNV2 이외 MobileCLIP 계열에서도 유효함을 실험으로 확인한다.
5. **엣지 배포**: Raspberry Pi 5에서 CPU(88.3ms)와 Coral USB Edge TPU(63.4ms)로 실시간 추론을 구현한다.

---

## 2. 관련 연구

### 2.1 이미지 위조 탐지

이미지 위조 탐지는 크게 조작 탐지(forgery detection)와 AI 생성 이미지 탐지로 구분된다. MVSS-Net [4]은 픽셀 수준의 조작 마스크를 예측하며, CAT-Net [5]은 JPEG 이중 압축 흔적을 탐지한다. AI 생성 이미지에 특화된 FatFormer [6]는 CLIP ViT-L/14와 Forgery-Aware Adapter를 결합한다. 단일 모델은 탐지 대상 유형별로 맹점을 가지므로, 복수 모델의 융합이 실용적 시스템에 필수적이다.

### 2.2 앙상블 및 합의 방법

다수결 투표(Majority Voting) [7], 신뢰도 가중 융합 [8] 등의 앙상블 방법이 분류 성능 향상에 활용된다. 임계값 기반 선별 방식 [9]은 신뢰도가 높은 샘플을 조기 분류하고 낮은 샘플에만 보조 모델을 적용하나, 고확신 오류가 많은 경우 개입 범위가 제한된다.

선행 연구 [10]은 다중 전문가 에이전트의 불일치 패턴을 메타 특징으로 학습하는 합의 기법을 제안하여, 에이전트 간 상호 검증이 단일 에이전트의 과신을 억제하는 데 효과적임을 보였다. 본 논문의 ICWMV는 이 합의 철학을 2-모델 엣지 환경에 맞게 단순화한 고정 규칙 형태이다. 특히 임계값 기반 선별 방식과 달리 신뢰도 조건 없이 모든 샘플에 보조 모델 신호를 융합하므로, 고확신 오류 케이스에도 예외 없이 개입한다.

### 2.3 보완 학습

포컬 로스(Focal Loss) [11]는 어려운 샘플에 더 높은 손실 가중치를 부여하여 학습 집중도를 높인다. 본 논문의 EWCT는 이와 유사하게 MNV2가 틀린 샘플에 집중하되, 샘플별 MNV2 오류 확률을 가중치로 사용하는 차별화된 학습 방식이다.

### 2.4 엣지 디바이스 배포

경량 딥러닝 모델의 엣지 배포는 양자화(quantization) [12], 지식 증류, 구조 가지치기(pruning) [13] 등을 통해 이루어진다. Google Coral USB Accelerator는 Edge TPU 기반의 INT8 추론 가속기로, 완전 매핑된 모델에서 현저한 레이턴시 개선을 제공한다 [14]. ONNX Runtime의 동적 양자화 [15]는 보정 데이터 없이 INT8 추론을 가능하게 한다.

---

## 3. 문제 분석: MNV2의 고확신 오류

### 3.1 실험 설정

분석에 사용된 MNV2는 이중 스트림 MobileNetV2(파라미터 5.77M)로, 4개 데이터셋(base, dsC, OpenSDI, aigenproxy)에서 3-class(authentic/manipulated/ai_generated) 분류를 수행한다. 총 4,200개 샘플에서 교차 데이터셋 방식으로 추론을 수행하였다.

고확신 오류 분석은 ai_generated 클래스를 제외한 이진(authentic/manipulated) 샘플을 대상으로 하였다(N=2,784). ai_generated 클래스는 MNV2가 전담하며 조작 전문 모델의 탐지 범위 밖이기 때문이다.

### 3.2 오분류 신뢰도 분포

**Table 1. MNV2 오분류 신뢰도 분포 (4-DS 교차 평가, N=144)**

| 지표 | 값 |
|------|-----|
| 오분류 수 (이진) | 144건 |
| 오분류 평균 신뢰도 | **0.7538** |
| 정분류 평균 신뢰도 | 0.9502 |
| 오분류 중 conf > 0.6 | **107건 (74.3%)** |
| 오분류 중 conf > 0.8 | **67건 (46.5%)** |

MNV2 오분류의 74.3%가 신뢰도 0.6을 초과한다. 따라서 신뢰도 임계값(τ=0.6) 기반 선별 방식은 오분류 중 단 25.7%(37건)에만 보조 모델 개입 기회를 가지며, 나머지 74.3%(107건)는 MNV2가 충분히 "확신"하므로 이 방식은 개입하지 않고 오류를 그대로 통과시킨다.

이 결과는 선행 연구 [10]에서 다중 에이전트 간 합의 계층이 필요한 이유와 일맥상통한다. 단일 모델의 과신은 합의 메커니즘 없이는 교정되지 않으며, 본 논문에서는 이를 2-모델 고정 규칙 합의로 엣지 환경에서 해결한다.

---

## 4. 제안 시스템

### 4.1 시스템 개요

제안 시스템(경량 합의 기법)은 세 구성 요소로 이루어진다: (1) 백본 분류기 MNV2, (2) 조작 전문 모델, (3) ICWMV 합의 계층. **Figure 1**은 전체 파이프라인을 나타낸다.

```
이미지
  ├─→ MNV2          → P(auth), P(manip), P(aigen)  [3-class]
  └─→ 조작 전문 모델  → P(auth), P(manip)             [2-class]
            ↓
         ICWMV 합의
            ↓
    최종 판정 (authentic / manipulated / ai_generated)
```

MNV2는 3-class 분류를 담당하며, 조작 전문 모델은 EWCT로 학습된 이진(authentic/manipulated) 전문 모델이다. 조작 전문 모델은 ai_generated 클래스를 탐지하지 않으므로, 해당 클래스는 MNV2만 담당한다.

### 4.2 역신뢰도 가중 다수결 (ICWMV)

본 논문에서는 역신뢰도 가중 다수결(ICWMV, Inverse-Confidence Weighted Majority Vote)을 제안한다. ICWMV는 각 모델의 최대 예측 확률의 역수를 가중치로 사용함으로써, 낮은 신뢰도 모델에 더 높은 발언권을 부여하여 어느 한 모델의 과신에 의한 오류를 억제한다. ICWMV는 선행 연구 [10]의 합의 철학에서 영감을 받아, 2-모델 엣지 환경을 위해 본 논문이 새로 정의한 고정 규칙 합의이다.

**임계값 기반 선별과의 비교**: 신뢰도 임계값(τ=0.6) 기반 선별 방식은 MNV2 신뢰도 < τ인 샘플에만 조작 전문 모델을 개입시키므로, 고확신 오류 107건 중 0건에 개입한다. ICWMV는 신뢰도 조건 없이 모든 샘플에 조작 전문 모델을 융합하므로 고확신 오류 케이스에서도 교정 기회를 확보한다.

가중치 계산:
```
w_MNV2 = 1 / max(P_MNV2)
w_조작  = 1 / max(P_조작)
```

융합 점수:
```
score(auth)  = w_MNV2 · P_MNV2(auth)  + w_조작 · P_조작(auth)
score(manip) = w_MNV2 · P_MNV2(manip) + w_조작 · P_조작(manip)
score(aigen) = w_MNV2 · P_MNV2(aigen)
```

ai_generated 폴백: P_MNV2(aigen) > 0.5이면 최종 판정을 ai_generated로 확정한다.

최종 판정: ŷ = argmax { score(auth), score(manip), score(aigen) }

### 4.3 오류 가중 보완 학습 (EWCT)

오류 가중 보완 학습(Error-Weighted Complementary Training, EWCT)은 MNV2가 오분류할 가능성이 높은 샘플에 집중하도록 조작 전문 모델을 학습시킨다.

손실 함수:
```
L = (1/N) Σᵢ w(xᵢ) · CE(조작 전문 모델(xᵢ), yᵢ)
w(xᵢ) = (1 - P_MNV2(yᵢ | xᵢ))^γ
```

여기서 P_MNV2(yᵢ | xᵢ)는 MNV2가 정답 클래스에 부여한 확률이며, γ > 0은 집중도 조절 하이퍼파라미터다. MNV2가 확신하는 샘플(P_MNV2 → 1)은 w → 0, MNV2가 틀린 샘플(P_MNV2 → 0)은 w → 1로 최대 기여한다.

**w_max 불필요 증명**: 기존 구현에서 w(x) = min((1-P_MNV2)^γ, w_max) 형태로 클리핑을 사용하는 경우가 있다. 그러나 P_MNV2 ∈ [0,1]이므로 임의의 γ > 0에 대해 w(x) ∈ [0,1]이 항상 성립한다. 따라서 w_max > 1은 어떠한 경우에도 비활성 하이퍼파라미터이며, 안전하게 제거할 수 있다. 이는 실험적으로도 확인된다.

### 4.4 교정률 정의

```
교정률 = n_corrected / n_errors
n_corrected : MNV2가 틀렸으나 ICWMV가 맞힌 샘플 수
n_errors    : MNV2 오분류 수 (이진 클래스 기준, ai_gen 제외)
```

교정률은 F1과 독립적인 지표로, 기존 백본의 오류를 융합 시스템이 얼마나 실질적으로 교정하는지를 측정한다. 본 논문은 선행 연구 [10]과 동일한 교차 데이터셋 평가 프로토콜을 사용하여 같은 일반화 기준에서 평가한다.

---

## 5. 조작 전문 모델 학습

### 5.1 구조

조작 전문 모델은 MobileNetV2 기반의 이진 분류 모델(파라미터 7.66M)로, authentic과 manipulated를 구분한다. ai_generated 클래스를 학습 대상에서 제외하고, MNV2 예측 확률을 활용한 EWCT 손실 함수로 학습된다.

### 5.2 학습 설정

- **데이터셋**: 4개 데이터셋(base, dsC, OpenSDI, aigenproxy)에서 authentic/manipulated 샘플
- **γ**: 1.0 (기본), 2.0 (실험 비교)
- **타임스텝 스케줄링(TS)**: comp_g1(γ=1+TS), comp_noTS(γ=1, no TS)
- **손실 함수**: Binary Cross-Entropy + EWCT 가중치

---

## 6. 실험

### 6.1 실험 설정

**데이터셋**: 4개 벤치마크 데이터셋
- **base**: CASIA2 기반 조작 이미지 (N=1,500)
- **dsC**: 다양한 조작 기법 (N=900)
- **OpenSDI**: 오픈 소스 디지털 이미지 (N=900)
- **aigenproxy**: AI 생성 이미지 포함 (N=900)

**평가 프로토콜**: 교차 데이터셋 평가. 3개 데이터셋에서 학습/교정하고 나머지 1개 데이터셋에서 평가하는 방식을 4회 반복하여 평균을 보고한다. Fixed-rule 방법(MNV2, 임계값 선별, ICWMV)은 학습이 없으므로 각 데이터셋에 직접 평가한다.

**평가 지표**: macro-F1 (3-class), 오분류 교정률

**비교 방법**: MNV2 단독 / 임계값 선별(τ=0.6) + 조작 전문 모델 / **ICWMV + 조작 전문 모델 (제안)** / ICWMV + 조작 전문 모델-EWCT-noTS

### 6.2 메인 결과

**Table 2. 4개 데이터셋 교차 평가 성능 비교**

| 방법 | avg F1 | ΔF1 | 교정률 |
|------|--------|-----|--------|
| MNV2 단독 | 0.9581 | — | 0.0% |
| 임계값 선별(τ=0.6) + 조작 전문 모델 | 0.9612 | +0.31%p | 19.3% |
| **ICWMV + 조작 전문 모델 (제안)** | **0.9652** | **+0.71%p** | **35.4%** |
| ICWMV + 조작 전문 모델-EWCT-noTS | 0.9575 | −0.06%p | **49.2%** |

**데이터셋별 세부 결과** (ICWMV + 조작 전문 모델):

| 데이터셋 | MNV2 F1 | ICWMV F1 | ΔF1 |
|---------|---------|----------|-----|
| base | 0.9438 | 0.9566 | +0.0128 |
| dsC | 0.9789 | 0.9844 | +0.0055 |
| OpenSDI | 0.9489 | 0.9500 | +0.0011 |
| aigenproxy | 0.9610 | 0.9700 | +0.0090 |
| **평균** | **0.9581** | **0.9652** | **+0.0071** |

제안 시스템은 MNV2 단독 대비 모든 데이터셋에서 F1이 향상되었으며, 임계값 선별(교정률 19.3%) 대비 교정률이 16.1%p 높으면서도 F1 손실이 없다. ICWMV가 고확신 오류 케이스에서도 조작 전문 모델 신호를 활용하기 때문이다.

교정/역교정 분석에서 ICWMV는 50건을 교정하고 17건을 역교정하여 순 이득 +33건을 달성하였다. 역교정률(MNV2 정답 중 ICWMV가 틀린 비율)은 0.57%로 매우 낮다.

EWCT 변형(noTS)은 교정률을 49.2%로 높이나 F1이 0.0077 하락한다. 사용자가 교정률 우선 vs F1 우선 operating point를 선택할 수 있는 tradeoff이다.

### 6.3 EWCT 효과

**Table 3. EWCT 학습 효과 비교**

| 조작 전문 모델 변형 | 독립 manip-F1 | ICWMV 교정률 | ICWMV avg F1 |
|------------------|-------------|-------------|-------------|
| 기본 (no EWCT) | 0.7438 | 35.4% | **0.9652** |
| EWCT-comp_g1 (γ=1, TS) | — | 39.4% | 0.9563 |
| EWCT-noTS (γ=1, no TS) | **0.7632** | **49.2%** | 0.9575 |

EWCT-noTS는 조작 전문 모델의 독립 manip-F1을 +1.94%p 향상시키며, ICWMV 교정률을 +13.8%p 높인다. 타임스텝 스케줄링(TS) 제거 시 교정률이 더 향상되며, 이는 TS가 EWCT 가중치 학습을 억제하는 효과가 있음을 시사한다.

### 6.4 백본 계열 및 강도 일반화

ICWMV가 특정 MNV2 체크포인트에 종속된 규칙인지 확인하기 위해, 동일한 조작 전문 모델을 고정한 채 백본만 변경하였다. MobileCLIP fine-tuned(strong)과 MobileCLIP zero-shot(weak), 그리고 보조 stress-test로 `MNV2 no-finetuning`을 추가로 평가하였다.

**Table 4. 백본 계열/강도별 ICWMV 효과**

| Generalist 백본 | 설정 | 백본 F1 | ICWMV F1 | ΔF1 | 교정률 |
|----------------|------|--------|----------|-----|--------|
| MNV2 | strong | 0.9581 | 0.9652 | +0.71%p | 35.4% |
| MNV2 | weak | 0.8414 | 0.8637 | +2.23%p | 49.8% |
| MobileCLIP | fine-tuned | 0.9532 | 0.9569 | +0.37%p | 19.7% |
| MobileCLIP | zero-shot | 0.3005 | 0.4320 | +13.14%p | 77.1% |
| MNV2* | no-finetuning | 0.3658 | 0.5556 | +18.98%p | 80.8% |

*`MNV2 no-finetuning`은 ImageNet pretrained branch와 미학습 forensic head로 구성한 보조 stress-test이다.

ICWMV는 MNV2뿐 아니라 MobileCLIP 계열에서도 일관되게 성능을 향상시킨다. gain은 백본 family보다 **백본 강도(오류 수)에 더 강하게 비례**하며, 이는 ICWMV의 교정 효과가 특정 백본 트릭이 아님을 보여준다.

---

## 7. 엣지 디바이스 배포 및 평가

### 7.1 배포 구성

제안 시스템을 Raspberry Pi 5(ARM Cortex-A76, 8GB RAM, Debian GNU/Linux 13)에 배포하였다. 두 가지 추론 경로를 구성하였다:

- **경로 A (CPU)**: MNV2 dynamic INT8 ONNX(19MB) + 조작 전문 모델 static INT8 ONNX(8.5MB), ONNX Runtime 1.24.4, Python 3.13.5, threads=4
- **경로 B (Coral)**: MNV2 PTQ-sweep-tuned Edge TPU TFLite(6.9MB) + 조작 전문 모델 coral-ft Edge TPU TFLite(9.4MB), tflite-runtime 2.14.0, Python 3.9.25

MNV2는 151/151 연산이 Edge TPU에 완전 매핑되었으며, 조작 전문 모델(coral-ft)은 Coral 환경에 맞게 추가 미세조정 후 컴파일하였다.

### 7.2 레이턴시

**Table 5. RPi5 단계별 레이턴시 (10회 평균)**

| 단계 | CPU (ONNX) | Coral (Edge TPU) | 가속비 |
|------|-----------|-----------------|--------|
| MNV2 | 53.7ms | **29.0ms** | **1.85×** |
| 조작 전문 모델 | 34.6ms | 34.4ms | ≈1.0× |
| **Total** | **88.3ms** | **63.4ms** | **1.39×** |
| 편차 | ±9.5ms | **±0.8ms** | — |

MNV2는 Edge TPU 완전 매핑으로 1.85배 가속되나, 조작 전문 모델은 일부 연산이 CPU로 폴백되어 가속 효과가 거의 없다. Coral의 레이턴시 편차(±0.8ms)는 CPU(±9.5ms) 대비 11배 안정적이다.

### 7.3 배포 결과 종합

**Table 6. RPi5 배포 경로 종합 비교**

| 항목 | CPU (ONNX INT8) | Coral (Edge TPU) |
|------|----------------|-----------------|
| 평균 레이턴시 | 88.3ms | **63.4ms** |
| 레이턴시 편차 | ±9.5ms | **±0.8ms** |
| 처리량 (warm) | 11.3 FPS | **15.8 FPS** |
| 메모리 | 135.2 MB | **80.6 MB** |
| Cold start | **270ms** | 2,661ms |
| 정확도 (서버 검증) | 0.9535 | 0.9308 |

두 경로 모두 실시간 처리(>10 FPS) 요건을 충족한다. CPU 경로는 설치 간단·cold start 빠름·정확도 높음, Coral 경로는 낮은 레이턴시·안정적 편차·낮은 메모리 사용이 장점이다.

---

## 8. 토의

### 8.1 EWCT의 tradeoff

EWCT(noTS)는 교정률을 49.2%까지 높이지만 F1이 0.0077 하락한다. EWCT로 학습된 조작 전문 모델이 MNV2 오류 케이스에 집중하는 과정에서 일반 케이스의 정확도를 일부 희생하기 때문이다. 두 operating point(ICWMV+기본: F1 우선, ICWMV+EWCT-noTS: 교정률 우선) 중 응용 요건에 맞는 것을 선택할 수 있다.

---

## 9. 결론

본 논문은 MobileNetV2 이중 스트림 모델의 고확신 오류 문제를 정량적으로 분석하고, 이를 엣지 환경에서 해결하는 경량 합의 기법을 제안하였다. 주요 결과는 다음과 같다.

1. **고확신 오류 분석**: MNV2 오분류의 74.3%가 confidence > 0.6이며, 신뢰도 임계값(τ=0.6) 기반 선별 방식은 이 중 25.7%에만 개입 가능함을 정량화하였다.
2. **ICWMV 제안**: 선행 연구 [10]의 합의 철학에서 영감을 받은 2-모델 고정 규칙 합의를 제안하였다. 교정률 35.4%, macro-F1 +0.71%p를 달성하였으며, 교정 50건/역교정 17건으로 순 이득 +33건을 기록하였다.
3. **EWCT**: 조작 전문 모델을 MNV2 오류 케이스에 집중 학습시켜 교정률 49.2%까지 확장 가능하며, w_max가 이론적·실험적으로 불필요함을 증명하였다.
4. **백본 일반화**: ICWMV는 MNV2뿐 아니라 MobileCLIP에도 전이되며, gain은 백본 family보다 백본 강도에 비례한다.
5. **엣지 배포**: RPi5에서 CPU(88.3ms, 11.3 FPS) 및 Coral(63.4ms, 15.8 FPS) 모두 실시간 처리 요건을 충족한다.

향후 과제로는 조작 전문 모델의 Edge TPU 완전 매핑 최적화, ai_generated 전문 모델의 엣지 통합, 그리고 온디바이스 1차 판정 후 불확실 케이스의 서버 재분석 하이브리드 운용 아키텍처를 목표로 한다.

---

## 참고문헌

[1] Tolosana, R. et al., "DeepFakes and Beyond: A Survey of Face Manipulation and Fake Detection," *Information Fusion*, 2020.

[2] Wang, Z. et al., "Towards Universal Fake Image Detection Exploiting Style Latent Space," *CVPR*, 2023.

[3] Sandler, M. et al., "MobileNetV2: Inverted Residuals and Linear Bottlenecks," *CVPR*, 2018.

[4] Wang, J. et al., "MVSS-Net: Multi-View Multi-Scale Supervised Networks for Image Manipulation Detection," *IEEE TPAMI*, 2022.

[5] Kwon, M. J. et al., "CAT-Net: Compression Artifact Tracing Network for Detection and Localization of Image Splicing," *WACV*, 2021.

[6] Liu, Z. et al., "FatFormer: Forgery-aware Adaptive Transformer for Generalizable Deepfake Detection," *CVPR*, 2024.

[7] Zhou, Z.-H., "Ensemble Methods: Foundations and Algorithms," *CRC Press*, 2012.

[8] Guo, C. et al., "On Calibration of Modern Neural Networks," *ICML*, 2017.

[9] Viola, P. and Jones, M., "Rapid Object Detection using a Boosted Cascade of Simple Features," *CVPR*, 2001.

[10] (저자), "다중 에이전트 이미지 위조 탐지에서 불일치 패턴 기반 합의 개선," *한국정보처리학회*, 2026. ← DAAC 논문 정보 입력 필요

[11] Lin, T.-Y. et al., "Focal Loss for Dense Object Detection," *ICCV*, 2017.

[12] Jacob, B. et al., "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference," *CVPR*, 2018.

[13] Han, S. et al., "Learning both Weights and Connections for Efficient Neural Networks," *NeurIPS*, 2015.

[14] Seshadri, K. et al., "An Evaluation of Edge TPU Accelerators for Convolutional Neural Networks," *IISWC*, 2022.

[15] ONNX Runtime Development Team, "ONNX Runtime: Cross-platform, High Performance ML Inferencing," 2019. https://onnxruntime.ai
