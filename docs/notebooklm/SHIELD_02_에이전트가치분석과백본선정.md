# SHIELD Phase 1-2: 에이전트 가치 분석과 경량 백본 선정

## 1. Phase 1의 핵심 질문: "어떤 에이전트가 얼마나 필요한가?"

DAAC에서 4개 에이전트 모두를 사용하는 이유는 각자가 다른 탐지 능력을 갖기 때문이다. 하지만 경량화를 위해서는 "정말로 4개 모두 필요한가?"라는 질문에 답해야 한다. 단순히 성능이 높은 것을 고르는 것이 아니라, **어떤 에이전트가 다른 에이전트가 제공하지 못하는 고유한 정보를 제공하는가**를 측정해야 한다.

이를 위해 세 가지 분석 도구를 사용했다: Model Shapley, STII(Shapley-Taylor Interaction Index), PID(Partial Information Decomposition), 그리고 CKA(Centered Kernel Alignment).

---

## 2. Model Shapley: 에이전트별 기여도 정량화

**Model Shapley**는 협력 게임 이론에서 나온 개념으로, N=4일 때 2⁴=16개 모든 부분집합을 평가하여 각 에이전트의 한계 기여도를 정확히 계산한다.

계산 방법: 16개 에이전트 조합 각각에 대해 DAAC 메타 분류기를 재학습하고 Macro-F1을 측정한다. 이후 각 에이전트가 없을 때와 있을 때의 성능 차이를 모든 가능한 맥락에서 평균낸다.

### 결과

| 에이전트 | Shapley Value φ | 해석 |
|----------|----------------|------|
| FrequencyAgent (CAT-Net) | φ = +0.2690 | 가장 높은 고유 기여도 |
| FatFormerAgent | φ = +0.1216 | 두 번째 기여도 |
| NoiseAgent (MVSS-Net) | φ = +0.0886 | 세 번째 기여도 |
| SpatialAgent (Mesorch) | φ = +0.0547 | 가장 낮은 기여도 |

Frequency가 가장 높은 이유는 DAAC에서 가장 중요한 특징인 `disagree_frequency_fatformer`가 직접적으로 이 에이전트의 판정에 의존하기 때문이다.

---

## 3. STII: 에이전트 간 상호작용 측정

**STII(Shapley-Taylor Interaction Index)**는 두 에이전트가 함께 있을 때 시너지가 발생하는지, 아니면 서로 중복인지를 측정한다. 양수면 시너지(서로를 보완), 음수면 대체재(중복).

### 결과: 모든 쌍이 음수

| 에이전트 쌍 | STII | 해석 |
|-----------|------|------|
| Frequency ↔ FatFormer | -0.1823 | 가장 강한 대체재 관계 |
| Noise ↔ FatFormer | -0.1150 | 두 번째 |
| Frequency ↔ Noise | -0.0891 | 세 번째 |
| 나머지 쌍 | 모두 음수 | |

모든 쌍이 음수라는 것은 에이전트들이 서로 보완적이지 않고 중복된 측면이 있음을 의미한다. 이는 에이전트 선택 시 단순히 가장 기여도 높은 것을 고르면 된다는 단서다.

---

## 4. PID: 고유/중복/시너지 정보 분해

**PID(Partial Information Decomposition)**는 정보론 관점에서 에이전트들의 정보를 분해한다. 각 에이전트가 제공하는 정보 중:
- **Unique**: 해당 에이전트만 제공하는 고유 정보
- **Redundant**: 다른 에이전트와 중복되는 정보
- **Synergistic**: 두 에이전트가 함께 있을 때만 나타나는 정보

### 결과

| 에이전트 | Unique Information |
|----------|------------------|
| FrequencyAgent | 0.2029 (가장 높음) |
| FatFormerAgent | 0.0382 |
| NoiseAgent | 0.0311 |
| **SpatialAgent** | **0.0000 (전 데이터셋에서)** |

**핵심 발견**: Spatial 에이전트의 Unique information이 모든 데이터셋에서 0이다. 이것은 SpatialAgent가 제공하는 정보를 다른 에이전트들이 이미 모두 커버하고 있다는 의미다.

가장 높은 시너지: Noise ↔ FatFormer (+0.1093)

---

## 5. Cross-Dataset 검증: 실제로는 더 복잡하다

Phase 1.1~1.4 분석은 CASIA2 + GenImage BigGAN 데이터셋에서 수행했다. 이 결론이 다른 데이터셋에서도 동일하게 유지되는지 확인하기 위해 **4개 데이터셋 Cross-Dataset 검증**을 수행했다.

### 핵심 발견

- **SpatialAgent 제거**: 4/4 데이터셋에서 Unique information=0 확정 → 경량화에서 제외 결정
- **Frequency 1위 순위**: 1/4 데이터셋에서만 성립 (CASIA 편향)
- **FatFormer**: 일부 데이터셋에서 음수 Shapley → 특정 도메인에서 역효과 가능
- **결론**: "무거운 모델 기준 Shapley가 경량 모델 배포 시 보장되지 않는다"

이 발견은 중요한 방향 전환을 만들었다. 원래 모델 그대로 순위를 매기는 것이 아니라, **경량 백본으로 교체한 후에 다시 평가해야 한다**.

---

## 6. Phase 2: 경량 백본 후보 수급 및 평가

### 경량 백본 후보 선정

딥리서치(6개 PDF 분석) 결과 각 에이전트 슬롯에 대한 최유력 후보:

| 슬롯 | 현재 모델 | 경량 후보 | 근거 |
|------|---------|---------|------|
| FatFormer (~890MB) | CLIP ViT-L/14 + FAA | **MobileCLIP-S2** | FAA가 backbone-agnostic (ViT-B/16, Swin-B, Swin-L 모두 동작 확인) |
| Frequency (~150MB) | CAT-Net (HRNet-W48) | **ForMa** (VMamba 기반) | freq+noise 통합 가능 |
| Noise (~120MB) | MVSS-Net | **MobileNetV2 dual-stream** | RGB+SRM dual-stream, 5.77M |
| Spatial (~100MB) | Mesorch | **Tiny-LaDeDa** | Cascade Tier-1 스크리너 용도 |

### ForMa 평가 결과 (4개 데이터셋)

ForMa(37.3M, VMamba 기반)을 4개 데이터셋에서 단독 평가:
- Authentic recall: 0.837~0.937 (높음)
- Manipulated recall: 0.067~0.150 (매우 낮음)
- AI-gen recall: 0 (탐지 불가)
- **평균 accuracy: 0.335** (3-class에서 매우 낮음)

ForMa의 문제: 3-class 분류에서 단독으로는 쓸 수 없는 수준. 게다가 CPU 추론 시간이 **1,613ms**(RPi5에서는 더 느림)로 심각한 병목이 된다.

### MobileCLIP-S2 파인튜닝 결과

MobileCLIP-S2를 forensics 데이터셋에 파인튜닝했다. 두 단계로 진행:

**ft0 (Linear probe)**: CLIP 백본 완전 freeze, head만 학습
- val macro-recall: 0.790

**ft4 (Last 4 blocks FT)**: 마지막 4개 블록까지 unfreeze
- val macro: **0.806**
- 4개 데이터셋 평균 accuracy: **0.953** (base=0.942, dsC=0.974, opensdi=0.953, aigen=0.943)

크기: 99.4M params, GPU 15.5ms, CPU 123.8ms. 서버용으로는 적합하지만 RPi5에서는 너무 느리다.

### MobileNetV2 dual-stream 결과

RGB 이미지와 SRM(Spatial Rich Model) 잔차 노이즈를 동시에 처리하는 dual-stream 아키텍처:
- 5.77M params, GPU 18.9ms, **CPU 35.9ms** (RPi5 친화적)
- val macro: **0.806**
- 4개 데이터셋 avg: **0.958** (base=0.944, dsC=0.979, opensdi=0.949, aigen=0.961)

MNV2와 MobileCLIP이 거의 동등한 성능을 보이면서, MNV2는 훨씬 가볍고 CPU에서 빠르다.

### Tiny-LaDeDa 평가 결과

WildRF 데이터셋으로 학습된 Tiny-LaDeDa(0.0013M, 1,300 params):
- AI-generated recall: 73~86% (준수)
- Manipulated recall: **0%** (binary 모델이라 조작 탐지 불가)
- 용도: Cascade Tier-1 스크리너 전용

---

## 7. Phase 3.2: 경량 모델 Shapley+CKA 재분석

경량 백본 기반으로 Phase 1 분석을 재실행했다. 이 결과가 전체 연구 방향을 결정하는 핵심 발견이다.

### Shapley 결과 (경량 모델)

| 모델 | Shapley φ |
|------|-----------|
| MobileNetV2 dual-stream | +0.304 |
| MobileCLIP-ft4 | +0.300 |
| ForMa | +0.008 |
| Tiny-LaDeDa | +0.008 |

**MNV2와 MobileCLIP이 거의 동등하게 높고, ForMa와 Tiny-LaDeDa는 거의 기여가 없다.**

### CKA 결과 (경량 모델)

**CKA(Centered Kernel Alignment)**로 두 모델의 특징 유사도를 측정했다:

| 모델 쌍 | CKA |
|--------|-----|
| MobileCLIP ↔ MNV2 | **0.922** (매우 높은 중복!) |
| 나머지 모든 쌍 | < 0.02 (거의 독립) |

충격적인 발견: MobileCLIP과 MNV2가 특징 공간에서 92% 유사하다. 이것은 두 모델이 사실상 같은 정보를 처리하고 있다는 의미다. 따라서 함께 쓰면 정보가 중복될 가능성이 높다.

### PID: MobileCLIP ↔ MNV2 Redundancy = 0.599

PID 분석에서도 두 모델의 Redundancy(중복 정보)가 0.599로 매우 높았다. STII에서도 -0.584로 가장 강한 대체재 관계.

### 3-Track 비교 실험 결론

| Track | 구성 | 4-DS avg F1 |
|-------|------|-----------|
| Track 1 | ForMa + MNV2 + MobileCLIP | 0.9564 (in-dist) |
| Track 2 | ForMa + MobileCLIP (통합) | 낮음 |
| Track 3 | Tiny-LaDeDa(Tier1) + ForMa + MobileCLIP | 낮음 |

Fair LOO 재평가 (OOD):
- MobileCLIP 단독: avg F1 = 0.6386
- Track1 앙상블: avg F1 = **0.7309** (+0.092)

**ForMa를 전면 제거**했다. 이유: CPU 추론 1,613ms 병목 + Shapley ≈ 0.

**최종 결론**: RPi5 배포용 **2-모델 조합: MNV2 + MobileCLIP**가 최적. 단, MNV2와 MobileCLIP이 높은 CKA 중복을 가지므로, 이 중복을 줄이는 독립적인 **Binary Specialist** 에이전트를 추가하면 시스템을 더 강화할 수 있다. 이것이 Phase 3.5의 시작이다.
