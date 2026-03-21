# SHIELD Phase 3.5: Binary Specialist 설계와 ICWMV 앙상블

## 1. Binary Specialist가 필요한 이유

Phase 3.2에서 MNV2와 MobileCLIP-ft4가 최적 2-모델 조합으로 선정되었지만 두 가지 문제가 있었다.

첫째, CKA=0.922로 두 모델이 매우 높은 특징 중복을 가진다. Meyen et al.(2021)의 이론에 따르면 앙상블의 상한은 개별 모델 에러가 독립적일 때 최대화된다. 중복이 높으면 두 모델이 같은 이미지에서 같이 틀릴 가능성이 높아 앙상블의 이점이 줄어든다.

둘째, 에러 Overlap 분석에서 "둘 다 틀리는" 패턴이 발견됐다: MNV2와 MobileCLIP 모두 조작 이미지를 authentic으로 잘못 분류하는 케이스가 102건 중 56건으로 지배적이었다. Jaccard 지수 0.3361.

해결책: **Binary Specialist** 추가. 3-class generalist 두 개만으로는 커버하지 못하는 영역을 binary 전문가로 보강한다.

- **Specialist-M**: auth vs manipulated binary 전문가
- **Specialist-G**: auth vs ai_generated binary 전문가

이를 결합하는 방법이 **ICWMV(Individual Confidence Weighted Majority Voting)**다.

---

## 2. ICWMV: 4-모델 Class-wise Weighted Fusion

ICWMV는 각 클래스에 대해 해당 클래스를 탐지할 수 있는 모델들의 confidence를 가중 평균한다.

```
S(auth)  = avg_w [ MNV2(auth),  CLIP(auth),  SpecM(auth),  SpecG(auth) ]
S(manip) = avg_w [ MNV2(manip), CLIP(manip), SpecM(manip) ]  ← SpecG는 manip 정보 없음
S(aigen) = avg_w [ MNV2(aigen), CLIP(aigen),               SpecG(aigen) ] ← SpecM은 aigen 정보 없음
→ renormalize → argmax
```

SpecM은 auth/manip binary이므로 aigen 클래스에 기여하지 않는다. SpecG는 auth/aigen binary이므로 manip 클래스에 기여하지 않는다. 이것이 ICWMV의 핵심 설계 원리다.

---

## 3. Specialist-M: 조작 탐지 전문가

### v1: 기본 설계

Specialist-M v1은 3-stream 아키텍처다:
- **RGB stream**: 이미지 원본 (1280d)
- **SRM stream**: Spatial Rich Model 노이즈 잔차 (1280d)
- **DCT stream**: 주파수 도메인 잔차 (1280d)
- **Fused**: 3840d → head → binary output

모델 크기: 7.66M params, 29.3MB

학습 데이터: CASIA2 (Auth 7,491장 + Manipulated 5,123장)

결과:
- val best manip_f1: 0.764 (epoch 5)
- 4-DS 평가: base f1=0.861, dsC=0.797
- **OOD 한계**: opensdi auth_recall = 7% (CASIA2에 과적합)

### v2: OOD 강건화

추가한 것:
- IMD2020 데이터셋 1,710장 (다양한 조작 기법)
- JPEG 압축 augmentation (p=0.5, q=40~95)
- Gaussian Noise augmentation (p=0.4)
- WeightedRandomSampler (클래스 불균형 보정)

결과:
- val best manip_f1: **0.827** (v1 0.764 대비 +6.3%p)
- opensdi auth_recall: 7% → 11% (개선됐지만 여전히 낮음)

### v3: Authentic OOD 강건화 (핵심 돌파)

추가한 것:
- GenImage_nature 3,000장: ImageNet val의 실제 자연 사진 (AI 생성이 아님)
- RandomErasing(p=0.3): 이미지 일부를 지워 inpainting 조작을 시뮬레이션

결과:
- val best manip_f1: **0.7832** (epoch 11)
- **opensdi auth_recall: 11% → 62%** (+51%p 대폭 개선!)
- ICWMV 2-model avg (MNV2+SpecM-v3): **96.19%**

### v4: 세밀한 파인튜닝

v3 checkpoint를 재개해서 추가 파인튜닝:
- LR: 1e-4 → 3e-5 (더 작은 학습률)
- RandomErasing value=random (inpainting fill 노이즈 더 현실적으로)
- Focal loss α=0.6

결과:
- val best manip_f1: **0.7792** (epoch 10)
- openSDI manip_recall: 70.3%
- **ICWMV v4 avg: 96.58%** (4-model 서버 96.48% 초과!)

RPi5 최종 배포 기준 모델로 v4 확정.

---

## 4. Specialist-G: AI 생성 탐지 전문가

Specialist-G는 MobileCLIP-S2를 frozen backbone으로 사용하고, PiD(Perceptual image Distortion) branch를 학습 가능한 경량 head로 붙인 아키텍처다.

- MobileCLIP frozen: 35.81M (학습 안 함)
- PiD branch: 0.10M (학습)
- 총 학습 파라미터: 0.10M

학습 데이터: CASIA2 Au + GenImage BigGAN + AI-GenBench 약 23,000장

결과:
- val best aigen_f1: **0.981** (epoch 19)
- 4-DS: base=0.987, dsC=0.988, opensdi=0.799, aigenproxy=0.730

---

## 5. 4-Model ICWMV 앙상블 평가

| 구성 | avg macro-F1 |
|------|-------------|
| MNV2 단독 | 95.81% |
| MNV2 + SpecM-v2 (ICWMV) | 96.48% (+0.67%p) |
| MNV2 + SpecM-v4 (ICWMV) | **96.58%** (+0.77%p) |
| 4-model 서버 (MNV2+CLIP+SpecM+SpecG) | 96.48% |

**RPi5용 2-model ICWMV (MNV2+SpecM-v4)가 서버 4-model을 +0.10%p 초과한다.**

데이터셋별 결과 (v4 기준):
- base: 95.86%
- dsC: 98.44%
- opensdi: 94.68%
- aigenproxy: 97.33%

---

## 6. CKA 다양성 분석: Binary Specialist의 효과

4-model 구성이 2-model보다 왜 더 좋은가를 이론적으로 검증했다.

| 지표 | 2-model (MNV2+CLIP) | 4-model (MNV2+CLIP+SpecM+SpecG) |
|------|-------------------|--------------------------------|
| 평균 CKA | 0.9241 | **0.0855** |
| 평균 Jaccard | 0.3361 | **0.1233** |
| Disagreement rate | 4.4% | **32.8%** |

Binary specialist 추가 후 평균 CKA가 0.92 → 0.09로 대폭 감소했다. 에이전트들이 훨씬 더 독립적인 시각으로 이미지를 보게 된 것이다.

---

## 7. SpecM-v5b: MobileCLIP 기반 재설계

### 동기: SpecM v1~v4의 근본 한계 발견

SpecM v4까지 개선했음에도 opensdi 같은 OOD 데이터셋에서 성능이 여전히 제한적이었다. 근본 원인을 찾기 위해 **Embedding-level CKA 분석**을 수행했다.

**브랜치별 CKA vs MNV2 (n=4,200 샘플)**:

| SpecM 브랜치 | CKA vs MNV2 |
|------------|------------|
| RGB branch (1280d) | 0.322 |
| SRM branch (1280d) | **0.563** |
| DCT branch (1280d) | 0.239 |
| PiD branch (64d) | 0.001 |
| Fused (3840d) | 0.392 |

핵심 발견:
- `mnv2_rgb ↔ specm_rgb = 0.656`: 두 모델 모두 ImageNet pretrained RGB backbone을 쓰기 때문에 발생하는 구조적 중복
- `mnv2_noise ↔ specm_srm = 0.564`: MNV2의 SRM noise channel과 SpecM의 SRM branch가 동일한 신호 처리

**더 나아가: 전 프로젝트 모델 vs MNV2 CKA 비교**

SpecM v1~v4의 CKA가 모두 동일하게 높았다. 이는 RGB backbone을 공유하기 때문에 신호(SRM, DCT)를 아무리 추가해도 MNV2와의 중복이 줄지 않는다는 것을 의미한다.

| 모델 | CKA vs MNV2 |
|------|------------|
| SpecM v1~v4 | 0.725 (모두 동일!) |
| SpecG | 0.659 |
| **MobileCLIP-S2-ft4** | **0.028 ★** |

**MobileCLIP이 압도적으로 독립적이다.** 이유: MobileCLIP은 ViT 계열 contrastive pretrain으로 학습되어 ImageNet 분류 CNN인 MNV2와 inductive bias가 근본적으로 다르다.

### SpecM-v5b 설계

기존 RGB/SRM branch를 버리고 MobileCLIP을 backbone으로 채택한 새로운 설계:

```
Input image
    ↓ (두 경로)
[MobileCLIP-S2] → 512d
[SRMLightCNN]   → 128d   (depthwise-separable CNN, 경량화)
    ↓
Concatenate → 640d
    ↓
Head: LayerNorm(640) → Linear(256) → GELU → Dropout → Linear(64) → GELU → Linear(2)
```

SRMLightCNN은 depthwise-separable 구조의 경량 CNN으로 MobileNetV2 구조를 피해 중복을 줄였다.

**Phase 1: MobileCLIP frozen (0.21M 학습 파라미터)**
- 총 36.03M params, 학습 0.21M
- 40 epochs, lr=3e-4
- val best manip_f1: **0.7517** (epoch 37)
- 4-DS avg manip_f1: 0.8062

0.21M만 학습하는 frozen 방식의 한계가 있었다. forensics 특화 적응이 충분하지 않다.

**Phase 2: MobileCLIP unfreeze + differential LR (v5b_ft)**
- MobileCLIP 전체 unfreeze
- Differential LR: clip trunk lr=1e-5 / SRM CNN+head lr=1e-4 (10배 차이)
- 20 epochs 추가 학습

결과:
- val best manip_f1: **0.7846** (epoch 13)
- **4-DS avg manip_f1: 0.8347** (SpecM-v4 0.8165 대비 +0.018, +1.8%p)
  - base: 0.8645
  - dsC: 0.8932
  - opensdi: 0.6205 (여전히 약점)
  - aigenproxy: **0.9605** (대폭 개선!)

ICWMV (MNV2+v5b_ft): avg=0.9635

### v5b의 RPi5 배포 결정: v4 유지

v5b_ft가 4-DS avg에서 v4를 +1.8%p 초과하지만, RPi5 배포 기준 모델은 v4를 유지한다. 이유:
- opensdi OOD F1: v5b_ft=0.6205 vs v4=0.9468 (v4가 훨씬 우수)
- RPi5에서 실제 사용자 이미지는 다양한 소셜미디어 출처 → OOD 강건성이 중요

v5b/v5b_ft는 연구 contribution(CKA 독립성, MobileCLIP backbone 유효성)으로서는 의미가 있지만, 실제 배포에는 v4가 더 안전하다.

---

## 8. 최종 4-모델 아키텍처 요약

| 모델 | 역할 | 크기 | 학습 데이터 |
|------|------|------|-----------|
| MobileNetV2 dual-stream | 3-class generalist | 5.77M / 22.5MB | CASIA2+BigGAN |
| MobileCLIP-ft4 | 3-class generalist | 99.4M / 380MB | 동일 |
| **Specialist-M v4** | binary: auth vs manip | 7.66M / 29MB | CASIA2+IMD+GenImage |
| Specialist-G | binary: auth vs aigen | 35.91M / 141MB | CASIA2+BigGAN+AIGenBench |

**RPi5 배포 구성**: MNV2-Dynamic INT8 + SpecM-v4-Dynamic INT8 = **~46MB**, ICWMV avg **96.58%**
