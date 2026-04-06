# 노이즈 분석 도메인 지식
**Noise Analysis — MVSS-Net 백엔드 (PRNU/SRM fallback)**

> ⚠️ **백엔드 구성**: 실험/운영 환경에서는 MVSS-Net 딥러닝 모델(backend=mvss)을 기본 사용. 체크포인트 부재 시 PRNU/SRM 경로로 자동 전환. **두 백엔드 모두 AI-generated 탐지 능력 없음(F1=0.000).**

---

## 📚 과학적 근거

### 핵심 논문
1. **"MVSS-Net: Multi-View Multi-Scale Supervised Networks for Image Manipulation Detection" (Chen et al., ICCV 2021)**
   - 픽셀 수준 조작 마스크 예측 딥러닝 모델
   - 노이즈 뷰 + 분할 가이드 뷰 결합

2. **"Digital camera identification from sensor pattern noise" (Lukas et al., 2006)**
   - PRNU는 카메라 센서의 고유한 "지문" (PRNU fallback 경로)

3. **"Camera model identification using sensor noise" (Chen et al., 2008)**
   - PRNU의 주파수 특성

---

## 🔬 분석 원리

### 1. MVSS-Net (기본 백엔드)
```
입력 이미지 → Noise View (SRM 필터) + Segmentation View →
Multi-Scale 피처 통합 → 픽셀 수준 조작 마스크 예측
```

**동작 원리**:
- SRM(Spatial Rich Model) 필터로 노이즈 잔여물 추출
- 딥러닝으로 조작 영역과 원본 영역의 노이즈 분포 차이를 학습
- 출력: 0~1 범위의 조작 확률 마스크
- `mvss_threshold`: 마스크 이진화 임계값 (기본 0.655)

**판정 로직**:
- 조작 픽셀 비율 < auth_threshold → AUTHENTIC
- 조작 픽셀 비율 > manipulated_threshold → MANIPULATED
- 그 외 → UNCERTAIN

**한계**: AI 생성 이미지(전체 생성)는 "조작"이 없으므로 탐지 불가 → **AI-generated F1=0.000 맹점**

### 2. PRNU란? (Fallback 백엔드)
```
카메라 센서의 제조 과정:
  실리콘 웨이퍼 → 포토다이오드 어레이 → 미세한 불균일성
```

**물리적 원인**:
- 각 픽셀의 감광 민감도가 미세하게 다름
- 제조 과정의 불완전성 (균일하게 만들 수 없음)
- 이 패턴은 **카메라마다 고유함** (지문과 같음)

**수학적 모델**:
```
I(x, y) = I₀(x, y) × [1 + K(x, y)] + Θ(x, y)

I: 관측 이미지
I₀: 실제 장면
K: PRNU 패턴 (카메라 고유)
Θ: 기타 노이즈 (shot noise, read noise)
```

### 2. PRNU 추출 방법
```
이미지 → Denoising Filter → 노이즈 잔여물 → PRNU 패턴
```

**사용 필터**:
- Wiener Filter (전통적 방법)
- BM3D (고급 denoising)
- DnCNN (딥러닝 기반)

**추출된 PRNU의 특징**:
- 고주파 성분 (카메라 센서의 미세 패턴)
- 이미지 내용과 독립적
- 동일 카메라로 찍은 모든 사진에 공통

---

## 📊 메트릭 해석 가이드

> ⚠️ **백엔드에 따라 출력 키가 다릅니다**: MVSS-Net 백엔드(기본)와 PRNU/SRM 폴백은 서로 다른 evidence 키를 사용합니다. `evidence["backend"]`로 확인하세요.

### 🔷 MVSS-Net 백엔드 메트릭 (기본)

#### 1. mvss_score (MVSS 조작 점수)
**측정 방법**: 조작 예측 마스크의 최대값

| 범위 | 의미 |
|------|------|
| > `mvss_threshold` (기본 0.655) | MANIPULATED 판정 |
| < auth threshold | AUTHENTIC 판정 |
| 중간 구간 | UNCERTAIN 판정 |

#### 2. manipulation_ratio (조작 픽셀 비율)
**측정 방법**: 이진화된 마스크에서 양성 픽셀 비율

| 범위 | 의미 |
|------|------|
| 높음 | 광범위한 조작 영역 |
| 낮음 | 국소적 조작 또는 조작 없음 |

#### 3. mask_mean / mask_max
- `mask_mean`: 조작 마스크 평균값 (0~1)
- `mask_max`: 조작 마스크 최대값 (판정 기준)

---

### 🔷 PRNU/SRM 폴백 백엔드 메트릭

### 1. prnu_consistency (PRNU 일관성)
**측정 방법**: 이미지 전체에서 PRNU 패턴의 일관성 측정

| 범위 | 의미 | 과학적 근거 |
|------|------|------------|
| 0.8 - 1.0 | 매우 일관적 | 실제 카메라 촬영. PRNU 패턴 명확 |
| 0.5 - 0.8 | 중간 일관성 | 압축/편집된 실제 사진 또는 혼합 이미지 |
| 0.0 - 0.5 | 일관성 없음 | AI 생성 또는 심하게 조작된 이미지 |

**AI 생성 이미지의 특징**:
- GAN/Diffusion은 PRNU를 "학습"하지 않음
- 훈련 데이터의 PRNU는 모두 다르므로 학습 불가
- 결과: PRNU 패턴이 없거나 무작위

### 2. noise_pattern_presence (노이즈 패턴 존재 여부)
**측정 방법**: 카메라 센서 노이즈 패턴 탐지

**의미**:
- `True`: 카메라 고유 노이즈 검출
- `False`: 노이즈 없음 또는 인위적 노이즈

**판별 기준**:
```python
if correlation(extracted_noise, natural_prnu_template) > threshold:
    → AUTHENTIC (실제 카메라)
else:
    → AI_GENERATED or MANIPULATED
```

### 3. sensor_fingerprint_match (센서 지문 매칭)
**측정 방법**: 알려진 카메라 모델의 PRNU와 비교

| 값 | 의미 | 설명 |
|-----|------|------|
| 0.9 - 1.0 | 특정 카메라 일치 | 카메라 모델 식별 가능 |
| 0.6 - 0.9 | 유사 카메라 | 같은 제조사/모델 계열 |
| < 0.6 | 매칭 실패 | AI 생성 또는 알려지지 않은 카메라 |

**주의사항**:
- 새로운 카메라 모델은 데이터베이스에 없을 수 있음
- "매칭 실패 ≠ 무조건 AI 생성"

---

## ⚖️ 분석의 강점과 한계

### 강점
✅ **물리적 근거**
   - PRNU는 카메라 하드웨어의 물리적 특성
   - 소프트웨어로 위조 매우 어려움

✅ **픽셀 수준 조작 탐지** (MVSS-Net)
   - 단순 분류가 아닌 마스크 출력 → 어느 영역이 조작됐는지 확인 가능
   - CASIA2 기준 F1 ≈ 0.76 달성

✅ **물리적 근거** (PRNU fallback)
   - PRNU는 카메라 하드웨어의 물리적 특성, 소프트웨어로 위조 어려움
   - PRNU 있음 → 실제 카메라 촬영 확실

### 한계
❌ **AI-generated 이미지 탐지 불가 (F1=0.000)**
   - 두 백엔드 모두 해당: AI 생성 이미지는 픽셀 수준 "조작"이 없음
   - **FatFormerAgent가 AI-generated 탐지를 전담해야 함**

❌ **압축/편집에 취약** (특히 PRNU)
   - JPEG 고압축 → PRNU 약화
   - 강한 필터링 → 패턴 손상

---

## 🤝 다른 분석과의 관계

### vs Frequency Analysis (CAT-Net)
**보완 관계**:
- Frequency (CAT-Net): JPEG 이중 압축 흔적 탐지 (스플라이싱)
- Noise (MVSS-Net): 픽셀 수준 조작 마스크 예측

**상충 시 해석**:
```
Frequency: MANIPULATED (JPEG 이중 압축 흔적 탐지)
Noise: AUTHENTIC (MVSS 조작 마스크 없음)

→ 해석: 압축 이력 불일치이나 픽셀 수준 흔적이 약한 경우
→ 추천: 두 에이전트 모두 Manipulated 탐지 전문이므로 일치 시 높은 신뢰도
```

### vs FatFormer (CLIP+DWT)
**역할 분담**:
- NoiseAgent: 조작(MANIPULATED) 탐지 전담, AI-generated 맹점
- FatFormerAgent: AI-generated 탐지 전담, Manipulated 맹점
- DAAC에서 두 에이전트 불일치(`disagree_noise_fatformer`)는 중요 탐지 신호(GBM 중요도 8.4%)

### vs Spatial Analysis
**보완 관계**:
- Noise: 전역 일관성 (global consistency)
- Spatial: 지역 불일치 (local inconsistency)

**조작 탐지 시**:
```
1. MVSS-Net으로 조작 마스크 예측 → 조작 여부 판단
2. Spatial로 조작 위치 상세 특정
```

---

## 💡 해석 예시

### 🔷 MVSS-Net 백엔드 예시 (기본)

### Case 1: 조작 이미지 (MVSS 탐지)
```
backend: mvss
mvss_score: 0.82
manipulation_ratio: 0.073
mask_mean: 0.041
verdict: MANIPULATED
```

**해석**:
"MVSS-Net 조작 마스크에서 전체 픽셀의 7.3%가 조작된 것으로 탐지되었습니다.
조작 점수(0.82)가 임계값(0.655)을 명확히 초과하여 MANIPULATED로 판정됩니다.
SpatialAgent와 함께 조작 영역의 위치를 특정하세요."

### Case 2: 원본 이미지 (MVSS 탐지)
```
backend: mvss
mvss_score: 0.21
manipulation_ratio: 0.002
mask_mean: 0.008
verdict: AUTHENTIC
```

**해석**:
"MVSS-Net 조작 마스크에서 유의미한 조작 패턴이 탐지되지 않았습니다.
조작 점수(0.21)가 임계값을 하회하여 AUTHENTIC으로 판정됩니다.
⚠️ MVSS-Net은 AI 생성 이미지(전체 생성)에 대한 탐지 능력이 없습니다 — FatFormerAgent 결과를 참조하세요."

### Case 3: AI 생성 이미지 (MVSS 탐지 실패)
```
backend: mvss
mvss_score: 0.18
manipulation_ratio: 0.001
verdict: AUTHENTIC  ← 오판!
```

**해석**:
"MVSS-Net은 전체 AI 생성 이미지를 AUTHENTIC으로 오판합니다.
⚠️ AI 생성 이미지는 픽셀 수준 '조작'이 없으므로 MVSS-Net의 탐지 대상이 아닙니다.
**이 경우 FatFormerAgent 판정을 우선 참조하세요.**"

---

### 🔷 PRNU/SRM 폴백 백엔드 예시

### Case 4: 실제 카메라 촬영 (PRNU fallback)
```
backend: prnu
prnu_consistency: 0.87
noise_pattern_presence: True
sensor_fingerprint_match: 0.92
```

**해석**:
"명확한 PRNU 패턴이 검출되었습니다(일관성 0.87).
카메라 센서의 고유한 노이즈 지문이 이미지 전체에서 일관되게 나타납니다.
이는 실제 카메라로 촬영된 원본 이미지임을 강력히 시사합니다."

### Case 5: AI 생성 이미지 (PRNU fallback)
```
backend: prnu
prnu_consistency: 0.12
noise_pattern_presence: False
sensor_fingerprint_match: 0.05
```

**해석**:
"PRNU 패턴이 거의 검출되지 않았습니다(일관성 0.12).
카메라 센서의 고유 노이즈가 존재하지 않습니다.
AI 생성 모델은 물리적 카메라 센서가 없으므로 PRNU를 생성할 수 없습니다."

---

## 🔍 특수 케이스

### 1. 스마트폰 카메라
**특징**:
- 강한 후처리 (HDR, AI 필터)
- PRNU가 약화될 수 있음
- 하지만 완전히 사라지지는 않음

**판단**:
- prnu_consistency < 0.7이어도 AUTHENTIC 가능
- noise_pattern_presence가 더 중요

### 2. 스크린샷
**특징**:
- PRNU 없음 (카메라로 찍지 않음)
- 하지만 AI 생성도 아님

**판단**:
- PRNU만으로는 판단 불가
- 다른 증거 필요 (워터마크, 주파수 등)

### 3. RAW vs JPEG
**RAW**:
- PRNU 최대한 보존
- 가장 정확한 분석 가능

**JPEG**:
- 압축으로 PRNU 약화
- 하지만 여전히 탐지 가능 (완전 소멸은 아님)

---

## 📖 참고문헌

1. Chen et al., "MVSS-Net: Multi-View Multi-Scale Supervised Networks for Image Manipulation Detection", ICCV 2021
2. Lukas et al., "Digital Camera Identification from Sensor Pattern Noise", IEEE TIFS 2006
3. Chen et al., "Determining Image Origin and Integrity Using Sensor Noise", IEEE TIFS 2008
4. Fridrich & Kodovsky, "Rich Models for Steganalysis of Digital Images", IEEE TIFS 2012

---

**최종 업데이트**: 2026-03-08 (MVSS-Net 기본 백엔드 메트릭 섹션 추가, 해석 예시 MVSS/PRNU 분리, Frequency 비교 CAT-Net 반영)
