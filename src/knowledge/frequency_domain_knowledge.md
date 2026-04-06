# 주파수 분석 도메인 지식
**Frequency Domain Analysis — JPEG 압축 흔적 탐지 (CAT-Net 백엔드)**

> ⚠️ **현재 백엔드**: CAT-Net (JPEG 이중 압축 탐지 전문). FFT 기반 GAN 아티팩트 탐지는 fallback 경로이며 기본 동작이 아님.

---

## 📚 과학적 근거

### 핵심 논문
1. **"CAT-Net: Compression Artifact Tracing Network" (Kwon et al., IEEE TIFS 2021)**
   - JPEG 이중 압축 흔적을 DCT 도메인에서 직접 학습
   - HRNet 백본 + DCT/JPEG 사전학습 가중치 활용
   - 스플라이싱(splicing) 조작 탐지에 특화

2. **"Rich Models for Steganalysis of Digital Images" (Fridrich & Kodovsky, IEEE TIFS 2012)**
   - SRM 필터 기반 압축 아티팩트 특성화

---

## 🔬 분석 원리

### 1. JPEG 이중 압축 (Double JPEG Compression)
```
원본 이미지 → 1차 JPEG 저장 → 조작(스플라이싱) → 2차 JPEG 저장
```

**핵심 원리**:
- JPEG는 8×8 블록 단위 DCT 변환으로 압축
- 이미지를 편집하면 조작된 영역과 원본 영역의 압축 히스토리가 불일치
- 이중 압축 흔적(double quantization artifact)이 픽셀 수준 불연속으로 나타남

### 2. CAT-Net의 탐지 방식
```
입력 이미지 → DCT 계수 추출 → HRNet 피처 추출 →
이중 압축 마스크 예측 → 조작 영역 확률 맵
```

- `mask_threshold`: 마스크 이진화 임계값 (기본 0.35)
- `authentic_ratio_threshold`: 조작 픽셀 비율이 이 이하면 AUTHENTIC 판정 (기본 0.0048)
- `manipulated_ratio_threshold`: 조작 픽셀 비율이 이 이상이면 MANIPULATED 판정 (기본 0.0048)

---

## 📊 메트릭 해석 가이드

### 1. compression_score (압축 이상 점수)
**측정 방법**: 조작 예측 마스크에서 양성 픽셀 비율

| 범위 | 의미 |
|------|------|
| 높음 (> manipulated_ratio_threshold) | JPEG 이중 압축 흔적 탐지 → MANIPULATED 판정 |
| 낮음 (< authentic_ratio_threshold) | 단일 압축 이력 → AUTHENTIC 판정 |
| 중간 | UNCERTAIN 판정 |

### 2. 판정 로직 요약
- JPEG 스플라이싱된 조작 이미지 탐지에 특화
- AI-generated 이미지에 대해서는 **탐지 능력 없음 (F1=0.000)**
  - AI 이미지는 카메라 JPEG 이력이 없어 CAT-Net의 가정이 성립하지 않음

---

## ⚖️ 분석의 강점과 한계

### 강점
✅ **JPEG 스플라이싱 탐지에 특화**
   - CASIA2 기준 F1 ≈ 0.70 달성
   - DCT 도메인 직접 학습으로 압축 아티팩트 정밀 분석

✅ **픽셀 수준 조작 영역 예측**
   - 단순 분류가 아닌 마스크 출력 → 어느 영역이 조작되었는지 보조 정보 제공

### 한계
❌ **AI-generated 이미지 탐지 불가 (F1=0.000)**
   - AI 이미지는 카메라 촬영 → JPEG 압축 이력이 없음
   - CAT-Net의 "이중 압축" 가정이 AI 이미지에 적용되지 않음
   - **FatFormerAgent가 AI-generated 탐지를 담당**

❌ **PNG 원본 이미지에 취약**
   - JPEG 압축 이력이 없는 PNG 이미지에서 false positive 발생 가능

❌ **고배율 리사이징 후 흔적 소멸**
   - 극단적인 리사이징 → 압축 블록 경계 왜곡 → 탐지 어려움

---

## 🤝 다른 분석과의 관계

### vs FatFormer (CLIP+DWT)
- **상호 보완 관계** (DAAC의 핵심):
  - FrequencyAgent: Manipulated(JPEG 조작) 탐지 전문, AI-generated 맹점
  - FatFormerAgent: AI-generated 탐지 전문, Manipulated 맹점
  - 두 에이전트의 불일치(`disagree_frequency_fatformer`)가 DAAC 최상위 특징(56.5%)

### vs Noise Analysis (MVSS-Net/PRNU)
- **보완 관계**: 압축 흔적 vs 픽셀 잡음 불일치
  - 두 에이전트 모두 Manipulated 탐지 전문이며 AI-generated 맹점 공유
  - 의견 일치 시 신뢰도 상승

### vs Spatial Analysis (Mesorch)
- **보완 관계**:
  - 주파수(압축 이력): 전역 JPEG 블록 불일치 탐지
  - 공간(Mesorch): 지역적 픽셀 불연속 탐지
  - 함께 MANIPULATED 판정 시 높은 신뢰도

---

## 💡 해석 예시

### Case 1: JPEG 스플라이싱 탐지
```
compression_score: 0.023 (> manipulated_ratio_threshold 0.0048)
verdict: MANIPULATED
confidence: 0.85
```

**해석**:
"이미지 2.3%의 픽셀에서 이중 JPEG 압축 흔적이 탐지되었습니다.
조작된 영역(스플라이싱)이 다른 압축 이력을 가짐을 의미합니다.
다른 에이전트와 함께 종합 판정이 필요합니다."

### Case 2: AI 생성 이미지 (탐지 실패)
```
compression_score: 0.002 (< authentic_ratio_threshold)
verdict: AUTHENTIC
confidence: 0.60
```

**해석**:
"JPEG 압축 흔적이 없어 AUTHENTIC으로 판정되었습니다.
⚠️ 그러나 AI 생성 이미지도 압축 이력이 없으면 동일 결과를 낳습니다.
FatFormerAgent 결과를 우선 참조하세요."

### Case 3: 정상 사진
```
compression_score: 0.001
verdict: AUTHENTIC
confidence: 0.75
```

**해석**:
"단일 JPEG 압축 이력, 조작 흔적 없음. 자연 촬영 이미지로 판단됩니다."

---

## ⚠️ Fallback 모드 (FFT 기반)

CAT-Net 체크포인트 미로드 시 `frequency_tool.py`의 FFT 경로로 자동 전환됨.
FFT 모드는 GAN upsampling 격자 아티팩트를 탐지하나, 현재 운영 환경에서 기본 경로가 아님.

---

## 📖 참고문헌

1. Kwon et al., "CAT-Net: Compression Artifact Tracing Network for Detection and Localization of Image Splicing", IEEE TIFS 2021
2. Fridrich & Kodovsky, "Rich Models for Steganalysis of Digital Images", IEEE TIFS 2012
3. Wang et al., "CNN-generated images are surprisingly easy to spot... for now", CVPR 2020 (FFT fallback 참고)

---

**최종 업데이트**: 2026-03-07 (CAT-Net 백엔드 반영)
