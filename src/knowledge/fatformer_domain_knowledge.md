# FatFormer AI 생성 탐지 도메인 지식
**FatFormer: Forgery-Aware Transformer for AI-Generated Image Detection**

---

## 📚 과학적 근거

### 핵심 논문
1. **"FatFormer: A Frequency-Aware Transformer for Detecting AI-Generated Images" (Liu et al., CVPR 2024)**
   - CLIP ViT-L/14 백본에 Forgery-Aware Adapter 삽입
   - 공간적 의미론 + DWT 주파수 경로 이중 분석
   - Language-Guided Alignment으로 텍스트-이미지 대조 학습

2. **"Learning Transferable Visual Models From Natural Language Supervision" (Radford et al., 2021)**
   - CLIP의 대규모 이미지-텍스트 사전학습이 제공하는 범용 시각 표현
   - 다양한 도메인에 대한 강건한 전이 학습 능력

3. **"GenImage: A Large-Scale Benchmark for AI-Generated Image Detection" (Zhu et al., NeurIPS 2024)**
   - 8개 생성 모델에 대한 교차 평가 벤치마크
   - FatFormer가 교차 생성기 일반화에서 우수한 성능

---

## 🔬 분석 원리

### 1. CLIP 기반 의미론적 분석
```
이미지 → CLIP ViT-L/14 → 패치 토큰 (196개) + CLS 토큰
                         ↓
           의미론적 특징 (1024차원)
```

**원리**:
- CLIP의 대규모 사전학습은 "자연스러운 이미지"의 분포를 학습
- AI 생성 이미지는 의미론적 수준에서 미세한 불일치를 보임
- 텍스트-이미지 정렬을 통해 "real" vs "fake" 개념 학습

### 2. Forgery-Aware Adapter (FAA)
```
ViT Layer (7, 15, 23번에 삽입):
  ┌─ Spatial Pathway: Conv1d 병목 → 공간적 위조 패턴
  └─ Frequency Pathway: DWT → Intra/Inter-band Attention → IDWT
                                     ↓
              주파수 도메인 위조 아티팩트 탐지
```

**공간 경로 (Spatial Pathway)**:
- 1D 컨볼루션 병목 구조
- 로컬 패치 간 공간적 불일치 포착
- AI 생성 이미지의 텍스처 불연속성 탐지

**주파수 경로 (Frequency Pathway)**:
- DWT (Discrete Wavelet Transform)로 주파수 분해
  - LL (저주파): 전반적 구조
  - LH, HL, HH (고주파): 디테일, 엣지, 텍스처
- Intra-band Attention: 각 대역 내 이상 패턴 탐지
- Inter-band Attention: 대역 간 일관성 검증
- IDWT로 주파수 특징을 공간 특징에 재결합

### 3. Language-Guided Alignment (LGA)
```
학습 가능한 프롬프트 (8개 컨텍스트 토큰)
  + "real" / "fake" 클래스 이름
  → 텍스트 인코더 → 텍스트 특징
  ↕ (대조 학습)
이미지 특징 ← 이미지 인코더
```

**원리**:
- 학습 가능한 프롬프트가 위조 탐지에 최적화된 텍스트 표현 생성
- Patch-Based Enhancer: 패치별 이미지 특징으로 프롬프트 강화
- Text-Guided Interactor: 텍스트 특징으로 이미지 패치 해석 보강

---

## 📊 메트릭 해석 가이드

### AI 생성 확률 (fake_probability)
| 범위 | 해석 | 판정 |
|------|------|------|
| 0.0 – 0.3 | AI 생성 특징 미감지 | AUTHENTIC |
| 0.3 – 0.5 | 약간의 이상 신호, 불확실 | UNCERTAIN |
| 0.5 – 0.7 | AI 생성 가능성 높음 (경계) | AI_GENERATED (낮은 신뢰도) |
| 0.7 – 1.0 | AI 생성 확실 | AI_GENERATED (높은 신뢰도) |

### 신뢰도 해석
- **높은 신뢰도 (>0.7)**: CLIP 의미론과 DWT 주파수 모두 일치된 판단
- **중간 신뢰도 (0.5-0.7)**: 한쪽 경로에서만 강한 신호
- **낮은 신뢰도 (<0.5)**: 경계 사례, 다른 도구 교차 검증 필요

---

## 💡 해석 예시

### 예시 1: Diffusion 생성 이미지
```json
{
  "fake_probability": 0.92,
  "real_probability": 0.08,
  "model_backbone": "CLIP:ViT-L/14"
}
```
**해석**: CLIP 의미론적 분석에서 Diffusion 특유의 과도한 부드러움과
DWT 주파수 경로에서 고주파 에너지 감소 패턴이 명확히 감지됨.
AI 생성 이미지로 높은 신뢰도로 판정.

### 예시 2: GAN 생성 이미지 (JPEG 압축 후)
```json
{
  "fake_probability": 0.78,
  "real_probability": 0.22,
  "model_backbone": "CLIP:ViT-L/14"
}
```
**해석**: JPEG 압축으로 일부 고주파 아티팩트가 손실되었으나,
CLIP 의미론적 레벨에서 AI 생성 패턴이 유지됨.
FatFormer의 JPEG 강건성이 발휘된 사례.

### 예시 3: 실제 카메라 촬영 이미지
```json
{
  "fake_probability": 0.05,
  "real_probability": 0.95,
  "model_backbone": "CLIP:ViT-L/14"
}
```
**해석**: 자연스러운 센서 노이즈와 카메라 ISP 특성이 보존됨.
DWT 주파수 경로에서 자연 이미지의 1/f 노이즈 법칙과 일치.
원본 이미지로 높은 신뢰도로 판정.

---

## 🔍 특수 케이스

### 1. Inpainting / 부분 조작
- FatFormer는 전체 이미지 수준 분류기이므로 부분 조작 탐지에는 한계
- 조작 비율이 작으면 AUTHENTIC으로 판정될 수 있음
- → SpatialAgent의 픽셀 수준 마스크와 교차 검증 필요

### 2. 매우 작은 이미지 (<100px)
- 224x224로 업스케일링 시 보간 아티팩트 발생 가능
- 신뢰도를 낮게 보고하고 다른 분석 결과 참고

### 3. 스크린샷 / UI 이미지
- 렌더링된 텍스트, 아이콘 등은 AI 생성이 아니지만
  규칙적 패턴으로 인해 오탐 가능성 존재
- → 다른 에이전트 결과와의 교차 검증으로 보정

### 4. 학습 데이터에 없는 새로운 생성 모델
- CLIP 사전학습의 일반화 능력으로 일정 수준 대응 가능
- 단, 완전히 새로운 아키텍처의 경우 성능 저하 가능
