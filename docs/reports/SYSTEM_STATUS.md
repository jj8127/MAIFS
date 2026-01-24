# MAIFS 시스템 상태

**마지막 업데이트**: 2026-01-21
**전체 시스템 상태**: ✅ 준비 완료

---

## 🎯 현재 상태

### 핵심 구성 요소
| 구성요소 | 상태 | 비고 |
|---------|------|------|
| 4개 전문가 에이전트 | ✅ | Frequency, Noise, Watermark, Spatial |
| COBRA 합의 알고리즘 | ✅ | RoT, DRWA, AVGA 모두 작동 |
| 토론 시스템 | ✅ | 동기, 비동기, 구조화 프로토콜 |
| OmniGuard 체크포인트 | ✅ | 5개 파일 다운로드 완료 |
| 테스트 커버리지 | ✅ | 94/94 테스트 통과 |

---

## 📊 테스트 결과

### 전체 테스트 통과
```
test_tools.py .......................... 21/21 ✅
test_cobra.py .......................... 18/18 ✅
test_debate.py ......................... 19/19 ✅
test_e2e.py ............................ 21/21 ✅
test_checkpoint_loading.py ............. 15/15 ✅
─────────────────────────────────────────────────
합계 .................................. 94/94 ✅
```

### 주요 기능 검증
- [x] 개별 Tool 분석 (Frequency, Noise, Watermark, Spatial)
- [x] COBRA 합의 알고리즘 (RoT, DRWA, AVGA)
- [x] Debate 프로토콜 (Sync, Async, Structured)
- [x] End-to-End MAIFS 파이프라인
- [x] 체크포인트 로드 및 검증
- [x] 에러 처리 및 Fallback 모드

---

## 🔧 설정 및 경로

### OmniGuard 경로
```
현재 위치: OmniGuard-main/
체크포인트: OmniGuard-main/checkpoint/
```

### 설정 파일
```
메인 설정: configs/settings.py
- OS 자동 감지 (Windows/Linux/macOS)
- 동적 체크포인트 경로
- 장치 자동 선택 (GPU/CPU)
```

---

## 📝 현재 기능

### MAIFS 코어 기능
1. **이미지 분석**
   - 다양한 형식 지원 (NumPy, PIL, 파일 경로)
   - 자동 전처리 (그레이스케일 → RGB 변환)

2. **전문가 에이전트**
   - Frequency Agent: FFT 기반 주파수 분석
   - Noise Agent: PRNU 노이즈 분석
   - Watermark Agent: 워터마크 탐지
   - Spatial Agent: 공간 조작 탐지

3. **합의 엔진**
   - RoT (Root-of-Trust): 신뢰 기반 결정
   - DRWA: 동적 가중치 조정
   - AVGA: 어텐션 기반 집계

4. **토론 시스템**
   - 자동 의견 불일치 감지
   - 라운드 기반 토론
   - 수렴 조건 확인

5. **결과 처리**
   - JSON 직렬화
   - Dict 변환
   - 상세 설명 생성

---

## 🚀 사용 방법

### 기본 사용법
```python
from src.maifs import MAIFS

# MAIFS 초기화
maifs = MAIFS(enable_debate=True, consensus_algorithm="drwa")

# 이미지 분석
result = maifs.analyze("path/to/image.jpg")

# 결과 확인
print(result.verdict)          # Verdict (AUTHENTIC, AI_GENERATED, etc)
print(result.confidence)       # 신뢰도 (0.0-1.0)
print(result.to_json())       # JSON 형식
```

### 커스텀 설정
```python
# 합의 알고리즘 선택
maifs = MAIFS(consensus_algorithm="rot")  # RoT 사용

# 토론 비활성화
maifs = MAIFS(enable_debate=False)

# 토론 임계값 조정
maifs = MAIFS(debate_threshold=0.2)
```

---

## 🔬 테스트 실행

### 모든 테스트 실행
```bash
cd /path/to/MAIFS
python -m pytest tests/ -v
```

### 특정 테스트 실행
```bash
# 체크포인트 테스트만
python -m pytest tests/test_checkpoint_loading.py -v

# 특정 클래스
python -m pytest tests/test_checkpoint_loading.py::TestCheckpointAvailability -v

# 특정 테스트
python -m pytest tests/test_e2e.py::TestMAIFSAnalysis::test_analyze_numpy_array -v
```

---

## 📁 주요 파일 구조

```
MAIFS/
├── configs/
│   └── settings.py              # 중앙 설정 (경로, 모델, 알고리즘)
├── src/
│   ├── maifs.py                 # MAIFS 메인 시스템
│   ├── tools/
│   │   ├── base_tool.py
│   │   ├── frequency_tool.py
│   │   ├── noise_tool.py
│   │   ├── watermark_tool.py
│   │   └── spatial_tool.py
│   ├── agents/
│   │   └── base_agent.py
│   ├── consensus/
│   │   └── cobra.py             # RoT, DRWA, AVGA 알고리즘
│   └── debate/
│       ├── protocols.py          # 토론 프로토콜
│       └── debate_chamber.py     # 토론 시스템
├── tests/
│   ├── test_tools.py            # 21 테스트
│   ├── test_cobra.py            # 18 테스트
│   ├── test_debate.py           # 19 테스트
│   ├── test_e2e.py              # 21 테스트
│   └── test_checkpoint_loading.py # 15 테스트
└── OmniGuard-main/
    └── checkpoint/              # 다운로드된 체크포인트 (1.6 GB)
```

---

## ⚙️ 설정 옵션

### ModelConfig
```python
# 체크포인트 경로
omniguard_checkpoint_dir: Path
hinet_checkpoint: Path
vit_checkpoint: Path
unet_checkpoint: Path

# 모델 파라미터
vit_input_size: int = 1024
vit_patch_size: int = 16

# 디바이스
device: str = "cuda" or "cpu"
```

### COBRAConfig
```python
# 신뢰도 설정
trust_threshold: float = 0.7
initial_trust: Dict[str, float]

# 알고리즘 선택
consensus_algorithm: str = "drwa"

# 알고리즘 파라미터
drwa_epsilon: float = 0.1
avga_temperature: float = 1.0
rot_alpha: float = 0.3
```

### DebateConfig
```python
# 토론 활성화
enable_debate: bool = True

# 토론 조건
disagreement_threshold: float = 0.3
max_rounds: int = 3

# 모드 선택
debate_mode: str = "asynchronous"
```

---

## 🔮 다음 단계

### Phase 1: LLM 통합 (예정)
- Manager Agent 구현
- Claude API 또는 로컬 LLM 연동
- 자동 분석 리포트 생성

### Phase 2: 성능 최적화 (예정)
- 모델 가중치 로드 검증
- 추론 속도 최적화
- 메모리 사용량 감소

### Phase 3: 확장 (예정)
- 웹 API 개발
- 웹 UI 개발
- 배치 처리 지원

---

## 🐛 알려진 제한사항

1. **모델 로드 상태**: Fallback 모드 작동 중
   - WatermarkTool과 SpatialTool은 현재 규칙 기반 분석 사용
   - 실제 모델 가중치 로드는 추후 개선 필요

2. **HiNet/ViT 모델**: 형식 호환성 미확인
   - checkpoint-175.pth가 HiNet 모델을 포함하는지 확인 필요
   - model_checkpoint_*.pt가 ViT 모델인지 확인 필요

3. **LLM 통합**: 미구현
   - 현재는 규칙 기반 분석만 사용
   - Manager Agent 미구현

---

## 📞 문제 해결

### 체크포인트를 찾을 수 없음
```
해결: configs/settings.py에서 OmniGuard 경로 확인
python -c "from configs.settings import config; print(config.model.omniguard_checkpoint_dir)"
```

### 테스트 실패
```
해결: pytest 로그 확인
python -m pytest tests/test_*.py -v --tb=short
```

### 메모리 부족
```
해결: 배치 크기 감소 또는 이미지 크기 조정
python -c "from PIL import Image; img = Image.open(...); img.resize((512, 512))"
```

---

**마지막 검증**: 2026-01-21 ✅
