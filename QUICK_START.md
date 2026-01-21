# MAIFS 빠른 시작 가이드

---

## 🚀 30초 시작

```bash
# 1. 저장소 이동
cd /path/to/MAIFS

# 2. 기본 분석 실행
python -c "
from src.maifs import MAIFS
maifs = MAIFS(enable_debate=False)
result = maifs.analyze('path/to/image.jpg')
print(f'결과: {result.verdict.value} ({result.confidence:.1%})')
"

# 3. 테스트 실행
python -m pytest tests/ -v --tb=short
```

---

## 💻 기본 사용 예제

### 예제 1: 간단한 분석
```python
from src.maifs import MAIFS
import numpy as np

# MAIFS 초기화
maifs = MAIFS(enable_debate=False)

# 분석 (더미 이미지)
image = np.random.rand(512, 512, 3)
result = maifs.analyze(image)

# 결과 확인
print(f"판정: {result.verdict.value}")
print(f"신뢰도: {result.confidence:.1%}")
```

### 예제 2: 파일 분석
```python
from src.maifs import MAIFS

maifs = MAIFS()
result = maifs.analyze('/path/to/image.jpg')

# 판정 설명 출력
print(result.get_verdict_explanation())

# JSON 저장
with open('result.json', 'w') as f:
    f.write(result.to_json())
```

### 예제 3: 커스텀 설정
```python
from src.maifs import MAIFS

# DRWA 알고리즘 사용, 토론 활성화
maifs = MAIFS(
    consensus_algorithm="drwa",
    enable_debate=True,
    debate_threshold=0.2
)

result = maifs.analyze('/path/to/image.jpg')

# 토론 결과 확인
if result.debate_result:
    print(f"토론 라운드: {result.debate_result.total_rounds}")
    print(result.debate_result.get_summary())
```

### 예제 4: 배치 처리
```python
from src.maifs import MAIFS
from pathlib import Path

maifs = MAIFS(enable_debate=False)
results = []

for img_path in Path('/path/to/images').glob('*.jpg'):
    result = maifs.analyze(img_path)
    results.append({
        'image': img_path.name,
        'verdict': result.verdict.value,
        'confidence': result.confidence
    })

# 결과 출력
for r in results:
    print(f"{r['image']}: {r['verdict']} ({r['confidence']:.1%})")
```

---

## 📊 합의 알고리즘 선택

```python
from src.maifs import MAIFS

# 신뢰도 기반 (Root-of-Trust)
maifs = MAIFS(consensus_algorithm="rot")

# 동적 가중치 (기본값)
maifs = MAIFS(consensus_algorithm="drwa")

# 어텐션 기반
maifs = MAIFS(consensus_algorithm="avga")

# 자동 선택
maifs = MAIFS(consensus_algorithm="auto")
```

---

## 🔍 토론 옵션

```python
from src.maifs import MAIFS

# 토론 활성화 (기본값)
maifs = MAIFS(enable_debate=True)

# 토론 비활성화
maifs = MAIFS(enable_debate=False)

# 토론 임계값 조정
maifs = MAIFS(
    enable_debate=True,
    debate_threshold=0.3  # 0.0-1.0
)

# 토론 프로토콜 선택
from configs.settings import config
config.debate.debate_mode = "asynchronous"  # or "synchronous", "structured"
```

---

## 📁 파일 입력 형식

```python
from src.maifs import MAIFS
from PIL import Image
import numpy as np

maifs = MAIFS()

# 1️⃣ 파일 경로 (문자열)
result = maifs.analyze('image.jpg')

# 2️⃣ Path 객체
from pathlib import Path
result = maifs.analyze(Path('image.jpg'))

# 3️⃣ PIL Image
img = Image.open('image.jpg')
result = maifs.analyze(img)

# 4️⃣ NumPy 배열 (RGB)
arr = np.random.rand(512, 512, 3)
result = maifs.analyze(arr)

# 5️⃣ 그레이스케일 → 자동 RGB 변환
gray = np.random.rand(512, 512)
result = maifs.analyze(gray)
```

---

## 🧪 테스트 실행

```bash
# 모든 테스트 (94개)
python -m pytest tests/ -v

# 특정 모듈만
python -m pytest tests/test_e2e.py -v
python -m pytest tests/test_checkpoint_loading.py -v

# 특정 클래스
python -m pytest tests/test_e2e.py::TestMAIFSAnalysis -v

# 특정 테스트
python -m pytest tests/test_e2e.py::TestMAIFSAnalysis::test_analyze_numpy_array -v

# 상세 로그
python -m pytest tests/ -v -s

# 짧은 로그
python -m pytest tests/ -q
```

---

## 🔧 설정 확인

```bash
# 시스템 설정 출력
python -c "from configs.settings import config; config.print_info()"

# 체크포인트 확인
python -c "from configs.settings import config; print(config.model.get_available_checkpoints())"

# 최적 HiNet 체크포인트
python -c "from configs.settings import config; print(config.model.get_best_hinet_checkpoint())"

# 디바이스 확인
python -c "from configs.settings import config; print(f'Device: {config.model.device}')"
```

---

## 📊 결과 포맷

### 결과 객체 구조
```python
result = maifs.analyze(image)

# 주요 속성
result.verdict              # Verdict (AUTHENTIC, AI_GENERATED, MANIPULATED, UNCERTAIN)
result.confidence          # float (0.0-1.0)
result.processing_time     # float (초)
result.agent_responses     # Dict[str, AgentResponse]
result.consensus_result    # ConsensusResult
result.debate_result       # Optional[DebateResult]
result.image_info          # Dict (파일명, 크기 등)

# 메서드
result.to_dict()           # Dict 변환
result.to_json()           # JSON 문자열 변환
result.get_verdict_explanation()  # 설명 문자열
```

### 에이전트 응답 구조
```python
for name, response in result.agent_responses.items():
    print(f"에이전트: {name}")
    print(f"  판정: {response.verdict.value}")
    print(f"  신뢰도: {response.confidence:.1%}")
    print(f"  근거: {response.evidence}")
    print(f"  주장: {response.arguments}")
```

---

## 🐛 문제 해결

### 체크포인트를 찾을 수 없음
```bash
# 경로 확인
python -c "from configs.settings import config; print(config.model.omniguard_checkpoint_dir)"

# 파일 목록 확인
ls -lh OmniGuard-main/checkpoint/
```

### 테스트 실패
```bash
# 상세 로그와 함께 실행
python -m pytest tests/ -v --tb=long

# 특정 테스트만 디버그
python -m pytest tests/test_e2e.py::TestMAIFSAnalysis::test_analyze_numpy_array -vvv
```

### 메모리 부족
```python
# 이미지 크기 감소
from PIL import Image
img = Image.open('large_image.jpg')
img = img.resize((512, 512))
result = maifs.analyze(img)
```

### GPU 사용하고 싶음
```bash
# CUDA 디바이스 활성화
export CUDA_VISIBLE_DEVICES=0
python your_script.py
```

---

## 📈 성능 측정

```python
import time
from src.maifs import MAIFS

maifs = MAIFS(enable_debate=False)

# 워밍업
maifs.analyze(np.random.rand(512, 512, 3))

# 성능 측정
start = time.time()
result = maifs.analyze(np.random.rand(512, 512, 3))
elapsed = time.time() - start

print(f"처리 시간: {elapsed:.2f}초")
print(f"신뢰도: {result.confidence:.1%}")
```

---

## 🔐 입력 검증

```python
from src.maifs import MAIFS

maifs = MAIFS()

# ❌ 잘못된 입력
try:
    maifs.analyze("nonexistent.jpg")  # 파일 없음 → ValueError
except ValueError as e:
    print(f"에러: {e}")

# ✅ 올바른 입력
import numpy as np
image = np.random.rand(512, 512, 3)
result = maifs.analyze(image)
```

---

## 📚 추가 리소스

- 전체 문서: `SYSTEM_STATUS.md`
- 체크포인트 보고서: `CHECKPOINT_VALIDATION_REPORT.md`
- 변경 사항: `CHANGES_SUMMARY.md`
- 구현 계획: `MAIFS_IMPLEMENTATION_PLAN.md`

---

## ✨ 자주 사용하는 명령어

```bash
# 시스템 상태 확인
python -c "from configs.settings import config; config.print_info()"

# 모든 테스트 실행
python -m pytest tests/ -v

# 체크포인트 테스트만
python -m pytest tests/test_checkpoint_loading.py -v

# E2E 파이프라인 테스트
python -m pytest tests/test_e2e.py::TestMAIFSWithRealImages -v -s
```

---

**🎉 준비 완료! 시작하세요!**
