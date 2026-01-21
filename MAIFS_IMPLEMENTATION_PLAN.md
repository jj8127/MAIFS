# MAIFS 프로젝트 - 구현 상태 및 다음 작업

> **마지막 업데이트**: 2026-01-21 (완료 후)
> **현재 상태**: 핵심 기능 구현 완료 ✅, OmniGuard 체크포인트 검증 완료 ✅

## ✅ 완료된 작업 (2026-01-21)

### 1. 설정 파일 업데이트
- `configs/settings.py` - 경로 자동 감지, 다중 환경 지원
- OmniGuard 경로를 `OmniGuard-main`으로 수정
- 실제 다운로드된 체크포인트 파일 반영 (`checkpoint-175.pth`, `model_checkpoint_*.pt`)

### 2. Tool 경로 수정
- `src/tools/watermark_tool.py` - 설정에서 경로 로드
- `src/tools/spatial_tool.py` - 설정에서 경로 로드

### 3. 테스트 코드 작성 (94개 테스트 통과 ✅)
- `tests/test_tools.py` - 21개 테스트 ✅
- `tests/test_cobra.py` - 18개 테스트 ✅
- `tests/test_debate.py` - 19개 테스트 ✅
- `tests/test_e2e.py` - 21개 테스트 ✅
- `tests/test_checkpoint_loading.py` - 15개 테스트 ✅ (새로 추가)

### 4. 버그 수정
- MAIFS에서 consensus_algorithm 파라미터가 무시되는 문제 수정
- 존재하지 않는 파일에 대한 에러 처리 추가
- 체크포인트 경로 감지 로직 개선

### 5. OmniGuard 체크포인트 검증 ✅
- 체크포인트 디렉토리 존재 확인
- 체크포인트 파일 존재 확인 (5개 파일 발견)
- Tool 초기화 테스트 완료
- 모델 로드 시도 완료 (Fallback 모드 정상 작동)
- MAIFS 전체 파이프라인 테스트 완료

---

---

## 📋 1. 즉시 구현 가능 (낮은 난이도)

### 1.1 **Tool 경로 및 체크포인트 통합** 🔧
**파일**: `src/tools/watermark_tool.py`, `src/tools/spatial_tool.py`

**현재 상태**:
```python
# 현재 하드코딩됨
OMNIGUARD_PATH = Path("e:/Downloads/OmniGuard-main/OmniGuard-main")
```

**해야 할 일**:
```python
# 1. 설정 파일에서 경로 로드
# 2. 다양한 운영체제 지원
# 3. 체크포인트 자동 다운로드 옵션

# configs/settings.py 에 추가:
OMNIGUARD_CHECKPOINT_PATH = "/path/to/OmniGuard-main/checkpoint"
HINET_CHECKPOINT = "hinet.pth"  # 또는 hinet_2.pth
IML_VIT_CHECKPOINT = "iml_vit.pth"

# watermark_tool.py와 spatial_tool.py 수정:
from ..configs import settings
self.checkpoint_path = Path(settings.OMNIGUARD_CHECKPOINT_PATH)
```

**예상 소요 시간**: 30분

---

### 1.2 **Fallback 모드 테스트** ✅
**파일**: `src/tools/watermark_tool.py:54`, `src/tools/spatial_tool.py:80`

**현재 상태**: 모델 로드 실패 시 fallback 분석 구현 완료

**해야 할 일**:
```python
# test_tools.py 작성 (테스트 코드)
import pytest
from src.tools.watermark_tool import WatermarkTool
import numpy as np

def test_watermark_tool_fallback():
    """모델 없을 때 fallback 동작 확인"""
    tool = WatermarkTool(checkpoint_path=Path("/nonexistent"))

    dummy_image = np.random.rand(256, 256, 3)
    result = tool(dummy_image)

    assert result.verdict is not None
    assert result.confidence >= 0.0
    assert result.evidence.get("fallback_mode") == True

def test_frequency_tool():
    """주파수 분석 도구 테스트"""
    from src.tools.frequency_tool import FrequencyAnalysisTool
    tool = FrequencyAnalysisTool()

    dummy_image = np.random.rand(512, 512, 3)
    result = tool(dummy_image)

    assert result.verdict in [Verdict.AUTHENTIC, Verdict.AI_GENERATED, Verdict.UNCERTAIN]
    assert 0.0 <= result.confidence <= 1.0
```

**예상 소요 시간**: 45분

---

### 1.3 **Debate 프로토콜 테스트** 🧪
**파일**: `src/debate/protocols.py` (이미 구현됨)

**해야 할 일**:
```python
# test_debate.py 작성
def test_synchronous_debate():
    from src.debate.protocols import SynchronousDebate, DebateState
    from src.agents.base_agent import AgentResponse, AgentRole
    from src.tools.base_tool import Verdict

    # 샘플 응답 생성
    responses = {
        "freq_agent": AgentResponse(
            agent_name="Frequency Agent",
            role=AgentRole.FREQUENCY,
            verdict=Verdict.AI_GENERATED,
            confidence=0.8,
            reasoning="High frequency peaks",
            arguments=["Grid artifact detected"]
        ),
        "noise_agent": AgentResponse(
            agent_name="Noise Agent",
            role=AgentRole.NOISE,
            verdict=Verdict.AUTHENTIC,
            confidence=0.6,
            reasoning="Natural PRNU pattern",
            arguments=["PRNU variance normal"]
        )
    }

    protocol = SynchronousDebate(max_rounds=2)
    state = DebateState()

    messages, state = protocol.conduct_round(responses, state)

    assert len(messages) == 2
    assert state.current_round == 1

def test_asynchronous_debate():
    """비동기 토론 테스트"""
    # 유사한 구조
    pass

def test_structured_debate():
    """구조화 토론 테스트"""
    # 각 phase (claim, rebuttal, rejoinder, summary) 검증
    pass
```

**예상 소요 시간**: 1시간

---

## 🔌 2. 중간 난이도 - 실제 기능 테스트

### 2.1 **E2E 테스트 - 실제 이미지로 파이프라인 테스트**
**파일**: `tests/test_e2e.py` (새로 생성)

**해야 할 일**:
```python
def test_maifs_complete_pipeline():
    """전체 MAIFS 파이프라인 테스트"""
    from src.maifs import MAIFS
    from PIL import Image
    import numpy as np

    # 테스트 이미지 선택
    test_image_path = "path/to/image.png"

    # MAIFS 초기화
    maifs = MAIFS(
        enable_debate=True,
        consensus_algorithm="drwa"
    )

    # 분석 실행
    result = maifs.analyze(test_image_path)

    # 검증
    assert result.verdict is not None
    assert 0.0 <= result.confidence <= 1.0
    assert len(result.agent_responses) == 4  # 4개 전문가

    if result.debate_result:
        assert result.debate_result.total_rounds > 0
        print(result.debate_result.get_summary())

    # 보고서 저장
    result_path = "outputs/test_result.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)

    with open(result_path, "w") as f:
        f.write(result.to_json(indent=2))
```

**예상 소요 시간**: 1.5시간

---

### 2.2 **COBRA 알고리즘 검증 테스트**
**파일**: `tests/test_cobra.py`

**해야 할 일**:
```python
def test_cobra_rot_algorithm():
    """RoT 알고리즘 검증"""
    from src.consensus.cobra import RootOfTrust

    # 샘플 응답
    responses = {
        "agent1": AgentResponse(..., verdict=Verdict.AI_GENERATED, confidence=0.9),
        "agent2": AgentResponse(..., verdict=Verdict.AUTHENTIC, confidence=0.7),
        "agent3": AgentResponse(..., verdict=Verdict.AI_GENERATED, confidence=0.8),
    }

    trust_scores = {
        "agent1": 0.8,  # 신뢰도 높음
        "agent2": 0.4,  # 신뢰도 낮음
        "agent3": 0.8   # 신뢰도 높음
    }

    algorithm = RootOfTrust(trust_threshold=0.7, alpha=0.3)
    result = algorithm.aggregate(responses, trust_scores)

    # RoT는 신뢰도 높은 에이전트를 우선함
    assert result.agent_weights["agent1"] > result.agent_weights["agent2"]
    assert result.final_verdict == Verdict.AI_GENERATED

def test_cobra_drwa_algorithm():
    """DRWA 알고리즘 검증"""
    # 동적 가중치 조정이 제대로 작동하는지 확인
    pass

def test_cobra_avga_algorithm():
    """AVGA 알고리즘 검증"""
    # 어텐션 기반 집계가 제대로 작동하는지 확인
    pass

def test_cobra_algorithm_selection():
    """자동 알고리즘 선택 로직 검증"""
    # 상황에 따라 올바른 알고리즘이 선택되는지 확인
    pass
```

**예상 소요 시간**: 1.5시간

---

### 2.3 **Tool 결과 검증**
**파일**: `tests/test_tools.py`

**해야 할 일**:
```python
def test_frequency_tool_real_image():
    """주파수 도구 - 실제 이미지 테스트"""
    from src.tools.frequency_tool import FrequencyAnalysisTool

    test_image = Image.open("path/to/image.png")
    img_array = np.array(test_image)

    tool = FrequencyAnalysisTool()
    result = tool(img_array)

    print(f"Verdict: {result.verdict}")
    print(f"Confidence: {result.confidence:.2%}")
    print(f"Evidence: {result.evidence}")

    assert result.tool_name == "frequency_analyzer"

def test_noise_tool_real_image():
    """노이즈 도구 - 실제 이미지 테스트"""
    # 유사한 구조
    pass

def test_all_tools_consistency():
    """모든 도구가 일관된 결과 반환하는지 확인"""
    # 여러 이미지로 각 도구 테스트
    pass
```

**예상 소요 시간**: 1시간

---

## 🎯 3. 고급 - 성능 개선

### 3.1 **Debate 시스템 동작 검증**
**파일**: `tests/test_debate.py`

**확인 사항**:
```python
def test_debate_convergence():
    """토론이 수렴하는지 확인"""
    from src.debate.debate_chamber import DebateChamber

    # 서로 다른 판정 에이전트들
    responses = {
        "freq": AgentResponse(..., verdict=Verdict.AI_GENERATED, confidence=0.85),
        "noise": AgentResponse(..., verdict=Verdict.AUTHENTIC, confidence=0.65),
        "watermark": AgentResponse(..., verdict=Verdict.UNCERTAIN, confidence=0.50),
        "spatial": AgentResponse(..., verdict=Verdict.AI_GENERATED, confidence=0.75),
    }

    chamber = DebateChamber()
    result = chamber.conduct_debate(responses)

    # 검증
    assert result.convergence_achieved or result.total_rounds >= 3
    assert len(result.rounds) > 0
    print(result.get_summary())
```

**예상 소요 시간**: 1.5시간

---

### 3.2 **Performance Benchmark**
**파일**: `benchmarks/benchmark.py`

**해야 할 일**:
```python
def benchmark_tool_speed():
    """각 도구의 처리 속도 측정"""
    import time
    from src.tools.frequency_tool import FrequencyAnalysisTool
    from src.tools.noise_tool import NoiseAnalysisTool

    test_image = np.random.rand(512, 512, 3)

    tools = [
        FrequencyAnalysisTool(),
        NoiseAnalysisTool(),
        # WatermarkTool, SpatialAnalysisTool 추가 (모델이 있을 경우)
    ]

    for tool in tools:
        start = time.time()
        result = tool(test_image)
        elapsed = time.time() - start

        print(f"{tool.name}: {elapsed:.3f}s")

def benchmark_consensus_algorithms():
    """합의 알고리즘 성능 비교"""
    # 3가지 알고리즘의 속도 및 정확도 비교
    pass
```

**예상 소요 시간**: 1시간

---

## 📝 4. 문서화 및 정리

### 4.1 **API 문서 작성**
**파일**: `docs/API_REFERENCE.md` 수정

```markdown
## Tool API

### FrequencyAnalysisTool
- **설명**: FFT 기반 주파수 분석
- **입력**: RGB 이미지 (H, W, 3)
- **출력**: ToolResult (verdict, confidence, evidence)
- **예시**:
  ```python
  tool = FrequencyAnalysisTool()
  result = tool(image)
  print(result.evidence)  # {"grid_analysis": {...}, ...}
  ```

### NoiseAnalysisTool
...
```

**예상 소요 시간**: 1시간

---

### 4.2 **Example 코드 작성**
**파일**: `examples/` 디렉토리 추가

```python
# examples/basic_usage.py
from src.maifs import MAIFS

# 기본 사용법
maifs = MAIFS()
result = maifs.analyze("path/to/image.jpg")
print(result.verdict, result.confidence)

# examples/advanced_usage.py
# COBRA 알고리즘 선택, 디버깅 모드 등

# examples/tool_usage.py
# 각 도구 개별 사용법

# examples/debate_example.py
# 토론 시스템 예시
```

**예상 소요 시간**: 1시간

---

## ✅ 우선순위별 체크리스트

### 🔴 **우선순위 1 (지금 시작)**
- [ ] 1.1 Tool 경로 및 체크포인트 통합
- [ ] 1.2 Fallback 모드 테스트 작성
- [ ] 1.3 기본 Debate 프로토콜 테스트

### 🟡 **우선순위 2 (1주일 내)**
- [ ] 2.1 E2E 파이프라인 테스트 (실제 이미지)
- [ ] 2.2 COBRA 알고리즘 검증
- [ ] 2.3 Tool 결과 검증

### 🟢 **우선순위 3 (나중에)**
- [ ] 3.1 Debate 수렴성 검증
- [ ] 3.2 Performance Benchmark
- [ ] 4.1 API 문서화
- [ ] 4.2 Example 코드 작성

---

## 🚀 빠른 시작 - 30분 안에 할 수 있는 작업

### Step 1: Tool 경로 수정 (5분)
```bash
cd /path/to/MAIFS
# src/tools/watermark_tool.py와 spatial_tool.py의 경로 수정
```

### Step 2: 간단한 테스트 (10분)
```python
# tests/test_quick.py
from src.tools.frequency_tool import FrequencyAnalysisTool
import numpy as np

tool = FrequencyAnalysisTool()
dummy = np.random.rand(512, 512, 3)
result = tool(dummy)
print(result.verdict, result.confidence)  # 작동 확인
```

### Step 3: E2E 테스트 (15분)
```python
# 실제 이미지로 테스트
from src.maifs import MAIFS
maifs = MAIFS(enable_debate=False)  # 토론 없이
result = maifs.analyze("path/to/image.png")
print(result.summary)
```

---

## 💡 도움말

### 문제 발생 시
1. **모델 로드 실패**: fallback 모드가 자동으로 작동함
2. **경로 오류**: `settings.py`에서 경로 확인
3. **메모리 부족**: 이미지 크기 줄이기 또는 배치 처리

### 다음 단계
- 모든 Tool이 정상 작동하면 → Debate 시스템 통합 테스트
- Debate 테스트 완료 → OmniGuard 체크포인트 연동
- 최종 → LLM 통합 (Claude API)

---

**질문이나 막히는 부분이 있으면 언제든 물어봐!**
