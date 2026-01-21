# MAIFS Architecture Guide

## 시스템 아키텍처 개요

MAIFS는 다층 구조의 다중 에이전트 시스템으로, 각 계층이 명확한 책임을 갖습니다.

---

## 계층 구조

```
┌─────────────────────────────────────────────────────────────────┐
│                       Application Layer                          │
│              (main.py, app.py - CLI & Web UI)                   │
└───────────────────────────────┬─────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────┐
│                      Orchestration Layer                         │
│                   (MAIFS, ManagerAgent)                          │
│         - 분석 파이프라인 조율                                    │
│         - 결과 수집 및 통합                                       │
└───────────────────────────────┬─────────────────────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
        ▼                       ▼                       ▼
┌───────────────┐       ┌───────────────┐       ┌───────────────┐
│   Consensus   │       │    Debate     │       │   Specialist  │
│    Layer      │       │    Layer      │       │    Agents     │
│   (COBRA)     │       │  (Chamber)    │       │   Layer       │
└───────┬───────┘       └───────┬───────┘       └───────┬───────┘
        │                       │                       │
        └───────────────────────┴───────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────┐
│                         Tool Layer                               │
│        (FrequencyTool, NoiseTool, WatermarkTool, SpatialTool)   │
│                    - 실제 분석 로직 수행                          │
└───────────────────────────────┬─────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────┐
│                       External Models                            │
│              (OmniGuard: HiNet, ViT, UNet)                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 핵심 컴포넌트

### 1. Tool Layer

포렌식 분석의 실제 로직을 담당합니다.

```
BaseTool (Abstract)
├── load_model()     # 모델 로드
├── analyze()        # 분석 실행
└── unload_model()   # 메모리 해제

구현체:
├── FrequencyAnalysisTool  # FFT 기반 분석
├── NoiseAnalysisTool      # SRM/PRNU 분석
├── WatermarkTool          # HiNet 워터마크
└── SpatialAnalysisTool    # ViT 공간 분석
```

#### ToolResult 구조

```python
@dataclass
class ToolResult:
    tool_name: str              # 도구 이름
    verdict: Verdict            # 판정 (Enum)
    confidence: float           # 신뢰도 [0, 1]
    evidence: Dict[str, Any]    # 증거 데이터
    explanation: str            # 설명
    manipulation_mask: Optional[np.ndarray]  # 조작 마스크
    processing_time: float      # 처리 시간
```

### 2. Agent Layer

Tool을 사용하고 결과를 해석하는 전문가 에이전트입니다.

```
BaseAgent (Abstract)
├── analyze()              # 분석 수행
├── generate_reasoning()   # 추론 생성
├── respond_to_challenge() # 반론 대응
└── register_tool()        # 도구 등록

구현체:
├── FrequencyAgent
├── NoiseAgent
├── WatermarkAgent
├── SpatialAgent
└── ManagerAgent (특수)
```

#### AgentResponse 구조

```python
@dataclass
class AgentResponse:
    agent_name: str
    role: AgentRole
    verdict: Verdict
    confidence: float
    reasoning: str              # 추론 설명
    evidence: Dict[str, Any]
    tool_results: List[ToolResult]
    arguments: List[str]        # 토론용 논거
    counter_arguments: Dict     # 반론
    processing_time: float
```

### 3. Consensus Layer (COBRA)

에이전트 응답을 집계하여 합의를 도출합니다.

```
COBRAConsensus
├── aggregate()           # 합의 집계
└── _select_algorithm()   # 알고리즘 자동 선택

알고리즘:
├── RootOfTrust (RoT)
│   └── 신뢰/비신뢰 코호트 분리
├── DRWA
│   └── 동적 가중치 조정
└── AVGA
    └── 분산 기반 어텐션
```

#### 알고리즘 선택 기준

| 상황 | 선택 알고리즘 | 이유 |
|------|-------------|------|
| 신뢰도 편차 큼 | RoT | 신뢰 코호트 분리 필요 |
| 판정 3개 이상 | AVGA | 분산 기반 균등 분배 |
| 일반적 | DRWA | 동적 가중치 조정 |

### 4. Debate Layer

에이전트 간 의견 충돌을 해결합니다.

```
DebateChamber
├── should_debate()        # 토론 필요성 판단
├── conduct_debate()       # 토론 진행
└── generate_transcript()  # 기록 생성

프로토콜:
├── SynchronousDebate   # 동시 발언
├── AsynchronousDebate  # 순차 발언 (권장)
└── StructuredDebate    # 역할 기반
```

#### 토론 흐름

```
Round 1: Claim
├── Agent A: "AI_GENERATED, 신뢰도 0.85"
├── Agent B: "AUTHENTIC, 신뢰도 0.72"
└── Agent C: "AI_GENERATED, 신뢰도 0.78"

Round 2: Rebuttal
├── Agent B → A, C: "주파수 패턴은 압축 아티팩트일 수 있음"
└── Agent A → B: "노이즈 분산이 자연 이미지와 다름"

Round 3: Resolution
└── 합의: AI_GENERATED, 신뢰도 0.81
```

---

## 데이터 흐름

### 1. 분석 파이프라인

```python
# 1. 입력
image: np.ndarray  # (H, W, 3) RGB

# 2. 병렬 에이전트 분석
responses: Dict[str, AgentResponse] = {
    "frequency": FrequencyAgent.analyze(image),
    "noise": NoiseAgent.analyze(image),
    "watermark": WatermarkAgent.analyze(image),
    "spatial": SpatialAgent.analyze(image),
}

# 3. 합의 계산
consensus: ConsensusResult = COBRA.aggregate(responses, trust_scores)

# 4. 토론 (필요시)
if disagreement > threshold:
    debate_result: DebateResult = DebateChamber.conduct_debate(responses)
    consensus = debate_result.consensus_result

# 5. 최종 결과
result: MAIFSResult = MAIFSResult(
    verdict=consensus.final_verdict,
    confidence=consensus.confidence,
    ...
)
```

### 2. 신뢰도 전파

```
Tool Confidence (0.0 ~ 1.0)
        │
        ▼
Agent Confidence = Tool Confidence × Agent Trust Score
        │
        ▼
Consensus Confidence = Weighted Average (COBRA)
        │
        ▼ (토론 시)
Final Confidence = Debate-Adjusted Consensus
```

---

## 확장 포인트

### 1. 새 분석 도구 추가

```python
# 1. BaseTool 상속
class MyTool(BaseTool):
    def analyze(self, image: np.ndarray) -> ToolResult:
        ...

# 2. __init__.py에 등록
# src/tools/__init__.py
from .my_tool import MyTool
```

### 2. 새 에이전트 추가

```python
# 1. BaseAgent 상속
class MyAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="...", role=AgentRole.SPATIAL, ...)
        self._tool = MyTool()

# 2. ManagerAgent에 등록
# src/agents/manager_agent.py
self.specialists["my_agent"] = MyAgent()
self.trust_scores["my_agent"] = 0.80
```

### 3. 새 합의 알고리즘 추가

```python
# 1. ConsensusAlgorithm 상속
class MyAlgorithm(ConsensusAlgorithm):
    def aggregate(...) -> ConsensusResult:
        ...

# 2. COBRAConsensus에 등록
self.algorithms["my_algo"] = MyAlgorithm()
```

### 4. 새 토론 프로토콜 추가

```python
# 1. DebateProtocol 상속
class MyDebateProtocol(DebateProtocol):
    def conduct_round(...) -> Tuple[List[DebateMessage], DebateState]:
        ...

# 2. DebateChamber에 사용
chamber = DebateChamber(protocol=MyDebateProtocol())
```

---

## 인터페이스 계약

### Tool Interface

```python
class BaseTool(ABC):
    @abstractmethod
    def load_model(self) -> None:
        """모델을 메모리에 로드"""
        pass

    @abstractmethod
    def analyze(self, image: np.ndarray) -> ToolResult:
        """
        이미지 분석 수행

        Preconditions:
            - image.shape == (H, W, 3)
            - image.dtype in [np.uint8, np.float32]

        Postconditions:
            - result.confidence in [0.0, 1.0]
            - result.verdict in Verdict
        """
        pass
```

### Agent Interface

```python
class BaseAgent(ABC):
    @abstractmethod
    def analyze(
        self,
        image: np.ndarray,
        context: Optional[Dict] = None
    ) -> AgentResponse:
        """
        전문가 분석 수행

        Preconditions:
            - 등록된 Tool이 1개 이상

        Postconditions:
            - response.reasoning은 비어있지 않음
            - response.tool_results에 최소 1개 결과
        """
        pass
```

### Consensus Interface

```python
class ConsensusAlgorithm(ABC):
    @abstractmethod
    def aggregate(
        self,
        responses: Dict[str, AgentResponse],
        trust_scores: Dict[str, float]
    ) -> ConsensusResult:
        """
        에이전트 응답 집계

        Preconditions:
            - len(responses) >= 1
            - all trust_scores in [0.0, 1.0]

        Postconditions:
            - result.final_verdict in Verdict
            - result.confidence in [0.0, 1.0]
        """
        pass
```

---

## 설정 및 튜닝

### 주요 파라미터

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `debate_threshold` | 0.3 | 토론 개시 불일치 임계값 |
| `max_debate_rounds` | 3 | 최대 토론 라운드 |
| `trust_threshold` | 0.7 | RoT 신뢰 코호트 임계값 |
| `drwa_epsilon` | 0.2 | DRWA 가중치 조정 계수 |
| `convergence_threshold` | 0.1 | 토론 수렴 판단 임계값 |

### 성능 최적화

1. **배치 처리**: 여러 이미지 동시 분석 시 GPU 배치 활용
2. **모델 캐싱**: `load_model()` 호출 최소화
3. **조기 종료**: 높은 신뢰도 합의 시 토론 생략
4. **비동기 분석**: 에이전트 병렬 실행

---

## 참고 문헌

1. AIFo: Agent-based Image Forensics Framework
2. COBRA: Consensus-Based Reward Aggregation
3. MAD-Sherlock: Multi-Agent Debate for Misinformation Detection
4. OmniGuard: Hybrid Manipulation Localization
