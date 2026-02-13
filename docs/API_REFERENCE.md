# MAIFS API Reference

## Core Classes

### MAIFS

메인 시스템 클래스입니다.

```python
class MAIFS:
    """Multi-Agent Image Forensic System"""

    VERSION = "0.1.0"

    def __init__(
        self,
        enable_debate: bool = True,
        debate_threshold: float = 0.3,
        consensus_algorithm: str = "drwa",
        device: str = "cuda"
    ) -> None:
        """
        MAIFS 인스턴스를 생성합니다.

        Args:
            enable_debate: 토론 기능 활성화 여부
            debate_threshold: 토론 개시 불일치 임계값 (0.0 ~ 1.0)
            consensus_algorithm: 합의 알고리즘
                - "rot": Root-of-Trust
                - "drwa": Dynamic Reliability Weighted Aggregation
                - "avga": Adaptive Variance-Guided Attention
            device: 연산 디바이스 ("cuda" 또는 "cpu")
        """

    def analyze(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        include_debate: Optional[bool] = None,
        save_report: Optional[Path] = None
    ) -> MAIFSResult:
        """
        이미지 분석을 수행합니다.

        Args:
            image: 분석할 이미지
                - str/Path: 이미지 파일 경로
                - np.ndarray: RGB 이미지 (H, W, 3)
                - PIL.Image: PIL 이미지 객체
            include_debate: 토론 포함 여부
                - None: 자동 결정 (enable_debate 설정 따름)
                - True: 강제 토론
                - False: 토론 생략
            save_report: 보고서 저장 경로
                - .json: JSON 형식
                - 기타: 텍스트 형식

        Returns:
            MAIFSResult: 분석 결과 객체

        Raises:
            ValueError: 지원하지 않는 이미지 형식
            FileNotFoundError: 이미지 파일 없음
            RuntimeError: 분석 중 오류

        Examples:
            >>> maifs = MAIFS()
            >>> result = maifs.analyze("photo.jpg")
            >>> print(result.verdict)
            Verdict.AUTHENTIC
        """
```

---

### MAIFSResult

분석 결과를 담는 데이터 클래스입니다.

```python
@dataclass
class MAIFSResult:
    """MAIFS 분석 최종 결과"""

    verdict: Verdict
    """최종 판정 (Verdict enum)"""

    confidence: float
    """판정 신뢰도 (0.0 ~ 1.0)"""

    summary: str
    """결과 요약 문자열"""

    detailed_report: str
    """상세 분석 보고서"""

    agent_responses: Dict[str, AgentResponse]
    """에이전트별 응답 딕셔너리"""

    consensus_result: Optional[ConsensusResult]
    """COBRA 합의 결과"""

    debate_result: Optional[DebateResult]
    """토론 결과 (토론 진행 시)"""

    image_info: Dict[str, Any]
    """이미지 메타데이터"""

    processing_time: float
    """총 처리 시간 (초)"""

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""

    def to_json(self, indent: int = 2) -> str:
        """JSON 문자열로 변환"""

    def get_verdict_explanation(self) -> str:
        """판정에 대한 설명 반환"""
```

---

### Verdict

판정 결과 열거형입니다.

```python
class Verdict(Enum):
    """판정 결과"""

    AUTHENTIC = "authentic"
    """원본 이미지: 조작이나 AI 생성 흔적 없음"""

    MANIPULATED = "manipulated"
    """조작된 이미지: 부분 수정, 합성, 편집됨"""

    AI_GENERATED = "ai_generated"
    """AI 생성 이미지: GAN, Diffusion 등으로 생성"""

    UNCERTAIN = "uncertain"
    """판단 불가: 추가 분석 필요"""
```

---

## Tools

### BaseTool

모든 분석 도구의 기본 클래스입니다.

```python
class BaseTool(ABC):
    """분석 도구 기본 클래스"""

    def __init__(
        self,
        name: str,
        description: str,
        device: str = "cuda"
    ) -> None:
        """
        Args:
            name: 도구 고유 이름
            description: 도구 설명
            device: 연산 디바이스
        """

    @property
    def tool_name(self) -> str:
        """도구 이름"""

    @property
    def tool_description(self) -> str:
        """도구 설명"""

    @abstractmethod
    def load_model(self) -> None:
        """모델을 메모리에 로드"""

    @abstractmethod
    def analyze(self, image: np.ndarray) -> ToolResult:
        """
        이미지 분석 수행

        Args:
            image: RGB 이미지 (H, W, 3)

        Returns:
            ToolResult: 분석 결과
        """

    def __call__(self, image: np.ndarray) -> ToolResult:
        """도구를 함수처럼 호출"""

    def unload_model(self) -> None:
        """모델 언로드 (메모리 해제)"""

    def get_schema(self) -> Dict[str, Any]:
        """LangChain Tool 스키마 반환"""
```

### ToolResult

도구 분석 결과입니다.

```python
@dataclass
class ToolResult:
    """Tool 분석 결과"""

    tool_name: str
    """도구 이름"""

    verdict: Verdict
    """판정 결과"""

    confidence: float
    """신뢰도 (0.0 ~ 1.0)"""

    evidence: Dict[str, Any]
    """증거 데이터"""

    explanation: str
    """설명 문자열"""

    manipulation_mask: Optional[np.ndarray]
    """조작 영역 마스크 (있는 경우)"""

    raw_output: Optional[Any]
    """원시 모델 출력"""

    processing_time: float
    """처리 시간 (초)"""

    @property
    def confidence_level(self) -> ConfidenceLevel:
        """신뢰도 레벨 (VERY_LOW ~ VERY_HIGH)"""

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
```

### 구현된 도구들

#### FrequencyAnalysisTool

```python
class FrequencyAnalysisTool(BaseTool):
    """FFT 기반 주파수 분석 도구"""

    def analyze(self, image: np.ndarray) -> ToolResult:
        """
        주파수 스펙트럼 분석

        탐지 항목:
        - GAN 격자 아티팩트
        - 고주파 영역 이상
        - Radial Energy 분포

        Evidence 키:
        - ai_generation_score: float
        - grid_analysis: Dict
        - high_frequency_analysis: Dict
        """
```

#### NoiseAnalysisTool

```python
class NoiseAnalysisTool(BaseTool):
    """SRM/PRNU 기반 노이즈 분석 도구"""

    def analyze(self, image: np.ndarray) -> ToolResult:
        """
        노이즈 패턴 분석

        탐지 항목:
        - PRNU 센서 노이즈
        - SRM 필터 응답
        - 노이즈 일관성

        Evidence 키:
        - prnu_stats: Dict
        - consistency_analysis: Dict
        - ai_detection: Dict
        """
```

#### FatFormerTool

```python
class FatFormerTool(BaseTool):
    """FatFormer (CLIP+DWT) 기반 AI 생성 탐지 도구"""

    def analyze(self, image: np.ndarray) -> ToolResult:
        """
        AI 생성 이미지 탐지

        탐지 항목:
        - CLIP ViT-L/14 특징 추출
        - DWT 주파수 분석
        - Forgery-Aware Adapter 판정

        Evidence 키:
        - is_ai_generated: bool
        - fatformer_score: float
        - signal_detected: bool
        """
```

#### SpatialAnalysisTool

```python
class SpatialAnalysisTool(BaseTool):
    """ViT 기반 공간 분석 도구"""

    def analyze(self, image: np.ndarray) -> ToolResult:
        """
        픽셀 수준 조작 영역 탐지

        탐지 항목:
        - 조작 영역 마스크
        - 조작 비율
        - 경계 불일치

        Evidence 키:
        - manipulation_ratio: float
        - max_intensity: float
        - mean_intensity: float

        manipulation_mask: 조작 영역 마스크 (H, W)
        """
```

---

## Agents

### BaseAgent

모든 에이전트의 기본 클래스입니다.

```python
class BaseAgent(ABC):
    """에이전트 기본 클래스"""

    def __init__(
        self,
        name: str,
        role: AgentRole,
        description: str,
        llm_model: Optional[str] = None
    ) -> None:
        """
        Args:
            name: 에이전트 이름
            role: 에이전트 역할 (AgentRole enum)
            description: 에이전트 설명
            llm_model: LLM 모델 (선택적)
        """

    @property
    def trust_score(self) -> float:
        """현재 신뢰도 점수"""

    def register_tool(self, tool: BaseTool) -> None:
        """도구 등록"""

    @abstractmethod
    def analyze(
        self,
        image: np.ndarray,
        context: Optional[Dict] = None
    ) -> AgentResponse:
        """분석 수행"""

    @abstractmethod
    def generate_reasoning(
        self,
        tool_results: List[ToolResult],
        context: Optional[Dict] = None
    ) -> str:
        """추론 생성"""

    def respond_to_challenge(
        self,
        challenger_name: str,
        challenge: str,
        my_response: AgentResponse
    ) -> str:
        """반론에 대응"""

    def update_trust(self, delta: float) -> None:
        """신뢰도 업데이트"""

    def get_system_prompt(self) -> str:
        """LLM 시스템 프롬프트"""
```

### AgentResponse

에이전트 응답 데이터입니다.

```python
@dataclass
class AgentResponse:
    """에이전트 응답"""

    agent_name: str
    role: AgentRole
    verdict: Verdict
    confidence: float
    reasoning: str
    evidence: Dict[str, Any]
    tool_results: List[ToolResult]
    arguments: List[str]
    counter_arguments: Dict[str, str]
    processing_time: float

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
```

### AgentRole

에이전트 역할 열거형입니다.

```python
class AgentRole(Enum):
    MANAGER = "manager"
    FREQUENCY = "frequency"
    NOISE = "noise"
    FATFORMER = "fatformer"
    SPATIAL = "spatial"
    SEMANTIC = "semantic"
```

---

## Consensus

### COBRAConsensus

COBRA 합의 시스템입니다.

```python
class COBRAConsensus:
    """COBRA 합의 시스템"""

    def __init__(
        self,
        default_algorithm: str = "drwa",
        trust_threshold: float = 0.7,
        epsilon: float = 0.2,
        temperature: float = 1.0
    ) -> None:
        """
        Args:
            default_algorithm: 기본 알고리즘
            trust_threshold: RoT 신뢰 임계값
            epsilon: DRWA 조정 계수
            temperature: AVGA 온도
        """

    def aggregate(
        self,
        responses: Dict[str, AgentResponse],
        trust_scores: Dict[str, float],
        algorithm: Optional[str] = None
    ) -> ConsensusResult:
        """
        합의 도출

        Args:
            responses: 에이전트 응답
            trust_scores: 신뢰도 점수
            algorithm: 알고리즘 (None이면 자동 선택)

        Returns:
            ConsensusResult: 합의 결과
        """
```

### ConsensusResult

합의 결과 데이터입니다.

```python
@dataclass
class ConsensusResult:
    final_verdict: Verdict
    confidence: float
    algorithm_used: str
    verdict_scores: Dict[str, float]
    agent_weights: Dict[str, float]
    cohort_info: Dict[str, Any]
    disagreement_level: float

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
```

---

## Debate

### DebateChamber

토론 관리 클래스입니다.

```python
class DebateChamber:
    """토론 관리자"""

    def __init__(
        self,
        protocol: Optional[DebateProtocol] = None,
        consensus_engine: Optional[COBRAConsensus] = None,
        disagreement_threshold: float = 0.3
    ) -> None:
        """
        Args:
            protocol: 토론 프로토콜 (기본: 비동기식)
            consensus_engine: 합의 엔진
            disagreement_threshold: 토론 개시 임계값
        """

    def should_debate(
        self,
        responses: Dict[str, AgentResponse]
    ) -> bool:
        """토론 필요성 판단"""

    def conduct_debate(
        self,
        responses: Dict[str, AgentResponse],
        agents: Optional[Dict[str, BaseAgent]] = None,
        trust_scores: Optional[Dict[str, float]] = None
    ) -> DebateResult:
        """토론 진행"""

    def generate_debate_transcript(
        self,
        result: DebateResult
    ) -> str:
        """토론 기록 생성"""
```

### DebateResult

토론 결과 데이터입니다.

```python
@dataclass
class DebateResult:
    initial_verdicts: Dict[str, Verdict]
    final_verdicts: Dict[str, Verdict]
    rounds: List[DebateRound]
    total_rounds: int
    convergence_achieved: bool
    convergence_reason: str
    consensus_result: Optional[ConsensusResult]
    total_duration: float

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""

    def get_summary(self) -> str:
        """요약 문자열"""
```

---

## Utility Functions

### analyze_image

편의 함수입니다.

```python
def analyze_image(
    image_path: Union[str, Path],
    **kwargs
) -> MAIFSResult:
    """
    이미지 분석 편의 함수

    Args:
        image_path: 이미지 경로
        **kwargs: MAIFS 생성자 인자

    Returns:
        MAIFSResult: 분석 결과

    Example:
        >>> result = analyze_image("test.jpg", enable_debate=False)
    """
```

---

## LLM Integration

### QwenClient (NEW - v0.6.0)

vLLM 서버와 통신하는 Qwen 클라이언트입니다.

```python
class QwenClient:
    """Qwen vLLM 클라이언트"""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: float = 60.0,
        max_retries: int = 3
    ) -> None:
        """
        Args:
            base_url: vLLM 서버 URL
            timeout: 요청 타임아웃 (초)
            max_retries: 최대 재시도 횟수
        """

    async def health_check(self) -> bool:
        """서버 상태 확인"""

    async def infer(
        self,
        role: AgentRole,
        tool_results: Dict[str, Any],
        use_guided_json: bool = True,
        context: Optional[Dict] = None
    ) -> InferenceResult:
        """
        단일 에이전트 추론

        Args:
            role: 에이전트 역할 (AgentRole enum)
            tool_results: Tool 분석 결과
            use_guided_json: JSON 스키마 강제 여부
            context: 추가 컨텍스트

        Returns:
            InferenceResult: 추론 결과
        """

    async def batch_infer(
        self,
        tool_results_map: Dict[AgentRole, Dict[str, Any]],
        use_guided_json: bool = True
    ) -> Dict[AgentRole, InferenceResult]:
        """
        배치 추론 (4개 에이전트 동시 처리)

        Args:
            tool_results_map: {AgentRole: tool_results} 매핑
            use_guided_json: JSON 스키마 강제 여부

        Returns:
            Dict[AgentRole, InferenceResult]: 각 에이전트별 결과
        """

    async def debate_respond(
        self,
        role: AgentRole,
        my_verdict: str,
        my_confidence: float,
        my_evidence: Dict[str, Any],
        challenger_name: str,
        challenge: str
    ) -> InferenceResult:
        """
        토론 응답 생성

        Args:
            role: 응답하는 에이전트 역할
            my_verdict: 내 판정
            my_confidence: 내 신뢰도
            my_evidence: 내 증거
            challenger_name: 반론 제기자 이름
            challenge: 반론 내용

        Returns:
            InferenceResult: 토론 응답
        """

    async def close(self) -> None:
        """세션 종료"""
```

### InferenceResult

추론 결과 데이터입니다.

```python
@dataclass
class InferenceResult:
    """추론 결과"""

    role: AgentRole
    """에이전트 역할"""

    content: str
    """응답 텍스트"""

    parsed_json: Optional[Dict]
    """파싱된 JSON (있는 경우)"""

    raw_response: Optional[Dict]
    """원시 API 응답"""

    latency_ms: float
    """지연 시간 (밀리초)"""

    success: bool
    """성공 여부"""

    error: Optional[str]
    """에러 메시지 (실패 시)"""
```

### QwenMAIFSAdapter (NEW - v0.6.0)

MAIFS 시스템과 Qwen을 연결하는 어댑터입니다.

```python
class QwenMAIFSAdapter:
    """Qwen MAIFS 어댑터"""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        enable_debate: bool = True,
        max_debate_rounds: int = 3,
        consensus_threshold: float = 0.7
    ) -> None:
        """
        Args:
            base_url: vLLM 서버 URL
            enable_debate: 토론 활성화 여부
            max_debate_rounds: 최대 토론 라운드
            consensus_threshold: 합의 임계값
        """

    async def analyze_with_qwen(
        self,
        tool_results_map: Dict[str, Dict[str, Any]]
    ) -> Dict[str, QwenAnalysisResult]:
        """
        Tool 결과를 Qwen으로 분석

        Args:
            tool_results_map: {"frequency": {...}, "noise": {...}, ...}

        Returns:
            Dict[str, QwenAnalysisResult]: 에이전트별 분석 결과
        """

    async def analyze_single(
        self,
        agent_name: str,
        tool_results: Dict[str, Any]
    ) -> QwenAnalysisResult:
        """
        단일 에이전트 분석

        Args:
            agent_name: 에이전트 이름
            tool_results: Tool 분석 결과

        Returns:
            QwenAnalysisResult: 분석 결과
        """

    async def conduct_debate(
        self,
        analysis_results: Dict[str, QwenAnalysisResult]
    ) -> Dict[str, Any]:
        """
        토론 수행

        Args:
            analysis_results: 에이전트별 분석 결과

        Returns:
            Dict: 토론 결과
                - debate_conducted: bool
                - rounds: int
                - consensus_reached: bool
                - final_verdicts: Dict[str, str]
                - history: List[Dict]
                - updated_results: Dict[str, QwenAnalysisResult]
        """

    async def close(self) -> None:
        """리소스 정리"""
```

### QwenAnalysisResult

Qwen 분석 결과입니다.

```python
@dataclass
class QwenAnalysisResult:
    """Qwen 분석 결과"""

    verdict: Verdict
    """판정 결과"""

    confidence: float
    """신뢰도 (0.0 ~ 1.0)"""

    reasoning: str
    """추론 근거"""

    key_evidence: List[str]
    """핵심 증거 목록"""

    uncertainties: List[str]
    """불확실한 점 목록"""

    raw_result: Optional[InferenceResult]
    """원시 추론 결과"""

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
```

### JSON Output Schemas

Guided Decoding에 사용되는 JSON 스키마입니다.

```python
# 에이전트 분석 결과 스키마
AGENT_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["AUTHENTIC", "MANIPULATED", "AI_GENERATED", "UNCERTAIN"]
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0
        },
        "reasoning": {
            "type": "string",
            "description": "판정에 대한 논리적 근거"
        },
        "key_evidence": {
            "type": "array",
            "items": {"type": "string"},
            "description": "핵심 증거 목록"
        },
        "uncertainties": {
            "type": "array",
            "items": {"type": "string"},
            "description": "불확실한 점 목록"
        }
    },
    "required": ["verdict", "confidence", "reasoning"]
}

# 토론 응답 스키마
DEBATE_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "response_type": {
            "type": "string",
            "enum": ["defense", "concession", "counter", "clarification"]
        },
        "content": {
            "type": "string",
            "description": "응답 내용"
        },
        "verdict_changed": {
            "type": "boolean"
        },
        "new_verdict": {
            "type": ["string", "null"],
            "enum": ["AUTHENTIC", "MANIPULATED", "AI_GENERATED", "UNCERTAIN"]
        },
        "new_confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0
        },
        "reasoning": {
            "type": "string"
        }
    },
    "required": ["response_type", "content", "verdict_changed"]
}
```

### 사용 예시

```python
import asyncio
from src.llm import QwenMAIFSAdapter
from src.tools import (
    FrequencyAnalysisTool,
    NoiseAnalysisTool,
    FatFormerTool,
    SpatialAnalysisTool
)

async def analyze_image(image):
    # 1. Tool 분석 수행
    tools = {
        "frequency": FrequencyAnalysisTool(),
        "noise": NoiseAnalysisTool(),
        "fatformer": FatFormerTool(),
        "spatial": SpatialAnalysisTool()
    }

    tool_results = {}
    for name, tool in tools.items():
        result = tool.analyze(image)
        tool_results[name] = result.evidence

    # 2. Qwen LLM으로 해석
    adapter = QwenMAIFSAdapter(base_url="http://localhost:8000")

    analysis_results = await adapter.analyze_with_qwen(tool_results)

    # 3. 토론 수행 (불일치 시)
    debate_result = await adapter.conduct_debate(analysis_results)

    # 4. 결과 출력
    for name, result in analysis_results.items():
        print(f"{name}: {result.verdict.value} ({result.confidence:.1%})")

    if debate_result.get("debate_conducted"):
        print(f"토론 결과: {debate_result['final_verdicts']}")

    await adapter.close()

# 실행
import numpy as np
image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
asyncio.run(analyze_image(image))
```

### Technical Notes

#### vLLM Continuous Batching

**중요**: vLLM의 continuous batching은 **KV-cache를 공유하지 않습니다**.

**실제 동작**:
```python
# batch_infer() 호출 시
Agent 1: 독립적인 KV-cache[0] (시스템 프롬프트 + 입력)
Agent 2: 독립적인 KV-cache[1] (시스템 프롬프트 + 입력)
Agent 3: 독립적인 KV-cache[2] (시스템 프롬프트 + 입력)
Agent 4: 독립적인 KV-cache[3] (시스템 프롬프트 + 입력)
```

**Continuous Batching의 장점**:
- ✅ 여러 요청을 동시에 스케줄링
- ✅ GPU 활용도 향상 (idle time 최소화)
- ✅ Throughput 증가
- ❌ KV-cache 공유 (각 요청은 독립적)

**KV-cache 공유를 원한다면**:
```bash
# vLLM Prefix Caching 활성화 필요
--enable-prefix-caching
```

Prefix caching을 사용하면 공통 시스템 프롬프트 부분이 캐싱되어 재사용됩니다:
```
공통 prefix: "당신은 MAIFS의..." → 1번만 연산
개별 suffix: 각 에이전트의 tool_results → 4번 연산
```

#### GPU Allocation Strategy

MAIFS는 3+1 GPU 분할을 사용합니다:

```
GPU 0,1,2: Qwen LLM (Tensor Parallel 3)
GPU 3:     Vision Tools (전용)
```

**이유**:
- ✅ LLM과 Vision Tools의 자원 충돌 방지
- ✅ 병렬 실행 가능 → 전체 파이프라인 ~30% 빠름
- ⚠️ LLM 속도 ~15% 감소 (TP4 → TP3)

자세한 내용은 [docs/GPU_ALLOCATION.md](GPU_ALLOCATION.md)를 참조하세요.

#### Guided Decoding Guarantees

Guided JSON decoding (Outlines backend)이 보장하는 것:
- ✅ JSON 파싱 성공률 ~100%
- ✅ 스키마 준수 (enum, type, required)
- ✅ 일관된 출력 형식

보장하지 **않는** 것:
- ❌ 내용의 타당성 (논리적 오류 방지 불가)
- ❌ 추론의 정확성 (사실 검증 불가)
- ❌ 증거의 신뢰성 (품질 보장 불가)

추가 검증 로직이 필요한 경우:
```python
def validate_result(result: QwenAnalysisResult) -> bool:
    # 논리 검증
    if len(result.reasoning) < 50:
        return False

    # 증거 검증
    if result.confidence > 0.8 and len(result.key_evidence) < 2:
        return False

    return True
```

---

## Meta Module (DAAC)

DAAC (Disagreement-Aware Adaptive Consensus) 메타 분류기 모듈입니다.

### AgentSimulator

Gaussian copula 기반 합성 에이전트 출력 생성기입니다.

```python
class AgentSimulator:
    """Gaussian copula 기반 에이전트 시뮬레이터"""

    def __init__(
        self,
        profiles: Optional[Dict[str, AgentProfile]] = None,
        correlation: Optional[np.ndarray] = None,
        label_distribution: Optional[Dict[str, float]] = None,
        seed: int = 42
    ) -> None:
        """
        Args:
            profiles: 에이전트별 성능 프로파일 (None이면 문헌 기반 기본값)
            correlation: 에이전트 간 상관 행렬 (4×4)
            label_distribution: 레이블 비율 (None이면 균등)
            seed: 난수 시드
        """

    def generate(
        self,
        n_samples: int = 10000,
        sub_type_distribution: Optional[Dict] = None
    ) -> List[SimulatedOutput]:
        """합성 에이전트 출력 생성"""

    def generate_split(
        self,
        n_train: int = 8000,
        n_val: int = 1000,
        n_test: int = 1000,
        sub_type_split: bool = True
    ) -> Tuple[List[SimulatedOutput], List[SimulatedOutput], List[SimulatedOutput]]:
        """학습/검증/테스트 분할 생성"""
```

### MetaFeatureExtractor

43-dim 메타 특징 추출기입니다.

```python
class MetaFeatureExtractor:
    """43-dim 메타 특징 추출기"""

    def __init__(self, config: Optional[FeatureConfig] = None) -> None:
        """
        Args:
            config: 특징 추출 설정 (포함/제외할 특징 그룹)
        """

    def extract_single(self, sample: SimulatedOutput) -> np.ndarray:
        """단일 샘플에서 메타 특징 추출 → (D,) ndarray"""

    def extract_dataset(
        self,
        samples: List[SimulatedOutput]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        데이터셋 전체 특징 + 레이블 추출

        Returns:
            X: (N, D) feature matrix
            y: (N,) label indices (0=auth, 1=manip, 2=ai_gen)
        """
```

### MetaTrainer

메타 분류기 학습/추론 관리자입니다.

```python
class MetaTrainer:
    """LogReg, GradientBoosting, MLP 메타 분류기"""

    def train_all(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Dict[str, TrainResult]:
        """모든 모델 학습"""

    def predict(self, model_name: str, X: np.ndarray) -> np.ndarray:
        """예측"""

    def predict_proba(self, model_name: str, X: np.ndarray) -> np.ndarray:
        """확률 예측 (N, 3)"""
```

### MetaEvaluator

평가 지표 및 통계 검정입니다.

```python
class MetaEvaluator:
    """Macro-F1, AUROC, ECE, McNemar, Bootstrap CI"""

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        model_name: str = "unknown"
    ) -> EvalResult:
        """전체 평가 지표 계산"""

    def compare(
        self,
        y_true: np.ndarray,
        y_pred_a: np.ndarray,
        y_pred_b: np.ndarray,
        model_a: str = "model_a",
        model_b: str = "model_b"
    ) -> ComparisonResult:
        """McNemar 검정 + Bootstrap CI 비교"""
```

### 실행 예시

```python
from src.meta import AgentSimulator, MetaFeatureExtractor, MetaTrainer, MetaEvaluator

# 1. 합성 데이터 생성
sim = AgentSimulator(seed=42)
train, val, test = sim.generate_split(n_train=8000)

# 2. 메타 특징 추출
extractor = MetaFeatureExtractor()
X_train, y_train = extractor.extract_dataset(train)
X_test, y_test = extractor.extract_dataset(test)

# 3. 메타 분류기 학습
trainer = MetaTrainer()
trainer.train_all(X_train, y_train)

# 4. 평가
evaluator = MetaEvaluator()
y_pred = trainer.predict("gradient_boosting", X_test)
result = evaluator.evaluate(y_test, y_pred)
print(f"Macro-F1: {result.macro_f1:.4f}")
```

---

## Error Handling

### 예외 클래스

```python
class MAIFSError(Exception):
    """MAIFS 기본 예외"""

class ModelLoadError(MAIFSError):
    """모델 로드 실패"""

class AnalysisError(MAIFSError):
    """분석 중 오류"""

class ConsensusError(MAIFSError):
    """합의 도출 실패"""
```

### 에러 처리 예시

```python
from src.maifs import MAIFS, MAIFSError

try:
    maifs = MAIFS()
    result = maifs.analyze("image.jpg")
except FileNotFoundError:
    print("이미지 파일을 찾을 수 없습니다")
except MAIFSError as e:
    print(f"분석 중 오류: {e}")
```
