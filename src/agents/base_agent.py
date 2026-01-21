"""
MAIFS Agent 기본 클래스
모든 포렌식 분석 에이전트의 기본 인터페이스 정의
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List
from enum import Enum
import numpy as np
import time

from ..tools.base_tool import ToolResult, Verdict


class AgentRole(Enum):
    """에이전트 역할"""
    MANAGER = "manager"           # 총괄 관리자
    FREQUENCY = "frequency"       # 주파수 분석 전문가
    NOISE = "noise"               # 노이즈 분석 전문가
    WATERMARK = "watermark"       # 워터마크 분석 전문가
    SPATIAL = "spatial"           # 공간 분석 전문가
    SEMANTIC = "semantic"         # 의미론적 분석 전문가


@dataclass
class AgentResponse:
    """에이전트 응답"""
    agent_name: str
    role: AgentRole
    verdict: Verdict
    confidence: float
    reasoning: str
    evidence: Dict[str, Any] = field(default_factory=dict)
    tool_results: List[ToolResult] = field(default_factory=list)
    processing_time: float = 0.0

    # 토론용 필드
    arguments: List[str] = field(default_factory=list)
    counter_arguments: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "agent_name": self.agent_name,
            "role": self.role.value,
            "verdict": self.verdict.value,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "evidence": self.evidence,
            "tool_results": [tr.to_dict() for tr in self.tool_results],
            "arguments": self.arguments,
            "processing_time": self.processing_time
        }


class BaseAgent(ABC):
    """
    포렌식 분석 에이전트 기본 클래스

    각 전문 에이전트는 이 클래스를 상속받아 구현합니다.
    Tool을 사용하여 분석을 수행하고, LLM으로 결과를 해석합니다.
    """

    def __init__(
        self,
        name: str,
        role: AgentRole,
        description: str,
        llm_model: Optional[str] = None
    ):
        self.name = name
        self.role = role
        self.description = description
        self.llm_model = llm_model
        self._tools = []
        self._trust_score = 0.8  # 초기 신뢰도

    @property
    def trust_score(self) -> float:
        return self._trust_score

    @trust_score.setter
    def trust_score(self, value: float):
        self._trust_score = max(0.0, min(1.0, value))

    def register_tool(self, tool) -> None:
        """Tool 등록"""
        self._tools.append(tool)

    @abstractmethod
    def analyze(self, image: np.ndarray, context: Optional[Dict] = None) -> AgentResponse:
        """
        이미지 분석 수행

        Args:
            image: RGB 이미지 (H, W, 3)
            context: 추가 컨텍스트 (다른 에이전트 결과 등)

        Returns:
            AgentResponse: 분석 결과 및 추론
        """
        pass

    @abstractmethod
    def generate_reasoning(
        self,
        tool_results: List[ToolResult],
        context: Optional[Dict] = None
    ) -> str:
        """
        Tool 결과를 바탕으로 추론 생성

        Args:
            tool_results: Tool 분석 결과들
            context: 추가 컨텍스트

        Returns:
            추론 설명 문자열
        """
        pass

    def respond_to_challenge(
        self,
        challenger_name: str,
        challenge: str,
        my_response: AgentResponse
    ) -> str:
        """
        다른 에이전트의 반론에 대응

        Args:
            challenger_name: 반론 제기한 에이전트 이름
            challenge: 반론 내용
            my_response: 내 원래 응답

        Returns:
            반박 또는 수정된 의견
        """
        # 기본 구현: 증거 기반 방어
        return (
            f"[{self.name}] {challenger_name}의 지적에 대해:\n"
            f"내 분석 결과 {my_response.verdict.value}은 다음 증거에 기반합니다:\n"
            f"- 신뢰도: {my_response.confidence:.2%}\n"
            f"- 주요 증거: {list(my_response.evidence.keys())}\n"
            f"추가 검토 필요 시 토론을 계속하겠습니다."
        )

    def update_trust(self, delta: float) -> None:
        """신뢰도 업데이트"""
        self.trust_score = self._trust_score + delta

    def get_system_prompt(self) -> str:
        """LLM용 시스템 프롬프트 반환"""
        return f"""당신은 {self.name}입니다. {self.description}

역할: {self.role.value}
전문 분야: {self._get_expertise()}

분석 시 다음 원칙을 따르세요:
1. 객관적인 증거에 기반하여 판단하세요
2. 불확실한 경우 명확히 표시하세요
3. 다른 전문가의 의견과 충돌 시 근거를 명확히 제시하세요
4. 신뢰도 점수를 정직하게 보고하세요
"""

    def _get_expertise(self) -> str:
        """전문 분야 설명"""
        expertise_map = {
            AgentRole.FREQUENCY: "FFT 기반 주파수 스펙트럼 분석, GAN 아티팩트 탐지",
            AgentRole.NOISE: "SRM 필터, PRNU 센서 노이즈 패턴 분석",
            AgentRole.WATERMARK: "HiNet 기반 워터마크 탐지 및 추출",
            AgentRole.SPATIAL: "ViT 기반 픽셀 수준 조작 영역 탐지",
            AgentRole.SEMANTIC: "VLM 기반 의미론적 불일치 탐지",
            AgentRole.MANAGER: "다중 에이전트 조율 및 최종 판단"
        }
        return expertise_map.get(self.role, "이미지 포렌식 분석")
