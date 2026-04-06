"""
MAIFS Tool 기본 클래스
모든 포렌식 분석 도구의 기본 인터페이스 정의
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List
from pathlib import Path
import numpy as np
from enum import Enum


class ConfidenceLevel(Enum):
    """신뢰도 수준"""
    VERY_LOW = "very_low"      # 0.0 - 0.2
    LOW = "low"                # 0.2 - 0.4
    MEDIUM = "medium"          # 0.4 - 0.6
    HIGH = "high"              # 0.6 - 0.8
    VERY_HIGH = "very_high"    # 0.8 - 1.0


class Verdict(Enum):
    """판정 결과"""
    AUTHENTIC = "authentic"          # 원본 이미지
    MANIPULATED = "manipulated"      # 조작된 이미지
    AI_GENERATED = "ai_generated"    # AI 생성 이미지
    UNCERTAIN = "uncertain"          # 판단 불가


@dataclass
class ToolResult:
    """Tool 분석 결과"""
    tool_name: str
    verdict: Verdict
    confidence: float  # 0.0 ~ 1.0
    evidence: Dict[str, Any] = field(default_factory=dict)
    explanation: str = ""
    manipulation_mask: Optional[np.ndarray] = None  # 조작 영역 마스크
    raw_output: Optional[Any] = None
    processing_time: float = 0.0  # 초 단위

    @property
    def confidence_level(self) -> ConfidenceLevel:
        """신뢰도 수치를 레벨로 변환"""
        if self.confidence < 0.2:
            return ConfidenceLevel.VERY_LOW
        elif self.confidence < 0.4:
            return ConfidenceLevel.LOW
        elif self.confidence < 0.6:
            return ConfidenceLevel.MEDIUM
        elif self.confidence < 0.8:
            return ConfidenceLevel.HIGH
        else:
            return ConfidenceLevel.VERY_HIGH

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환 (직렬화용)"""
        return {
            "tool_name": self.tool_name,
            "verdict": self.verdict.value,
            "confidence": self.confidence,
            "confidence_level": self.confidence_level.value,
            "evidence": self.evidence,
            "explanation": self.explanation,
            "has_mask": self.manipulation_mask is not None,
            "processing_time": self.processing_time
        }


class BaseTool(ABC):
    """
    포렌식 분석 도구 기본 클래스

    모든 전문 분석 도구는 이 클래스를 상속받아 구현합니다.
    LangChain/CrewAI와 호환되는 인터페이스를 제공합니다.
    """

    def __init__(self, name: str, description: str, device: str = "cuda"):
        self.name = name
        self.description = description
        self.device = device
        self._model = None
        self._is_loaded = False

    @property
    def tool_name(self) -> str:
        return self.name

    @property
    def tool_description(self) -> str:
        return self.description

    @abstractmethod
    def load_model(self) -> None:
        """모델 로드"""
        pass

    @abstractmethod
    def analyze(self, image: np.ndarray) -> ToolResult:
        """
        이미지 분석 실행

        Args:
            image: RGB 이미지 (H, W, 3) numpy array

        Returns:
            ToolResult: 분석 결과
        """
        pass

    def __call__(self, image: np.ndarray) -> ToolResult:
        """Tool을 함수처럼 호출"""
        if not self._is_loaded:
            self.load_model()
        return self.analyze(image)

    def unload_model(self) -> None:
        """모델 언로드 (메모리 해제)"""
        self._model = None
        self._is_loaded = False

    def get_schema(self) -> Dict[str, Any]:
        """LangChain Tool 스키마 반환"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "분석할 이미지 파일 경로"
                    }
                },
                "required": ["image_path"]
            }
        }
