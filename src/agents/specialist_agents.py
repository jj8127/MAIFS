"""
전문가 에이전트들
각 포렌식 분석 영역을 담당하는 전문 에이전트 구현
"""
from typing import Dict, Optional, List
import numpy as np
import time

from .base_agent import BaseAgent, AgentRole, AgentResponse
from ..tools.base_tool import ToolResult, Verdict
from ..tools.frequency_tool import FrequencyAnalysisTool
from ..tools.noise_tool import NoiseAnalysisTool
from ..tools.watermark_tool import WatermarkTool
from ..tools.spatial_tool import SpatialAnalysisTool


class FrequencyAgent(BaseAgent):
    """
    주파수 분석 전문가 에이전트

    FFT 기반 주파수 스펙트럼을 분석하여
    GAN/Diffusion 생성 이미지의 특징적 패턴을 탐지합니다.
    """

    def __init__(self, llm_model: Optional[str] = None):
        super().__init__(
            name="주파수 분석 전문가 (Frequency Expert)",
            role=AgentRole.FREQUENCY,
            description="FFT 기반 주파수 스펙트럼 분석 전문가. "
                       "AI 생성 이미지의 격자 아티팩트와 비정상적 주파수 패턴을 탐지합니다.",
            llm_model=llm_model
        )
        self._tool = FrequencyAnalysisTool()
        self.register_tool(self._tool)

    def analyze(self, image: np.ndarray, context: Optional[Dict] = None) -> AgentResponse:
        """주파수 분석 수행"""
        start_time = time.time()

        # Tool 실행
        tool_result = self._tool(image)

        # 추론 생성
        reasoning = self.generate_reasoning([tool_result], context)

        # 주요 논거 추출
        arguments = self._extract_arguments(tool_result)

        processing_time = time.time() - start_time

        return AgentResponse(
            agent_name=self.name,
            role=self.role,
            verdict=tool_result.verdict,
            confidence=tool_result.confidence * self.trust_score,
            reasoning=reasoning,
            evidence=tool_result.evidence,
            tool_results=[tool_result],
            arguments=arguments,
            processing_time=processing_time
        )

    def generate_reasoning(
        self,
        tool_results: List[ToolResult],
        context: Optional[Dict] = None
    ) -> str:
        """추론 생성"""
        if not tool_results:
            return "분석 결과가 없습니다."

        result = tool_results[0]
        evidence = result.evidence

        reasoning_parts = [
            f"[주파수 분석 결과]",
            f"판정: {result.verdict.value}",
            f"신뢰도: {result.confidence:.2%}",
            "",
            "근거:"
        ]

        # 격자 아티팩트 분석
        if "grid_analysis" in evidence:
            grid = evidence["grid_analysis"]
            reasoning_parts.append(
                f"- 격자 패턴 분석: "
                f"수평 피크 {grid.get('horizontal_peaks', 0)}개, "
                f"수직 피크 {grid.get('vertical_peaks', 0)}개"
            )
            if grid.get("is_grid_pattern"):
                reasoning_parts.append(
                    "  → GAN 특유의 규칙적인 주파수 피크가 탐지됨"
                )

        # 고주파 분석
        if "high_frequency_analysis" in evidence:
            hf = evidence["high_frequency_analysis"]
            reasoning_parts.append(
                f"- 고주파/저주파 비율: {hf.get('hf_lf_ratio', 0):.3f}"
            )
            if hf.get("abnormality_score", 0) > 0.5:
                reasoning_parts.append(
                    "  → 자연 이미지와 다른 고주파 특성 발견"
                )

        return "\n".join(reasoning_parts)

    def _extract_arguments(self, tool_result: ToolResult) -> List[str]:
        """토론용 논거 추출"""
        arguments = []
        evidence = tool_result.evidence

        if evidence.get("ai_generation_score", 0) > 0.6:
            arguments.append(
                "주파수 스펙트럼에서 AI 생성 모델 특유의 패턴이 명확히 관찰됩니다."
            )

        if evidence.get("grid_analysis", {}).get("is_grid_pattern"):
            arguments.append(
                "격자형 아티팩트가 탐지되었으며, 이는 GAN 업샘플링의 전형적인 특징입니다."
            )

        if evidence.get("high_frequency_analysis", {}).get("abnormality_score", 0) > 0.5:
            arguments.append(
                "고주파 영역의 에너지 분포가 자연 이미지의 1/f 법칙을 벗어납니다."
            )

        return arguments


class NoiseAgent(BaseAgent):
    """
    노이즈 분석 전문가 에이전트

    SRM 필터와 PRNU 분석을 통해
    카메라 센서 노이즈와 AI 생성 노이즈를 구분합니다.
    """

    def __init__(self, llm_model: Optional[str] = None):
        super().__init__(
            name="노이즈 분석 전문가 (Noise Expert)",
            role=AgentRole.NOISE,
            description="SRM/PRNU 기반 노이즈 패턴 분석 전문가. "
                       "카메라 센서 고유 노이즈와 AI 생성 노이즈를 구분합니다.",
            llm_model=llm_model
        )
        self._tool = NoiseAnalysisTool()
        self.register_tool(self._tool)

    def analyze(self, image: np.ndarray, context: Optional[Dict] = None) -> AgentResponse:
        """노이즈 분석 수행"""
        start_time = time.time()

        tool_result = self._tool(image)
        reasoning = self.generate_reasoning([tool_result], context)
        arguments = self._extract_arguments(tool_result)

        processing_time = time.time() - start_time

        return AgentResponse(
            agent_name=self.name,
            role=self.role,
            verdict=tool_result.verdict,
            confidence=tool_result.confidence * self.trust_score,
            reasoning=reasoning,
            evidence=tool_result.evidence,
            tool_results=[tool_result],
            arguments=arguments,
            processing_time=processing_time
        )

    def generate_reasoning(
        self,
        tool_results: List[ToolResult],
        context: Optional[Dict] = None
    ) -> str:
        """추론 생성"""
        if not tool_results:
            return "분석 결과가 없습니다."

        result = tool_results[0]
        evidence = result.evidence

        reasoning_parts = [
            f"[노이즈 분석 결과]",
            f"판정: {result.verdict.value}",
            f"신뢰도: {result.confidence:.2%}",
            "",
            "근거:"
        ]

        # PRNU 분석
        if "prnu_stats" in evidence:
            prnu = evidence["prnu_stats"]
            reasoning_parts.append(
                f"- PRNU 분산: {prnu.get('variance', 0):.6f}"
            )
            reasoning_parts.append(
                f"- PRNU 첨도: {prnu.get('kurtosis', 0):.3f}"
            )

        # 일관성 분석
        if "consistency_analysis" in evidence:
            consistency = evidence["consistency_analysis"]
            reasoning_parts.append(
                f"- 노이즈 일관성 점수: {consistency.get('consistency_score', 0):.2%}"
            )

        # AI 탐지
        if "ai_detection" in evidence:
            ai = evidence["ai_detection"]
            reasoning_parts.append(
                f"- AI 생성 점수: {ai.get('ai_generation_score', 0):.2%}"
            )

        return "\n".join(reasoning_parts)

    def _extract_arguments(self, tool_result: ToolResult) -> List[str]:
        """토론용 논거 추출"""
        arguments = []
        evidence = tool_result.evidence

        ai_score = evidence.get("ai_detection", {}).get("ai_generation_score", 0)
        if ai_score > 0.6:
            arguments.append(
                "센서 노이즈(PRNU)가 거의 탐지되지 않아 AI 생성 이미지로 판단됩니다."
            )

        consistency = evidence.get("consistency_analysis", {}).get("consistency_score", 1)
        if consistency < 0.4:
            arguments.append(
                "이미지 영역별 노이즈 패턴이 불일치하여 부분 조작이 의심됩니다."
            )

        prnu_var = evidence.get("prnu_stats", {}).get("variance", 0)
        if prnu_var < 0.0001:
            arguments.append(
                "PRNU 분산이 매우 낮아 실제 카메라 촬영 이미지가 아닐 가능성이 높습니다."
            )

        return arguments


class WatermarkAgent(BaseAgent):
    """
    워터마크 분석 전문가 에이전트

    HiNet 기반 역변환 신경망을 사용하여
    비가시성 워터마크를 탐지하고 무결성을 검증합니다.
    """

    def __init__(self, llm_model: Optional[str] = None):
        super().__init__(
            name="워터마크 분석 전문가 (Watermark Expert)",
            role=AgentRole.WATERMARK,
            description="HiNet 기반 워터마크 탐지 전문가. "
                       "비가시성 워터마크를 추출하고 이미지 무결성을 검증합니다.",
            llm_model=llm_model
        )
        self._tool = WatermarkTool()
        self.register_tool(self._tool)

    def analyze(self, image: np.ndarray, context: Optional[Dict] = None) -> AgentResponse:
        """워터마크 분석 수행"""
        start_time = time.time()

        tool_result = self._tool(image)
        reasoning = self.generate_reasoning([tool_result], context)
        arguments = self._extract_arguments(tool_result)

        processing_time = time.time() - start_time

        return AgentResponse(
            agent_name=self.name,
            role=self.role,
            verdict=tool_result.verdict,
            confidence=tool_result.confidence * self.trust_score,
            reasoning=reasoning,
            evidence=tool_result.evidence,
            tool_results=[tool_result],
            arguments=arguments,
            processing_time=processing_time
        )

    def generate_reasoning(
        self,
        tool_results: List[ToolResult],
        context: Optional[Dict] = None
    ) -> str:
        """추론 생성"""
        if not tool_results:
            return "분석 결과가 없습니다."

        result = tool_results[0]
        evidence = result.evidence

        reasoning_parts = [
            f"[워터마크 분석 결과]",
            f"판정: {result.verdict.value}",
            f"신뢰도: {result.confidence:.2%}",
            "",
            "근거:"
        ]

        if "has_watermark" in evidence:
            has_wm = evidence["has_watermark"]
            ber = evidence.get("bit_error_rate", 1.0)

            reasoning_parts.append(
                f"- 워터마크 존재: {'예' if has_wm else '아니오'}"
            )
            reasoning_parts.append(
                f"- 비트 오류율 (BER): {ber:.2%}"
            )

            if has_wm:
                reasoning_parts.append(
                    "  → 유효한 워터마크가 복원되어 원본 무결성이 확인됨"
                )
            else:
                reasoning_parts.append(
                    "  → 워터마크가 없거나 손상됨 (조작 또는 비보호 이미지)"
                )

        return "\n".join(reasoning_parts)

    def _extract_arguments(self, tool_result: ToolResult) -> List[str]:
        """토론용 논거 추출"""
        arguments = []
        evidence = tool_result.evidence

        if evidence.get("has_watermark"):
            arguments.append(
                "유효한 워터마크가 탐지되어 이미지의 원본 무결성이 확인됩니다."
            )
        else:
            ber = evidence.get("bit_error_rate", 1.0)
            if ber > 0.5:
                arguments.append(
                    f"워터마크가 탐지되지 않았습니다 (BER: {ber:.2%}). "
                    "워터마크가 없는 원본이거나 조작으로 워터마크가 손상되었을 수 있습니다."
                )

        return arguments


class SpatialAgent(BaseAgent):
    """
    공간 분석 전문가 에이전트

    ViT 기반 모델을 사용하여
    픽셀 수준의 조작 영역을 탐지합니다.
    """

    def __init__(self, llm_model: Optional[str] = None):
        super().__init__(
            name="공간 분석 전문가 (Spatial Expert)",
            role=AgentRole.SPATIAL,
            description="ViT 기반 공간 분석 전문가. "
                       "픽셀 수준에서 조작된 영역을 탐지하고 마스크로 시각화합니다.",
            llm_model=llm_model
        )
        self._tool = SpatialAnalysisTool()
        self.register_tool(self._tool)

    def analyze(self, image: np.ndarray, context: Optional[Dict] = None) -> AgentResponse:
        """공간 분석 수행"""
        start_time = time.time()

        tool_result = self._tool(image)
        reasoning = self.generate_reasoning([tool_result], context)
        arguments = self._extract_arguments(tool_result)

        processing_time = time.time() - start_time

        return AgentResponse(
            agent_name=self.name,
            role=self.role,
            verdict=tool_result.verdict,
            confidence=tool_result.confidence * self.trust_score,
            reasoning=reasoning,
            evidence=tool_result.evidence,
            tool_results=[tool_result],
            arguments=arguments,
            processing_time=processing_time
        )

    def generate_reasoning(
        self,
        tool_results: List[ToolResult],
        context: Optional[Dict] = None
    ) -> str:
        """추론 생성"""
        if not tool_results:
            return "분석 결과가 없습니다."

        result = tool_results[0]
        evidence = result.evidence

        reasoning_parts = [
            f"[공간 분석 결과]",
            f"판정: {result.verdict.value}",
            f"신뢰도: {result.confidence:.2%}",
            "",
            "근거:"
        ]

        if "manipulation_ratio" in evidence:
            ratio = evidence["manipulation_ratio"]
            reasoning_parts.append(
                f"- 조작 영역 비율: {ratio:.2%}"
            )

            if ratio < 0.05:
                reasoning_parts.append(
                    "  → 조작 영역이 거의 탐지되지 않음 (원본)"
                )
            elif ratio > 0.8:
                reasoning_parts.append(
                    "  → 이미지 대부분이 생성/조작됨 (AI 생성 의심)"
                )
            else:
                reasoning_parts.append(
                    "  → 부분적 조작 영역 탐지 (합성/편집 의심)"
                )

        return "\n".join(reasoning_parts)

    def _extract_arguments(self, tool_result: ToolResult) -> List[str]:
        """토론용 논거 추출"""
        arguments = []
        evidence = tool_result.evidence

        ratio = evidence.get("manipulation_ratio", 0)

        if ratio > 0.8:
            arguments.append(
                f"이미지의 {ratio:.0%}가 조작/생성된 것으로 탐지되어 AI 생성 이미지로 판단됩니다."
            )
        elif ratio > 0.05:
            arguments.append(
                f"이미지의 {ratio:.0%} 영역에서 조작 흔적이 탐지되었습니다. "
                "마스크를 통해 조작 위치를 확인할 수 있습니다."
            )
        else:
            arguments.append(
                "픽셀 수준 분석에서 유의미한 조작 영역이 탐지되지 않았습니다."
            )

        return arguments
