"""
전문가 에이전트들
각 포렌식 분석 영역을 담당하는 전문 에이전트 구현

LLM 통합으로 도메인 지식 기반 추론 및 토론 기능 지원
"""
from typing import Dict, Optional, List
import numpy as np
import time

from .base_agent import BaseAgent, AgentRole, AgentResponse
from ..tools.base_tool import ToolResult, Verdict
from ..tools.catnet_tool import CATNetAnalysisTool
from ..tools.noise_tool import NoiseAnalysisTool
from ..tools.fatformer_tool import FatFormerTool
from ..tools.spatial_tool import SpatialAnalysisTool
from ..llm.subagent_llm import SubAgentLLM, AgentDomain, DebateResponse


class FrequencyAgent(BaseAgent):
    """
    압축/주파수 분석 전문가 에이전트

    CAT-Net 압축 아티팩트 분석을 우선 사용하고,
    필요 시 주파수 기반 fallback으로 조작 신호를 탐지합니다.

    LLM 통합: 도메인 지식 기반 추론 및 토론 기능 지원
    """

    def __init__(self, llm_model: Optional[str] = None, use_llm: bool = True):
        super().__init__(
            name="압축 아티팩트 분석 전문가 (Compression Expert)",
            role=AgentRole.FREQUENCY,
            description="CAT-Net 기반 압축 아티팩트 분석 전문가. "
                       "JPEG 이중 압축/압축 흔적 기반 조작 영역을 탐지합니다.",
            llm_model=llm_model
        )
        self._tool = CATNetAnalysisTool()
        self.register_tool(self._tool)

        # LLM 통합
        self._use_llm = use_llm
        self._llm = SubAgentLLM(AgentDomain.FREQUENCY, model=llm_model) if use_llm else None

    def analyze(self, image: np.ndarray, context: Optional[Dict] = None) -> AgentResponse:
        """압축/주파수 분석 수행"""
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
        """추론 생성 - LLM 사용 시 도메인 지식 기반 해석"""
        if not tool_results:
            return "분석 결과가 없습니다."

        result = tool_results[0]
        evidence = result.evidence

        # LLM 기반 추론 (가능한 경우)
        if self._use_llm and self._llm and self._llm.is_available:
            try:
                reasoning_result = self._llm.interpret_results(evidence, context)
                return self._format_llm_reasoning(result, reasoning_result)
            except Exception as e:
                print(f"[FrequencyAgent] LLM 추론 실패, 규칙 기반 사용: {e}")

        # 규칙 기반 추론 (폴백)
        return self._generate_rule_based_reasoning(result, evidence)

    def _format_llm_reasoning(self, result: ToolResult, reasoning_result) -> str:
        """LLM 추론 결과 포맷팅"""
        parts = [
            f"[압축 아티팩트 분석 결과 - LLM 해석]",
            f"판정: {result.verdict.value}",
            f"신뢰도: {result.confidence:.2%}",
            "",
            "## 해석",
            reasoning_result.interpretation,
            "",
            "## 추론",
            reasoning_result.reasoning,
            "",
            "## 판정 근거",
            reasoning_result.verdict_rationale,
        ]

        if reasoning_result.key_findings:
            parts.extend(["", "## 핵심 발견"])
            for finding in reasoning_result.key_findings:
                parts.append(f"- {finding}")

        if reasoning_result.uncertainties:
            parts.extend(["", "## 불확실성"])
            for uncertainty in reasoning_result.uncertainties:
                parts.append(f"- {uncertainty}")

        return "\n".join(parts)

    def _generate_rule_based_reasoning(self, result: ToolResult, evidence: Dict) -> str:
        """규칙 기반 추론 생성 (LLM 폴백)"""
        reasoning_parts = [
            f"[압축 아티팩트 분석 결과]",
            f"판정: {result.verdict.value}",
            f"신뢰도: {result.confidence:.2%}",
            "",
            "근거:"
        ]

        if "compression_artifact_score" in evidence:
            reasoning_parts.append(
                f"- 압축 아티팩트 점수: {evidence.get('compression_artifact_score', 0):.3f}"
            )
            reasoning_parts.append(
                f"- 추정 조작 비율: {evidence.get('manipulation_ratio', 0):.2%}"
            )

        if evidence.get("backend") == "frequency_fallback":
            reasoning_parts.append(
                f"- CAT-Net 비가용으로 fallback 사용: {evidence.get('catnet_error', 'unknown')}"
            )

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

    def respond_to_challenge(
        self,
        challenger_name: str,
        challenge: str,
        my_response: AgentResponse
    ) -> DebateResponse:
        """
        다른 에이전트의 반론에 대응 - LLM 기반 토론

        Args:
            challenger_name: 반론 제기한 에이전트 이름
            challenge: 반론 내용
            my_response: 내 원래 응답

        Returns:
            DebateResponse: 토론 응답
        """
        # LLM 기반 응답 (가능한 경우)
        if self._use_llm and self._llm and self._llm.is_available:
            return self._llm.respond_to_challenge(
                challenger_name=challenger_name,
                challenge=challenge,
                my_verdict=my_response.verdict.value,
                my_confidence=my_response.confidence,
                my_evidence=my_response.evidence,
                my_reasoning=my_response.reasoning
            )

        # 규칙 기반 응답 (폴백)
        return DebateResponse(
            response_type="defense",
            content=(
                f"[{self.name}] {challenger_name}의 지적에 대해:\n"
                f"내 분석 결과 {my_response.verdict.value}은 다음 증거에 기반합니다:\n"
                f"- 신뢰도: {my_response.confidence:.2%}\n"
                f"- 주요 증거: {list(my_response.evidence.keys())}\n"
                f"추가 검토 필요 시 토론을 계속하겠습니다."
            ),
            verdict_changed=False,
            reasoning="규칙 기반 방어 응답"
        )

    def generate_challenge(
        self,
        target_response: AgentResponse,
        my_response: Optional[AgentResponse] = None
    ) -> str:
        """
        다른 에이전트에 대한 반론 생성

        Args:
            target_response: 대상 에이전트의 응답
            my_response: 내 응답 (선택적)

        Returns:
            str: 반론 내용
        """
        my_verdict = my_response.verdict.value if my_response else ""
        my_evidence = my_response.evidence if my_response else {}

        if self._use_llm and self._llm and self._llm.is_available:
            return self._llm.generate_challenge(
                target_verdict=target_response.verdict.value,
                target_confidence=target_response.confidence,
                target_evidence=target_response.evidence,
                my_verdict=my_verdict,
                my_evidence=my_evidence
            )

        return (
            f"압축 아티팩트 분석 관점에서 {target_response.verdict.value} 판정에 대해 "
            f"질문드립니다. JPEG/압축 도메인 증거를 제시해주실 수 있습니까?"
        )

    def _extract_arguments(self, tool_result: ToolResult) -> List[str]:
        """토론용 논거 추출"""
        arguments = []
        evidence = tool_result.evidence

        if evidence.get("compression_artifact_score", 0) > 0.12:
            arguments.append(
                "압축 아티팩트 점수가 높아 조작 영역이 존재할 가능성이 큽니다."
            )

        if evidence.get("backend") == "frequency_fallback":
            arguments.append(
                "현재 CAT-Net 체크포인트/환경 이슈로 fallback 분석을 사용했습니다."
            )

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

    LLM 통합: 도메인 지식 기반 추론 및 토론 기능 지원
    """

    def __init__(self, llm_model: Optional[str] = None, use_llm: bool = True):
        super().__init__(
            name="노이즈 분석 전문가 (Noise Expert)",
            role=AgentRole.NOISE,
            description="SRM/PRNU 기반 노이즈 패턴 분석 전문가. "
                       "카메라 센서 고유 노이즈와 AI 생성 노이즈를 구분합니다.",
            llm_model=llm_model
        )
        self._tool = NoiseAnalysisTool()
        self.register_tool(self._tool)

        # LLM 통합
        self._use_llm = use_llm
        self._llm = SubAgentLLM(AgentDomain.NOISE, model=llm_model) if use_llm else None

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
        """추론 생성 - LLM 사용 시 도메인 지식 기반 해석"""
        if not tool_results:
            return "분석 결과가 없습니다."

        result = tool_results[0]
        evidence = result.evidence

        # LLM 기반 추론 (가능한 경우)
        if self._use_llm and self._llm and self._llm.is_available:
            try:
                reasoning_result = self._llm.interpret_results(evidence, context)
                return self._format_llm_reasoning(result, reasoning_result)
            except Exception as e:
                print(f"[NoiseAgent] LLM 추론 실패, 규칙 기반 사용: {e}")

        # 규칙 기반 추론 (폴백)
        return self._generate_rule_based_reasoning(result, evidence)

    def _format_llm_reasoning(self, result: ToolResult, reasoning_result) -> str:
        """LLM 추론 결과 포맷팅"""
        parts = [
            f"[노이즈 분석 결과 - LLM 해석]",
            f"판정: {result.verdict.value}",
            f"신뢰도: {result.confidence:.2%}",
            "",
            "## 해석",
            reasoning_result.interpretation,
            "",
            "## 추론",
            reasoning_result.reasoning,
            "",
            "## 판정 근거",
            reasoning_result.verdict_rationale,
        ]

        if reasoning_result.key_findings:
            parts.extend(["", "## 핵심 발견"])
            for finding in reasoning_result.key_findings:
                parts.append(f"- {finding}")

        if reasoning_result.uncertainties:
            parts.extend(["", "## 불확실성"])
            for uncertainty in reasoning_result.uncertainties:
                parts.append(f"- {uncertainty}")

        return "\n".join(parts)

    def _generate_rule_based_reasoning(self, result: ToolResult, evidence: Dict) -> str:
        """규칙 기반 추론 생성 (LLM 폴백)"""
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

    def respond_to_challenge(
        self,
        challenger_name: str,
        challenge: str,
        my_response: AgentResponse
    ) -> DebateResponse:
        """다른 에이전트의 반론에 대응 - LLM 기반 토론"""
        if self._use_llm and self._llm and self._llm.is_available:
            return self._llm.respond_to_challenge(
                challenger_name=challenger_name,
                challenge=challenge,
                my_verdict=my_response.verdict.value,
                my_confidence=my_response.confidence,
                my_evidence=my_response.evidence,
                my_reasoning=my_response.reasoning
            )

        return DebateResponse(
            response_type="defense",
            content=(
                f"[{self.name}] {challenger_name}의 지적에 대해:\n"
                f"PRNU/SRM 분석 결과 {my_response.verdict.value}은 "
                f"신뢰도 {my_response.confidence:.2%}의 증거에 기반합니다."
            ),
            verdict_changed=False,
            reasoning="규칙 기반 방어 응답"
        )

    def generate_challenge(
        self,
        target_response: AgentResponse,
        my_response: Optional[AgentResponse] = None
    ) -> str:
        """다른 에이전트에 대한 반론 생성"""
        my_verdict = my_response.verdict.value if my_response else ""
        my_evidence = my_response.evidence if my_response else {}

        if self._use_llm and self._llm and self._llm.is_available:
            return self._llm.generate_challenge(
                target_verdict=target_response.verdict.value,
                target_confidence=target_response.confidence,
                target_evidence=target_response.evidence,
                my_verdict=my_verdict,
                my_evidence=my_evidence
            )

        return (
            f"노이즈 분석 관점에서 {target_response.verdict.value} 판정에 대해 "
            f"질문드립니다. 센서 노이즈(PRNU) 패턴에 대한 분석 결과가 있습니까?"
        )


class FatFormerAgent(BaseAgent):
    """
    AI 생성 탐지 전문가 에이전트 (FatFormer)

    CLIP ViT-L/14 + Forgery-Aware Adapter를 사용하여
    AI 생성 이미지를 탐지합니다.

    LLM 통합: 도메인 지식 기반 추론 및 토론 기능 지원
    """

    def __init__(self, llm_model: Optional[str] = None, use_llm: bool = True):
        super().__init__(
            name="AI 생성 탐지 전문가 (FatFormer Expert)",
            role=AgentRole.FATFORMER,
            description="FatFormer (CLIP+DWT) 기반 AI 생성 이미지 탐지 전문가. "
                       "의미론적 특징과 주파수 특징을 결합하여 AI 생성 여부를 판별합니다.",
            llm_model=llm_model
        )
        self._tool = FatFormerTool()
        self.register_tool(self._tool)

        # LLM 통합
        self._use_llm = use_llm
        self._llm = SubAgentLLM(AgentDomain.FATFORMER, model=llm_model) if use_llm else None

    def analyze(self, image: np.ndarray, context: Optional[Dict] = None) -> AgentResponse:
        """AI 생성 탐지 분석 수행"""
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
        """추론 생성 - LLM 사용 시 도메인 지식 기반 해석"""
        if not tool_results:
            return "분석 결과가 없습니다."

        result = tool_results[0]
        evidence = result.evidence

        # LLM 기반 추론 (가능한 경우)
        if self._use_llm and self._llm and self._llm.is_available:
            try:
                reasoning_result = self._llm.interpret_results(evidence, context)
                return self._format_llm_reasoning(result, reasoning_result)
            except Exception as e:
                print(f"[FatFormerAgent] LLM 추론 실패, 규칙 기반 사용: {e}")

        # 규칙 기반 추론 (폴백)
        return self._generate_rule_based_reasoning(result, evidence)

    def _format_llm_reasoning(self, result: ToolResult, reasoning_result) -> str:
        """LLM 추론 결과 포맷팅"""
        parts = [
            f"[AI 생성 탐지 결과 - LLM 해석]",
            f"판정: {result.verdict.value}",
            f"신뢰도: {result.confidence:.2%}",
            "",
            "## 해석",
            reasoning_result.interpretation,
            "",
            "## 추론",
            reasoning_result.reasoning,
            "",
            "## 판정 근거",
            reasoning_result.verdict_rationale,
        ]

        if reasoning_result.key_findings:
            parts.extend(["", "## 핵심 발견"])
            for finding in reasoning_result.key_findings:
                parts.append(f"- {finding}")

        if reasoning_result.uncertainties:
            parts.extend(["", "## 불확실성"])
            for uncertainty in reasoning_result.uncertainties:
                parts.append(f"- {uncertainty}")

        return "\n".join(parts)

    def _generate_rule_based_reasoning(self, result: ToolResult, evidence: Dict) -> str:
        """규칙 기반 추론 생성 (LLM 폴백)"""
        reasoning_parts = [
            f"[AI 생성 탐지 결과 (FatFormer)]",
            f"판정: {result.verdict.value}",
            f"신뢰도: {result.confidence:.2%}",
            "",
            "근거:"
        ]

        fake_prob = evidence.get("fake_probability")
        real_prob = evidence.get("real_probability")

        if fake_prob is not None:
            reasoning_parts.append(
                f"- AI 생성 확률: {fake_prob:.2%}"
            )
            reasoning_parts.append(
                f"- 실제 이미지 확률: {real_prob:.2%}"
            )

            if fake_prob > 0.7:
                reasoning_parts.append(
                    "  → CLIP 의미론적 분석과 DWT 주파수 분석 모두에서 AI 생성 특징 감지"
                )
            elif fake_prob > 0.5:
                reasoning_parts.append(
                    "  → AI 생성 가능성이 높으나 경계 사례로 추가 검증 권장"
                )
            elif fake_prob > 0.3:
                reasoning_parts.append(
                    "  → AI 생성 여부 불확실, 다른 분석 결과와의 교차 검증 필요"
                )
            else:
                reasoning_parts.append(
                    "  → AI 생성 특징이 감지되지 않음, 실제 이미지로 판단"
                )

        if evidence.get("fallback_mode"):
            reasoning_parts.append(
                "- ⚠ Fallback 모드: 모델 미로드로 기본 분석만 수행됨"
            )

        return "\n".join(reasoning_parts)

    def _extract_arguments(self, tool_result: ToolResult) -> List[str]:
        """토론용 논거 추출"""
        arguments = []
        evidence = tool_result.evidence

        fake_prob = evidence.get("fake_probability", 0)

        if fake_prob > 0.7:
            arguments.append(
                f"CLIP ViT-L/14 의미론적 분석과 DWT 주파수 경로 분석 모두에서 "
                f"AI 생성 특징이 명확히 감지되었습니다 (AI 생성 확률: {fake_prob:.2%})."
            )
        elif fake_prob > 0.5:
            arguments.append(
                f"FatFormer가 AI 생성 이미지로 분류했으나 경계 사례입니다 "
                f"(AI 생성 확률: {fake_prob:.2%}). "
                f"Forgery-Aware Adapter의 주파수 경로에서 약한 위조 신호가 감지되었습니다."
            )
        elif fake_prob > 0:
            arguments.append(
                f"FatFormer의 CLIP 기반 의미론적 분석에서 AI 생성 특징이 "
                f"감지되지 않았습니다 (AI 생성 확률: {fake_prob:.2%})."
            )

        return arguments

    def respond_to_challenge(
        self,
        challenger_name: str,
        challenge: str,
        my_response: AgentResponse
    ) -> DebateResponse:
        """다른 에이전트의 반론에 대응 - LLM 기반 토론"""
        if self._use_llm and self._llm and self._llm.is_available:
            return self._llm.respond_to_challenge(
                challenger_name=challenger_name,
                challenge=challenge,
                my_verdict=my_response.verdict.value,
                my_confidence=my_response.confidence,
                my_evidence=my_response.evidence,
                my_reasoning=my_response.reasoning
            )

        return DebateResponse(
            response_type="defense",
            content=(
                f"[{self.name}] {challenger_name}의 지적에 대해:\n"
                f"FatFormer 분석 결과 {my_response.verdict.value}은 "
                f"AI 생성 확률 {my_response.evidence.get('fake_probability', 'N/A'):.2%}에 기반합니다."
            ),
            verdict_changed=False,
            reasoning="규칙 기반 방어 응답"
        )

    def generate_challenge(
        self,
        target_response: AgentResponse,
        my_response: Optional[AgentResponse] = None
    ) -> str:
        """다른 에이전트에 대한 반론 생성"""
        my_verdict = my_response.verdict.value if my_response else ""
        my_evidence = my_response.evidence if my_response else {}

        if self._use_llm and self._llm and self._llm.is_available:
            return self._llm.generate_challenge(
                target_verdict=target_response.verdict.value,
                target_confidence=target_response.confidence,
                target_evidence=target_response.evidence,
                my_verdict=my_verdict,
                my_evidence=my_evidence
            )

        return (
            f"CLIP 기반 의미론적 분석 관점에서 {target_response.verdict.value} 판정에 대해 "
            f"질문드립니다. 이미지의 의미론적 특징과 주파수 특징에서 AI 생성 흔적이 확인되었습니까?"
        )


class SpatialAgent(BaseAgent):
    """
    공간 분석 전문가 에이전트

    ViT 기반 모델을 사용하여
    픽셀 수준의 조작 영역을 탐지합니다.

    LLM 통합: 도메인 지식 기반 추론 및 토론 기능 지원
    """

    def __init__(self, llm_model: Optional[str] = None, use_llm: bool = True):
        super().__init__(
            name="공간 분석 전문가 (Spatial Expert)",
            role=AgentRole.SPATIAL,
            description="ViT 기반 공간 분석 전문가. "
                       "픽셀 수준에서 조작된 영역을 탐지하고 마스크로 시각화합니다.",
            llm_model=llm_model
        )
        self._tool = SpatialAnalysisTool()
        self.register_tool(self._tool)

        # LLM 통합
        self._use_llm = use_llm
        self._llm = SubAgentLLM(AgentDomain.SPATIAL, model=llm_model) if use_llm else None

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
        """추론 생성 - LLM 사용 시 도메인 지식 기반 해석"""
        if not tool_results:
            return "분석 결과가 없습니다."

        result = tool_results[0]
        evidence = result.evidence

        # LLM 기반 추론 (가능한 경우)
        if self._use_llm and self._llm and self._llm.is_available:
            try:
                reasoning_result = self._llm.interpret_results(evidence, context)
                return self._format_llm_reasoning(result, reasoning_result)
            except Exception as e:
                print(f"[SpatialAgent] LLM 추론 실패, 규칙 기반 사용: {e}")

        # 규칙 기반 추론 (폴백)
        return self._generate_rule_based_reasoning(result, evidence)

    def _format_llm_reasoning(self, result: ToolResult, reasoning_result) -> str:
        """LLM 추론 결과 포맷팅"""
        parts = [
            f"[공간 분석 결과 - LLM 해석]",
            f"판정: {result.verdict.value}",
            f"신뢰도: {result.confidence:.2%}",
            "",
            "## 해석",
            reasoning_result.interpretation,
            "",
            "## 추론",
            reasoning_result.reasoning,
            "",
            "## 판정 근거",
            reasoning_result.verdict_rationale,
        ]

        if reasoning_result.key_findings:
            parts.extend(["", "## 핵심 발견"])
            for finding in reasoning_result.key_findings:
                parts.append(f"- {finding}")

        if reasoning_result.uncertainties:
            parts.extend(["", "## 불확실성"])
            for uncertainty in reasoning_result.uncertainties:
                parts.append(f"- {uncertainty}")

        return "\n".join(parts)

    def _generate_rule_based_reasoning(self, result: ToolResult, evidence: Dict) -> str:
        """규칙 기반 추론 생성 (LLM 폴백)"""
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

    def respond_to_challenge(
        self,
        challenger_name: str,
        challenge: str,
        my_response: AgentResponse
    ) -> DebateResponse:
        """다른 에이전트의 반론에 대응 - LLM 기반 토론"""
        if self._use_llm and self._llm and self._llm.is_available:
            return self._llm.respond_to_challenge(
                challenger_name=challenger_name,
                challenge=challenge,
                my_verdict=my_response.verdict.value,
                my_confidence=my_response.confidence,
                my_evidence=my_response.evidence,
                my_reasoning=my_response.reasoning
            )

        return DebateResponse(
            response_type="defense",
            content=(
                f"[{self.name}] {challenger_name}의 지적에 대해:\n"
                f"공간 분석 결과 {my_response.verdict.value}은 "
                f"조작 영역 {my_response.evidence.get('manipulation_ratio', 0):.1%}에 기반합니다."
            ),
            verdict_changed=False,
            reasoning="규칙 기반 방어 응답"
        )

    def generate_challenge(
        self,
        target_response: AgentResponse,
        my_response: Optional[AgentResponse] = None
    ) -> str:
        """다른 에이전트에 대한 반론 생성"""
        my_verdict = my_response.verdict.value if my_response else ""
        my_evidence = my_response.evidence if my_response else {}

        if self._use_llm and self._llm and self._llm.is_available:
            return self._llm.generate_challenge(
                target_verdict=target_response.verdict.value,
                target_confidence=target_response.confidence,
                target_evidence=target_response.evidence,
                my_verdict=my_verdict,
                my_evidence=my_evidence
            )

        return (
            f"공간 분석 관점에서 {target_response.verdict.value} 판정에 대해 "
            f"질문드립니다. 픽셀 수준의 조작 영역 분석 결과는 어떻습니까?"
        )
