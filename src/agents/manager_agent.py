"""
Manager Agent
다중 에이전트를 조율하고 최종 판단을 내리는 총괄 에이전트
Claude API를 활용한 지능형 판단 시스템
"""
from typing import Dict, Optional, List, Any
import numpy as np
import time
import json
from dataclasses import dataclass, field

from .base_agent import BaseAgent, AgentRole, AgentResponse
from .specialist_agents import (
    FrequencyAgent,
    NoiseAgent,
    FatFormerAgent,
    SpatialAgent
)
from ..consensus.cobra import COBRAConsensus, ConsensusResult
from ..debate.debate_chamber import DebateChamber, DebateResult
from ..tools.base_tool import Verdict
from configs.trust import resolve_trust

# LLM 클라이언트 (선택적)
try:
    from ..llm.claude_client import ClaudeClient, LLMResponse
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    ClaudeClient = None
    LLMResponse = None


@dataclass
class ForensicReport:
    """포렌식 분석 최종 보고서"""
    final_verdict: Verdict
    confidence: float
    summary: str
    detailed_reasoning: str
    agent_responses: Dict[str, AgentResponse] = field(default_factory=dict)
    consensus_info: Dict[str, Any] = field(default_factory=dict)
    debate_history: List[Dict] = field(default_factory=list)
    consensus_result: Optional[ConsensusResult] = None
    debate_result: Optional[DebateResult] = None
    total_processing_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "final_verdict": self.final_verdict.value,
            "confidence": self.confidence,
            "summary": self.summary,
            "detailed_reasoning": self.detailed_reasoning,
            "agent_responses": {
                k: v.to_dict() for k, v in self.agent_responses.items()
            },
            "consensus_info": self.consensus_info,
            "consensus": self.consensus_result.to_dict() if self.consensus_result else None,
            "debate": self.debate_result.to_dict() if self.debate_result else None,
            "debate_rounds": len(self.debate_history),
            "total_processing_time": self.total_processing_time
        }


class ManagerAgent(BaseAgent):
    """
    Manager Agent (총괄 관리자)

    역할:
    1. 전문가 에이전트들에게 분석 작업 분배
    2. 분석 결과 수집 및 종합
    3. COBRA 합의 알고리즘으로 최종 판단
    4. 불일치 발생 시 토론 조율
    5. 최종 판정 및 설명 생성
    """

    SYSTEM_PROMPT = """당신은 MAIFS(Multi-Agent Image Forensic System)의 Manager Agent입니다.

역할: 다중 이미지 포렌식 전문가들을 조율하여 이미지의 진위 여부를 판단합니다.

전문가 팀:
1. 주파수 분석 전문가: FFT 기반 GAN 아티팩트 탐지
2. 노이즈 분석 전문가: PRNU/SRM 기반 센서 노이즈 분석
3. AI 생성 탐지 전문가: FatFormer (CLIP ViT-L/14 + DWT) 기반 AI 이미지 탐지
4. 공간 분석 전문가: ViT 기반 조작 영역 탐지

판단 기준:
- AUTHENTIC: 원본 이미지로 확인됨
- MANIPULATED: 부분적으로 조작/편집됨
- AI_GENERATED: AI에 의해 생성된 이미지
- UNCERTAIN: 판단 불가, 추가 분석 필요

분석 시:
1. 각 전문가의 분석 결과를 종합하세요
2. 상충되는 의견이 있으면 증거의 강도를 비교하세요
3. COBRA 합의 알고리즘에 따라 신뢰도를 가중 평균하세요
4. 최종 판정에 대한 명확한 근거를 제시하세요
"""

    def __init__(
        self,
        llm_model: str = "claude-sonnet-4-20250514",
        api_key: Optional[str] = None,
        use_llm: bool = True,
        consensus_algorithm: str = "drwa",
        enable_debate: bool = True,
        debate_threshold: float = 0.3,
        trust_scores: Optional[Dict[str, float]] = None,
        trust_metrics: Optional[Dict[str, Dict[str, float]]] = None,
    ):
        super().__init__(
            name="Manager Agent (총괄 관리자)",
            role=AgentRole.MANAGER,
            description="다중 에이전트 조율 및 최종 판단을 담당하는 총괄 관리자",
            llm_model=llm_model
        )

        # 전문가 에이전트 초기화
        self.specialists: Dict[str, BaseAgent] = {
            "frequency": FrequencyAgent(),
            "noise": NoiseAgent(),
            "fatformer": FatFormerAgent(),
            "spatial": SpatialAgent(),
        }

        # 에이전트별 신뢰도 (COBRA용, configs/trust.py SSOT)
        self.agent_trust: Dict[str, float] = resolve_trust(
            trust_scores,
            metrics_override=trust_metrics,
        )
        self.consensus_algorithm = consensus_algorithm
        self.enable_debate = enable_debate
        self.debate_threshold = debate_threshold

        # MAIFS와 동일한 합의/토론 엔진을 사용해 경로 분기를 없앤다.
        self.consensus_engine = COBRAConsensus(default_algorithm=consensus_algorithm)
        self.debate_chamber = DebateChamber(
            consensus_engine=self.consensus_engine,
            disagreement_threshold=debate_threshold
        )

        # LLM 클라이언트 초기화
        self.use_llm = use_llm and LLM_AVAILABLE
        self.llm_client: Optional[ClaudeClient] = None

        if self.use_llm and ClaudeClient is not None:
            self.llm_client = ClaudeClient(api_key=api_key, model=llm_model)
            if not self.llm_client.is_available:
                print("[ManagerAgent] LLM 사용 불가, 규칙 기반 모드로 전환")
                self.use_llm = False

    def analyze(self, image: np.ndarray, context: Optional[Dict] = None) -> ForensicReport:
        """
        전체 포렌식 분석 수행

        Args:
            image: 분석할 RGB 이미지
            context: 추가 컨텍스트 (메타데이터 등)

        Returns:
            ForensicReport: 종합 분석 보고서
        """
        start_time = time.time()

        # 1. 모든 전문가에게 분석 요청
        agent_responses = self._collect_analyses(image, context)

        consensus_result, debate_result, consensus, debate_history = self._run_consensus_pipeline(agent_responses)
        final_verdict = consensus_result.final_verdict
        confidence = consensus_result.confidence

        # 5. 보고서 생성
        summary = self._generate_summary(final_verdict, confidence, agent_responses)
        detailed_reasoning = self._generate_detailed_reasoning(
            agent_responses, consensus, debate_history
        )

        total_time = time.time() - start_time

        return ForensicReport(
            final_verdict=final_verdict,
            confidence=confidence,
            summary=summary,
            detailed_reasoning=detailed_reasoning,
            agent_responses=agent_responses,
            consensus_info=consensus,
            debate_history=debate_history,
            consensus_result=consensus_result,
            debate_result=debate_result,
            total_processing_time=total_time
        )

    def _run_consensus_pipeline(
        self,
        responses: Dict[str, AgentResponse]
    ) -> tuple[ConsensusResult, Optional[DebateResult], Dict[str, Any], List[Dict[str, Any]]]:
        """
        MAIFS 런타임과 동일한 합의/토론 파이프라인.
        """
        consensus_result = self.consensus_engine.aggregate(
            responses,
            self.agent_trust,
            algorithm=None if self.consensus_algorithm == "auto" else self.consensus_algorithm,
        )

        debate_result: Optional[DebateResult] = None
        if self.enable_debate and self.debate_chamber.should_debate(responses):
            debate_result = self.debate_chamber.conduct_debate(
                responses,
                trust_scores=self.agent_trust,
            )
            if debate_result.consensus_result is not None:
                consensus_result = debate_result.consensus_result

        consensus_info = consensus_result.to_dict()
        debate_history: List[Dict[str, Any]] = []
        if debate_result:
            for round_info in debate_result.rounds:
                debate_history.append(
                    {
                        "round": round_info.round_number,
                        "messages": [msg.to_dict() for msg in round_info.messages],
                        "duration": round_info.duration,
                    }
                )

        return consensus_result, debate_result, consensus_info, debate_history

    def _collect_analyses(
        self,
        image: np.ndarray,
        context: Optional[Dict]
    ) -> Dict[str, AgentResponse]:
        """모든 전문가의 분석 수집"""
        responses = {}

        for name, agent in self.specialists.items():
            try:
                response = agent.analyze(image, context)
                responses[name] = response
                print(f"[Manager] {name} 분석 완료: {response.verdict.value}")
            except Exception as e:
                print(f"[Manager] {name} 분석 실패: {e}")

        return responses

    def _compute_consensus(
        self,
        responses: Dict[str, AgentResponse]
    ) -> Dict[str, Any]:
        """
        COBRA 기반 합의 계산 (호환용 래퍼)

        Returns:
            합의 정보 (가중 평균 신뢰도, 불일치 수준 등)
        """
        consensus = self.consensus_engine.aggregate(
            responses,
            self.agent_trust,
            algorithm=None if self.consensus_algorithm == "auto" else self.consensus_algorithm,
        )
        return {
            "weighted_confidence": consensus.confidence,
            "disagreement_level": consensus.disagreement_level,
            "verdict_distribution": consensus.verdict_scores,
            "dominant_verdict": consensus.final_verdict,
            "algorithm_used": consensus.algorithm_used,
        }

    def _conduct_debate(
        self,
        responses: Dict[str, AgentResponse]
    ) -> List[Dict]:
        """
        에이전트 간 토론 진행 (호환용 래퍼)

        Returns:
            토론 기록
        """
        result = self.debate_chamber.conduct_debate(
            responses,
            trust_scores=self.agent_trust,
        )
        history: List[Dict[str, Any]] = []
        for round_info in result.rounds:
            history.append(
                {
                    "round": round_info.round_number,
                    "messages": [msg.to_dict() for msg in round_info.messages],
                    "duration": round_info.duration,
                }
            )
        return history

    def _make_final_decision(
        self,
        responses: Dict[str, AgentResponse],
        consensus: Dict[str, Any]
    ) -> tuple:
        """최종 판정 결정 (호환용 래퍼)"""
        dominant = consensus.get("dominant_verdict", Verdict.UNCERTAIN)
        confidence = float(consensus.get("weighted_confidence", 0.0))
        if not isinstance(dominant, Verdict):
            try:
                dominant = Verdict(str(dominant).lower())
            except ValueError:
                dominant = Verdict.UNCERTAIN
        return dominant, max(0.0, min(1.0, confidence))

    def _generate_summary(
        self,
        verdict: Verdict,
        confidence: float,
        responses: Dict[str, AgentResponse]
    ) -> str:
        """요약 생성"""
        verdict_text = {
            Verdict.AUTHENTIC: "원본 이미지",
            Verdict.MANIPULATED: "조작된 이미지",
            Verdict.AI_GENERATED: "AI 생성 이미지",
            Verdict.UNCERTAIN: "판단 불가"
        }

        summary = (
            f"[최종 판정] {verdict_text.get(verdict, '알 수 없음')}\n"
            f"신뢰도: {confidence:.1%}\n\n"
            f"분석 참여 전문가 {len(responses)}명의 의견을 종합한 결과입니다."
        )

        return summary

    def _generate_detailed_reasoning(
        self,
        responses: Dict[str, AgentResponse],
        consensus: Dict[str, Any],
        debate_history: List[Dict]
    ) -> str:
        """상세 추론 생성"""
        parts = ["[상세 분석 보고서]", ""]

        # 각 전문가 의견
        parts.append("== 전문가별 분석 결과 ==")
        for name, response in responses.items():
            parts.append(f"\n{response.reasoning}")

        # 합의 정보
        parts.append("\n== 합의 분석 ==")
        parts.append(f"불일치 수준: {consensus.get('disagreement_level', 0):.1%}")
        parts.append(f"판정 분포: {consensus.get('verdict_distribution', {})}")

        # 토론 기록
        if debate_history:
            parts.append("\n== 토론 기록 ==")
            for round_info in debate_history:
                parts.append(f"라운드 {round_info['round']}:")
                messages = round_info.get("messages", [])
                if messages:
                    for message in messages:
                        agent_name = message.get("agent_name", "unknown")
                        content = message.get("content", "")
                        parts.append(f"  [{agent_name}] {content}")
                else:
                    # legacy format compatibility
                    for exchange in round_info.get("exchanges", []):
                        challenge = exchange.get("challenge", "")
                        if challenge:
                            parts.append(f"  {challenge}")

        return "\n".join(parts)

    def generate_reasoning(
        self,
        tool_results: List,
        context: Optional[Dict] = None
    ) -> str:
        """BaseAgent 추상 메서드 구현"""
        return "Manager Agent는 직접적인 Tool 분석을 수행하지 않습니다."

    def analyze_with_llm(
        self,
        image: np.ndarray,
        context: Optional[Dict] = None
    ) -> ForensicReport:
        """
        LLM을 활용한 고급 포렌식 분석

        Args:
            image: 분석할 RGB 이미지
            context: 추가 컨텍스트

        Returns:
            ForensicReport: LLM 기반 종합 분석 보고서
        """
        start_time = time.time()

        # 1. 모든 전문가에게 분석 요청
        agent_responses = self._collect_analyses(image, context)

        consensus_result, debate_result, consensus, debate_history = self._run_consensus_pipeline(agent_responses)

        # 4. LLM 기반 분석 (사용 가능한 경우)
        if self.use_llm and self.llm_client and self.llm_client.is_available:
            llm_response = self.llm_client.analyze_forensics(
                agent_responses=agent_responses,
                consensus_info=consensus,
                debate_history=debate_history
            )

            # LLM 응답 파싱
            try:
                llm_result = json.loads(llm_response.content)
                verdict_raw = str(llm_result.get("verdict", "uncertain")).strip().lower()
                try:
                    final_verdict = Verdict(verdict_raw)
                except ValueError:
                    final_verdict = consensus_result.final_verdict
                confidence = float(llm_result.get("confidence", consensus_result.confidence))
                confidence = max(0.0, min(1.0, confidence))
                summary = llm_result.get("summary", "")
                detailed_reasoning = llm_result.get("reasoning", "")
            except (json.JSONDecodeError, ValueError):
                # JSON 파싱 실패 시 기본 분석 사용
                final_verdict = consensus_result.final_verdict
                confidence = consensus_result.confidence
                summary = self._generate_summary(final_verdict, confidence, agent_responses)
                detailed_reasoning = llm_response.content
        else:
            # 규칙 기반 분석
            final_verdict = consensus_result.final_verdict
            confidence = consensus_result.confidence
            summary = self._generate_summary(final_verdict, confidence, agent_responses)
            detailed_reasoning = self._generate_detailed_reasoning(
                agent_responses, consensus, debate_history
            )

        total_time = time.time() - start_time

        return ForensicReport(
            final_verdict=final_verdict,
            confidence=confidence,
            summary=summary,
            detailed_reasoning=detailed_reasoning,
            agent_responses=agent_responses,
            consensus_info=consensus,
            debate_history=debate_history,
            consensus_result=consensus_result,
            debate_result=debate_result,
            total_processing_time=total_time
        )

    def generate_human_report(
        self,
        report: ForensicReport,
        language: str = "ko"
    ) -> str:
        """
        사람이 읽기 쉬운 보고서 생성

        Args:
            report: ForensicReport 객체
            language: 출력 언어 (ko/en)

        Returns:
            str: 포맷팅된 보고서
        """
        if self.use_llm and self.llm_client and self.llm_client.is_available:
            return self.llm_client.generate_report(
                verdict=report.final_verdict.value,
                confidence=report.confidence,
                agent_responses=report.agent_responses,
                language=language
            )
        else:
            return self._generate_fallback_report(report, language)

    def _generate_fallback_report(
        self,
        report: ForensicReport,
        language: str = "ko"
    ) -> str:
        """규칙 기반 보고서 생성"""
        if language == "ko":
            verdict_text = {
                Verdict.AUTHENTIC: "원본 이미지",
                Verdict.MANIPULATED: "조작된 이미지",
                Verdict.AI_GENERATED: "AI 생성 이미지",
                Verdict.UNCERTAIN: "판단 불가"
            }

            lines = [
                "=" * 60,
                "          MAIFS 이미지 포렌식 분석 보고서",
                "=" * 60,
                "",
                f"▶ 최종 판정: {verdict_text.get(report.final_verdict, '알 수 없음')}",
                f"▶ 신뢰도: {report.confidence:.1%}",
                f"▶ 처리 시간: {report.total_processing_time:.2f}초",
                "",
                "-" * 60,
                "전문가 분석 요약",
                "-" * 60,
            ]

            for name, response in report.agent_responses.items():
                agent_name = {
                    "frequency": "주파수 분석",
                    "noise": "노이즈 분석",
                    "fatformer": "AI 생성 탐지 (FatFormer)",
                    "spatial": "공간 분석"
                }.get(name, name)

                lines.append(f"  [{agent_name}]")
                lines.append(f"    판정: {response.verdict.value}")
                lines.append(f"    신뢰도: {response.confidence:.1%}")
                lines.append("")

            if report.debate_history:
                lines.append("-" * 60)
                lines.append(f"토론 진행: {len(report.debate_history)} 라운드")
                lines.append("-" * 60)

            lines.extend([
                "",
                "=" * 60,
                "        분석 완료 - MAIFS Multi-Agent System",
                "=" * 60
            ])

            return "\n".join(lines)
        else:
            # English version
            lines = [
                "=" * 60,
                "          MAIFS Image Forensic Analysis Report",
                "=" * 60,
                "",
                f"▶ Final Verdict: {report.final_verdict.value}",
                f"▶ Confidence: {report.confidence:.1%}",
                f"▶ Processing Time: {report.total_processing_time:.2f}s",
                "",
                "-" * 60,
                "Expert Analysis Summary",
                "-" * 60,
            ]

            for name, response in report.agent_responses.items():
                lines.append(f"  [{name.upper()}]")
                lines.append(f"    Verdict: {response.verdict.value}")
                lines.append(f"    Confidence: {response.confidence:.1%}")
                lines.append("")

            lines.extend([
                "",
                "=" * 60,
                "        Analysis Complete - MAIFS Multi-Agent System",
                "=" * 60
            ])

            return "\n".join(lines)
