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
from ..tools.base_tool import Verdict

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
        use_llm: bool = True
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

        # 에이전트별 신뢰도 (COBRA용)
        self.agent_trust: Dict[str, float] = {
            "frequency": 0.85,
            "noise": 0.80,
            "fatformer": 0.85,
            "spatial": 0.85,
        }

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

        # 2. 합의 도출
        consensus = self._compute_consensus(agent_responses)

        # 3. 불일치 확인 및 토론
        debate_history = []
        if consensus["disagreement_level"] > 0.3:
            debate_history = self._conduct_debate(agent_responses)
            # 토론 후 합의 재계산
            consensus = self._compute_consensus(agent_responses)

        # 4. 최종 판정
        final_verdict, confidence = self._make_final_decision(
            agent_responses, consensus
        )

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
            total_processing_time=total_time
        )

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
        COBRA 기반 합의 계산

        Returns:
            합의 정보 (가중 평균 신뢰도, 불일치 수준 등)
        """
        if not responses:
            return {
                "weighted_confidence": 0.0,
                "disagreement_level": 1.0,
                "verdict_distribution": {}
            }

        # 판정별 분포
        verdict_distribution: Dict[Verdict, List[float]] = {}
        total_weight = 0.0

        for name, response in responses.items():
            trust = self.agent_trust.get(name, 0.5)
            weighted_conf = response.confidence * trust

            if response.verdict not in verdict_distribution:
                verdict_distribution[response.verdict] = []
            verdict_distribution[response.verdict].append(weighted_conf)
            total_weight += trust

        # 불일치 수준 계산
        # 서로 다른 판정 개수가 많을수록 불일치
        num_verdicts = len(verdict_distribution)
        disagreement = (num_verdicts - 1) / 3.0  # 최대 4개 판정

        # 가중 평균 신뢰도
        weighted_confidence = sum(
            sum(confs) for confs in verdict_distribution.values()
        ) / total_weight if total_weight > 0 else 0.0

        return {
            "weighted_confidence": weighted_confidence,
            "disagreement_level": disagreement,
            "verdict_distribution": {
                k.value: sum(v) / len(v) for k, v in verdict_distribution.items()
            },
            "dominant_verdict": max(
                verdict_distribution.keys(),
                key=lambda v: sum(verdict_distribution[v])
            ) if verdict_distribution else Verdict.UNCERTAIN
        }

    def _conduct_debate(
        self,
        responses: Dict[str, AgentResponse]
    ) -> List[Dict]:
        """
        에이전트 간 토론 진행

        Returns:
            토론 기록
        """
        debate_history = []

        # 불일치하는 에이전트 쌍 찾기
        verdicts = {name: r.verdict for name, r in responses.items()}
        unique_verdicts = set(verdicts.values())

        if len(unique_verdicts) <= 1:
            return debate_history  # 토론 불필요

        # 라운드 1: 반대 의견 제시
        round_1 = {"round": 1, "exchanges": []}

        for name_a, verdict_a in verdicts.items():
            for name_b, verdict_b in verdicts.items():
                if name_a < name_b and verdict_a != verdict_b:
                    # A가 B에게 반론
                    challenge = (
                        f"[{name_a}→{name_b}] "
                        f"나는 {verdict_a.value}로 판단했는데, "
                        f"당신은 왜 {verdict_b.value}로 판단했나요? "
                        f"내 근거: {responses[name_a].arguments[:1]}"
                    )

                    # B의 응답
                    rebuttal = responses[name_b].tool_results[0].explanation if responses[name_b].tool_results else "증거 없음"

                    round_1["exchanges"].append({
                        "challenger": name_a,
                        "challenged": name_b,
                        "challenge": challenge,
                        "rebuttal": rebuttal
                    })

        debate_history.append(round_1)

        return debate_history

    def _make_final_decision(
        self,
        responses: Dict[str, AgentResponse],
        consensus: Dict[str, Any]
    ) -> tuple:
        """최종 판정 결정"""
        if not responses:
            return Verdict.UNCERTAIN, 0.0

        # DRWA (Dynamic Reliability Weighted Aggregation) 적용
        verdict_scores: Dict[Verdict, float] = {}

        for name, response in responses.items():
            trust = self.agent_trust.get(name, 0.5)

            # 동적 가중치 = 신뢰도 * 분석 신뢰도
            weight = trust * response.confidence

            if response.verdict not in verdict_scores:
                verdict_scores[response.verdict] = 0.0
            verdict_scores[response.verdict] += weight

        # 가장 높은 점수의 판정 선택
        final_verdict = max(verdict_scores.keys(), key=lambda v: verdict_scores[v])
        total_score = sum(verdict_scores.values())
        confidence = verdict_scores[final_verdict] / total_score if total_score > 0 else 0.0

        return final_verdict, confidence

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
                for exchange in round_info.get("exchanges", []):
                    parts.append(f"  {exchange['challenge']}")

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

        # 2. 합의 도출
        consensus = self._compute_consensus(agent_responses)

        # 3. 불일치 확인 및 토론
        debate_history = []
        if consensus["disagreement_level"] > 0.3:
            debate_history = self._conduct_debate(agent_responses)
            consensus = self._compute_consensus(agent_responses)

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
                final_verdict = Verdict(llm_result.get("verdict", "UNCERTAIN"))
                confidence = llm_result.get("confidence", 0.5)
                summary = llm_result.get("summary", "")
                detailed_reasoning = llm_result.get("reasoning", "")
            except (json.JSONDecodeError, ValueError):
                # JSON 파싱 실패 시 기본 분석 사용
                final_verdict, confidence = self._make_final_decision(agent_responses, consensus)
                summary = self._generate_summary(final_verdict, confidence, agent_responses)
                detailed_reasoning = llm_response.content
        else:
            # 규칙 기반 분석
            final_verdict, confidence = self._make_final_decision(agent_responses, consensus)
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
