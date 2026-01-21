"""
Debate Chamber - 다중 에이전트 토론 관리

에이전트 간 의견 충돌 시 토론을 통해 합의 도출
"""
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import time

from ..agents.base_agent import AgentResponse, BaseAgent
from ..tools.base_tool import Verdict
from ..consensus.cobra import COBRAConsensus, ConsensusResult
from .protocols import (
    DebateProtocol,
    DebateState,
    DebateMessage,
    AsynchronousDebate
)


@dataclass
class DebateRound:
    """토론 라운드 정보"""
    round_number: int
    messages: List[DebateMessage]
    verdict_snapshot: Dict[str, str]
    confidence_snapshot: Dict[str, float]
    duration: float = 0.0


@dataclass
class DebateResult:
    """토론 최종 결과"""
    initial_verdicts: Dict[str, Verdict]
    final_verdicts: Dict[str, Verdict]
    rounds: List[DebateRound]
    total_rounds: int
    convergence_achieved: bool
    convergence_reason: str
    consensus_result: Optional[ConsensusResult] = None
    total_duration: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "initial_verdicts": {k: v.value for k, v in self.initial_verdicts.items()},
            "final_verdicts": {k: v.value for k, v in self.final_verdicts.items()},
            "total_rounds": self.total_rounds,
            "convergence_achieved": self.convergence_achieved,
            "convergence_reason": self.convergence_reason,
            "consensus": self.consensus_result.to_dict() if self.consensus_result else None,
            "duration": self.total_duration
        }

    def get_summary(self) -> str:
        """토론 요약"""
        lines = [
            "=== 토론 결과 요약 ===",
            f"총 {self.total_rounds} 라운드 진행",
            f"수렴 여부: {'예' if self.convergence_achieved else '아니오'}",
            f"수렴 이유: {self.convergence_reason}",
            "",
            "초기 판정:",
        ]

        for name, verdict in self.initial_verdicts.items():
            lines.append(f"  - {name}: {verdict.value}")

        lines.append("")
        lines.append("최종 판정:")
        for name, verdict in self.final_verdicts.items():
            changed = verdict != self.initial_verdicts.get(name)
            marker = " (변경됨)" if changed else ""
            lines.append(f"  - {name}: {verdict.value}{marker}")

        if self.consensus_result:
            lines.extend([
                "",
                f"합의 결과: {self.consensus_result.final_verdict.value}",
                f"합의 신뢰도: {self.consensus_result.confidence:.1%}"
            ])

        return "\n".join(lines)


class DebateChamber:
    """
    Debate Chamber - 토론 관리자

    역할:
    1. 토론 필요성 판단
    2. 적절한 토론 프로토콜 선택
    3. 토론 라운드 진행
    4. 수렴 확인 및 결과 집계
    """

    def __init__(
        self,
        protocol: Optional[DebateProtocol] = None,
        consensus_engine: Optional[COBRAConsensus] = None,
        disagreement_threshold: float = 0.3
    ):
        """
        Args:
            protocol: 토론 프로토콜 (기본: 비동기식)
            consensus_engine: COBRA 합의 엔진
            disagreement_threshold: 토론 개시 불일치 임계값
        """
        self.protocol = protocol or AsynchronousDebate(max_rounds=3)
        self.consensus_engine = consensus_engine or COBRAConsensus()
        self.disagreement_threshold = disagreement_threshold

    def should_debate(self, responses: Dict[str, AgentResponse]) -> bool:
        """
        토론 필요성 판단

        다음 조건에서 토론 개시:
        1. 판정이 2개 이상으로 나뉜 경우
        2. 신뢰도 차이가 큰 경우
        3. 증거가 상충하는 경우
        """
        if len(responses) < 2:
            return False

        # 판정 다양성 확인
        verdicts = set(r.verdict for r in responses.values())
        if len(verdicts) >= 2:
            return True

        # 신뢰도 분산 확인
        confidences = [r.confidence for r in responses.values()]
        if max(confidences) - min(confidences) > self.disagreement_threshold:
            return True

        return False

    def conduct_debate(
        self,
        responses: Dict[str, AgentResponse],
        agents: Optional[Dict[str, BaseAgent]] = None,
        trust_scores: Optional[Dict[str, float]] = None
    ) -> DebateResult:
        """
        토론 진행

        Args:
            responses: 에이전트별 초기 응답
            agents: 에이전트 인스턴스 (선택적, 동적 응답 생성용)
            trust_scores: 에이전트별 신뢰도

        Returns:
            DebateResult: 토론 결과
        """
        start_time = time.time()

        # 초기 상태
        initial_verdicts = {name: r.verdict for name, r in responses.items()}
        state = DebateState()
        rounds: List[DebateRound] = []

        # 기본 신뢰도
        if trust_scores is None:
            trust_scores = {name: 0.8 for name in responses}

        # 현재 응답 (토론 중 업데이트될 수 있음)
        current_responses = dict(responses)

        # 토론 라운드 진행
        while not state.is_converged and state.current_round < self.protocol.max_rounds:
            round_start = time.time()

            # 라운드 진행
            messages, state = self.protocol.conduct_round(current_responses, state)

            # 라운드 기록
            debate_round = DebateRound(
                round_number=state.current_round,
                messages=messages,
                verdict_snapshot={
                    name: r.verdict.value for name, r in current_responses.items()
                },
                confidence_snapshot={
                    name: r.confidence for name, r in current_responses.items()
                },
                duration=time.time() - round_start
            )
            rounds.append(debate_round)

            # 토론을 통한 응답 업데이트 (신뢰도 조정)
            current_responses = self._update_responses_from_debate(
                current_responses, messages, trust_scores
            )

            # 수렴 확인
            self.protocol.check_convergence(current_responses, state)

        # 최종 합의 도출
        consensus_result = self.consensus_engine.aggregate(
            current_responses, trust_scores
        )

        total_duration = time.time() - start_time

        return DebateResult(
            initial_verdicts=initial_verdicts,
            final_verdicts={name: r.verdict for name, r in current_responses.items()},
            rounds=rounds,
            total_rounds=state.current_round,
            convergence_achieved=state.is_converged,
            convergence_reason=state.convergence_reason,
            consensus_result=consensus_result,
            total_duration=total_duration
        )

    def _update_responses_from_debate(
        self,
        responses: Dict[str, AgentResponse],
        messages: List[DebateMessage],
        trust_scores: Dict[str, float]
    ) -> Dict[str, AgentResponse]:
        """
        토론 메시지를 바탕으로 응답 업데이트

        다른 에이전트의 강력한 증거에 의해 신뢰도가 조정될 수 있음
        """
        updated = {}

        for name, response in responses.items():
            # 이 에이전트에게 보낸 반론 찾기
            challenges = [
                m for m in messages
                if m.target_agent == name or (
                    m.agent_name != name and
                    responses.get(m.agent_name, response).verdict != response.verdict
                )
            ]

            # 반론의 강도에 따라 신뢰도 조정
            confidence_delta = 0.0
            for challenge in challenges:
                challenger_trust = trust_scores.get(challenge.agent_name, 0.5)
                # 신뢰도 높은 에이전트의 반론은 더 큰 영향
                if len(challenge.evidence) > len(response.arguments):
                    confidence_delta -= 0.05 * challenger_trust

            # 새 신뢰도 적용
            new_confidence = max(0.1, min(1.0, response.confidence + confidence_delta))

            # 응답 복사 및 업데이트
            updated[name] = AgentResponse(
                agent_name=response.agent_name,
                role=response.role,
                verdict=response.verdict,
                confidence=new_confidence,
                reasoning=response.reasoning,
                evidence=response.evidence,
                tool_results=response.tool_results,
                arguments=response.arguments,
                processing_time=response.processing_time
            )

        return updated

    def generate_debate_transcript(self, result: DebateResult) -> str:
        """토론 기록 생성"""
        lines = [
            "=" * 60,
            "MAIFS 다중 에이전트 토론 기록",
            "=" * 60,
            ""
        ]

        for round_info in result.rounds:
            lines.append(f"--- 라운드 {round_info.round_number} ---")
            lines.append("")

            for msg in round_info.messages:
                lines.append(f"[{msg.agent_name}] ({msg.role.value})")
                lines.append(f"  {msg.content}")
                if msg.evidence:
                    lines.append(f"  증거: {msg.evidence[0][:50]}...")
                lines.append("")

            lines.append(f"라운드 소요 시간: {round_info.duration:.2f}초")
            lines.append("")

        lines.extend([
            "=" * 60,
            result.get_summary(),
            "=" * 60
        ])

        return "\n".join(lines)
