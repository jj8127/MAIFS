"""
Debate Protocol
Multi-Agent 토론 프로토콜 및 종료 조건 관리
"""
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

from ..agents.base_agent import AgentResponse
from ..tools.base_tool import Verdict
from ..llm.subagent_llm import DebateResponse


class DebateTerminationReason(Enum):
    """토론 종료 이유"""
    CONSENSUS_REACHED = "consensus_reached"  # 합의 도달
    MAX_ROUNDS_REACHED = "max_rounds"  # 최대 라운드 도달
    STALEMATE = "stalemate"  # 교착 상태 (같은 주장 반복)
    NO_PROGRESS = "no_progress"  # 진전 없음 (판정 변화 없음)
    HIGH_CONFIDENCE_DEADLOCK = "high_confidence_deadlock"  # 양측 모두 높은 신뢰도


@dataclass
class DebateTurn:
    """토론의 한 턴"""
    round_number: int
    challenger: str  # Agent 이름
    challenged: str
    challenge: str  # 도전 내용
    response: str  # 응답 내용
    verdict_before: Verdict  # 응답 전 판정
    verdict_after: Verdict  # 응답 후 판정
    confidence_before: float
    confidence_after: float
    verdict_changed: bool = False


@dataclass
class DebateResult:
    """토론 결과"""
    total_rounds: int
    turns: List[DebateTurn] = field(default_factory=list)
    final_verdicts: Dict[str, Verdict] = field(default_factory=dict)
    termination_reason: DebateTerminationReason = DebateTerminationReason.CONSENSUS_REACHED
    consensus_reached: bool = False
    disagreement_level_before: float = 0.0
    disagreement_level_after: float = 0.0

    def to_dict(self) -> Dict:
        """딕셔너리로 변환"""
        return {
            "total_rounds": self.total_rounds,
            "termination_reason": self.termination_reason.value,
            "consensus_reached": self.consensus_reached,
            "disagreement_before": self.disagreement_level_before,
            "disagreement_after": self.disagreement_level_after,
            "turns": [
                {
                    "round": turn.round_number,
                    "challenger": turn.challenger,
                    "challenged": turn.challenged,
                    "verdict_changed": turn.verdict_changed,
                    "verdict_before": turn.verdict_before.value,
                    "verdict_after": turn.verdict_after.value
                }
                for turn in self.turns
            ],
            "final_verdicts": {k: v.value for k, v in self.final_verdicts.items()}
        }


class DebateProtocol:
    """토론 프로토콜 관리자"""

    def __init__(
        self,
        max_rounds: int = 3,
        consensus_threshold: float = 0.3,
        stalemate_threshold: int = 2,
        high_confidence_threshold: float = 0.85
    ):
        """
        Args:
            max_rounds: 최대 토론 라운드
            consensus_threshold: 합의 판정 임계값 (disagreement < threshold)
            stalemate_threshold: 교착 상태 판정 (N 라운드 동안 변화 없음)
            high_confidence_threshold: 높은 신뢰도 기준
        """
        self.max_rounds = max_rounds
        self.consensus_threshold = consensus_threshold
        self.stalemate_threshold = stalemate_threshold
        self.high_confidence_threshold = high_confidence_threshold

    def should_debate(self, responses: Dict[str, AgentResponse]) -> bool:
        """토론 필요 여부 판단"""
        disagreement = self._compute_disagreement(responses)
        return disagreement > self.consensus_threshold

    def _compute_disagreement(self, responses: Dict[str, AgentResponse]) -> float:
        """
        불일치 수준 계산

        Returns:
            0.0 ~ 1.0 (0: 완전 합의, 1: 완전 불일치)
        """
        verdicts = [r.verdict for r in responses.values()]
        if not verdicts:
            return 0.0

        unique_verdicts = len(set(verdicts))
        total_verdicts = len(verdicts)

        # 정규화: 모두 다르면 1.0, 모두 같으면 0.0
        return (unique_verdicts - 1) / max(total_verdicts - 1, 1)

    def conduct_debate(
        self,
        agents: Dict[str, 'BaseAgent'],
        responses: Dict[str, AgentResponse]
    ) -> DebateResult:
        """
        토론 진행

        Args:
            agents: Agent 인스턴스들
            responses: 초기 분석 결과

        Returns:
            DebateResult: 토론 결과
        """
        turns = []
        current_round = 1

        disagreement_before = self._compute_disagreement(responses)

        # 판정 변경 추적 (교착 상태 감지용)
        rounds_without_change = 0

        # 토론 대상 쌍 찾기
        debate_pairs = self._find_debate_pairs(responses)

        if not debate_pairs:
            # 토론 대상 없음 (이미 합의됨)
            return DebateResult(
                total_rounds=0,
                turns=[],
                final_verdicts={name: r.verdict for name, r in responses.items()},
                termination_reason=DebateTerminationReason.CONSENSUS_REACHED,
                consensus_reached=True,
                disagreement_level_before=disagreement_before,
                disagreement_level_after=disagreement_before
            )

        while current_round <= self.max_rounds:
            round_turns = []
            any_change_this_round = False

            for agent_a_name, agent_b_name in debate_pairs:
                agent_a = agents[agent_a_name]
                agent_b = agents[agent_b_name]

                # A가 B에게 도전
                turn = self._execute_challenge(
                    challenger=agent_a,
                    challenged=agent_b,
                    challenger_response=responses[agent_a_name],
                    challenged_response=responses[agent_b_name],
                    round_number=current_round
                )

                round_turns.append(turn)

                # 판정 변경 확인
                if turn.verdict_changed:
                    any_change_this_round = True
                    # responses 업데이트
                    responses[agent_b_name].verdict = turn.verdict_after
                    responses[agent_b_name].confidence = turn.confidence_after

            turns.extend(round_turns)

            # 종료 조건 확인
            termination_reason, should_terminate = self._check_termination(
                responses=responses,
                current_round=current_round,
                any_change_this_round=any_change_this_round,
                rounds_without_change=rounds_without_change
            )

            if should_terminate:
                return DebateResult(
                    total_rounds=current_round,
                    turns=turns,
                    final_verdicts={name: r.verdict for name, r in responses.items()},
                    termination_reason=termination_reason,
                    consensus_reached=(termination_reason == DebateTerminationReason.CONSENSUS_REACHED),
                    disagreement_level_before=disagreement_before,
                    disagreement_level_after=self._compute_disagreement(responses)
                )

            # 변화 추적
            if not any_change_this_round:
                rounds_without_change += 1
            else:
                rounds_without_change = 0

            current_round += 1

        # 최대 라운드 도달
        return DebateResult(
            total_rounds=self.max_rounds,
            turns=turns,
            final_verdicts={name: r.verdict for name, r in responses.items()},
            termination_reason=DebateTerminationReason.MAX_ROUNDS_REACHED,
            consensus_reached=self._check_consensus(responses),
            disagreement_level_before=disagreement_before,
            disagreement_level_after=self._compute_disagreement(responses)
        )

    def _check_termination(
        self,
        responses: Dict[str, AgentResponse],
        current_round: int,
        any_change_this_round: bool,
        rounds_without_change: int
    ) -> Tuple[DebateTerminationReason, bool]:
        """
        종료 조건 확인

        Returns:
            (종료 이유, 종료 여부)
        """
        # 1. 합의 도달
        if self._check_consensus(responses):
            return (DebateTerminationReason.CONSENSUS_REACHED, True)

        # 2. 교착 상태 (N 라운드 동안 변화 없음)
        if rounds_without_change >= self.stalemate_threshold:
            return (DebateTerminationReason.STALEMATE, True)

        # 3. 양측 모두 높은 신뢰도 (합의 불가능)
        if self._check_high_confidence_deadlock(responses):
            return (DebateTerminationReason.HIGH_CONFIDENCE_DEADLOCK, True)

        # 4. 최대 라운드
        if current_round >= self.max_rounds:
            return (DebateTerminationReason.MAX_ROUNDS_REACHED, True)

        return (DebateTerminationReason.CONSENSUS_REACHED, False)

    def _check_consensus(self, responses: Dict[str, AgentResponse]) -> bool:
        """합의 도달 여부 확인"""
        disagreement = self._compute_disagreement(responses)
        return disagreement < self.consensus_threshold

    def _check_high_confidence_deadlock(
        self,
        responses: Dict[str, AgentResponse]
    ) -> bool:
        """
        양측 모두 높은 신뢰도로 다른 판정 → 합의 불가능

        예:
          Frequency: AI_GENERATED (0.92)
          Noise: AUTHENTIC (0.88)
          → 양측 모두 확신 → 더 토론해도 의미 없음
        """
        verdicts = {}
        for name, response in responses.items():
            verdict = response.verdict
            confidence = response.confidence

            if verdict not in verdicts:
                verdicts[verdict] = []
            verdicts[verdict].append(confidence)

        # 서로 다른 판정이 2개 이상 있고, 모두 높은 신뢰도
        if len(verdicts) >= 2:
            high_confidence_groups = 0
            for verdict, confidences in verdicts.items():
                avg_confidence = sum(confidences) / len(confidences)
                if avg_confidence >= self.high_confidence_threshold:
                    high_confidence_groups += 1

            # 2개 이상의 그룹이 모두 높은 신뢰도
            if high_confidence_groups >= 2:
                return True

        return False

    def _find_debate_pairs(
        self,
        responses: Dict[str, AgentResponse]
    ) -> List[Tuple[str, str]]:
        """
        토론할 Agent 쌍 찾기

        전략: 신뢰도 높은 쪽이 낮은 쪽에게 도전
        """
        pairs = []
        agent_names = list(responses.keys())

        for i, name_a in enumerate(agent_names):
            for name_b in agent_names[i+1:]:
                # 판정이 다르면 토론 대상
                if responses[name_a].verdict != responses[name_b].verdict:
                    # 신뢰도 높은 쪽이 도전자
                    if responses[name_a].confidence >= responses[name_b].confidence:
                        pairs.append((name_a, name_b))
                    else:
                        pairs.append((name_b, name_a))

        return pairs

    def _execute_challenge(
        self,
        challenger: 'BaseAgent',
        challenged: 'BaseAgent',
        challenger_response: AgentResponse,
        challenged_response: AgentResponse,
        round_number: int
    ) -> DebateTurn:
        """
        한 턴의 도전-응답 실행

        Args:
            challenger: 도전하는 Agent
            challenged: 도전받는 Agent
            challenger_response: 도전자의 분석 결과
            challenged_response: 피도전자의 분석 결과
            round_number: 현재 라운드 번호

        Returns:
            DebateTurn: 토론 턴 결과
        """
        # 1. 도전자의 주장 생성
        challenge = self._generate_challenge(
            challenger=challenger,
            my_response=challenger_response,
            opponent_response=challenged_response
        )

        # 2. 피도전자의 응답
        response_result = self._generate_response(
            challenged=challenged,
            challenge=challenge,
            challenger_name=challenger.name,
            my_response=challenged_response,
            opponent_response=challenger_response
        )

        # 3. 판정 변경 확인
        verdict_before = challenged_response.verdict
        confidence_before = challenged_response.confidence

        verdict_after = response_result.get("verdict_after", verdict_before)
        confidence_after = response_result.get("confidence_after", confidence_before)
        verdict_changed = response_result.get("verdict_changed", verdict_after != verdict_before)

        return DebateTurn(
            round_number=round_number,
            challenger=challenger.name,
            challenged=challenged.name,
            challenge=challenge,
            response=response_result['response'],
            verdict_before=verdict_before,
            verdict_after=verdict_after,
            confidence_before=confidence_before,
            confidence_after=confidence_after,
            verdict_changed=verdict_changed
        )

    def _generate_challenge(
        self,
        challenger: 'BaseAgent',
        my_response: AgentResponse,
        opponent_response: AgentResponse
    ) -> str:
        """도전 생성 (Agent의 메서드 호출)"""
        if hasattr(challenger, "generate_challenge"):
            try:
                return challenger.generate_challenge(opponent_response, my_response)
            except TypeError:
                try:
                    return challenger.generate_challenge(opponent_response)
                except TypeError:
                    pass

        return (
            f"저는 {my_response.verdict.value}로 판단했습니다 "
            f"(신뢰도 {my_response.confidence:.1%}). "
            f"귀하는 {opponent_response.verdict.value}로 판단한 근거가 무엇입니까?"
        )

    def _coerce_verdict(self, verdict_value: Any, fallback: Verdict) -> Verdict:
        """문자열/Enum 판정을 Verdict로 정규화"""
        if isinstance(verdict_value, Verdict):
            return verdict_value

        if isinstance(verdict_value, str):
            value = verdict_value.strip().lower()
            for verdict in Verdict:
                if value == verdict.value:
                    return verdict
            for verdict in Verdict:
                if value == verdict.name.lower():
                    return verdict

        return fallback

    def _normalize_response(
        self,
        raw_response: Any,
        challenged_response: AgentResponse
    ) -> Dict[str, Any]:
        """응답 타입을 표준 딕셔너리로 정규화"""
        verdict_after = challenged_response.verdict
        confidence_after = challenged_response.confidence
        verdict_changed = False

        if isinstance(raw_response, DebateResponse):
            response_text = raw_response.content
            verdict_changed = raw_response.verdict_changed
            if raw_response.new_verdict:
                verdict_after = self._coerce_verdict(
                    raw_response.new_verdict, verdict_after
                )
            if raw_response.new_confidence is not None:
                confidence_after = raw_response.new_confidence
        elif isinstance(raw_response, dict):
            response_text = raw_response.get("response") or raw_response.get("content") or ""
            verdict_after = self._coerce_verdict(
                raw_response.get("verdict_after", verdict_after), verdict_after
            )
            confidence_after = raw_response.get("confidence_after", confidence_after)
            verdict_changed = raw_response.get(
                "verdict_changed", verdict_after != challenged_response.verdict
            )
        else:
            response_text = str(raw_response)

        if verdict_after != challenged_response.verdict:
            verdict_changed = True

        confidence_after = max(0.0, min(1.0, float(confidence_after)))

        return {
            "response": response_text,
            "verdict_after": verdict_after,
            "confidence_after": confidence_after,
            "verdict_changed": verdict_changed
        }

    def _generate_response(
        self,
        challenged: 'BaseAgent',
        challenge: str,
        challenger_name: str,
        my_response: AgentResponse,
        opponent_response: AgentResponse
    ) -> Dict[str, Any]:
        """응답 생성 (Agent의 메서드 호출)"""
        raw_response = None

        if hasattr(challenged, "respond_to_challenge"):
            try:
                raw_response = challenged.respond_to_challenge(
                    challenger_name=challenger_name,
                    challenge=challenge,
                    my_response=my_response
                )
            except TypeError:
                try:
                    raw_response = challenged.respond_to_challenge(
                        challenge=challenge,
                        challenger_name=challenger_name,
                        my_current_verdict=my_response.verdict,
                        my_evidence=my_response.evidence,
                        my_confidence=my_response.confidence,
                        opponent_evidence=opponent_response.evidence
                    )
                except TypeError:
                    raw_response = None

        if raw_response is None:
            raw_response = "제 분석 도구의 결과를 바탕으로 판단했습니다."

        return self._normalize_response(raw_response, my_response)
