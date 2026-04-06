"""
토론 프로토콜 정의

MAD-Sherlock 논문 기반 다중 에이전트 토론 프로토콜:
1. Synchronous: 모든 에이전트가 동시에 발언
2. Asynchronous: 순차적으로 발언, 이전 발언 참조 가능
3. Structured: 역할 기반 구조화된 토론
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum

from ..agents.base_agent import AgentResponse
from ..tools.base_tool import Verdict


class DebateRole(Enum):
    """토론 역할"""
    PROPONENT = "proponent"      # 지지자 (현재 판정 옹호)
    OPPONENT = "opponent"        # 반대자 (반론 제기)
    MODERATOR = "moderator"      # 중재자
    JUDGE = "judge"              # 심판


@dataclass
class DebateMessage:
    """토론 메시지"""
    agent_name: str
    role: DebateRole
    content: str
    evidence: List[str] = field(default_factory=list)
    target_agent: Optional[str] = None  # 특정 에이전트에게 보내는 메시지
    round_number: int = 0
    confidence_change: float = 0.0  # 이 발언으로 인한 신뢰도 변화

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent": self.agent_name,
            "role": self.role.value,
            "content": self.content,
            "evidence": self.evidence,
            "target": self.target_agent,
            "round": self.round_number
        }


@dataclass
class DebateState:
    """토론 상태"""
    current_round: int = 0
    messages: List[DebateMessage] = field(default_factory=list)
    verdict_history: List[Dict[str, Verdict]] = field(default_factory=list)
    confidence_history: List[Dict[str, float]] = field(default_factory=list)
    is_converged: bool = False
    convergence_reason: str = ""


class DebateProtocol(ABC):
    """토론 프로토콜 기본 클래스"""

    def __init__(self, max_rounds: int = 3, convergence_threshold: float = 0.1):
        """
        Args:
            max_rounds: 최대 토론 라운드 수
            convergence_threshold: 수렴 판단 임계값
        """
        self.max_rounds = max_rounds
        self.convergence_threshold = convergence_threshold

    @abstractmethod
    def conduct_round(
        self,
        responses: Dict[str, AgentResponse],
        state: DebateState
    ) -> Tuple[List[DebateMessage], DebateState]:
        """
        토론 라운드 진행

        Args:
            responses: 현재 에이전트 응답들
            state: 현재 토론 상태

        Returns:
            (생성된 메시지들, 업데이트된 상태)
        """
        pass

    def check_convergence(
        self,
        responses: Dict[str, AgentResponse],
        state: DebateState
    ) -> bool:
        """수렴 여부 확인"""
        if state.current_round >= self.max_rounds:
            state.is_converged = True
            state.convergence_reason = "최대 라운드 도달"
            return True

        # 모든 에이전트가 같은 판정인지 확인
        verdicts = set(r.verdict for r in responses.values())
        if len(verdicts) == 1:
            state.is_converged = True
            state.convergence_reason = "만장일치"
            return True

        # 신뢰도 변화가 임계값 이하인지 확인
        if len(state.confidence_history) >= 2:
            prev = state.confidence_history[-2]
            curr = state.confidence_history[-1]

            max_change = max(
                abs(curr.get(name, 0) - prev.get(name, 0))
                for name in set(prev.keys()) | set(curr.keys())
            )

            if max_change < self.convergence_threshold:
                state.is_converged = True
                state.convergence_reason = "신뢰도 수렴"
                return True

        return False


class SynchronousDebate(DebateProtocol):
    """
    동기식 토론 프로토콜

    모든 에이전트가 동시에 자신의 의견을 제시
    각 라운드에서 다른 에이전트들의 이전 라운드 발언만 참조 가능
    """

    def conduct_round(
        self,
        responses: Dict[str, AgentResponse],
        state: DebateState
    ) -> Tuple[List[DebateMessage], DebateState]:
        """동기식 토론 라운드"""
        messages = []
        state.current_round += 1

        # 각 에이전트의 현재 판정 수집
        current_verdicts = {name: r.verdict for name, r in responses.items()}
        majority_verdict = self._get_majority_verdict(responses)

        for name, response in responses.items():
            # 역할 결정: 다수 의견이면 지지자, 아니면 반대자
            if response.verdict == majority_verdict:
                role = DebateRole.PROPONENT
                content = self._generate_support_argument(response)
            else:
                role = DebateRole.OPPONENT
                content = self._generate_counter_argument(response, majority_verdict)

            message = DebateMessage(
                agent_name=name,
                role=role,
                content=content,
                evidence=response.arguments[:2],  # 상위 2개 논거
                round_number=state.current_round
            )
            messages.append(message)

        # 상태 업데이트
        state.messages.extend(messages)
        state.verdict_history.append(current_verdicts)
        state.confidence_history.append({
            name: r.confidence for name, r in responses.items()
        })

        return messages, state

    def _get_majority_verdict(self, responses: Dict[str, AgentResponse]) -> Verdict:
        """다수 의견 판정"""
        verdict_counts: Dict[Verdict, float] = {}
        for response in responses.values():
            v = response.verdict
            verdict_counts[v] = verdict_counts.get(v, 0) + response.confidence

        return max(verdict_counts.keys(), key=lambda v: verdict_counts[v])

    def _generate_support_argument(self, response: AgentResponse) -> str:
        """지지 논거 생성"""
        return (
            f"저는 {response.verdict.value} 판정을 지지합니다. "
            f"근거: {response.arguments[0] if response.arguments else response.reasoning[:100]}"
        )

    def _generate_counter_argument(
        self,
        response: AgentResponse,
        majority: Verdict
    ) -> str:
        """반론 생성"""
        return (
            f"다수 의견인 {majority.value}에 동의하지 않습니다. "
            f"저는 {response.verdict.value}가 맞다고 봅니다. "
            f"근거: {response.arguments[0] if response.arguments else response.reasoning[:100]}"
        )


class AsynchronousDebate(DebateProtocol):
    """
    비동기식 토론 프로토콜

    에이전트가 순차적으로 발언
    이전 모든 발언을 참조하여 더 정교한 논증 가능
    MAD-Sherlock 논문에서 가장 좋은 성능을 보인 방식
    """

    def __init__(
        self,
        max_rounds: int = 3,
        convergence_threshold: float = 0.1,
        speaking_order: Optional[List[str]] = None
    ):
        super().__init__(max_rounds, convergence_threshold)
        self.speaking_order = speaking_order  # 발언 순서 (None이면 신뢰도 순)

    def conduct_round(
        self,
        responses: Dict[str, AgentResponse],
        state: DebateState
    ) -> Tuple[List[DebateMessage], DebateState]:
        """비동기식 토론 라운드"""
        messages = []
        state.current_round += 1

        # 발언 순서 결정
        order = self._determine_speaking_order(responses)

        # 이 라운드에서 축적되는 컨텍스트
        round_context = []

        for name in order:
            response = responses[name]

            # 이전 발언들을 참조한 응답 생성
            content = self._generate_contextual_response(
                response, round_context, state.messages
            )

            # 역할 결정
            role = self._determine_role(response, responses, round_context)

            message = DebateMessage(
                agent_name=name,
                role=role,
                content=content,
                evidence=response.arguments,
                round_number=state.current_round
            )

            messages.append(message)
            round_context.append(message)

        # 상태 업데이트
        state.messages.extend(messages)
        state.verdict_history.append({
            name: r.verdict for name, r in responses.items()
        })
        state.confidence_history.append({
            name: r.confidence for name, r in responses.items()
        })

        return messages, state

    def _determine_speaking_order(
        self,
        responses: Dict[str, AgentResponse]
    ) -> List[str]:
        """발언 순서 결정"""
        if self.speaking_order:
            return [n for n in self.speaking_order if n in responses]

        # 신뢰도 역순 (낮은 신뢰도부터 발언)
        return sorted(
            responses.keys(),
            key=lambda n: responses[n].confidence
        )

    def _generate_contextual_response(
        self,
        response: AgentResponse,
        round_context: List[DebateMessage],
        history: List[DebateMessage]
    ) -> str:
        """컨텍스트 기반 응답 생성"""
        parts = []

        # 이전 발언에 대한 반응
        if round_context:
            last_msg = round_context[-1]
            if last_msg.role == DebateRole.OPPONENT:
                parts.append(f"{last_msg.agent_name}의 반론에 대해, ")

        # 자신의 주장
        parts.append(f"저의 분석 결과 {response.verdict.value}입니다. ")

        # 증거 제시
        if response.arguments:
            parts.append(f"핵심 근거: {response.arguments[0]}")

        return "".join(parts)

    def _determine_role(
        self,
        response: AgentResponse,
        all_responses: Dict[str, AgentResponse],
        context: List[DebateMessage]
    ) -> DebateRole:
        """역할 결정"""
        if not context:
            return DebateRole.PROPONENT

        # 이전 발언자와 의견이 다르면 반대자
        last_verdict = all_responses.get(
            context[-1].agent_name, response
        ).verdict

        if response.verdict != last_verdict:
            return DebateRole.OPPONENT
        return DebateRole.PROPONENT


class StructuredDebate(DebateProtocol):
    """
    구조화된 토론 프로토콜

    역할 기반 토론:
    1. 주장 (Claim): 초기 판정 제시
    2. 반론 (Rebuttal): 다른 의견 제시
    3. 재반론 (Rejoinder): 반론에 대한 방어
    4. 요약 (Summary): 최종 입장 정리
    """

    def __init__(self, max_rounds: int = 4, convergence_threshold: float = 0.1):
        super().__init__(max_rounds, convergence_threshold)
        self.phases = ["claim", "rebuttal", "rejoinder", "summary"]

    def conduct_round(
        self,
        responses: Dict[str, AgentResponse],
        state: DebateState
    ) -> Tuple[List[DebateMessage], DebateState]:
        """구조화된 토론 라운드"""
        messages = []
        state.current_round += 1

        phase = self.phases[min(state.current_round - 1, len(self.phases) - 1)]

        for name, response in responses.items():
            content = self._generate_phase_content(response, phase, state)
            role = self._phase_to_role(phase, response, responses)

            message = DebateMessage(
                agent_name=name,
                role=role,
                content=content,
                evidence=response.arguments,
                round_number=state.current_round
            )
            messages.append(message)

        state.messages.extend(messages)
        state.verdict_history.append({
            name: r.verdict for name, r in responses.items()
        })
        state.confidence_history.append({
            name: r.confidence for name, r in responses.items()
        })

        return messages, state

    def _generate_phase_content(
        self,
        response: AgentResponse,
        phase: str,
        state: DebateState
    ) -> str:
        """단계별 콘텐츠 생성"""
        if phase == "claim":
            return (
                f"[주장] {response.verdict.value}로 판정합니다. "
                f"신뢰도: {response.confidence:.1%}"
            )
        elif phase == "rebuttal":
            return (
                f"[반론] 다른 의견에 대해: "
                f"{response.arguments[0] if response.arguments else '추가 증거 필요'}"
            )
        elif phase == "rejoinder":
            return (
                f"[재반론] 제 분석의 핵심 증거: "
                f"{response.evidence.get('key_finding', response.reasoning[:50])}"
            )
        else:  # summary
            return (
                f"[요약] 최종 판정 {response.verdict.value}, "
                f"신뢰도 {response.confidence:.1%}를 유지합니다."
            )

    def _phase_to_role(
        self,
        phase: str,
        response: AgentResponse,
        all_responses: Dict[str, AgentResponse]
    ) -> DebateRole:
        """단계를 역할로 변환"""
        role_map = {
            "claim": DebateRole.PROPONENT,
            "rebuttal": DebateRole.OPPONENT,
            "rejoinder": DebateRole.PROPONENT,
            "summary": DebateRole.MODERATOR
        }
        return role_map.get(phase, DebateRole.PROPONENT)
