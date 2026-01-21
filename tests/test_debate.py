"""
Debate 시스템 테스트

토론 프로토콜과 DebateChamber의 기능을 검증합니다.
"""
import pytest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tools.base_tool import Verdict
from src.agents.base_agent import AgentResponse, AgentRole
from src.debate.protocols import (
    SynchronousDebate,
    AsynchronousDebate,
    StructuredDebate,
    DebateState,
    DebateMessage,
    DebateRole
)
from src.debate.debate_chamber import DebateChamber, DebateResult


def create_mock_response(
    name: str,
    role: AgentRole,
    verdict: Verdict,
    confidence: float,
    arguments: list = None
) -> AgentResponse:
    """테스트용 모의 AgentResponse 생성"""
    return AgentResponse(
        agent_name=name,
        role=role,
        verdict=verdict,
        confidence=confidence,
        reasoning=f"{name} analysis result",
        evidence={"ai_generation_score": confidence if verdict == Verdict.AI_GENERATED else 1 - confidence},
        arguments=arguments or [f"{name} primary argument", f"{name} secondary argument"]
    )


class TestSynchronousDebate:
    """동기식 토론 프로토콜 테스트"""

    def setup_method(self):
        """테스트 전 설정"""
        self.protocol = SynchronousDebate(max_rounds=3, convergence_threshold=0.1)

    def test_conduct_round(self):
        """라운드 진행 테스트"""
        responses = {
            "freq": create_mock_response("Frequency Agent", AgentRole.FREQUENCY,
                                         Verdict.AI_GENERATED, 0.85),
            "noise": create_mock_response("Noise Agent", AgentRole.NOISE,
                                          Verdict.AUTHENTIC, 0.70),
        }
        state = DebateState()

        messages, new_state = self.protocol.conduct_round(responses, state)

        assert len(messages) == 2
        assert new_state.current_round == 1
        assert len(new_state.messages) == 2

    def test_message_structure(self):
        """메시지 구조 검증"""
        responses = {
            "freq": create_mock_response("Frequency Agent", AgentRole.FREQUENCY,
                                         Verdict.AI_GENERATED, 0.85),
        }
        state = DebateState()

        messages, _ = self.protocol.conduct_round(responses, state)

        msg = messages[0]
        assert isinstance(msg, DebateMessage)
        assert msg.agent_name == "freq"
        assert msg.role in [DebateRole.PROPONENT, DebateRole.OPPONENT]
        assert msg.content is not None
        assert msg.round_number == 1

    def test_proponent_opponent_assignment(self):
        """지지자/반대자 역할 할당 테스트"""
        # 다수가 AI_GENERATED를 지지
        responses = {
            "agent1": create_mock_response("Agent1", AgentRole.FREQUENCY,
                                           Verdict.AI_GENERATED, 0.9),
            "agent2": create_mock_response("Agent2", AgentRole.NOISE,
                                           Verdict.AI_GENERATED, 0.8),
            "agent3": create_mock_response("Agent3", AgentRole.SPATIAL,
                                           Verdict.AUTHENTIC, 0.7),
        }
        state = DebateState()

        messages, _ = self.protocol.conduct_round(responses, state)

        # agent3은 소수 의견이므로 OPPONENT
        agent3_msg = [m for m in messages if m.agent_name == "agent3"][0]
        assert agent3_msg.role == DebateRole.OPPONENT

    def test_convergence_check_unanimous(self):
        """만장일치 수렴 확인"""
        responses = {
            "agent1": create_mock_response("Agent1", AgentRole.FREQUENCY,
                                           Verdict.AI_GENERATED, 0.9),
            "agent2": create_mock_response("Agent2", AgentRole.NOISE,
                                           Verdict.AI_GENERATED, 0.85),
        }
        state = DebateState()

        is_converged = self.protocol.check_convergence(responses, state)

        assert is_converged == True
        assert state.is_converged == True
        assert state.convergence_reason == "만장일치"


class TestAsynchronousDebate:
    """비동기식 토론 프로토콜 테스트"""

    def setup_method(self):
        """테스트 전 설정"""
        self.protocol = AsynchronousDebate(max_rounds=3)

    def test_speaking_order(self):
        """발언 순서 테스트 (신뢰도 낮은 순)"""
        responses = {
            "high_conf": create_mock_response("High Confidence", AgentRole.FREQUENCY,
                                              Verdict.AI_GENERATED, 0.95),
            "low_conf": create_mock_response("Low Confidence", AgentRole.NOISE,
                                             Verdict.AUTHENTIC, 0.5),
            "mid_conf": create_mock_response("Mid Confidence", AgentRole.SPATIAL,
                                             Verdict.AI_GENERATED, 0.75),
        }
        state = DebateState()

        messages, _ = self.protocol.conduct_round(responses, state)

        # 첫 번째 발언자는 신뢰도가 가장 낮은 에이전트
        assert messages[0].agent_name == "low_conf"

    def test_contextual_response_generation(self):
        """컨텍스트 기반 응답 생성"""
        responses = {
            "agent1": create_mock_response("Agent1", AgentRole.FREQUENCY,
                                           Verdict.AI_GENERATED, 0.9),
            "agent2": create_mock_response("Agent2", AgentRole.NOISE,
                                           Verdict.AUTHENTIC, 0.7),
        }
        state = DebateState()

        messages, _ = self.protocol.conduct_round(responses, state)

        # 두 번째 메시지는 첫 번째를 참조해야 함
        assert len(messages) == 2
        # 컨텍스트 응답은 내용에 반영됨


class TestStructuredDebate:
    """구조화된 토론 프로토콜 테스트"""

    def setup_method(self):
        """테스트 전 설정"""
        self.protocol = StructuredDebate(max_rounds=4)

    def test_phase_progression(self):
        """단계별 진행 테스트"""
        responses = {
            "agent1": create_mock_response("Agent1", AgentRole.FREQUENCY,
                                           Verdict.AI_GENERATED, 0.9),
        }
        state = DebateState()

        # Round 1: claim
        messages1, state = self.protocol.conduct_round(responses, state)
        assert "[주장]" in messages1[0].content

        # Round 2: rebuttal
        messages2, state = self.protocol.conduct_round(responses, state)
        assert "[반론]" in messages2[0].content

        # Round 3: rejoinder
        messages3, state = self.protocol.conduct_round(responses, state)
        assert "[재반론]" in messages3[0].content

        # Round 4: summary
        messages4, state = self.protocol.conduct_round(responses, state)
        assert "[요약]" in messages4[0].content


class TestDebateChamber:
    """DebateChamber 통합 테스트"""

    def setup_method(self):
        """테스트 전 설정"""
        self.chamber = DebateChamber(disagreement_threshold=0.3)

    def test_should_debate_different_verdicts(self):
        """다른 판정 시 토론 필요"""
        responses = {
            "agent1": create_mock_response("Agent1", AgentRole.FREQUENCY,
                                           Verdict.AI_GENERATED, 0.9),
            "agent2": create_mock_response("Agent2", AgentRole.NOISE,
                                           Verdict.AUTHENTIC, 0.7),
        }

        should_debate = self.chamber.should_debate(responses)
        assert should_debate == True

    def test_should_not_debate_same_verdicts(self):
        """같은 판정 시 토론 불필요"""
        responses = {
            "agent1": create_mock_response("Agent1", AgentRole.FREQUENCY,
                                           Verdict.AI_GENERATED, 0.9),
            "agent2": create_mock_response("Agent2", AgentRole.NOISE,
                                           Verdict.AI_GENERATED, 0.85),
        }

        should_debate = self.chamber.should_debate(responses)
        assert should_debate == False

    def test_conduct_debate(self):
        """토론 진행 테스트"""
        responses = {
            "freq": create_mock_response("Frequency Agent", AgentRole.FREQUENCY,
                                         Verdict.AI_GENERATED, 0.85,
                                         ["Grid artifact detected", "High frequency peaks"]),
            "noise": create_mock_response("Noise Agent", AgentRole.NOISE,
                                          Verdict.AUTHENTIC, 0.65,
                                          ["Natural PRNU pattern", "Low noise variance"]),
            "spatial": create_mock_response("Spatial Agent", AgentRole.SPATIAL,
                                            Verdict.AI_GENERATED, 0.75,
                                            ["Large manipulation area", "Texture inconsistency"]),
        }
        trust_scores = {"freq": 0.85, "noise": 0.80, "spatial": 0.85}

        result = self.chamber.conduct_debate(responses, trust_scores=trust_scores)

        assert isinstance(result, DebateResult)
        assert result.total_rounds > 0
        assert len(result.rounds) > 0
        assert result.consensus_result is not None

    def test_debate_result_structure(self):
        """토론 결과 구조 검증"""
        responses = {
            "agent1": create_mock_response("Agent1", AgentRole.FREQUENCY,
                                           Verdict.AI_GENERATED, 0.9),
            "agent2": create_mock_response("Agent2", AgentRole.NOISE,
                                           Verdict.AUTHENTIC, 0.7),
        }

        result = self.chamber.conduct_debate(responses)

        # 필수 필드 확인
        assert hasattr(result, 'initial_verdicts')
        assert hasattr(result, 'final_verdicts')
        assert hasattr(result, 'rounds')
        assert hasattr(result, 'convergence_achieved')
        assert hasattr(result, 'consensus_result')

    def test_debate_convergence(self):
        """토론 수렴 테스트"""
        # 의견이 크게 다른 경우
        responses = {
            "agent1": create_mock_response("Agent1", AgentRole.FREQUENCY,
                                           Verdict.AI_GENERATED, 0.95),
            "agent2": create_mock_response("Agent2", AgentRole.NOISE,
                                           Verdict.AUTHENTIC, 0.3),
        }

        result = self.chamber.conduct_debate(responses)

        # 최대 라운드 내에서 수렴 또는 종료
        assert result.total_rounds <= self.chamber.protocol.max_rounds
        assert result.convergence_reason in ["만장일치", "신뢰도 수렴", "최대 라운드 도달"]

    def test_to_dict(self):
        """to_dict 메서드 테스트"""
        responses = {
            "agent1": create_mock_response("Agent1", AgentRole.FREQUENCY,
                                           Verdict.AI_GENERATED, 0.9),
            "agent2": create_mock_response("Agent2", AgentRole.NOISE,
                                           Verdict.AUTHENTIC, 0.7),
        }

        result = self.chamber.conduct_debate(responses)
        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert "initial_verdicts" in result_dict
        assert "final_verdicts" in result_dict
        assert "total_rounds" in result_dict
        assert "convergence_achieved" in result_dict

    def test_get_summary(self):
        """요약 생성 테스트"""
        responses = {
            "agent1": create_mock_response("Agent1", AgentRole.FREQUENCY,
                                           Verdict.AI_GENERATED, 0.9),
            "agent2": create_mock_response("Agent2", AgentRole.NOISE,
                                           Verdict.AUTHENTIC, 0.7),
        }

        result = self.chamber.conduct_debate(responses)
        summary = result.get_summary()

        assert isinstance(summary, str)
        assert "토론 결과 요약" in summary
        assert "라운드 진행" in summary

    def test_generate_debate_transcript(self):
        """토론 기록 생성 테스트"""
        responses = {
            "agent1": create_mock_response("Agent1", AgentRole.FREQUENCY,
                                           Verdict.AI_GENERATED, 0.9),
            "agent2": create_mock_response("Agent2", AgentRole.NOISE,
                                           Verdict.AUTHENTIC, 0.7),
        }

        result = self.chamber.conduct_debate(responses)
        transcript = self.chamber.generate_debate_transcript(result)

        assert isinstance(transcript, str)
        assert "MAIFS 다중 에이전트 토론 기록" in transcript


class TestDebateProtocolComparison:
    """토론 프로토콜 비교 테스트"""

    def test_all_protocols_produce_valid_output(self):
        """모든 프로토콜이 유효한 출력 생성"""
        protocols = [
            SynchronousDebate(max_rounds=2),
            AsynchronousDebate(max_rounds=2),
            StructuredDebate(max_rounds=2),
        ]

        responses = {
            "agent1": create_mock_response("Agent1", AgentRole.FREQUENCY,
                                           Verdict.AI_GENERATED, 0.9),
            "agent2": create_mock_response("Agent2", AgentRole.NOISE,
                                           Verdict.AUTHENTIC, 0.7),
        }

        for protocol in protocols:
            chamber = DebateChamber(protocol=protocol)
            result = chamber.conduct_debate(responses)

            assert result is not None
            assert result.total_rounds > 0
            assert len(result.rounds) > 0


class TestEdgeCases:
    """엣지 케이스 테스트"""

    def setup_method(self):
        """테스트 전 설정"""
        self.chamber = DebateChamber()

    def test_single_agent_no_debate(self):
        """단일 에이전트 - 토론 불필요"""
        responses = {
            "agent1": create_mock_response("Agent1", AgentRole.FREQUENCY,
                                           Verdict.AI_GENERATED, 0.9),
        }

        should_debate = self.chamber.should_debate(responses)
        assert should_debate == False

    def test_all_uncertain_debate(self):
        """모두 UNCERTAIN인 경우"""
        responses = {
            "agent1": create_mock_response("Agent1", AgentRole.FREQUENCY,
                                           Verdict.UNCERTAIN, 0.5),
            "agent2": create_mock_response("Agent2", AgentRole.NOISE,
                                           Verdict.UNCERTAIN, 0.4),
        }

        result = self.chamber.conduct_debate(responses)

        # 에러 없이 완료되어야 함
        assert result is not None
        assert result.consensus_result.final_verdict == Verdict.UNCERTAIN

    def test_high_confidence_disparity(self):
        """신뢰도 차이가 큰 경우"""
        responses = {
            "high": create_mock_response("High", AgentRole.FREQUENCY,
                                         Verdict.AI_GENERATED, 0.99),
            "low": create_mock_response("Low", AgentRole.NOISE,
                                        Verdict.AUTHENTIC, 0.01),
        }

        # 신뢰도 차이가 임계값 초과
        should_debate = self.chamber.should_debate(responses)
        assert should_debate == True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
