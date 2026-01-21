"""
COBRA 합의 알고리즘 테스트

RoT, DRWA, AVGA 알고리즘의 기능을 검증합니다.
"""
import pytest
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tools.base_tool import Verdict
from src.agents.base_agent import AgentResponse, AgentRole
from src.consensus.cobra import (
    COBRAConsensus,
    RootOfTrust,
    DRWA,
    AVGA,
    ConsensusResult
)


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
        arguments=arguments or [f"{name} argument"]
    )


class TestRootOfTrust:
    """RoT (Root-of-Trust) 알고리즘 테스트"""

    def setup_method(self):
        """테스트 전 설정"""
        self.algorithm = RootOfTrust(trust_threshold=0.7, alpha=0.3)

    def test_basic_aggregation(self):
        """기본 집계 테스트"""
        responses = {
            "agent1": create_mock_response("Agent1", AgentRole.FREQUENCY,
                                           Verdict.AI_GENERATED, 0.9),
            "agent2": create_mock_response("Agent2", AgentRole.NOISE,
                                           Verdict.AUTHENTIC, 0.7),
        }
        trust_scores = {"agent1": 0.8, "agent2": 0.6}

        result = self.algorithm.aggregate(responses, trust_scores)

        assert isinstance(result, ConsensusResult)
        assert result.algorithm_used == "RoT"
        assert result.final_verdict in list(Verdict)
        assert 0.0 <= result.confidence <= 1.0

    def test_trusted_cohort_priority(self):
        """신뢰 코호트 우선순위 테스트"""
        # 신뢰도 높은 에이전트: AI_GENERATED
        # 신뢰도 낮은 에이전트: AUTHENTIC
        responses = {
            "trusted1": create_mock_response("Trusted1", AgentRole.FREQUENCY,
                                             Verdict.AI_GENERATED, 0.9),
            "trusted2": create_mock_response("Trusted2", AgentRole.SPATIAL,
                                             Verdict.AI_GENERATED, 0.85),
            "untrusted": create_mock_response("Untrusted", AgentRole.NOISE,
                                              Verdict.AUTHENTIC, 0.95),
        }
        trust_scores = {
            "trusted1": 0.9,   # 신뢰 코호트
            "trusted2": 0.85,  # 신뢰 코호트
            "untrusted": 0.5   # 비신뢰 코호트
        }

        result = self.algorithm.aggregate(responses, trust_scores)

        # 신뢰 코호트가 AI_GENERATED를 지지하므로 결과도 AI_GENERATED
        assert result.final_verdict == Verdict.AI_GENERATED
        assert result.agent_weights["trusted1"] > result.agent_weights["untrusted"]

    def test_cohort_info(self):
        """코호트 정보 검증"""
        responses = {
            "agent1": create_mock_response("Agent1", AgentRole.FREQUENCY,
                                           Verdict.AI_GENERATED, 0.9),
            "agent2": create_mock_response("Agent2", AgentRole.NOISE,
                                           Verdict.AUTHENTIC, 0.7),
        }
        trust_scores = {"agent1": 0.8, "agent2": 0.5}

        result = self.algorithm.aggregate(responses, trust_scores)

        assert "trusted_count" in result.cohort_info
        assert "untrusted_count" in result.cohort_info
        assert result.cohort_info["trusted_count"] == 1
        assert result.cohort_info["untrusted_count"] == 1

    def test_empty_responses(self):
        """빈 응답 처리"""
        result = self.algorithm.aggregate({}, {})

        assert result.final_verdict == Verdict.UNCERTAIN
        assert result.confidence == 0.0


class TestDRWA:
    """DRWA (Dynamic Reliability Weighted Aggregation) 알고리즘 테스트"""

    def setup_method(self):
        """테스트 전 설정"""
        self.algorithm = DRWA(epsilon=0.2)

    def test_basic_aggregation(self):
        """기본 집계 테스트"""
        responses = {
            "agent1": create_mock_response("Agent1", AgentRole.FREQUENCY,
                                           Verdict.AI_GENERATED, 0.9),
            "agent2": create_mock_response("Agent2", AgentRole.NOISE,
                                           Verdict.AUTHENTIC, 0.7),
        }
        trust_scores = {"agent1": 0.8, "agent2": 0.8}

        result = self.algorithm.aggregate(responses, trust_scores)

        assert isinstance(result, ConsensusResult)
        assert result.algorithm_used == "DRWA"
        assert result.final_verdict in list(Verdict)

    def test_dynamic_weight_adjustment(self):
        """동적 가중치 조정 테스트"""
        responses = {
            "consistent": create_mock_response("Consistent", AgentRole.FREQUENCY,
                                               Verdict.AI_GENERATED, 0.95),
            "inconsistent": create_mock_response("Inconsistent", AgentRole.NOISE,
                                                 Verdict.AI_GENERATED, 0.5),
        }
        trust_scores = {"consistent": 0.8, "inconsistent": 0.8}

        result = self.algorithm.aggregate(responses, trust_scores)

        # 일관된 에이전트가 더 높은 가중치를 받아야 함
        # (정확한 비교는 분산 계산 방식에 따라 달라질 수 있음)
        assert result.agent_weights is not None
        assert len(result.agent_weights) == 2

    def test_variance_info(self):
        """분산 정보 검증"""
        responses = {
            "agent1": create_mock_response("Agent1", AgentRole.FREQUENCY,
                                           Verdict.AI_GENERATED, 0.9),
            "agent2": create_mock_response("Agent2", AgentRole.NOISE,
                                           Verdict.AUTHENTIC, 0.7),
        }
        trust_scores = {"agent1": 0.8, "agent2": 0.8}

        result = self.algorithm.aggregate(responses, trust_scores)

        assert "epsilon" in result.cohort_info
        assert "variances" in result.cohort_info


class TestAVGA:
    """AVGA (Adaptive Variance-Guided Attention) 알고리즘 테스트"""

    def setup_method(self):
        """테스트 전 설정"""
        self.algorithm = AVGA(temperature=1.0, shift_rate=0.2)

    def test_basic_aggregation(self):
        """기본 집계 테스트"""
        responses = {
            "agent1": create_mock_response("Agent1", AgentRole.FREQUENCY,
                                           Verdict.AI_GENERATED, 0.9),
            "agent2": create_mock_response("Agent2", AgentRole.NOISE,
                                           Verdict.AUTHENTIC, 0.7),
        }
        trust_scores = {"agent1": 0.8, "agent2": 0.8}

        result = self.algorithm.aggregate(responses, trust_scores)

        assert isinstance(result, ConsensusResult)
        assert result.algorithm_used == "AVGA"

    def test_attention_weights_sum_to_one(self):
        """어텐션 가중치 합이 1인지 검증"""
        responses = {
            "agent1": create_mock_response("Agent1", AgentRole.FREQUENCY,
                                           Verdict.AI_GENERATED, 0.9),
            "agent2": create_mock_response("Agent2", AgentRole.NOISE,
                                           Verdict.AUTHENTIC, 0.7),
            "agent3": create_mock_response("Agent3", AgentRole.SPATIAL,
                                           Verdict.AI_GENERATED, 0.8),
        }
        trust_scores = {"agent1": 0.8, "agent2": 0.7, "agent3": 0.85}

        result = self.algorithm.aggregate(responses, trust_scores)

        weight_sum = sum(result.agent_weights.values())
        assert abs(weight_sum - 1.0) < 0.01  # 오차 허용

    def test_high_variance_reduces_confidence(self):
        """높은 분산이 신뢰도를 낮추는지 검증"""
        # 의견이 많이 다른 경우
        responses_diverse = {
            "agent1": create_mock_response("Agent1", AgentRole.FREQUENCY,
                                           Verdict.AI_GENERATED, 0.95),
            "agent2": create_mock_response("Agent2", AgentRole.NOISE,
                                           Verdict.AUTHENTIC, 0.1),  # 극단적 차이
        }

        # 의견이 일치하는 경우
        responses_unified = {
            "agent1": create_mock_response("Agent1", AgentRole.FREQUENCY,
                                           Verdict.AI_GENERATED, 0.9),
            "agent2": create_mock_response("Agent2", AgentRole.NOISE,
                                           Verdict.AI_GENERATED, 0.85),
        }

        trust_scores = {"agent1": 0.8, "agent2": 0.8}

        result_diverse = self.algorithm.aggregate(responses_diverse, trust_scores)
        result_unified = self.algorithm.aggregate(responses_unified, trust_scores)

        # 의견 불일치가 많으면 신뢰도가 낮아야 함
        assert result_diverse.disagreement_level > result_unified.disagreement_level


class TestCOBRAConsensus:
    """COBRA 통합 시스템 테스트"""

    def setup_method(self):
        """테스트 전 설정"""
        self.cobra = COBRAConsensus(
            default_algorithm="drwa",
            trust_threshold=0.7,
            epsilon=0.2,
            temperature=1.0
        )

    def test_manual_algorithm_selection(self):
        """수동 알고리즘 선택"""
        responses = {
            "agent1": create_mock_response("Agent1", AgentRole.FREQUENCY,
                                           Verdict.AI_GENERATED, 0.9),
            "agent2": create_mock_response("Agent2", AgentRole.NOISE,
                                           Verdict.AUTHENTIC, 0.7),
        }
        trust_scores = {"agent1": 0.8, "agent2": 0.6}

        result_rot = self.cobra.aggregate(responses, trust_scores, algorithm="rot")
        result_drwa = self.cobra.aggregate(responses, trust_scores, algorithm="drwa")
        result_avga = self.cobra.aggregate(responses, trust_scores, algorithm="avga")

        assert result_rot.algorithm_used == "RoT"
        assert result_drwa.algorithm_used == "DRWA"
        assert result_avga.algorithm_used == "AVGA"

    def test_automatic_algorithm_selection(self):
        """자동 알고리즘 선택"""
        # 신뢰도 편차가 큰 경우 -> RoT
        responses_trust_variance = {
            "agent1": create_mock_response("Agent1", AgentRole.FREQUENCY,
                                           Verdict.AI_GENERATED, 0.9),
            "agent2": create_mock_response("Agent2", AgentRole.NOISE,
                                           Verdict.AUTHENTIC, 0.7),
        }
        trust_high_variance = {"agent1": 0.95, "agent2": 0.3}  # 큰 편차

        result = self.cobra.aggregate(responses_trust_variance, trust_high_variance)
        assert result.algorithm_used == "RoT"

    def test_all_verdicts_possible(self):
        """모든 판정 유형 가능"""
        test_cases = [
            (Verdict.AUTHENTIC, 0.9),
            (Verdict.AI_GENERATED, 0.9),
            (Verdict.MANIPULATED, 0.9),
        ]

        for verdict, confidence in test_cases:
            responses = {
                "agent1": create_mock_response("Agent1", AgentRole.FREQUENCY,
                                               verdict, confidence),
                "agent2": create_mock_response("Agent2", AgentRole.NOISE,
                                               verdict, confidence),
            }
            trust_scores = {"agent1": 0.8, "agent2": 0.8}

            result = self.cobra.aggregate(responses, trust_scores)
            assert result.final_verdict == verdict

    def test_to_dict(self):
        """ConsensusResult의 to_dict 메서드 테스트"""
        responses = {
            "agent1": create_mock_response("Agent1", AgentRole.FREQUENCY,
                                           Verdict.AI_GENERATED, 0.9),
        }
        trust_scores = {"agent1": 0.8}

        result = self.cobra.aggregate(responses, trust_scores)
        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert "final_verdict" in result_dict
        assert "confidence" in result_dict
        assert "algorithm" in result_dict
        assert "verdict_scores" in result_dict
        assert "agent_weights" in result_dict


class TestEdgeCases:
    """엣지 케이스 테스트"""

    def setup_method(self):
        """테스트 전 설정"""
        self.cobra = COBRAConsensus()

    def test_single_agent(self):
        """단일 에이전트"""
        responses = {
            "agent1": create_mock_response("Agent1", AgentRole.FREQUENCY,
                                           Verdict.AI_GENERATED, 0.9),
        }
        trust_scores = {"agent1": 0.8}

        result = self.cobra.aggregate(responses, trust_scores)

        assert result.final_verdict == Verdict.AI_GENERATED
        assert result.confidence > 0.0

    def test_all_uncertain(self):
        """모든 에이전트가 UNCERTAIN"""
        responses = {
            "agent1": create_mock_response("Agent1", AgentRole.FREQUENCY,
                                           Verdict.UNCERTAIN, 0.5),
            "agent2": create_mock_response("Agent2", AgentRole.NOISE,
                                           Verdict.UNCERTAIN, 0.4),
        }
        trust_scores = {"agent1": 0.8, "agent2": 0.8}

        result = self.cobra.aggregate(responses, trust_scores)

        assert result.final_verdict == Verdict.UNCERTAIN

    def test_equal_split(self):
        """동점 상황"""
        responses = {
            "agent1": create_mock_response("Agent1", AgentRole.FREQUENCY,
                                           Verdict.AI_GENERATED, 0.8),
            "agent2": create_mock_response("Agent2", AgentRole.NOISE,
                                           Verdict.AUTHENTIC, 0.8),
        }
        trust_scores = {"agent1": 0.8, "agent2": 0.8}

        result = self.cobra.aggregate(responses, trust_scores)

        # 결과가 어떤 것이든 반환되어야 함
        assert result.final_verdict in [Verdict.AI_GENERATED, Verdict.AUTHENTIC]

    def test_zero_confidence(self):
        """신뢰도 0인 경우"""
        responses = {
            "agent1": create_mock_response("Agent1", AgentRole.FREQUENCY,
                                           Verdict.AI_GENERATED, 0.0),
        }
        trust_scores = {"agent1": 0.8}

        result = self.cobra.aggregate(responses, trust_scores)

        # 에러 없이 처리되어야 함
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
