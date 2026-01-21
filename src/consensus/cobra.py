"""
COBRA (COnsensus-Based RewArd) 합의 알고리즘 구현

논문 기반: "Enhancing Alignment of LLMs via Multi-Rater Preference Aggregation"

세 가지 합의 전략:
1. RoT (Root-of-Trust): 신뢰 코호트 기반 가중 평균
2. DRWA (Dynamic Reliability Weighted Aggregation): 동적 신뢰도 조정
3. AVGA (Adaptive Variance-Guided Attention): 분산 기반 어텐션
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import numpy as np

from ..tools.base_tool import Verdict
from ..agents.base_agent import AgentResponse


@dataclass
class ConsensusResult:
    """합의 결과"""
    final_verdict: Verdict
    confidence: float
    algorithm_used: str
    verdict_scores: Dict[str, float] = field(default_factory=dict)
    agent_weights: Dict[str, float] = field(default_factory=dict)
    cohort_info: Dict[str, Any] = field(default_factory=dict)
    disagreement_level: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "final_verdict": self.final_verdict.value,
            "confidence": self.confidence,
            "algorithm": self.algorithm_used,
            "verdict_scores": self.verdict_scores,
            "agent_weights": self.agent_weights,
            "cohort_info": self.cohort_info,
            "disagreement_level": self.disagreement_level
        }


class ConsensusAlgorithm(ABC):
    """합의 알고리즘 기본 클래스"""

    @abstractmethod
    def aggregate(
        self,
        responses: Dict[str, AgentResponse],
        trust_scores: Dict[str, float]
    ) -> ConsensusResult:
        """
        에이전트 응답 집계

        Args:
            responses: 에이전트별 응답
            trust_scores: 에이전트별 신뢰도 점수

        Returns:
            ConsensusResult: 합의 결과
        """
        pass


class RootOfTrust(ConsensusAlgorithm):
    """
    RoT (Root-of-Trust) 합의 알고리즘

    신뢰할 수 있는 에이전트(Trusted Cohort)와
    신뢰할 수 없는 에이전트(Untrusted Cohort)를 구분하여 집계

    수식:
    S_RoT = (Σ_{i∈Tc} w_i * v_i + α * Σ_{j∈Uc} w_j * v_j) / (Σ w_i + α * Σ w_j)

    - Tc: Trusted Cohort
    - Uc: Untrusted Cohort
    - α: Untrusted 가중치 감쇠 계수 (0 < α < 1)
    """

    def __init__(self, trust_threshold: float = 0.7, alpha: float = 0.3):
        """
        Args:
            trust_threshold: 신뢰 코호트 분류 임계값
            alpha: 비신뢰 코호트 가중치 감쇠 계수
        """
        self.trust_threshold = trust_threshold
        self.alpha = alpha

    def aggregate(
        self,
        responses: Dict[str, AgentResponse],
        trust_scores: Dict[str, float]
    ) -> ConsensusResult:
        """RoT 집계"""
        if not responses:
            return ConsensusResult(
                final_verdict=Verdict.UNCERTAIN,
                confidence=0.0,
                algorithm_used="RoT"
            )

        # 코호트 분류
        trusted_cohort = []
        untrusted_cohort = []

        for name, response in responses.items():
            trust = trust_scores.get(name, 0.5)
            if trust >= self.trust_threshold:
                trusted_cohort.append((name, response, trust))
            else:
                untrusted_cohort.append((name, response, trust))

        # 판정별 점수 집계
        verdict_scores: Dict[Verdict, float] = {}
        total_weight = 0.0

        # Trusted Cohort (가중치 1.0)
        for name, response, trust in trusted_cohort:
            weight = trust * response.confidence
            verdict = response.verdict

            if verdict not in verdict_scores:
                verdict_scores[verdict] = 0.0
            verdict_scores[verdict] += weight
            total_weight += weight

        # Untrusted Cohort (가중치 α)
        for name, response, trust in untrusted_cohort:
            weight = self.alpha * trust * response.confidence
            verdict = response.verdict

            if verdict not in verdict_scores:
                verdict_scores[verdict] = 0.0
            verdict_scores[verdict] += weight
            total_weight += weight

        # 최종 판정
        if not verdict_scores:
            return ConsensusResult(
                final_verdict=Verdict.UNCERTAIN,
                confidence=0.0,
                algorithm_used="RoT"
            )

        final_verdict = max(verdict_scores.keys(), key=lambda v: verdict_scores[v])
        confidence = verdict_scores[final_verdict] / total_weight if total_weight > 0 else 0.0

        # 불일치 수준
        disagreement = self._calculate_disagreement(verdict_scores, total_weight)

        # 에이전트 가중치
        agent_weights = {}
        for name, _, trust in trusted_cohort:
            agent_weights[name] = trust
        for name, _, trust in untrusted_cohort:
            agent_weights[name] = self.alpha * trust

        return ConsensusResult(
            final_verdict=final_verdict,
            confidence=confidence,
            algorithm_used="RoT",
            verdict_scores={v.value: s / total_weight for v, s in verdict_scores.items()},
            agent_weights=agent_weights,
            cohort_info={
                "trusted_count": len(trusted_cohort),
                "untrusted_count": len(untrusted_cohort),
                "trust_threshold": self.trust_threshold,
                "alpha": self.alpha
            },
            disagreement_level=disagreement
        )

    def _calculate_disagreement(
        self,
        verdict_scores: Dict[Verdict, float],
        total_weight: float
    ) -> float:
        """불일치 수준 계산 (엔트로피 기반)"""
        if total_weight == 0 or len(verdict_scores) <= 1:
            return 0.0

        # 정규화된 확률
        probs = [s / total_weight for s in verdict_scores.values()]

        # 엔트로피 계산
        entropy = -sum(p * np.log2(p + 1e-10) for p in probs if p > 0)

        # 최대 엔트로피로 정규화
        max_entropy = np.log2(len(verdict_scores))

        return entropy / max_entropy if max_entropy > 0 else 0.0


class DRWA(ConsensusAlgorithm):
    """
    DRWA (Dynamic Reliability Weighted Aggregation) 합의 알고리즘

    각 에이전트의 신뢰도를 동적으로 조정하여 집계
    일관성 있는 에이전트의 가중치를 높이고, 편향된 에이전트의 가중치를 낮춤

    수식:
    ω_t = w_t + ε * (1 - σ_t / σ_max)

    - w_t: 기본 신뢰도
    - ε: 조정 계수
    - σ_t: 에이전트 t의 예측 분산
    - σ_max: 최대 분산
    """

    def __init__(self, epsilon: float = 0.2):
        """
        Args:
            epsilon: 동적 가중치 조정 계수
        """
        self.epsilon = epsilon

    def aggregate(
        self,
        responses: Dict[str, AgentResponse],
        trust_scores: Dict[str, float]
    ) -> ConsensusResult:
        """DRWA 집계"""
        if not responses:
            return ConsensusResult(
                final_verdict=Verdict.UNCERTAIN,
                confidence=0.0,
                algorithm_used="DRWA"
            )

        # 에이전트별 분산 계산 (신뢰도의 일관성)
        variances = self._calculate_variances(responses)
        max_variance = max(variances.values()) if variances else 1.0

        # 동적 가중치 계산
        dynamic_weights = {}
        for name, response in responses.items():
            base_trust = trust_scores.get(name, 0.5)
            variance = variances.get(name, 0.5)

            # 분산이 낮을수록 (일관성 높을수록) 가중치 증가
            variance_factor = 1 - (variance / (max_variance + 1e-10))
            dynamic_weight = base_trust + self.epsilon * variance_factor

            dynamic_weights[name] = max(0.1, min(1.0, dynamic_weight))

        # 판정별 점수 집계
        verdict_scores: Dict[Verdict, float] = {}
        total_weight = 0.0

        for name, response in responses.items():
            weight = dynamic_weights[name] * response.confidence
            verdict = response.verdict

            if verdict not in verdict_scores:
                verdict_scores[verdict] = 0.0
            verdict_scores[verdict] += weight
            total_weight += weight

        # 최종 판정
        if not verdict_scores:
            return ConsensusResult(
                final_verdict=Verdict.UNCERTAIN,
                confidence=0.0,
                algorithm_used="DRWA"
            )

        final_verdict = max(verdict_scores.keys(), key=lambda v: verdict_scores[v])
        confidence = verdict_scores[final_verdict] / total_weight if total_weight > 0 else 0.0

        # 불일치 수준
        disagreement = len(verdict_scores) / 4.0  # 최대 4개 판정

        return ConsensusResult(
            final_verdict=final_verdict,
            confidence=confidence,
            algorithm_used="DRWA",
            verdict_scores={v.value: s / total_weight for v, s in verdict_scores.items()},
            agent_weights=dynamic_weights,
            cohort_info={
                "epsilon": self.epsilon,
                "variances": variances,
                "max_variance": max_variance
            },
            disagreement_level=disagreement
        )

    def _calculate_variances(
        self,
        responses: Dict[str, AgentResponse]
    ) -> Dict[str, float]:
        """
        각 에이전트의 예측 분산 계산

        여기서는 증거의 일관성을 기반으로 분산을 추정
        """
        variances = {}

        for name, response in responses.items():
            # 증거 기반 분산 추정
            evidence = response.evidence

            # 다양한 지표의 분산을 종합
            variance_indicators = []

            # AI 점수와 신뢰도의 일치 여부
            if "ai_generation_score" in evidence:
                ai_score = evidence["ai_generation_score"]
                if response.verdict == Verdict.AI_GENERATED:
                    variance_indicators.append(1.0 - ai_score)  # 낮으면 일관적
                else:
                    variance_indicators.append(ai_score)  # 낮으면 일관적

            # 기본 분산
            if not variance_indicators:
                variance_indicators.append(0.5)

            variances[name] = np.mean(variance_indicators)

        return variances


class AVGA(ConsensusAlgorithm):
    """
    AVGA (Adaptive Variance-Guided Attention) 합의 알고리즘

    분산을 기반으로 어텐션을 적응적으로 조정
    불확실성이 높은 경우 더 보수적인 판단

    수식:
    A_t = softmax(Q * K_t^T / √d) * V_t
    여기서 Q는 분산 기반 쿼리, K는 신뢰도, V는 판정
    """

    def __init__(self, temperature: float = 1.0, shift_rate: float = 0.2):
        """
        Args:
            temperature: Softmax 온도 파라미터
            shift_rate: 어텐션 이동률
        """
        self.temperature = temperature
        self.shift_rate = shift_rate

    def aggregate(
        self,
        responses: Dict[str, AgentResponse],
        trust_scores: Dict[str, float]
    ) -> ConsensusResult:
        """AVGA 집계"""
        if not responses:
            return ConsensusResult(
                final_verdict=Verdict.UNCERTAIN,
                confidence=0.0,
                algorithm_used="AVGA"
            )

        # 전체 신뢰도의 분산 계산
        confidences = [r.confidence for r in responses.values()]
        overall_variance = np.var(confidences)

        # 분산에 따른 어텐션 조정
        # 분산이 높으면 더 균등한 가중치, 낮으면 최고 신뢰도에 집중
        attention_weights = self._compute_attention(responses, trust_scores, overall_variance)

        # 판정별 점수 집계
        verdict_scores: Dict[Verdict, float] = {}
        total_weight = 0.0

        for name, response in responses.items():
            weight = attention_weights[name]
            verdict = response.verdict

            if verdict not in verdict_scores:
                verdict_scores[verdict] = 0.0
            verdict_scores[verdict] += weight
            total_weight += weight

        # 최종 판정
        if not verdict_scores:
            return ConsensusResult(
                final_verdict=Verdict.UNCERTAIN,
                confidence=0.0,
                algorithm_used="AVGA"
            )

        final_verdict = max(verdict_scores.keys(), key=lambda v: verdict_scores[v])
        confidence = verdict_scores[final_verdict] / total_weight if total_weight > 0 else 0.0

        # 불확실성이 높으면 신뢰도 감쇠
        if overall_variance > 0.1:
            confidence *= (1 - self.shift_rate * overall_variance)

        return ConsensusResult(
            final_verdict=final_verdict,
            confidence=confidence,
            algorithm_used="AVGA",
            verdict_scores={v.value: s / total_weight for v, s in verdict_scores.items()},
            agent_weights=attention_weights,
            cohort_info={
                "temperature": self.temperature,
                "shift_rate": self.shift_rate,
                "overall_variance": overall_variance
            },
            disagreement_level=min(overall_variance * 3, 1.0)
        )

    def _compute_attention(
        self,
        responses: Dict[str, AgentResponse],
        trust_scores: Dict[str, float],
        overall_variance: float
    ) -> Dict[str, float]:
        """어텐션 가중치 계산"""
        # 기본 점수 계산
        scores = {}
        for name, response in responses.items():
            trust = trust_scores.get(name, 0.5)
            scores[name] = trust * response.confidence

        # 분산에 따른 온도 조정
        # 분산이 높으면 온도를 높여 균등 분포에 가깝게
        adjusted_temp = self.temperature * (1 + overall_variance)

        # Softmax
        score_values = np.array(list(scores.values()))
        exp_scores = np.exp(score_values / adjusted_temp)
        softmax_scores = exp_scores / (np.sum(exp_scores) + 1e-10)

        # 딕셔너리로 변환
        return {
            name: float(softmax_scores[i])
            for i, name in enumerate(scores.keys())
        }


class COBRAConsensus:
    """
    COBRA 합의 시스템

    상황에 따라 적절한 합의 알고리즘을 선택하여 적용
    """

    def __init__(
        self,
        default_algorithm: str = "drwa",
        trust_threshold: float = 0.7,
        epsilon: float = 0.2,
        temperature: float = 1.0
    ):
        """
        Args:
            default_algorithm: 기본 알고리즘 ("rot", "drwa", "avga")
            trust_threshold: RoT 신뢰 임계값
            epsilon: DRWA 조정 계수
            temperature: AVGA 온도
        """
        self.algorithms = {
            "rot": RootOfTrust(trust_threshold=trust_threshold),
            "drwa": DRWA(epsilon=epsilon),
            "avga": AVGA(temperature=temperature)
        }
        self.default_algorithm = default_algorithm

    def aggregate(
        self,
        responses: Dict[str, AgentResponse],
        trust_scores: Dict[str, float],
        algorithm: Optional[str] = None
    ) -> ConsensusResult:
        """
        합의 도출

        Args:
            responses: 에이전트 응답
            trust_scores: 신뢰도 점수
            algorithm: 사용할 알고리즘 (None이면 자동 선택)

        Returns:
            ConsensusResult: 합의 결과
        """
        # 알고리즘 선택
        if algorithm is None:
            algorithm = self._select_algorithm(responses, trust_scores)

        algo = self.algorithms.get(algorithm, self.algorithms[self.default_algorithm])

        return algo.aggregate(responses, trust_scores)

    def _select_algorithm(
        self,
        responses: Dict[str, AgentResponse],
        trust_scores: Dict[str, float]
    ) -> str:
        """
        상황에 따른 최적 알고리즘 선택

        - 신뢰도 편차가 큰 경우: RoT (신뢰 코호트 분리)
        - 판정 불일치가 심한 경우: AVGA (분산 기반 어텐션)
        - 일반적인 경우: DRWA (동적 가중치)
        """
        if not responses:
            return self.default_algorithm

        # 신뢰도 편차
        trusts = list(trust_scores.values())
        trust_variance = np.var(trusts) if trusts else 0

        # 판정 다양성
        verdicts = [r.verdict for r in responses.values()]
        unique_verdicts = len(set(verdicts))

        # 알고리즘 선택
        if trust_variance > 0.1:
            return "rot"  # 신뢰도 편차 큼 → 코호트 분리
        elif unique_verdicts >= 3:
            return "avga"  # 의견 분분 → 분산 기반
        else:
            return "drwa"  # 일반적 → 동적 가중치
