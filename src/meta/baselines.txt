"""
베이스라인 합의 방법

1. MajorityVoteBaseline: 단순 다수결 (동률 시 UNCERTAIN → 최다 confidence로 결정)
2. COBRABaseline: 런타임 COBRAConsensus(RoT/DRWA/AVGA) 재사용 래퍼
"""
import numpy as np
from collections import Counter
from typing import Dict, List, Optional

from .simulator import SimulatedOutput, AGENT_NAMES, TRUE_LABELS
from configs.trust import DEFAULT_TRUST as _DEFAULT_TRUST
from configs.trust import resolve_trust
from ..tools.base_tool import Verdict
from ..agents.base_agent import AgentResponse, AgentRole
from ..consensus.cobra import COBRAConsensus

# MAIFS 기본 trust scores — configs/trust.py SSOT에서 가져옴
DEFAULT_TRUST_SCORES = dict(_DEFAULT_TRUST)

_AGENT_ROLE_MAP = {
    "frequency": AgentRole.FREQUENCY,
    "noise": AgentRole.NOISE,
    "fatformer": AgentRole.FATFORMER,
    "spatial": AgentRole.SPATIAL,
}


class MajorityVoteBaseline:
    """
    단순 다수결 베이스라인

    4개 에이전트의 verdict를 다수결로 결정.
    동률 시 confidence가 높은 에이전트의 verdict를 채택.
    """

    def predict_single(self, sample: SimulatedOutput) -> int:
        """단일 샘플 예측 → label index"""
        verdicts = list(sample.agent_verdicts.values())
        confidences = sample.agent_confidences

        # uncertain 을 제외한 투표
        real_verdicts = [v for v in verdicts if v != "uncertain"]

        if not real_verdicts:
            # 모두 uncertain → 가장 높은 confidence 에이전트의 verdict
            best_agent = max(AGENT_NAMES, key=lambda a: confidences[a])
            best_verdict = sample.agent_verdicts[best_agent]
            if best_verdict == "uncertain":
                return 0  # 기본값: authentic
            return TRUE_LABELS.index(best_verdict) if best_verdict in TRUE_LABELS else 0

        counts = Counter(real_verdicts)
        max_count = counts.most_common(1)[0][1]
        tied = [v for v, c in counts.items() if c == max_count]

        if len(tied) == 1:
            winner = tied[0]
        else:
            # 동률: confidence 합이 높은 verdict
            verdict_conf_sum = {}
            for agent in AGENT_NAMES:
                v = sample.agent_verdicts[agent]
                if v in tied:
                    verdict_conf_sum[v] = verdict_conf_sum.get(v, 0) + confidences[agent]
            winner = max(verdict_conf_sum, key=verdict_conf_sum.get)

        return TRUE_LABELS.index(winner) if winner in TRUE_LABELS else 0

    def predict(self, samples: List[SimulatedOutput]) -> np.ndarray:
        """데이터셋 예측"""
        return np.array([self.predict_single(s) for s in samples], dtype=np.int64)


class COBRABaseline:
    """
    COBRA 가중 합의 베이스라인 (래퍼)

    MAIFS 런타임과 동일한 COBRAConsensus(RoT/DRWA/AVGA)를
    시뮬레이션 데이터에 적용한다.
    """

    def __init__(
        self,
        trust_scores: Optional[Dict[str, float]] = None,
        algorithm: str = "drwa",
    ):
        """
        Args:
            trust_scores: 에이전트별 고정 trust score
            algorithm: COBRA 알고리즘 ("rot", "drwa", "avga", "auto")
        """
        # YAML override가 있으면 trust.py의 키 검증/마이그레이션 정책을 적용한다.
        self.trust_scores = resolve_trust(trust_scores)
        self.algorithm = algorithm
        self.consensus_engine = COBRAConsensus(default_algorithm="drwa")

    def _to_agent_response(self, agent: str, sample: SimulatedOutput) -> AgentResponse:
        """SimulatedOutput의 agent 출력 -> AgentResponse 변환."""
        verdict_str = str(sample.agent_verdicts.get(agent, "uncertain")).lower()
        confidence = float(sample.agent_confidences.get(agent, 0.5))

        try:
            verdict = Verdict(verdict_str)
        except ValueError:
            verdict = Verdict.UNCERTAIN

        # DRWA 분산 휴리스틱용 최소 evidence 제공
        ai_score = confidence if verdict == Verdict.AI_GENERATED else (1.0 - confidence)
        evidence = {
            "ai_generation_score": max(0.0, min(1.0, ai_score)),
            "source": "simulated_baseline",
        }

        return AgentResponse(
            agent_name=agent,
            role=_AGENT_ROLE_MAP[agent],
            verdict=verdict,
            confidence=max(0.0, min(1.0, confidence)),
            reasoning="",
            evidence=evidence,
            arguments=[],
        )

    def _aggregate_sample(self, sample: SimulatedOutput):
        responses = {
            agent: self._to_agent_response(agent, sample)
            for agent in AGENT_NAMES
        }
        selected_algorithm = None if self.algorithm == "auto" else self.algorithm
        return self.consensus_engine.aggregate(
            responses=responses,
            trust_scores=self.trust_scores,
            algorithm=selected_algorithm,
        )

    def predict_single(self, sample: SimulatedOutput) -> int:
        """단일 샘플 예측 → label index"""
        result = self._aggregate_sample(sample)
        verdict = result.final_verdict.value
        if verdict not in TRUE_LABELS:
            return 0
        return TRUE_LABELS.index(verdict)

    def predict(self, samples: List[SimulatedOutput]) -> np.ndarray:
        """데이터셋 예측"""
        return np.array([self.predict_single(s) for s in samples], dtype=np.int64)

    def predict_proba(self, samples: List[SimulatedOutput]) -> np.ndarray:
        """데이터셋 확률 예측 (N, 3)"""
        proba = []
        for sample in samples:
            result = self._aggregate_sample(sample)
            scores = np.array([
                float(result.verdict_scores.get(label, 0.0)) for label in TRUE_LABELS
            ])
            total = scores.sum()
            if total > 0:
                scores /= total
            else:
                scores = np.ones(3) / 3
            proba.append(scores)

        return np.array(proba, dtype=np.float64)


class WeightedMajorityVoteBaseline:
    """
    Trust score × confidence 가중 다수결 베이스라인

    각 레이블에 대해 (trust_score × confidence) 합산 후 argmax.
    """

    def __init__(self, trust_scores: Optional[Dict[str, float]] = None):
        self.trust_scores = resolve_trust(trust_scores)

    def predict_single(self, sample: SimulatedOutput) -> int:
        label_scores: Dict[str, float] = {label: 0.0 for label in TRUE_LABELS}
        for agent in AGENT_NAMES:
            verdict = sample.agent_verdicts[agent]
            if verdict not in TRUE_LABELS:
                continue
            label_scores[verdict] += (
                self.trust_scores.get(agent, 1.0) * float(sample.agent_confidences[agent])
            )
        best = max(label_scores, key=label_scores.get)  # type: ignore[arg-type]
        return TRUE_LABELS.index(best)

    def predict(self, samples: List[SimulatedOutput]) -> np.ndarray:
        return np.array([self.predict_single(s) for s in samples], dtype=np.int64)
