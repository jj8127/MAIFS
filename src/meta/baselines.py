"""
베이스라인 합의 방법

1. MajorityVoteBaseline: 단순 다수결 (동률 시 UNCERTAIN → 최다 confidence로 결정)
2. COBRABaseline: 기존 COBRA 고정 가중 합의 래퍼
"""
import numpy as np
from collections import Counter
from typing import Dict, List, Optional

from .simulator import SimulatedOutput, AGENT_NAMES, TRUE_LABELS, VERDICTS


# MAIFS 기본 trust scores (configs/settings.py 기준)
DEFAULT_TRUST_SCORES = {
    "frequency": 0.85,
    "noise": 0.80,
    "fatformer": 0.85,
    "spatial": 0.85,
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

    MAIFS의 기존 COBRA 알고리즘을 시뮬레이션 데이터에 적용.
    고정 trust score × confidence 가중 투표.
    """

    def __init__(
        self,
        trust_scores: Optional[Dict[str, float]] = None,
        algorithm: str = "drwa",
    ):
        """
        Args:
            trust_scores: 에이전트별 고정 trust score
            algorithm: COBRA 알고리즘 ("rot", "drwa", "avga")
        """
        self.trust_scores = trust_scores or DEFAULT_TRUST_SCORES.copy()
        self.algorithm = algorithm

    def predict_single(self, sample: SimulatedOutput) -> int:
        """단일 샘플 예측 → label index"""
        # 판정별 가중 점수 계산
        verdict_scores: Dict[str, float] = {}

        for agent in AGENT_NAMES:
            verdict = sample.agent_verdicts[agent]
            confidence = sample.agent_confidences[agent]
            trust = self.trust_scores.get(agent, 0.5)

            weight = trust * confidence
            verdict_scores[verdict] = verdict_scores.get(verdict, 0.0) + weight

        # UNCERTAIN은 최종 출력에서 제외하고 남은 것 중 최대
        real_scores = {
            v: s for v, s in verdict_scores.items() if v in TRUE_LABELS
        }

        if not real_scores:
            # 모두 uncertain
            return 0  # 기본값: authentic

        winner = max(real_scores, key=real_scores.get)
        return TRUE_LABELS.index(winner)

    def predict(self, samples: List[SimulatedOutput]) -> np.ndarray:
        """데이터셋 예측"""
        return np.array([self.predict_single(s) for s in samples], dtype=np.int64)

    def predict_proba(self, samples: List[SimulatedOutput]) -> np.ndarray:
        """데이터셋 확률 예측 (N, 3)"""
        proba = []
        for sample in samples:
            verdict_scores = {}
            for agent in AGENT_NAMES:
                verdict = sample.agent_verdicts[agent]
                confidence = sample.agent_confidences[agent]
                trust = self.trust_scores.get(agent, 0.5)
                weight = trust * confidence
                verdict_scores[verdict] = verdict_scores.get(verdict, 0.0) + weight

            # TRUE_LABELS 순서로 점수 추출
            scores = np.array([
                verdict_scores.get(label, 0.0) for label in TRUE_LABELS
            ])
            total = scores.sum()
            if total > 0:
                scores /= total
            else:
                scores = np.ones(3) / 3
            proba.append(scores)

        return np.array(proba, dtype=np.float64)
