"""
43-dim 메타 특징 추출기

Per-agent features (20 dim):
    verdict one-hot  : 4 categories × 4 agents = 16 dim
    confidence       : 4 agents = 4 dim

Pairwise disagreement features (18 dim, 6 pairs):
    binary disagreement    : 같은 verdict인지 (0/1) = 6 dim
    confidence difference  : |c_i - c_j| = 6 dim
    conflict strength      : 불일치 시 |c_i - c_j|, 일치 시 0 = 6 dim

Aggregate features (5 dim):
    confidence variance  : var(c₁..c₄)
    verdict entropy      : -Σ p(v) log p(v)
    max-min gap          : max(c) - min(c)
    unique verdict count : |{v₁..v₄}|
    majority ratio       : 다수파 비율 (0.25~1.0)

합계: 20 + 18 + 5 = 43 dim
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from itertools import combinations
from dataclasses import dataclass

from .simulator import AGENT_NAMES, VERDICTS, TRUE_LABELS, SimulatedOutput


# 에이전트 쌍 (순서 고정, 6쌍)
AGENT_PAIRS: List[Tuple[str, str]] = list(combinations(AGENT_NAMES, 2))


@dataclass
class FeatureConfig:
    """특징 추출 설정"""
    include_verdict: bool = True      # verdict one-hot (16 dim)
    include_confidence: bool = True   # confidence (4 dim)
    include_disagreement: bool = True # pairwise features (18 dim)
    include_aggregate: bool = True    # aggregate features (5 dim)

    @property
    def dim(self) -> int:
        """특징 차원 수"""
        d = 0
        if self.include_verdict:
            d += len(AGENT_NAMES) * len(VERDICTS)  # 16
        if self.include_confidence:
            d += len(AGENT_NAMES)                    # 4
        if self.include_disagreement:
            d += len(AGENT_PAIRS) * 3                # 18
        if self.include_aggregate:
            d += 5
        return d

    @property
    def feature_names(self) -> List[str]:
        """특징 이름 목록"""
        names = []
        if self.include_verdict:
            for agent in AGENT_NAMES:
                for verdict in VERDICTS:
                    names.append(f"{agent}_verdict_{verdict}")
        if self.include_confidence:
            for agent in AGENT_NAMES:
                names.append(f"{agent}_confidence")
        if self.include_disagreement:
            for a1, a2 in AGENT_PAIRS:
                names.append(f"disagree_{a1}_{a2}")
            for a1, a2 in AGENT_PAIRS:
                names.append(f"conf_diff_{a1}_{a2}")
            for a1, a2 in AGENT_PAIRS:
                names.append(f"conflict_{a1}_{a2}")
        if self.include_aggregate:
            names.extend([
                "confidence_variance",
                "verdict_entropy",
                "max_min_gap",
                "unique_verdicts",
                "majority_ratio",
            ])
        return names


# 사전 정의된 ablation 설정 (DAAC_RESEARCH_PLAN.md Section 3.6)
ABLATION_CONFIGS = {
    "A1_confidence_only": FeatureConfig(
        include_verdict=False, include_confidence=True,
        include_disagreement=False, include_aggregate=False,
    ),
    "A2_verdict_only": FeatureConfig(
        include_verdict=True, include_confidence=False,
        include_disagreement=False, include_aggregate=False,
    ),
    "A3_disagreement_only": FeatureConfig(
        include_verdict=False, include_confidence=False,
        include_disagreement=True, include_aggregate=True,
    ),
    "A4_verdict_confidence": FeatureConfig(
        include_verdict=True, include_confidence=True,
        include_disagreement=False, include_aggregate=False,
    ),
    "A5_full": FeatureConfig(
        include_verdict=True, include_confidence=True,
        include_disagreement=True, include_aggregate=True,
    ),
}


class MetaFeatureExtractor:
    """
    43-dim 메타 특징 추출기

    Usage:
        extractor = MetaFeatureExtractor()
        X, y = extractor.extract_dataset(simulated_outputs)
        # X: (N, 43) ndarray
        # y: (N,) ndarray of label indices
    """

    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()

    @property
    def dim(self) -> int:
        return self.config.dim

    @property
    def feature_names(self) -> List[str]:
        return self.config.feature_names

    def extract_single(self, sample: SimulatedOutput) -> np.ndarray:
        """단일 샘플에서 메타 특징 추출"""
        features = []

        verdicts = sample.agent_verdicts
        confidences = sample.agent_confidences

        # --- Per-agent features ---
        if self.config.include_verdict:
            for agent in AGENT_NAMES:
                one_hot = np.zeros(len(VERDICTS), dtype=np.float32)
                v_idx = VERDICTS.index(verdicts[agent])
                one_hot[v_idx] = 1.0
                features.append(one_hot)

        if self.config.include_confidence:
            conf_vec = np.array(
                [confidences[agent] for agent in AGENT_NAMES],
                dtype=np.float32,
            )
            features.append(conf_vec)

        # --- Pairwise disagreement features ---
        if self.config.include_disagreement:
            binary_disagree = []
            conf_diff = []
            conflict_strength = []

            for a1, a2 in AGENT_PAIRS:
                v1, v2 = verdicts[a1], verdicts[a2]
                c1, c2 = confidences[a1], confidences[a2]

                disagree = 0.0 if v1 == v2 else 1.0
                binary_disagree.append(disagree)
                conf_diff.append(abs(c1 - c2))
                # conflict: 불일치 시 두 confidence의 합 (둘 다 확신하면서 다르면 높음)
                conflict_strength.append((c1 + c2) * disagree)

            features.append(np.array(binary_disagree, dtype=np.float32))
            features.append(np.array(conf_diff, dtype=np.float32))
            features.append(np.array(conflict_strength, dtype=np.float32))

        # --- Aggregate features ---
        if self.config.include_aggregate:
            conf_values = np.array(
                [confidences[a] for a in AGENT_NAMES], dtype=np.float32,
            )
            verdict_list = [verdicts[a] for a in AGENT_NAMES]

            agg = np.array([
                np.var(conf_values),                              # confidence 분산
                self._verdict_entropy(verdict_list),              # verdict 엔트로피
                float(np.max(conf_values) - np.min(conf_values)), # max-min gap
                float(len(set(verdict_list))),                    # 고유 verdict 수
                self._majority_ratio(verdict_list),               # majority 비율
            ], dtype=np.float32)
            features.append(agg)

        return np.concatenate(features)

    def extract_dataset(
        self,
        samples: List[SimulatedOutput],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        데이터셋 전체에서 메타 특징 + 레이블 추출

        Returns:
            X: (N, D) feature matrix
            y: (N,) label indices (0=authentic, 1=manipulated, 2=ai_generated)
        """
        X_list = []
        y_list = []

        for sample in samples:
            X_list.append(self.extract_single(sample))
            y_list.append(TRUE_LABELS.index(sample.true_label))

        X = np.stack(X_list, axis=0)
        y = np.array(y_list, dtype=np.int64)
        return X, y

    @staticmethod
    def _verdict_entropy(verdict_list: List[str]) -> float:
        """verdict 분포의 엔트로피"""
        from collections import Counter
        counts = Counter(verdict_list)
        n = len(verdict_list)
        if n == 0:
            return 0.0
        entropy = 0.0
        for count in counts.values():
            p = count / n
            if p > 0:
                entropy -= p * np.log2(p)
        return float(entropy)

    @staticmethod
    def _majority_ratio(verdict_list: List[str]) -> float:
        """다수파 비율 (0.25 ~ 1.0)"""
        from collections import Counter
        if not verdict_list:
            return 0.25
        counts = Counter(verdict_list)
        return counts.most_common(1)[0][1] / len(verdict_list)

    def extract_with_agent_removal(
        self,
        sample: SimulatedOutput,
        exclude_agent: str,
    ) -> np.ndarray:
        """
        특정 에이전트를 제외한 메타 특징 추출 (A6 ablation용)

        제외된 에이전트의 특징은 0으로 마스킹
        """
        features = self.extract_single(sample)

        # 제외할 에이전트의 인덱스
        agent_idx = AGENT_NAMES.index(exclude_agent)
        names = self.feature_names

        mask = np.ones(len(features), dtype=np.float32)
        for i, name in enumerate(names):
            if exclude_agent in name:
                mask[i] = 0.0

        return features * mask
