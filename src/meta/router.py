"""
Phase 2: Adaptive Routing (Path B: 시뮬레이션용 구현)

목표:
    - "이미지 특성"에 해당하는 경량 feature vector로부터
      에이전트 가중치 [w_freq, w_noise, w_fat, w_spatial] 를 예측한다.
    - Phase 1의 43-dim 메타 특징에 router weights(4-dim)를 추가하여
      (43+4=47) 메타 분류기의 입력으로 사용한다.

NOTE:
    현재 레포의 Path B는 실제 이미지가 없으므로, SyntheticImageEncoder로
    (true_label/sub_type) 기반의 합성 "image feature"를 생성해 Phase 2를 실험한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .simulator import AGENT_NAMES, VERDICT_TO_IDX, AgentSimulator, SimulatedOutput


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(x)
    denom = np.sum(exp, axis=axis, keepdims=True)
    denom = np.where(denom <= 0.0, 1.0, denom)
    return exp / denom


@dataclass
class SyntheticImageEncoderConfig:
    feature_dim: int = 32
    prototype_scale: float = 3.0
    noise_std: float = 1.0
    seed: int = 42


class SyntheticImageEncoder:
    """
    Path B에서 Phase 2를 재현하기 위한 합성 "이미지 인코더".

    - key = sample.sub_type (없으면 sample.true_label)
    - key별로 prototype(mean) 벡터를 하나 만들고,
      prototype + N(0, noise_std) 로 feature를 생성한다.
    """

    def __init__(self, config: Optional[SyntheticImageEncoderConfig] = None):
        self.config = config or SyntheticImageEncoderConfig()
        if int(self.config.feature_dim) <= 0:
            raise ValueError("feature_dim must be > 0")
        self.rng = np.random.default_rng(int(self.config.seed))
        self._prototypes: Dict[str, np.ndarray] = {}

    @staticmethod
    def _key(sample: SimulatedOutput) -> str:
        return str(sample.sub_type if sample.sub_type is not None else sample.true_label)

    def _get_prototype(self, key: str) -> np.ndarray:
        if key not in self._prototypes:
            # prototype_scale로 군집 간 분리를 크게 만들어 router가 학습 가능하도록 한다.
            mean = self.rng.standard_normal(int(self.config.feature_dim)).astype(np.float32)
            mean *= float(self.config.prototype_scale)
            self._prototypes[key] = mean
        return self._prototypes[key]

    def encode_single(self, sample: SimulatedOutput) -> np.ndarray:
        key = self._key(sample)
        proto = self._get_prototype(key)
        noise = self.rng.standard_normal(int(self.config.feature_dim)).astype(np.float32)
        return proto + noise * float(self.config.noise_std)

    def encode_dataset(self, samples: List[SimulatedOutput]) -> np.ndarray:
        feats = [self.encode_single(s) for s in samples]
        return np.stack(feats, axis=0).astype(np.float32)


@dataclass
class OracleWeightConfig:
    """Oracle weight 생성 규칙 (시뮬레이터 confusion row 기반)."""

    power: float = 2.0
    eps: float = 1e-6
    label_smoothing: float = 0.0
    confidence_power: float = 0.0
    uncertain_penalty: float = 1.0
    entropy_power: float = 0.0
    verdict_power: float = 0.0


class OracleWeightComputer:
    """
    시뮬레이터 프로파일(confusion matrix)에서
    샘플 단위로 에이전트 신뢰도를 oracle로 계산한다.
    """

    def __init__(self, simulator: AgentSimulator, config: Optional[OracleWeightConfig] = None):
        self.simulator = simulator
        self.config = config or OracleWeightConfig()

    @staticmethod
    def _row_entropy(row: np.ndarray, eps: float = 1e-12) -> float:
        """정규화 엔트로피 [0, 1] 계산."""
        p = np.asarray(row, dtype=np.float64)
        p = np.clip(p, eps, 1.0)
        p /= float(p.sum())
        h = -float(np.sum(p * np.log(p)))
        h_max = float(np.log(len(p)))
        if h_max <= 0.0:
            return 0.0
        return h / h_max

    def compute_single(self, sample: SimulatedOutput) -> np.ndarray:
        # 각 에이전트의 "정답 확률"에 신뢰도/불확실성/엔트로피 보정을 반영.
        scores = []
        conf_power = float(self.config.confidence_power)
        uncertain_penalty = float(np.clip(float(self.config.uncertain_penalty), 0.0, 1.0))
        entropy_power = float(self.config.entropy_power)
        verdict_power = float(self.config.verdict_power)
        true_idx = VERDICT_TO_IDX[sample.true_label]

        for agent in AGENT_NAMES:
            profile = self.simulator.profiles[agent]
            cm_row = self.simulator._get_cm_row(profile, sample.true_label, sample.sub_type)  # noqa: SLF001
            p_correct = max(float(cm_row[true_idx]), float(self.config.eps))

            # 기존 동작과 호환: 기본 score는 (p_correct ^ power)
            score = p_correct ** float(self.config.power)

            if entropy_power > 0.0:
                # confusion row 엔트로피가 높을수록(혼란할수록) 가중치 감쇠
                norm_entropy = self._row_entropy(cm_row, eps=max(self.config.eps, 1e-12))
                entropy_factor = max(1.0 - norm_entropy, float(self.config.eps))
                score *= entropy_factor ** entropy_power

            verdict = sample.agent_verdicts.get(agent, "uncertain")
            v_idx = VERDICT_TO_IDX.get(verdict, VERDICT_TO_IDX["uncertain"])
            if verdict_power > 0.0:
                p_verdict = max(float(cm_row[v_idx]), float(self.config.eps))
                score *= p_verdict ** verdict_power

            if conf_power > 0.0:
                conf = float(np.clip(sample.agent_confidences.get(agent, 0.5), self.config.eps, 1.0))
                score *= conf ** conf_power

            if verdict == "uncertain":
                score *= uncertain_penalty

            scores.append(max(score, float(self.config.eps)))

        w = np.array(scores, dtype=np.float64)
        s = float(w.sum())
        if s <= 0.0:
            return (np.ones(len(AGENT_NAMES), dtype=np.float64) / len(AGENT_NAMES))
        w = (w / s).astype(np.float64)

        # 라벨 스무딩으로 지나치게 뾰족한 oracle target을 완화한다.
        smooth = float(np.clip(float(self.config.label_smoothing), 0.0, 1.0))
        if smooth > 0.0:
            uniform = 1.0 / len(AGENT_NAMES)
            w = (1.0 - smooth) * w + smooth * uniform
            w = w / max(float(w.sum()), float(self.config.eps))

        return w

    def compute_dataset(self, samples: List[SimulatedOutput]) -> np.ndarray:
        w = [self.compute_single(s) for s in samples]
        return np.stack(w, axis=0).astype(np.float64)


@dataclass
class MetaRouterConfig:
    """
    scikit-learn 기반 Meta-Router 설정.

    Router는 oracle weight의 log를 예측(logits)하고, softmax로 weight를 만든다.
    """

    hidden_layer_sizes: Tuple[int, ...] = (64, 64)
    activation: str = "relu"
    solver: str = "adam"
    alpha: float = 1e-4
    learning_rate_init: float = 1e-3
    max_iter: int = 500
    early_stopping: bool = True
    validation_fraction: float = 0.15
    n_iter_no_change: int = 20
    random_state: int = 42
    regressor: str = "mlp"  # mlp | ridge | gbrt
    ridge_alpha: float = 1.0
    gbr_n_estimators: int = 200
    gbr_learning_rate: float = 0.05
    gbr_max_depth: int = 3
    gbr_subsample: float = 1.0


@dataclass
class RouterMetrics:
    mse_weights: float
    kl_weights: float
    top1_match: float


class MetaRouter:
    """
    scikit-learn MLPRegressor 기반 Meta-Router.
    """

    def __init__(self, config: Optional[MetaRouterConfig] = None, eps: float = 1e-6):
        self.config = config or MetaRouterConfig()
        self.eps = float(eps)
        self.model: Optional[Pipeline] = None

    @staticmethod
    def weights_to_logits(weights: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        w = np.asarray(weights, dtype=np.float64)
        w = np.clip(w, eps, 1.0)
        # weights는 row-wise sum=1이라고 가정. softmax(log(w)) == w
        return np.log(w)

    def fit(self, X_img: np.ndarray, oracle_weights: np.ndarray) -> "MetaRouter":
        X_img = np.asarray(X_img, dtype=np.float32)
        w = np.asarray(oracle_weights, dtype=np.float64)
        if X_img.ndim != 2:
            raise ValueError(f"X_img must be 2D, got shape={X_img.shape}")
        if w.ndim != 2 or w.shape[1] != len(AGENT_NAMES):
            raise ValueError(f"oracle_weights must be (N, {len(AGENT_NAMES)}), got shape={w.shape}")
        if len(X_img) != len(w):
            raise ValueError("X_img and oracle_weights must have same length")

        y_logits = self.weights_to_logits(w, eps=max(self.eps, 1e-12))

        regressor_name = str(self.config.regressor).lower()
        if regressor_name == "mlp":
            regressor = MLPRegressor(
                hidden_layer_sizes=tuple(int(v) for v in self.config.hidden_layer_sizes),
                activation=str(self.config.activation),
                solver=str(self.config.solver),
                alpha=float(self.config.alpha),
                learning_rate_init=float(self.config.learning_rate_init),
                max_iter=int(self.config.max_iter),
                early_stopping=bool(self.config.early_stopping),
                validation_fraction=float(self.config.validation_fraction),
                n_iter_no_change=int(self.config.n_iter_no_change),
                random_state=int(self.config.random_state),
            )
        elif regressor_name == "ridge":
            regressor = Ridge(alpha=float(self.config.ridge_alpha))
        elif regressor_name == "gbrt":
            base = GradientBoostingRegressor(
                n_estimators=int(self.config.gbr_n_estimators),
                learning_rate=float(self.config.gbr_learning_rate),
                max_depth=int(self.config.gbr_max_depth),
                subsample=float(self.config.gbr_subsample),
                random_state=int(self.config.random_state),
            )
            regressor = MultiOutputRegressor(base)
        else:
            raise ValueError(f"unsupported router regressor: {self.config.regressor}")

        self.model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("regressor", regressor),
            ]
        )
        self.model.fit(X_img, y_logits)
        return self

    def predict_logits(self, X_img: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("MetaRouter not fitted yet")
        X_img = np.asarray(X_img, dtype=np.float32)
        logits = self.model.predict(X_img)
        return np.asarray(logits, dtype=np.float64)

    def predict_weights(self, X_img: np.ndarray) -> np.ndarray:
        logits = self.predict_logits(X_img)
        return softmax(logits, axis=1)

    def evaluate(self, X_img: np.ndarray, oracle_weights: np.ndarray) -> RouterMetrics:
        w_true = np.asarray(oracle_weights, dtype=np.float64)
        w_pred = self.predict_weights(X_img)

        mse = float(np.mean((w_pred - w_true) ** 2))

        # KL(true || pred)
        w_t = np.clip(w_true, self.eps, 1.0)
        w_p = np.clip(w_pred, self.eps, 1.0)
        kl = float(np.mean(np.sum(w_t * (np.log(w_t) - np.log(w_p)), axis=1)))

        top1 = float(np.mean(np.argmax(w_pred, axis=1) == np.argmax(w_true, axis=1)))

        return RouterMetrics(mse_weights=mse, kl_weights=kl, top1_match=top1)
