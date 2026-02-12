"""
메타 분류기 학습/추론

베이스라인:
    - Logistic Regression (선형 메타 분류기)
    - XGBoost / GradientBoosting (비선형 앙상블)
    - MLP (43 → 32 → 16 → 3, 소형 신경망)

모든 분류기는 scikit-learn 호환 인터페이스를 따른다.
"""
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, field

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

try:
    from sklearn.ensemble import GradientBoostingClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


@dataclass
class TrainResult:
    """학습 결과"""
    model_name: str
    train_accuracy: float
    val_accuracy: float
    best_params: Dict[str, Any] = field(default_factory=dict)
    feature_dim: int = 0


# 모델 기본 하이퍼파라미터
DEFAULT_CONFIGS = {
    "logistic_regression": {
        "C": 1.0,
        "max_iter": 1000,
        "solver": "lbfgs",
        "class_weight": "balanced",
        "random_state": 42,
    },
    "gradient_boosting": {
        "n_estimators": 200,
        "max_depth": 4,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "random_state": 42,
    },
    "mlp": {
        "hidden_layer_sizes": (32, 16),
        "activation": "relu",
        "solver": "adam",
        "max_iter": 500,
        "early_stopping": True,
        "validation_fraction": 0.15,
        "learning_rate_init": 0.001,
        "random_state": 42,
    },
}


class MetaTrainer:
    """
    메타 분류기 학습/추론 관리자

    Usage:
        trainer = MetaTrainer()
        results = trainer.train_all(X_train, y_train, X_val, y_val)
        preds = trainer.predict("mlp", X_test)
        proba = trainer.predict_proba("mlp", X_test)
    """

    def __init__(
        self,
        model_configs: Optional[Dict[str, Dict[str, Any]]] = None,
        scale_features: bool = True,
    ):
        """
        Args:
            model_configs: 모델별 하이퍼파라미터 오버라이드
            scale_features: StandardScaler 적용 여부
        """
        self.model_configs = {**DEFAULT_CONFIGS}
        if model_configs:
            for name, cfg in model_configs.items():
                if name in self.model_configs:
                    self.model_configs[name].update(cfg)
                else:
                    self.model_configs[name] = cfg

        self.scale_features = scale_features
        self.models: Dict[str, Pipeline] = {}
        self.train_results: Dict[str, TrainResult] = {}

    def _build_model(self, name: str) -> Pipeline:
        """모델 + 스케일러 파이프라인 생성"""
        cfg = dict(self.model_configs.get(name, {}))

        if name == "logistic_regression":
            # scikit-learn 1.8+: multi_class 제거됨
            cfg.pop("multi_class", None)
            estimator = LogisticRegression(**cfg)
        elif name == "gradient_boosting":
            estimator = GradientBoostingClassifier(**cfg)
        elif name == "mlp":
            estimator = MLPClassifier(**cfg)
        else:
            raise ValueError(f"Unknown model: {name}")

        steps = []
        if self.scale_features:
            steps.append(("scaler", StandardScaler()))
        steps.append(("classifier", estimator))
        return Pipeline(steps)

    def train(
        self,
        model_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> TrainResult:
        """
        단일 모델 학습

        Args:
            model_name: 모델 이름
            X_train: 학습 특징 (N, D)
            y_train: 학습 레이블 (N,)
            X_val: 검증 특징
            y_val: 검증 레이블

        Returns:
            TrainResult: 학습 결과
        """
        pipeline = self._build_model(model_name)
        pipeline.fit(X_train, y_train)

        train_acc = float(pipeline.score(X_train, y_train))
        val_acc = 0.0
        if X_val is not None and y_val is not None:
            val_acc = float(pipeline.score(X_val, y_val))

        self.models[model_name] = pipeline

        result = TrainResult(
            model_name=model_name,
            train_accuracy=train_acc,
            val_accuracy=val_acc,
            best_params=self.model_configs.get(model_name, {}),
            feature_dim=X_train.shape[1],
        )
        self.train_results[model_name] = result
        return result

    def train_all(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict[str, TrainResult]:
        """모든 모델 학습"""
        results = {}
        for name in ["logistic_regression", "gradient_boosting", "mlp"]:
            results[name] = self.train(name, X_train, y_train, X_val, y_val)
        return results

    def predict(self, model_name: str, X: np.ndarray) -> np.ndarray:
        """예측"""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not trained yet")
        return self.models[model_name].predict(X)

    def predict_proba(self, model_name: str, X: np.ndarray) -> np.ndarray:
        """확률 예측 (N, 3)"""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not trained yet")
        return self.models[model_name].predict_proba(X)

    def get_feature_importance(self, model_name: str) -> Optional[np.ndarray]:
        """특징 중요도 반환 (지원 모델만)"""
        if model_name not in self.models:
            return None

        pipeline = self.models[model_name]
        classifier = pipeline.named_steps["classifier"]

        if hasattr(classifier, "feature_importances_"):
            return classifier.feature_importances_
        elif hasattr(classifier, "coef_"):
            # LogisticRegression: 클래스별 계수의 절댓값 평균
            return np.mean(np.abs(classifier.coef_), axis=0)
        return None
