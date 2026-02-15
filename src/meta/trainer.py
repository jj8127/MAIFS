"""
메타 분류기 학습/추론

베이스라인:
    - Logistic Regression (선형 메타 분류기)
    - GradientBoosting / XGBoost (비선형 앙상블)
    - MLP (43 → 32 → 16 → 3)

GPU 사용 정책:
    - `mlp.backend: auto` + CUDA 가능 시 PyTorch MLP(CUDA) 사용
    - `gradient_boosting.backend: auto` + XGBoost 설치 시 GPU 트리 사용 시도
    - 그 외는 기존 scikit-learn CPU 경로로 폴백
"""
import os
import copy
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from sklearn.ensemble import GradientBoostingClassifier
except Exception:
    GradientBoostingClassifier = None

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except Exception:
    XGBClassifier = None
    HAS_XGBOOST = False

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except Exception:
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None
    TORCH_AVAILABLE = False


@dataclass
class TrainResult:
    """학습 결과"""
    model_name: str
    train_accuracy: float
    val_accuracy: float
    best_params: Dict[str, Any] = field(default_factory=dict)
    feature_dim: int = 0


if TORCH_AVAILABLE:

    class _TorchMLPNet(nn.Module):
        """소형 MLP 분류기 네트워크"""

        def __init__(
            self,
            input_dim: int,
            hidden_layer_sizes: Tuple[int, ...],
            output_dim: int,
            activation: str = "relu",
        ):
            super().__init__()
            act_map = {
                "relu": nn.ReLU,
                "tanh": nn.Tanh,
                "gelu": nn.GELU,
                "identity": nn.Identity,
                "logistic": nn.Sigmoid,
            }
            if activation not in act_map:
                raise ValueError(f"Unsupported activation for TorchMLP: {activation}")

            layers = []
            prev_dim = input_dim
            for h in hidden_layer_sizes:
                layers.append(nn.Linear(prev_dim, int(h)))
                layers.append(act_map[activation]())
                prev_dim = int(h)
            layers.append(nn.Linear(prev_dim, output_dim))
            self.net = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)


    class TorchMLPClassifier(BaseEstimator, ClassifierMixin):
        """
        sklearn 호환 PyTorch MLP 분류기

        NOTE:
            `Pipeline` 내부에서 동작하도록 `fit/predict/predict_proba` 인터페이스 유지.
        """

        def __init__(
            self,
            hidden_layer_sizes: Tuple[int, ...] = (32, 16),
            activation: str = "relu",
            max_iter: int = 500,
            learning_rate_init: float = 0.001,
            batch_size: int = 256,
            early_stopping: bool = True,
            validation_fraction: float = 0.15,
            random_state: int = 42,
            patience: int = 30,
            weight_decay: float = 0.0,
            verbose: bool = False,
            device: str = "cuda",
        ):
            self.hidden_layer_sizes = hidden_layer_sizes
            self.activation = activation
            self.max_iter = max_iter
            self.learning_rate_init = learning_rate_init
            self.batch_size = batch_size
            self.early_stopping = early_stopping
            self.validation_fraction = validation_fraction
            self.random_state = random_state
            self.patience = patience
            self.weight_decay = weight_decay
            self.verbose = verbose
            self.device = device

        def _resolve_device(self) -> str:
            requested = (self.device or "cpu").lower()
            if requested in {"cuda", "gpu"}:
                if TORCH_AVAILABLE and torch.cuda.is_available():
                    return "cuda"
                return "cpu"
            if requested == "auto":
                if TORCH_AVAILABLE and torch.cuda.is_available():
                    return "cuda"
                return "cpu"
            return "cpu"

        def fit(self, X: np.ndarray, y: np.ndarray):
            X = np.asarray(X, dtype=np.float32)
            y_raw = np.asarray(y)
            self.classes_, y_encoded = np.unique(y_raw, return_inverse=True)
            y_encoded = y_encoded.astype(np.int64)

            if X.ndim != 2:
                raise ValueError(f"X must be 2D, got shape={X.shape}")
            if len(X) != len(y_encoded):
                raise ValueError("X and y must have same length")

            # 재현성
            torch.manual_seed(int(self.random_state))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(self.random_state))

            use_val = (
                self.early_stopping
                and 0.0 < float(self.validation_fraction) < 1.0
                and len(X) >= 20
                and len(np.unique(y_encoded)) > 1
            )

            if use_val:
                X_train, X_val, y_train, y_val = train_test_split(
                    X,
                    y_encoded,
                    test_size=float(self.validation_fraction),
                    random_state=int(self.random_state),
                    stratify=y_encoded,
                )
            else:
                X_train, y_train = X, y_encoded
                X_val, y_val = None, None

            self.device_ = self._resolve_device()
            self.n_features_in_ = X.shape[1]
            n_classes = len(self.classes_)

            self.model_ = _TorchMLPNet(
                input_dim=self.n_features_in_,
                hidden_layer_sizes=tuple(int(v) for v in self.hidden_layer_sizes),
                output_dim=n_classes,
                activation=str(self.activation).lower(),
            ).to(self.device_)

            optimizer = torch.optim.Adam(
                self.model_.parameters(),
                lr=float(self.learning_rate_init),
                weight_decay=float(self.weight_decay),
            )
            loss_fn = nn.CrossEntropyLoss()

            ds = TensorDataset(
                torch.from_numpy(X_train),
                torch.from_numpy(y_train),
            )
            loader = DataLoader(
                ds,
                batch_size=max(1, min(int(self.batch_size), len(ds))),
                shuffle=True,
            )

            best_state = None
            best_val = float("inf")
            patience_left = int(self.patience)

            for epoch in range(int(self.max_iter)):
                self.model_.train()
                total_loss = 0.0
                for xb, yb in loader:
                    xb = xb.to(self.device_)
                    yb = yb.to(self.device_)

                    optimizer.zero_grad(set_to_none=True)
                    logits = self.model_(xb)
                    loss = loss_fn(logits, yb)
                    loss.backward()
                    optimizer.step()
                    total_loss += float(loss.item()) * len(xb)

                epoch_loss = total_loss / max(1, len(ds))

                monitor = epoch_loss
                if X_val is not None and y_val is not None:
                    self.model_.eval()
                    with torch.no_grad():
                        xv = torch.from_numpy(X_val).to(self.device_)
                        yv = torch.from_numpy(y_val).to(self.device_)
                        val_logits = self.model_(xv)
                        monitor = float(loss_fn(val_logits, yv).item())

                if monitor < best_val - 1e-6:
                    best_val = monitor
                    best_state = copy.deepcopy(self.model_.state_dict())
                    patience_left = int(self.patience)
                else:
                    patience_left -= 1

                if self.verbose and ((epoch + 1) % 20 == 0 or epoch == 0):
                    print(
                        f"[TorchMLP] epoch={epoch + 1}/{self.max_iter} "
                        f"loss={epoch_loss:.5f} monitor={monitor:.5f} device={self.device_}"
                    )

                if self.early_stopping and patience_left <= 0:
                    break

            if best_state is not None:
                self.model_.load_state_dict(best_state)
            self.model_.eval()
            return self

        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            if not hasattr(self, "model_"):
                raise ValueError("Model not fitted yet")
            X = np.asarray(X, dtype=np.float32)
            with torch.no_grad():
                xb = torch.from_numpy(X).to(self.device_)
                logits = self.model_(xb)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
            return probs

        def predict(self, X: np.ndarray) -> np.ndarray:
            probs = self.predict_proba(X)
            idx = np.argmax(probs, axis=1)
            return self.classes_[idx]

else:

    class TorchMLPClassifier(BaseEstimator, ClassifierMixin):
        """PyTorch 미설치 환경에서의 placeholder (import-time crash 방지)."""

        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def fit(self, X: np.ndarray, y: np.ndarray):
            raise RuntimeError("TorchMLPClassifier requires PyTorch (torch is not installed)")



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
        "backend": "auto",  # auto|sklearn|xgboost
        "n_estimators": 200,
        "max_depth": 4,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "random_state": 42,
    },
    "mlp": {
        "backend": "auto",  # auto|sklearn|torch
        "hidden_layer_sizes": (32, 16),
        "activation": "relu",
        "solver": "adam",  # sklearn 전용
        "max_iter": 500,
        "early_stopping": True,
        "validation_fraction": 0.15,
        "learning_rate_init": 0.001,
        "random_state": 42,
        # torch 전용
        "batch_size": 256,
        "patience": 30,
        "weight_decay": 0.0,
        "device": "auto",
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
        use_gpu: Optional[bool] = None,
    ):
        """
        Args:
            model_configs: 모델별 하이퍼파라미터 오버라이드
            scale_features: StandardScaler 적용 여부
            use_gpu: None이면 env(`MAIFS_META_USE_GPU`, 기본 1) 기준 자동
        """
        self.model_configs = copy.deepcopy(DEFAULT_CONFIGS)
        if model_configs:
            for name, cfg in model_configs.items():
                if name in self.model_configs:
                    self.model_configs[name].update(cfg)
                else:
                    self.model_configs[name] = cfg

        if use_gpu is None:
            env_flag = os.environ.get("MAIFS_META_USE_GPU", "1").strip().lower()
            use_gpu = env_flag in {"1", "true", "yes", "on"}

        self.scale_features = scale_features
        self.gpu_available = bool(TORCH_AVAILABLE and torch.cuda.is_available())
        self.use_gpu = bool(use_gpu) and self.gpu_available
        self.default_device = "cuda" if self.use_gpu else "cpu"

        self.models: Dict[str, Pipeline] = {}
        self.train_results: Dict[str, TrainResult] = {}
        self.runtime_info: Dict[str, Dict[str, Any]] = {}

    def _build_model(self, name: str) -> Pipeline:
        """모델 + 스케일러 파이프라인 생성"""
        cfg = dict(self.model_configs.get(name, {}))

        runtime = {"backend": "sklearn", "device": "cpu"}

        if name == "logistic_regression":
            cfg.pop("multi_class", None)  # scikit-learn 1.8+ 제거 예정 옵션 방어
            estimator = LogisticRegression(**cfg)

        elif name == "gradient_boosting":
            backend = str(cfg.pop("backend", "auto")).lower()

            use_xgb = backend in {"xgboost", "xgb"} or (backend == "auto" and HAS_XGBOOST)
            if use_xgb and HAS_XGBOOST:
                xgb_cfg = {
                    "n_estimators": int(cfg.get("n_estimators", 200)),
                    "max_depth": int(cfg.get("max_depth", 4)),
                    "learning_rate": float(cfg.get("learning_rate", 0.1)),
                    "subsample": float(cfg.get("subsample", 0.8)),
                    "random_state": int(cfg.get("random_state", 42)),
                    "objective": "multi:softprob",
                    "num_class": 3,
                    "eval_metric": "mlogloss",
                    "verbosity": 0,
                }
                if self.use_gpu:
                    # XGBoost 2.x 방식: hist + device=cuda
                    xgb_cfg.setdefault("tree_method", "hist")
                    xgb_cfg.setdefault("device", "cuda")
                    runtime = {"backend": "xgboost", "device": "cuda"}
                else:
                    xgb_cfg.setdefault("tree_method", "hist")
                    runtime = {"backend": "xgboost", "device": "cpu"}

                estimator = XGBClassifier(**xgb_cfg)
            else:
                if GradientBoostingClassifier is None:
                    raise RuntimeError("GradientBoostingClassifier is unavailable")
                estimator = GradientBoostingClassifier(**cfg)
                runtime = {"backend": "sklearn", "device": "cpu"}

        elif name == "mlp":
            backend = str(cfg.pop("backend", "auto")).lower()

            use_torch = backend == "torch" or (backend == "auto" and self.use_gpu and TORCH_AVAILABLE)
            if use_torch:
                if not TORCH_AVAILABLE:
                    raise RuntimeError("mlp backend=torch requested but torch is unavailable")

                device_cfg = str(cfg.pop("device", "auto")).lower()
                if device_cfg in {"cuda", "gpu"}:
                    device = "cuda" if self.gpu_available else "cpu"
                elif device_cfg == "cpu":
                    device = "cpu"
                else:
                    device = self.default_device

                torch_cfg = {
                    "hidden_layer_sizes": tuple(cfg.get("hidden_layer_sizes", (32, 16))),
                    "activation": str(cfg.get("activation", "relu")),
                    "max_iter": int(cfg.get("max_iter", 500)),
                    "learning_rate_init": float(cfg.get("learning_rate_init", 0.001)),
                    "batch_size": int(cfg.get("batch_size", 256)),
                    "early_stopping": bool(cfg.get("early_stopping", True)),
                    "validation_fraction": float(cfg.get("validation_fraction", 0.15)),
                    "random_state": int(cfg.get("random_state", 42)),
                    "patience": int(cfg.get("patience", 30)),
                    "weight_decay": float(cfg.get("weight_decay", 0.0)),
                    "verbose": bool(cfg.get("verbose", False)),
                    "device": device,
                }
                estimator = TorchMLPClassifier(**torch_cfg)
                runtime = {"backend": "torch", "device": device}
            else:
                # sklearn MLP에서 torch 전용 파라미터 제거
                cfg.pop("batch_size", None)
                cfg.pop("patience", None)
                cfg.pop("weight_decay", None)
                cfg.pop("device", None)
                cfg.pop("backend", None)
                estimator = MLPClassifier(**cfg)
                runtime = {"backend": "sklearn", "device": "cpu"}

        else:
            raise ValueError(f"Unknown model: {name}")

        self.runtime_info[name] = runtime
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
        try:
            pipeline.fit(X_train, y_train)
        except Exception as e:
            runtime = self.runtime_info.get(model_name, {})
            # GPU XGBoost 실패 시 CPU XGBoost/CPU sklearn으로 자동 폴백
            if (
                model_name == "gradient_boosting"
                and runtime.get("backend") == "xgboost"
                and runtime.get("device") == "cuda"
            ):
                fallback_cfg = dict(self.model_configs.get("gradient_boosting", {}))
                fallback_cfg["backend"] = "xgboost" if HAS_XGBOOST else "sklearn"
                self.model_configs["gradient_boosting"] = fallback_cfg

                fallback = self._build_model("gradient_boosting")
                # 강제 CPU
                classifier = fallback.named_steps["classifier"]
                if HAS_XGBOOST and isinstance(classifier, XGBClassifier):
                    classifier.set_params(tree_method="hist", device="cpu")
                    self.runtime_info["gradient_boosting"] = {
                        "backend": "xgboost",
                        "device": "cpu",
                        "note": "fallback_from_cuda",
                    }
                else:
                    self.runtime_info["gradient_boosting"] = {
                        "backend": "sklearn",
                        "device": "cpu",
                        "note": "fallback_from_cuda",
                    }

                fallback.fit(X_train, y_train)
                pipeline = fallback
            else:
                raise

        train_acc = float(pipeline.score(X_train, y_train))
        val_acc = 0.0
        if X_val is not None and y_val is not None:
            val_acc = float(pipeline.score(X_val, y_val))

        self.models[model_name] = pipeline

        best_params = dict(self.model_configs.get(model_name, {}))
        best_params["runtime_backend"] = self.runtime_info.get(model_name, {}).get("backend")
        best_params["runtime_device"] = self.runtime_info.get(model_name, {}).get("device")

        result = TrainResult(
            model_name=model_name,
            train_accuracy=train_acc,
            val_accuracy=val_acc,
            best_params=best_params,
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
        if hasattr(classifier, "coef_"):
            # LogisticRegression: 클래스별 계수 절댓값 평균
            return np.mean(np.abs(classifier.coef_), axis=0)
        return None
