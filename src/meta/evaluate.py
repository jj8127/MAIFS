"""
평가 모듈

지표:
    Macro-F1             : 클래스 불균형 보정 주요 지표
    Balanced Accuracy    : 보조 지표
    AUROC (OvR)          : 확률 기반 평가
    ECE                  : Expected Calibration Error
    Brier Score          : calibration 보조
    Class-wise Confusion : 에이전트/조작유형별 분석

통계 검정:
    McNemar Test         : 분류 성능 쌍대 비교
    Bootstrap CI (95%)   : 1000회 리샘플링 신뢰 구간
"""
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, field

from sklearn.metrics import (
    f1_score,
    balanced_accuracy_score,
    roc_auc_score,
    brier_score_loss,
    confusion_matrix,
    classification_report,
)


@dataclass
class EvalResult:
    """평가 결과"""
    model_name: str
    macro_f1: float
    balanced_accuracy: float
    auroc: Optional[float] = None
    ece: Optional[float] = None
    brier: Optional[float] = None
    confusion: Optional[np.ndarray] = None
    class_report: Optional[str] = None
    per_class_f1: Dict[str, float] = field(default_factory=dict)


@dataclass
class ComparisonResult:
    """모델 간 비교 결과"""
    model_a: str
    model_b: str
    mcnemar_statistic: float
    mcnemar_pvalue: float
    f1_diff: float
    f1_ci_lower: float
    f1_ci_upper: float
    significant: bool  # p < 0.05


class MetaEvaluator:
    """
    메타 분류기 평가자

    Usage:
        evaluator = MetaEvaluator(label_names=["authentic", "manipulated", "ai_generated"])
        result = evaluator.evaluate(y_true, y_pred, y_proba, model_name="mlp")
        comparison = evaluator.compare(y_true, y_pred_a, y_pred_b, "cobra", "mlp")
    """

    def __init__(
        self,
        label_names: Optional[List[str]] = None,
        n_bootstrap: int = 1000,
        ci_level: float = 0.95,
        seed: int = 42,
    ):
        self.label_names = label_names or ["authentic", "manipulated", "ai_generated"]
        self.n_bootstrap = n_bootstrap
        self.ci_level = ci_level
        self.rng = np.random.default_rng(seed)

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        model_name: str = "unknown",
    ) -> EvalResult:
        """
        전체 평가 지표 계산

        Args:
            y_true: 실제 레이블 (N,)
            y_pred: 예측 레이블 (N,)
            y_proba: 예측 확률 (N, 3) — 확률 기반 지표에 필요
            model_name: 모델 이름

        Returns:
            EvalResult: 평가 결과
        """
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        bal_acc = balanced_accuracy_score(y_true, y_pred)

        # 클래스별 F1
        per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        per_class_f1 = {
            self.label_names[i]: float(per_class[i])
            for i in range(len(self.label_names))
        }

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(self.label_names))))

        # Classification report
        report = classification_report(
            y_true, y_pred,
            target_names=self.label_names,
            zero_division=0,
        )

        # 확률 기반 지표
        auroc = None
        ece = None
        brier = None

        if y_proba is not None:
            try:
                auroc = roc_auc_score(
                    y_true, y_proba,
                    multi_class="ovr",
                    average="macro",
                )
            except ValueError:
                pass

            ece = self._expected_calibration_error(y_true, y_proba)
            brier = self._multiclass_brier(y_true, y_proba)

        return EvalResult(
            model_name=model_name,
            macro_f1=macro_f1,
            balanced_accuracy=bal_acc,
            auroc=auroc,
            ece=ece,
            brier=brier,
            confusion=cm,
            class_report=report,
            per_class_f1=per_class_f1,
        )

    def compare(
        self,
        y_true: np.ndarray,
        y_pred_a: np.ndarray,
        y_pred_b: np.ndarray,
        model_a: str = "model_a",
        model_b: str = "model_b",
        y_proba_a: Optional[np.ndarray] = None,
        y_proba_b: Optional[np.ndarray] = None,
    ) -> ComparisonResult:
        """
        McNemar 검정 + Bootstrap CI로 두 모델 비교

        Args:
            y_true: 실제 레이블
            y_pred_a: 모델 A 예측
            y_pred_b: 모델 B 예측
            model_a, model_b: 모델 이름

        Returns:
            ComparisonResult: 비교 결과
        """
        # McNemar test
        statistic, pvalue = self._mcnemar_test(y_true, y_pred_a, y_pred_b)

        # Bootstrap CI for F1 difference
        f1_a = f1_score(y_true, y_pred_a, average="macro", zero_division=0)
        f1_b = f1_score(y_true, y_pred_b, average="macro", zero_division=0)
        f1_diff = f1_b - f1_a

        ci_lower, ci_upper = self._bootstrap_f1_diff_ci(
            y_true, y_pred_a, y_pred_b,
        )

        return ComparisonResult(
            model_a=model_a,
            model_b=model_b,
            mcnemar_statistic=statistic,
            mcnemar_pvalue=pvalue,
            f1_diff=f1_diff,
            f1_ci_lower=ci_lower,
            f1_ci_upper=ci_upper,
            significant=pvalue < 0.05,
        )

    def _mcnemar_test(
        self,
        y_true: np.ndarray,
        y_pred_a: np.ndarray,
        y_pred_b: np.ndarray,
    ) -> Tuple[float, float]:
        """
        McNemar 검정 (이진 → 정답/오답 변환)

        b: A 맞고 B 틀린 횟수
        c: A 틀리고 B 맞은 횟수
        statistic = (|b - c| - 1)² / (b + c)  [연속성 보정]
        """
        correct_a = (y_pred_a == y_true)
        correct_b = (y_pred_b == y_true)

        b = np.sum(correct_a & ~correct_b)   # A 맞고 B 틀림
        c = np.sum(~correct_a & correct_b)   # A 틀리고 B 맞음

        if b + c == 0:
            return 0.0, 1.0  # 차이 없음

        # 연속성 보정 McNemar
        statistic = (abs(b - c) - 1) ** 2 / (b + c)

        # chi-squared 분포 (df=1) p-value
        from scipy.stats import chi2
        pvalue = float(chi2.sf(statistic, df=1))

        return float(statistic), pvalue

    def _bootstrap_f1_diff_ci(
        self,
        y_true: np.ndarray,
        y_pred_a: np.ndarray,
        y_pred_b: np.ndarray,
    ) -> Tuple[float, float]:
        """Bootstrap 95% CI for Macro-F1 difference (B - A)"""
        n = len(y_true)
        diffs = []

        for _ in range(self.n_bootstrap):
            idx = self.rng.integers(0, n, size=n)
            f1_a = f1_score(y_true[idx], y_pred_a[idx], average="macro", zero_division=0)
            f1_b = f1_score(y_true[idx], y_pred_b[idx], average="macro", zero_division=0)
            diffs.append(f1_b - f1_a)

        diffs = np.array(diffs)
        alpha = 1 - self.ci_level
        lower = float(np.percentile(diffs, 100 * alpha / 2))
        upper = float(np.percentile(diffs, 100 * (1 - alpha / 2)))
        return lower, upper

    @staticmethod
    def _expected_calibration_error(
        y_true: np.ndarray,
        y_proba: np.ndarray,
        n_bins: int = 15,
    ) -> float:
        """
        Expected Calibration Error (ECE)

        각 bin에서 |accuracy - mean_confidence| 의 가중 평균
        """
        # 다중 클래스: 예측 확률의 최댓값과 예측 정확성 비교
        max_proba = np.max(y_proba, axis=1)
        preds = np.argmax(y_proba, axis=1)
        correct = (preds == y_true).astype(float)

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        total = len(y_true)

        for i in range(n_bins):
            mask = (max_proba > bin_boundaries[i]) & (max_proba <= bin_boundaries[i + 1])
            count = mask.sum()
            if count == 0:
                continue
            bin_acc = correct[mask].mean()
            bin_conf = max_proba[mask].mean()
            ece += (count / total) * abs(bin_acc - bin_conf)

        return float(ece)

    @staticmethod
    def _multiclass_brier(
        y_true: np.ndarray,
        y_proba: np.ndarray,
    ) -> float:
        """
        다중 클래스 Brier Score

        BS = (1/N) Σ Σ (p_{ik} - y_{ik})²
        """
        n_classes = y_proba.shape[1]
        n_samples = len(y_true)

        # One-hot 인코딩
        y_onehot = np.zeros((n_samples, n_classes))
        y_onehot[np.arange(n_samples), y_true] = 1.0

        brier = np.mean(np.sum((y_proba - y_onehot) ** 2, axis=1))
        return float(brier)

    def format_result(self, result: EvalResult) -> str:
        """평가 결과를 포맷팅"""
        lines = [
            f"=== {result.model_name} ===",
            f"  Macro-F1:          {result.macro_f1:.4f}",
            f"  Balanced Accuracy: {result.balanced_accuracy:.4f}",
        ]
        if result.auroc is not None:
            lines.append(f"  AUROC (OvR):       {result.auroc:.4f}")
        if result.ece is not None:
            lines.append(f"  ECE:               {result.ece:.4f}")
        if result.brier is not None:
            lines.append(f"  Brier Score:       {result.brier:.4f}")

        lines.append("  Per-class F1:")
        for name, f1 in result.per_class_f1.items():
            lines.append(f"    {name:15s}: {f1:.4f}")

        return "\n".join(lines)

    def format_comparison(self, comp: ComparisonResult) -> str:
        """비교 결과를 포맷팅"""
        sig = "***" if comp.significant else "(n.s.)"
        return (
            f"--- {comp.model_a} vs {comp.model_b} ---\n"
            f"  F1 diff ({comp.model_b} - {comp.model_a}): "
            f"{comp.f1_diff:+.4f}  [{comp.f1_ci_lower:+.4f}, {comp.f1_ci_upper:+.4f}]\n"
            f"  McNemar: χ²={comp.mcnemar_statistic:.2f}, "
            f"p={comp.mcnemar_pvalue:.4f} {sig}"
        )
