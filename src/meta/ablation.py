"""
Ablation 실행기

DAAC_RESEARCH_PLAN.md Section 3.6에 정의된 6가지 ablation:

    A1: confidence만 (4 dim) — confidence 단독 정보량
    A2: verdict만 (16 dim) — verdict 패턴 정보량
    A3: disagreement만 (23 dim) — **핵심: 불일치 단독 신호**
    A4: verdict + confidence (20 dim) — 기존 앙상블 수준
    A5: Full (V+C+D) (43 dim) — DAAC 완전체
    A6: 에이전트 제거 (−1) (각 ~33 dim) — 각 에이전트의 기여도
"""
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

from .simulator import SimulatedOutput, AGENT_NAMES
from .features import MetaFeatureExtractor, FeatureConfig, ABLATION_CONFIGS
from .trainer import MetaTrainer
from .evaluate import MetaEvaluator, EvalResult, ComparisonResult


@dataclass
class AblationResult:
    """단일 ablation 결과"""
    ablation_id: str
    feature_dim: int
    model_results: Dict[str, EvalResult] = field(default_factory=dict)
    best_model: str = ""
    best_f1: float = 0.0


@dataclass
class AblationSummary:
    """전체 ablation 요약"""
    results: Dict[str, AblationResult] = field(default_factory=dict)
    comparisons: List[ComparisonResult] = field(default_factory=list)
    go_no_go: Dict[str, Any] = field(default_factory=dict)


class AblationRunner:
    """
    Ablation 실행기

    Usage:
        runner = AblationRunner()
        summary = runner.run(train_data, val_data, test_data)
        runner.print_summary(summary)
    """

    def __init__(
        self,
        models: Optional[List[str]] = None,
        evaluator: Optional[MetaEvaluator] = None,
    ):
        self.model_names = models or [
            "logistic_regression",
            "gradient_boosting",
            "mlp",
        ]
        self.evaluator = evaluator or MetaEvaluator()

    def run(
        self,
        train_data: List[SimulatedOutput],
        val_data: List[SimulatedOutput],
        test_data: List[SimulatedOutput],
        run_a6: bool = True,
    ) -> AblationSummary:
        """
        전체 ablation 실행

        Args:
            train_data: 학습 데이터
            val_data: 검증 데이터
            test_data: 테스트 데이터
            run_a6: A6 (에이전트 제거) 실행 여부

        Returns:
            AblationSummary: 전체 ablation 요약
        """
        summary = AblationSummary()

        # A1 ~ A5
        for ablation_id, config in ABLATION_CONFIGS.items():
            result = self._run_single_ablation(
                ablation_id, config, train_data, val_data, test_data,
            )
            summary.results[ablation_id] = result

        # A6: 에이전트 제거
        if run_a6:
            for agent_name in AGENT_NAMES:
                ablation_id = f"A6_remove_{agent_name}"
                result = self._run_agent_removal(
                    ablation_id, agent_name, train_data, val_data, test_data,
                )
                summary.results[ablation_id] = result

        # 핵심 비교: A5 (Full) vs COBRA 베이스라인
        # (COBRA 비교는 run_phase1에서 수행, 여기서는 ablation 간 비교)
        self._add_comparisons(summary, test_data)

        # Go/No-Go 평가
        summary.go_no_go = self._evaluate_go_no_go(summary)

        return summary

    def _run_single_ablation(
        self,
        ablation_id: str,
        config: FeatureConfig,
        train_data: List[SimulatedOutput],
        val_data: List[SimulatedOutput],
        test_data: List[SimulatedOutput],
    ) -> AblationResult:
        """단일 feature set으로 ablation 실행"""
        extractor = MetaFeatureExtractor(config)

        X_train, y_train = extractor.extract_dataset(train_data)
        X_val, y_val = extractor.extract_dataset(val_data)
        X_test, y_test = extractor.extract_dataset(test_data)

        result = AblationResult(
            ablation_id=ablation_id,
            feature_dim=config.dim,
        )

        trainer = MetaTrainer()
        best_f1 = 0.0
        best_model = ""

        for model_name in self.model_names:
            trainer.train(model_name, X_train, y_train, X_val, y_val)
            y_pred = trainer.predict(model_name, X_test)
            y_proba = trainer.predict_proba(model_name, X_test)

            eval_result = self.evaluator.evaluate(
                y_test, y_pred, y_proba, model_name=model_name,
            )
            result.model_results[model_name] = eval_result

            if eval_result.macro_f1 > best_f1:
                best_f1 = eval_result.macro_f1
                best_model = model_name

        result.best_model = best_model
        result.best_f1 = best_f1
        return result

    def _run_agent_removal(
        self,
        ablation_id: str,
        exclude_agent: str,
        train_data: List[SimulatedOutput],
        val_data: List[SimulatedOutput],
        test_data: List[SimulatedOutput],
    ) -> AblationResult:
        """에이전트 제거 ablation (A6)"""
        extractor = MetaFeatureExtractor()  # Full config

        # 제외된 에이전트의 특징을 0으로 마스킹
        X_train = np.array([
            extractor.extract_with_agent_removal(s, exclude_agent) for s in train_data
        ])
        y_train = np.array([
            ["authentic", "manipulated", "ai_generated"].index(s.true_label)
            for s in train_data
        ], dtype=np.int64)

        X_val = np.array([
            extractor.extract_with_agent_removal(s, exclude_agent) for s in val_data
        ])
        y_val = np.array([
            ["authentic", "manipulated", "ai_generated"].index(s.true_label)
            for s in val_data
        ], dtype=np.int64)

        X_test = np.array([
            extractor.extract_with_agent_removal(s, exclude_agent) for s in test_data
        ])
        y_test = np.array([
            ["authentic", "manipulated", "ai_generated"].index(s.true_label)
            for s in test_data
        ], dtype=np.int64)

        result = AblationResult(
            ablation_id=ablation_id,
            feature_dim=extractor.dim,
        )

        trainer = MetaTrainer()
        best_f1 = 0.0
        best_model = ""

        for model_name in self.model_names:
            trainer.train(model_name, X_train, y_train, X_val, y_val)
            y_pred = trainer.predict(model_name, X_test)
            y_proba = trainer.predict_proba(model_name, X_test)

            eval_result = self.evaluator.evaluate(
                y_test, y_pred, y_proba, model_name=model_name,
            )
            result.model_results[model_name] = eval_result

            if eval_result.macro_f1 > best_f1:
                best_f1 = eval_result.macro_f1
                best_model = model_name

        result.best_model = best_model
        result.best_f1 = best_f1
        return result

    def _add_comparisons(
        self,
        summary: AblationSummary,
        test_data: List[SimulatedOutput],
    ) -> None:
        """핵심 ablation 간 비교 추가"""
        # A3 (disagreement only) vs random baseline
        # A5 (full) vs A4 (verdict+confidence)
        pairs = [
            ("A3_disagreement_only", "A5_full"),
            ("A4_verdict_confidence", "A5_full"),
        ]

        for id_a, id_b in pairs:
            if id_a not in summary.results or id_b not in summary.results:
                continue

            res_a = summary.results[id_a]
            res_b = summary.results[id_b]

            # best 모델 기준으로 비교 (GradientBoosting 우선, 없으면 best)
            model = "gradient_boosting"
            if model not in res_a.model_results:
                model = res_a.best_model

            # 비교를 위해 해당 모델의 예측을 다시 생성해야 하므로
            # 여기서는 F1 차이만 기록
            if model in res_a.model_results and model in res_b.model_results:
                eval_a = res_a.model_results[model]
                eval_b = res_b.model_results[model]
                comp = ComparisonResult(
                    model_a=f"{id_a}({model})",
                    model_b=f"{id_b}({model})",
                    mcnemar_statistic=0.0,  # 별도 predict 필요
                    mcnemar_pvalue=1.0,
                    f1_diff=eval_b.macro_f1 - eval_a.macro_f1,
                    f1_ci_lower=0.0,
                    f1_ci_upper=0.0,
                    significant=False,
                )
                summary.comparisons.append(comp)

    def _evaluate_go_no_go(self, summary: AblationSummary) -> Dict[str, Any]:
        """
        Go/No-Go 판정 (Phase 1 → Phase 2 진행 조건)

        1. A5 (Full)이 COBRA 대비 Macro-F1 유의 개선 → run_phase1에서 확인
        2. 교차 데이터셋 성능 유지 → run_phase1에서 확인
        3. A3 (disagreement only)이 random baseline보다 유의하게 높음
        """
        go_no_go = {
            "criteria": [],
            "overall": None,
        }

        # 조건 3: A3 > random (random baseline = 1/3 ≈ 0.333)
        random_f1 = 1.0 / 3.0
        a3_result = summary.results.get("A3_disagreement_only")
        if a3_result:
            a3_f1 = a3_result.best_f1
            margin = a3_f1 - random_f1
            passed = margin > 0.05  # 5%p 이상 개선
            go_no_go["criteria"].append({
                "id": "C3_disagreement_signal",
                "description": "A3 (disagreement only) > random baseline",
                "value": a3_f1,
                "threshold": random_f1 + 0.05,
                "margin": margin,
                "passed": passed,
            })

        # 조건 1, 2는 run_phase1에서 COBRA와 비교하여 판정
        go_no_go["criteria"].append({
            "id": "C1_full_vs_cobra",
            "description": "A5 (Full) > COBRA (McNemar p < 0.05)",
            "value": None,
            "threshold": None,
            "passed": None,
            "note": "run_phase1에서 별도 평가",
        })

        go_no_go["criteria"].append({
            "id": "C2_cross_dataset",
            "description": "교차 데이터셋 Macro-F1 하락 < 5%p",
            "value": None,
            "threshold": None,
            "passed": None,
            "note": "run_phase1에서 별도 평가",
        })

        return go_no_go

    def print_summary(self, summary: AblationSummary) -> str:
        """요약 출력"""
        lines = ["=" * 70, "DAAC Phase 1 Ablation Summary", "=" * 70, ""]

        # 결과 테이블
        lines.append(f"{'Ablation':<30} {'Dim':>4} {'Best Model':<25} {'Macro-F1':>8}")
        lines.append("-" * 70)

        for ablation_id, result in sorted(summary.results.items()):
            lines.append(
                f"{ablation_id:<30} {result.feature_dim:>4} "
                f"{result.best_model:<25} {result.best_f1:>8.4f}"
            )

        lines.append("")

        # 모델별 상세
        for ablation_id, result in sorted(summary.results.items()):
            lines.append(f"\n--- {ablation_id} (dim={result.feature_dim}) ---")
            for model_name, eval_result in result.model_results.items():
                lines.append(self.evaluator.format_result(eval_result))

        # 비교
        if summary.comparisons:
            lines.append("\n" + "=" * 70)
            lines.append("Comparisons")
            lines.append("=" * 70)
            for comp in summary.comparisons:
                lines.append(self.evaluator.format_comparison(comp))

        # Go/No-Go
        lines.append("\n" + "=" * 70)
        lines.append("Go/No-Go Criteria")
        lines.append("=" * 70)
        for criterion in summary.go_no_go.get("criteria", []):
            status = (
                "PASS" if criterion["passed"] is True
                else "FAIL" if criterion["passed"] is False
                else "PENDING"
            )
            lines.append(f"  [{status:7s}] {criterion['description']}")
            if criterion.get("value") is not None:
                lines.append(
                    f"           value={criterion['value']:.4f}, "
                    f"threshold={criterion['threshold']:.4f}, "
                    f"margin={criterion.get('margin', 0):.4f}"
                )
            if criterion.get("note"):
                lines.append(f"           note: {criterion['note']}")

        output = "\n".join(lines)
        print(output)
        return output
