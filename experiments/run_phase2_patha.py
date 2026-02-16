"""
DAAC Phase 2 Path A 파이프라인 (실데이터 수집 기반).

개요:
    1) 실제 데이터셋에서 4개 에이전트 출력 수집
    2) Phase 1(43-dim) 메타 분류기 학습/평가
    3) Proxy image feature(20-dim) -> MetaRouter -> weights(4-dim) 예측
    4) Phase 2(47-dim) 메타 분류기 학습/평가
    5) Best Phase2 vs Best Phase1 통계 비교
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import yaml
from sklearn.metrics import balanced_accuracy_score, f1_score

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.meta.baselines import COBRABaseline, MajorityVoteBaseline
from src.meta.collector import (
    AgentOutputCollector,
    build_empirical_simulator,
    build_proxy_image_features,
)
from src.meta.evaluate import MetaEvaluator
from src.meta.features import MetaFeatureExtractor
from src.meta.router import (
    MetaRouter,
    MetaRouterConfig,
    OracleWeightComputer,
    OracleWeightConfig,
)
from src.meta.trainer import MetaTrainer


def load_config(config_path: str = "") -> Dict:
    if not config_path:
        config_path = str(_project_root / "experiments" / "configs" / "phase2_patha.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _coerce_hidden_layer_sizes(model_configs: Dict) -> None:
    mlp = model_configs.get("mlp", {})
    h = mlp.get("hidden_layer_sizes")
    if isinstance(h, list):
        mlp["hidden_layer_sizes"] = tuple(h)


def _evaluate_models(
    trainer: MetaTrainer,
    evaluator: MetaEvaluator,
    X_test: np.ndarray,
    y_test: np.ndarray,
    prefix: str,
) -> Tuple[Dict, Dict]:
    result_dict: Dict = {}
    pred_dict: Dict = {}
    for model_name in ["logistic_regression", "gradient_boosting", "mlp"]:
        y_pred = trainer.predict(model_name, X_test)
        y_proba = trainer.predict_proba(model_name, X_test)
        er = evaluator.evaluate(y_test, y_pred, y_proba, model_name=f"{prefix}_{model_name}")
        result_dict[model_name] = {
            "macro_f1": er.macro_f1,
            "balanced_accuracy": er.balanced_accuracy,
            "auroc": er.auroc,
            "ece": er.ece,
            "brier": er.brier,
            "per_class_f1": er.per_class_f1,
        }
        pred_dict[model_name] = y_pred
    return result_dict, pred_dict


def _evaluate_subgroups(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    test_data: List,
) -> Dict:
    """sub_type / true_label 단위 보조 지표."""
    out: Dict = {"by_sub_type": {}, "by_true_label": {}}

    def _safe_macro_f1(y_t: np.ndarray, y_p: np.ndarray) -> float:
        labels = np.unique(y_t)
        if labels.size == 0:
            return 0.0
        # 부분집합 평가에서 y_true에 없는 클래스 예측으로 발생하는 경고를 방지한다.
        return float(
            f1_score(
                y_t,
                y_p,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        )

    def _safe_bal_acc(y_t: np.ndarray, y_p: np.ndarray) -> float:
        labels = np.unique(y_t)
        if labels.size == 0:
            return 0.0
        if labels.size == 1:
            # single-class subset에서는 해당 클래스 recall == accuracy와 동일하다.
            return float(np.mean(y_t == y_p))
        return float(balanced_accuracy_score(y_t, y_p))

    sub_types = sorted({str(s.sub_type) for s in test_data})
    for st in sub_types:
        idx = np.array([i for i, s in enumerate(test_data) if str(s.sub_type) == st], dtype=int)
        if idx.size == 0:
            continue
        out["by_sub_type"][st] = {
            "n_samples": int(idx.size),
            "macro_f1": _safe_macro_f1(y_true[idx], y_pred[idx]),
            "balanced_accuracy": _safe_bal_acc(y_true[idx], y_pred[idx]),
        }

    true_labels = sorted({str(s.true_label) for s in test_data})
    for tl in true_labels:
        idx = np.array([i for i, s in enumerate(test_data) if str(s.true_label) == tl], dtype=int)
        if idx.size == 0:
            continue
        out["by_true_label"][tl] = {
            "n_samples": int(idx.size),
            "macro_f1": _safe_macro_f1(y_true[idx], y_pred[idx]),
            "balanced_accuracy": _safe_bal_acc(y_true[idx], y_pred[idx]),
        }

    return out


def run_phase2_patha(config: Dict) -> Dict:
    start_time = time.time()
    seed = int(config.get("collector", {}).get("seed", 42))
    split_seed = int(config.get("split", {}).get("seed", seed))

    print("=" * 70)
    print("DAAC Phase 2: Path A (Real Data Collector)")
    print("=" * 70)

    # ---------------------------------------------------------------
    # Step 1: 실제 에이전트 출력 수집
    # ---------------------------------------------------------------
    print("\n[Step 1] 실제 에이전트 출력 수집...")
    collector_cfg = config.get("collector", {})
    collector = AgentOutputCollector(
        device=str(collector_cfg.get("device", "cuda")),
        noise_backend=str(collector_cfg.get("noise_backend", "mvss")),
        spatial_backend=str(collector_cfg.get("spatial_backend", "mesorch")),
        seed=seed,
    )

    precollected_jsonl = str(collector_cfg.get("precollected_jsonl", "")).strip()
    loaded_from_jsonl = False
    if precollected_jsonl:
        precollected_path = Path(precollected_jsonl)
        if not precollected_path.exists():
            raise FileNotFoundError(f"precollected_jsonl not found: {precollected_jsonl}")
        samples, records = AgentOutputCollector.load_jsonl(precollected_path)
        errors = []
        loaded_from_jsonl = True
        print(f"  loaded precollected records: {len(samples)} from {precollected_path}")
    else:
        class_specs = config.get("datasets", {}).get("classes", {})
        samples, records, errors = collector.collect(
            class_specs=class_specs,
            verbose_every=int(collector_cfg.get("verbose_every", 20)),
        )

    if not samples:
        raise RuntimeError("No collected samples from Path A dataset specs")
    print(f"  collected: {len(samples)} samples, errors: {len(errors)}")

    split_cfg = config.get("split", {})
    split_strategy = str(split_cfg.get("strategy", "stratified")).strip().lower()
    if split_strategy in {"stratified", "ratio"}:
        train_data, val_data, test_data = collector.stratified_split(
            samples,
            train_ratio=float(split_cfg.get("train_ratio", 0.6)),
            val_ratio=float(split_cfg.get("val_ratio", 0.2)),
            test_ratio=float(split_cfg.get("test_ratio", 0.2)),
            seed=split_seed,
        )
        split_meta = {
            "strategy": "stratified",
            "seed": split_seed,
            "train_ratio": float(split_cfg.get("train_ratio", 0.6)),
            "val_ratio": float(split_cfg.get("val_ratio", 0.2)),
            "test_ratio": float(split_cfg.get("test_ratio", 0.2)),
        }
    elif split_strategy in {"kfold", "stratified_kfold"}:
        k_folds = int(split_cfg.get("k_folds", 5))
        test_fold = int(split_cfg.get("test_fold", 0))
        val_fold = int(split_cfg.get("val_fold", (test_fold + 1) % max(1, k_folds)))
        train_data, val_data, test_data = collector.stratified_kfold_split(
            samples,
            k_folds=k_folds,
            test_fold=test_fold,
            val_fold=val_fold,
            seed=split_seed,
        )
        split_meta = {
            "strategy": "kfold",
            "seed": split_seed,
            "k_folds": k_folds,
            "test_fold": int(test_fold) % k_folds,
            "val_fold": int(val_fold) % k_folds,
        }
    else:
        raise ValueError(f"unsupported split.strategy: {split_strategy}")
    print(f"  split: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")

    # ---------------------------------------------------------------
    # Step 2: Phase 1 43-dim 메타 특징
    # ---------------------------------------------------------------
    print("\n[Step 2] Phase 1 43-dim 메타 특징 추출...")
    extractor = MetaFeatureExtractor()
    X_train_meta, y_train = extractor.extract_dataset(train_data)
    X_val_meta, y_val = extractor.extract_dataset(val_data)
    X_test_meta, y_test = extractor.extract_dataset(test_data)
    print(f"  X_train_meta: {X_train_meta.shape}, X_test_meta: {X_test_meta.shape}")

    # ---------------------------------------------------------------
    # Step 3: 베이스라인
    # ---------------------------------------------------------------
    print("\n[Step 3] 베이스라인 평가...")
    eval_cfg = config.get("evaluation", {})
    evaluator = MetaEvaluator(
        n_bootstrap=int(eval_cfg.get("n_bootstrap", 200)),
        ci_level=float(eval_cfg.get("ci_level", 0.95)),
        seed=seed,
    )

    mv = MajorityVoteBaseline()
    y_pred_mv = mv.predict(test_data)
    eval_mv = evaluator.evaluate(y_test, y_pred_mv, model_name="majority_vote")
    print(evaluator.format_result(eval_mv))

    cobra_cfg = config.get("cobra", {})
    cobra = COBRABaseline(
        trust_scores=cobra_cfg.get("trust_scores"),
        algorithm=str(cobra_cfg.get("algorithm", "drwa")),
    )
    y_pred_cobra = cobra.predict(test_data)
    y_proba_cobra = cobra.predict_proba(test_data)
    eval_cobra = evaluator.evaluate(y_test, y_pred_cobra, y_proba_cobra, "cobra")
    print(evaluator.format_result(eval_cobra))

    # ---------------------------------------------------------------
    # Step 4: Router 학습 (Proxy image features)
    # ---------------------------------------------------------------
    print("\n[Step 4] Meta-Router 학습 (Path A proxy features)...")
    empirical_sim = build_empirical_simulator(train_data, seed=seed)
    router_cfg = config.get("router", {})
    oracle_cfg = router_cfg.get("oracle", {})
    model_cfg = router_cfg.get("model", {})
    feature_cfg = router_cfg.get("features", {})
    feature_set = str(feature_cfg.get("profile", "base20"))

    oracle = OracleWeightComputer(
        simulator=empirical_sim,
        config=OracleWeightConfig(
            power=float(oracle_cfg.get("power", 2.0)),
            eps=float(oracle_cfg.get("eps", 1e-6)),
            label_smoothing=float(oracle_cfg.get("label_smoothing", 0.0)),
            confidence_power=float(oracle_cfg.get("confidence_power", 0.0)),
            uncertain_penalty=float(oracle_cfg.get("uncertain_penalty", 1.0)),
            entropy_power=float(oracle_cfg.get("entropy_power", 0.0)),
            verdict_power=float(oracle_cfg.get("verdict_power", 0.0)),
        ),
    )

    hls = model_cfg.get("hidden_layer_sizes", [64, 64])
    if isinstance(hls, list):
        hls = tuple(int(v) for v in hls)

    router = MetaRouter(
        config=MetaRouterConfig(
            hidden_layer_sizes=tuple(hls),
            activation=str(model_cfg.get("activation", "relu")),
            solver=str(model_cfg.get("solver", "adam")),
            alpha=float(model_cfg.get("alpha", 1e-4)),
            learning_rate_init=float(model_cfg.get("learning_rate_init", 1e-3)),
            max_iter=int(model_cfg.get("max_iter", 500)),
            early_stopping=bool(model_cfg.get("early_stopping", True)),
            validation_fraction=float(model_cfg.get("validation_fraction", 0.15)),
            n_iter_no_change=int(model_cfg.get("n_iter_no_change", 20)),
            random_state=int(model_cfg.get("random_state", seed)),
            regressor=str(model_cfg.get("regressor", "mlp")),
            ridge_alpha=float(model_cfg.get("ridge_alpha", 1.0)),
            gbr_n_estimators=int(model_cfg.get("gbr_n_estimators", 200)),
            gbr_learning_rate=float(model_cfg.get("gbr_learning_rate", 0.05)),
            gbr_max_depth=int(model_cfg.get("gbr_max_depth", 3)),
            gbr_subsample=float(model_cfg.get("gbr_subsample", 1.0)),
        ),
        eps=float(oracle_cfg.get("eps", 1e-6)),
    )

    X_train_img = build_proxy_image_features(train_data, feature_set=feature_set)
    X_val_img = build_proxy_image_features(val_data, feature_set=feature_set)
    X_test_img = build_proxy_image_features(test_data, feature_set=feature_set)

    W_train_oracle = oracle.compute_dataset(train_data)
    W_val_oracle = oracle.compute_dataset(val_data)
    W_test_oracle = oracle.compute_dataset(test_data)

    router.fit(X_train_img, W_train_oracle)
    m_train = router.evaluate(X_train_img, W_train_oracle)
    m_val = router.evaluate(X_val_img, W_val_oracle)
    m_test = router.evaluate(X_test_img, W_test_oracle)

    print(
        f"  Router metrics (train): mse={m_train.mse_weights:.6f} "
        f"kl={m_train.kl_weights:.6f} top1={m_train.top1_match:.4f}"
    )
    print(
        f"  Router metrics (val)  : mse={m_val.mse_weights:.6f} "
        f"kl={m_val.kl_weights:.6f} top1={m_val.top1_match:.4f}"
    )
    print(
        f"  Router metrics (test) : mse={m_test.mse_weights:.6f} "
        f"kl={m_test.kl_weights:.6f} top1={m_test.top1_match:.4f}"
    )

    W_train_pred = router.predict_weights(X_train_img)
    W_val_pred = router.predict_weights(X_val_img)
    W_test_pred = router.predict_weights(X_test_img)

    # ---------------------------------------------------------------
    # Step 5: Phase 1 vs Phase 2 메타 분류기 비교
    # ---------------------------------------------------------------
    print("\n[Step 5] 메타 분류기 학습/평가 (Path A Phase1 vs Phase2)...")
    model_configs = config.get("models", {})
    _coerce_hidden_layer_sizes(model_configs)

    trainer_p1 = MetaTrainer(model_configs=model_configs)
    trainer_p1.train_all(X_train_meta, y_train, X_val_meta, y_val)
    phase1_results, phase1_preds = _evaluate_models(
        trainer=trainer_p1,
        evaluator=evaluator,
        X_test=X_test_meta,
        y_test=y_test,
        prefix="phase1",
    )

    X_train_p2 = np.concatenate([X_train_meta, W_train_pred], axis=1)
    X_val_p2 = np.concatenate([X_val_meta, W_val_pred], axis=1)
    X_test_p2 = np.concatenate([X_test_meta, W_test_pred], axis=1)

    trainer_p2 = MetaTrainer(model_configs=model_configs)
    trainer_p2.train_all(X_train_p2, y_train, X_val_p2, y_val)
    phase2_results, phase2_preds = _evaluate_models(
        trainer=trainer_p2,
        evaluator=evaluator,
        X_test=X_test_p2,
        y_test=y_test,
        prefix="phase2",
    )

    # ---------------------------------------------------------------
    # Step 6: Best Phase2 vs Best Phase1
    # ---------------------------------------------------------------
    print("\n[Step 6] Best Phase2 vs Best Phase1 비교...")
    best_p1 = max(phase1_results, key=lambda k: phase1_results[k]["macro_f1"])
    best_p2 = max(phase2_results, key=lambda k: phase2_results[k]["macro_f1"])

    cmp_res = evaluator.compare(
        y_true=y_test,
        y_pred_a=phase1_preds[best_p1],
        y_pred_b=phase2_preds[best_p2],
        model_a=f"phase1_{best_p1}",
        model_b=f"phase2_{best_p2}",
    )

    print(
        f"  F1 diff ({best_p2} - {best_p1}): {cmp_res.f1_diff:+.4f}  "
        f"[{cmp_res.f1_ci_lower:+.4f}, {cmp_res.f1_ci_upper:+.4f}]"
    )
    print(
        f"  McNemar p={cmp_res.mcnemar_pvalue:.4f}, significant={cmp_res.significant} "
        f"(b={cmp_res.discordant_a_correct_b_wrong}, c={cmp_res.discordant_a_wrong_b_correct})"
    )

    subgroup_phase1_best = _evaluate_subgroups(y_true=y_test, y_pred=phase1_preds[best_p1], test_data=test_data)
    subgroup_phase2_best = _evaluate_subgroups(y_true=y_test, y_pred=phase2_preds[best_p2], test_data=test_data)

    # ---------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------
    output_cfg = config.get("output", {})
    save_dir = Path(output_cfg.get("save_dir", "experiments/results/phase2_patha"))
    save_dir.mkdir(parents=True, exist_ok=True)
    now_tag = datetime.now().strftime("%Y%m%d_%H%M%S")

    if loaded_from_jsonl:
        records_path = Path(precollected_jsonl)
    else:
        records_path = save_dir / f"patha_agent_outputs_{now_tag}.jsonl"
        AgentOutputCollector.save_jsonl(records, records_path)

    elapsed = time.time() - start_time
    results = {
        "timestamp": datetime.now().isoformat(),
        "mode": "PathA_real_data_collector",
        "seed": seed,
        "split_seed": split_seed,
        "config": config,
        "artifacts": {
            "agent_outputs_jsonl": str(records_path),
            "data_source": "precollected_jsonl" if loaded_from_jsonl else "live_collection",
            "precollected_jsonl": precollected_jsonl or None,
            "collector_errors": errors,
        },
        "split_sizes": {
            "all": len(samples),
            "train": len(train_data),
            "val": len(val_data),
            "test": len(test_data),
        },
        "split": split_meta,
        "baselines": {
            "majority_vote": {
                "macro_f1": eval_mv.macro_f1,
                "balanced_accuracy": eval_mv.balanced_accuracy,
                "per_class_f1": eval_mv.per_class_f1,
            },
            "cobra": {
                "macro_f1": eval_cobra.macro_f1,
                "balanced_accuracy": eval_cobra.balanced_accuracy,
                "auroc": eval_cobra.auroc,
                "ece": eval_cobra.ece,
                "brier": eval_cobra.brier,
                "per_class_f1": eval_cobra.per_class_f1,
            },
        },
        "router": {
            "proxy_feature_dim": int(X_train_img.shape[1]),
            "feature_set": feature_set,
            "oracle_config": {
                "power": float(oracle_cfg.get("power", 2.0)),
                "eps": float(oracle_cfg.get("eps", 1e-6)),
                "label_smoothing": float(oracle_cfg.get("label_smoothing", 0.0)),
                "confidence_power": float(oracle_cfg.get("confidence_power", 0.0)),
                "uncertain_penalty": float(oracle_cfg.get("uncertain_penalty", 1.0)),
                "entropy_power": float(oracle_cfg.get("entropy_power", 0.0)),
                "verdict_power": float(oracle_cfg.get("verdict_power", 0.0)),
            },
            "model_config": {
                "regressor": str(model_cfg.get("regressor", "mlp")),
                "hidden_layer_sizes": list(hls),
                "activation": str(model_cfg.get("activation", "relu")),
                "solver": str(model_cfg.get("solver", "adam")),
                "alpha": float(model_cfg.get("alpha", 1e-4)),
                "learning_rate_init": float(model_cfg.get("learning_rate_init", 1e-3)),
                "max_iter": int(model_cfg.get("max_iter", 500)),
                "early_stopping": bool(model_cfg.get("early_stopping", True)),
                "validation_fraction": float(model_cfg.get("validation_fraction", 0.15)),
                "n_iter_no_change": int(model_cfg.get("n_iter_no_change", 20)),
                "random_state": int(model_cfg.get("random_state", seed)),
                "ridge_alpha": float(model_cfg.get("ridge_alpha", 1.0)),
                "gbr_n_estimators": int(model_cfg.get("gbr_n_estimators", 200)),
                "gbr_learning_rate": float(model_cfg.get("gbr_learning_rate", 0.05)),
                "gbr_max_depth": int(model_cfg.get("gbr_max_depth", 3)),
                "gbr_subsample": float(model_cfg.get("gbr_subsample", 1.0)),
            },
            "metrics": {
                "train": asdict(m_train),
                "val": asdict(m_val),
                "test": asdict(m_test),
            },
        },
        "protocol": config.get("protocol", {}),
        "phase1_meta": phase1_results,
        "phase2_meta": phase2_results,
        "subgroup_metrics": {
            "phase1_best": subgroup_phase1_best,
            "phase2_best": subgroup_phase2_best,
        },
        "phase2_vs_phase1_best": {
            "phase1_best": best_p1,
            "phase2_best": best_p2,
            "f1_diff": cmp_res.f1_diff,
            "mcnemar_statistic": cmp_res.mcnemar_statistic,
            "mcnemar_pvalue": cmp_res.mcnemar_pvalue,
            "mcnemar_b": cmp_res.discordant_a_correct_b_wrong,
            "mcnemar_c": cmp_res.discordant_a_wrong_b_correct,
            "n_test_samples": cmp_res.n_samples,
            "significant": cmp_res.significant,
            "ci_lower": cmp_res.f1_ci_lower,
            "ci_upper": cmp_res.f1_ci_upper,
        },
        "elapsed_seconds": elapsed,
    }

    result_path = save_dir / f"phase2_patha_results_{now_tag}.json"
    results["result_path"] = str(result_path)
    with result_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n총 실행 시간: {elapsed:.1f}초")
    print(f"결과 저장: {result_path}")
    return results


if __name__ == "__main__":
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else ""
    cfg = load_config(cfg_path)
    run_phase2_patha(cfg)
