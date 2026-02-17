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
import warnings
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

MODEL_NAMES = ("logistic_regression", "gradient_boosting", "mlp")


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


def _safe_macro_f1(y_t: np.ndarray, y_p: np.ndarray) -> float:
    labels = np.unique(y_t)
    if labels.size == 0:
        return 0.0
    return float(
        f1_score(
            y_t,
            y_p,
            labels=labels,
            average="macro",
            zero_division=0,
        )
    )


def _router_guard_score(
    weights: np.ndarray,
    *,
    mode: str = "max_minus_entropy",
    eps: float = 1e-12,
) -> np.ndarray:
    if weights.ndim != 2:
        raise ValueError(f"weights must be 2-D array, got shape={weights.shape}")
    if weights.shape[0] == 0:
        return np.zeros(0, dtype=float)
    if weights.shape[1] <= 0:
        raise ValueError(f"weights must have positive width, got shape={weights.shape}")

    w = np.clip(weights.astype(float), 0.0, None)
    row_sums = np.sum(w, axis=1, keepdims=True)
    row_sums = np.where(row_sums <= eps, 1.0, row_sums)
    w = w / row_sums

    max_w = np.max(w, axis=1)
    n_agents = int(w.shape[1])
    if n_agents == 1:
        entropy = np.zeros_like(max_w)
        margin_top2 = np.ones_like(max_w)
    else:
        entropy = -np.sum(w * np.log(np.clip(w, eps, 1.0)), axis=1) / np.log(float(n_agents))
        top2 = np.sort(np.partition(w, kth=n_agents - 2, axis=1)[:, -2:], axis=1)
        margin_top2 = top2[:, 1] - top2[:, 0]

    key = str(mode).strip().lower()
    if key == "max_weight":
        return max_w
    if key == "neg_entropy":
        return 1.0 - entropy
    if key == "top2_margin":
        return margin_top2
    if key == "max_minus_entropy":
        return max_w - entropy
    raise ValueError(f"unsupported router guard score mode: {mode}")


def _build_guard_threshold_grid(
    scores: np.ndarray,
    *,
    threshold_grid: List[float] | None = None,
    n_thresholds: int = 11,
) -> List[float]:
    if threshold_grid:
        grid = sorted({float(x) for x in threshold_grid})
    else:
        if scores.size == 0:
            grid = []
        else:
            n = max(2, int(n_thresholds))
            q = np.linspace(0.0, 1.0, n)
            grid = sorted({float(v) for v in np.quantile(scores, q)})
    grid.append(float("inf"))
    return grid


def _apply_router_guard(
    phase1_pred: np.ndarray,
    phase2_pred: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> Tuple[np.ndarray, float]:
    if phase1_pred.shape != phase2_pred.shape:
        raise ValueError("phase1_pred and phase2_pred must share shape")
    if phase1_pred.shape[0] != scores.shape[0]:
        raise ValueError("prediction length and score length must match")
    use_phase2 = scores >= float(threshold)
    mixed = np.where(use_phase2, phase2_pred, phase1_pred)
    return mixed, float(np.mean(use_phase2))


def _select_guard_threshold(
    y_val: np.ndarray,
    phase1_val_pred: np.ndarray,
    phase2_val_pred: np.ndarray,
    scores_val: np.ndarray,
    thresholds: List[float],
    *,
    min_val_gain: float = 0.0,
    min_phase2_val_gain: float = -1.0,
    max_route_rate: float | None = None,
) -> Dict[str, float | bool]:
    phase1_f1 = _safe_macro_f1(y_val, phase1_val_pred)
    phase2_f1 = _safe_macro_f1(y_val, phase2_val_pred)
    phase2_gain = float(phase2_f1 - phase1_f1)
    tol = 1e-12
    raw_phase2_gate_pass = phase2_gain + tol >= float(min_phase2_val_gain)

    best = {
        "threshold": float("inf"),
        "val_macro_f1": phase1_f1,
        "val_gain_vs_phase1": 0.0,
        "phase2_route_rate": 0.0,
        "phase1_val_macro_f1": phase1_f1,
        "phase2_val_macro_f1": phase2_f1,
        "raw_phase2_val_gain": phase2_gain,
        "raw_phase2_gate_pass": bool(raw_phase2_gate_pass),
        "min_phase2_val_gain": float(min_phase2_val_gain),
    }
    if not raw_phase2_gate_pass:
        return best

    for thr in thresholds:
        y_hybrid, route_rate = _apply_router_guard(
            phase1_pred=phase1_val_pred,
            phase2_pred=phase2_val_pred,
            scores=scores_val,
            threshold=float(thr),
        )
        if max_route_rate is not None and route_rate > float(max_route_rate) + tol:
            continue
        val_f1 = _safe_macro_f1(y_val, y_hybrid)
        gain = float(val_f1 - phase1_f1)
        if gain + tol < float(min_val_gain):
            continue
        better_f1 = val_f1 > best["val_macro_f1"] + tol
        same_f1_more_conservative = abs(val_f1 - best["val_macro_f1"]) <= tol and route_rate < (
            best["phase2_route_rate"] - tol
        )
        if better_f1 or same_f1_more_conservative:
            best = {
                "threshold": float(thr),
                "val_macro_f1": float(val_f1),
                "val_gain_vs_phase1": gain,
                "phase2_route_rate": float(route_rate),
                "phase1_val_macro_f1": phase1_f1,
                "phase2_val_macro_f1": phase2_f1,
                "raw_phase2_val_gain": phase2_gain,
                "raw_phase2_gate_pass": bool(raw_phase2_gate_pass),
                "min_phase2_val_gain": float(min_phase2_val_gain),
            }
    return best


def _apply_non_regression_veto(
    *,
    phase1_results: Dict,
    phase1_preds: Dict,
    phase2_results: Dict,
    phase2_preds: Dict,
    best_p1: str,
    best_p2: str,
    tolerance: float = 0.0,
    fallback_prefix: str = "phase1_fallback_",
) -> Tuple[str, Dict[str, float | str | bool]]:
    p1_f1 = float(phase1_results[best_p1]["macro_f1"])
    p2_f1 = float(phase2_results[best_p2]["macro_f1"])
    tol = float(tolerance)
    applied = bool(p2_f1 + tol < p1_f1)
    fallback_key = ""
    selected = best_p2

    if applied:
        fallback_key = f"{fallback_prefix}{best_p1}"
        if fallback_key not in phase2_results:
            phase2_results[fallback_key] = dict(phase1_results[best_p1])
        if fallback_key not in phase2_preds:
            phase2_preds[fallback_key] = phase1_preds[best_p1]
        selected = fallback_key
        p2_f1 = float(phase2_results[selected]["macro_f1"])

    return selected, {
        "applied": applied,
        "tolerance": tol,
        "before_best_phase2": best_p2,
        "after_best_phase2": selected,
        "phase1_best_model": best_p1,
        "phase1_best_macro_f1": p1_f1,
        "phase2_selected_macro_f1": p2_f1,
        "fallback_key": fallback_key,
    }


def _evaluate_models(
    trainer: MetaTrainer,
    evaluator: MetaEvaluator,
    X_test: np.ndarray,
    y_test: np.ndarray,
    prefix: str,
) -> Tuple[Dict, Dict]:
    result_dict: Dict = {}
    pred_dict: Dict = {}
    for model_name in MODEL_NAMES:
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


def _build_macro_f1_result_dict(y_true: np.ndarray, pred_dict: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for model_name, y_pred in pred_dict.items():
        out[model_name] = {"macro_f1": _safe_macro_f1(y_true, y_pred)}
    return out


def _select_best_models_for_comparison(
    *,
    phase1_val_results: Dict[str, Dict[str, float]],
    phase2_val_results: Dict[str, Dict[str, float]],
    guard_enabled: bool,
    selection_scope: str,
    model_key_prefix: str,
) -> Tuple[str, str, Dict[str, Dict[str, float]], str]:
    if not phase1_val_results:
        raise ValueError("phase1_val_results is empty")
    if not phase2_val_results:
        raise ValueError("phase2_val_results is empty")

    resolved_scope = str(selection_scope).strip().lower()
    selection_results = phase2_val_results
    if guard_enabled and resolved_scope == "hybrid_only":
        prefix = f"{model_key_prefix}_"
        filtered = {k: v for k, v in phase2_val_results.items() if k.startswith(prefix)}
        if filtered:
            selection_results = filtered
        else:
            resolved_scope = "all"

    best_p1 = max(phase1_val_results, key=lambda k: phase1_val_results[k]["macro_f1"])
    best_p2 = max(selection_results, key=lambda k: selection_results[k]["macro_f1"])
    return best_p1, best_p2, selection_results, resolved_scope


def _evaluate_subgroups(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    test_data: List,
) -> Dict:
    """sub_type / true_label 단위 보조 지표."""
    out: Dict = {"by_sub_type": {}, "by_true_label": {}}

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
    _raw_split_seed = config.get("split", {}).get("seed")

    # Seed decoupling 마이그레이션 1단계:
    # split.seed가 명시되지 않으면 기존 동작(collector seed 연동)을 유지하되
    # 경고를 발생시키고 결과에 legacy_coupled_default 플래그를 남긴다.
    # 2단계: 실험 yaml 26개에 split.seed 명시 후 이 fallback 제거.
    if _raw_split_seed is None:
        warnings.warn(
            f"split.seed가 config에 명시되지 않아 collector.seed={seed}를 기본값으로 사용합니다. "
            "실험 config에 'split: {{seed: N}}'을 추가하면 이 경고가 사라집니다. "
            "이 기본값 동작은 향후 릴리스에서 제거될 예정입니다.",
            stacklevel=2,
        )
        split_seed = seed
        _split_seed_coupled = True
    else:
        split_seed = int(_raw_split_seed)
        _split_seed_coupled = False

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
        train_idx, val_idx, test_idx = collector.stratified_split(
            samples,
            train_ratio=float(split_cfg.get("train_ratio", 0.6)),
            val_ratio=float(split_cfg.get("val_ratio", 0.2)),
            test_ratio=float(split_cfg.get("test_ratio", 0.2)),
            seed=split_seed,
            return_indices=True,
        )
        train_data = [samples[i] for i in train_idx]
        val_data = [samples[i] for i in val_idx]
        test_data = [samples[i] for i in test_idx]
        train_records = [records[i] for i in train_idx]
        val_records = [records[i] for i in val_idx]
        test_records = [records[i] for i in test_idx]
        split_meta = {
            "strategy": "stratified",
            "seed": split_seed,
            "train_ratio": float(split_cfg.get("train_ratio", 0.6)),
            "val_ratio": float(split_cfg.get("val_ratio", 0.2)),
            "test_ratio": float(split_cfg.get("test_ratio", 0.2)),
            "legacy_coupled_default": _split_seed_coupled,
        }
    elif split_strategy in {"kfold", "stratified_kfold"}:
        k_folds = int(split_cfg.get("k_folds", 5))
        test_fold = int(split_cfg.get("test_fold", 0))
        val_fold = int(split_cfg.get("val_fold", (test_fold + 1) % max(1, k_folds)))
        train_idx, val_idx, test_idx = collector.stratified_kfold_split(
            samples,
            k_folds=k_folds,
            test_fold=test_fold,
            val_fold=val_fold,
            seed=split_seed,
            return_indices=True,
        )
        train_data = [samples[i] for i in train_idx]
        val_data = [samples[i] for i in val_idx]
        test_data = [samples[i] for i in test_idx]
        train_records = [records[i] for i in train_idx]
        val_records = [records[i] for i in val_idx]
        test_records = [records[i] for i in test_idx]
        split_meta = {
            "strategy": "kfold",
            "seed": split_seed,
            "k_folds": k_folds,
            "test_fold": int(test_fold) % k_folds,
            "val_fold": int(val_fold) % k_folds,
            "legacy_coupled_default": _split_seed_coupled,
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
            target_mode=str(oracle_cfg.get("target_mode", "accuracy")),
            label_smoothing=float(oracle_cfg.get("label_smoothing", 0.0)),
            confidence_power=float(oracle_cfg.get("confidence_power", 0.0)),
            uncertain_penalty=float(oracle_cfg.get("uncertain_penalty", 1.0)),
            entropy_power=float(oracle_cfg.get("entropy_power", 0.0)),
            verdict_power=float(oracle_cfg.get("verdict_power", 0.0)),
            margin_power=float(oracle_cfg.get("margin_power", 0.0)),
            wrong_peak_power=float(oracle_cfg.get("wrong_peak_power", 0.0)),
            majority_agreement_power=float(oracle_cfg.get("majority_agreement_power", 0.0)),
            disagreement_penalty=float(oracle_cfg.get("disagreement_penalty", 1.0)),
            uncertain_extra_penalty=float(oracle_cfg.get("uncertain_extra_penalty", 1.0)),
            adaptive_smoothing_scale=float(oracle_cfg.get("adaptive_smoothing_scale", 0.0)),
            adaptive_smoothing_max=float(oracle_cfg.get("adaptive_smoothing_max", 0.4)),
        ),
    )

    hls = model_cfg.get("hidden_layer_sizes", [64, 64])
    if isinstance(hls, list):
        hls = tuple(int(v) for v in hls)
    router_random_state = int(model_cfg.get("random_state", seed))

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
            random_state=router_random_state,
            regressor=str(model_cfg.get("regressor", "mlp")),
            ridge_alpha=float(model_cfg.get("ridge_alpha", 1.0)),
            gbr_n_estimators=int(model_cfg.get("gbr_n_estimators", 200)),
            gbr_learning_rate=float(model_cfg.get("gbr_learning_rate", 0.05)),
            gbr_max_depth=int(model_cfg.get("gbr_max_depth", 3)),
            gbr_subsample=float(model_cfg.get("gbr_subsample", 1.0)),
        ),
        eps=float(oracle_cfg.get("eps", 1e-6)),
    )

    use_evidence_records = feature_set == "evidence_2ch"
    X_train_img = build_proxy_image_features(
        train_data,
        feature_set=feature_set,
        records=train_records if use_evidence_records else None,
    )
    X_val_img = build_proxy_image_features(
        val_data,
        feature_set=feature_set,
        records=val_records if use_evidence_records else None,
    )
    X_test_img = build_proxy_image_features(
        test_data,
        feature_set=feature_set,
        records=test_records if use_evidence_records else None,
    )

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
    phase1_val_preds = {model_name: trainer_p1.predict(model_name, X_val_meta) for model_name in MODEL_NAMES}
    phase1_val_results = _build_macro_f1_result_dict(y_true=y_val, pred_dict=phase1_val_preds)

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
    phase2_val_preds = {model_name: trainer_p2.predict(model_name, X_val_p2) for model_name in MODEL_NAMES}
    phase2_val_results = _build_macro_f1_result_dict(y_true=y_val, pred_dict=phase2_val_preds)

    guard_cfg = router_cfg.get("guard", {})
    guard_enabled = bool(guard_cfg.get("enabled", False))
    guard_tuning: Dict[str, Dict] = {}
    guard_veto_info: Dict[str, float | str | bool] = {}
    model_key_prefix = str(guard_cfg.get("model_key_prefix", "hybrid_guard"))
    selection_scope = str(guard_cfg.get("selection_scope", "all")).strip().lower()
    if guard_enabled:
        print("\n  - Router guard fallback 활성화: val 기준 임계값 자동 선택")
        score_mode = str(guard_cfg.get("score_mode", "max_minus_entropy"))
        scores_val = _router_guard_score(W_val_pred, mode=score_mode)
        scores_test = _router_guard_score(W_test_pred, mode=score_mode)

        raw_threshold_grid = guard_cfg.get("threshold_grid", [])
        if isinstance(raw_threshold_grid, (int, float)):
            threshold_grid = [float(raw_threshold_grid)]
        elif isinstance(raw_threshold_grid, list):
            threshold_grid = [float(x) for x in raw_threshold_grid]
        else:
            threshold_grid = []
        thresholds = _build_guard_threshold_grid(
            scores_val,
            threshold_grid=threshold_grid,
            n_thresholds=int(guard_cfg.get("n_thresholds", 11)),
        )
        min_val_gain = float(guard_cfg.get("min_val_gain", 0.0))
        min_phase2_val_gain = float(guard_cfg.get("min_phase2_val_gain", -1.0))
        max_route_rate = guard_cfg.get("max_route_rate", None)
        if max_route_rate is not None:
            max_route_rate = float(max_route_rate)

        for model_name in MODEL_NAMES:
            phase1_val_pred = phase1_val_preds[model_name]
            phase2_val_pred = phase2_val_preds[model_name]
            selection = _select_guard_threshold(
                y_val=y_val,
                phase1_val_pred=phase1_val_pred,
                phase2_val_pred=phase2_val_pred,
                scores_val=scores_val,
                thresholds=thresholds,
                min_val_gain=min_val_gain,
                min_phase2_val_gain=min_phase2_val_gain,
                max_route_rate=max_route_rate,
            )

            hybrid_val_pred, route_rate_val = _apply_router_guard(
                phase1_pred=phase1_val_pred,
                phase2_pred=phase2_val_pred,
                scores=scores_val,
                threshold=float(selection["threshold"]),
            )
            hybrid_test_pred, route_rate_test = _apply_router_guard(
                phase1_pred=phase1_preds[model_name],
                phase2_pred=phase2_preds[model_name],
                scores=scores_test,
                threshold=float(selection["threshold"]),
            )
            er_hybrid = evaluator.evaluate(y_test, hybrid_test_pred, model_name=f"phase2_hybrid_{model_name}")
            hybrid_key = f"{model_key_prefix}_{model_name}"
            phase2_results[hybrid_key] = {
                "macro_f1": er_hybrid.macro_f1,
                "balanced_accuracy": er_hybrid.balanced_accuracy,
                "auroc": er_hybrid.auroc,
                "ece": er_hybrid.ece,
                "brier": er_hybrid.brier,
                "per_class_f1": er_hybrid.per_class_f1,
            }
            phase2_preds[hybrid_key] = hybrid_test_pred
            phase2_val_preds[hybrid_key] = hybrid_val_pred
            phase2_val_results[hybrid_key] = {
                "macro_f1": _safe_macro_f1(y_val, hybrid_val_pred),
            }
            guard_tuning[hybrid_key] = {
                "score_mode": score_mode,
                "threshold": float(selection["threshold"]),
                "phase2_route_rate_val": float(route_rate_val),
                "phase2_route_rate_test": float(route_rate_test),
                "val_macro_f1": float(selection["val_macro_f1"]),
                "val_gain_vs_phase1": float(selection["val_gain_vs_phase1"]),
                "phase1_val_macro_f1": float(selection["phase1_val_macro_f1"]),
                "phase2_val_macro_f1": float(selection["phase2_val_macro_f1"]),
                "raw_phase2_val_gain": float(selection["raw_phase2_val_gain"]),
                "raw_phase2_gate_pass": bool(selection["raw_phase2_gate_pass"]),
                "min_val_gain": float(min_val_gain),
                "min_phase2_val_gain": float(min_phase2_val_gain),
                "max_route_rate": max_route_rate,
            }
            print(
                f"    {hybrid_key}: thr={selection['threshold']:.4f}, "
                f"route_val={selection['phase2_route_rate']:.3f}, "
                f"route_test={route_rate_test:.3f}, "
                f"val_gain={selection['val_gain_vs_phase1']:+.4f}"
            )
    else:
        score_mode = str(guard_cfg.get("score_mode", "max_minus_entropy"))

    # ---------------------------------------------------------------
    # Step 6: Best Phase2 vs Best Phase1
    # ---------------------------------------------------------------
    print("\n[Step 6] Best Phase2 vs Best Phase1 비교 (val 선택 -> test 평가)...")
    best_p1, best_p2, phase2_selection_val_results, resolved_selection_scope = _select_best_models_for_comparison(
        phase1_val_results=phase1_val_results,
        phase2_val_results=phase2_val_results,
        guard_enabled=guard_enabled,
        selection_scope=selection_scope,
        model_key_prefix=model_key_prefix,
    )
    if guard_enabled and selection_scope == "hybrid_only":
        if resolved_selection_scope == "hybrid_only":
            print(f"  - Step6 selection scope: hybrid_only ({len(phase2_selection_val_results)} models)")
        else:
            print("  - Step6 selection scope=hybrid_only but no hybrid models found; fallback to all phase2 models")
    selection_scope = resolved_selection_scope

    if guard_enabled and bool(guard_cfg.get("enforce_non_regression", False)):
        best_p2, guard_veto_info = _apply_non_regression_veto(
            phase1_results=phase1_val_results,
            phase1_preds=phase1_val_preds,
            phase2_results=phase2_val_results,
            phase2_preds=phase2_val_preds,
            best_p1=best_p1,
            best_p2=best_p2,
            tolerance=float(guard_cfg.get("non_regression_tolerance", 0.0)),
            fallback_prefix=str(guard_cfg.get("fallback_prefix", "phase1_fallback_")),
        )
        if bool(guard_veto_info.get("applied", False)):
            print(
                f"  - Non-regression veto: {guard_veto_info['before_best_phase2']} -> "
                f"{guard_veto_info['after_best_phase2']}"
            )
            fallback_key = str(guard_veto_info.get("fallback_key", "")).strip()
            if fallback_key:
                # 최종 test 비교용 prediction/result에도 fallback 엔트리를 주입한다.
                if fallback_key not in phase2_results:
                    phase2_results[fallback_key] = dict(phase1_results[best_p1])
                if fallback_key not in phase2_preds:
                    phase2_preds[fallback_key] = phase1_preds[best_p1]
    else:
        guard_veto_info = {
            "applied": False,
            "tolerance": float(guard_cfg.get("non_regression_tolerance", 0.0)),
            "before_best_phase2": best_p2,
            "after_best_phase2": best_p2,
            "phase1_best_model": best_p1,
            "phase1_best_macro_f1": float(phase1_val_results[best_p1]["macro_f1"]),
            "phase2_selected_macro_f1": float(phase2_val_results[best_p2]["macro_f1"]),
            "fallback_key": "",
        }

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
        "seed_meta": {
            "collector_seed": seed,
            "split_seed": split_seed,
            "router_random_state": router_random_state,
            "split_seed_source": "config.split.seed" if _raw_split_seed is not None else "legacy_collector_seed_fallback",
            "router_seed_source": "config.router.model.random_state" if "random_state" in model_cfg else "collector_seed_fallback",
        },
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
                "target_mode": str(oracle_cfg.get("target_mode", "accuracy")),
                "label_smoothing": float(oracle_cfg.get("label_smoothing", 0.0)),
                "confidence_power": float(oracle_cfg.get("confidence_power", 0.0)),
                "uncertain_penalty": float(oracle_cfg.get("uncertain_penalty", 1.0)),
                "entropy_power": float(oracle_cfg.get("entropy_power", 0.0)),
                "verdict_power": float(oracle_cfg.get("verdict_power", 0.0)),
                "margin_power": float(oracle_cfg.get("margin_power", 0.0)),
                "wrong_peak_power": float(oracle_cfg.get("wrong_peak_power", 0.0)),
                "majority_agreement_power": float(oracle_cfg.get("majority_agreement_power", 0.0)),
                "disagreement_penalty": float(oracle_cfg.get("disagreement_penalty", 1.0)),
                "uncertain_extra_penalty": float(oracle_cfg.get("uncertain_extra_penalty", 1.0)),
                "adaptive_smoothing_scale": float(oracle_cfg.get("adaptive_smoothing_scale", 0.0)),
                "adaptive_smoothing_max": float(oracle_cfg.get("adaptive_smoothing_max", 0.4)),
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
            "guard": {
                "enabled": bool(guard_enabled),
                "score_mode": score_mode,
                "selection_scope": selection_scope,
                "tuning": guard_tuning,
                "non_regression_veto": guard_veto_info,
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
        "phase1_val_meta": phase1_val_results,
        "phase2_val_meta": phase2_val_results,
        "model_selection": {
            "metric": "val_macro_f1",
            "selection_scope": selection_scope,
            "phase1_best_val_macro_f1": float(phase1_val_results[best_p1]["macro_f1"]),
            "phase2_best_val_macro_f1": float(phase2_val_results[best_p2]["macro_f1"]),
        },
        "subgroup_metrics": {
            "phase1_best": subgroup_phase1_best,
            "phase2_best": subgroup_phase2_best,
        },
        "phase2_vs_phase1_best": {
            "phase1_best": best_p1,
            "phase2_best": best_p2,
            "selection_metric": "val_macro_f1",
            "phase1_best_val_macro_f1": float(phase1_val_results[best_p1]["macro_f1"]),
            "phase2_best_val_macro_f1": float(phase2_val_results[best_p2]["macro_f1"]),
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
