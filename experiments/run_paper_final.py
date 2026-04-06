"""
Paper-quality multi-seed experiment: baselines + ablation for DAAC.

Loads the precollected JSONL to skip agent inference.
Loops over 10 split seeds → mean ± std for all metrics.

Output:
  Table 1 — DAAC vs baselines (macro-F1 mean ± std + Wilcoxon p vs COBRA)
  Table 2 — Per-class F1: COBRA vs DAAC-GBM vs Logistic Stacking
  Table 3 — Feature group ablation A1~A5 (GBM)
  Table 4 — GBM feature importance (top 15, averaged over seeds)
  JSON    — experiments/results/paper_final/paper_final_{ts}.json

Usage:
  .venv-qwen/bin/python experiments/run_paper_final.py experiments/configs/paper_final.yaml
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml
from scipy.stats import wilcoxon
from sklearn.metrics import f1_score

from src.meta.baselines import COBRABaseline, MajorityVoteBaseline, WeightedMajorityVoteBaseline
from src.meta.collector import AgentOutputCollector
from src.meta.features import ABLATION_CONFIGS, MetaFeatureExtractor
from src.meta.simulator import SimulatedOutput
from src.meta.trainer import MetaTrainer

CLASSES = ["authentic", "manipulated", "ai_generated"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _split(
    samples: List[SimulatedOutput],
    seed: int,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
) -> Tuple[List, List, List]:
    collector = AgentOutputCollector.__new__(AgentOutputCollector)
    train_idx, val_idx, test_idx = collector.stratified_split(
        samples,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        return_indices=True,
    )
    return (
        [samples[i] for i in train_idx],
        [samples[i] for i in val_idx],
        [samples[i] for i in test_idx],
    )


def _perclass(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Per-class F1 dict keyed by class name."""
    pf = f1_score(y_true, y_pred, average=None, zero_division=0, labels=[0, 1, 2])
    return {cls: float(pf[i]) for i, cls in enumerate(CLASSES)}


def _run_seed(
    samples: List[SimulatedOutput],
    seed: int,
    trust_scores: Dict[str, float],
    split_cfg: Dict,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Run every method for one split seed.

    Returns:
        results  — {method: macro_f1}  (for aggregation)
        extras   — {perclass, feature_importances}
    """
    train_data, val_data, test_data = _split(
        samples, seed,
        train_ratio=float(split_cfg.get("train_ratio", 0.6)),
        val_ratio=float(split_cfg.get("val_ratio", 0.2)),
        test_ratio=float(split_cfg.get("test_ratio", 0.2)),
    )

    # Full-feature extractor (A5, 43-dim)
    ext_full = MetaFeatureExtractor()
    X_tr, y_tr = ext_full.extract_dataset(train_data)
    X_va, y_va = ext_full.extract_dataset(val_data)
    X_te, y_te = ext_full.extract_dataset(test_data)

    results: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Baselines — save raw predictions for per-class F1
    # ------------------------------------------------------------------
    mv_pred = MajorityVoteBaseline().predict(test_data)
    results["majority_vote"] = float(f1_score(y_te, mv_pred, average="macro", zero_division=0))

    wmv_pred = WeightedMajorityVoteBaseline(trust_scores).predict(test_data)
    results["weighted_mv"] = float(f1_score(y_te, wmv_pred, average="macro", zero_division=0))

    cobra_pred = COBRABaseline(trust_scores, algorithm="drwa").predict(test_data)
    results["cobra"] = float(f1_score(y_te, cobra_pred, average="macro", zero_division=0))

    # ------------------------------------------------------------------
    # Naive stacking — LR on A4 (verdict + confidence, 20-dim)
    # ------------------------------------------------------------------
    ext_a4 = MetaFeatureExtractor(ABLATION_CONFIGS["A4_verdict_confidence"])
    X_tr_a4, _ = ext_a4.extract_dataset(train_data)
    X_va_a4, _ = ext_a4.extract_dataset(val_data)
    X_te_a4, _ = ext_a4.extract_dataset(test_data)

    trainer_lr_a4 = MetaTrainer()
    trainer_lr_a4.train("logistic_regression", X_tr_a4, y_tr, X_va_a4, y_va)
    lstacking_pred = trainer_lr_a4.predict("logistic_regression", X_te_a4)
    results["logistic_stacking"] = float(f1_score(y_te, lstacking_pred, average="macro", zero_division=0))

    # ------------------------------------------------------------------
    # DAAC — all 3 models on A5 full features (43-dim)
    # ------------------------------------------------------------------
    trainer_full = MetaTrainer()
    trainer_full.train_all(X_tr, y_tr, X_va, y_va)

    daac_preds: Dict[str, np.ndarray] = {}
    for mname in ("logistic_regression", "gradient_boosting", "mlp"):
        daac_preds[mname] = trainer_full.predict(mname, X_te)
        results[f"daac_{mname}"] = float(
            f1_score(y_te, daac_preds[mname], average="macro", zero_division=0)
        )

    # ------------------------------------------------------------------
    # Ablation — A1~A5, GBM only
    # ------------------------------------------------------------------
    for ablation_id, abl_cfg in ABLATION_CONFIGS.items():
        ext_abl = MetaFeatureExtractor(abl_cfg)
        X_tr_abl, _ = ext_abl.extract_dataset(train_data)
        X_va_abl, _ = ext_abl.extract_dataset(val_data)
        X_te_abl, _ = ext_abl.extract_dataset(test_data)

        trainer_abl = MetaTrainer()
        trainer_abl.train("gradient_boosting", X_tr_abl, y_tr, X_va_abl, y_va)
        results[f"abl_{ablation_id}"] = float(
            f1_score(y_te, trainer_abl.predict("gradient_boosting", X_te_abl),
                     average="macro", zero_division=0)
        )

    # ------------------------------------------------------------------
    # Extras: per-class F1 + feature importances
    # ------------------------------------------------------------------
    perclass = {
        "cobra":                  _perclass(y_te, cobra_pred),
        "logistic_stacking":      _perclass(y_te, lstacking_pred),
        "daac_gradient_boosting": _perclass(y_te, daac_preds["gradient_boosting"]),
        "daac_logistic_regression": _perclass(y_te, daac_preds["logistic_regression"]),
        "daac_mlp":               _perclass(y_te, daac_preds["mlp"]),
    }
    fi = trainer_full.get_feature_importance("gradient_boosting")  # (43,) or None

    extras: Dict[str, Any] = {
        "perclass": perclass,
        "feature_importances": fi.tolist() if fi is not None else None,
    }
    return results, extras


# ---------------------------------------------------------------------------
# Statistical significance
# ---------------------------------------------------------------------------

def _pvalue_str(p: float) -> str:
    if np.isnan(p):
        return "  —  "
    if p < 0.001:
        return f"{p:.1e}***"
    if p < 0.01:
        return f"{p:.4f}** "
    if p < 0.05:
        return f"{p:.4f}*  "
    return f"{p:.4f}   "


def _compute_wilcoxon(
    all_results: List[Dict[str, float]],
    reference: str = "cobra",
) -> Dict[str, float]:
    ref_runs = [r[reference] for r in all_results]
    pvalues: Dict[str, float] = {}
    for key in all_results[0]:
        if key == reference:
            pvalues[key] = float("nan")
            continue
        method_runs = [r[key] for r in all_results]
        try:
            _, p = wilcoxon(method_runs, ref_runs, alternative="two-sided")
        except Exception:
            p = float("nan")
        pvalues[key] = float(p)
    return pvalues


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def _print_table1(agg: Dict[str, Dict], pvalues: Dict[str, float]) -> str:
    METHOD_ORDER = [
        ("majority_vote",            "Majority Vote"),
        ("weighted_mv",              "Weighted Majority Vote"),
        ("cobra",                    "COBRA (drwa)"),
        ("logistic_stacking",        "Logistic Stacking (A4+LR)"),
        ("daac_logistic_regression", "DAAC-LR  (A5+LR)"),
        ("daac_gradient_boosting",   "DAAC-GBM (A5+GBM)  ← proposed"),
        ("daac_mlp",                 "DAAC-MLP (A5+MLP)"),
    ]
    header = f"{'Method':<38} {'F1 mean':>9} {'±std':>7}  {'p (vs COBRA)'}"
    sep = "-" * 72
    lines = ["\n=== Table 1: Method Comparison (macro-F1, 10 seeds) ===", header, sep]
    for key, label in METHOD_ORDER:
        if key in agg:
            m, s = agg[key]["mean"], agg[key]["std"]
            p_str = _pvalue_str(pvalues.get(key, float("nan")))
            lines.append(f"{label:<38} {m:>9.4f} {s:>7.4f}  {p_str}")
    lines.append("\n  *** p<0.001  ** p<0.01  * p<0.05  (Wilcoxon signed-rank, 10 seeds)")
    out = "\n".join(lines)
    print(out)
    return out


def _print_perclass_table(all_extras: List[Dict]) -> str:
    """Per-class F1 averaged over seeds for key methods."""
    METHODS = [
        ("cobra",                  "COBRA (drwa)"),
        ("logistic_stacking",      "Logistic Stacking"),
        ("daac_gradient_boosting", "DAAC-GBM (proposed)"),
    ]
    header = f"{'Method':<26} {'authentic':>10} {'manipulated':>13} {'ai_generated':>14}"
    sep = "-" * 67
    lines = ["\n=== Table 2: Per-class F1 (mean over 10 seeds) ===", header, sep]
    for key, label in METHODS:
        runs_by_class: Dict[str, List[float]] = {c: [] for c in CLASSES}
        for e in all_extras:
            pc = e["perclass"].get(key, {})
            for c in CLASSES:
                runs_by_class[c].append(pc.get(c, float("nan")))
        row = f"{label:<26}"
        for c in CLASSES:
            vals = [v for v in runs_by_class[c] if not np.isnan(v)]
            mu = np.mean(vals) if vals else float("nan")
            row += f" {mu:>10.4f}    "
        lines.append(row.rstrip())
    out = "\n".join(lines)
    print(out)
    return out


def _print_table3(agg: Dict[str, Dict]) -> str:
    ABLATION_ORDER = [
        ("abl_A1_confidence_only",    "A1  confidence only",    4),
        ("abl_A2_verdict_only",       "A2  verdict only",       16),
        ("abl_A3_disagreement_only",  "A3  disagreement only",  23),
        ("abl_A4_verdict_confidence", "A4  verdict+confidence", 20),
        ("abl_A5_full",               "A5  full DAAC",          43),
    ]
    header = f"{'Ablation':<28} {'Dim':>4} {'GBM F1':>9} {'±std':>7}"
    sep = "-" * 52
    lines = ["\n=== Table 3: Feature Group Ablation (GBM, 10 seeds) ===", header, sep]
    for key, label, dim in ABLATION_ORDER:
        if key in agg:
            m, s = agg[key]["mean"], agg[key]["std"]
            lines.append(f"{label:<28} {dim:>4} {m:>9.4f} {s:>7.4f}")
    out = "\n".join(lines)
    print(out)
    return out


def _print_feature_importance(all_extras: List[Dict]) -> str:
    """Top 15 GBM feature importances averaged over all seeds."""
    fi_arrays = [
        np.array(e["feature_importances"])
        for e in all_extras
        if e.get("feature_importances") is not None
    ]
    if not fi_arrays:
        return ""
    avg_fi = np.mean(fi_arrays, axis=0)
    feature_names = MetaFeatureExtractor().feature_names
    top_idx = np.argsort(avg_fi)[::-1][:15]

    lines = ["\n=== Table 4: Top-15 GBM Feature Importances (avg 10 seeds) ===",
             f"  {'Rank':<5} {'Feature':<45} {'Importance':>10}",
             "  " + "-" * 64]
    for rank, i in enumerate(top_idx, 1):
        lines.append(f"  {rank:<5} {feature_names[i]:<45} {avg_fi[i]:>10.4f}")
    out = "\n".join(lines)
    print(out)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    config_path = sys.argv[1] if len(sys.argv) > 1 else (
        str(Path(__file__).parent / "configs" / "paper_final.yaml")
    )
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    jsonl_path = cfg["precollected_jsonl"]
    seeds: List[int] = [int(s) for s in cfg.get("seeds", list(range(300, 310)))]
    trust_scores: Dict[str, float] = cfg.get("trust_scores", {})
    split_cfg = cfg.get("split", {})
    save_dir = Path(cfg.get("output", {}).get("save_dir", "experiments/results/paper_final"))
    label = cfg.get("label", Path(config_path).stem)

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"Loading JSONL: {jsonl_path}")
    samples, _records = AgentOutputCollector.load_jsonl(Path(jsonl_path))
    print(f"  {len(samples)} samples loaded  |  seeds: {seeds}")

    start = time.time()
    all_results: List[Dict[str, float]] = []
    all_extras: List[Dict[str, Any]] = []

    for seed in seeds:
        t0 = time.time()
        res, extras = _run_seed(samples, seed, trust_scores, split_cfg)
        all_results.append(res)
        all_extras.append(extras)
        elapsed = time.time() - t0
        daac_gbm = res.get("daac_gradient_boosting", float("nan"))
        print(f"  seed={seed}: daac_gbm={daac_gbm:.4f}  ({elapsed:.1f}s)")

    # Aggregate macro-F1
    all_keys = list(all_results[0].keys())
    agg: Dict[str, Dict] = {}
    for key in all_keys:
        runs = [r[key] for r in all_results]
        agg[key] = {
            "mean": float(np.mean(runs)),
            "std":  float(np.std(runs, ddof=1)),
            "runs": runs,
        }

    # Wilcoxon p-values (vs COBRA)
    pvalues = _compute_wilcoxon(all_results, reference="cobra")

    # Print tables
    table1 = _print_table1(agg, pvalues)
    table2 = _print_perclass_table(all_extras)
    table3 = _print_table3(agg)
    table4 = _print_feature_importance(all_extras)

    total_elapsed = time.time() - start
    print(f"\n총 실행 시간: {total_elapsed:.1f}s")

    # Save
    save_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = save_dir / f"paper_final_{label}_{ts}.json"

    # Serialize extras (convert perclass nested dicts and feature_importances lists)
    payload = {
        "timestamp": datetime.now().isoformat(),
        "label": label,
        "config_path": config_path,
        "jsonl_path": jsonl_path,
        "n_samples": len(samples),
        "seeds": seeds,
        "trust_scores": trust_scores,
        "results_per_seed": all_results,
        "aggregated": agg,
        "pvalues_vs_cobra": pvalues,
        "extras_per_seed": all_extras,
        "tables": {
            "table1_methods": table1,
            "table2_perclass": table2,
            "table3_ablation": table3,
            "table4_feature_importance": table4,
        },
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n결과 저장: {out_path}")


if __name__ == "__main__":
    main()
