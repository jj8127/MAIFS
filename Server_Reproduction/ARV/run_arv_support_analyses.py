#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR / "data"
REPO_ROOT = SCRIPT_DIR.parents[1]
OUT_DIR = ROOT / "experiments" / "results" / "paper_support"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SCALAR_RESULT = (
    ROOT
    / "experiments"
    / "results"
    / "hema_icwmv_veto"
    / "hema_icwmv_veto_loo_cd_20260323_114321.json"
)
BEST_RICHER_RESULT = (
    ROOT
    / "experiments"
    / "results"
    / "hema_icwmv_veto"
    / "comp_nots_richer_veto_20260325_084631.json"
)
REPEAT_RESULT = (
    ROOT
    / "experiments"
    / "results"
    / "repeat_arv"
    / "arv_strong_backbone_repeats_20260328_124619.json"
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


HEMA = load_module("hema_support", ROOT / "experiments" / "run_hema_icwmv_veto_loo_cd.py")
ARV = load_module("arv_support_exp", SCRIPT_DIR / "run_arv_experiment.py")
ARV_RUNTIME = load_module(
    "arv_stage2_runtime_support",
    REPO_ROOT
    / "Raspberry_pi5_Experiment"
    / "ARV_EndToEnd_RPi5"
    / "common"
    / "scripts"
    / "arv_stage2_runtime.py",
)

DATASETS = HEMA.DATASETS
CLS2IDX = HEMA.CLS2IDX
IDX2CLS = HEMA.IDX2CLS
MNV2_LABEL = "strong"
MODEL_KEY = "comp_noTS"
REVERSE_AUTH_WEIGHT = 2.0
NON_CASIA_HARM_WEIGHT = 1.5


def macro_f1(preds: np.ndarray, labels: np.ndarray) -> float:
    present = sorted(set(labels.tolist()))
    f1s = []
    for cls_idx in present:
        tp = int(((preds == cls_idx) & (labels == cls_idx)).sum())
        fp = int(((preds == cls_idx) & (labels != cls_idx)).sum())
        fn = int(((preds != cls_idx) & (labels == cls_idx)).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1s.append(2 * precision * recall / max(precision + recall, 1e-8))
    return float(np.mean(f1s))


def bootstrap_delta_ci(
    labels: np.ndarray,
    preds_a: np.ndarray,
    preds_b: np.ndarray,
    n_boot: int = 2000,
    seed: int = 42,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(labels)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        deltas.append(macro_f1(preds_b[idx], labels[idx]) - macro_f1(preds_a[idx], labels[idx]))
    arr = np.asarray(deltas, dtype=np.float64)
    return {
        "mean": round(float(arr.mean()), 4),
        "ci95_low": round(float(np.quantile(arr, 0.025)), 4),
        "ci95_high": round(float(np.quantile(arr, 0.975)), 4),
    }


def mcnemar_exact_pvalue(correct_a: np.ndarray, correct_b: np.ndarray) -> Dict[str, float | int]:
    b = int(((correct_a == 1) & (correct_b == 0)).sum())
    c = int(((correct_a == 0) & (correct_b == 1)).sum())
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "pvalue": 1.0}
    k = min(b, c)
    prob = 0.0
    for i in range(k + 1):
        prob += math.comb(n, i) * (0.5 ** n)
    pvalue = min(1.0, 2.0 * prob)
    return {"b": b, "c": c, "pvalue": round(float(pvalue), 6)}


def summarize_seed_series(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": round(float(arr.mean()), 4),
        "std": round(float(arr.std(ddof=0)), 4),
        "min": round(float(arr.min()), 4),
        "max": round(float(arr.max()), 4),
    }


def simple_average_fuse_single(m: Dict, s: Dict) -> int:
    if HEMA.is_ai_lock(m):
        return CLS2IDX["ai_generated"]
    p_auth_bin, p_manip_bin = HEMA.mnv2_binary_probs(m)
    auth = 0.5 * (p_auth_bin + float(s["authentic_score"]))
    manip = 0.5 * (p_manip_bin + float(s["manip_score"]))
    aigen = float(m["scores"].get("ai_generated", 0.0))
    raw = np.array([auth, manip, aigen], dtype=np.float32)
    raw = raw / max(float(raw.sum()), 1e-8)
    return int(np.argmax(raw))


def stack_feature(m: Dict, s: Dict) -> List[float]:
    p_auth = float(m["scores"]["authentic"])
    p_manip = float(m["scores"]["manipulated"])
    p_ai = float(m["scores"].get("ai_generated", 0.0))
    p_auth_bin, p_manip_bin = HEMA.mnv2_binary_probs(m)
    specm_a = float(s["authentic_score"])
    specm_m = float(s["manip_score"])
    return [
        p_auth,
        p_manip,
        p_ai,
        p_auth_bin,
        p_manip_bin,
        specm_a,
        specm_m,
        float(m["confidence"]),
        float(s["confidence"]),
        float(p_manip_bin - p_auth_bin),
        float(specm_m - specm_a),
    ]


def train_logistic_stacking(
    train_pairs: List[Tuple[List[Dict], List[Dict]]],
    c_value: float = 1.0,
):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    xs, ys = [], []
    for mnv2_recs, specm_recs in train_pairs:
        for m, s in zip(mnv2_recs, specm_recs):
            xs.append(stack_feature(m, s))
            ys.append(CLS2IDX[m["true_label"]])
    x_tr = np.asarray(xs, dtype=np.float32)
    y_tr = np.asarray(ys, dtype=np.int64)
    clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=float(c_value),
                    max_iter=1000,
                    multi_class="multinomial",
                    random_state=42,
                ),
            ),
        ]
    )
    clf.fit(x_tr, y_tr)
    return clf


def stacking_predict(model, mnv2_recs: List[Dict], specm_recs: List[Dict]) -> np.ndarray:
    x = np.asarray([stack_feature(m, s) for m, s in zip(mnv2_recs, specm_recs)], dtype=np.float32)
    return model.predict(x).astype(np.int64)


def collect_totals(per_ds: Dict[str, Dict]) -> Dict[str, int]:
    return {
        "n_errors": int(sum(per_ds[ds]["err_corr"]["n_errors"] for ds in DATASETS)),
        "n_corrected": int(sum(per_ds[ds]["err_corr"]["n_corrected"] for ds in DATASETS)),
        "n_broken": int(sum(per_ds[ds]["err_corr"]["n_broken"] for ds in DATASETS)),
        "net_gain": int(sum(per_ds[ds]["err_corr"]["net_gain"] for ds in DATASETS)),
    }


def aggregate_metric(per_ds: Dict[str, Dict], key: str) -> float:
    return round(float(np.mean([per_ds[ds][key] for ds in DATASETS])), 4)


def aggregate_corr(per_ds: Dict[str, Dict]) -> float:
    return round(float(np.mean([per_ds[ds]["err_corr"]["rate"] for ds in DATASETS])), 4)


def load_aligned() -> Tuple[Dict[Tuple[str, str], Tuple[List[Dict], List[Dict]]], str]:
    scalar = json.loads(SCALAR_RESULT.read_text(encoding="utf-8"))
    mnv2_ts = scalar["config"]["mnv2_versions"][MNV2_LABEL]
    aligned = {}
    for ds_name in DATASETS:
        mnv2_recs = HEMA.load_mnv2(ds_name, mnv2_ts)
        specm_recs = HEMA.load_specm(MODEL_KEY, ds_name)
        aligned[(ds_name, MODEL_KEY)] = HEMA.align_records(mnv2_recs, specm_recs)
    return aligned, mnv2_ts


def deterministic_arv_results(
    aligned: Dict[Tuple[str, str], Tuple[List[Dict], List[Dict]]],
    reverse_manip_weight: float = 6.0,
    non_casia_harm_weight: float = 1.5,
) -> Dict[str, Any]:
    per_ds = {}
    preds_store = {}
    labels_store = {}
    runtime = ARV_RUNTIME.ARVStage2Runtime()
    for test_ds in DATASETS:
        mnv2_test, specm_test = aligned[(test_ds, MODEL_KEY)]
        preds = []
        actions: Dict[str, int] = defaultdict(int)
        print(f"  - deterministic {test_ds}: runtime inference on {len(mnv2_test)} samples", flush=True)
        for m, s in zip(mnv2_test, specm_test):
            decision = runtime.decide(
                model_key=test_ds,
                base_scores3=m["scores"],
                aux_scores2={
                    "authentic": float(s["authentic_score"]),
                    "manipulated": float(s["manip_score"]),
                },
                base_conf=float(m["confidence"]),
                aux_conf=float(s["confidence"]),
                sub_type=str(m.get("sub_type", "")),
            )
            preds.append(CLS2IDX[decision.final_label])
            actions[decision.action] += 1
        preds = np.asarray(preds, dtype=np.int64)
        res = HEMA.eval_preds(preds, mnv2_test, actions)
        res["cfg"] = runtime.model_meta(test_ds)
        per_ds[test_ds] = res
        preds_store[test_ds] = preds
        labels_store[test_ds] = np.asarray([CLS2IDX[m["true_label"]] for m in mnv2_test], dtype=np.int64)
    return {
        "per_ds": per_ds,
        "avg_f1": aggregate_metric(per_ds, "macro_f1"),
        "avg_corr": aggregate_corr(per_ds),
        "avg_net_gain": round(float(np.mean([per_ds[ds]["err_corr"]["net_gain"] for ds in DATASETS])), 4),
        "totals": collect_totals(per_ds),
        "preds": preds_store,
        "labels": labels_store,
    }


def base_and_stage1_results(
    aligned: Dict[Tuple[str, str], Tuple[List[Dict], List[Dict]]]
) -> Dict[str, Any]:
    base_per_ds = {}
    stage1_per_ds = {}
    avg_ens_per_ds = {}
    preds = {"base": {}, "stage1": {}, "avg_ensemble": {}}
    labels_store = {}

    for ds in DATASETS:
        mnv2_recs, specm_recs = aligned[(ds, MODEL_KEY)]
        base_preds = np.asarray([CLS2IDX[m["pred_label"]] for m in mnv2_recs], dtype=np.int64)
        stage1_preds = np.asarray([HEMA.icwmv_single(m, s) for m, s in zip(mnv2_recs, specm_recs)], dtype=np.int64)
        avg_preds = np.asarray([simple_average_fuse_single(m, s) for m, s in zip(mnv2_recs, specm_recs)], dtype=np.int64)

        base_per_ds[ds] = HEMA.eval_preds(base_preds, mnv2_recs, {"mnv2_only": len(mnv2_recs)})
        stage1_per_ds[ds] = HEMA.eval_preds(stage1_preds, mnv2_recs, {"icwmv_comp_noTS": len(mnv2_recs)})
        avg_ens_per_ds[ds] = HEMA.eval_preds(avg_preds, mnv2_recs, {"simple_avg_ensemble": len(mnv2_recs)})

        preds["base"][ds] = base_preds
        preds["stage1"][ds] = stage1_preds
        preds["avg_ensemble"][ds] = avg_preds
        labels_store[ds] = np.asarray([CLS2IDX[m["true_label"]] for m in mnv2_recs], dtype=np.int64)

    def pack(per_ds: Dict[str, Dict]) -> Dict[str, Any]:
        return {
            "per_ds": per_ds,
            "avg_f1": aggregate_metric(per_ds, "macro_f1"),
            "avg_corr": aggregate_corr(per_ds),
            "avg_net_gain": round(float(np.mean([per_ds[ds]["err_corr"]["net_gain"] for ds in DATASETS])), 4),
            "totals": collect_totals(per_ds),
        }

    return {
        "base": pack(base_per_ds),
        "stage1": pack(stage1_per_ds),
        "avg_ensemble": pack(avg_ens_per_ds),
        "preds": preds,
        "labels": labels_store,
    }


def logistic_stacking_results(
    aligned: Dict[Tuple[str, str], Tuple[List[Dict], List[Dict]]]
) -> Dict[str, Any]:
    per_ds = {}
    preds_store = {}
    for test_ds in DATASETS:
        train_dss = [ds for ds in DATASETS if ds != test_ds]
        train_pairs = [aligned[(ds, MODEL_KEY)] for ds in train_dss]
        model = train_logistic_stacking(train_pairs, c_value=1.0)
        mnv2_test, specm_test = aligned[(test_ds, MODEL_KEY)]
        preds = stacking_predict(model, mnv2_test, specm_test)
        per_ds[test_ds] = HEMA.eval_preds(preds, mnv2_test, {"logistic_stacking": len(mnv2_test)})
        preds_store[test_ds] = preds
    return {
        "per_ds": per_ds,
        "avg_f1": aggregate_metric(per_ds, "macro_f1"),
        "avg_corr": aggregate_corr(per_ds),
        "avg_net_gain": round(float(np.mean([per_ds[ds]["err_corr"]["net_gain"] for ds in DATASETS])), 4),
        "totals": collect_totals(per_ds),
        "preds": preds_store,
    }


def pooled_arrays(preds_per_ds: Dict[str, np.ndarray], labels_per_ds: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    preds = np.concatenate([preds_per_ds[ds] for ds in DATASETS], axis=0)
    labels = np.concatenate([labels_per_ds[ds] for ds in DATASETS], axis=0)
    return preds, labels


def main_result_statistics(
    base_preds: Dict[str, np.ndarray],
    arv_preds: Dict[str, np.ndarray],
    labels_per_ds: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    preds_base, labels = pooled_arrays(base_preds, labels_per_ds)
    preds_arv, _ = pooled_arrays(arv_preds, labels_per_ds)
    correct_base = (preds_base == labels).astype(np.int64)
    correct_arv = (preds_arv == labels).astype(np.int64)
    return {
        "macro_f1": {
            "base": round(macro_f1(preds_base, labels), 4),
            "arv": round(macro_f1(preds_arv, labels), 4),
            "delta": round(macro_f1(preds_arv, labels) - macro_f1(preds_base, labels), 4),
            "bootstrap_delta_ci": bootstrap_delta_ci(labels, preds_base, preds_arv),
        },
        "mcnemar_base_vs_arv": mcnemar_exact_pvalue(correct_base, correct_arv),
        "n_samples_pooled": int(len(labels)),
    }


def repeated_stats_summary() -> Dict[str, Any]:
    repeat = json.loads(REPEAT_RESULT.read_text(encoding="utf-8"))
    mnv2_rows = repeat["backbones"]["mnv2_strong"]
    baseline_f1 = [row["baseline_avg_f1"] for row in mnv2_rows]
    stage1_f1 = [row["plain_icwmv"]["avg_f1"] for row in mnv2_rows]
    arv_f1 = [row["arv"]["avg_f1"] for row in mnv2_rows]
    arv_broken = [row["arv"]["totals"]["n_broken"] for row in mnv2_rows]
    arv_gain = [row["arv"]["totals"]["net_gain"] for row in mnv2_rows]
    return {
        "mnv2_strong_10seed": {
            "baseline_avg_f1": summarize_seed_series(baseline_f1),
            "stage1_avg_f1": summarize_seed_series(stage1_f1),
            "arv_avg_f1": summarize_seed_series(arv_f1),
            "arv_broken_total": summarize_seed_series(arv_broken),
            "arv_net_gain_total": summarize_seed_series(arv_gain),
        }
    }


def feature_importance_summary(
    aligned: Dict[Tuple[str, str], Tuple[List[Dict], List[Dict]]]
) -> Dict[str, Any]:
    feature_names = [
        "p_auth_bin",
        "p_manip_bin",
        "specm_auth",
        "specm_manip",
        "ic_auth",
        "ic_manip",
        "ic_margin_signed",
        "specm_minus_pmanip",
        "specm_minus_pauth",
        "ai_margin",
        "base_conf",
        "aux_conf",
        "conf_ratio",
        "base_bin_pred",
        "stage1_bin_pred",
        "specm_base_abs_gap",
    ] + [f"subtype_{name}" for name in HEMA.SUBTYPE_VOCAB] + [
        "family_casia",
        "family_biggan",
        "family_inpaint",
        "family_opensdi",
        "family_aigenproxy",
        "family_empty",
        "mnv2_margin",
        "specm_margin",
        "ic_margin_abs",
        "specm_margin_minus_mnv2_margin",
        "conf_gap",
        "conf_product",
        "base_pred_auth",
        "base_pred_manip",
        "stage1_pred_auth",
        "stage1_pred_manip",
        "toward_auth",
        "toward_manip",
        "non_casia_flag",
        "ood_manip_family_flag",
    ]

    runtime = ARV_RUNTIME.ARVStage2Runtime()
    agg_gain = defaultdict(float)
    model_tables = {}
    for test_ds in DATASETS:
        booster = runtime.load_model(test_ds)
        fmap = {f"f{i}": feature_names[i] for i in range(len(feature_names))}
        score = booster.get_score(importance_type="gain")
        named = {fmap.get(k, k): float(v) for k, v in score.items()}
        total = sum(named.values()) or 1.0
        norm = {k: v / total for k, v in named.items()}
        for k, v in norm.items():
            agg_gain[k] += v
        model_tables[test_ds] = dict(sorted(norm.items(), key=lambda kv: kv[1], reverse=True)[:10])
    avg = {k: v / len(DATASETS) for k, v in agg_gain.items()}
    top_avg = dict(sorted(avg.items(), key=lambda kv: kv[1], reverse=True)[:10])
    return {"top_average_gain": top_avg, "per_dataset_top10": model_tables}


def train_fast_veto_model(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    sample_weight: np.ndarray,
):
    if len(x_tr) == 0:
        return HEMA.ConstantKeepModel(prob_keep=1.0)
    uniq = sorted(set(y_tr.tolist()))
    if len(uniq) == 1:
        return HEMA.ConstantKeepModel(prob_keep=float(uniq[0]))

    import xgboost as xgb

    clf = xgb.XGBClassifier(
        n_estimators=80,
        max_depth=2,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=3.0,
        min_child_weight=4,
        eval_metric="logloss",
        objective="binary:logistic",
        tree_method="hist",
        random_state=42,
        n_jobs=1,
        verbosity=0,
    )
    clf.fit(x_tr, y_tr, sample_weight=sample_weight)
    return clf


def cost_sensitivity(
    aligned: Dict[Tuple[str, str], Tuple[List[Dict], List[Dict]]]
) -> Dict[str, Any]:
    scalar = json.loads(SCALAR_RESULT.read_text(encoding="utf-8"))
    best_ref = json.loads(BEST_RICHER_RESULT.read_text(encoding="utf-8"))
    rows = []
    for ratio, reverse_manip_weight in [("2:1", 4.0), ("3:1", 6.0), ("4:1", 8.0), ("6:1", 12.0)]:
        per_ds = {}
        for test_ds in DATASETS:
            base_cfg = scalar["results"][MNV2_LABEL]["models"][MODEL_KEY]["hema_icwmv_veto"]["per_ds"][test_ds]["best_cfg"]
            richer_cfg = best_ref["richer_veto"]["per_ds"][test_ds]["cfg"]
            train_dss = [ds for ds in DATASETS if ds != test_ds]
            x_tr, y_tr, w_tr = ARV.richer_override_dataset(
                HEMA,
                aligned,
                train_dss,
                model_key=MODEL_KEY,
                pos_weight=float(base_cfg["pos_weight"]),
                reverse_manip_weight=float(reverse_manip_weight),
                reverse_auth_weight=REVERSE_AUTH_WEIGHT,
                non_casia_harm_weight=NON_CASIA_HARM_WEIGHT,
            )
            veto_model = train_fast_veto_model(x_tr, y_tr, w_tr)
            mnv2_test, specm_test = aligned[(test_ds, MODEL_KEY)]
            preds, actions = ARV.apply_richer_veto(
                HEMA,
                mnv2_test,
                specm_test,
                veto_model,
                tau=float(richer_cfg["tau"]),
            )
            per_ds[test_ds] = HEMA.eval_preds(preds, mnv2_test, actions)
        rows.append(
            {
                "ratio": ratio,
                "reverse_manip_weight": reverse_manip_weight,
                "reverse_auth_weight": REVERSE_AUTH_WEIGHT,
                "avg_f1": aggregate_metric(per_ds, "macro_f1"),
                "avg_corr": aggregate_corr(per_ds),
                "avg_net_gain": round(float(np.mean([per_ds[ds]["err_corr"]["net_gain"] for ds in DATASETS])), 4),
                "total_broken": int(sum(per_ds[ds]["err_corr"]["n_broken"] for ds in DATASETS)),
                "opensdi_broken": int(per_ds["opensdi"]["err_corr"]["n_broken"]),
            }
        )
    return {"rows": rows}


def aigenproxy_failure_analysis(
    stage1_res: Dict[str, Any],
    arv_res: Dict[str, Any],
) -> Dict[str, Any]:
    s = stage1_res["per_ds"]["aigenproxy"]["err_corr"]
    a = arv_res["per_ds"]["aigenproxy"]["err_corr"]
    return {
        "stage1": s,
        "arv": a,
        "interpretation": {
            "lost_net_gain": int(s["net_gain"] - a["net_gain"]),
            "lost_auth_to_manip_corrections": int(
                s["patterns"].get("authentic→manipulated", {}).get("corrected", 0)
                - a["patterns"].get("authentic→manipulated", {}).get("corrected", 0)
            ),
            "lost_manip_to_auth_corrections": int(
                s["patterns"].get("manipulated→authentic", {}).get("corrected", 0)
                - a["patterns"].get("manipulated→authentic", {}).get("corrected", 0)
            ),
            "broken_reduction": int(s["n_broken"] - a["n_broken"]),
        },
    }


def main() -> None:
    print("[1/7] loading aligned records...", flush=True)
    aligned, mnv2_ts = load_aligned()
    print("[2/7] deterministic ARV results...", flush=True)
    deterministic = deterministic_arv_results(aligned)
    print("[3/7] base/stage1/simple-average baselines...", flush=True)
    base_stage1 = base_and_stage1_results(aligned)
    print("[4/7] logistic stacking baseline...", flush=True)
    stacking = logistic_stacking_results(aligned)

    print("[5/7] statistics + repeat summary...", flush=True)
    stats = main_result_statistics(
        base_preds=base_stage1["preds"]["base"],
        arv_preds=deterministic["preds"],
        labels_per_ds=base_stage1["labels"],
    )
    repeat = repeated_stats_summary()
    print("[6/7] feature importance...", flush=True)
    importance = feature_importance_summary(aligned)
    print("[7/7] cost sensitivity + aigen analysis...", flush=True)
    cost = cost_sensitivity(aligned)
    aigen = aigenproxy_failure_analysis(base_stage1["stage1"], deterministic)

    out = {
        "timestamp": datetime.now().isoformat(),
        "experiment": "arv_paper_support_analyses",
        "mnv2_label": MNV2_LABEL,
        "mnv2_ts": mnv2_ts,
        "model_key": MODEL_KEY,
        "main_repeated_stats": repeat,
        "main_result_statistics": stats,
        "external_baselines": {
            "simple_average_ensemble": {
                "avg_f1": base_stage1["avg_ensemble"]["avg_f1"],
                "avg_corr": base_stage1["avg_ensemble"]["avg_corr"],
                "avg_net_gain": base_stage1["avg_ensemble"]["avg_net_gain"],
                "total_broken": base_stage1["avg_ensemble"]["totals"]["n_broken"],
                "per_ds": base_stage1["avg_ensemble"]["per_ds"],
            },
            "logistic_stacking": {
                "avg_f1": stacking["avg_f1"],
                "avg_corr": stacking["avg_corr"],
                "avg_net_gain": stacking["avg_net_gain"],
                "total_broken": stacking["totals"]["n_broken"],
                "per_ds": stacking["per_ds"],
            },
            "stage1_reference": {
                "avg_f1": base_stage1["stage1"]["avg_f1"],
                "avg_corr": base_stage1["stage1"]["avg_corr"],
                "avg_net_gain": base_stage1["stage1"]["avg_net_gain"],
                "total_broken": base_stage1["stage1"]["totals"]["n_broken"],
            },
            "arv_reference": {
                "avg_f1": deterministic["avg_f1"],
                "avg_corr": deterministic["avg_corr"],
                "avg_net_gain": deterministic["avg_net_gain"],
                "total_broken": deterministic["totals"]["n_broken"],
            },
        },
        "aigenproxy_analysis": aigen,
        "cost_sensitivity": cost,
        "feature_importance": importance,
    }
    out_path = OUT_DIR / f"arv_paper_support_analyses_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()
