#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR / "data" / "experiments" / "results" / "repeat_arv"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(range(10)),
        help="반복 실험에 사용할 random seed 목록",
    )
    return parser.parse_args()


def seeded_train_veto_model(x_tr, y_tr, sample_weight, model_key: str, seed: int):
    if len(x_tr) == 0:
        from types import SimpleNamespace

        class ConstantKeepModel:
            def __init__(self, prob_keep: float):
                self.prob_keep = prob_keep

            def predict_proba(self, x):
                p = np.full(len(x), self.prob_keep, dtype=np.float32)
                return np.stack([1.0 - p, p], axis=1)

        return ConstantKeepModel(prob_keep=1.0)

    uniq = sorted(set(np.asarray(y_tr).tolist()))
    if len(uniq) == 1:
        class ConstantKeepModel:
            def __init__(self, prob_keep: float):
                self.prob_keep = prob_keep

            def predict_proba(self, x):
                p = np.full(len(x), self.prob_keep, dtype=np.float32)
                return np.stack([1.0 - p, p], axis=1)

        return ConstantKeepModel(prob_keep=float(uniq[0]))

    if model_key == "logreg":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=seed)),
        ])
        clf.fit(x_tr, y_tr, clf__sample_weight=sample_weight)
        return clf

    if model_key in {"xgb_stump", "xgb_depth2", "xgb_meta_depth2"}:
        import xgboost as xgb

        max_depth = 1 if model_key == "xgb_stump" else 2
        clf = xgb.XGBClassifier(
            n_estimators=120,
            max_depth=max_depth,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=3.0,
            min_child_weight=4,
            eval_metric="logloss",
            objective="binary:logistic",
            tree_method="hist",
            random_state=seed,
            verbosity=0,
            n_jobs=1,
        )
        clf.fit(x_tr, y_tr, sample_weight=sample_weight)
        return clf

    raise ValueError(f"Unknown veto model: {model_key}")


def seeded_train_richer_model(x_tr, y_tr, sample_weight, seed: int):
    try:
        import xgboost as xgb

        clf = xgb.XGBClassifier(
            n_estimators=60,
            max_depth=2,
            learning_rate=0.07,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=3.0,
            min_child_weight=2,
            eval_metric="logloss",
            objective="binary:logistic",
            tree_method="hist",
            random_state=seed,
            verbosity=0,
            n_jobs=1,
        )
        clf.fit(x_tr, y_tr, sample_weight=sample_weight)
        return clf
    except Exception:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=500, random_state=seed)),
        ])
        clf.fit(x_tr, y_tr, clf__sample_weight=sample_weight)
        return clf


def collect_totals(per_ds: Dict) -> Dict[str, int]:
    n_corrected = 0
    n_broken = 0
    n_errors = 0
    net_gain = 0
    for ds_name, res in per_ds.items():
        if not isinstance(res, dict) or "err_corr" not in res:
            continue
        err = res["err_corr"]
        n_corrected += int(err["n_corrected"])
        n_broken += int(err["n_broken"])
        n_errors += int(err["n_errors"])
        net_gain += int(err["net_gain"])
    return {
        "n_corrected": n_corrected,
        "n_broken": n_broken,
        "n_errors": n_errors,
        "net_gain": net_gain,
    }


def summarize_numeric(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": round(float(arr.mean()), 4),
        "std": round(float(arr.std(ddof=0)), 4),
        "min": round(float(arr.min()), 4),
        "max": round(float(arr.max()), 4),
    }


def run_mnv2_strong(seed: int) -> Dict:
    module = load_module(
        f"arv_exp_seed_{seed}",
        SCRIPT_DIR / "run_arv_experiment.py",
    )
    hema = module.load_hema_module()
    hema.train_veto_model = lambda x_tr, y_tr, w_tr, model_key: seeded_train_veto_model(
        x_tr, y_tr, w_tr, model_key, seed
    )

    scalar = json.loads(Path(module.DEFAULT_SCALAR_RESULT).read_text(encoding="utf-8"))
    mnv2_label = "strong"
    model_key = "comp_noTS"
    scalar_models = scalar["results"][mnv2_label]["models"]
    mnv2_ts = scalar["config"]["mnv2_versions"][mnv2_label]

    aligned = {}
    for ds_name in hema.DATASETS:
        mnv2_recs = hema.load_mnv2(ds_name, mnv2_ts)
        specm_recs = hema.load_specm(model_key, ds_name)
        aligned[(ds_name, model_key)] = hema.align_records(mnv2_recs, specm_recs)

    baseline_f1s = []
    plain_f1s = []
    plain_corrs = []
    plain_gains = []
    for ds_name in hema.DATASETS:
        mnv2_recs, specm_recs = aligned[(ds_name, model_key)]
        baseline_preds = np.asarray([hema.CLS2IDX[r["pred_label"]] for r in mnv2_recs], dtype=np.int64)
        baseline_res = hema.eval_preds(baseline_preds, mnv2_recs, {"mnv2_only": len(mnv2_recs)})
        baseline_f1s.append(baseline_res["macro_f1"])

        plain_preds = np.asarray([hema.icwmv_single(m, s) for m, s in zip(mnv2_recs, specm_recs)], dtype=np.int64)
        plain_res = hema.eval_preds(plain_preds, mnv2_recs, {"icwmv_comp_noTS": len(mnv2_recs)})
        plain_f1s.append(plain_res["macro_f1"])
        plain_corrs.append(plain_res["err_corr"]["rate"])
        plain_gains.append(plain_res["err_corr"]["net_gain"])

    f1s, corrs, gains = [], [], []
    richer_per_ds = {}
    for test_ds in hema.DATASETS:
        base_cfg = scalar_models[model_key]["hema_icwmv_veto"]["per_ds"][test_ds]["best_cfg"]
        train_dss = [ds for ds in hema.DATASETS if ds != test_ds]
        best_grid_cfg, _ = module.tune_richer_grid(
            hema,
            aligned,
            train_dss,
            scalar_models,
            model_key=model_key,
            reverse_auth_weight=2.0,
            reverse_manip_grid=[4.0, 6.0, 8.0],
            non_casia_grid=[1.0, 1.5, 2.0],
            tau_deltas=[-0.05, 0.0, 0.05],
        )
        x_tr, y_tr, w_tr = module.richer_override_dataset(
            hema,
            aligned,
            train_dss,
            model_key=model_key,
            pos_weight=float(base_cfg["pos_weight"]),
            reverse_manip_weight=float(best_grid_cfg["reverse_manip_weight"]),
            reverse_auth_weight=2.0,
            non_casia_harm_weight=float(best_grid_cfg["non_casia_harm_weight"]),
        )
        veto_model = hema.train_veto_model(x_tr, y_tr, w_tr, "xgb_depth2")
        mnv2_test, specm_test = aligned[(test_ds, model_key)]
        preds, actions = module.apply_richer_veto(
            hema,
            mnv2_test,
            specm_test,
            veto_model,
            tau=module.clipped_tau(float(base_cfg["tau"]), float(best_grid_cfg["tau_delta"])),
        )
        res = hema.eval_preds(preds, mnv2_test, actions)
        richer_per_ds[test_ds] = res
        f1s.append(res["macro_f1"])
        corrs.append(res["err_corr"]["rate"])
        gains.append(res["err_corr"]["net_gain"])

    plain = scalar_models[model_key]["icwmv"]
    return {
        "seed": seed,
        "label": "MNV2 strong",
        "baseline_avg_f1": round(float(np.mean(baseline_f1s)), 4),
        "plain_icwmv": {
            "avg_f1": round(float(np.mean(plain_f1s)), 4),
            "avg_corr": round(float(np.mean(plain_corrs)), 4),
            "avg_net_gain": round(float(np.mean(plain_gains)), 4),
            "totals": collect_totals(plain["per_ds"]),
        },
        "arv": {
            "avg_f1": round(float(np.mean(f1s)), 4),
            "avg_corr": round(float(np.mean(corrs)), 4),
            "avg_net_gain": round(float(np.mean(gains)), 4),
            "totals": collect_totals(richer_per_ds),
            "per_ds": richer_per_ds,
        },
    }


def run_clip_ft4_strong(seed: int) -> Dict:
    clip_mod = load_module(
        f"arv_generalist_seed_{seed}",
        SCRIPT_DIR / "run_arv_generalist.py",
    )
    helper = clip_mod.load_helper_module()
    clip_mod.train_richer_model = lambda x_tr, y_tr, w_tr: seeded_train_richer_model(
        x_tr, y_tr, w_tr, seed
    )

    args = SimpleNamespace(
        backbones=["clip_ft4_strong"],
        pos_weight_grid=[1.0, 2.0],
        reverse_manip_grid=[4.0, 6.0],
        reverse_auth_grid=[1.5],
        tau_grid=[0.45, 0.55, 0.65],
    )
    res = clip_mod.evaluate_backbone(helper, "clip_ft4_strong", args)
    return {
        "seed": seed,
        "label": res["label"],
        "baseline_avg_f1": float(res["baseline"]["avg_f1"]),
        "plain_icwmv": {
            "avg_f1": float(res["plain_icwmv_comp_noTS"]["avg_f1"]),
            "avg_corr": float(res["plain_icwmv_comp_noTS"]["avg_corr"]),
            "avg_net_gain": round(
                float(np.mean([v["err_corr"]["net_gain"] for k, v in res["plain_icwmv_comp_noTS"].items() if isinstance(v, dict)])),
                4,
            ),
            "totals": collect_totals(res["plain_icwmv_comp_noTS"]),
        },
        "arv": {
            "avg_f1": float(res["richer_veto"]["avg_f1"]),
            "avg_corr": float(res["richer_veto"]["avg_corr"]),
            "avg_net_gain": float(res["richer_veto"]["avg_net_gain"]),
            "totals": collect_totals(res["richer_veto"]["per_ds"]),
            "per_ds": res["richer_veto"]["per_ds"],
        },
    }


def main() -> None:
    args = parse_args()
    seeds = [int(s) for s in args.seeds]
    results = {
        "timestamp": datetime.now().isoformat(),
        "experiment": "arv_strong_backbone_repeats",
        "seeds": seeds,
        "backbones": {
            "mnv2_strong": [],
            "clip_ft4_strong": [],
        },
    }

    for seed in seeds:
        np.random.seed(seed)
        print(f"\n[seed={seed}] MNV2 strong")
        mn = run_mnv2_strong(seed)
        results["backbones"]["mnv2_strong"].append(mn)
        print(
            f"  plain broken={mn['plain_icwmv']['totals']['n_broken']} "
            f"arv broken={mn['arv']['totals']['n_broken']} "
            f"plain f1={mn['plain_icwmv']['avg_f1']:.4f} "
            f"arv f1={mn['arv']['avg_f1']:.4f}"
        )

        print(f"[seed={seed}] MobileCLIP-ft4 strong")
        clip = run_clip_ft4_strong(seed)
        results["backbones"]["clip_ft4_strong"].append(clip)
        print(
            f"  plain broken={clip['plain_icwmv']['totals']['n_broken']} "
            f"arv broken={clip['arv']['totals']['n_broken']} "
            f"plain f1={clip['plain_icwmv']['avg_f1']:.4f} "
            f"arv f1={clip['arv']['avg_f1']:.4f}"
        )

    summary = {}
    for key, rows in results["backbones"].items():
        plain_broken = [r["plain_icwmv"]["totals"]["n_broken"] for r in rows]
        arv_broken = [r["arv"]["totals"]["n_broken"] for r in rows]
        plain_f1 = [r["plain_icwmv"]["avg_f1"] for r in rows]
        arv_f1 = [r["arv"]["avg_f1"] for r in rows]
        plain_net = [r["plain_icwmv"]["totals"]["net_gain"] for r in rows]
        arv_net = [r["arv"]["totals"]["net_gain"] for r in rows]
        summary[key] = {
            "plain_broken": summarize_numeric(plain_broken),
            "arv_broken": summarize_numeric(arv_broken),
            "plain_f1": summarize_numeric(plain_f1),
            "arv_f1": summarize_numeric(arv_f1),
            "plain_net_gain": summarize_numeric(plain_net),
            "arv_net_gain": summarize_numeric(arv_net),
        }

    pooled_plain_broken = []
    pooled_arv_broken = []
    pooled_plain_net = []
    pooled_arv_net = []
    for i in range(len(seeds)):
        mn = results["backbones"]["mnv2_strong"][i]
        cl = results["backbones"]["clip_ft4_strong"][i]
        pooled_plain_broken.append(
            mn["plain_icwmv"]["totals"]["n_broken"] + cl["plain_icwmv"]["totals"]["n_broken"]
        )
        pooled_arv_broken.append(
            mn["arv"]["totals"]["n_broken"] + cl["arv"]["totals"]["n_broken"]
        )
        pooled_plain_net.append(
            mn["plain_icwmv"]["totals"]["net_gain"] + cl["plain_icwmv"]["totals"]["net_gain"]
        )
        pooled_arv_net.append(
            mn["arv"]["totals"]["net_gain"] + cl["arv"]["totals"]["net_gain"]
        )

    reduction_pct = [
        100.0 * (1.0 - a / max(p, 1))
        for p, a in zip(pooled_plain_broken, pooled_arv_broken)
    ]
    summary["pooled_strong"] = {
        "plain_broken": summarize_numeric(pooled_plain_broken),
        "arv_broken": summarize_numeric(pooled_arv_broken),
        "plain_net_gain": summarize_numeric(pooled_plain_net),
        "arv_net_gain": summarize_numeric(pooled_arv_net),
        "reverse_reduction_pct": summarize_numeric(reduction_pct),
    }

    results["summary"] = summary

    out_path = OUT_DIR / f"arv_strong_backbone_repeats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n저장: {out_path}")


if __name__ == "__main__":
    main()
