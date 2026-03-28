#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCALAR_RESULT = (
    ROOT
    / "experiments"
    / "results"
    / "hema_icwmv_veto"
    / "hema_icwmv_veto_loo_cd_20260323_114321.json"
)
DEFAULT_META_WARMSTART = (
    ROOT
    / "experiments"
    / "results"
    / "hema_icwmv_veto"
    / "hema_icwmv_veto_meta_warmstart_20260325_082654.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scalar-result",
        type=str,
        default=str(DEFAULT_SCALAR_RESULT),
    )
    parser.add_argument(
        "--meta-warmstart-result",
        type=str,
        default=str(DEFAULT_META_WARMSTART),
    )
    parser.add_argument(
        "--mnv2-label",
        type=str,
        default="strong",
    )
    parser.add_argument(
        "--reverse-manip-weight",
        type=float,
        default=6.0,
        help="Extra harmful weight for manipulated -> authentic overrides",
    )
    parser.add_argument(
        "--reverse-auth-weight",
        type=float,
        default=2.0,
        help="Extra harmful weight for authentic -> manipulated overrides",
    )
    parser.add_argument(
        "--non-casia-harm-weight",
        type=float,
        default=1.5,
        help="Multiplier for harmful non-casia overrides",
    )
    parser.add_argument(
        "--reverse-manip-grid",
        nargs="+",
        type=float,
        default=[4.0, 6.0, 8.0],
        help="Small grid for manipulated -> authentic harmful weight",
    )
    parser.add_argument(
        "--non-casia-grid",
        nargs="+",
        type=float,
        default=[1.0, 1.5, 2.0],
        help="Small grid for harmful non-casia multiplier",
    )
    parser.add_argument(
        "--tau-deltas",
        nargs="+",
        type=float,
        default=[-0.05, 0.0, 0.05],
        help="Small grid around the scalar best tau",
    )
    return parser.parse_args()


def load_hema_module():
    script_path = ROOT / "experiments" / "run_hema_icwmv_veto_loo_cd.py"
    spec = importlib.util.spec_from_file_location("hema_veto", script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["hema_veto"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def is_non_casia(sub_type: str) -> bool:
    token = str(sub_type).strip().lower()
    return bool(token) and not token.startswith("casia")


def is_ood_manip_family(sub_type: str) -> bool:
    token = str(sub_type).strip().lower()
    return token in {
        "imd2020_inpainting",
        "aigen_proxy_manipulated",
        "opensdi_partial_fake",
        "opensdi_entire_fake",
    }


def richer_feature(module, m: Dict, s: Dict) -> List[float]:
    base = module.build_veto_feature(m, s, feature_mode="meta")
    p_auth_bin, p_manip_bin = module.mnv2_binary_probs(m)
    specm_a = float(s["authentic_score"])
    specm_m = float(s["manip_score"])
    ic_auth, ic_manip = module.weighted_binary_scores(m, s)
    mnv2_bin = module.mnv2_binary_pred_idx(m)
    ic_pred = int(ic_manip >= ic_auth)
    m_conf = float(m["confidence"])
    s_conf = float(s["confidence"])
    sub_type = str(m.get("sub_type", "")).strip().lower()

    mnv2_margin = abs(p_manip_bin - p_auth_bin)
    specm_margin = abs(specm_m - specm_a)
    ic_margin = abs(ic_manip - ic_auth)
    conf_gap = s_conf - m_conf
    conf_prod = s_conf * m_conf
    toward_auth = 1.0 if mnv2_bin == module.CLS2IDX["manipulated"] and ic_pred == module.CLS2IDX["authentic"] else 0.0
    toward_manip = 1.0 if mnv2_bin == module.CLS2IDX["authentic"] and ic_pred == module.CLS2IDX["manipulated"] else 0.0

    return base + [
        mnv2_margin,
        specm_margin,
        ic_margin,
        specm_margin - mnv2_margin,
        conf_gap,
        conf_prod,
        float(mnv2_bin == module.CLS2IDX["authentic"]),
        float(mnv2_bin == module.CLS2IDX["manipulated"]),
        float(ic_pred == module.CLS2IDX["authentic"]),
        float(ic_pred == module.CLS2IDX["manipulated"]),
        toward_auth,
        toward_manip,
        float(is_non_casia(sub_type)),
        float(is_ood_manip_family(sub_type)),
    ]


def richer_override_dataset(
    module,
    aligned: Dict[Tuple[str, str], Tuple[List[Dict], List[Dict]]],
    train_dss: List[str],
    model_key: str,
    pos_weight: float,
    reverse_manip_weight: float,
    reverse_auth_weight: float,
    non_casia_harm_weight: float,
):
    feats, labels, weights = [], [], []
    for ds_name in train_dss:
        mnv2_recs, specm_recs = aligned[(ds_name, model_key)]
        for m, s in zip(mnv2_recs, specm_recs):
            if module.is_ai_lock(m):
                continue
            true_idx = module.CLS2IDX[m["true_label"]]
            if true_idx == module.CLS2IDX["ai_generated"]:
                continue

            mnv2_bin = module.mnv2_binary_pred_idx(m)
            ic_pred = int(np.argmax(module.weighted_binary_scores(m, s)))
            if ic_pred == mnv2_bin:
                continue

            keep_override = int(ic_pred == true_idx)
            sample_weight = float(pos_weight if keep_override else 1.0)
            sub_type = str(m.get("sub_type", "")).strip().lower()
            if not keep_override:
                if m["true_label"] == "manipulated" and ic_pred == module.CLS2IDX["authentic"]:
                    sample_weight *= float(reverse_manip_weight)
                elif m["true_label"] == "authentic" and ic_pred == module.CLS2IDX["manipulated"]:
                    sample_weight *= float(reverse_auth_weight)
                if is_non_casia(sub_type):
                    sample_weight *= float(non_casia_harm_weight)

            feats.append(richer_feature(module, m, s))
            labels.append(keep_override)
            weights.append(sample_weight)

    if not feats:
        return (
            np.zeros((0, 46), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.float32),
        )

    return (
        np.asarray(feats, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
        np.asarray(weights, dtype=np.float32),
    )


def richer_keep_prob(module, model, m: Dict, s: Dict) -> float:
    x = np.asarray([richer_feature(module, m, s)], dtype=np.float32)
    return float(model.predict_proba(x)[0, 1])


def apply_richer_veto(module, mnv2_recs: List[Dict], specm_recs: List[Dict], model, tau: float):
    preds = []
    actions = {}
    for m, s in zip(mnv2_recs, specm_recs):
        if module.is_ai_lock(m):
            preds.append(module.CLS2IDX["ai_generated"])
            actions["ai_lock"] = actions.get("ai_lock", 0) + 1
            continue

        mnv2_bin = module.mnv2_binary_pred_idx(m)
        ic_pred = int(np.argmax(module.weighted_binary_scores(m, s)))
        if ic_pred == mnv2_bin:
            preds.append(ic_pred)
            actions["icwmv_no_override"] = actions.get("icwmv_no_override", 0) + 1
            continue

        p_keep = richer_keep_prob(module, model, m, s)
        if p_keep >= tau:
            preds.append(ic_pred)
            actions["keep_icwmv_override"] = actions.get("keep_icwmv_override", 0) + 1
        else:
            preds.append(mnv2_bin)
            actions["revert_to_mnv2"] = actions.get("revert_to_mnv2", 0) + 1
    return np.asarray(preds, dtype=np.int64), actions


def clipped_tau(base_tau: float, tau_delta: float) -> float:
    return float(np.clip(float(base_tau) + float(tau_delta), 0.05, 0.95))


def tune_richer_grid(
    module,
    aligned: Dict[Tuple[str, str], Tuple[List[Dict], List[Dict]]],
    train_dss: List[str],
    scalar_models: Dict,
    model_key: str,
    reverse_auth_weight: float,
    reverse_manip_grid: List[float],
    non_casia_grid: List[float],
    tau_deltas: List[float],
) -> Tuple[Dict, List[Dict]]:
    candidates: List[Dict] = []
    best = None

    for reverse_manip_weight in reverse_manip_grid:
        for non_casia_harm_weight in non_casia_grid:
            for tau_delta in tau_deltas:
                inner_f1s = []
                inner_corrs = []
                inner_gains = []
                inner_broken = []

                for val_ds in train_dss:
                    base_cfg = scalar_models[model_key]["hema_icwmv_veto"]["per_ds"][val_ds]["best_cfg"]
                    inner_train = [ds for ds in train_dss if ds != val_ds]
                    x_tr, y_tr, w_tr = richer_override_dataset(
                        module,
                        aligned,
                        inner_train,
                        model_key=model_key,
                        pos_weight=float(base_cfg["pos_weight"]),
                        reverse_manip_weight=float(reverse_manip_weight),
                        reverse_auth_weight=float(reverse_auth_weight),
                        non_casia_harm_weight=float(non_casia_harm_weight),
                    )
                    veto_model = module.train_veto_model(x_tr, y_tr, w_tr, "xgb_depth2")
                    mnv2_val, specm_val = aligned[(val_ds, model_key)]
                    preds, actions = apply_richer_veto(
                        module,
                        mnv2_val,
                        specm_val,
                        veto_model,
                        tau=clipped_tau(float(base_cfg["tau"]), float(tau_delta)),
                    )
                    res = module.eval_preds(preds, mnv2_val, actions)
                    inner_f1s.append(res["macro_f1"])
                    inner_corrs.append(res["err_corr"]["rate"])
                    inner_gains.append(res["err_corr"]["net_gain"])
                    inner_broken.append(res["err_corr"]["n_broken"])

                cand = {
                    "reverse_manip_weight": float(reverse_manip_weight),
                    "non_casia_harm_weight": float(non_casia_harm_weight),
                    "tau_delta": float(tau_delta),
                    "avg_f1": float(np.mean(inner_f1s)),
                    "avg_corr": float(np.mean(inner_corrs)),
                    "avg_net_gain": float(np.mean(inner_gains)),
                    "avg_broken": float(np.mean(inner_broken)),
                }
                candidates.append(cand)
                key = (
                    cand["avg_f1"],
                    cand["avg_net_gain"],
                    -cand["avg_broken"],
                    cand["avg_corr"],
                )
                if best is None or key > (
                    best["avg_f1"],
                    best["avg_net_gain"],
                    -best["avg_broken"],
                    best["avg_corr"],
                ):
                    best = cand

    assert best is not None
    candidates_sorted = sorted(
        candidates,
        key=lambda x: (x["avg_f1"], x["avg_net_gain"], -x["avg_broken"], x["avg_corr"]),
        reverse=True,
    )
    return best, candidates_sorted


def main() -> None:
    args = parse_args()
    module = load_hema_module()

    scalar = json.loads(Path(args.scalar_result).read_text(encoding="utf-8"))
    meta = json.loads(Path(args.meta_warmstart_result).read_text(encoding="utf-8"))
    scalar_models = scalar["results"][args.mnv2_label]["models"]
    meta_models = meta["models"]
    mnv2_ts = scalar["config"]["mnv2_versions"][args.mnv2_label]

    model_key = "comp_noTS"
    aligned = {}
    for ds_name in module.DATASETS:
        mnv2_recs = module.load_mnv2(ds_name, mnv2_ts)
        specm_recs = module.load_specm(model_key, ds_name)
        aligned[(ds_name, model_key)] = module.align_records(mnv2_recs, specm_recs)

    final_results = {
        "timestamp": datetime.now().isoformat(),
        "experiment": "comp_noTS_richer_veto",
        "mnv2_label": args.mnv2_label,
        "mnv2_ts": mnv2_ts,
        "weights": {
            "reverse_manip_weight": float(args.reverse_manip_weight),
            "reverse_auth_weight": float(args.reverse_auth_weight),
            "non_casia_harm_weight": float(args.non_casia_harm_weight),
        },
        "grid": {
            "reverse_manip_grid": [float(x) for x in args.reverse_manip_grid],
            "non_casia_grid": [float(x) for x in args.non_casia_grid],
            "tau_deltas": [float(x) for x in args.tau_deltas],
        },
        "references": {
            "scalar_result": str(Path(args.scalar_result)),
            "meta_warmstart_result": str(Path(args.meta_warmstart_result)),
        },
        "baselines": {
            "icwmv": deepcopy(scalar_models[model_key]["icwmv"]),
            "scalar_veto": deepcopy(scalar_models[model_key]["hema_icwmv_veto"]),
            "meta_warmstart_veto": deepcopy(meta_models[model_key]["meta_warmstart_veto"]),
        },
        "richer_veto": {"per_ds": {}},
        "tuning": {},
    }

    f1s = []
    corrs = []
    gains = []
    for test_ds in module.DATASETS:
        base_cfg = scalar_models[model_key]["hema_icwmv_veto"]["per_ds"][test_ds]["best_cfg"]
        train_dss = [ds for ds in module.DATASETS if ds != test_ds]
        best_grid_cfg, all_candidates = tune_richer_grid(
            module,
            aligned,
            train_dss,
            scalar_models,
            model_key=model_key,
            reverse_auth_weight=float(args.reverse_auth_weight),
            reverse_manip_grid=[float(x) for x in args.reverse_manip_grid],
            non_casia_grid=[float(x) for x in args.non_casia_grid],
            tau_deltas=[float(x) for x in args.tau_deltas],
        )
        final_results["tuning"][test_ds] = {
            "base_cfg": {
                "tau": float(base_cfg["tau"]),
                "pos_weight": float(base_cfg["pos_weight"]),
                "scalar_model_key": base_cfg["model_key"],
            },
            "selected_grid_cfg": best_grid_cfg,
            "top_candidates": all_candidates[:5],
        }
        x_tr, y_tr, w_tr = richer_override_dataset(
            module,
            aligned,
            train_dss,
            model_key=model_key,
            pos_weight=float(base_cfg["pos_weight"]),
            reverse_manip_weight=float(best_grid_cfg["reverse_manip_weight"]),
            reverse_auth_weight=float(args.reverse_auth_weight),
            non_casia_harm_weight=float(best_grid_cfg["non_casia_harm_weight"]),
        )
        veto_model = module.train_veto_model(x_tr, y_tr, w_tr, "xgb_depth2")
        mnv2_test, specm_test = aligned[(test_ds, model_key)]
        preds, actions = apply_richer_veto(
            module,
            mnv2_test,
            specm_test,
            veto_model,
            tau=clipped_tau(float(base_cfg["tau"]), float(best_grid_cfg["tau_delta"])),
        )
        res = module.eval_preds(preds, mnv2_test, actions)
        res["cfg"] = {
            "tau": clipped_tau(float(base_cfg["tau"]), float(best_grid_cfg["tau_delta"])),
            "tau_base": float(base_cfg["tau"]),
            "tau_delta": float(best_grid_cfg["tau_delta"]),
            "pos_weight": float(base_cfg["pos_weight"]),
            "scalar_model_key": base_cfg["model_key"],
            "richer_model_key": "xgb_depth2",
            "reverse_manip_weight": float(best_grid_cfg["reverse_manip_weight"]),
            "reverse_auth_weight": float(args.reverse_auth_weight),
            "non_casia_harm_weight": float(best_grid_cfg["non_casia_harm_weight"]),
        }
        final_results["richer_veto"]["per_ds"][test_ds] = res
        f1s.append(res["macro_f1"])
        corrs.append(res["err_corr"]["rate"])
        gains.append(res["err_corr"]["net_gain"])
        print(
            f"[{test_ds:10s}] richer F1={res['macro_f1']:.4f} "
            f"corr={res['err_corr']['rate']:.3f} "
            f"net={res['err_corr']['net_gain']} "
            f"broken={res['err_corr']['n_broken']} "
            f"(rmw={best_grid_cfg['reverse_manip_weight']:.1f}, "
            f"nchw={best_grid_cfg['non_casia_harm_weight']:.1f}, "
            f"tau={res['cfg']['tau']:.2f})"
        )

    final_results["richer_veto"]["avg_f1"] = round(float(np.mean(f1s)), 4)
    final_results["richer_veto"]["avg_corr"] = round(float(np.mean(corrs)), 4)
    final_results["richer_veto"]["avg_net_gain"] = round(float(np.mean(gains)), 4)

    out_dir = ROOT / "experiments" / "results" / "hema_icwmv_veto"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"comp_nots_richer_veto_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(final_results, indent=2), encoding="utf-8")
    print(f"\n저장: {out_path}")


if __name__ == "__main__":
    main()
