#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


_SCRIPT_DIR = Path(__file__).resolve().parent
_LOCAL_ROOT = _SCRIPT_DIR / "data"
ROOT = _LOCAL_ROOT if _LOCAL_ROOT.exists() else Path(__file__).resolve().parents[1]
BE_DIR = ROOT / "experiments" / "results" / "backbone_eval"
COMP_DIR = ROOT / "experiments" / "results" / "specm_complementary_eval"
OUT_DIR = ROOT / "experiments" / "results" / "generalist_arv"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODULE_PATH = ROOT / "experiments" / "run_hema_icwmv_veto_loo_cd.py"

DATASETS = ["base", "dsC", "opensdi", "aigenproxy"]
BACKBONES = {
    "clip_ft4_strong": {
        "label": "MobileCLIP-ft4 strong",
        "prefix": "mobileclip_s2_finetuned",
        "timestamp": "20260319_061834",
    },
    "clip_zeroshot_weak": {
        "label": "MobileCLIP zero-shot weak",
        "prefix": "mobileclip_s2_zeroshot_scored",
        "timestamp": "20260323_141929",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backbones",
        nargs="+",
        default=["clip_ft4_strong", "clip_zeroshot_weak"],
        choices=sorted(BACKBONES.keys()),
    )
    parser.add_argument(
        "--pos-weight-grid",
        nargs="+",
        type=float,
        default=[1.0, 2.0],
    )
    parser.add_argument(
        "--reverse-manip-grid",
        nargs="+",
        type=float,
        default=[4.0, 6.0],
    )
    parser.add_argument(
        "--reverse-auth-grid",
        nargs="+",
        type=float,
        default=[1.5],
    )
    parser.add_argument(
        "--tau-grid",
        nargs="+",
        type=float,
        default=[0.45, 0.55, 0.65],
    )
    return parser.parse_args()


def load_helper_module():
    spec = importlib.util.spec_from_file_location("hema_veto_helper", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["hema_veto_helper"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def train_richer_model(x_tr: np.ndarray, y_tr: np.ndarray, sample_weight: np.ndarray):
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
            random_state=42,
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
            ("clf", LogisticRegression(max_iter=500, random_state=42)),
        ])
        clf.fit(x_tr, y_tr, clf__sample_weight=sample_weight)
        return clf


def load_jsonl(path: Path) -> List[Dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def find_comp_jsonl(ds_name: str) -> Path:
    cands = sorted(COMP_DIR.glob(f"specm_comp_gamma1.0_wmax10_noTS_{ds_name}_*.jsonl"))
    if not cands:
        raise FileNotFoundError(f"Missing comp_noTS JSONL for {ds_name}")
    return cands[-1]


def load_backbone_records(backbone_key: str) -> Dict[str, List[Dict]]:
    cfg = BACKBONES[backbone_key]
    out = {}
    for ds in DATASETS:
        path = BE_DIR / f"{cfg['prefix']}_{ds}_{cfg['timestamp']}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Missing backbone JSONL: {path}")
        out[ds] = load_jsonl(path)
    return out


def align_records(gen_recs: List[Dict], specm_recs: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    specm_map = {r["image_path"]: r for r in specm_recs}
    aligned_g, aligned_s = [], []
    for g in gen_recs:
        s = specm_map.get(g["image_path"])
        if s:
            aligned_g.append(g)
            aligned_s.append(s)
    return aligned_g, aligned_s


def rich_feature(module, g: Dict, s: Dict) -> List[float]:
    p_auth = float(g["scores"]["authentic"])
    p_manip = float(g["scores"]["manipulated"])
    p_ai = float(g["scores"].get("ai_generated", 0.0))
    p_auth_bin, p_manip_bin = module.mnv2_binary_probs(g)
    specm_a = float(s["authentic_score"])
    specm_m = float(s["manip_score"])
    ic_auth, ic_manip = module.weighted_binary_scores(g, s)
    m_conf = float(g["confidence"])
    s_conf = float(s["confidence"])
    mnv2_pred = module.mnv2_binary_pred_idx(g)
    ic_pred = int(ic_manip >= ic_auth)
    ai_margin = p_ai - max(p_auth, p_manip)
    mnv2_margin = abs(p_manip_bin - p_auth_bin)
    specm_margin = abs(specm_m - specm_a)
    ic_margin = abs(ic_manip - ic_auth)
    conf_gap = s_conf - m_conf
    conf_prod = s_conf * m_conf
    return [
        p_auth_bin,
        p_manip_bin,
        specm_a,
        specm_m,
        ic_auth,
        ic_manip,
        ic_manip - ic_auth,
        specm_m - p_manip_bin,
        specm_a - p_auth_bin,
        ai_margin,
        m_conf,
        s_conf,
        s_conf / max(m_conf, 1e-3),
        float(mnv2_pred),
        float(ic_pred),
        abs(specm_m - p_manip_bin),
        mnv2_margin,
        specm_margin,
        ic_margin,
        conf_gap,
        conf_prod,
    ]


def build_override_dataset(
    module,
    aligned: Dict[Tuple[str, str], Tuple[List[Dict], List[Dict]]],
    train_dss: List[str],
    backbone_key: str,
    pos_weight: float,
    reverse_manip_weight: float,
    reverse_auth_weight: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    feats, labels, weights = [], [], []
    for ds_name in train_dss:
        gen_recs, specm_recs = aligned[(ds_name, backbone_key)]
        for g, s in zip(gen_recs, specm_recs):
            if module.is_ai_lock(g):
                continue
            true_idx = module.CLS2IDX[g["true_label"]]
            if true_idx == module.CLS2IDX["ai_generated"]:
                continue

            mnv2_bin = module.mnv2_binary_pred_idx(g)
            ic_pred = int(np.argmax(module.weighted_binary_scores(g, s)))
            if ic_pred == mnv2_bin:
                continue

            keep_override = int(ic_pred == true_idx)
            sample_weight = float(pos_weight if keep_override else 1.0)
            if not keep_override:
                if g["true_label"] == "manipulated" and ic_pred == module.CLS2IDX["authentic"]:
                    sample_weight *= float(reverse_manip_weight)
                elif g["true_label"] == "authentic" and ic_pred == module.CLS2IDX["manipulated"]:
                    sample_weight *= float(reverse_auth_weight)

            feats.append(rich_feature(module, g, s))
            labels.append(keep_override)
            weights.append(sample_weight)

    if not feats:
        return (
            np.zeros((0, 21), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.float32),
        )

    return (
        np.asarray(feats, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
        np.asarray(weights, dtype=np.float32),
    )


def apply_richer_veto(module, gen_recs: List[Dict], specm_recs: List[Dict], model, tau: float):
    preds = []
    actions = defaultdict(int)
    for g, s in zip(gen_recs, specm_recs):
        if module.is_ai_lock(g):
            preds.append(module.CLS2IDX["ai_generated"])
            actions["ai_lock"] += 1
            continue

        mnv2_bin = module.mnv2_binary_pred_idx(g)
        ic_pred = int(np.argmax(module.weighted_binary_scores(g, s)))
        if ic_pred == mnv2_bin:
            preds.append(ic_pred)
            actions["icwmv_no_override"] += 1
            continue

        p_keep = float(model.predict_proba(np.asarray([rich_feature(module, g, s)], dtype=np.float32))[0, 1])
        if p_keep >= tau:
            preds.append(ic_pred)
            actions["keep_icwmv_override"] += 1
        else:
            preds.append(mnv2_bin)
            actions["revert_to_backbone"] += 1
    return np.asarray(preds, dtype=np.int64), dict(actions)


def tune_richer_grid(
    module,
    aligned: Dict[Tuple[str, str], Tuple[List[Dict], List[Dict]]],
    train_dss: List[str],
    backbone_key: str,
    pos_weight_grid: List[float],
    reverse_manip_grid: List[float],
    reverse_auth_grid: List[float],
    tau_grid: List[float],
) -> Tuple[Dict, List[Dict]]:
    candidates: List[Dict] = []
    best = None

    for pos_weight in pos_weight_grid:
        for reverse_manip_weight in reverse_manip_grid:
            for reverse_auth_weight in reverse_auth_grid:
                for tau in tau_grid:
                    inner_f1s, inner_corrs, inner_gains, inner_broken = [], [], [], []
                    for val_ds in train_dss:
                        inner_train = [ds for ds in train_dss if ds != val_ds]
                        x_tr, y_tr, w_tr = build_override_dataset(
                            module,
                            aligned,
                            inner_train,
                            backbone_key=backbone_key,
                            pos_weight=float(pos_weight),
                            reverse_manip_weight=float(reverse_manip_weight),
                            reverse_auth_weight=float(reverse_auth_weight),
                        )
                        veto_model = train_richer_model(x_tr, y_tr, w_tr)
                        gen_val, specm_val = aligned[(val_ds, backbone_key)]
                        preds, actions = apply_richer_veto(module, gen_val, specm_val, veto_model, tau=float(tau))
                        res = module.eval_preds(preds, gen_val, actions)
                        inner_f1s.append(res["macro_f1"])
                        inner_corrs.append(res["err_corr"]["rate"])
                        inner_gains.append(res["err_corr"]["net_gain"])
                        inner_broken.append(res["err_corr"]["n_broken"])

                    cand = {
                        "pos_weight": float(pos_weight),
                        "reverse_manip_weight": float(reverse_manip_weight),
                        "reverse_auth_weight": float(reverse_auth_weight),
                        "tau": float(tau),
                        "avg_f1": float(np.mean(inner_f1s)),
                        "avg_corr": float(np.mean(inner_corrs)),
                        "avg_net_gain": float(np.mean(inner_gains)),
                        "avg_broken": float(np.mean(inner_broken)),
                    }
                    candidates.append(cand)
                    key = (cand["avg_f1"], cand["avg_net_gain"], -cand["avg_broken"], cand["avg_corr"])
                    if best is None or key > (
                        best["avg_f1"], best["avg_net_gain"], -best["avg_broken"], best["avg_corr"]
                    ):
                        best = cand

    assert best is not None
    candidates_sorted = sorted(
        candidates,
        key=lambda x: (x["avg_f1"], x["avg_net_gain"], -x["avg_broken"], x["avg_corr"]),
        reverse=True,
    )
    return best, candidates_sorted


def evaluate_backbone(module, backbone_key: str, args: argparse.Namespace) -> Dict:
    gen_data = load_backbone_records(backbone_key)
    specm_data = {ds: load_jsonl(find_comp_jsonl(ds)) for ds in DATASETS}

    aligned = {}
    for ds in DATASETS:
        aligned[(ds, backbone_key)] = align_records(gen_data[ds], specm_data[ds])

    results = {
        "label": BACKBONES[backbone_key]["label"],
        "baseline": {},
        "plain_icwmv_comp_noTS": {},
        "richer_veto": {"per_ds": {}},
        "tuning": {},
        "coverage": {},
    }

    baseline_vals, icwmv_vals, corr_vals = [], [], []
    for ds in DATASETS:
        gen_aligned, specm_aligned = aligned[(ds, backbone_key)]
        baseline_preds = np.asarray([module.CLS2IDX[r["pred_label"]] for r in gen_aligned], dtype=np.int64)
        baseline_res = module.eval_preds(baseline_preds, gen_aligned, {"backbone_only": len(gen_aligned)})

        icwmv_preds = np.asarray([module.icwmv_single(g, s) for g, s in zip(gen_aligned, specm_aligned)], dtype=np.int64)
        icwmv_res = module.eval_preds(icwmv_preds, gen_aligned, {"icwmv_comp_noTS": len(gen_aligned)})

        results["coverage"][ds] = len(gen_aligned)
        results["baseline"][ds] = baseline_res
        results["plain_icwmv_comp_noTS"][ds] = icwmv_res
        baseline_vals.append(baseline_res["macro_f1"])
        icwmv_vals.append(icwmv_res["macro_f1"])
        corr_vals.append(icwmv_res["err_corr"]["rate"])

    results["baseline"]["avg_f1"] = round(float(np.mean(baseline_vals)), 4)
    results["plain_icwmv_comp_noTS"]["avg_f1"] = round(float(np.mean(icwmv_vals)), 4)
    results["plain_icwmv_comp_noTS"]["avg_corr"] = round(float(np.mean(corr_vals)), 4)

    for test_ds in DATASETS:
        train_dss = [ds for ds in DATASETS if ds != test_ds]
        best_cfg, all_candidates = tune_richer_grid(
            module,
            aligned,
            train_dss,
            backbone_key=backbone_key,
            pos_weight_grid=[float(x) for x in args.pos_weight_grid],
            reverse_manip_grid=[float(x) for x in args.reverse_manip_grid],
            reverse_auth_grid=[float(x) for x in args.reverse_auth_grid],
            tau_grid=[float(x) for x in args.tau_grid],
        )
        results["tuning"][test_ds] = {
            "best_cfg": best_cfg,
            "top_candidates": all_candidates[:5],
        }
        x_tr, y_tr, w_tr = build_override_dataset(
            module,
            aligned,
            train_dss,
            backbone_key=backbone_key,
            pos_weight=float(best_cfg["pos_weight"]),
            reverse_manip_weight=float(best_cfg["reverse_manip_weight"]),
            reverse_auth_weight=float(best_cfg["reverse_auth_weight"]),
        )
        veto_model = train_richer_model(x_tr, y_tr, w_tr)
        gen_test, specm_test = aligned[(test_ds, backbone_key)]
        preds, actions = apply_richer_veto(module, gen_test, specm_test, veto_model, tau=float(best_cfg["tau"]))
        res = module.eval_preds(preds, gen_test, actions)
        res["cfg"] = dict(best_cfg)
        res["n_common"] = len(gen_test)
        results["richer_veto"]["per_ds"][test_ds] = res
        print(
            f"[{backbone_key}/{test_ds}] ARV F1={res['macro_f1']:.4f} "
            f"corr={res['err_corr']['rate']:.3f} net={res['err_corr']['net_gain']} "
            f"broken={res['err_corr']['n_broken']} cfg={best_cfg}"
        )

    results["richer_veto"]["avg_f1"] = round(
        float(np.mean([v["macro_f1"] for v in results["richer_veto"]["per_ds"].values()])), 4
    )
    results["richer_veto"]["avg_corr"] = round(
        float(np.mean([v["err_corr"]["rate"] for v in results["richer_veto"]["per_ds"].values()])), 4
    )
    results["richer_veto"]["avg_net_gain"] = round(
        float(np.mean([v["err_corr"]["net_gain"] for v in results["richer_veto"]["per_ds"].values()])), 4
    )
    return results


def main() -> None:
    args = parse_args()
    module = load_helper_module()

    final = {
        "timestamp": datetime.now().isoformat(),
        "experiment": "generalist_comp_noTS_richer_veto",
        "backbones": {},
        "references": {
            "helper_module": str(MODULE_PATH),
        },
        "config": {
            "backbones": list(args.backbones),
            "pos_weight_grid": [float(x) for x in args.pos_weight_grid],
            "reverse_manip_grid": [float(x) for x in args.reverse_manip_grid],
            "reverse_auth_grid": [float(x) for x in args.reverse_auth_grid],
            "tau_grid": [float(x) for x in args.tau_grid],
        },
    }

    for backbone_key in args.backbones:
        print(f"\n[run] {backbone_key}")
        try:
            final["backbones"][backbone_key] = evaluate_backbone(module, backbone_key, args)
        except FileNotFoundError as exc:
            raise SystemExit(
                "입력 자산이 부족해 generalist ARV 실험을 시작할 수 없습니다.\n"
                f"- generalist: {backbone_key}\n"
                f"- missing: {exc}\n"
                "- 현재 최소 서버 재현 번들에는 MobileCLIP JSONL이 포함되어 있지 않습니다.\n"
                "- 이 스크립트를 실행하려면 mobileclip_s2_finetuned_*.jsonl 과 "
                "mobileclip_s2_zeroshot_scored_*.jsonl 을 추가로 넣어야 합니다."
            ) from exc

    out_path = OUT_DIR / f"generalist_comp_nots_richer_veto_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(final, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n저장: {out_path}")

    print("\n=== SUMMARY ===")
    for backbone_key in args.backbones:
        res = final["backbones"][backbone_key]
        print(
            f"{res['label']}: baseline={res['baseline']['avg_f1']:.4f} "
            f"icwmv={res['plain_icwmv_comp_noTS']['avg_f1']:.4f}/"
            f"{res['plain_icwmv_comp_noTS']['avg_corr']:.4f} "
            f"arv={res['richer_veto']['avg_f1']:.4f}/{res['richer_veto']['avg_corr']:.4f}"
        )


if __name__ == "__main__":
    main()
