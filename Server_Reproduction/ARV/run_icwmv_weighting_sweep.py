#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np


_SCRIPT_DIR = Path(__file__).resolve().parent
_LOCAL_ROOT = _SCRIPT_DIR / "data"
ROOT = _LOCAL_ROOT if _LOCAL_ROOT.exists() else Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "experiments" / "results" / "paper_support"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mnv2-label",
        type=str,
        default="strong",
    )
    parser.add_argument(
        "--mnv2-ts",
        type=str,
        default="20260319_070725",
    )
    parser.add_argument(
        "--specm-model",
        type=str,
        default="comp_noTS",
    )
    parser.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=[0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
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


def weighted_binary_scores_alpha(module, m: Dict, s: Dict, alpha: float) -> np.ndarray:
    p_auth_bin, p_manip_bin = module.mnv2_binary_probs(m)
    mnv2_scores = np.array([p_auth_bin, p_manip_bin], dtype=np.float32)
    specm_scores = np.array(
        [float(s["authentic_score"]), float(s["manip_score"])],
        dtype=np.float32,
    )
    if alpha == 0.0:
        w_m = 1.0
        w_s = 1.0
    else:
        w_m = 1.0 / max(float(m["confidence"]), 1e-3) ** alpha
        w_s = 1.0 / max(float(s["confidence"]), 1e-3) ** alpha
    combined = w_m * mnv2_scores + w_s * specm_scores
    total = float(combined.sum())
    if total > 0.0:
        combined /= total
    return combined


def icwmv_single_alpha(module, m: Dict, s: Dict, alpha: float) -> int:
    if module.is_ai_lock(m):
        return module.CLS2IDX["ai_generated"]
    return int(np.argmax(weighted_binary_scores_alpha(module, m, s, alpha)))


def main() -> None:
    args = parse_args()
    module = load_hema_module()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    aligned = {}
    for ds_name in module.DATASETS:
        mnv2_recs = module.load_mnv2(ds_name, args.mnv2_ts)
        specm_recs = module.load_specm(args.specm_model, ds_name)
        if specm_recs is None:
            raise FileNotFoundError(f"Missing SpecM records for {args.specm_model}/{ds_name}")
        aligned[ds_name] = module.align_records(mnv2_recs, specm_recs)

    results: List[Dict] = []
    best = None
    for alpha in args.alphas:
        per_ds = {}
        f1s: List[float] = []
        corrs: List[float] = []
        gains: List[float] = []
        broken: List[float] = []
        for ds_name in module.DATASETS:
            mnv2_recs, specm_recs = aligned[ds_name]
            preds = np.array(
                [icwmv_single_alpha(module, m, s, alpha) for m, s in zip(mnv2_recs, specm_recs)],
                dtype=np.int64,
            )
            res = module.eval_preds(preds, mnv2_recs, {"alpha": alpha})
            per_ds[ds_name] = res
            f1s.append(res["macro_f1"])
            corrs.append(res["err_corr"]["rate"])
            gains.append(res["err_corr"]["net_gain"])
            broken.append(res["err_corr"]["n_broken"])

        row = {
            "alpha": float(alpha),
            "avg_f1": round(float(np.mean(f1s)), 4),
            "avg_corr": round(float(np.mean(corrs)), 4),
            "avg_net_gain": round(float(np.mean(gains)), 4),
            "avg_broken": round(float(np.mean(broken)), 4),
            "per_ds": per_ds,
        }
        results.append(row)
        key = (row["avg_f1"], row["avg_net_gain"], -row["avg_broken"], row["avg_corr"])
        if best is None or key > (
            best["avg_f1"],
            best["avg_net_gain"],
            -best["avg_broken"],
            best["avg_corr"],
        ):
            best = row

    payload = {
        "timestamp": datetime.now().isoformat(),
        "experiment": "icwmv_weighting_sweep",
        "mnv2_label": args.mnv2_label,
        "mnv2_ts": args.mnv2_ts,
        "specm_model": args.specm_model,
        "alpha_definition": "w = 1 / confidence^alpha; alpha=0 means equal-weight averaging",
        "results": results,
        "best": best,
    }
    out_path = OUT_DIR / f"icwmv_weighting_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
