#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCALAR_RESULT = (
    ROOT
    / "experiments"
    / "results"
    / "hema_icwmv_veto"
    / "hema_icwmv_veto_loo_cd_20260323_114321.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scalar-result",
        type=str,
        default=str(DEFAULT_SCALAR_RESULT),
        help="Path to the completed scalar veto result JSON",
    )
    parser.add_argument(
        "--mnv2-label",
        type=str,
        default="strong",
        help="MNV2 label to reuse from the scalar result JSON",
    )
    parser.add_argument(
        "--specm-models",
        nargs="+",
        default=["comp_noTS", "comp_g1"],
        help="SpecM models to evaluate with meta warm-start",
    )
    parser.add_argument(
        "--weighting-mode",
        choices=["inverse_confidence", "equal_weight"],
        default="inverse_confidence",
        help="1단계 ICWMV 결합 방식",
    )
    parser.add_argument(
        "--weighting-alpha",
        type=float,
        default=1.0,
        help="w = 1 / confidence^alpha 에서 alpha 값 (equal_weight 모드에서는 무시)",
    )
    return parser.parse_args()


def effective_weighting_alpha(weighting_mode: str, weighting_alpha: float) -> float:
    if weighting_mode == "equal_weight":
        return 0.0
    return max(float(weighting_alpha), 0.0)


def alpha_tag(weighting_mode: str, weighting_alpha: float) -> str:
    return f"alpha{effective_weighting_alpha(weighting_mode, weighting_alpha):.1f}".replace(".", "p")


def load_hema_module():
    script_path = ROOT / "experiments" / "run_hema_icwmv_veto_loo_cd.py"
    spec = importlib.util.spec_from_file_location("hema_veto", script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["hema_veto"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_aligned_data(module, mnv2_ts: str, specm_models: List[str]) -> Dict:
    aligned = {}
    for model_key in specm_models:
        for ds_name in module.DATASETS:
            mnv2_recs = module.load_mnv2(ds_name, mnv2_ts)
            specm_recs = module.load_specm(model_key, ds_name)
            if specm_recs is None:
                raise FileNotFoundError(f"SpecM JSONL missing for model={model_key}, ds={ds_name}")
            aligned_m, aligned_s = module.align_records(mnv2_recs, specm_recs)
            aligned[(ds_name, model_key)] = (aligned_m, aligned_s)
    return aligned


def main() -> None:
    args = parse_args()
    module = load_hema_module()
    module.WEIGHTING_MODE = args.weighting_mode
    module.WEIGHTING_ALPHA = float(args.weighting_alpha)

    scalar_path = Path(args.scalar_result)
    scalar = json.loads(scalar_path.read_text(encoding="utf-8"))
    scalar_models = scalar["results"][args.mnv2_label]["models"]
    mnv2_ts = scalar["config"]["mnv2_versions"][args.mnv2_label]

    aligned = load_aligned_data(module, mnv2_ts, list(args.specm_models))

    final_results = {
        "timestamp": datetime.now().isoformat(),
        "experiment": "meta_warmstart_from_scalar_best_cfg",
        "scalar_result": str(scalar_path),
        "mnv2_label": args.mnv2_label,
        "mnv2_ts": mnv2_ts,
        "weighting_mode": args.weighting_mode,
        "weighting_alpha": float(args.weighting_alpha),
        "effective_weighting_alpha": effective_weighting_alpha(args.weighting_mode, args.weighting_alpha),
        "models": {},
    }

    for model_key in args.specm_models:
        model_out = {
            "icwmv": deepcopy(scalar_models[model_key]["icwmv"]),
            "scalar_veto": deepcopy(scalar_models[model_key]["hema_icwmv_veto"]),
            "meta_warmstart_veto": {"per_ds": {}},
        }
        meta_f1s = []
        meta_corrs = []
        meta_gains = []
        for test_ds in module.DATASETS:
            base_cfg = scalar_models[model_key]["hema_icwmv_veto"]["per_ds"][test_ds]["best_cfg"]
            train_dss = [ds for ds in module.DATASETS if ds != test_ds]
            x_tr, y_tr, w_tr, _ = module.concat_override_data(
                aligned=aligned,
                train_dss=train_dss,
                specm_model=model_key,
                pos_weight=float(base_cfg["pos_weight"]),
                feature_mode="meta",
            )
            veto_model = module.train_veto_model(x_tr, y_tr, w_tr, "xgb_meta_depth2")
            setattr(veto_model, "_feature_mode", "meta")

            mnv2_test, specm_test = aligned[(test_ds, model_key)]
            preds, actions = module.apply_icwmv_veto(
                mnv2_test,
                specm_test,
                veto_model,
                tau=float(base_cfg["tau"]),
            )
            res = module.eval_preds(preds, mnv2_test, actions)
            res["warmstart_cfg"] = {
                "tau": float(base_cfg["tau"]),
                "pos_weight": float(base_cfg["pos_weight"]),
                "scalar_model_key": base_cfg["model_key"],
                "meta_model_key": "xgb_meta_depth2",
            }
            model_out["meta_warmstart_veto"]["per_ds"][test_ds] = res

            meta_f1s.append(res["macro_f1"])
            meta_corrs.append(res["err_corr"]["rate"])
            meta_gains.append(res["err_corr"]["net_gain"])
            print(
                f"[{model_key:10s}][{test_ds:10s}] "
                f"meta F1={res['macro_f1']:.4f} corr={res['err_corr']['rate']:.3f} "
                f"net={res['err_corr']['net_gain']} broken={res['err_corr']['n_broken']}"
            )

        model_out["meta_warmstart_veto"]["avg_f1"] = round(float(np.mean(meta_f1s)), 4)
        model_out["meta_warmstart_veto"]["avg_corr"] = round(float(np.mean(meta_corrs)), 4)
        model_out["meta_warmstart_veto"]["avg_net_gain"] = round(float(np.mean(meta_gains)), 4)
        final_results["models"][model_key] = model_out

        print(
            f"  -> [{model_key}] meta avg_F1={model_out['meta_warmstart_veto']['avg_f1']:.4f} "
            f"avg_corr={model_out['meta_warmstart_veto']['avg_corr']:.3f} "
            f"avg_net_gain={model_out['meta_warmstart_veto']['avg_net_gain']:.2f}"
        )

    out_dir = ROOT / "experiments" / "results" / "hema_icwmv_veto"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"hema_icwmv_veto_meta_warmstart_{alpha_tag(args.weighting_mode, args.weighting_alpha)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(final_results, indent=2), encoding="utf-8")
    print(f"\n저장: {out_path}")


if __name__ == "__main__":
    main()
