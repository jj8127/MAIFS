#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
SERVER_ROOT = ROOT / "Server_Reproduction" / "ARV"
SERVER_DATA_ROOT = SERVER_ROOT / "data"
OUT_DIR = Path(__file__).resolve().parent.parent / "models" / "arv_stage2"
SCALAR_RESULT = SERVER_DATA_ROOT / "experiments" / "results" / "hema_icwmv_veto" / "hema_icwmv_veto_loo_cd_20260323_114321.json"
BEST_RICHER_RESULT = SERVER_DATA_ROOT / "experiments" / "results" / "hema_icwmv_veto" / "comp_nots_richer_veto_20260325_084631.json"

REVERSE_MANIP_WEIGHT = 6.0
REVERSE_AUTH_WEIGHT = 2.0
NON_CASIA_HARM_WEIGHT = 1.5
MNV2_LABEL = "strong"
MODEL_KEY = "comp_noTS"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    hema = load_module("hema_veto_export", SERVER_ROOT / "data" / "experiments" / "run_hema_icwmv_veto_loo_cd.py")
    richer = load_module("richer_export", SERVER_ROOT / "data" / "experiments" / "run_comp_nots_richer_veto.py")

    scalar = json.loads(SCALAR_RESULT.read_text(encoding="utf-8"))
    richer_ref = json.loads(BEST_RICHER_RESULT.read_text(encoding="utf-8"))
    scalar_models = scalar["results"][MNV2_LABEL]["models"]
    mnv2_ts = scalar["config"]["mnv2_versions"][MNV2_LABEL]

    aligned = {}
    for ds_name in hema.DATASETS:
        mnv2_recs = hema.load_mnv2(ds_name, mnv2_ts)
        specm_recs = hema.load_specm(MODEL_KEY, ds_name)
        aligned[(ds_name, MODEL_KEY)] = hema.align_records(mnv2_recs, specm_recs)

    manifest = {
        "generated_from": {
            "scalar_result": "Server_Reproduction/ARV/data/experiments/results/hema_icwmv_veto/hema_icwmv_veto_loo_cd_20260323_114321.json",
            "best_richer_result": "Server_Reproduction/ARV/data/experiments/results/hema_icwmv_veto/comp_nots_richer_veto_20260325_084631.json",
            "mnv2_label": MNV2_LABEL,
            "mnv2_ts": mnv2_ts,
            "model_key": MODEL_KEY,
        },
        "feature_dim": 46,
        "model_format": "xgboost_booster_json",
        "default_model_key": "base",
        "subtype_vocab": hema.SUBTYPE_VOCAB,
        "weights": {
            "reverse_manip_weight": REVERSE_MANIP_WEIGHT,
            "reverse_auth_weight": REVERSE_AUTH_WEIGHT,
            "non_casia_harm_weight": NON_CASIA_HARM_WEIGHT,
        },
        "models": {},
    }

    for test_ds in hema.DATASETS:
        base_cfg = scalar_models[MODEL_KEY]["hema_icwmv_veto"]["per_ds"][test_ds]["best_cfg"]
        richer_cfg = richer_ref["richer_veto"]["per_ds"][test_ds]["cfg"]
        train_dss = [ds for ds in hema.DATASETS if ds != test_ds]
        x_tr, y_tr, w_tr = richer.richer_override_dataset(
            hema,
            aligned,
            train_dss,
            model_key=MODEL_KEY,
            pos_weight=float(base_cfg["pos_weight"]),
            reverse_manip_weight=REVERSE_MANIP_WEIGHT,
            reverse_auth_weight=REVERSE_AUTH_WEIGHT,
            non_casia_harm_weight=NON_CASIA_HARM_WEIGHT,
        )
        model = hema.train_veto_model(x_tr, y_tr, w_tr, "xgb_depth2")
        if not hasattr(model, "get_booster"):
            raise RuntimeError(f"{test_ds}: xgb_depth2 모델 export 실패")
        out_name = f"arv_comp_nots_{test_ds}.json"
        model.get_booster().save_model(str(OUT_DIR / out_name))
        manifest["models"][test_ds] = {
            "path": out_name,
            "tau": float(richer_cfg["tau"]),
            "pos_weight": float(richer_cfg["pos_weight"]),
            "scalar_model_key": str(richer_cfg["scalar_model_key"]),
            "richer_model_key": str(richer_cfg["richer_model_key"]),
            "train_rows": int(len(x_tr)),
            "beneficial_weighting": {
                "reverse_manip_weight": REVERSE_MANIP_WEIGHT,
                "reverse_auth_weight": REVERSE_AUTH_WEIGHT,
                "non_casia_harm_weight": NON_CASIA_HARM_WEIGHT,
            },
        }
        print(f"[exported] {test_ds} -> {OUT_DIR / out_name}")

    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[saved] {OUT_DIR / 'manifest.json'}")


if __name__ == "__main__":
    main()
