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


_SCRIPT_DIR = Path(__file__).resolve().parent
_LOCAL_ROOT = _SCRIPT_DIR / "data"
ROOT = _LOCAL_ROOT if _LOCAL_ROOT.exists() else Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "experiments" / "results" / "arv_backbone_transfer"
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
        "--backbones",
        nargs="+",
        default=["mnv2_strong", "mnv2_weak", "mnv2_nofinetune"],
        help=(
            "현재 최소 서버 재현 번들에서 바로 재현 가능한 기본값은 MNV2 3종이다. "
            "MobileCLIP 비교를 하려면 추가 JSONL/datasets 자산이 필요하다."
        ),
    )
    parser.add_argument(
        "--specm-model",
        type=str,
        default="comp_noTS",
    )
    parser.add_argument(
        "--taus",
        nargs="+",
        type=float,
        default=[0.35, 0.45, 0.55, 0.65],
    )
    parser.add_argument(
        "--pos-weights",
        nargs="+",
        type=float,
        default=[1.0, 2.0, 4.0],
    )
    parser.add_argument(
        "--scalar-models",
        nargs="+",
        default=["logreg", "xgb_stump", "xgb_depth2"],
    )
    parser.add_argument(
        "--reverse-auth-weight",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--reverse-manip-grid",
        nargs="+",
        type=float,
        default=[4.0, 6.0, 8.0],
    )
    parser.add_argument(
        "--non-casia-grid",
        nargs="+",
        type=float,
        default=[1.0, 1.5],
    )
    parser.add_argument(
        "--tau-deltas",
        nargs="+",
        type=float,
        default=[0.0, 0.05],
    )
    parser.add_argument(
        "--subtype-ts",
        type=str,
        default="20260319_070725",
        help="MNV2 JSONL timestamp used only to recover subtype context by image_path join",
    )
    return parser.parse_args()


HEMA = load_module("hema_veto_backbone", ROOT / "experiments" / "run_hema_icwmv_veto_loo_cd.py")
COMP = load_module("comp_richer_backbone", ROOT / "experiments" / "run_comp_nots_richer_veto.py")
TRANSFER = load_module("icwmv_transfer_backbone", ROOT / "experiments" / "run_icwmv_backbone_transfer.py")


def load_jsonl(path: Path) -> List[Dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def subtype_map_for_dataset(ds_name: str, subtype_ts: str) -> Dict[str, str]:
    path = ROOT / "experiments" / "results" / "backbone_eval" / f"mobilenetv2_dualstream_{ds_name}_{subtype_ts}.jsonl"
    records = load_jsonl(path)
    return {r["image_path"]: str(r.get("sub_type", "")) for r in records}


def augment_with_subtype(records: List[Dict], subtype_map: Dict[str, str]) -> List[Dict]:
    out = []
    for rec in records:
        item = dict(rec)
        item["sub_type"] = subtype_map.get(rec["image_path"], "")
        out.append(item)
    return out


def apply_plain_icwmv(gen_recs: List[Dict], specm_recs: List[Dict]) -> np.ndarray:
    preds = []
    for g, s in zip(gen_recs, specm_recs):
        preds.append(HEMA.icwmv_single(g, s))
    return np.asarray(preds, dtype=np.int64)


def tune_scalar(
    aligned: Dict[Tuple[str, str], Tuple[List[Dict], List[Dict]]],
    specm_model: str,
    taus: List[float],
    pos_weights: List[float],
    model_keys: List[str],
) -> Tuple[Dict[str, Dict], Dict]:
    per_ds = {}
    f1s = []
    corrs = []
    gains = []
    for test_ds in HEMA.DATASETS:
        train_dss = [ds for ds in HEMA.DATASETS if ds != test_ds]
        best_cfg = HEMA.tune_veto(
            aligned,
            train_dss,
            specm_model,
            taus=taus,
            pos_weights=pos_weights,
            model_keys=model_keys,
        )
        feature_mode = "meta" if best_cfg["model_key"] == "xgb_meta_depth2" else "base"
        x_tr, y_tr, w_tr, _ = HEMA.concat_override_data(
            aligned,
            train_dss,
            specm_model,
            float(best_cfg["pos_weight"]),
            feature_mode=feature_mode,
        )
        veto_model = HEMA.train_veto_model(x_tr, y_tr, w_tr, best_cfg["model_key"])
        setattr(veto_model, "_feature_mode", feature_mode)
        gen_test, specm_test = aligned[(test_ds, specm_model)]
        preds, actions = HEMA.apply_icwmv_veto(gen_test, specm_test, veto_model, tau=float(best_cfg["tau"]))
        res = HEMA.eval_preds(preds, gen_test, actions)
        res["best_cfg"] = deepcopy(best_cfg)
        per_ds[test_ds] = res
        f1s.append(res["macro_f1"])
        corrs.append(res["err_corr"]["rate"])
        gains.append(res["err_corr"]["net_gain"])
    return per_ds, {
        "avg_f1": round(float(np.mean(f1s)), 4),
        "avg_corr": round(float(np.mean(corrs)), 4),
        "avg_net_gain": round(float(np.mean(gains)), 4),
    }


def tune_richer_grid(
    aligned: Dict[Tuple[str, str], Tuple[List[Dict], List[Dict]]],
    train_dss: List[str],
    specm_model: str,
    scalar_per_ds: Dict[str, Dict],
    reverse_auth_weight: float,
    reverse_manip_grid: List[float],
    non_casia_grid: List[float],
    tau_deltas: List[float],
) -> Tuple[Dict, List[Dict]]:
    candidates = []
    best = None

    for reverse_manip_weight in reverse_manip_grid:
        for non_casia_harm_weight in non_casia_grid:
            for tau_delta in tau_deltas:
                inner_f1s = []
                inner_corrs = []
                inner_gains = []
                inner_broken = []

                for val_ds in train_dss:
                    base_cfg = scalar_per_ds[val_ds]["best_cfg"]
                    inner_train = [ds for ds in train_dss if ds != val_ds]
                    x_tr, y_tr, w_tr = COMP.richer_override_dataset(
                        HEMA,
                        aligned,
                        inner_train,
                        model_key=specm_model,
                        pos_weight=float(base_cfg["pos_weight"]),
                        reverse_manip_weight=float(reverse_manip_weight),
                        reverse_auth_weight=float(reverse_auth_weight),
                        non_casia_harm_weight=float(non_casia_harm_weight),
                    )
                    veto_model = HEMA.train_veto_model(x_tr, y_tr, w_tr, "xgb_depth2")
                    gen_val, specm_val = aligned[(val_ds, specm_model)]
                    preds, actions = COMP.apply_richer_veto(
                        HEMA,
                        gen_val,
                        specm_val,
                        veto_model,
                        tau=COMP.clipped_tau(float(base_cfg["tau"]), float(tau_delta)),
                    )
                    res = HEMA.eval_preds(preds, gen_val, actions)
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
    return best, sorted(
        candidates,
        key=lambda x: (x["avg_f1"], x["avg_net_gain"], -x["avg_broken"], x["avg_corr"]),
        reverse=True,
    )


def evaluate_backbone(
    backbone_key: str,
    specm_model: str,
    subtype_ts: str,
    taus: List[float],
    pos_weights: List[float],
    scalar_model_keys: List[str],
    reverse_auth_weight: float,
    reverse_manip_grid: List[float],
    non_casia_grid: List[float],
    tau_deltas: List[float],
) -> Dict:
    gen_data = TRANSFER.load_backbone_records(backbone_key)
    aligned: Dict[Tuple[str, str], Tuple[List[Dict], List[Dict]]] = {}

    baseline_per_ds = {}
    icwmv_per_ds = {}
    baseline_f1s = []
    icwmv_f1s = []
    icwmv_corrs = []
    icwmv_gains = []

    for ds_name in HEMA.DATASETS:
        subtype_map = subtype_map_for_dataset(ds_name, subtype_ts)
        gen_recs = augment_with_subtype(gen_data[ds_name], subtype_map)
        specm_recs = HEMA.load_specm(specm_model, ds_name)
        assert specm_recs is not None
        gen_aligned, specm_aligned = HEMA.align_records(gen_recs, specm_recs)
        aligned[(ds_name, specm_model)] = (gen_aligned, specm_aligned)

        base_preds = np.asarray([HEMA.CLS2IDX[r["pred_label"]] for r in gen_aligned], dtype=np.int64)
        base_res = HEMA.eval_preds(base_preds, gen_aligned)
        baseline_per_ds[ds_name] = base_res
        baseline_f1s.append(base_res["macro_f1"])

        icwmv_preds = apply_plain_icwmv(gen_aligned, specm_aligned)
        icwmv_res = HEMA.eval_preds(icwmv_preds, gen_aligned)
        icwmv_per_ds[ds_name] = icwmv_res
        icwmv_f1s.append(icwmv_res["macro_f1"])
        icwmv_corrs.append(icwmv_res["err_corr"]["rate"])
        icwmv_gains.append(icwmv_res["err_corr"]["net_gain"])

    scalar_per_ds, scalar_avg = tune_scalar(
        aligned,
        specm_model=specm_model,
        taus=taus,
        pos_weights=pos_weights,
        model_keys=scalar_model_keys,
    )

    richer_per_ds = {}
    richer_tuning = {}
    richer_f1s = []
    richer_corrs = []
    richer_gains = []
    for test_ds in HEMA.DATASETS:
        train_dss = [ds for ds in HEMA.DATASETS if ds != test_ds]
        best_grid_cfg, all_candidates = tune_richer_grid(
            aligned,
            train_dss=train_dss,
            specm_model=specm_model,
            scalar_per_ds=scalar_per_ds,
            reverse_auth_weight=reverse_auth_weight,
            reverse_manip_grid=reverse_manip_grid,
            non_casia_grid=non_casia_grid,
            tau_deltas=tau_deltas,
        )
        base_cfg = scalar_per_ds[test_ds]["best_cfg"]
        x_tr, y_tr, w_tr = COMP.richer_override_dataset(
            HEMA,
            aligned,
            train_dss,
            model_key=specm_model,
            pos_weight=float(base_cfg["pos_weight"]),
            reverse_manip_weight=float(best_grid_cfg["reverse_manip_weight"]),
            reverse_auth_weight=float(reverse_auth_weight),
            non_casia_harm_weight=float(best_grid_cfg["non_casia_harm_weight"]),
        )
        veto_model = HEMA.train_veto_model(x_tr, y_tr, w_tr, "xgb_depth2")
        gen_test, specm_test = aligned[(test_ds, specm_model)]
        tau = COMP.clipped_tau(float(base_cfg["tau"]), float(best_grid_cfg["tau_delta"]))
        preds, actions = COMP.apply_richer_veto(
            HEMA,
            gen_test,
            specm_test,
            veto_model,
            tau=tau,
        )
        res = HEMA.eval_preds(preds, gen_test, actions)
        res["cfg"] = {
            "tau": float(tau),
            "tau_base": float(base_cfg["tau"]),
            "tau_delta": float(best_grid_cfg["tau_delta"]),
            "pos_weight": float(base_cfg["pos_weight"]),
            "scalar_model_key": base_cfg["model_key"],
            "richer_model_key": "xgb_depth2",
            "reverse_manip_weight": float(best_grid_cfg["reverse_manip_weight"]),
            "reverse_auth_weight": float(reverse_auth_weight),
            "non_casia_harm_weight": float(best_grid_cfg["non_casia_harm_weight"]),
        }
        richer_per_ds[test_ds] = res
        richer_tuning[test_ds] = {
            "base_cfg": deepcopy(base_cfg),
            "selected_grid_cfg": deepcopy(best_grid_cfg),
            "top_candidates": deepcopy(all_candidates[:5]),
        }
        richer_f1s.append(res["macro_f1"])
        richer_corrs.append(res["err_corr"]["rate"])
        richer_gains.append(res["err_corr"]["net_gain"])

    return {
        "label": TRANSFER.BACKBONES[backbone_key]["label"],
        "specm_model": specm_model,
        "baseline": {
            "avg_f1": round(float(np.mean(baseline_f1s)), 4),
            "per_ds": baseline_per_ds,
        },
        "icwmv": {
            "avg_f1": round(float(np.mean(icwmv_f1s)), 4),
            "avg_corr": round(float(np.mean(icwmv_corrs)), 4),
            "avg_net_gain": round(float(np.mean(icwmv_gains)), 4),
            "per_ds": icwmv_per_ds,
        },
        "scalar_veto": {
            **scalar_avg,
            "per_ds": scalar_per_ds,
        },
        "richer_veto": {
            "avg_f1": round(float(np.mean(richer_f1s)), 4),
            "avg_corr": round(float(np.mean(richer_corrs)), 4),
            "avg_net_gain": round(float(np.mean(richer_gains)), 4),
            "per_ds": richer_per_ds,
        },
        "tuning": {
            "scalar_search": {
                "taus": [float(x) for x in taus],
                "pos_weights": [float(x) for x in pos_weights],
                "model_keys": list(scalar_model_keys),
            },
            "richer_search": {
                "reverse_auth_weight": float(reverse_auth_weight),
                "reverse_manip_grid": [float(x) for x in reverse_manip_grid],
                "non_casia_grid": [float(x) for x in non_casia_grid],
                "tau_deltas": [float(x) for x in tau_deltas],
                "per_ds": richer_tuning,
            },
        },
    }


def main() -> None:
    args = parse_args()
    results = {
        "timestamp": datetime.now().isoformat(),
        "experiment": "arv_backbone_transfer",
        "specm_model": args.specm_model,
        "subtype_ts": args.subtype_ts,
        "backbones": {},
    }

    for backbone_key in args.backbones:
        print(f"\n[Backbone={backbone_key}]")
        try:
            res = evaluate_backbone(
                backbone_key=backbone_key,
                specm_model=args.specm_model,
                subtype_ts=args.subtype_ts,
                taus=[float(x) for x in args.taus],
                pos_weights=[float(x) for x in args.pos_weights],
                scalar_model_keys=list(args.scalar_models),
                reverse_auth_weight=float(args.reverse_auth_weight),
                reverse_manip_grid=[float(x) for x in args.reverse_manip_grid],
                non_casia_grid=[float(x) for x in args.non_casia_grid],
                tau_deltas=[float(x) for x in args.tau_deltas],
            )
        except FileNotFoundError as exc:
            raise SystemExit(
                "입력 자산이 부족해 backbone transfer를 계속할 수 없습니다.\n"
                f"- backbone: {backbone_key}\n"
                f"- missing: {exc}\n"
                "- 현재 최소 서버 재현 번들에서 기본 지원되는 실행은 "
                "`--backbones mnv2_strong mnv2_weak mnv2_nofinetune` 입니다.\n"
                "- MobileCLIP 비교를 하려면 mobileclip JSONL, seed JSONL, datasets 이미지, "
                "specm_v4 JSONL을 추가로 넣어야 합니다."
            ) from exc
        results["backbones"][backbone_key] = res
        print(
            f"  baseline={res['baseline']['avg_f1']:.4f} | "
            f"icwmv={res['icwmv']['avg_f1']:.4f} | "
            f"scalar={res['scalar_veto']['avg_f1']:.4f} | "
            f"richer={res['richer_veto']['avg_f1']:.4f}"
        )

    out_path = OUT_DIR / f"arv_backbone_transfer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n저장: {out_path}")


if __name__ == "__main__":
    main()
