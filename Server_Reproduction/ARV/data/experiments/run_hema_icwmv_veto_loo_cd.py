#!/usr/bin/env python3
"""
HEMA ICWMV-Veto LOO-CD Comparison
=================================

아이디어:
  1. 기본 fusion은 ICWMV를 그대로 사용한다.
  2. 단, ICWMV가 MNV2 binary prediction을 실제로 "override"한 샘플만 따로 본다.
  3. learned veto는 그 override가 유익한지(keep) 해로운지(revert)를 예측한다.

핵심 차이:
  - 기존 HEMA/action-gate는 모든 disagreement slice를 다시 배운다.
  - 이 스크립트는 ICWMV의 강한 inductive bias를 유지한 채,
    "ICWMV가 실수하는 override"만 선택적으로 되돌리는 데 집중한다.

목표:
  ICWMV보다 macro-F1과 error-correction rate를 동시에 개선할 수 있는지 확인한다.

실행:
  .venv-qwen/bin/python experiments/run_hema_icwmv_veto_loo_cd.py
  .venv-qwen/bin/python experiments/run_hema_icwmv_veto_loo_cd.py --mnv2 strong
"""

from __future__ import annotations

import argparse
import json
import warnings
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
BE_DIR = ROOT / "experiments" / "results" / "backbone_eval"
SPECM_EVAL_DIR = ROOT / "experiments" / "results" / "specm_eval"
LEGACY_COMP_DIR = ROOT / "experiments" / "results" / "specm_complementary_eval"

DATASETS = ["base", "dsC", "opensdi", "aigenproxy"]
CLS2IDX = {"authentic": 0, "manipulated": 1, "ai_generated": 2}
IDX2CLS = {0: "authentic", 1: "manipulated", 2: "ai_generated"}

SUBTYPE_VOCAB = [
    "casia_au",
    "casia_tp",
    "biggan",
    "imd2020_inpainting",
    "opensdi_real",
    "opensdi_partial_fake",
    "opensdi_entire_fake",
    "aigen_proxy_real",
    "aigen_proxy_manipulated",
    "aigen_proxy_ai_generated",
]
SUBTYPE2IDX = {name: idx for idx, name in enumerate(SUBTYPE_VOCAB)}

MNV2_PRESETS = {
    "strong": "20260319_070725",
    "weak": "20260319_064748",
}


@dataclass
class ConstantKeepModel:
    prob_keep: float

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        p = np.full(len(x), self.prob_keep, dtype=np.float32)
        return np.stack([1.0 - p, p], axis=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mnv2",
        nargs="+",
        default=["strong", "weak"],
        help="MNV2 preset(strong/weak) 또는 직접 timestamp",
    )
    parser.add_argument(
        "--specm-models",
        nargs="+",
        default=["v4", "comp_noTS", "comp_g1"],
        help="비교할 SpecM 모델 키",
    )
    parser.add_argument(
        "--taus",
        nargs="+",
        type=float,
        default=[0.35, 0.45, 0.55, 0.65],
        help="keep-override threshold 후보",
    )
    parser.add_argument(
        "--pos-weights",
        nargs="+",
        type=float,
        default=[1.0, 2.0, 4.0],
        help="beneficial-override positive weight 후보",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["logreg", "xgb_stump", "xgb_depth2"],
        help="veto model 후보",
    )
    return parser.parse_args()


def resolve_mnv2_versions(raw_values: Sequence[str]) -> Dict[str, str]:
    resolved = {}
    for value in raw_values:
        resolved[value] = MNV2_PRESETS.get(value, value)
    return resolved


def load_mnv2(ds_name: str, ts_be: str) -> List[Dict]:
    path = BE_DIR / f"mobilenetv2_dualstream_{ds_name}_{ts_be}.jsonl"
    with open(path) as f:
        return [json.loads(line) for line in f]


def find_specm_jsonl(model_key: str, ds_name: str) -> Optional[Path]:
    cands = sorted(SPECM_EVAL_DIR.glob(f"specm_{model_key}_{ds_name}_*.jsonl"))
    if cands:
        return cands[-1]

    tag_map = {
        "comp_g1": "gamma1.0_wmax10",
        "comp_g2": "gamma2.0_wmax10",
        "comp_g3": "gamma3.0_wmax10",
        "comp_noTS": "gamma1.0_wmax10_noTS",
    }
    tag = tag_map.get(model_key)
    if tag:
        cands = sorted(LEGACY_COMP_DIR.glob(f"specm_comp_{tag}_{ds_name}_*.jsonl"))
        if cands:
            return cands[-1]
    return None


def load_specm(model_key: str, ds_name: str) -> Optional[List[Dict]]:
    path = find_specm_jsonl(model_key, ds_name)
    if not path:
        return None
    with open(path) as f:
        return [json.loads(line) for line in f]


def align_records(mnv2_recs: List[Dict], specm_recs: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    specm_map = {r["image_path"]: r for r in specm_recs}
    aligned_m, aligned_s = [], []
    for m in mnv2_recs:
        s = specm_map.get(m["image_path"])
        if s:
            aligned_m.append(m)
            aligned_s.append(s)
    return aligned_m, aligned_s


def mnv2_binary_probs(m: Dict) -> Tuple[float, float]:
    p_auth = float(m["scores"]["authentic"])
    p_manip = float(m["scores"]["manipulated"])
    total = p_auth + p_manip
    if total <= 0.0:
        return 0.5, 0.5
    return p_auth / total, p_manip / total


def mnv2_binary_pred_idx(m: Dict) -> int:
    p_auth_bin, p_manip_bin = mnv2_binary_probs(m)
    return 1 if p_manip_bin >= p_auth_bin else 0


def is_ai_lock(m: Dict, threshold: float = 0.5) -> bool:
    p_aigen = float(m["scores"].get("ai_generated", 0.0))
    return m["pred_label"] == "ai_generated" or p_aigen > threshold


def subtype_context_features(m: Dict) -> List[float]:
    sub_type = str(m.get("sub_type", "")).strip().lower()
    exact = [0.0] * len(SUBTYPE_VOCAB)
    idx = SUBTYPE2IDX.get(sub_type)
    if idx is not None:
        exact[idx] = 1.0

    family = [
        1.0 if sub_type.startswith("casia") else 0.0,
        1.0 if sub_type == "biggan" else 0.0,
        1.0 if "inpaint" in sub_type or sub_type.startswith("imd2020") else 0.0,
        1.0 if sub_type.startswith("opensdi") else 0.0,
        1.0 if sub_type.startswith("aigen_proxy") else 0.0,
        1.0 if not sub_type else 0.0,
    ]
    return exact + family


def weighted_binary_scores(m: Dict, s: Dict) -> np.ndarray:
    p_auth_bin, p_manip_bin = mnv2_binary_probs(m)
    mnv2_scores = np.array([p_auth_bin, p_manip_bin], dtype=np.float32)
    specm_scores = np.array([
        float(s["authentic_score"]),
        float(s["manip_score"]),
    ], dtype=np.float32)
    w_m = 1.0 / max(float(m["confidence"]), 1e-3)
    w_s = 1.0 / max(float(s["confidence"]), 1e-3)
    combined = w_m * mnv2_scores + w_s * specm_scores
    total = float(combined.sum())
    if total > 0.0:
        combined /= total
    return combined


def icwmv_single(m: Dict, s: Dict) -> int:
    if is_ai_lock(m):
        return CLS2IDX["ai_generated"]
    return int(np.argmax(weighted_binary_scores(m, s)))


def build_veto_feature(m: Dict, s: Dict, feature_mode: str = "base") -> List[float]:
    p_auth = float(m["scores"]["authentic"])
    p_manip = float(m["scores"]["manipulated"])
    p_aigen = float(m["scores"].get("ai_generated", 0.0))
    p_auth_bin, p_manip_bin = mnv2_binary_probs(m)
    specm_a = float(s["authentic_score"])
    specm_m = float(s["manip_score"])
    ic_auth, ic_manip = weighted_binary_scores(m, s)
    w_m = 1.0 / max(float(m["confidence"]), 1e-3)
    w_s = 1.0 / max(float(s["confidence"]), 1e-3)
    ai_margin = p_aigen - max(p_auth, p_manip)
    mnv2_pred = mnv2_binary_pred_idx(m)
    ic_pred = int(ic_manip >= ic_auth)

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
        float(m["confidence"]),
        float(s["confidence"]),
        w_s / max(w_m, 1e-6),
        float(mnv2_pred),
        float(ic_pred),
        abs(specm_m - p_manip_bin),
    ] + (subtype_context_features(m) if feature_mode == "meta" else [])


def build_override_dataset(
    mnv2_recs: List[Dict],
    specm_recs: List[Dict],
    pos_weight: float,
    feature_mode: str = "base",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    feats, labels, weights = [], [], []
    stats = defaultdict(int)

    for m, s in zip(mnv2_recs, specm_recs):
        if is_ai_lock(m):
            stats["ai_lock"] += 1
            continue

        true_idx = CLS2IDX[m["true_label"]]
        if true_idx == CLS2IDX["ai_generated"]:
            stats["true_ai_generated"] += 1
            continue

        mnv2_bin = mnv2_binary_pred_idx(m)
        ic_pred = int(np.argmax(weighted_binary_scores(m, s)))
        if ic_pred == mnv2_bin:
            stats["no_override"] += 1
            continue

        keep_override = int(ic_pred == true_idx)
        feats.append(build_veto_feature(m, s, feature_mode=feature_mode))
        labels.append(keep_override)
        weights.append(pos_weight if keep_override else 1.0)
        stats["override_candidates"] += 1
        if keep_override:
            stats["beneficial_overrides"] += 1
        else:
            stats["harmful_overrides"] += 1

    if not feats:
        return (
            np.zeros((0, 16 + (16 if feature_mode == "meta" else 0)), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.float32),
            dict(stats),
        )

    return (
        np.array(feats, dtype=np.float32),
        np.array(labels, dtype=np.int64),
        np.array(weights, dtype=np.float32),
        dict(stats),
    )


def train_veto_model(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    sample_weight: np.ndarray,
    model_key: str,
):
    if len(x_tr) == 0:
        return ConstantKeepModel(prob_keep=1.0)

    uniq = sorted(set(y_tr.tolist()))
    if len(uniq) == 1:
        return ConstantKeepModel(prob_keep=float(uniq[0]))

    if model_key == "logreg":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=42,
            )),
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
            random_state=42,
            verbosity=0,
        )
        clf.fit(x_tr, y_tr, sample_weight=sample_weight)
        return clf

    raise ValueError(f"Unknown veto model: {model_key}")


def veto_keep_prob(model, m: Dict, s: Dict) -> float:
    feature_mode = "meta" if getattr(model, "_feature_mode", "base") == "meta" else "base"
    x = np.array([build_veto_feature(m, s, feature_mode=feature_mode)], dtype=np.float32)
    return float(model.predict_proba(x)[0, 1])


def apply_icwmv_veto(
    mnv2_recs: List[Dict],
    specm_recs: List[Dict],
    veto_model,
    tau: float,
) -> Tuple[np.ndarray, Dict[str, int]]:
    preds = []
    actions = defaultdict(int)

    for m, s in zip(mnv2_recs, specm_recs):
        if is_ai_lock(m):
            preds.append(CLS2IDX["ai_generated"])
            actions["ai_lock"] += 1
            continue

        mnv2_bin = mnv2_binary_pred_idx(m)
        ic_scores = weighted_binary_scores(m, s)
        ic_pred = int(np.argmax(ic_scores))

        if ic_pred == mnv2_bin:
            preds.append(ic_pred)
            actions["icwmv_no_override"] += 1
            continue

        p_keep = veto_keep_prob(veto_model, m, s)
        if p_keep >= tau:
            preds.append(ic_pred)
            actions["keep_icwmv_override"] += 1
        else:
            preds.append(mnv2_bin)
            actions["revert_to_mnv2"] += 1

    return np.array(preds, dtype=np.int64), dict(actions)


def apply_oracle_veto(
    mnv2_recs: List[Dict],
    specm_recs: List[Dict],
) -> Tuple[np.ndarray, Dict[str, int]]:
    preds = []
    actions = defaultdict(int)
    for m, s in zip(mnv2_recs, specm_recs):
        if is_ai_lock(m):
            preds.append(CLS2IDX["ai_generated"])
            actions["ai_lock"] += 1
            continue

        true_idx = CLS2IDX[m["true_label"]]
        mnv2_bin = mnv2_binary_pred_idx(m)
        ic_pred = int(np.argmax(weighted_binary_scores(m, s)))

        if ic_pred == mnv2_bin:
            preds.append(ic_pred)
            actions["icwmv_no_override"] += 1
            continue

        if true_idx == ic_pred:
            preds.append(ic_pred)
            actions["oracle_keep"] += 1
        else:
            preds.append(mnv2_bin)
            actions["oracle_revert"] += 1

    return np.array(preds, dtype=np.int64), dict(actions)


def eval_preds(preds: np.ndarray, mnv2_recs: List[Dict], actions: Optional[Dict[str, int]] = None) -> Dict:
    labels = np.array([CLS2IDX[m["true_label"]] for m in mnv2_recs])
    present = sorted(set(labels.tolist()))

    f1s = []
    for cls_idx in present:
        tp = int(((preds == cls_idx) & (labels == cls_idx)).sum())
        fp = int(((preds == cls_idx) & (labels != cls_idx)).sum())
        fn = int(((preds != cls_idx) & (labels == cls_idx)).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1s.append(2 * precision * recall / max(precision + recall, 1e-8))

    n_err = n_corr = n_broken = 0
    patterns = defaultdict(lambda: {"total": 0, "corrected": 0, "broken": 0})
    for i, m in enumerate(mnv2_recs):
        true_label = m["true_label"]
        pred_label = m["pred_label"]
        fused_label = IDX2CLS[int(preds[i])]

        if true_label == "ai_generated" or pred_label == "ai_generated":
            continue

        if pred_label != true_label:
            n_err += 1
            pat = f"{true_label}→{pred_label}"
            patterns[pat]["total"] += 1
            if fused_label == true_label:
                n_corr += 1
                patterns[pat]["corrected"] += 1
        elif fused_label != true_label:
            n_broken += 1
            pat = f"{true_label}✓"
            patterns[pat]["broken"] += 1

    for p in patterns.values():
        total = p.get("total", 0)
        p["rate"] = round(p.get("corrected", 0) / max(total, 1), 4)

    return {
        "macro_f1": round(float(np.mean(f1s)), 4),
        "accuracy": round(float((preds == labels).mean()), 4),
        "n": int(len(labels)),
        "err_corr": {
            "n_errors": n_err,
            "n_corrected": n_corr,
            "rate": round(n_corr / max(n_err, 1), 4),
            "n_broken": n_broken,
            "net_gain": int(n_corr - n_broken),
            "patterns": dict(patterns),
        },
        "actions": actions or {},
    }


def concat_override_data(
    aligned: Dict[Tuple[str, str], Tuple[List[Dict], List[Dict]]],
    train_dss: Sequence[str],
    specm_model: str,
    pos_weight: float,
    feature_mode: str = "base",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    xs, ys, ws = [], [], []
    agg_stats = defaultdict(int)
    for ds_name in train_dss:
        mnv2_recs, specm_recs = aligned[(ds_name, specm_model)]
        x_ds, y_ds, w_ds, stats = build_override_dataset(
            mnv2_recs,
            specm_recs,
            pos_weight,
            feature_mode=feature_mode,
        )
        if len(x_ds):
            xs.append(x_ds)
            ys.append(y_ds)
            ws.append(w_ds)
        for key, value in stats.items():
            agg_stats[key] += value

    if not xs:
        return (
            np.zeros((0, 16 + (16 if feature_mode == "meta" else 0)), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.float32),
            dict(agg_stats),
        )

    return (
        np.vstack(xs),
        np.hstack(ys),
        np.hstack(ws),
        dict(agg_stats),
    )


def tune_veto(
    aligned: Dict[Tuple[str, str], Tuple[List[Dict], List[Dict]]],
    train_dss: Sequence[str],
    specm_model: str,
    taus: Sequence[float],
    pos_weights: Sequence[float],
    model_keys: Sequence[str],
) -> Dict[str, float]:
    best = None

    for pos_weight in pos_weights:
        for model_key in model_keys:
            feature_mode = "meta" if model_key == "xgb_meta_depth2" else "base"
            x_all, y_all, w_all, stats = concat_override_data(
                aligned,
                train_dss,
                specm_model,
                pos_weight,
                feature_mode=feature_mode,
            )
            if len(x_all) == 0:
                continue

            for tau in taus:
                inner_f1s, inner_corrs, inner_gain = [], [], []
                for val_ds in train_dss:
                    inner_train = [d for d in train_dss if d != val_ds]
                    x_tr, y_tr, w_tr, _ = concat_override_data(
                        aligned,
                        inner_train,
                        specm_model,
                        pos_weight,
                        feature_mode=feature_mode,
                    )
                    veto_model = train_veto_model(x_tr, y_tr, w_tr, model_key)
                    setattr(veto_model, "_feature_mode", feature_mode)
                    mnv2_val, specm_val = aligned[(val_ds, specm_model)]
                    preds, actions = apply_icwmv_veto(mnv2_val, specm_val, veto_model, tau=tau)
                    res = eval_preds(preds, mnv2_val, actions)
                    inner_f1s.append(res["macro_f1"])
                    inner_corrs.append(res["err_corr"]["rate"])
                    inner_gain.append(res["err_corr"]["net_gain"])

                cand = {
                    "tau": float(tau),
                    "pos_weight": float(pos_weight),
                    "model_key": model_key,
                    "avg_f1": float(np.mean(inner_f1s)),
                    "avg_corr": float(np.mean(inner_corrs)),
                    "avg_net_gain": float(np.mean(inner_gain)),
                    "train_override_candidates": int(stats.get("override_candidates", 0)),
                    "train_keep_rate": round(
                        float(stats.get("beneficial_overrides", 0)) / max(float(stats.get("override_candidates", 0)), 1.0),
                        4,
                    ),
                }
                key = (cand["avg_f1"], cand["avg_corr"], cand["avg_net_gain"])
                if best is None or key > (best["avg_f1"], best["avg_corr"], best["avg_net_gain"]):
                    best = cand

    if best is None:
        best = {
            "tau": 0.5,
            "pos_weight": 1.0,
            "model_key": "logreg",
            "avg_f1": 0.0,
            "avg_corr": 0.0,
            "avg_net_gain": 0.0,
            "train_override_candidates": 0,
            "train_keep_rate": 0.0,
        }
    return best


def run_single_mnv2(
    mnv2_label: str,
    ts_be: str,
    specm_models: Sequence[str],
    taus: Sequence[float],
    pos_weights: Sequence[float],
    model_keys: Sequence[str],
) -> Dict:
    print(f"\n{'=' * 78}")
    print(f"  HEMA ICWMV-Veto LOO-CD | MNV2={mnv2_label} ({ts_be})")
    print(f"{'=' * 78}")

    mnv2_data: Dict[str, List[Dict]] = {}
    aligned: Dict[Tuple[str, str], Tuple[List[Dict], List[Dict]]] = {}

    print("\n  [데이터 로드]")
    for ds_name in DATASETS:
        mnv2_data[ds_name] = load_mnv2(ds_name, ts_be)
        print(f"    MNV2  [{ds_name}] n={len(mnv2_data[ds_name])}")

    valid_specm_models = []
    for model_key in specm_models:
        coverage_ok = True
        for ds_name in DATASETS:
            specm_recs = load_specm(model_key, ds_name)
            if specm_recs is None:
                coverage_ok = False
                print(f"    SpecM [{model_key}][{ds_name}] 없음")
                break
            aligned_m, aligned_s = align_records(mnv2_data[ds_name], specm_recs)
            if len(aligned_m) != len(mnv2_data[ds_name]):
                coverage_ok = False
                print(
                    f"    SpecM [{model_key}][{ds_name}] aligned={len(aligned_m)} / mnv2={len(mnv2_data[ds_name])} "
                    "-> full coverage 아님, 제외"
                )
                break
            aligned[(ds_name, model_key)] = (aligned_m, aligned_s)
            print(f"    SpecM [{model_key}][{ds_name}] aligned={len(aligned_m)} / mnv2={len(mnv2_data[ds_name])}")
        if coverage_ok:
            valid_specm_models.append(model_key)

    results = {
        "mnv2_label": mnv2_label,
        "mnv2_ts": ts_be,
        "models": {},
    }

    for model_key in valid_specm_models:
        print(f"\n  [SpecM={model_key}]")
        icwmv_f1s, icwmv_corrs = [], []
        veto_f1s, veto_corrs = [], []
        oracle_f1s, oracle_corrs = [], []
        model_res = {
            "icwmv": {"per_ds": {}},
            "hema_icwmv_veto": {"per_ds": {}},
            "oracle_veto": {"per_ds": {}},
        }

        for test_ds in DATASETS:
            train_dss = [d for d in DATASETS if d != test_ds]
            best_cfg = tune_veto(
                aligned,
                train_dss,
                model_key,
                taus=taus,
                pos_weights=pos_weights,
                model_keys=model_keys,
            )
            feature_mode = "meta" if best_cfg["model_key"] == "xgb_meta_depth2" else "base"
            x_tr, y_tr, w_tr, _ = concat_override_data(
                aligned,
                train_dss,
                model_key,
                best_cfg["pos_weight"],
                feature_mode=feature_mode,
            )
            veto_model = train_veto_model(x_tr, y_tr, w_tr, best_cfg["model_key"])
            setattr(veto_model, "_feature_mode", feature_mode)

            mnv2_test, specm_test = aligned[(test_ds, model_key)]
            icwmv_preds = np.array([icwmv_single(m, s) for m, s in zip(mnv2_test, specm_test)], dtype=np.int64)
            veto_preds, veto_actions = apply_icwmv_veto(mnv2_test, specm_test, veto_model, tau=best_cfg["tau"])
            oracle_preds, oracle_actions = apply_oracle_veto(mnv2_test, specm_test)

            icwmv_res = eval_preds(icwmv_preds, mnv2_test, {"ai_lock_or_icwmv": len(mnv2_test)})
            veto_res = eval_preds(veto_preds, mnv2_test, veto_actions)
            oracle_res = eval_preds(oracle_preds, mnv2_test, oracle_actions)
            veto_res["best_cfg"] = best_cfg

            model_res["icwmv"]["per_ds"][test_ds] = icwmv_res
            model_res["hema_icwmv_veto"]["per_ds"][test_ds] = veto_res
            model_res["oracle_veto"]["per_ds"][test_ds] = oracle_res

            icwmv_f1s.append(icwmv_res["macro_f1"])
            icwmv_corrs.append(icwmv_res["err_corr"]["rate"])
            veto_f1s.append(veto_res["macro_f1"])
            veto_corrs.append(veto_res["err_corr"]["rate"])
            oracle_f1s.append(oracle_res["macro_f1"])
            oracle_corrs.append(oracle_res["err_corr"]["rate"])

            print(
                f"    [{test_ds:10s}] "
                f"ICWMV F1={icwmv_res['macro_f1']:.4f} corr={icwmv_res['err_corr']['rate']:.3f} | "
                f"Veto F1={veto_res['macro_f1']:.4f} corr={veto_res['err_corr']['rate']:.3f} | "
                f"Oracle F1={oracle_res['macro_f1']:.4f} corr={oracle_res['err_corr']['rate']:.3f} "
                f"(tau={best_cfg['tau']:.2f}, w+={best_cfg['pos_weight']:.1f}, model={best_cfg['model_key']})"
            )

        model_res["icwmv"]["avg_f1"] = round(float(np.mean(icwmv_f1s)), 4)
        model_res["icwmv"]["avg_corr"] = round(float(np.mean(icwmv_corrs)), 4)
        model_res["icwmv"]["avg_net_gain"] = round(
            float(np.mean([
                model_res["icwmv"]["per_ds"][ds]["err_corr"]["net_gain"] for ds in DATASETS
            ])),
            4,
        )
        model_res["hema_icwmv_veto"]["avg_f1"] = round(float(np.mean(veto_f1s)), 4)
        model_res["hema_icwmv_veto"]["avg_corr"] = round(float(np.mean(veto_corrs)), 4)
        model_res["hema_icwmv_veto"]["avg_net_gain"] = round(
            float(np.mean([
                model_res["hema_icwmv_veto"]["per_ds"][ds]["err_corr"]["net_gain"] for ds in DATASETS
            ])),
            4,
        )
        model_res["oracle_veto"]["avg_f1"] = round(float(np.mean(oracle_f1s)), 4)
        model_res["oracle_veto"]["avg_corr"] = round(float(np.mean(oracle_corrs)), 4)
        model_res["oracle_veto"]["avg_net_gain"] = round(
            float(np.mean([
                model_res["oracle_veto"]["per_ds"][ds]["err_corr"]["net_gain"] for ds in DATASETS
            ])),
            4,
        )

        results["models"][model_key] = model_res
        print(
            f"    -> avg ICWMV F1={model_res['icwmv']['avg_f1']:.4f} corr={model_res['icwmv']['avg_corr']:.3f} | "
            f"Veto F1={model_res['hema_icwmv_veto']['avg_f1']:.4f} corr={model_res['hema_icwmv_veto']['avg_corr']:.3f} | "
            f"Oracle F1={model_res['oracle_veto']['avg_f1']:.4f} corr={model_res['oracle_veto']['avg_corr']:.3f} "
            f"ΔF1={model_res['hema_icwmv_veto']['avg_f1'] - model_res['icwmv']['avg_f1']:+.4f} "
            f"Δcorr={model_res['hema_icwmv_veto']['avg_corr'] - model_res['icwmv']['avg_corr']:+.3f}"
        )

    return results


def main() -> None:
    args = parse_args()
    mnv2_versions = resolve_mnv2_versions(args.mnv2)

    final_results = {
        "config": {
            "mnv2_versions": mnv2_versions,
            "specm_models": list(args.specm_models),
            "taus": list(args.taus),
            "pos_weights": list(args.pos_weights),
            "models": list(args.models),
        },
        "results": {},
    }

    for label, ts_be in mnv2_versions.items():
        final_results["results"][label] = run_single_mnv2(
            mnv2_label=label,
            ts_be=ts_be,
            specm_models=args.specm_models,
            taus=args.taus,
            pos_weights=args.pos_weights,
            model_keys=args.models,
        )

    out_dir = ROOT / "experiments" / "results" / "hema_icwmv_veto"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"hema_icwmv_veto_loo_cd_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(final_results, f, indent=2)

    print(f"\n저장: {out_path}")


if __name__ == "__main__":
    main()
