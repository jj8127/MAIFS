#!/usr/bin/env python3
"""
HEMA Action-Gate LOO-CD Comparison
==================================

아이디어:
  1. ai_generated는 MNV2가 맡는다 (ai-gen lock).
  2. auth/manip 구간에서도 MNV2와 SpecM이 "disagree"하는 샘플만 learned gate가 본다.
  3. gate는 최종 3-class를 직접 맞히지 않고,
     "keep MNV2" vs "override with SpecM"만 예측한다.
  4. gate가 애매하면 ICWMV로 fallback 한다.

목표:
  기존 HEMA-XGBoost처럼 모든 샘플에서 3-class를 재분류하지 않고,
  실제로 override가 필요한 slice만 학습해서 ICWMV보다
  더 높은 교정률과 macro-F1을 동시에 노린다.

실행:
  .venv-qwen/bin/python experiments/run_hema_action_gate_loo_cd.py
  .venv-qwen/bin/python experiments/run_hema_action_gate_loo_cd.py --mnv2 strong
  .venv-qwen/bin/python experiments/run_hema_action_gate_loo_cd.py --mnv2 weak --specm-models v4 comp_noTS
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

MNV2_PRESETS = {
    "strong": "20260319_070725",
    "weak": "20260319_064748",
}


@dataclass
class ConstantGate:
    prob_override: float

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        n = len(x)
        p = np.full(n, self.prob_override, dtype=np.float32)
        return np.stack([1.0 - p, p], axis=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mnv2",
        nargs="+",
        default=["strong", "weak"],
        help="MNV2 버전 preset(strong/weak) 또는 직접 timestamp",
    )
    parser.add_argument(
        "--specm-models",
        nargs="+",
        default=["v4", "comp_noTS", "comp_g1"],
        help="비교할 SpecM 모델 키",
    )
    parser.add_argument(
        "--deltas",
        nargs="+",
        type=float,
        default=[0.0, 0.05, 0.10, 0.15],
        help="gate 불확실 구간 폭: |p-0.5| <= delta 이면 ICWMV fallback",
    )
    parser.add_argument(
        "--pos-weights",
        nargs="+",
        type=float,
        default=[2.0, 3.0, 4.0],
        help="override-positive 샘플 가중치 후보",
    )
    parser.add_argument(
        "--c-grid",
        nargs="+",
        type=float,
        default=[0.25, 1.0, 4.0],
        help="LogisticRegression C 후보",
    )
    return parser.parse_args()


def resolve_mnv2_versions(raw_values: Sequence[str]) -> Dict[str, str]:
    resolved = {}
    for value in raw_values:
        if value in MNV2_PRESETS:
            resolved[value] = MNV2_PRESETS[value]
        else:
            resolved[value] = value
    return resolved


def load_mnv2(ds_name: str, ts_be: str) -> List[Dict]:
    path = BE_DIR / f"mobilenetv2_dualstream_{ds_name}_{ts_be}.jsonl"
    with open(path) as f:
        return [json.loads(line) for line in f]


def find_specm_jsonl(model_key: str, ds_name: str) -> Optional[Path]:
    # 1) full-coverage eval results 우선
    cands = sorted(SPECM_EVAL_DIR.glob(f"specm_{model_key}_{ds_name}_*.jsonl"))
    if cands:
        return cands[-1]

    # 2) 과거 complementary subset 결과 fallback (공정 비교용으로는 비권장)
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


def mnv2_binary_pred(m: Dict) -> str:
    p_auth_bin, p_manip_bin = mnv2_binary_probs(m)
    return "manipulated" if p_manip_bin >= p_auth_bin else "authentic"


def specm_pred(s: Dict) -> str:
    return "manipulated" if s["manip_score"] >= s["authentic_score"] else "authentic"


def is_ai_lock(m: Dict, threshold: float = 0.5) -> bool:
    p_aigen = float(m["scores"].get("ai_generated", 0.0))
    return m["pred_label"] == "ai_generated" or p_aigen > threshold


def mnv2_entropy(m: Dict) -> float:
    probs = np.array([
        float(m["scores"]["authentic"]),
        float(m["scores"]["manipulated"]),
        float(m["scores"].get("ai_generated", 0.0)),
    ])
    return float(-(probs * np.log(probs + 1e-8)).sum())


def specm_entropy(s: Dict) -> float:
    p = np.array([float(s["authentic_score"]), float(s["manip_score"])])
    return float(-(p * np.log(p + 1e-8)).sum())


def icwmv_single(m: Dict, s: Dict) -> int:
    p_aigen = float(m["scores"].get("ai_generated", 0.0))
    if m["pred_label"] == "ai_generated" or p_aigen > 0.5:
        return CLS2IDX["ai_generated"]

    p_auth_bin, p_manip_bin = mnv2_binary_probs(m)
    mnv2_scores = np.array([p_auth_bin, p_manip_bin, 0.0], dtype=np.float32)
    specm_scores = np.array([
        float(s["authentic_score"]),
        float(s["manip_score"]),
        0.0,
    ], dtype=np.float32)
    w_m = 1.0 / max(float(m["confidence"]), 1e-3)
    w_s = 1.0 / max(float(s["confidence"]), 1e-3)
    combined = w_m * mnv2_scores + w_s * specm_scores
    return int(np.argmax(combined[:2]))


def build_gate_feature(m: Dict, s: Dict) -> List[float]:
    p_auth = float(m["scores"]["authentic"])
    p_manip = float(m["scores"]["manipulated"])
    p_aigen = float(m["scores"].get("ai_generated", 0.0))
    p_auth_bin, p_manip_bin = mnv2_binary_probs(m)
    specm_a = float(s["authentic_score"])
    specm_m = float(s["manip_score"])
    disagree_dir = 1.0 if (specm_m >= specm_a and p_auth_bin > p_manip_bin) else -1.0
    ai_margin = p_aigen - max(p_auth, p_manip)

    return [
        p_auth_bin,
        p_manip_bin,
        p_aigen,
        float(m["confidence"]),
        p_manip_bin - p_auth_bin,
        mnv2_entropy(m),
        specm_a,
        specm_m,
        float(s["confidence"]),
        specm_m - specm_a,
        specm_m - p_manip_bin,
        specm_a - p_auth_bin,
        ai_margin,
        disagree_dir,
    ]


def build_gate_dataset(
    mnv2_recs: List[Dict],
    specm_recs: List[Dict],
    pos_weight: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    feats, labels, weights = [], [], []
    stats = defaultdict(int)

    for m, s in zip(mnv2_recs, specm_recs):
        if is_ai_lock(m):
            stats["ai_locked"] += 1
            continue

        m_bin = mnv2_binary_pred(m)
        s_bin = specm_pred(s)
        if m_bin == s_bin:
            stats["agreement"] += 1
            continue

        true_label = m["true_label"]
        override_good = (
            true_label in ("authentic", "manipulated")
            and s_bin == true_label
            and m_bin != true_label
        )
        harmful_override = (
            true_label == "ai_generated"
            or (true_label in ("authentic", "manipulated") and m_bin == true_label)
        )

        feats.append(build_gate_feature(m, s))
        labels.append(1 if override_good else 0)
        if override_good:
            weights.append(pos_weight)
            stats["override_positive"] += 1
        elif harmful_override:
            weights.append(2.0)
            stats["override_negative_harmful"] += 1
        else:
            weights.append(1.0)
            stats["override_negative_neutral"] += 1

    if not feats:
        return (
            np.zeros((0, 14), dtype=np.float32),
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


def train_gate(x_tr: np.ndarray, y_tr: np.ndarray, sample_weight: np.ndarray, c_value: float):
    if len(x_tr) == 0:
        return ConstantGate(prob_override=0.0)

    uniq = sorted(set(y_tr.tolist()))
    if len(uniq) == 1:
        return ConstantGate(prob_override=float(uniq[0]))

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=c_value,
                max_iter=1000,
                random_state=42,
            )),
        ])
        clf.fit(x_tr, y_tr, clf__sample_weight=sample_weight)
        return clf
    except Exception:
        from sklearn.ensemble import RandomForestClassifier

        clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=5,
            min_samples_leaf=8,
            random_state=42,
            n_jobs=-1,
        )
        clf.fit(x_tr, y_tr, sample_weight=sample_weight)
        return clf


def gate_proba(gate, m: Dict, s: Dict) -> float:
    feat = np.array([build_gate_feature(m, s)], dtype=np.float32)
    return float(gate.predict_proba(feat)[0, 1])


def apply_action_gate(
    mnv2_recs: List[Dict],
    specm_recs: List[Dict],
    gate,
    delta: float,
) -> Tuple[np.ndarray, Dict[str, int]]:
    preds = []
    actions = defaultdict(int)

    for m, s in zip(mnv2_recs, specm_recs):
        if is_ai_lock(m):
            preds.append(CLS2IDX["ai_generated"])
            actions["ai_lock"] += 1
            continue

        m_bin = mnv2_binary_pred(m)
        s_bin = specm_pred(s)
        if m_bin == s_bin:
            preds.append(CLS2IDX[m_bin])
            actions["agreement_keep"] += 1
            continue

        p_override = gate_proba(gate, m, s)
        if p_override >= 0.5 + delta:
            preds.append(CLS2IDX[s_bin])
            actions["override_specm"] += 1
        elif p_override <= 0.5 - delta:
            preds.append(CLS2IDX[m_bin])
            actions["keep_mnv2"] += 1
        else:
            preds.append(icwmv_single(m, s))
            actions["fallback_icwmv"] += 1

    return np.array(preds, dtype=np.int64), dict(actions)


def eval_preds(preds: np.ndarray, mnv2_recs: List[Dict], actions: Optional[Dict[str, int]] = None) -> Dict:
    labels = np.array([CLS2IDX[m["true_label"]] for m in mnv2_recs])
    present = sorted(set(labels.tolist()))

    f1s = []
    per_class = {}
    for cls_idx in present:
        tp = int(((preds == cls_idx) & (labels == cls_idx)).sum())
        fp = int(((preds == cls_idx) & (labels != cls_idx)).sum())
        fn = int(((preds != cls_idx) & (labels == cls_idx)).sum())
        pr = tp / max(tp + fp, 1)
        rc = tp / max(tp + fn, 1)
        f1 = 2 * pr * rc / max(pr + rc, 1e-8)
        f1s.append(f1)
        per_class[IDX2CLS[cls_idx]] = {
            "f1": round(f1, 4),
            "precision": round(pr, 4),
            "recall": round(rc, 4),
            "n": int((labels == cls_idx).sum()),
        }

    n_err = n_corr = n_broken = 0
    patterns = defaultdict(lambda: {"total": 0, "corrected": 0, "broken": 0})
    for i, m in enumerate(mnv2_recs):
        true_label = m["true_label"]
        pred_label = m["pred_label"]
        if true_label == "ai_generated" or pred_label == "ai_generated":
            continue
        fused_label = IDX2CLS[int(preds[i])]

        if pred_label != true_label:
            n_err += 1
            pat = f"{true_label}→{pred_label}"
            patterns[pat]["total"] += 1
            if fused_label == true_label:
                n_corr += 1
                patterns[pat]["corrected"] += 1
        else:
            if fused_label != true_label:
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
        "per_class": per_class,
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


def concat_gate_data(
    aligned: Dict[Tuple[str, str], Tuple[List[Dict], List[Dict]]],
    train_dss: Sequence[str],
    specm_model: str,
    pos_weight: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    xs, ys, ws = [], [], []
    agg_stats = defaultdict(int)
    for ds_name in train_dss:
        mnv2_recs, specm_recs = aligned[(ds_name, specm_model)]
        x_ds, y_ds, w_ds, stats = build_gate_dataset(mnv2_recs, specm_recs, pos_weight=pos_weight)
        if len(x_ds):
            xs.append(x_ds)
            ys.append(y_ds)
            ws.append(w_ds)
        for key, value in stats.items():
            agg_stats[key] += value

    if not xs:
        return (
            np.zeros((0, 14), dtype=np.float32),
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


def tune_action_gate(
    aligned: Dict[Tuple[str, str], Tuple[List[Dict], List[Dict]]],
    train_dss: Sequence[str],
    specm_model: str,
    deltas: Sequence[float],
    pos_weights: Sequence[float],
    c_grid: Sequence[float],
) -> Dict[str, float]:
    best = None
    for pos_weight in pos_weights:
        for c_value in c_grid:
            x_tr_all, y_tr_all, w_tr_all, stats = concat_gate_data(
                aligned, train_dss, specm_model, pos_weight=pos_weight
            )
            if len(x_tr_all) == 0:
                continue

            for delta in deltas:
                inner_f1s, inner_corrs, inner_gain = [], [], []
                for val_ds in train_dss:
                    inner_train_dss = [d for d in train_dss if d != val_ds]
                    x_tr, y_tr, w_tr, _ = concat_gate_data(
                        aligned, inner_train_dss, specm_model, pos_weight=pos_weight
                    )
                    gate = train_gate(x_tr, y_tr, w_tr, c_value=c_value)
                    mnv2_val, specm_val = aligned[(val_ds, specm_model)]
                    preds, actions = apply_action_gate(mnv2_val, specm_val, gate, delta=delta)
                    res = eval_preds(preds, mnv2_val, actions)
                    inner_f1s.append(res["macro_f1"])
                    inner_corrs.append(res["err_corr"]["rate"])
                    inner_gain.append(res["err_corr"]["net_gain"])

                cand = {
                    "delta": float(delta),
                    "pos_weight": float(pos_weight),
                    "c_value": float(c_value),
                    "avg_f1": float(np.mean(inner_f1s)),
                    "avg_corr": float(np.mean(inner_corrs)),
                    "avg_net_gain": float(np.mean(inner_gain)),
                    "train_disagree": int(stats.get("override_positive", 0) + stats.get("override_negative_harmful", 0) + stats.get("override_negative_neutral", 0)),
                }
                key = (cand["avg_f1"], cand["avg_corr"], cand["avg_net_gain"])
                if best is None or key > (best["avg_f1"], best["avg_corr"], best["avg_net_gain"]):
                    best = cand

    if best is None:
        best = {
            "delta": 0.0,
            "pos_weight": 2.0,
            "c_value": 1.0,
            "avg_f1": 0.0,
            "avg_corr": 0.0,
            "avg_net_gain": 0.0,
            "train_disagree": 0,
        }
    return best


def run_single_mnv2(
    mnv2_label: str,
    ts_be: str,
    specm_models: Sequence[str],
    deltas: Sequence[float],
    pos_weights: Sequence[float],
    c_grid: Sequence[float],
) -> Dict:
    print(f"\n{'=' * 78}")
    print(f"  HEMA Action-Gate LOO-CD | MNV2={mnv2_label} ({ts_be})")
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
                continue
            am, as_ = align_records(mnv2_data[ds_name], specm_recs)
            aligned[(ds_name, model_key)] = (am, as_)
            print(f"    SpecM [{model_key}][{ds_name}] aligned={len(am)} / mnv2={len(mnv2_data[ds_name])}")
            if len(am) != len(mnv2_data[ds_name]):
                coverage_ok = False
        if coverage_ok:
            valid_specm_models.append(model_key)
        else:
            print(f"    -> [{model_key}] full coverage 아님: 공정 비교에서 제외")

    if not valid_specm_models:
        raise RuntimeError("full-coverage SpecM 결과가 없습니다. run_specm_eval.py로 먼저 생성해 주세요.")

    results = {
        "mnv2_label": mnv2_label,
        "mnv2_ts": ts_be,
        "mnv2_only": None,
        "models": {},
    }

    # MNV2-only baseline
    mnv2_per_ds = {}
    for ds_name in DATASETS:
        preds = np.array([CLS2IDX[r["pred_label"]] for r in mnv2_data[ds_name]], dtype=np.int64)
        mnv2_per_ds[ds_name] = eval_preds(preds, mnv2_data[ds_name], {"mnv2_only": len(preds)})
    results["mnv2_only"] = {
        "avg_f1": round(float(np.mean([r["macro_f1"] for r in mnv2_per_ds.values()])), 4),
        "avg_corr": 0.0,
        "avg_net_gain": 0.0,
        "per_ds": mnv2_per_ds,
    }

    for model_key in valid_specm_models:
        print(f"\n  [SpecM={model_key}]")
        icwmv_per_ds = {}
        gate_per_ds = {}
        selected_params = {}

        for test_ds in DATASETS:
            train_dss = [d for d in DATASETS if d != test_ds]
            mnv2_te, specm_te = aligned[(test_ds, model_key)]

            # Fixed-rule baseline
            icwmv_preds = np.array([icwmv_single(m, s) for m, s in zip(mnv2_te, specm_te)], dtype=np.int64)
            icwmv_actions = {"ai_lock_or_icwmv": len(icwmv_preds)}
            icwmv_per_ds[test_ds] = eval_preds(icwmv_preds, mnv2_te, icwmv_actions)

            tuned = tune_action_gate(
                aligned,
                train_dss=train_dss,
                specm_model=model_key,
                deltas=deltas,
                pos_weights=pos_weights,
                c_grid=c_grid,
            )
            selected_params[test_ds] = tuned

            x_tr, y_tr, w_tr, stats = concat_gate_data(
                aligned,
                train_dss=train_dss,
                specm_model=model_key,
                pos_weight=tuned["pos_weight"],
            )
            gate = train_gate(x_tr, y_tr, w_tr, c_value=tuned["c_value"])
            preds, actions = apply_action_gate(mnv2_te, specm_te, gate, delta=tuned["delta"])
            res = eval_preds(preds, mnv2_te, actions)
            res["train_gate_stats"] = stats
            gate_per_ds[test_ds] = res

            print(
                f"    [{test_ds:10s}] "
                f"ICWMV F1={icwmv_per_ds[test_ds]['macro_f1']:.4f} corr={icwmv_per_ds[test_ds]['err_corr']['rate']:.3f} | "
                f"Gate F1={res['macro_f1']:.4f} corr={res['err_corr']['rate']:.3f} "
                f"(delta={tuned['delta']:.2f}, pos_w={tuned['pos_weight']:.1f}, C={tuned['c_value']:.2f})"
            )

        icwmv_avg_f1 = float(np.mean([r["macro_f1"] for r in icwmv_per_ds.values()]))
        icwmv_avg_corr = float(np.mean([r["err_corr"]["rate"] for r in icwmv_per_ds.values()]))
        icwmv_avg_gain = float(np.mean([r["err_corr"]["net_gain"] for r in icwmv_per_ds.values()]))

        gate_avg_f1 = float(np.mean([r["macro_f1"] for r in gate_per_ds.values()]))
        gate_avg_corr = float(np.mean([r["err_corr"]["rate"] for r in gate_per_ds.values()]))
        gate_avg_gain = float(np.mean([r["err_corr"]["net_gain"] for r in gate_per_ds.values()]))

        print(
            f"    -> avg ICWMV F1={icwmv_avg_f1:.4f} corr={icwmv_avg_corr:.3f} | "
            f"Gate F1={gate_avg_f1:.4f} corr={gate_avg_corr:.3f} "
            f"ΔF1={gate_avg_f1 - icwmv_avg_f1:+.4f} Δcorr={gate_avg_corr - icwmv_avg_corr:+.3f}"
        )

        results["models"][model_key] = {
            "icwmv": {
                "avg_f1": round(icwmv_avg_f1, 4),
                "avg_corr": round(icwmv_avg_corr, 4),
                "avg_net_gain": round(icwmv_avg_gain, 2),
                "per_ds": icwmv_per_ds,
            },
            "hema_action_gate": {
                "avg_f1": round(gate_avg_f1, 4),
                "avg_corr": round(gate_avg_corr, 4),
                "avg_net_gain": round(gate_avg_gain, 2),
                "per_ds": gate_per_ds,
                "selected_params": selected_params,
            },
        }

    return results


def main():
    args = parse_args()
    mnv2_versions = resolve_mnv2_versions(args.mnv2)
    all_results = {}

    for label, ts_be in mnv2_versions.items():
        all_results[label] = run_single_mnv2(
            mnv2_label=label,
            ts_be=ts_be,
            specm_models=args.specm_models,
            deltas=args.deltas,
            pos_weights=args.pos_weights,
            c_grid=args.c_grid,
        )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "experiments" / "results" / "hema_action_gate"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"hema_action_gate_loo_cd_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "config": {
                    "mnv2_versions": mnv2_versions,
                    "specm_models": args.specm_models,
                    "deltas": args.deltas,
                    "pos_weights": args.pos_weights,
                    "c_grid": args.c_grid,
                },
                "results": all_results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\n저장: {out_path}")


if __name__ == "__main__":
    main()
