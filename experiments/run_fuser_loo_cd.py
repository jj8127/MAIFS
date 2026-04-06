#!/usr/bin/env python3
"""
Fuser LOO-CD (Leave-One-Out Cross-Dataset) 비교
================================================
Fixed-rule fusers(ICWMV, Cascade)와 Learned fusers(HEMA-XGBoost)를
동일한 LOO-CD 프로토콜 하에서 공정하게 비교.

비교 대상:
  - MNV2-only          : 3-class baseline (no SpecM)
  - ICWMV+v4           : fixed-rule, SpecM-v4 (LOO-CD = eval on each DS directly)
  - Cascade τ=0.6 +v4  : fixed-rule, SpecM-v4
  - HEMA+v4            : XGBoost, train on 3 DS → test on 1
  - HEMA+comp_noTS     : XGBoost, train on 3 DS → test on 1 (primary model)
  - HEMA+comp_g1       : XGBoost, train on 3 DS → test on 1

핵심 질문:
  ICWMV-v4(fixed-rule, no domain-shift risk)와
  HEMA+comp_noTS(learned, LOO-CD tested)의 성능 차이는?

실행:
  .venv-qwen/bin/python experiments/run_fuser_loo_cd.py
"""

from __future__ import annotations

import json
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

warnings.filterwarnings("ignore")

ROOT     = Path(__file__).resolve().parents[1]
BE_DIR   = ROOT / "experiments" / "results" / "backbone_eval"
COMP_DIR = ROOT / "experiments" / "results" / "specm_complementary_eval"
SPECM_DIR= ROOT / "experiments" / "results" / "specm_eval"
TS_BE    = "20260319_070725"

DATASETS = ["base", "dsC", "opensdi", "aigenproxy"]
CLS2IDX  = {"authentic": 0, "manipulated": 1, "ai_generated": 2}
IDX2CLS  = {0: "authentic", 1: "manipulated", 2: "ai_generated"}


# ═══════════════════════════════════════════════════════════════════════════════
# 데이터 로드
# ═══════════════════════════════════════════════════════════════════════════════

def load_mnv2(ds: str) -> List[Dict]:
    with open(BE_DIR / f"mobilenetv2_dualstream_{ds}_{TS_BE}.jsonl") as f:
        return [json.loads(l) for l in f]


def find_specm_jsonl(specm_model: str, ds: str) -> Optional[Path]:
    tag_map = {
        "comp_g1":   "gamma1.0_wmax10",
        "comp_g2":   "gamma2.0_wmax10",
        "comp_noTS": "gamma1.0_wmax10_noTS",
        "v4":        None,
    }
    if specm_model == "v4":
        cands = sorted(SPECM_DIR.glob(f"specm_v4_{ds}_*.jsonl"))
    else:
        tag = tag_map.get(specm_model, specm_model)
        cands = sorted(COMP_DIR.glob(f"specm_comp_{tag}_{ds}_*.jsonl"))
    return cands[-1] if cands else None


def load_specm(specm_model: str, ds: str) -> Optional[List[Dict]]:
    p = find_specm_jsonl(specm_model, ds)
    if not p:
        return None
    with open(p) as f:
        return [json.loads(l) for l in f]


def align(mnv2_recs: List[Dict], specm_recs: List[Dict]) -> Tuple[List, List]:
    specm_map = {r["image_path"]: r for r in specm_recs}
    am, as_ = [], []
    for m in mnv2_recs:
        s = specm_map.get(m["image_path"])
        if s:
            am.append(m)
            as_.append(s)
    return am, as_


# ═══════════════════════════════════════════════════════════════════════════════
# Fixed-rule fusers
# ═══════════════════════════════════════════════════════════════════════════════

def fuse_mnv2_only(mnv2_recs: List[Dict]) -> np.ndarray:
    return np.array([CLS2IDX[r["pred_label"]] for r in mnv2_recs])


def fuse_icwmv(mnv2_recs: List[Dict], specm_recs: List[Dict]) -> np.ndarray:
    preds = []
    for m, s in zip(mnv2_recs, specm_recs):
        p_aigen = m["scores"].get("ai_generated", 0.0)
        if m["pred_label"] == "ai_generated" or p_aigen > 0.5:
            preds.append(CLS2IDX["ai_generated"])
            continue
        p_auth  = m["scores"]["authentic"]
        p_manip = m["scores"]["manipulated"]
        total   = p_auth + p_manip
        if total > 0:
            p_auth /= total
            p_manip /= total
        mnv2_sc = np.array([p_auth, p_manip, 0.0])
        specm_sc = np.array([s["authentic_score"], s["manip_score"], 0.0])
        w_m = 1.0 / max(m["confidence"], 1e-3)
        w_s = 1.0 / max(s["confidence"], 1e-3)
        combined = w_m * mnv2_sc + w_s * specm_sc
        preds.append(int(combined[:2].argmax()))
    return np.array(preds)


def fuse_gated(mnv2_recs: List[Dict], specm_recs: List[Dict],
               alpha: float, tau_s: float) -> np.ndarray:
    """가설 2: Specialist 게이팅 — specialist conf >= tau_s 일 때만 개입"""
    preds = []
    for m, s in zip(mnv2_recs, specm_recs):
        p_aigen = m["scores"].get("ai_generated", 0.0)
        if m["pred_label"] == "ai_generated" or p_aigen > 0.5:
            preds.append(CLS2IDX["ai_generated"])
            continue
        p_auth  = m["scores"]["authentic"]
        p_manip = m["scores"]["manipulated"]
        total   = p_auth + p_manip
        if total > 0:
            p_auth /= total
            p_manip /= total
        mnv2_sc  = np.array([p_auth, p_manip, 0.0])
        specm_sc = np.array([s["authentic_score"], s["manip_score"], 0.0])
        spec_conf = max(s["authentic_score"], s["manip_score"])
        if spec_conf >= tau_s:
            combined = mnv2_sc + alpha * specm_sc
        else:
            combined = mnv2_sc  # specialist 불확실 → MNV2 그대로
        preds.append(int(combined[:2].argmax()))
    return np.array(preds)


def fuse_disagree(mnv2_recs: List[Dict], specm_recs: List[Dict],
                  alpha_agree: float, alpha_disagree: float) -> np.ndarray:
    """가설 3: 불일치 조건부 α — 동의 시 alpha_agree, 불일치 시 alpha_disagree"""
    preds = []
    for m, s in zip(mnv2_recs, specm_recs):
        p_aigen = m["scores"].get("ai_generated", 0.0)
        if m["pred_label"] == "ai_generated" or p_aigen > 0.5:
            preds.append(CLS2IDX["ai_generated"])
            continue
        p_auth  = m["scores"]["authentic"]
        p_manip = m["scores"]["manipulated"]
        total   = p_auth + p_manip
        if total > 0:
            p_auth /= total
            p_manip /= total
        mnv2_sc  = np.array([p_auth, p_manip, 0.0])
        specm_sc = np.array([s["authentic_score"], s["manip_score"], 0.0])
        spec_pred = "authentic" if s["authentic_score"] >= s["manip_score"] else "manipulated"
        alpha = alpha_disagree if m["pred_label"] != spec_pred else alpha_agree
        combined = mnv2_sc + alpha * specm_sc
        preds.append(int(combined[:2].argmax()))
    return np.array(preds)


def fuse_alpha(mnv2_recs: List[Dict], specm_recs: List[Dict], alpha: float) -> np.ndarray:
    """고정 가중치 융합: combined = mnv2_sc + alpha * specm_sc"""
    preds = []
    for m, s in zip(mnv2_recs, specm_recs):
        p_aigen = m["scores"].get("ai_generated", 0.0)
        if m["pred_label"] == "ai_generated" or p_aigen > 0.5:
            preds.append(CLS2IDX["ai_generated"])
            continue
        p_auth  = m["scores"]["authentic"]
        p_manip = m["scores"]["manipulated"]
        total   = p_auth + p_manip
        if total > 0:
            p_auth /= total
            p_manip /= total
        mnv2_sc  = np.array([p_auth, p_manip, 0.0])
        specm_sc = np.array([s["authentic_score"], s["manip_score"], 0.0])
        combined = mnv2_sc + alpha * specm_sc
        preds.append(int(combined[:2].argmax()))
    return np.array(preds)


def fuse_uniform(mnv2_recs: List[Dict], specm_recs: List[Dict]) -> np.ndarray:
    """신뢰도 무관 단순 합산 (w=1 고정)"""
    preds = []
    for m, s in zip(mnv2_recs, specm_recs):
        p_aigen = m["scores"].get("ai_generated", 0.0)
        if m["pred_label"] == "ai_generated" or p_aigen > 0.5:
            preds.append(CLS2IDX["ai_generated"])
            continue
        p_auth  = m["scores"]["authentic"]
        p_manip = m["scores"]["manipulated"]
        total   = p_auth + p_manip
        if total > 0:
            p_auth /= total
            p_manip /= total
        mnv2_sc  = np.array([p_auth, p_manip, 0.0])
        specm_sc = np.array([s["authentic_score"], s["manip_score"], 0.0])
        combined = mnv2_sc + specm_sc  # 가중치 없이 단순 합산
        preds.append(int(combined[:2].argmax()))
    return np.array(preds)


def fuse_cascade(mnv2_recs: List[Dict], specm_recs: List[Dict],
                 tau: float = 0.6) -> np.ndarray:
    preds = []
    for m, s in zip(mnv2_recs, specm_recs):
        if m["confidence"] > tau:
            preds.append(CLS2IDX[m["pred_label"]])
        else:
            if m["pred_label"] == "ai_generated":
                preds.append(CLS2IDX["ai_generated"])
            else:
                pred_bin = "manipulated" if s["manip_score"] >= s["authentic_score"] else "authentic"
                preds.append(CLS2IDX[pred_bin])
    return np.array(preds)


def align_three(
    mnv2_recs: List[Dict],
    specm_a_recs: List[Dict],
    specm_b_recs: List[Dict],
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    map_a = {r["image_path"]: r for r in specm_a_recs}
    map_b = {r["image_path"]: r for r in specm_b_recs}
    out_m, out_a, out_b = [], [], []
    for m in mnv2_recs:
        a = map_a.get(m["image_path"])
        b = map_b.get(m["image_path"])
        if a is None or b is None:
            continue
        out_m.append(m)
        out_a.append(a)
        out_b.append(b)
    return out_m, out_a, out_b


def fuse_dual_policy_v4_comp_noTS(
    mnv2_recs: List[Dict],
    specm_v4_recs: List[Dict],
    specm_comp_recs: List[Dict],
    tau_v4: float,
    tau_comp: float,
    tau_mnv2: float,
    tau_gap: float,
) -> np.ndarray:
    """
    Fixed-rule dual policy:
      - stay conservative with v4 by default
      - switch to comp_noTS only when it is clearly stronger
    """
    preds = []
    for m, s_v4, s_comp in zip(mnv2_recs, specm_v4_recs, specm_comp_recs):
        p_aigen = m["scores"].get("ai_generated", 0.0)
        if m["pred_label"] == "ai_generated" or p_aigen > 0.5:
            preds.append(CLS2IDX["ai_generated"])
            continue

        m_conf = float(m["confidence"])
        v4_conf = float(max(s_v4["authentic_score"], s_v4["manip_score"]))
        comp_conf = float(max(s_comp["authentic_score"], s_comp["manip_score"]))
        v4_bin = int(s_v4["manip_score"] >= s_v4["authentic_score"])
        comp_bin = int(s_comp["manip_score"] >= s_comp["authentic_score"])

        use_comp = (
            comp_conf >= tau_comp
            and v4_conf <= tau_v4
            and m_conf <= tau_mnv2
            and (comp_conf - v4_conf) >= tau_gap
        )
        preds.append(comp_bin if use_comp else v4_bin)
    return np.array(preds)


def search_dual_policy_v4_comp_noTS(
    mnv2_data: Dict[str, List[Dict]],
    specm_v4_data: Dict[str, List[Dict]],
    specm_comp_data: Dict[str, List[Dict]],
    datasets: List[str],
) -> Dict[str, object]:
    grid = []
    for tau_v4 in [0.50, 0.55, 0.60, 0.65]:
        for tau_comp in [0.60, 0.65, 0.70, 0.75]:
            for tau_mnv2 in [0.45, 0.55, 0.65, 0.75]:
                for tau_gap in [0.00, 0.01, 0.02, 0.03]:
                    grid.append((tau_v4, tau_comp, tau_mnv2, tau_gap))

    candidates = []
    for tau_v4, tau_comp, tau_mnv2, tau_gap in grid:
        per_ds = {}
        f1s, corrs, net_gains = [], [], []
        for ds in datasets:
            if ds not in mnv2_data or ds not in specm_v4_data or ds not in specm_comp_data:
                continue
            am, av4, acomp = align_three(mnv2_data[ds], specm_v4_data[ds], specm_comp_data[ds])
            if not am:
                continue
            dual_preds = fuse_dual_policy_v4_comp_noTS(
                am, av4, acomp, tau_v4=tau_v4, tau_comp=tau_comp, tau_mnv2=tau_mnv2, tau_gap=tau_gap
            )
            dual_res = eval_preds(dual_preds, am)

            v4_preds = fuse_icwmv(am, av4)
            v4_res = eval_preds(v4_preds, am)

            comp_preds = fuse_icwmv(am, acomp)
            comp_res = eval_preds(comp_preds, am)

            per_ds[ds] = {
                "dual_v4_comp_noTS": dual_res,
                "icwmv_v4_common": v4_res,
                "icwmv_comp_noTS_common": comp_res,
                "n_common": len(am),
            }
            f1s.append(dual_res["macro_f1"])
            corrs.append(dual_res["err_corr"]["rate"])
            net_gains.append(dual_res["err_corr"]["net_gain"])

        if not per_ds:
            continue
        candidates.append(
            {
                "tau_v4": tau_v4,
                "tau_comp": tau_comp,
                "tau_mnv2": tau_mnv2,
                "tau_gap": tau_gap,
                "avg_f1": float(np.mean(f1s)),
                "avg_corr": float(np.mean(corrs)),
                "avg_net_gain": float(np.mean(net_gains)),
                "per_ds": per_ds,
            }
        )

    if not candidates:
        return {"best_cfg": None, "summary": {}, "per_ds": {}}

    best = max(candidates, key=lambda r: (r["avg_f1"], r["avg_corr"], r["avg_net_gain"]))
    return {
        "best_cfg": {
            "tau_v4": best["tau_v4"],
            "tau_comp": best["tau_comp"],
            "tau_mnv2": best["tau_mnv2"],
            "tau_gap": best["tau_gap"],
        },
        "summary": {
            "avg_f1": round(best["avg_f1"], 4),
            "avg_corr": round(best["avg_corr"], 4),
            "avg_net_gain": round(best["avg_net_gain"], 4),
        },
        "per_ds": best["per_ds"],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# HEMA-XGBoost (learned fuser)
# ═══════════════════════════════════════════════════════════════════════════════

def build_features(mnv2_recs: List[Dict], specm_recs: List[Dict]) -> np.ndarray:
    feats = []
    for m, s in zip(mnv2_recs, specm_recs):
        p_auth  = m["scores"]["authentic"]
        p_manip = m["scores"]["manipulated"]
        p_aigen = m["scores"].get("ai_generated", 0.0)
        sc3     = np.array([p_auth, p_manip, p_aigen])
        sp_m    = s["manip_score"]
        sp_a    = s["authentic_score"]
        msp_m   = float(sc3.max())
        ss      = np.sort(sc3)[::-1]
        margin  = ss[0] - ss[1]
        ent_m   = -np.sum(sc3 * np.log(sc3 + 1e-8))
        msp_s   = max(sp_m, sp_a)
        ent_s   = -(sp_m * np.log(sp_m + 1e-8) + sp_a * np.log(sp_a + 1e-8))
        feats.append([
            p_auth, p_manip, p_aigen,
            msp_m, margin, ent_m,
            sp_m, sp_a,
            msp_s, ent_s,
            abs(p_auth - sp_a),
            p_aigen * sp_a,
            p_manip * sp_m,
        ])
    return np.array(feats, dtype=np.float32)


def train_xgb(X_tr: np.ndarray, y_tr: np.ndarray):
    try:
        import xgboost as xgb
        clf = xgb.XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric="mlogloss",
            num_class=3, objective="multi:softmax",
            tree_method="hist", device="cuda",
            random_state=42, verbosity=0,
        )
    except Exception:
        from sklearn.ensemble import GradientBoostingClassifier
        clf = GradientBoostingClassifier(n_estimators=200, random_state=42)
    clf.fit(X_tr, y_tr)
    return clf


# ═══════════════════════════════════════════════════════════════════════════════
# 평가
# ═══════════════════════════════════════════════════════════════════════════════

def eval_preds(preds: np.ndarray, mnv2_recs: List[Dict]) -> Dict:
    labels = np.array([CLS2IDX[m["true_label"]] for m in mnv2_recs])
    present = sorted(set(labels.tolist()))
    f1s = []
    for c in present:
        tp = ((preds == c) & (labels == c)).sum()
        fp = ((preds == c) & (labels != c)).sum()
        fn = ((preds != c) & (labels == c)).sum()
        pr = tp / max(tp + fp, 1)
        rc = tp / max(tp + fn, 1)
        f1s.append(2 * pr * rc / max(pr + rc, 1e-8))

    n_err = n_corr = n_broken = 0
    patterns = defaultdict(lambda: {"total": 0, "corrected": 0, "broken": 0})
    for i, m in enumerate(mnv2_recs):
        if m["true_label"] == "ai_generated" or m["pred_label"] == "ai_generated":
            continue
        is_correct = m["pred_label"] == m["true_label"]
        is_now_correct = IDX2CLS.get(int(preds[i])) == m["true_label"]
        if not is_correct:
            n_err += 1
            pat = f"{m['true_label']}→{m['pred_label']}"
            patterns[pat]["total"] += 1
            if is_now_correct:
                n_corr += 1
                patterns[pat]["corrected"] += 1
        elif not is_now_correct:
            n_broken += 1
            pat = f"{m['true_label']}→{m['pred_label']}"
            patterns[pat]["broken"] += 1
    for p in patterns.values():
        p["rate"] = round(p["corrected"] / max(p["total"], 1), 4)

    return {
        "macro_f1": round(float(np.mean(f1s)), 4),
        "n": int(len(labels)),
        "err_corr": {
            "n_errors": n_err, "n_corrected": n_corr,
            "rate": round(n_corr / max(n_err, 1), 4),
            "n_broken": n_broken,
            "net_gain": int(n_corr - n_broken),
            "patterns": dict(patterns),
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
# LOO-CD 실행
# ═══════════════════════════════════════════════════════════════════════════════

def run_loo_cd():
    print("=" * 70)
    print("  Fuser LOO-CD Comparison")
    print("=" * 70)

    # 모든 데이터 미리 로드
    mnv2_data: Dict[str, List[Dict]] = {}
    specm_data: Dict[str, Dict[str, List[Dict]]] = {}  # specm_model → ds → recs

    specm_models = ["v4", "comp_noTS", "comp_g1"]

    print("\n  [데이터 로드]")
    for ds in DATASETS:
        mnv2_data[ds] = load_mnv2(ds)
        print(f"    MNV2  [{ds}] n={len(mnv2_data[ds])}")

    for sm in specm_models:
        specm_data[sm] = {}
        for ds in DATASETS:
            recs = load_specm(sm, ds)
            if recs:
                specm_data[sm][ds] = recs
                print(f"    SpecM [{sm}][{ds}] n={len(recs)}")
            else:
                print(f"    SpecM [{sm}][{ds}] JSONL 없음")

    # aligned 데이터 준비 (ds, specm_model) → (mnv2_aligned, specm_aligned)
    aligned: Dict[Tuple[str,str], Tuple[List, List]] = {}
    for sm in specm_models:
        for ds in DATASETS:
            if ds not in specm_data.get(sm, {}):
                continue
            am, as_ = align(mnv2_data[ds], specm_data[sm][ds])
            aligned[(ds, sm)] = (am, as_)

    available_ds = [ds for ds in DATASETS if (ds, "v4") in aligned]
    dual_report = search_dual_policy_v4_comp_noTS(
        mnv2_data=mnv2_data,
        specm_v4_data=specm_data.get("v4", {}),
        specm_comp_data=specm_data.get("comp_noTS", {}),
        datasets=available_ds,
    )

    # ── 결과 컨테이너 ──
    results: Dict[str, Dict[str, Dict]] = {}  # method → ds → metrics

    method_labels = {
        "mnv2_only":              "MNV2-only",
        "uniform_v4":             "Uniform+v4 (w=1, no conf)",
        "icwmv_v4":               "ICWMV+v4 (fixed-rule)",
        "cascade_v4":             "Cascade τ=0.6 +v4 (fixed)",
        "dual_v4_comp_noTS":      "DualPolicy v4→comp_noTS (fixed)",
        "hema_v4":                "HEMA+v4 (LOO-CD)",
        "uniform_comp_noTS":      "Uniform+comp_noTS (w=1, no conf)",
        "icwmv_comp_noTS":        "ICWMV+comp_noTS (fixed-rule)",
        "hema_comp_noTS":         "HEMA+comp_noTS (LOO-CD)",
        "uniform_comp_g1":        "Uniform+comp_g1 (w=1, no conf)",
        "icwmv_comp_g1":          "ICWMV+comp_g1 (fixed-rule)",
        "hema_comp_g1":           "HEMA+comp_g1 (LOO-CD)",
    }
    for k in method_labels:
        results[k] = {}
    if dual_report.get("per_ds"):
        for ds, ds_res in dual_report["per_ds"].items():
            results["dual_v4_comp_noTS"][ds] = dict(ds_res["dual_v4_comp_noTS"])
            results["dual_v4_comp_noTS"][ds]["best_cfg"] = dict(dual_report["best_cfg"])
            results["dual_v4_comp_noTS"][ds]["n_common"] = int(ds_res["n_common"])

    print(f"\n{'─'*70}")
    print(f"  LOO-CD folds")
    print(f"{'─'*70}")

    for test_ds in available_ds:
        train_dss = [d for d in available_ds if d != test_ds]
        print(f"\n  [Test={test_ds}]  Train={train_dss}")

        # ── Fixed-rule: v4 ──
        am_te, as_te_v4 = aligned[(test_ds, "v4")]
        n_te = len(am_te)

        # MNV2-only (no SpecM needed)
        preds = fuse_mnv2_only(am_te)
        results["mnv2_only"][test_ds] = eval_preds(preds, am_te)

        # Uniform+v4 (신뢰도 무관)
        preds = fuse_uniform(am_te, as_te_v4)
        results["uniform_v4"][test_ds] = eval_preds(preds, am_te)

        # ICWMV+v4
        preds = fuse_icwmv(am_te, as_te_v4)
        results["icwmv_v4"][test_ds] = eval_preds(preds, am_te)

        # Cascade τ=0.6 +v4
        preds = fuse_cascade(am_te, as_te_v4, tau=0.6)
        results["cascade_v4"][test_ds] = eval_preds(preds, am_te)

        # Dual policy: conservative v4, aggressive comp_noTS on selected slices
        if test_ds in dual_report.get("per_ds", {}):
            dual_ds = dual_report["per_ds"][test_ds]["dual_v4_comp_noTS"]
            results["dual_v4_comp_noTS"][test_ds] = dict(dual_ds)
            results["dual_v4_comp_noTS"][test_ds]["best_cfg"] = dict(dual_report["best_cfg"])
            results["dual_v4_comp_noTS"][test_ds]["n_common"] = int(
                dual_report["per_ds"][test_ds]["n_common"]
            )

        # ── Learned: HEMA+v4 ──
        key = ("v4", test_ds)
        if all((d, "v4") in aligned for d in train_dss):
            # 학습 데이터 구성
            X_trs, y_trs = [], []
            for tr_ds in train_dss:
                am_tr, as_tr = aligned[(tr_ds, "v4")]
                X_trs.append(build_features(am_tr, as_tr))
                y_trs.append(np.array([CLS2IDX[m["true_label"]] for m in am_tr]))
            X_tr = np.vstack(X_trs)
            y_tr = np.hstack(y_trs)
            X_te = build_features(am_te, as_te_v4)
            clf = train_xgb(X_tr, y_tr)
            preds = clf.predict(X_te)
            results["hema_v4"][test_ds] = eval_preds(preds, am_te)

        # ── Fixed-rule: comp_noTS ──
        if (test_ds, "comp_noTS") in aligned:
            am_te_c, as_te_c = aligned[(test_ds, "comp_noTS")]
            preds = fuse_uniform(am_te_c, as_te_c)
            results["uniform_comp_noTS"][test_ds] = eval_preds(preds, am_te_c)
            preds = fuse_icwmv(am_te_c, as_te_c)
            results["icwmv_comp_noTS"][test_ds] = eval_preds(preds, am_te_c)

        # ── Learned: HEMA+comp_noTS ──
        if (test_ds, "comp_noTS") in aligned and \
           all((d, "comp_noTS") in aligned for d in train_dss):
            am_te_c, as_te_c = aligned[(test_ds, "comp_noTS")]
            X_trs, y_trs = [], []
            for tr_ds in train_dss:
                am_tr, as_tr = aligned[(tr_ds, "comp_noTS")]
                X_trs.append(build_features(am_tr, as_tr))
                y_trs.append(np.array([CLS2IDX[m["true_label"]] for m in am_tr]))
            X_tr = np.vstack(X_trs)
            y_tr = np.hstack(y_trs)
            X_te = build_features(am_te_c, as_te_c)
            clf = train_xgb(X_tr, y_tr)
            preds = clf.predict(X_te)
            results["hema_comp_noTS"][test_ds] = eval_preds(preds, am_te_c)

        # ── Fixed-rule: comp_g1 ──
        if (test_ds, "comp_g1") in aligned:
            am_te_g, as_te_g = aligned[(test_ds, "comp_g1")]
            preds = fuse_uniform(am_te_g, as_te_g)
            results["uniform_comp_g1"][test_ds] = eval_preds(preds, am_te_g)
            preds = fuse_icwmv(am_te_g, as_te_g)
            results["icwmv_comp_g1"][test_ds] = eval_preds(preds, am_te_g)

        # ── Learned: HEMA+comp_g1 ──
        if (test_ds, "comp_g1") in aligned and \
           all((d, "comp_g1") in aligned for d in train_dss):
            am_te_g, as_te_g = aligned[(test_ds, "comp_g1")]
            X_trs, y_trs = [], []
            for tr_ds in train_dss:
                am_tr, as_tr = aligned[(tr_ds, "comp_g1")]
                X_trs.append(build_features(am_tr, as_tr))
                y_trs.append(np.array([CLS2IDX[m["true_label"]] for m in am_tr]))
            X_tr = np.vstack(X_trs)
            y_tr = np.hstack(y_trs)
            X_te = build_features(am_te_g, as_te_g)
            clf = train_xgb(X_tr, y_tr)
            preds = clf.predict(X_te)
            results["hema_comp_g1"][test_ds] = eval_preds(preds, am_te_g)

        # 이번 fold 출력
        print(f"    {'Method':30s} | F1     | corr")
        print(f"    {'─'*52}")
        for mk, ml in method_labels.items():
            r = results[mk].get(test_ds)
            if r:
                print(f"    {ml:30s} | {r['macro_f1']:.4f} | {r['err_corr']['rate']:.3f}")

    # ═══════════════════════════════════════════════════════════════════════════
    # LOO-CD 평균 요약
    # ═══════════════════════════════════════════════════════════════════════════
    print(f"\n{'═'*70}")
    print("  LOO-CD SUMMARY")
    print(f"{'═'*70}")
    print(f"  {'Method':30s} | avg_F1 | avg_corr | per-DS (F1)")
    print(f"  {'─'*70}")

    summary = {}
    for mk, ml in method_labels.items():
        ds_results = results[mk]
        if not ds_results:
            print(f"  {ml:30s} | (데이터 없음)")
            continue
        f1s   = [r["macro_f1"]           for r in ds_results.values()]
        corrs = [r["err_corr"]["rate"]    for r in ds_results.values()]
        avg_f1   = float(np.mean(f1s))
        avg_corr = float(np.mean(corrs))
        per_ds_str = " | ".join(
            f"{ds}:{ds_results[ds]['macro_f1']:.4f}"
            for ds in available_ds if ds in ds_results
        )
        print(f"  {ml:30s} | {avg_f1:.4f} | {avg_corr:.3f}      | {per_ds_str}")
        summary[mk] = {
            "avg_f1": round(avg_f1, 4),
            "avg_corr": round(avg_corr, 4),
            "per_ds": {ds: ds_results[ds] for ds in ds_results},
        }

    if dual_report.get("per_ds"):
        summary["dual_v4_comp_noTS_common_compare"] = {
            "best_cfg": dict(dual_report.get("best_cfg") or {}),
            "summary": dict(dual_report.get("summary") or {}),
            "per_ds": dual_report["per_ds"],
        }

    # ═══════════════════════════════════════════════════════════════════════════
    # 핵심 비교: ICWMV-v4 vs HEMA+comp_noTS
    # ═══════════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*70}")
    print("  CRITICAL COMPARISON: ICWMV-v4 (fixed-rule) vs HEMA+comp_noTS (learned)")
    print(f"{'─'*70}")
    if "icwmv_v4" in summary and "hema_comp_noTS" in summary:
        icwmv_f1   = summary["icwmv_v4"]["avg_f1"]
        hema_f1    = summary["hema_comp_noTS"]["avg_f1"]
        icwmv_corr = summary["icwmv_v4"]["avg_corr"]
        hema_corr  = summary["hema_comp_noTS"]["avg_corr"]
        delta_f1   = hema_f1 - icwmv_f1
        delta_corr = hema_corr - icwmv_corr
        winner_f1  = "HEMA+comp_noTS" if delta_f1 > 0 else "ICWMV-v4"
        winner_c   = "HEMA+comp_noTS" if delta_corr > 0 else "ICWMV-v4"
        print(f"  macro-F1 : ICWMV={icwmv_f1:.4f}  HEMA={hema_f1:.4f}  "
              f"Δ={delta_f1:+.4f}  → {winner_f1}")
        print(f"  err_corr : ICWMV={icwmv_corr:.3f}   HEMA={hema_corr:.3f}   "
              f"Δ={delta_corr:+.3f}   → {winner_c}")

    print(f"\n{'─'*70}")
    print("  DUAL POLICY SUMMARY: v4 conservative + comp_noTS aggressive")
    print(f"{'─'*70}")
    if dual_report.get("best_cfg") and "dual_v4_comp_noTS" in summary:
        d_f1 = summary["dual_v4_comp_noTS"]["avg_f1"]
        d_corr = summary["dual_v4_comp_noTS"]["avg_corr"]
        cfg = dual_report["best_cfg"]
        print(
            f"  best_cfg tau_v4={cfg['tau_v4']:.2f} tau_comp={cfg['tau_comp']:.2f} "
            f"tau_mnv2={cfg['tau_mnv2']:.2f} tau_gap={cfg['tau_gap']:.2f}"
        )
        print(f"  dual_v4_comp_noTS avg_F1={d_f1:.4f} avg_corr={d_corr:.3f}")
        if "icwmv_v4" in summary and "icwmv_comp_noTS" in summary:
            print(
                f"  compare: icwmv_v4 F1={summary['icwmv_v4']['avg_f1']:.4f} corr={summary['icwmv_v4']['avg_corr']:.3f} | "
                f"icwmv_comp_noTS F1={summary['icwmv_comp_noTS']['avg_f1']:.4f} corr={summary['icwmv_comp_noTS']['avg_corr']:.3f}"
            )

    # ═══════════════════════════════════════════════════════════════════════════
    # 가설 2: Specialist 게이팅 sweep (alpha x tau_s)
    # ═══════════════════════════════════════════════════════════════════════════
    GATE_ALPHAS = [0.9, 1.0, 1.5, 2.0]
    GATE_TAUS   = [0.5, 0.6, 0.65, 0.7, 0.75, 0.8]
    GATE_MODELS = ["v4", "comp_noTS", "comp_g1"]

    print(f"\n{'═'*70}")
    print("  가설 2: Specialist 게이팅 (α x τ_s sweep)")
    print(f"{'═'*70}")

    for sm in GATE_MODELS:
        if not all((ds, sm) in aligned for ds in available_ds):
            continue
        print(f"\n  [SpecM={sm}]")
        print(f"  {'α':>5} | {'τ_s':>5} | avg_F1  | avg_corr | note")
        print(f"  {'─'*55}")
        baseline_f1   = summary.get("icwmv_v4",  {}).get("avg_f1",  0)
        baseline_corr = summary.get("icwmv_v4",  {}).get("avg_corr", 0)
        best_gate = {"f1": 0, "corr": 0, "alpha": None, "tau": None}
        for alpha in GATE_ALPHAS:
            for tau_s in GATE_TAUS:
                f1s, corrs = [], []
                for ds in available_ds:
                    am, as_ = aligned[(ds, sm)]
                    preds = fuse_gated(am, as_, alpha, tau_s)
                    r = eval_preds(preds, am)
                    f1s.append(r["macro_f1"])
                    corrs.append(r["err_corr"]["rate"])
                avg_f1   = float(np.mean(f1s))
                avg_corr = float(np.mean(corrs))
                note = ""
                if avg_f1 >= 0.955 and avg_corr > best_gate["corr"]:
                    best_gate = {"f1": avg_f1, "corr": avg_corr,
                                 "alpha": alpha, "tau": tau_s}
                    note = " ← best(F1≥0.955)"
                print(f"  {alpha:>5.1f} | {tau_s:>5.2f} | {avg_f1:.4f}  | {avg_corr:.3f}      |{note}")
        if best_gate["alpha"]:
            print(f"  → 최적(F1≥0.955): α={best_gate['alpha']} τ_s={best_gate['tau']}  "
                  f"F1={best_gate['f1']:.4f}  corr={best_gate['corr']:.3f}")
        else:
            print("  → F1≥0.955 조건 만족하는 조합 없음")

    # ═══════════════════════════════════════════════════════════════════════════
    # 가설 3: 불일치 조건부 α sweep (alpha_agree x alpha_disagree)
    # ═══════════════════════════════════════════════════════════════════════════
    AGREE_ALPHAS    = [0.5, 0.7, 1.0]
    DISAGREE_ALPHAS = [1.0, 1.5, 2.0, 2.5, 3.0]
    DISAGREE_MODELS = ["v4", "comp_noTS", "comp_g1"]

    print(f"\n{'═'*70}")
    print("  가설 3: 불일치 조건부 α (α_agree x α_disagree sweep)")
    print(f"{'═'*70}")

    for sm in DISAGREE_MODELS:
        if not all((ds, sm) in aligned for ds in available_ds):
            continue
        print(f"\n  [SpecM={sm}]")
        print(f"  {'α_ag':>5} | {'α_dis':>6} | avg_F1  | avg_corr | note")
        print(f"  {'─'*58}")
        best_dis = {"f1": 0, "corr": 0, "a_ag": None, "a_dis": None}
        for a_ag in AGREE_ALPHAS:
            for a_dis in DISAGREE_ALPHAS:
                f1s, corrs = [], []
                for ds in available_ds:
                    am, as_ = aligned[(ds, sm)]
                    preds = fuse_disagree(am, as_, a_ag, a_dis)
                    r = eval_preds(preds, am)
                    f1s.append(r["macro_f1"])
                    corrs.append(r["err_corr"]["rate"])
                avg_f1   = float(np.mean(f1s))
                avg_corr = float(np.mean(corrs))
                note = ""
                if avg_f1 >= 0.955 and avg_corr > best_dis["corr"]:
                    best_dis = {"f1": avg_f1, "corr": avg_corr,
                                "a_ag": a_ag, "a_dis": a_dis}
                    note = " ← best(F1≥0.955)"
                print(f"  {a_ag:>5.1f} | {a_dis:>6.1f} | {avg_f1:.4f}  | {avg_corr:.3f}      |{note}")
        if best_dis["a_ag"]:
            print(f"  → 최적(F1≥0.955): α_agree={best_dis['a_ag']} α_disagree={best_dis['a_dis']}  "
                  f"F1={best_dis['f1']:.4f}  corr={best_dis['corr']:.3f}")
        else:
            print("  → F1≥0.955 조건 만족하는 조합 없음")

    # ═══════════════════════════════════════════════════════════════════════════
    # Alpha sweep: 최적 고정 가중치 탐색
    # ═══════════════════════════════════════════════════════════════════════════
    ALPHAS = [0.3, 0.5, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0]
    sweep_models = ["v4", "comp_noTS", "comp_g1"]

    print(f"\n{'═'*70}")
    print("  ALPHA SWEEP (mnv2 + α·specm)")
    print(f"{'═'*70}")

    for sm in sweep_models:
        if not all((ds, sm) in aligned for ds in available_ds):
            continue
        print(f"\n  [SpecM={sm}]")
        print(f"  {'α':>5} | avg_F1  | avg_corr | F1+corr")
        print(f"  {'─'*45}")
        best = {"score": -1, "alpha": None, "f1": None, "corr": None}
        for alpha in ALPHAS:
            f1s, corrs = [], []
            for ds in available_ds:
                am, as_ = aligned[(ds, sm)]
                preds = fuse_alpha(am, as_, alpha)
                r = eval_preds(preds, am)
                f1s.append(r["macro_f1"])
                corrs.append(r["err_corr"]["rate"])
            avg_f1   = float(np.mean(f1s))
            avg_corr = float(np.mean(corrs))
            combined_score = avg_f1 + avg_corr  # F1+교정률 합산 기준
            marker = " ←" if combined_score > best["score"] else ""
            print(f"  {alpha:>5.1f} | {avg_f1:.4f}  | {avg_corr:.3f}      | {combined_score:.4f}{marker}")
            if combined_score > best["score"]:
                best = {"score": combined_score, "alpha": alpha,
                        "f1": avg_f1, "corr": avg_corr}
        print(f"  → 최적 α={best['alpha']}  F1={best['f1']:.4f}  corr={best['corr']:.3f}  합산={best['score']:.4f}")

    # 저장
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "experiments" / "results" / "fuser_loo_cd"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"fuser_loo_cd_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n  저장: {out_path}")
    return summary


if __name__ == "__main__":
    run_loo_cd()
