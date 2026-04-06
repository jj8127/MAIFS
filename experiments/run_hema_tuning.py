#!/usr/bin/env python3
"""
HEMA LOO-CD 튜닝: ICWMV-v4를 F1+교정률 모두 이기는 설정 탐색
==============================================================
현재: ICWMV-v4  avg_F1=0.9652, avg_corr=0.354
목표: HEMA      avg_F1>0.9652, avg_corr>0.354 (both)

개선 축:
  1. Feature subset (13→5, 3, custom)  — overfitting 감소
  2. XGBoost 정규화 (max_depth↓, reg_lambda↑, n_estimators↓)
  3. SpecM 모델 선택 (v4 vs comp_noTS) — v4는 더 많은 aligned 샘플
  4. 조합 탐색

실행:
  .venv-qwen/bin/python experiments/run_hema_tuning.py
"""

from __future__ import annotations

import json
import warnings
from collections import defaultdict
from datetime import datetime
from itertools import product
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

# ── 데이터 로드 ──────────────────────────────────────────────────────────────

def load_mnv2(ds):
    with open(BE_DIR / f"mobilenetv2_dualstream_{ds}_{TS_BE}.jsonl") as f:
        return [json.loads(l) for l in f]

def find_specm(sm, ds):
    tag_map = {"comp_g1": "gamma1.0_wmax10", "comp_noTS": "gamma1.0_wmax10_noTS", "v4": None}
    if sm == "v4":
        c = sorted(SPECM_DIR.glob(f"specm_v4_{ds}_*.jsonl"))
    else:
        c = sorted(COMP_DIR.glob(f"specm_comp_{tag_map[sm]}_{ds}_*.jsonl"))
    return c[-1] if c else None

def load_specm(sm, ds):
    p = find_specm(sm, ds)
    if not p: return None
    with open(p) as f: return [json.loads(l) for l in f]

def align(mrecs, srecs):
    sm = {r["image_path"]: r for r in srecs}
    am, as_ = [], []
    for m in mrecs:
        s = sm.get(m["image_path"])
        if s: am.append(m); as_.append(s)
    return am, as_

# ── Feature 추출 ──────────────────────────────────────────────────────────

FEATURE_SUBSETS = {
    "full_13":       list(range(13)),
    "scores_5":      [0, 1, 2, 6, 7],           # raw probability scores only
    "cross_3":       [10, 11, 12],               # cross-modal only
    "mnv2_6":        list(range(6)),              # MNV2 features only
    "scores_cross_8":[0, 1, 2, 6, 7, 10, 11, 12], # scores + cross
    "no_entropy_10": [0, 1, 2, 3, 4, 6, 7, 8, 10, 11, 12],  # drop entropies (f5,f9)
    "top_signal_7":  [0, 1, 2, 4, 6, 7, 12],    # auth,manip,aigen,margin,sp_m,sp_a,manip_agree
}

def build_features(mrecs, srecs):
    feats = []
    for m, s in zip(mrecs, srecs):
        pa  = m["scores"]["authentic"]
        pm  = m["scores"]["manipulated"]
        pag = m["scores"].get("ai_generated", 0.0)
        sc3 = np.array([pa, pm, pag])
        sp_m, sp_a = s["manip_score"], s["authentic_score"]
        msp   = float(sc3.max())
        ss    = np.sort(sc3)[::-1]
        margin= ss[0] - ss[1]
        ent_m = -np.sum(sc3 * np.log(sc3 + 1e-8))
        msp_s = max(sp_m, sp_a)
        ent_s = -(sp_m * np.log(sp_m + 1e-8) + sp_a * np.log(sp_a + 1e-8))
        feats.append([
            pa, pm, pag,                   # 0-2
            msp, margin, ent_m,            # 3-5
            sp_m, sp_a,                    # 6-7
            msp_s, ent_s,                  # 8-9
            abs(pa - sp_a),                # 10
            pag * sp_a,                    # 11
            pm * sp_m,                     # 12
        ])
    return np.array(feats, dtype=np.float32)

# ── XGBoost ──────────────────────────────────────────────────────────────

def train_xgb(X_tr, y_tr, max_depth=4, n_est=300, reg_lambda=1.0,
              min_child=1, subsample=0.8, colsample=0.8):
    try:
        import xgboost as xgb
        clf = xgb.XGBClassifier(
            n_estimators=n_est, max_depth=max_depth, learning_rate=0.05,
            subsample=subsample, colsample_bytree=colsample,
            reg_lambda=reg_lambda, min_child_weight=min_child,
            use_label_encoder=False, eval_metric="mlogloss",
            num_class=3, objective="multi:softmax",
            tree_method="hist", device="cuda",
            random_state=42, verbosity=0,
        )
    except Exception:
        from sklearn.ensemble import GradientBoostingClassifier
        clf = GradientBoostingClassifier(n_estimators=n_est, max_depth=max_depth, random_state=42)
    clf.fit(X_tr, y_tr)
    return clf

# ── 평가 ──────────────────────────────────────────────────────────────

def eval_preds(preds, mrecs):
    labels = np.array([CLS2IDX[m["true_label"]] for m in mrecs])
    present = sorted(set(labels.tolist()))
    f1s = []
    for c in present:
        tp = ((preds == c) & (labels == c)).sum()
        fp = ((preds == c) & (labels != c)).sum()
        fn = ((preds != c) & (labels == c)).sum()
        pr = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1)
        f1s.append(2 * pr * rc / max(pr + rc, 1e-8))
    n_err = n_corr = 0
    for i, m in enumerate(mrecs):
        if m["true_label"] == "ai_generated" or m["pred_label"] == "ai_generated":
            continue
        if m["pred_label"] != m["true_label"]:
            n_err += 1
            if IDX2CLS.get(int(preds[i])) == m["true_label"]:
                n_corr += 1
    return {
        "macro_f1": round(float(np.mean(f1s)), 4),
        "err_corr": round(n_corr / max(n_err, 1), 4),
        "n_err": n_err, "n_corr": n_corr,
    }

# ── LOO-CD 평가 (하나의 설정) ──────────────────────────────────────

def loo_cd_eval(aligned_data, specm_model, feat_idx, xgb_params):
    """하나의 (specm, feature subset, xgb params) 조합에 대해 LOO-CD 평가"""
    available = [ds for ds in DATASETS if (ds, specm_model) in aligned_data]
    if len(available) < 4:
        return None

    ds_results = {}
    for test_ds in available:
        train_dss = [d for d in available if d != test_ds]
        # Train
        X_trs, y_trs = [], []
        for tr_ds in train_dss:
            am, as_ = aligned_data[(tr_ds, specm_model)]
            f = build_features(am, as_)[:, feat_idx]
            y = np.array([CLS2IDX[m["true_label"]] for m in am])
            X_trs.append(f); y_trs.append(y)
        X_tr = np.vstack(X_trs); y_tr = np.hstack(y_trs)
        # Test
        am_te, as_te = aligned_data[(test_ds, specm_model)]
        X_te = build_features(am_te, as_te)[:, feat_idx]
        clf = train_xgb(X_tr, y_tr, **xgb_params)
        preds = clf.predict(X_te)
        ds_results[test_ds] = eval_preds(preds, am_te)

    avg_f1   = np.mean([r["macro_f1"] for r in ds_results.values()])
    avg_corr = np.mean([r["err_corr"] for r in ds_results.values()])
    return {
        "avg_f1":   round(float(avg_f1), 4),
        "avg_corr": round(float(avg_corr), 4),
        "per_ds":   ds_results,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 메인
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 72)
    print("  HEMA LOO-CD Tuning — 목표: ICWMV-v4 (F1=0.9652, corr=0.354) 이기기")
    print("=" * 72)

    # 데이터 로드
    mnv2_data = {}
    aligned = {}
    specm_models = ["v4", "comp_noTS"]

    for ds in DATASETS:
        mnv2_data[ds] = load_mnv2(ds)

    for sm in specm_models:
        for ds in DATASETS:
            srecs = load_specm(sm, ds)
            if srecs:
                am, as_ = align(mnv2_data[ds], srecs)
                aligned[(ds, sm)] = (am, as_)

    print(f"  데이터 로드 완료")

    # ── 탐색 공간 정의 ──
    xgb_configs = {
        "default":     {"max_depth": 4, "n_est": 300, "reg_lambda": 1.0, "min_child": 1, "subsample": 0.8, "colsample": 0.8},
        "shallow":     {"max_depth": 2, "n_est": 200, "reg_lambda": 1.0, "min_child": 1, "subsample": 0.8, "colsample": 0.8},
        "shallow3":    {"max_depth": 3, "n_est": 200, "reg_lambda": 1.0, "min_child": 1, "subsample": 0.8, "colsample": 0.8},
        "reg_heavy":   {"max_depth": 3, "n_est": 200, "reg_lambda": 5.0, "min_child": 3, "subsample": 0.7, "colsample": 0.7},
        "reg_medium":  {"max_depth": 3, "n_est": 200, "reg_lambda": 3.0, "min_child": 2, "subsample": 0.8, "colsample": 0.8},
        "very_simple": {"max_depth": 2, "n_est": 100, "reg_lambda": 5.0, "min_child": 5, "subsample": 0.7, "colsample": 0.6},
        "stump":       {"max_depth": 1, "n_est": 300, "reg_lambda": 3.0, "min_child": 3, "subsample": 0.8, "colsample": 0.8},
    }

    # ── 탐색 실행 ──
    ICWMV_F1   = 0.9652
    ICWMV_CORR = 0.354

    all_results = []
    total = len(specm_models) * len(FEATURE_SUBSETS) * len(xgb_configs)
    print(f"  탐색 공간: {len(specm_models)} specm × {len(FEATURE_SUBSETS)} feats × {len(xgb_configs)} xgb = {total} 조합\n")

    count = 0
    for sm in specm_models:
        for feat_name, feat_idx in FEATURE_SUBSETS.items():
            for xgb_name, xgb_params in xgb_configs.items():
                count += 1
                r = loo_cd_eval(aligned, sm, feat_idx, xgb_params)
                if r is None:
                    continue

                beats_f1   = r["avg_f1"] >= ICWMV_F1
                beats_corr = r["avg_corr"] >= ICWMV_CORR
                marker = ""
                if beats_f1 and beats_corr:
                    marker = " ★★★ BOTH BEAT"
                elif beats_f1:
                    marker = " ★ F1 beat"
                elif beats_corr:
                    marker = " ★ corr beat"

                entry = {
                    "specm": sm, "features": feat_name, "xgb": xgb_name,
                    "n_feat": len(feat_idx),
                    **r,
                }
                all_results.append(entry)

                if marker:
                    print(f"  [{count:3d}/{total}] {sm:10s} {feat_name:16s} {xgb_name:12s} | "
                          f"F1={r['avg_f1']:.4f} corr={r['avg_corr']:.3f}{marker}")

    # ── 결과 정렬 (F1 + corr 합산 점수) ──
    print(f"\n{'═'*72}")
    print("  TOP 20 (sorted by F1+corr combined score)")
    print(f"{'═'*72}")
    print(f"  {'Rank':4s} {'SpecM':10s} {'Features':16s} {'XGB':12s} | F1     | corr  | vs ICWMV")
    print(f"  {'─'*78}")

    all_results.sort(key=lambda x: x["avg_f1"] + x["avg_corr"], reverse=True)
    for i, r in enumerate(all_results[:20]):
        df1 = r["avg_f1"] - ICWMV_F1
        dc  = r["avg_corr"] - ICWMV_CORR
        marker = ""
        if df1 >= 0 and dc >= 0: marker = "★★★"
        elif df1 >= 0: marker = "F1↑"
        elif dc >= 0: marker = "corr↑"
        print(f"  {i+1:4d} {r['specm']:10s} {r['features']:16s} {r['xgb']:12s} | "
              f"{r['avg_f1']:.4f} | {r['avg_corr']:.3f} | "
              f"ΔF1={df1:+.4f} Δcorr={dc:+.3f} {marker}")

    # ── ICWMV를 둘 다 이기는 설정들 ──
    both_beat = [r for r in all_results if r["avg_f1"] >= ICWMV_F1 and r["avg_corr"] >= ICWMV_CORR]
    print(f"\n  ICWMV를 F1+corr 모두 이기는 설정: {len(both_beat)}개")
    if both_beat:
        print(f"\n  {'SpecM':10s} {'Features':16s} {'XGB':12s} | F1     | corr  | per-DS F1")
        print(f"  {'─'*85}")
        for r in both_beat:
            per_ds = " ".join(f"{ds}:{r['per_ds'][ds]['macro_f1']:.4f}" for ds in DATASETS if ds in r['per_ds'])
            print(f"  {r['specm']:10s} {r['features']:16s} {r['xgb']:12s} | "
                  f"{r['avg_f1']:.4f} | {r['avg_corr']:.3f} | {per_ds}")

    # ── F1만 이기는 설정 중 corr가 가장 높은 것 ──
    f1_beat = [r for r in all_results if r["avg_f1"] >= ICWMV_F1]
    if f1_beat and not both_beat:
        f1_beat.sort(key=lambda x: x["avg_corr"], reverse=True)
        print(f"\n  F1≥ICWMV인 설정 중 corr 최고:")
        r = f1_beat[0]
        per_ds = " ".join(f"{ds}:{r['per_ds'][ds]['macro_f1']:.4f}" for ds in DATASETS if ds in r['per_ds'])
        print(f"  {r['specm']:10s} {r['features']:16s} {r['xgb']:12s} | "
              f"F1={r['avg_f1']:.4f} corr={r['avg_corr']:.3f} | {per_ds}")

    # 저장
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "experiments" / "results" / "hema_tuning"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"hema_tuning_{ts}.json"
    with open(out_path, "w") as f:
        json.dump({"icwmv_target": {"f1": ICWMV_F1, "corr": ICWMV_CORR},
                   "both_beat": both_beat,
                   "top20": all_results[:20],
                   "all": all_results}, f, indent=2, ensure_ascii=False)
    print(f"\n  저장: {out_path}")


if __name__ == "__main__":
    main()
