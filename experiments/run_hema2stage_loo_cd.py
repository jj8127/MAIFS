#!/usr/bin/env python3
"""
2-Stage HEMA LOO-CD 비교
========================
구조:
  Stage 1 (ai_gen lock): MNV2 p_aigen > τ_aigen → ai_generated 잠금
  Stage 2 (binary HEMA): auth/manip 구간만 XGBoost 이진 분류

Stage 2 features (signed-gap 중심, binary-renormalized):
  f1  = binary_mnv2_auth   (p_auth / (p_auth+p_manip))
  f2  = binary_mnv2_manip  (p_manip / (p_auth+p_manip))
  f3  = specm_manip
  f4  = specm_auth
  f5  = signed_gap         (specm_manip - binary_mnv2_manip)  ← 방향 신호
  f6  = aigen_margin       (p_aigen - max(p_auth,p_manip))   ← 경계 근접도
  f7  = disagree_indicator (MNV2 vs SpecM 예측 불일치 여부)
  f8  = override_signal    (specm_manip>0.7 AND binary_mnv2_manip<0.5)
  f9  = mnv2_confidence    (binary 재정규화 후 max)
  f10 = specm_confidence   (max(specm_manip, specm_auth))

목표: ICWMV+v4 (F1=0.9652, corr=0.354) F1+corr 모두 이기기

실행:
  .venv-qwen/bin/python experiments/run_hema2stage_loo_cd.py
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

# ─── 데이터 로드 ─────────────────────────────────────────────────────────────

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

# ─── Stage 1: ai_gen lock ────────────────────────────────────────────────────

def stage1_lock(mnv2_recs: List[Dict], tau_aigen: float = 0.5):
    """
    Returns:
      locked_mask: bool array, True = ai_gen으로 잠김
    """
    locked = []
    for m in mnv2_recs:
        p_aigen = m["scores"].get("ai_generated", 0.0)
        locked.append(m["pred_label"] == "ai_generated" or p_aigen > tau_aigen)
    return np.array(locked, dtype=bool)

# ─── Stage 2: binary features ────────────────────────────────────────────────

def build_binary_features(mnv2_recs: List[Dict], specm_recs: List[Dict]) -> np.ndarray:
    """10-dim binary-space feature (auth/manip 전용)"""
    feats = []
    for m, s in zip(mnv2_recs, specm_recs):
        pa   = m["scores"]["authentic"]
        pm   = m["scores"]["manipulated"]
        pag  = m["scores"].get("ai_generated", 0.0)
        total = pa + pm
        if total > 0:
            bpa = pa / total   # binary-renormalized auth
            bpm = pm / total   # binary-renormalized manip
        else:
            bpa, bpm = 0.5, 0.5

        sp_m = s["manip_score"]
        sp_a = s["authentic_score"]

        signed_gap      = sp_m - bpm                        # SpecM이 MNV2보다 얼마나 더 조작 의심
        aigen_margin    = pag - max(pa, pm)                 # 음수 = auth/manip 지배, 0에 가까울수록 ai_gen 경계
        disagree        = float((bpm >= 0.5) != (sp_m >= 0.5))  # MNV2 vs SpecM 예측 불일치
        override_signal = float(sp_m > 0.7 and bpm < 0.5)  # SpecM 강한 manip 의심 but MNV2는 auth
        binary_conf     = max(bpa, bpm)                     # binary 공간 confidence
        specm_conf      = max(sp_m, sp_a)

        feats.append([
            bpa, bpm,          # f1-f2: binary-renormalized MNV2
            sp_m, sp_a,        # f3-f4: SpecM scores
            signed_gap,        # f5: 방향 신호 (핵심)
            aigen_margin,      # f6: ai_gen 경계 근접도
            disagree,          # f7: 불일치 indicator
            override_signal,   # f8: SpecM override 후보
            binary_conf,       # f9: MNV2 binary confidence
            specm_conf,        # f10: SpecM confidence
        ])
    return np.array(feats, dtype=np.float32)

# ─── XGBoost (binary) ────────────────────────────────────────────────────────

def train_xgb_binary(X_tr, y_tr, max_depth=3, n_est=200, reg_lambda=2.0,
                     min_child=2, subsample=0.8, colsample=0.8):
    try:
        import xgboost as xgb
        clf = xgb.XGBClassifier(
            n_estimators=n_est, max_depth=max_depth, learning_rate=0.05,
            subsample=subsample, colsample_bytree=colsample,
            reg_lambda=reg_lambda, min_child_weight=min_child,
            use_label_encoder=False, eval_metric="logloss",
            objective="binary:logistic",
            tree_method="hist", device="cuda",
            random_state=42, verbosity=0,
        )
    except Exception:
        from sklearn.ensemble import GradientBoostingClassifier
        clf = GradientBoostingClassifier(n_estimators=n_est, max_depth=max_depth, random_state=42)
    clf.fit(X_tr, y_tr)
    return clf

# ─── 평가 ────────────────────────────────────────────────────────────────────

def eval_preds(preds, mrecs):
    labels = np.array([CLS2IDX[m["true_label"]] for m in mrecs])
    present = sorted(set(labels.tolist()))
    f1s = []
    for c in present:
        tp = ((preds == c) & (labels == c)).sum()
        fp = ((preds == c) & (labels != c)).sum()
        fn = ((preds != c) & (labels == c)).sum()
        pr = tp / max(tp+fp, 1); rc = tp / max(tp+fn, 1)
        f1s.append(2*pr*rc / max(pr+rc, 1e-8))
    n_err = n_corr = 0
    patterns = defaultdict(lambda: {"total": 0, "corrected": 0})
    for i, m in enumerate(mrecs):
        if m["true_label"] == "ai_generated" or m["pred_label"] == "ai_generated":
            continue
        if m["pred_label"] != m["true_label"]:
            n_err += 1
            pat = f"{m['true_label']}→{m['pred_label']}"
            patterns[pat]["total"] += 1
            if IDX2CLS.get(int(preds[i])) == m["true_label"]:
                n_corr += 1
                patterns[pat]["corrected"] += 1
    for p in patterns.values():
        p["rate"] = round(p["corrected"] / max(p["total"], 1), 4)
    return {
        "macro_f1": round(float(np.mean(f1s)), 4),
        "n": int(len(labels)),
        "err_corr": {
            "n_errors": n_err, "n_corrected": n_corr,
            "rate": round(n_corr / max(n_err, 1), 4),
            "patterns": dict(patterns),
        },
    }

# ─── ICWMV (reference baseline) ──────────────────────────────────────────────

def fuse_icwmv(mnv2_recs, specm_recs):
    preds = []
    for m, s in zip(mnv2_recs, specm_recs):
        p_aigen = m["scores"].get("ai_generated", 0.0)
        if m["pred_label"] == "ai_generated" or p_aigen > 0.5:
            preds.append(CLS2IDX["ai_generated"]); continue
        pa  = m["scores"]["authentic"]
        pm  = m["scores"]["manipulated"]
        tot = pa + pm
        if tot > 0: pa /= tot; pm /= tot
        w_m = 1.0 / max(m["confidence"], 1e-3)
        w_s = 1.0 / max(s["confidence"], 1e-3)
        combined = w_m * np.array([pa, pm, 0.0]) + w_s * np.array([s["authentic_score"], s["manip_score"], 0.0])
        preds.append(int(combined[:2].argmax()))
    return np.array(preds)

# ─── 2-Stage HEMA 예측 ───────────────────────────────────────────────────────

def predict_2stage(mnv2_recs, specm_recs, clf_binary,
                   tau_aigen=0.5) -> np.ndarray:
    """
    Stage 1: ai_gen lock
    Stage 2: binary XGBoost for remaining samples
    """
    locked = stage1_lock(mnv2_recs, tau_aigen)
    feats  = build_binary_features(mnv2_recs, specm_recs)
    binary_preds = clf_binary.predict(feats)

    final_preds = np.zeros(len(mnv2_recs), dtype=int)
    for i, m in enumerate(mnv2_recs):
        if locked[i]:
            final_preds[i] = CLS2IDX["ai_generated"]
        else:
            # binary pred: 0=auth, 1=manip (XGBoost label 그대로)
            final_preds[i] = int(binary_preds[i])
    return final_preds


# ─── LOO-CD 비교 실행 ────────────────────────────────────────────────────────

def run():
    print("=" * 72)
    print("  2-Stage HEMA LOO-CD vs ICWMV-v4")
    print("  목표: F1>0.9652, corr>0.354")
    print("=" * 72)

    ICWMV_F1   = 0.9652
    ICWMV_CORR = 0.354

    # 데이터 로드
    mnv2_data = {ds: load_mnv2(ds) for ds in DATASETS}
    aligned   = {}
    for sm in ["v4", "comp_noTS"]:
        for ds in DATASETS:
            srecs = load_specm(sm, ds)
            if srecs:
                am, as_ = align(mnv2_data[ds], srecs)
                aligned[(ds, sm)] = (am, as_)

    # tau_aigen 후보 탐색
    tau_candidates = [0.4, 0.5, 0.6]

    # XGBoost config 후보 (Stage 2 binary)
    xgb_configs = {
        "depth2_reg2":  {"max_depth": 2, "n_est": 200, "reg_lambda": 2.0, "min_child": 2},
        "depth3_reg2":  {"max_depth": 3, "n_est": 200, "reg_lambda": 2.0, "min_child": 2},
        "depth2_reg5":  {"max_depth": 2, "n_est": 150, "reg_lambda": 5.0, "min_child": 3},
        "stump_reg3":   {"max_depth": 1, "n_est": 300, "reg_lambda": 3.0, "min_child": 3},
        "depth3_reg5":  {"max_depth": 3, "n_est": 150, "reg_lambda": 5.0, "min_child": 5},
    }

    specm_models = ["v4", "comp_noTS"]

    total = len(specm_models) * len(tau_candidates) * len(xgb_configs)
    print(f"  탐색 공간: {len(specm_models)} specm × {len(tau_candidates)} τ × {len(xgb_configs)} xgb = {total} 조합\n")

    available_ds = [ds for ds in DATASETS if (ds, "v4") in aligned]
    all_results  = []
    count = 0

    for sm in specm_models:
        avail = [ds for ds in DATASETS if (ds, sm) in aligned]
        if len(avail) < 4:
            continue

        for tau in tau_candidates:
            for xname, xparams in xgb_configs.items():
                count += 1
                ds_results = {}

                for test_ds in avail:
                    train_dss = [d for d in avail if d != test_ds]

                    # Stage 2: 학습 데이터 구성 (binary labels, ai_gen 제외)
                    X_trs, y_trs = [], []
                    for tr_ds in train_dss:
                        am_tr, as_tr = aligned[(tr_ds, sm)]
                        locked_tr    = stage1_lock(am_tr, tau)
                        feats_tr     = build_binary_features(am_tr, as_tr)
                        # ai_gen locked 샘플 제외, 이진 레이블
                        for i, m in enumerate(am_tr):
                            if locked_tr[i]:
                                continue
                            true_lbl = m["true_label"]
                            if true_lbl == "ai_generated":
                                continue  # ai_gen이지만 lock 안된 경우 skip
                            y_bin = 1 if true_lbl == "manipulated" else 0
                            X_trs.append(feats_tr[i])
                            y_trs.append(y_bin)

                    if len(X_trs) < 10:
                        continue
                    X_tr = np.array(X_trs, dtype=np.float32)
                    y_tr = np.array(y_trs, dtype=int)

                    # 학습
                    clf = train_xgb_binary(X_tr, y_tr, **xparams)

                    # 예측
                    am_te, as_te = aligned[(test_ds, sm)]
                    preds = predict_2stage(am_te, as_te, clf, tau)
                    ds_results[test_ds] = eval_preds(preds, am_te)

                if len(ds_results) < 4:
                    continue

                avg_f1   = float(np.mean([r["macro_f1"] for r in ds_results.values()]))
                avg_corr = float(np.mean([r["err_corr"]["rate"] for r in ds_results.values()]))

                beats_f1   = avg_f1   >= ICWMV_F1
                beats_corr = avg_corr >= ICWMV_CORR
                marker = ""
                if beats_f1 and beats_corr: marker = " ★★★ BOTH"
                elif beats_f1:              marker = " ★ F1"
                elif beats_corr:            marker = " ★ corr"

                if marker:
                    print(f"  [{count:3d}/{total}] {sm:10s} τ={tau} {xname:15s} | "
                          f"F1={avg_f1:.4f} corr={avg_corr:.3f}{marker}")

                all_results.append({
                    "specm": sm, "tau_aigen": tau, "xgb": xname,
                    "avg_f1": round(avg_f1, 4), "avg_corr": round(avg_corr, 4),
                    "per_ds": ds_results,
                })

    # ── 결과 정렬 및 출력 ──
    all_results.sort(key=lambda x: x["avg_f1"] + x["avg_corr"], reverse=True)

    print(f"\n{'═'*72}")
    print("  TOP 15 (F1+corr combined)")
    print(f"{'═'*72}")
    print(f"  {'Rk':3} {'SpecM':10} {'τ':5} {'XGB':15} | F1     | corr  | vs ICWMV")
    print(f"  {'─'*72}")
    for i, r in enumerate(all_results[:15]):
        df1 = r["avg_f1"] - ICWMV_F1
        dc  = r["avg_corr"] - ICWMV_CORR
        m   = "★★★" if df1>=0 and dc>=0 else ("F1↑" if df1>=0 else ("corr↑" if dc>=0 else ""))
        print(f"  {i+1:3} {r['specm']:10} {r['tau_aigen']:<5} {r['xgb']:15} | "
              f"{r['avg_f1']:.4f} | {r['avg_corr']:.3f} | ΔF1={df1:+.4f} Δcorr={dc:+.3f} {m}")

    # ── 양쪽 다 이기는 설정 ──
    both = [r for r in all_results if r["avg_f1"] >= ICWMV_F1 and r["avg_corr"] >= ICWMV_CORR]
    print(f"\n  ICWMV를 F1+corr 모두 이기는 설정: {len(both)}개")
    if both:
        print(f"\n  {'SpecM':10} {'τ':5} {'XGB':15} | F1     | corr  | per-DS F1")
        print(f"  {'─'*80}")
        for r in both:
            per = " | ".join(f"{ds}:{r['per_ds'][ds]['macro_f1']:.4f}"
                             for ds in DATASETS if ds in r['per_ds'])
            print(f"  {r['specm']:10} {r['tau_aigen']:<5} {r['xgb']:15} | "
                  f"{r['avg_f1']:.4f} | {r['avg_corr']:.3f} | {per}")

    # ── 최선 설정 상세 per-dataset ──
    if all_results:
        best = all_results[0]
        print(f"\n  [최선 설정 상세] {best['specm']} τ={best['tau_aigen']} {best['xgb']}")
        print(f"  {'DS':12} | F1     | corr  | n_err | n_corr")
        print(f"  {'─'*52}")
        for ds in DATASETS:
            if ds in best['per_ds']:
                r = best['per_ds'][ds]
                ec = r['err_corr']
                print(f"  {ds:12} | {r['macro_f1']:.4f} | {ec['rate']:.3f} | "
                      f"{ec['n_errors']:5d} | {ec['n_corrected']:6d}")
        print(f"  {'avg':12} | {best['avg_f1']:.4f} | {best['avg_corr']:.3f}")

    # ── ICWMV 참조 ──
    print(f"\n  [참조] ICWMV+v4: F1={ICWMV_F1:.4f}, corr={ICWMV_CORR:.3f}")

    # 저장
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "experiments" / "results" / "hema_2stage"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"hema2stage_loo_cd_{ts}.json"
    with open(out_path, "w") as f:
        json.dump({
            "icwmv_target": {"f1": ICWMV_F1, "corr": ICWMV_CORR},
            "both_beat": both,
            "top15": all_results[:15],
        }, f, indent=2, ensure_ascii=False)
    print(f"\n  저장: {out_path}")


if __name__ == "__main__":
    run()
