#!/usr/bin/env python3
"""
HEMA-XGBoost Cross-Dataset Evaluation
======================================
공정한 HEMA-XGBoost 평가: Leave-One-Out cross-dataset 방식.

기존 문제:
  - 동일 데이터셋 내 70/30 split → in-domain 과대평가
  - 실제 환경에서 HEMA가 보지 않은 도메인에 얼마나 일반화되는지 측정 불가

개선:
  LOO-CD (Leave-One-Out Cross-Dataset):
    base로 테스트 → dsC+opensdi+aigenproxy로 학습
    dsC로 테스트 → base+opensdi+aigenproxy로 학습
    opensdi로 테스트 → base+dsC+aigenproxy로 학습
    aigenproxy로 테스트 → base+dsC+opensdi로 학습

추가 비교:
  All-In (oracle): 4개 데이터셋 전체 70/30 split (기존 방식의 개선판)
  Feature Ablation: 13-dim 중 어느 subset이 중요한지

실행:
  .venv-qwen/bin/python experiments/run_hema_xgb_crossdataset.py
  .venv-qwen/bin/python experiments/run_hema_xgb_crossdataset.py --specm comp_g1 --ablation
"""

from __future__ import annotations

import argparse
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

FEATURE_NAMES = [
    "mnv2_p_auth", "mnv2_p_manip", "mnv2_p_aigen",   # f1-f3
    "mnv2_msp", "mnv2_margin", "mnv2_entropy",         # f4-f6
    "specm_p_manip", "specm_p_auth",                   # f7-f8
    "specm_msp", "specm_entropy",                      # f9-f10
    "cross_auth_diff", "cross_null_signal",            # f11-f12
    "cross_manip_agree",                               # f13
]


# ═══════════════════════════════════════════════════════════════════════════════
# 데이터 로드 및 feature 추출
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


def build_features(mnv2_recs: List[Dict],
                   specm_recs: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    """13-dim HEMA feature + label 추출 (aligned)"""
    specm_map = {r["image_path"]: r for r in specm_recs}
    feats, labels = [], []
    for m in mnv2_recs:
        s = specm_map.get(m["image_path"])
        if not s:
            continue
        p_auth  = m["scores"]["authentic"]
        p_manip = m["scores"]["manipulated"]
        p_aigen = m["scores"].get("ai_generated", 0.0)
        sc3     = np.array([p_auth, p_manip, p_aigen])
        sp_m    = s["manip_score"]
        sp_a    = s["authentic_score"]

        msp_m   = float(sc3.max())
        sorted_s = np.sort(sc3)[::-1]
        margin  = sorted_s[0] - sorted_s[1]
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
        labels.append(CLS2IDX[m["true_label"]])
    return np.array(feats, dtype=np.float32), np.array(labels, dtype=int)


# ═══════════════════════════════════════════════════════════════════════════════
# HEMA XGBoost 학습 / 평가
# ═══════════════════════════════════════════════════════════════════════════════

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


def eval_preds(preds: np.ndarray, labels: np.ndarray,
               mnv2_recs: List[Dict]) -> Dict:
    f1s = []
    for c in sorted(set(labels)):
        tp = ((preds == c) & (labels == c)).sum()
        fp = ((preds == c) & (labels != c)).sum()
        fn = ((preds != c) & (labels == c)).sum()
        pr = tp / max(tp + fp, 1)
        rc = tp / max(tp + fn, 1)
        f1s.append(2 * pr * rc / max(pr + rc, 1e-8))

    n_err = n_corr = 0
    patterns = defaultdict(lambda: {"total": 0, "corrected": 0})
    for i, m in enumerate(mnv2_recs):
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
        "accuracy": round(float((preds == labels).mean()), 4),
        "n": int(len(labels)),
        "mnv2_err_corr": {
            "n_errors": n_err, "n_corrected": n_corr,
            "rate": round(n_corr / max(n_err, 1), 4),
            "patterns": dict(patterns),
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 평가 모드
# ═══════════════════════════════════════════════════════════════════════════════

def run_loo_crossdataset(specm_model: str) -> Dict:
    """Leave-One-Out Cross-Dataset 평가"""
    print(f"\n{'─'*60}")
    print(f"  LOO-CD: SpecM={specm_model}")
    print(f"{'─'*60}")

    # 전체 데이터 로드
    all_feats, all_labels, all_mnv2, all_ds = {}, {}, {}, {}
    for ds in DATASETS:
        mnv2 = load_mnv2(ds)
        specm = load_specm(specm_model, ds)
        if specm is None:
            print(f"  [{ds}] SpecM JSONL 없음 → 스킵")
            continue
        feats, labels = build_features(mnv2, specm)
        # mnv2_recs도 feats와 align
        specm_map = {r["image_path"]: r for r in specm}
        aligned_mnv2 = [m for m in mnv2 if m["image_path"] in specm_map]
        all_feats[ds]  = feats
        all_labels[ds] = labels
        all_mnv2[ds]   = aligned_mnv2
        print(f"  [{ds}] n={len(labels)}")

    available = list(all_feats.keys())
    results = {}

    print(f"\n  {'Test DS':12s} | Train DS (n)                    | macro_F1 | err_corr")
    print(f"  {'─'*72}")

    for test_ds in available:
        train_dss = [d for d in available if d != test_ds]
        X_tr = np.vstack([all_feats[d] for d in train_dss])
        y_tr = np.hstack([all_labels[d] for d in train_dss])
        X_te = all_feats[test_ds]
        y_te = all_labels[test_ds]
        mnv2_te = all_mnv2[test_ds]

        n_train_str = "+".join(f"{d}({len(all_labels[d])})" for d in train_dss)
        clf = train_xgb(X_tr, y_tr)
        preds = clf.predict(X_te)
        r = eval_preds(preds, y_te, mnv2_te)
        results[test_ds] = r

        pat_str = " | ".join(
            f"{k}:{v['corrected']}/{v['total']}({v['rate']:.0%})"
            for k, v in r["mnv2_err_corr"]["patterns"].items()
        )
        print(f"  {test_ds:12s} | {n_train_str[:33]:33s} | "
              f"{r['macro_f1']:.4f}   | {r['mnv2_err_corr']['rate']:.3f} | {pat_str}")

    avg_f1   = np.mean([r["macro_f1"] for r in results.values()])
    avg_corr = np.mean([r["mnv2_err_corr"]["rate"] for r in results.values()])
    print(f"\n  {'LOO-CD avg':12s} | {'─'*33} | {avg_f1:.4f}   | {avg_corr:.3f}")
    return results


def run_allin(specm_model: str) -> Dict:
    """All-In: 4개 데이터셋 통합 + stratified 5-fold CV"""
    print(f"\n{'─'*60}")
    print(f"  All-In 5-fold CV: SpecM={specm_model}")
    print(f"{'─'*60}")

    all_feats, all_labels, all_mnv2 = [], [], []
    for ds in DATASETS:
        mnv2 = load_mnv2(ds)
        specm = load_specm(specm_model, ds)
        if specm is None:
            continue
        feats, labels = build_features(mnv2, specm)
        specm_map = {r["image_path"]: r for r in specm}
        aligned_mnv2 = [m for m in mnv2 if m["image_path"] in specm_map]
        all_feats.append(feats)
        all_labels.append(labels)
        all_mnv2.extend(aligned_mnv2)

    X = np.vstack(all_feats)
    y = np.hstack(all_labels)

    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(y), dtype=int)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        clf = train_xgb(X[tr_idx], y[tr_idx])
        oof_preds[va_idx] = clf.predict(X[va_idx])

    mnv2_arr = all_mnv2
    r = eval_preds(oof_preds, y, mnv2_arr)
    print(f"  5-fold OOF: macro_F1={r['macro_f1']:.4f}, "
          f"err_corr={r['mnv2_err_corr']['rate']:.3f} "
          f"({r['mnv2_err_corr']['n_corrected']}/{r['mnv2_err_corr']['n_errors']})")
    return r


def run_feature_ablation(specm_model: str) -> Dict:
    """13-dim feature subset ablation (LOO-CD 기준)"""
    print(f"\n{'─'*60}")
    print(f"  Feature Ablation (LOO-CD): SpecM={specm_model}")
    print(f"{'─'*60}")

    all_feats, all_labels, all_mnv2 = {}, {}, {}
    for ds in DATASETS:
        mnv2 = load_mnv2(ds)
        specm = load_specm(specm_model, ds)
        if specm is None:
            continue
        feats, labels = build_features(mnv2, specm)
        specm_map = {r["image_path"]: r for r in specm}
        aligned_mnv2 = [m for m in mnv2 if m["image_path"] in specm_map]
        all_feats[ds]  = feats
        all_labels[ds] = labels
        all_mnv2[ds]   = aligned_mnv2

    subsets = {
        "Full (13-dim)":   list(range(13)),
        "MNV2-only (6)":   list(range(6)),
        "SpecM-only (4)":  list(range(6, 10)),
        "Cross (3)":       list(range(10, 13)),
        "MNV2+SpecM (10)": list(range(10)),
        "No-Cross (10)":   list(range(10)),
        "Scores-only (5)": [0, 1, 2, 6, 7],
        "w/o aigen (11)":  [i for i in range(13) if i != 2],
    }

    available = list(all_feats.keys())
    abl_results = {}

    print(f"  {'Subset':22s} | avg_F1 | avg_corr")
    print(f"  {'─'*45}")
    for name, feat_idx in subsets.items():
        f1s, corrs = [], []
        for test_ds in available:
            train_dss = [d for d in available if d != test_ds]
            X_tr = np.vstack([all_feats[d][:, feat_idx] for d in train_dss])
            y_tr = np.hstack([all_labels[d] for d in train_dss])
            X_te = all_feats[test_ds][:, feat_idx]
            y_te = all_labels[test_ds]
            mnv2_te = all_mnv2[test_ds]
            clf = train_xgb(X_tr, y_tr)
            preds = clf.predict(X_te)
            r = eval_preds(preds, y_te, mnv2_te)
            f1s.append(r["macro_f1"])
            corrs.append(r["mnv2_err_corr"]["rate"])
        avg_f1   = np.mean(f1s)
        avg_corr = np.mean(corrs)
        abl_results[name] = {"macro_f1": round(float(avg_f1), 4),
                              "err_corr": round(float(avg_corr), 4),
                              "per_ds": dict(zip(available, [round(f,4) for f in f1s]))}
        print(f"  {name:22s} | {avg_f1:.4f} | {avg_corr:.4f}")

    return abl_results


def run_model_comparison(datasets: List[str]) -> None:
    """comp_g1 vs v4 vs comp_noTS LOO-CD 비교"""
    print(f"\n{'═'*60}")
    print("  Model Comparison (LOO-CD avg)")
    print(f"{'═'*60}")
    models = ["comp_g1", "v4", "comp_noTS"]
    model_names = {
        "comp_g1":   "SpecM-Comp γ=1 (TS)",
        "v4":        "SpecM-v4 (baseline)",
        "comp_noTS": "SpecM-Comp γ=1 (no-TS)",
    }
    print(f"  {'Model':30s} | avg_F1 | avg_corr")
    print(f"  {'─'*50}")
    all_res = {}
    for m in models:
        available_ds = [ds for ds in DATASETS
                        if find_specm_jsonl(m, ds) is not None]
        if len(available_ds) < 2:
            print(f"  {model_names[m]:30s} | (데이터 부족 스킵)")
            continue
        f1s, corrs = [], []
        for test_ds in available_ds:
            train_dss = [d for d in available_ds if d != test_ds]
            mnv2_te   = load_mnv2(test_ds)
            specm_te  = load_specm(m, test_ds)
            X_te, y_te = build_features(mnv2_te, specm_te)
            specm_map = {r["image_path"]: r for r in specm_te}
            aligned_mnv2 = [rec for rec in mnv2_te if rec["image_path"] in specm_map]

            X_trs, y_trs = [], []
            for tr_ds in train_dss:
                s = load_specm(m, tr_ds)
                if s is None:
                    continue
                f, l = build_features(load_mnv2(tr_ds), s)
                X_trs.append(f); y_trs.append(l)
            if not X_trs:
                continue
            clf = train_xgb(np.vstack(X_trs), np.hstack(y_trs))
            preds = clf.predict(X_te)
            r = eval_preds(preds, y_te, aligned_mnv2)
            f1s.append(r["macro_f1"])
            corrs.append(r["mnv2_err_corr"]["rate"])
        avg_f1 = np.mean(f1s); avg_corr = np.mean(corrs)
        all_res[m] = {"macro_f1": round(float(avg_f1), 4),
                      "err_corr": round(float(avg_corr), 4)}
        print(f"  {model_names[m]:30s} | {avg_f1:.4f} | {avg_corr:.4f}")
    return all_res


# ═══════════════════════════════════════════════════════════════════════════════
# 메인
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--specm",    type=str, default="comp_g1")
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument("--compare",  action="store_true",
                        help="comp_g1 vs v4 vs comp_noTS 비교")
    args = parser.parse_args()

    print("=" * 65)
    print("HEMA-XGBoost Cross-Dataset Evaluation")
    print("=" * 65)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "experiments" / "results" / "hema_crossdataset"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    # LOO-CD
    loo_results = run_loo_crossdataset(args.specm)
    all_results["loo_cd"] = loo_results

    # All-In CV
    allin_result = run_allin(args.specm)
    all_results["allin_cv"] = allin_result

    # Feature Ablation
    if args.ablation:
        abl = run_feature_ablation(args.specm)
        all_results["feature_ablation"] = abl

    # Model Comparison
    if args.compare:
        cmp = run_model_comparison(DATASETS)
        all_results["model_comparison"] = cmp

    out_path = out_dir / f"hema_cd_{args.specm}_{ts}.json"
    with open(out_path, "w") as f:
        json.dump({"specm": args.specm, **all_results}, f,
                  indent=2, ensure_ascii=False)
    print(f"\n저장: {out_path}")


if __name__ == "__main__":
    main()
