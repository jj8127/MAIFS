#!/usr/bin/env python3
"""
Phase 3: Fuser 비교 실험
========================
MNV2(3-class) + SpecM-Comp(2-class) 출력을 결합하는 다양한 Fuser 전략 비교.

비교 전략:
  1. MNV2 단독       — 3-class softmax, baseline
  2. SpecM 단독      — 2-class softmax on binary (auth/manip), baseline
  3. ICWMV           — Inverse Confidence Weighted Majority Vote
  4. HEMA-XGBoost    — Heterogeneous Expert Meta-Aggregation (13-dim features, XGBoost)
  5. FoE-MLP         — Fusion of Experts MLP (ICLR 2024 style, feature concat)
  6. Cascade(τ=0.6)  — MNV2 uncertain → defer to SpecM

평가 지표 (3-layer):
  - Prediction: macro-F1, per-class accuracy
  - Selective:  MNV2 오분류 교정률 (binary 공간)
  - Calibration: ECE, Brier score

실행:
  .venv-qwen/bin/python experiments/run_fuser_comparison.py \
      --specm comp_g1   # comp_g1 / comp_g2 / v4 선택

  .venv-qwen/bin/python experiments/run_fuser_comparison.py --specm comp_g1 --all-fusers
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

ROOT = Path(__file__).resolve().parents[1]

# ── 경로 ──────────────────────────────────────────────────────────────────────
BE_DIR    = ROOT / "experiments" / "results" / "backbone_eval"
SPECM_DIR = ROOT / "experiments" / "results" / "specm_eval"
COMP_DIR  = ROOT / "experiments" / "results" / "specm_complementary_eval"
TS_BE     = "20260319_070725"

DATASETS  = ["base", "dsC", "opensdi", "aigenproxy"]
CLASSES   = ["authentic", "manipulated", "ai_generated"]
CLS2IDX   = {"authentic": 0, "manipulated": 1, "ai_generated": 2}
IDX2CLS   = {0: "authentic", 1: "manipulated", 2: "ai_generated"}
BIN_CLS   = {"authentic": 0, "manipulated": 1}

SPECM_JSONL_MAP = {
    # specm model key → {ds_name: jsonl_path}
}


# ═══════════════════════════════════════════════════════════════════════════════
# 1. 데이터 로드
# ═══════════════════════════════════════════════════════════════════════════════

def load_mnv2(ds_name: str) -> List[Dict]:
    path = BE_DIR / f"mobilenetv2_dualstream_{ds_name}_{TS_BE}.jsonl"
    records = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            records.append(r)
    return records


def find_specm_jsonl(specm_model: str, ds_name: str) -> Optional[Path]:
    """최신 SpecM eval JSONL 찾기"""
    # comp 계열은 specm_complementary_eval에서 찾기
    if specm_model.startswith("comp"):
        # gamma/wmax 태그로 폴더 내 검색
        tag_map = {
            "comp_g1":    "gamma1.0_wmax10",
            "comp_g2":    "gamma2.0_wmax10",
            "comp_g3":    "gamma3.0_wmax10",
            "comp_wmax1": "gamma1.0_wmax1",
            "comp_wmax5": "gamma1.0_wmax5",
            "comp_wmax20":"gamma1.0_wmax20",
            "comp_noTS":  "gamma1.0_wmax10_noTS",
        }
        tag = tag_map.get(specm_model, "")
        candidates = sorted(COMP_DIR.glob(f"specm_comp_{tag}_{ds_name}_*.jsonl"))
    else:
        # v4는 specm_eval에서 찾기
        candidates = sorted(SPECM_DIR.glob(f"specm_{specm_model}_{ds_name}_*.jsonl"))

    return candidates[-1] if candidates else None


def load_specm_results(specm_model: str, ds_name: str) -> Optional[List[Dict]]:
    path = find_specm_jsonl(specm_model, ds_name)
    if not path:
        return None
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def align_records(mnv2_recs: List[Dict], specm_recs: List[Dict]) -> Tuple[List, List]:
    """image_path 기준으로 정렬 및 매칭"""
    specm_map = {r["image_path"]: r for r in specm_recs}
    aligned_mnv2, aligned_specm = [], []
    for m in mnv2_recs:
        s = specm_map.get(m["image_path"])
        if s:
            aligned_mnv2.append(m)
            aligned_specm.append(s)
    return aligned_mnv2, aligned_specm


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Fuser 구현
# ═══════════════════════════════════════════════════════════════════════════════

def fuse_mnv2_only(mnv2_recs: List[Dict]) -> np.ndarray:
    """MNV2 단독 — 3-class pred"""
    return np.array([CLS2IDX[r["pred_label"]] for r in mnv2_recs])


def fuse_specm_only(specm_recs: List[Dict], mnv2_recs: List[Dict]) -> np.ndarray:
    """
    SpecM 단독 — binary pred.
    ai_generated 샘플은 MNV2 예측 사용 (SpecM은 2-class이므로)
    """
    preds = []
    for m, s in zip(mnv2_recs, specm_recs):
        if m["true_label"] == "ai_generated" or m["pred_label"] == "ai_generated":
            # SpecM이 처리 못하는 케이스는 MNV2 사용
            preds.append(CLS2IDX[m["pred_label"]])
        else:
            preds.append(BIN_CLS[s["pred_label"]])
    return np.array(preds)


def fuse_icwmv(mnv2_recs: List[Dict], specm_recs: List[Dict]) -> np.ndarray:
    """
    Inverse Confidence Weighted Majority Vote (ICWMV).
    각 모델의 confidence 역수로 가중 투표.
    MNV2: 3-class, SpecM: 2-class (ai_gen → MNV2 fallback)

    Fix: ai_gen 샘플에 대해 SpecM은 2-class라 ai_gen 처리 불가.
    MNV2 pred/score가 ai_gen을 가리키면 MNV2 결정을 따름.
    """
    preds = []
    for m, s in zip(mnv2_recs, specm_recs):
        p_aigen = m["scores"].get("ai_generated", 0.0)

        # ai_gen fallback: MNV2 ai_gen score가 지배적이면 MNV2 사용
        if m["pred_label"] == "ai_generated" or p_aigen > 0.5:
            preds.append(CLS2IDX["ai_generated"])
            continue

        mnv2_conf  = m["confidence"]
        specm_conf = s["confidence"]

        # MNV2 soft vote (auth/manip 부분만 재정규화)
        p_auth  = m["scores"]["authentic"]
        p_manip = m["scores"]["manipulated"]
        total   = p_auth + p_manip
        if total > 0:
            p_auth  /= total
            p_manip /= total

        mnv2_scores = np.array([p_auth, p_manip, 0.0])

        # SpecM soft vote (2-class → 3-class mapping, ai_gen=0)
        specm_scores = np.array([
            s["authentic_score"],
            s["manip_score"],
            0.0,
        ])

        # ICWMV: 낮은 confidence → 높은 가중치
        w_mnv2  = 1.0 / max(mnv2_conf, 1e-3)
        w_specm = 1.0 / max(specm_conf, 1e-3)

        combined = w_mnv2 * mnv2_scores + w_specm * specm_scores
        # argmax: 0=auth, 1=manip (ai_gen은 위에서 처리됨)
        preds.append(int(combined[:2].argmax()))
    return np.array(preds)


def fuse_cascade(mnv2_recs: List[Dict], specm_recs: List[Dict],
                 tau: float = 0.6) -> np.ndarray:
    """
    Confidence-gated Cascade.
    MNV2 confidence > τ → MNV2 결정
    MNV2 confidence ≤ τ → SpecM 결정 (binary, ai_gen은 MNV2 유지)
    """
    preds = []
    for m, s in zip(mnv2_recs, specm_recs):
        if m["confidence"] > tau:
            preds.append(CLS2IDX[m["pred_label"]])
        else:
            # uncertain → SpecM
            if m["pred_label"] == "ai_generated":
                # SpecM은 ai_gen 처리 못하므로 MNV2 유지
                preds.append(CLS2IDX["ai_generated"])
            else:
                preds.append(BIN_CLS[s["pred_label"]])
    return np.array(preds)


def build_hema_features(mnv2_recs: List[Dict], specm_recs: List[Dict]) -> np.ndarray:
    """
    HEMA 13-dim feature vector.
    From MNV2 (3-class softmax):
      f1=P(auth), f2=P(manip), f3=P(aigen)
      f4=MSP, f5=margin(top2), f6=entropy
    From SpecM (2-class softmax):
      f7=P(manip_SpecM), f8=P(auth_SpecM)
      f9=MSP_SpecM, f10=entropy_SpecM
    Cross-modal:
      f11=|P(auth_MNV2) - P(auth_SpecM)|
      f12=P(aigen_MNV2) * P(auth_SpecM)  (implicit null-class signal)
      f13=P(manip_MNV2) * P(manip_SpecM)
    """
    features = []
    for m, s in zip(mnv2_recs, specm_recs):
        p_auth  = m["scores"]["authentic"]
        p_manip = m["scores"]["manipulated"]
        p_aigen = m["scores"].get("ai_generated", 0.0)
        mnv2_scores = np.array([p_auth, p_manip, p_aigen])

        sp_manip = s["manip_score"]
        sp_auth  = s["authentic_score"]

        msp_mnv2   = float(mnv2_scores.max())
        sorted_s   = np.sort(mnv2_scores)[::-1]
        margin     = sorted_s[0] - sorted_s[1]
        ent_mnv2   = -np.sum(mnv2_scores * np.log(mnv2_scores + 1e-8))

        msp_specm  = max(sp_manip, sp_auth)
        ent_specm  = -(sp_manip * np.log(sp_manip + 1e-8) +
                       sp_auth  * np.log(sp_auth  + 1e-8))

        f11 = abs(p_auth - sp_auth)
        f12 = p_aigen * sp_auth      # implicit null-class signal
        f13 = p_manip * sp_manip

        features.append([
            p_auth, p_manip, p_aigen,          # f1-f3
            msp_mnv2, margin, ent_mnv2,         # f4-f6
            sp_manip, sp_auth,                  # f7-f8
            msp_specm, ent_specm,               # f9-f10
            f11, f12, f13,                      # f11-f13
        ])
    return np.array(features, dtype=np.float32)


def fuse_hema_xgb(train_features: np.ndarray, train_labels: np.ndarray,
                   test_features: np.ndarray) -> np.ndarray:
    """HEMA XGBoost 5-fold CV 학습 후 예측"""
    try:
        import xgboost as xgb
        from sklearn.model_selection import StratifiedKFold

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        oof_preds = np.zeros(len(train_labels), dtype=int)

        for fold, (tr_idx, va_idx) in enumerate(skf.split(train_features, train_labels)):
            clf = xgb.XGBClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                use_label_encoder=False, eval_metric="mlogloss",
                num_class=3, objective="multi:softmax",
                tree_method="hist", device="cuda",
                random_state=42, verbosity=0
            )
            clf.fit(train_features[tr_idx], train_labels[tr_idx],
                    eval_set=[(train_features[va_idx], train_labels[va_idx])],
                    verbose=False)
            oof_preds[va_idx] = clf.predict(train_features[va_idx])

        # 전체 학습 후 test 예측
        clf_full = xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            use_label_encoder=False, eval_metric="mlogloss",
            num_class=3, objective="multi:softmax",
            tree_method="hist", device="cuda",
            random_state=42, verbosity=0
        )
        clf_full.fit(train_features, train_labels)
        test_preds = clf_full.predict(test_features)
        return test_preds, clf_full

    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.multiclass import OneVsRestClassifier
        clf = GradientBoostingClassifier(n_estimators=200, random_state=42)
        clf.fit(train_features, train_labels)
        return clf.predict(test_features), clf


def build_foe_features(mnv2_recs: List[Dict], specm_recs: List[Dict]) -> np.ndarray:
    """
    FoE (Fusion of Experts) feature: expert outputs concatenation.
    MNV2: 3-dim softmax + SpecM: 2-dim softmax = 5-dim
    """
    features = []
    for m, s in zip(mnv2_recs, specm_recs):
        features.append([
            m["scores"]["authentic"],
            m["scores"]["manipulated"],
            m["scores"].get("ai_generated", 0.0),
            s["authentic_score"],
            s["manip_score"],
        ])
    return np.array(features, dtype=np.float32)


def train_foe_mlp(train_features: np.ndarray, train_labels: np.ndarray,
                   test_features: np.ndarray) -> np.ndarray:
    """FoE MLP fuser (sklearn MLPClassifier)"""
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(train_features)
    X_te   = scaler.transform(test_features)

    clf = MLPClassifier(
        hidden_layer_sizes=(64, 32), activation="relu",
        max_iter=500, random_state=42, early_stopping=True
    )
    clf.fit(X_tr, train_labels)
    return clf.predict(X_te), clf


# ═══════════════════════════════════════════════════════════════════════════════
# 3. 평가 함수
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(preds: np.ndarray, true_labels: List[str],
                    mnv2_recs: List[Dict]) -> Dict:
    """macro-F1, per-class, MNV2 오분류 교정률"""
    y_true = np.array([CLS2IDX[l] for l in true_labels])
    present = sorted(set(y_true.tolist()))

    per_class = {}
    f1s = []
    for cls_idx in present:
        cls_name = IDX2CLS[cls_idx]
        tp = int(((preds == cls_idx) & (y_true == cls_idx)).sum())
        fp = int(((preds == cls_idx) & (y_true != cls_idx)).sum())
        fn = int(((preds != cls_idx) & (y_true == cls_idx)).sum())
        prec = tp / max(tp+fp, 1)
        rec  = tp / max(tp+fn, 1)
        f1   = 2*prec*rec / max(prec+rec, 1e-8)
        per_class[cls_name] = {"f1": round(f1, 4), "precision": round(prec, 4),
                                "recall": round(rec, 4), "n": int((y_true==cls_idx).sum())}
        f1s.append(f1)

    macro_f1 = float(np.mean(f1s))
    accuracy = float((preds == y_true).mean())

    # MNV2 오분류 교정률 (binary 공간)
    n_err, n_corr = 0, 0
    patterns = defaultdict(lambda: {"total": 0, "corrected": 0})
    for i, m in enumerate(mnv2_recs):
        if m["pred_label"] == "ai_generated" or m["true_label"] == "ai_generated":
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
        "macro_f1":          round(macro_f1, 4),
        "accuracy":          round(accuracy, 4),
        "per_class":         per_class,
        "mnv2_err_correction": {
            "n_errors":    n_err,
            "n_corrected": n_corr,
            "rate":        round(n_corr / max(n_err, 1), 4),
            "patterns":    dict(patterns),
        },
    }


def compute_ece(scores: np.ndarray, labels: np.ndarray,
                n_bins: int = 15) -> float:
    """ECE (max-confidence 기준)"""
    conf  = scores.max(axis=1)
    corr  = (scores.argmax(axis=1) == labels).astype(float)
    bins  = np.linspace(0, 1, n_bins+1)
    ece   = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        m = (conf >= lo) & (conf < hi)
        if i == n_bins-1: m = (conf >= lo) & (conf <= hi)
        if m.sum() == 0: continue
        ece += (m.sum() / len(conf)) * abs(corr[m].mean() - conf[m].mean())
    return float(ece)


def compute_brier(probs: np.ndarray, labels: np.ndarray, n_cls: int) -> float:
    """Brier score (multi-class one-vs-rest)"""
    one_hot = np.zeros((len(labels), n_cls))
    for i, l in enumerate(labels):
        one_hot[i, l] = 1.0
    if probs.shape[1] < n_cls:
        pad = np.zeros((len(labels), n_cls - probs.shape[1]))
        probs = np.hstack([probs, pad])
    return float(np.mean((probs - one_hot)**2))


# ═══════════════════════════════════════════════════════════════════════════════
# 4. 메인
# ═══════════════════════════════════════════════════════════════════════════════

def run_dataset(ds_name: str, specm_model: str, cascade_tau: float = 0.6) -> Dict:
    print(f"\n  Dataset: {ds_name}")

    mnv2_recs  = load_mnv2(ds_name)
    specm_recs = load_specm_results(specm_model, ds_name)
    if not specm_recs:
        print(f"    SpecM {specm_model} JSONL 없음 → 스킵")
        return {}

    mnv2_recs, specm_recs = align_records(mnv2_recs, specm_recs)
    true_labels = [r["true_label"] for r in mnv2_recs]
    n = len(mnv2_recs)
    print(f"    n={n} (aligned)")

    results = {}

    # 1. MNV2 단독
    preds = fuse_mnv2_only(mnv2_recs)
    results["mnv2_only"] = compute_metrics(preds, true_labels, mnv2_recs)

    # 2. SpecM 단독 (binary 공간, ai_gen → MNV2 fallback)
    preds = fuse_specm_only(specm_recs, mnv2_recs)
    results["specm_only"] = compute_metrics(preds, true_labels, mnv2_recs)

    # 3. ICWMV
    preds = fuse_icwmv(mnv2_recs, specm_recs)
    results["icwmv"] = compute_metrics(preds, true_labels, mnv2_recs)

    # 4. Cascade
    preds = fuse_cascade(mnv2_recs, specm_recs, tau=cascade_tau)
    results[f"cascade_tau{cascade_tau}"] = compute_metrics(preds, true_labels, mnv2_recs)

    # 5. HEMA XGBoost (train=val split: 70/30)
    features = build_hema_features(mnv2_recs, specm_recs)
    y        = np.array([CLS2IDX[l] for l in true_labels])
    rng      = np.random.RandomState(42)
    idx      = rng.permutation(len(y))
    split    = int(len(y) * 0.7)
    tr_idx, te_idx = idx[:split], idx[split:]
    test_preds, _ = fuse_hema_xgb(features[tr_idx], y[tr_idx], features[te_idx])
    # test 인덱스에 해당하는 레코드로 평가
    te_true   = [true_labels[i] for i in te_idx]
    te_mnv2   = [mnv2_recs[i]   for i in te_idx]
    results["hema_xgb"] = compute_metrics(test_preds, te_true, te_mnv2)
    results["hema_xgb"]["note"] = f"30% test split ({len(te_idx)} samples)"

    # 6. FoE MLP
    foe_feat = build_foe_features(mnv2_recs, specm_recs)
    te_preds_foe, _ = train_foe_mlp(foe_feat[tr_idx], y[tr_idx], foe_feat[te_idx])
    results["foe_mlp"] = compute_metrics(te_preds_foe, te_true, te_mnv2)
    results["foe_mlp"]["note"] = f"30% test split ({len(te_idx)} samples)"

    # 결과 출력
    print(f"    {'Method':25s} | macro_F1 | err_corr")
    print(f"    {'-'*50}")
    for name, r in results.items():
        corr = r.get("mnv2_err_correction", {}).get("rate", "N/A")
        corr_str = f"{corr:.3f}" if isinstance(corr, float) else corr
        print(f"    {name:25s} | {r['macro_f1']:.4f}   | {corr_str}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--specm",     type=str, default="comp_g1",
                        help="SpecM 모델 키 (comp_g1, comp_g2, v4, ...)")
    parser.add_argument("--datasets",  nargs="+", default=DATASETS)
    parser.add_argument("--cascade-tau", type=float, default=0.6)
    args = parser.parse_args()

    print("=" * 65)
    print(f"Phase 3: Fuser 비교 — SpecM={args.specm}, τ={args.cascade_tau}")
    print("=" * 65)

    all_results = {}
    for ds in args.datasets:
        r = run_dataset(ds, args.specm, args.cascade_tau)
        if r:
            all_results[ds] = r

    # 전체 평균 요약
    if all_results:
        print("\n" + "=" * 65)
        print("  MACRO-F1 SUMMARY (전체 데이터셋 평균)")
        print("=" * 65)
        methods = list(next(iter(all_results.values())).keys())
        for method in methods:
            f1s = [all_results[ds][method]["macro_f1"]
                   for ds in all_results if method in all_results[ds]]
            corrs = [all_results[ds][method]["mnv2_err_correction"]["rate"]
                     for ds in all_results if method in all_results[ds]]
            avg_f1   = np.mean(f1s)  if f1s   else float("nan")
            avg_corr = np.mean(corrs) if corrs else float("nan")
            per_ds = " | ".join(f"{ds}:{all_results[ds].get(method, {}).get('macro_f1', 0):.4f}"
                                 for ds in all_results)
            print(f"  {method:25s}: avg_F1={avg_f1:.4f}, avg_corr={avg_corr:.3f} | {per_ds}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "experiments" / "results" / "fuser_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"fuser_{args.specm}_{ts}.json"
    with open(out_path, "w") as f:
        json.dump({"specm": args.specm, "cascade_tau": args.cascade_tau,
                   "results": all_results}, f, indent=2, ensure_ascii=False)
    print(f"\n저장: {out_path}")


if __name__ == "__main__":
    main()
