#!/usr/bin/env python3
"""
Phase 4: 3-Layer Evaluation
============================
SHIELD 논문 C6 섹션을 위한 종합 평가.

Layer 1 — Prediction:   macro-F1, per-class recall/precision
Layer 2 — Selective:    MNV2 오분류 교정률, AURC (Area Under Risk-Coverage)
Layer 3 — Calibration:  ECE (15-bin), Brier score

비교 대상:
  - MNV2 단독 (backbone_eval)
  - SpecM-v4 (기존 전문가)
  - SpecM-Comp γ=1 (신규 complementary)
  - HEMA-Comp (MNV2 + SpecM-Comp, XGBoost)
  - Cascade-Comp (τ=0.6)

실행:
  .venv-qwen/bin/python experiments/run_phase4_3layer_eval.py
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]

BE_DIR    = ROOT / "experiments" / "results" / "backbone_eval"
SPECM_DIR = ROOT / "experiments" / "results" / "specm_eval"
COMP_DIR  = ROOT / "experiments" / "results" / "specm_complementary_eval"
FUSER_DIR = ROOT / "experiments" / "results" / "fuser_comparison"
TS_BE     = "20260319_070725"

DATASETS = ["base", "dsC", "opensdi", "aigenproxy"]
CLASSES  = ["authentic", "manipulated", "ai_generated"]
CLS2IDX  = {"authentic": 0, "manipulated": 1, "ai_generated": 2}
IDX2CLS  = {0: "authentic", 1: "manipulated", 2: "ai_generated"}


# ═══════════════════════════════════════════════════════════════════════════════
# 데이터 로드
# ═══════════════════════════════════════════════════════════════════════════════

def load_mnv2(ds_name: str) -> List[Dict]:
    path = BE_DIR / f"mobilenetv2_dualstream_{ds_name}_{TS_BE}.jsonl"
    with open(path) as f:
        return [json.loads(l) for l in f]


def load_latest_jsonl(directory: Path, pattern: str) -> Optional[List[Dict]]:
    candidates = sorted(directory.glob(pattern))
    if not candidates:
        return None
    with open(candidates[-1]) as f:
        return [json.loads(l) for l in f]


def load_specm_v4(ds_name: str) -> Optional[List[Dict]]:
    return load_latest_jsonl(SPECM_DIR, f"specm_v4_{ds_name}_*.jsonl")


def load_specm_comp(specm_tag: str, ds_name: str) -> Optional[List[Dict]]:
    return load_latest_jsonl(COMP_DIR, f"specm_comp_{specm_tag}_{ds_name}_*.jsonl")


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 1: Prediction Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def macro_f1(preds: np.ndarray, labels: np.ndarray) -> Tuple[float, Dict]:
    classes = sorted(set(labels.tolist()))
    per_class, f1s = {}, []
    for c in classes:
        tp = int(((preds == c) & (labels == c)).sum())
        fp = int(((preds == c) & (labels != c)).sum())
        fn = int(((preds != c) & (labels == c)).sum())
        prec = tp / max(tp+fp, 1)
        rec  = tp / max(tp+fn, 1)
        f1   = 2*prec*rec / max(prec+rec, 1e-8)
        per_class[IDX2CLS.get(c, str(c))] = {
            "f1": round(f1, 4), "precision": round(prec, 4),
            "recall": round(rec, 4), "n": int((labels==c).sum())
        }
        f1s.append(f1)
    return float(np.mean(f1s)), per_class


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 2: Selective Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def mnv2_error_correction_rate(fused_preds: np.ndarray, mnv2_recs: List[Dict]) -> Dict:
    """MNV2 오분류 케이스에서 fused 예측이 교정하는 비율"""
    from collections import defaultdict
    patterns = defaultdict(lambda: {"total": 0, "corrected": 0})
    n_err, n_corr = 0, 0

    for i, m in enumerate(mnv2_recs):
        if m["true_label"] == "ai_generated" or m["pred_label"] == "ai_generated":
            continue
        if m["pred_label"] != m["true_label"]:
            n_err += 1
            pat = f"{m['true_label']}→{m['pred_label']}"
            patterns[pat]["total"] += 1
            if IDX2CLS.get(int(fused_preds[i])) == m["true_label"]:
                n_corr += 1
                patterns[pat]["corrected"] += 1

    for p in patterns.values():
        p["rate"] = round(p["corrected"] / max(p["total"], 1), 4)

    return {
        "n_errors":    n_err,
        "n_corrected": n_corr,
        "rate":        round(n_corr / max(n_err, 1), 4),
        "patterns":    dict(patterns),
    }


def compute_aurc(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    Area Under Risk-Coverage curve.
    confidence로 정렬하고 threshold를 낮추면서 risk(error rate)와 coverage를 계산.
    낮을수록 좋음.
    """
    conf   = scores.max(axis=1)
    correct = (scores.argmax(axis=1) == labels).astype(float)
    sorted_idx = np.argsort(conf)[::-1]
    n = len(labels)

    risks, coverages = [], []
    for k in range(1, n+1):
        idx  = sorted_idx[:k]
        risk = 1.0 - correct[idx].mean()
        cov  = k / n
        risks.append(risk)
        coverages.append(cov)

    # AUC via trapezoidal
    aurc = float(np.trapz(risks, coverages))
    return aurc


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 3: Calibration Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def compute_ece(scores: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    conf   = scores.max(axis=1)
    correct = (scores.argmax(axis=1) == labels).astype(float)
    bins   = np.linspace(0, 1, n_bins+1)
    ece    = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        m = (conf >= lo) & (conf < hi)
        if i == n_bins-1: m = (conf >= lo) & (conf <= hi)
        if m.sum() == 0: continue
        ece += (m.sum() / len(conf)) * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def compute_brier(probs: np.ndarray, labels: np.ndarray) -> float:
    n_cls  = probs.shape[1]
    one_hot = np.zeros((len(labels), n_cls))
    for i, l in enumerate(labels): one_hot[i, l] = 1.0
    return float(np.mean((probs - one_hot)**2))


# ═══════════════════════════════════════════════════════════════════════════════
# 데이터를 scores 배열로 변환
# ═══════════════════════════════════════════════════════════════════════════════

def mnv2_to_arrays(recs: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    """MNV2 → (scores [N,3], labels [N])"""
    scores = np.array([[r["scores"]["authentic"],
                        r["scores"]["manipulated"],
                        r["scores"].get("ai_generated", 0.0)] for r in recs])
    labels = np.array([CLS2IDX[r["true_label"]] for r in recs])
    return scores, labels


def specm_to_arrays(recs: List[Dict], mnv2_recs: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    """
    SpecM 2-class → (scores [N,3], labels [N]).
    ai_generated samples: SpecM은 2-class이므로 MNV2 scores로 fallback.
    """
    scores_list, labels_list = [], []
    for m, s in zip(mnv2_recs, recs):
        lbl = CLS2IDX[m["true_label"]]
        if m["true_label"] == "ai_generated":
            # SpecM 처리 불가 → MNV2 scores
            sc = np.array([m["scores"]["authentic"],
                           m["scores"]["manipulated"],
                           m["scores"].get("ai_generated", 0.0)])
        else:
            sc = np.array([s["authentic_score"], s["manip_score"], 0.0])
        scores_list.append(sc)
        labels_list.append(lbl)
    return np.array(scores_list), np.array(labels_list)


def cascade_to_arrays(mnv2_recs: List[Dict], specm_recs: List[Dict],
                       tau: float = 0.6) -> Tuple[np.ndarray, np.ndarray]:
    scores_list, labels_list = [], []
    for m, s in zip(mnv2_recs, specm_recs):
        lbl = CLS2IDX[m["true_label"]]
        if m["confidence"] > tau or m["true_label"] == "ai_generated":
            sc = np.array([m["scores"]["authentic"],
                           m["scores"]["manipulated"],
                           m["scores"].get("ai_generated", 0.0)])
        else:
            # SpecM 결정, 3번째 dim(aigen)=0
            sc = np.array([s["authentic_score"], s["manip_score"], 0.0])
        scores_list.append(sc)
        labels_list.append(lbl)
    return np.array(scores_list), np.array(labels_list)


def evaluate_model(scores: np.ndarray, labels: np.ndarray,
                   mnv2_recs: List[Dict]) -> Dict:
    """3-layer 평가"""
    preds = scores.argmax(axis=1)

    mf1, per_cls = macro_f1(preds, labels)
    corr         = mnv2_error_correction_rate(preds, mnv2_recs)
    aurc         = compute_aurc(scores, labels)
    ece          = compute_ece(scores, labels)
    brier        = compute_brier(scores, labels)

    return {
        "macro_f1":     round(mf1, 4),
        "accuracy":     round(float((preds == labels).mean()), 4),
        "per_class":    per_cls,
        "mnv2_err_corr": corr,
        "aurc":          round(aurc, 6),
        "ece":           round(ece, 4),
        "brier":         round(brier, 4),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 메인
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("Phase 4: 3-Layer Evaluation (SHIELD C6)")
    print("=" * 70)

    specm_tag = "gamma1.0_wmax10"  # best from ablation
    all_results = {}

    for ds_name in DATASETS:
        print(f"\n{'─'*60}")
        print(f"  Dataset: {ds_name}")
        print(f"{'─'*60}")

        mnv2_recs   = load_mnv2(ds_name)
        specm_v4    = load_specm_v4(ds_name)
        specm_comp  = load_specm_comp(specm_tag, ds_name)

        ds_results = {}

        # ── 1. MNV2 단독 ──
        sc, lb = mnv2_to_arrays(mnv2_recs)
        ds_results["mnv2"] = evaluate_model(sc, lb, mnv2_recs)
        print(f"  MNV2:         macro_F1={ds_results['mnv2']['macro_f1']:.4f} | "
              f"ECE={ds_results['mnv2']['ece']:.4f} | "
              f"AURC={ds_results['mnv2']['aurc']:.4f}")

        # ── 2. SpecM-v4 단독 ──
        if specm_v4:
            aligned_mnv2, aligned_v4 = [], []
            v4_map = {r["image_path"]: r for r in specm_v4}
            for m in mnv2_recs:
                if m["image_path"] in v4_map:
                    aligned_mnv2.append(m)
                    aligned_v4.append(v4_map[m["image_path"]])
            sc, lb = specm_to_arrays(aligned_v4, aligned_mnv2)
            ds_results["specm_v4"] = evaluate_model(sc, lb, aligned_mnv2)
            print(f"  SpecM-v4:     macro_F1={ds_results['specm_v4']['macro_f1']:.4f} | "
                  f"ECE={ds_results['specm_v4']['ece']:.4f} | "
                  f"AURC={ds_results['specm_v4']['aurc']:.4f}")
        else:
            print(f"  SpecM-v4:     JSONL 없음 (스킵)")

        # ── 3. SpecM-Comp γ=1 단독 ──
        if specm_comp:
            aligned_mnv2, aligned_comp = [], []
            comp_map = {r["image_path"]: r for r in specm_comp}
            for m in mnv2_recs:
                if m["image_path"] in comp_map:
                    aligned_mnv2.append(m)
                    aligned_comp.append(comp_map[m["image_path"]])
            sc, lb = specm_to_arrays(aligned_comp, aligned_mnv2)
            ds_results["specm_comp_g1"] = evaluate_model(sc, lb, aligned_mnv2)
            print(f"  SpecM-Comp γ=1: macro_F1={ds_results['specm_comp_g1']['macro_f1']:.4f} | "
                  f"ECE={ds_results['specm_comp_g1']['ece']:.4f} | "
                  f"AURC={ds_results['specm_comp_g1']['aurc']:.4f}")

            # ── 4. Cascade ──
            sc, lb = cascade_to_arrays(aligned_mnv2, aligned_comp, tau=0.6)
            ds_results["cascade_tau06"] = evaluate_model(sc, lb, aligned_mnv2)
            print(f"  Cascade(τ=0.6): macro_F1={ds_results['cascade_tau06']['macro_f1']:.4f} | "
                  f"ECE={ds_results['cascade_tau06']['ece']:.4f} | "
                  f"AURC={ds_results['cascade_tau06']['aurc']:.4f}")

            # MNV2 오분류 교정률 출력
            for name in ["specm_comp_g1", "cascade_tau06"]:
                corr = ds_results[name]["mnv2_err_corr"]
                print(f"  [{name}] 교정률={corr['rate']:.3f} "
                      f"({corr['n_corrected']}/{corr['n_errors']})")
                for pat, info in corr["patterns"].items():
                    print(f"    {pat}: {info['corrected']}/{info['total']} ({info['rate']:.1%})")

        all_results[ds_name] = ds_results

    # ── 전체 요약 테이블 ──
    print(f"\n{'='*70}")
    print("  SUMMARY (평균 across datasets)")
    print(f"{'='*70}")

    models = ["mnv2", "specm_v4", "specm_comp_g1", "cascade_tau06"]
    print(f"  {'Model':22s} | macro_F1 | err_corr | AURC    | ECE    | Brier")
    print(f"  {'-'*72}")
    for model in models:
        f1s, corrs, aurcs, eces, briers = [], [], [], [], []
        for ds in DATASETS:
            r = all_results.get(ds, {}).get(model)
            if r:
                f1s.append(r["macro_f1"])
                corrs.append(r["mnv2_err_corr"]["rate"])
                aurcs.append(r["aurc"])
                eces.append(r["ece"])
                briers.append(r["brier"])
        if not f1s: continue
        print(f"  {model:22s} | {np.mean(f1s):.4f}   | "
              f"{np.mean(corrs):.4f}   | {np.mean(aurcs):.4f}  | "
              f"{np.mean(eces):.4f} | {np.mean(briers):.4f}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "experiments" / "results" / "phase4_3layer"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"phase4_3layer_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n저장: {out_path}")


if __name__ == "__main__":
    main()
