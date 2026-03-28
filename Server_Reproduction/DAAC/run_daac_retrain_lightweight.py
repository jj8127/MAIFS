#!/usr/bin/env python3
"""
DAAC 메타 분류기 재학습 — 경량 모델 기준 (MNV2 + CLIP + SpecM + SpecG)
======================================================================

원래 DAAC는 4개 무거운 에이전트(Frequency/Noise/FatFormer/Spatial)의
43-dim 메타 특징을 GBM으로 학습.

이 스크립트는 경량 모델 2+2개의 출력으로 새로운 메타 특징을 구성하고
GBM/LR을 재학습합니다.

메타 특징 구조 (~30-dim):
  MNV2 그룹 (7): auth_s, manip_s, aigen_s, verdict_auth, verdict_manip, verdict_aigen, confidence
  CLIP 그룹 (7): auth_s, manip_s, aigen_s, verdict_auth, verdict_manip, verdict_aigen, confidence
  SpecM 그룹 (3, 해당 이미지에만): manip_score, auth_score, specm_available
  SpecG 그룹 (3, 해당 이미지에만): aigen_score, auth_score, specg_available
  Pairwise MNV2↔CLIP (7): disagreement, agree, score_diff×3, confidence_diff, kl_div
  Global (4): max_conf, min_conf, conf_entropy, label_entropy
  총 ~31-dim

실행:
  .venv-qwen/bin/python experiments/run_daac_retrain_lightweight.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

_SCRIPT_DIR = Path(__file__).resolve().parent
_LOCAL_ROOT = _SCRIPT_DIR / "data"
ROOT = _LOCAL_ROOT if _LOCAL_ROOT.exists() else Path(__file__).resolve().parents[1]
BE = ROOT / "experiments" / "results" / "backbone_eval"
SE = ROOT / "experiments" / "results" / "specialist_eval"
OUT_DIR = ROOT / "experiments" / "results" / "daac_retrain"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = ["base", "dsC", "opensdi", "aigenproxy"]
CLASSES  = ["authentic", "manipulated", "ai_generated"]
SEEDS    = list(range(42, 52))  # 10 seeds

MNV2_FILES = {
    ds: BE / f"mobilenetv2_dualstream_{ds}_20260319_070725.jsonl"
    for ds in DATASETS
}
CLIP_FILES = {
    ds: BE / f"mobileclip_s2_finetuned_{ds}_20260319_061834.jsonl"
    for ds in DATASETS
}
SPEC_M_FILES = {
    ds: (next(SE.glob(f"specialist_m_v2_{ds}_*.jsonl"), None)
         or next(SE.glob(f"specialist_m_{ds}_*.jsonl"), None))
    for ds in DATASETS
}
SPEC_G_FILES = {
    ds: next(SE.glob(f"specialist_g_{ds}_*.jsonl"), None)
    for ds in DATASETS
}


# ── 데이터 로드 ─────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> Dict[str, dict]:
    recs = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line.strip())
            recs[r["image_path"]] = r
    return recs


def load_dataset(ds: str) -> Optional[Dict[str, dict]]:
    """4개 모델 레코드를 image_path 기준으로 통합."""
    if not MNV2_FILES[ds].exists() or not CLIP_FILES[ds].exists():
        return None
    mnv2 = load_jsonl(MNV2_FILES[ds])
    clip  = load_jsonl(CLIP_FILES[ds])
    specm = load_jsonl(SPEC_M_FILES[ds]) if SPEC_M_FILES[ds] else {}
    specg = load_jsonl(SPEC_G_FILES[ds]) if SPEC_G_FILES[ds] else {}

    merged = {}
    for key in set(mnv2) & set(clip):
        merged[key] = {
            "mnv2":  mnv2[key],
            "clip":  clip[key],
            "specm": specm.get(key),
            "specg": specg.get(key),
            "true_label": mnv2[key]["true_label"],
        }
    return merged


# ── 메타 특징 추출 ──────────────────────────────────────────────────────────

def _kl_div(p: np.ndarray, q: np.ndarray, eps: float = 1e-8) -> float:
    p = np.clip(p, eps, 1.0); q = np.clip(q, eps, 1.0)
    return float(np.sum(p * np.log(p / q)))


def extract_features(rec: dict) -> np.ndarray:
    """단일 이미지 레코드 → 메타 특징 벡터 (25-dim, MNV2+CLIP 기반)."""
    mnv2  = rec["mnv2"]
    clip  = rec["clip"]
    specm = rec["specm"]
    specg = rec["specg"]

    # MNV2 그룹 (7-dim)
    ms = mnv2["scores"]
    m_scores = np.array([ms["authentic"], ms["manipulated"], ms["ai_generated"]])
    m_pred   = np.array([1 if mnv2["pred_label"] == c else 0 for c in CLASSES])
    m_conf   = mnv2["confidence"]

    # CLIP 그룹 (7-dim)
    cs = clip["scores"]
    c_scores = np.array([cs["authentic"], cs["manipulated"], cs["ai_generated"]])
    c_pred   = np.array([1 if clip["pred_label"] == c else 0 for c in CLASSES])
    c_conf   = clip["confidence"]

    # SpecM/SpecG: DAAC 메타 특징에는 포함하지 않음
    # (specialist는 ICWMV에서 별도 활용, DAAC는 generalist 2개 기반으로 공정 비교)

    # Pairwise MNV2↔CLIP (7-dim)
    disagree    = float(mnv2["pred_label"] != clip["pred_label"])
    agree       = 1.0 - disagree
    score_diffs = np.abs(m_scores - c_scores)   # 3-dim
    conf_diff   = abs(m_conf - c_conf)
    kl          = _kl_div(m_scores, c_scores)

    # Global (4-dim)
    all_confs  = [m_conf, c_conf]
    max_conf   = max(all_confs)
    min_conf   = min(all_confs)
    conf_std   = float(np.std(all_confs))
    avg_score  = (m_scores + c_scores) / 2.0
    lbl_entropy = float(-np.sum(avg_score * np.log(np.clip(avg_score, 1e-8, 1.0))))

    feat = np.concatenate([
        m_scores, m_pred, [m_conf],          # 7
        c_scores, c_pred, [c_conf],          # 7
        [disagree, agree], score_diffs,      # 5
        [conf_diff, kl],                     # 2
        [max_conf, min_conf, conf_std, lbl_entropy],  # 4
    ])                                       # 총 25-dim
    return feat.astype(np.float32)


def build_Xy(merged: Dict[str, dict]) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for rec in merged.values():
        X.append(extract_features(rec))
        y.append(rec["true_label"])
    return np.array(X), np.array(y)


# ── 베이스라인: MV, 2-model avg ─────────────────────────────────────────────

def majority_vote_f1(merged: Dict[str, dict]) -> float:
    y_true, y_pred = [], []
    for rec in merged.values():
        y_true.append(rec["true_label"])
        # 단순 다수결: MNV2 + CLIP 동의 시 → 그 레이블, 불일치 시 → 더 높은 confidence
        if rec["mnv2"]["pred_label"] == rec["clip"]["pred_label"]:
            y_pred.append(rec["mnv2"]["pred_label"])
        else:
            if rec["mnv2"]["confidence"] >= rec["clip"]["confidence"]:
                y_pred.append(rec["mnv2"]["pred_label"])
            else:
                y_pred.append(rec["clip"]["pred_label"])
    return f1_score(y_true, y_pred, average="macro")


def avg_score_f1(merged: Dict[str, dict]) -> float:
    y_true, y_pred = [], []
    for rec in merged.values():
        y_true.append(rec["true_label"])
        ms = rec["mnv2"]["scores"]
        cs = rec["clip"]["scores"]
        avg = {c: (ms[c] + cs[c]) / 2.0 for c in CLASSES}
        y_pred.append(max(avg, key=avg.get))
    return f1_score(y_true, y_pred, average="macro")


# ── 분류기 학습/평가 ────────────────────────────────────────────────────────

def train_and_eval(
    train_merged: Dict[str, dict],
    eval_datasets: Dict[str, Dict[str, dict]],
    seed: int = 42,
) -> dict:
    X_tr, y_tr = build_Xy(train_merged)
    le = LabelEncoder().fit(CLASSES)
    y_tr_enc = le.transform(y_tr)

    models = {
        "GBM": GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=seed),
        "LR":  LogisticRegression(
            max_iter=2000, C=1.0, random_state=seed, multi_class="multinomial"),
    }
    results = {}
    for name, clf in models.items():
        clf.fit(X_tr, y_tr_enc)
        results[name] = {}
        for ds_key, merged in eval_datasets.items():
            X_ev, y_ev = build_Xy(merged)
            y_pred_enc = clf.predict(X_ev)
            y_pred = le.inverse_transform(y_pred_enc)
            f1 = f1_score(y_ev, y_pred, average="macro")
            results[name][ds_key] = f1
    return results


# ── 메인 ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("DAAC 메타 분류기 재학습 — 경량 모델 (MNV2 + CLIP + SpecM + SpecG)")
    print("=" * 65)

    # 데이터 로드
    all_data = {}
    for ds in DATASETS:
        d = load_dataset(ds)
        if d:
            all_data[ds] = d
            specm_avail = sum(1 for r in d.values() if r["specm"] is not None)
            specg_avail = sum(1 for r in d.values() if r["specg"] is not None)
            print(f"  [{ds}] n={len(d)}, specm={specm_avail}, specg={specg_avail}")
        else:
            print(f"  [{ds}] 데이터 없음")

    if "base" not in all_data:
        print("[FAIL] base 데이터셋 없음")
        return

    # 특징 차원 확인
    sample_feat = extract_features(next(iter(all_data["base"].values())))
    print(f"\n  메타 특징 차원: {len(sample_feat)}-dim")

    # 베이스라인
    print("\n[베이스라인 (base 데이터셋)]")
    mv_f1   = majority_vote_f1(all_data["base"])
    avg_f1  = avg_score_f1(all_data["base"])
    print(f"  Majority Vote:  {mv_f1:.4f}")
    print(f"  Avg-Score:      {avg_f1:.4f}")

    # Protocol-P: 10-seed 학습/평가 (base 85%/15% split)
    print(f"\n[Protocol-P] base 10-seed 85/15 split 학습 → 4-DS 평가")
    keys_base = list(all_data["base"].keys())
    seed_results = defaultdict(lambda: defaultdict(list))

    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(keys_base))
        n_tr  = int(len(keys_base) * 0.85)
        tr_keys = [keys_base[i] for i in perm[:n_tr]]
        tr_data  = {k: all_data["base"][k] for k in tr_keys}
        eval_data = {ds: all_data[ds] for ds in DATASETS if ds in all_data}

        res = train_and_eval(tr_data, eval_data, seed=seed)
        for clf_name, ds_f1s in res.items():
            for ds, f1 in ds_f1s.items():
                seed_results[clf_name][ds].append(f1)

    print(f"\n{'모델':<8} {'base':>8} {'dsC':>8} {'opensdi':>8} {'aigenproxy':>10} {'avg':>8}")
    print("-" * 55)
    summary = {}
    for clf_name in ["GBM", "LR"]:
        row = {}
        for ds in DATASETS:
            if ds in seed_results[clf_name]:
                row[ds] = float(np.mean(seed_results[clf_name][ds]))
            else:
                row[ds] = 0.0
        avg = np.mean(list(row.values()))
        cells = "  ".join(f"{100*row[ds]:7.2f}%" for ds in DATASETS)
        print(f"{clf_name:<8} {cells}  {100*avg:7.2f}%")
        summary[clf_name] = {**row, "avg": float(avg)}

    # 원본 DAAC와 비교
    print(f"\n[비교 요약] (base 기준)")
    orig_daac_base = 0.8613  # 원본 DAAC GBM (무거운 에이전트 기반)
    orig_cobra     = 0.266
    new_gbm_base   = summary["GBM"].get("base", 0.0)
    new_lr_base    = summary["LR"].get("base", 0.0)
    print(f"  원본 COBRA (무거운):          {100*orig_cobra:.2f}%")
    print(f"  원본 DAAC-GBM (무거운):       {100*orig_daac_base:.2f}%")
    print(f"  경량 DAAC-GBM (MNV2+CLIP):   {100*new_gbm_base:.2f}%  (Δ {100*(new_gbm_base-orig_daac_base):+.2f}%p vs 원본)")
    print(f"  경량 DAAC-LR  (MNV2+CLIP):   {100*new_lr_base:.2f}%")
    print(f"  ICWMV 4-model avg (참고):     96.48%")

    # 특징 중요도 (GBM)
    print(f"\n[GBM 특징 중요도 Top-10]")
    feat_names = (
        ["mnv2_auth", "mnv2_manip", "mnv2_aigen", "mnv2_v_auth", "mnv2_v_manip", "mnv2_v_aigen", "mnv2_conf"] +
        ["clip_auth", "clip_manip", "clip_aigen", "clip_v_auth", "clip_v_manip", "clip_v_aigen", "clip_conf"] +
        ["disagree", "agree", "diff_auth", "diff_manip", "diff_aigen"] +
        ["conf_diff", "kl_div"] +
        ["max_conf", "min_conf", "conf_std", "lbl_entropy"]
    )  # 25-dim
    # 마지막 seed의 GBM으로 중요도 확인
    rng = np.random.default_rng(SEEDS[-1])
    perm = rng.permutation(len(keys_base))
    n_tr = int(len(keys_base) * 0.85)
    tr_keys = [keys_base[i] for i in perm[:n_tr]]
    tr_data = {k: all_data["base"][k] for k in tr_keys}
    X_tr, y_tr = build_Xy(tr_data)
    le = LabelEncoder().fit(CLASSES)
    gbm = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=SEEDS[-1])
    gbm.fit(X_tr, le.transform(y_tr))
    importances = gbm.feature_importances_
    top_idx = np.argsort(importances)[::-1][:10]
    for rank, idx in enumerate(top_idx, 1):
        name = feat_names[idx] if idx < len(feat_names) else f"feat_{idx}"
        print(f"  {rank:2d}. {name:<20} {100*importances[idx]:6.2f}%")

    # 결과 저장
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUT_DIR / f"daac_retrain_lightweight_{ts}.json"
    with open(out_path, "w") as f:
        json.dump({
            "description": "DAAC 메타 분류기 재학습 (경량 MNV2+CLIP+SpecM+SpecG 기준)",
            "n_features":  int(len(sample_feat)),
            "n_seeds":     len(SEEDS),
            "baselines":   {"majority_vote": mv_f1, "avg_score": avg_f1},
            "original_daac_gbm_base": orig_daac_base,
            "original_cobra_base":    orig_cobra,
            "results":     summary,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n결과 저장: {out_path}")


if __name__ == "__main__":
    main()
