#!/usr/bin/env python3
"""
ICWMV (Individual Confidence Weighted Majority Voting) Consensus
=================================================================

4개 모델 통합:
  1. MobileNetV2 dual-stream   (3-class: auth/manip/aigen)
  2. MobileCLIP-ft4            (3-class: auth/manip/aigen)
  3. Specialist-M              (binary: auth vs manip)
  4. Specialist-G              (binary: auth vs aigen)

Score Fusion (class-wise weighted average):
  S(auth)  = avg_w [ MNV2(auth), CLIP(auth), SpecM(auth), SpecG(auth) ]
  S(manip) = avg_w [ MNV2(manip), CLIP(manip), SpecM(manip) ]       ← SpecG 기여 없음
  S(aigen) = avg_w [ MNV2(aigen), CLIP(aigen), SpecG(aigen) ]       ← SpecM 기여 없음
  → renormalize → argmax

실행:
  .venv-qwen/bin/python experiments/run_icwmv_consensus.py
  .venv-qwen/bin/python experiments/run_icwmv_consensus.py --w-spec 1.5
  .venv-qwen/bin/python experiments/run_icwmv_consensus.py --grid-search
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT    = Path(__file__).resolve().parents[1]
RES     = ROOT / "experiments" / "results"
BE      = RES / "backbone_eval"
SE      = RES / "specialist_eval"
OUT_DIR = RES / "icwmv"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = ["base", "dsC", "opensdi", "aigenproxy"]
CLASSES  = ["authentic", "manipulated", "ai_generated"]

# ── 파일 경로 설정 ──────────────────────────────────────────────────────────

def _latest(pattern: str) -> Optional[Path]:
    """glob 패턴에서 가장 최신 파일 반환"""
    files = sorted(Path("/").glob(pattern.lstrip("/")))
    return files[-1] if files else None


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

# ── JSONL 로더 ─────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> Dict[str, dict]:
    records = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line.strip())
            records[r["image_path"]] = r
    return records


# ── Score Fusion ───────────────────────────────────────────────────────────

def fuse_scores(
    mnv2: dict,
    clip: dict,
    spec_m: Optional[dict],
    spec_g: Optional[dict],
    w_mnv2: float = 1.0,
    w_clip: float  = 1.0,
    w_spec: float  = 1.0,
) -> Tuple[str, np.ndarray]:
    """
    class-wise weighted average fusion.

    Returns:
        pred_label: str
        probs: np.ndarray shape (3,) [auth, manip, aigen]
    """
    # 각 클래스별 기여 합산
    auth_sum  = w_mnv2 * mnv2["scores"]["authentic"]  + w_clip * clip["scores"]["authentic"]
    manip_sum = w_mnv2 * mnv2["scores"]["manipulated"] + w_clip * clip["scores"]["manipulated"]
    aigen_sum = w_mnv2 * mnv2["scores"]["ai_generated"] + w_clip * clip["scores"]["ai_generated"]

    auth_w  = w_mnv2 + w_clip
    manip_w = w_mnv2 + w_clip
    aigen_w = w_mnv2 + w_clip

    # Specialist-M: authentic / manipulated에 기여
    if spec_m is not None:
        auth_sum  += w_spec * spec_m["authentic_score"]
        manip_sum += w_spec * spec_m["manip_score"]
        auth_w    += w_spec
        manip_w   += w_spec

    # Specialist-G: authentic / ai_generated에 기여
    if spec_g is not None:
        auth_sum  += w_spec * spec_g["authentic_score"]
        aigen_sum += w_spec * spec_g["aigen_score"]
        auth_w    += w_spec
        aigen_w   += w_spec

    # class-wise 평균 (각 클래스 독립적으로 정규화)
    s_auth  = auth_sum  / auth_w
    s_manip = manip_sum / manip_w
    s_aigen = aigen_sum / aigen_w

    # 전체 renormalize → 확률 분포로
    raw = np.array([s_auth, s_manip, s_aigen], dtype=float)
    probs = raw / raw.sum()

    pred_idx  = int(np.argmax(probs))
    pred_label = CLASSES[pred_idx]
    return pred_label, probs


# ── Per-dataset 평가 ────────────────────────────────────────────────────────

def evaluate_dataset(
    ds: str,
    w_mnv2: float = 1.0,
    w_clip: float  = 1.0,
    w_spec: float  = 1.0,
    verbose: bool  = False,
) -> Optional[dict]:
    for key, path in [("MNV2", MNV2_FILES[ds]), ("CLIP", CLIP_FILES[ds])]:
        if not path or not path.exists():
            if verbose:
                print(f"  [{ds}] {key} 파일 없음")
            return None

    mnv2_recs  = load_jsonl(MNV2_FILES[ds])
    clip_recs  = load_jsonl(CLIP_FILES[ds])
    spec_m_recs = load_jsonl(SPEC_M_FILES[ds]) if SPEC_M_FILES[ds] else {}
    spec_g_recs = load_jsonl(SPEC_G_FILES[ds]) if SPEC_G_FILES[ds] else {}

    # 공통 키 (MNV2 & CLIP 모두 있는 이미지)
    common_keys = sorted(set(mnv2_recs) & set(clip_recs))

    # 예측 수집
    y_true, y_pred = [], []
    both_wrong_fixed = both_wrong_total = 0
    both_wrong_details = defaultdict(int)

    for key in common_keys:
        true_label = mnv2_recs[key]["true_label"]
        spec_m = spec_m_recs.get(key)
        spec_g = spec_g_recs.get(key)

        pred_label, probs = fuse_scores(
            mnv2_recs[key], clip_recs[key],
            spec_m, spec_g,
            w_mnv2=w_mnv2, w_clip=w_clip, w_spec=w_spec,
        )
        y_true.append(true_label)
        y_pred.append(pred_label)

        # "둘 다 틀리는" 케이스 추적
        mnv2_wrong = (mnv2_recs[key]["pred_label"] != true_label)
        clip_wrong = (clip_recs[key]["pred_label"]  != true_label)
        if mnv2_wrong and clip_wrong:
            both_wrong_total += 1
            if pred_label == true_label:
                both_wrong_fixed += 1
            both_wrong_details[(true_label, pred_label)] += 1

    # Macro-F1
    per_class_stats = {}
    for cls in CLASSES:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class_stats[cls] = {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}

    classes_present = [c for c in CLASSES if any(t == c for t in y_true)]
    macro_f1  = np.mean([per_class_stats[c]["f1"] for c in classes_present])
    accuracy  = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)

    return {
        "dataset":          ds,
        "n":                len(common_keys),
        "accuracy":         accuracy,
        "macro_f1":         macro_f1,
        "per_class":        per_class_stats,
        "both_wrong_total": both_wrong_total,
        "both_wrong_fixed": both_wrong_fixed,
        "both_wrong_fix_rate": both_wrong_fixed / both_wrong_total if both_wrong_total else 0.0,
        "both_wrong_details": {str(k): v for k, v in sorted(both_wrong_details.items(), key=lambda x: -x[1])},
    }


# ── 단일 모델 baseline ──────────────────────────────────────────────────────

def eval_single_model(recs: Dict[str, dict], label_key: str = "pred_label") -> dict:
    """기존 3-class 모델 단독 평가"""
    y_true = [r["true_label"] for r in recs.values()]
    y_pred = [r[label_key]    for r in recs.values()]

    per_class = {}
    for cls in CLASSES:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        per_class[cls] = f1
    classes_present = [c for c in CLASSES if any(t == c for t in y_true)]
    macro_f1 = np.mean([per_class[c] for c in classes_present])
    acc = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)
    return {"accuracy": acc, "macro_f1": macro_f1, "per_class_f1": per_class}


# ── ICWMV 2-model (MNV2+CLIP only) ─────────────────────────────────────────

def eval_2model_icwmv(mnv2_recs, clip_recs, ds) -> dict:
    common = sorted(set(mnv2_recs) & set(clip_recs))
    y_true, y_pred = [], []
    for key in common:
        true_label = mnv2_recs[key]["true_label"]
        scores = {}
        for cls in CLASSES:
            scores[cls] = (mnv2_recs[key]["scores"][cls] + clip_recs[key]["scores"][cls]) / 2.0
        pred_label = max(scores, key=scores.get)
        y_true.append(true_label)
        y_pred.append(pred_label)

    per_class = {}
    for cls in CLASSES:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)
        p_ = tp / (tp + fp) if (tp + fp) else 0.0
        r_ = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p_ * r_ / (p_ + r_) if (p_ + r_) else 0.0
        per_class[cls] = f1
    classes_present = [c for c in CLASSES if any(t == c for t in y_true)]
    macro_f1 = np.mean([per_class[c] for c in classes_present])
    acc = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)
    return {"accuracy": acc, "macro_f1": macro_f1, "per_class_f1": per_class}


# ── Grid Search ─────────────────────────────────────────────────────────────

def grid_search(train_ds: str = "base") -> Tuple[float, dict]:
    """specialist weight 그리드 서치 (train_ds 기준)"""
    best_f1, best_cfg = -1, {}
    for w_spec in [0.3, 0.5, 0.7, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0]:
        r = evaluate_dataset(train_ds, w_spec=w_spec, verbose=False)
        if r and r["macro_f1"] > best_f1:
            best_f1 = r["macro_f1"]
            best_cfg = {"w_spec": w_spec, "macro_f1": best_f1}
    return best_f1, best_cfg


# ── 메인 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ICWMV 4-Model Consensus 평가")
    parser.add_argument("--w-spec",     type=float, default=1.0, help="Specialist 가중치")
    parser.add_argument("--grid-search", action="store_true",    help="w_spec 그리드 서치 (base 기준)")
    args = parser.parse_args()

    w_spec = args.w_spec

    if args.grid_search:
        print("\n[Grid Search] base 데이터셋 기준 w_spec 탐색...")
        _, best_cfg = grid_search("base")
        print(f"  best w_spec = {best_cfg['w_spec']} (macro_f1={best_cfg['macro_f1']:.4f})")
        w_spec = best_cfg["w_spec"]

    print(f"\n{'='*70}")
    print(f"ICWMV 4-Model Consensus Evaluation  (w_spec={w_spec})")
    print(f"{'='*70}")
    print(f"\n{'모델/데이터셋':<28} {'base':>8} {'dsC':>8} {'opensdi':>8} {'aigenproxy':>10}  {'avg':>8}")
    print("-" * 70)

    all_results = {}
    model_rows = {
        "MNV2 (단독)":      [],
        "CLIP (단독)":      [],
        "ICWMV 2모델":      [],
        f"ICWMV 4모델 (w={w_spec})": [],
    }

    both_wrong_rows = []

    for ds in DATASETS:
        if not MNV2_FILES[ds].exists() or not CLIP_FILES[ds].exists():
            for k in model_rows:
                model_rows[k].append(None)
            continue

        mnv2_recs = load_jsonl(MNV2_FILES[ds])
        clip_recs  = load_jsonl(CLIP_FILES[ds])

        mnv2_res  = eval_single_model(mnv2_recs)
        clip_res  = eval_single_model(clip_recs)
        two_res   = eval_2model_icwmv(mnv2_recs, clip_recs, ds)
        four_res  = evaluate_dataset(ds, w_spec=w_spec, verbose=True)

        model_rows["MNV2 (단독)"].append(mnv2_res["macro_f1"])
        model_rows["CLIP (단독)"].append(clip_res["macro_f1"])
        model_rows["ICWMV 2모델"].append(two_res["macro_f1"])
        model_rows[f"ICWMV 4모델 (w={w_spec})"].append(four_res["macro_f1"] if four_res else None)

        if four_res:
            all_results[ds] = four_res
            both_wrong_rows.append((ds, four_res["both_wrong_total"], four_res["both_wrong_fixed"], four_res["both_wrong_fix_rate"]))

    # 표 출력
    for row_name, vals in model_rows.items():
        cells = []
        valid = [v for v in vals if v is not None]
        for v in vals:
            cells.append(f"{100*v:7.2f}%" if v is not None else "    N/A ")
        avg = f"{100*np.mean(valid):7.2f}%" if valid else "    N/A "
        print(f"{row_name:<28} {'  '.join(cells)}  {avg}")

    # 상세 per-class 분석
    if all_results:
        print(f"\n{'='*70}")
        print("Per-class F1 상세 (ICWMV 4모델)")
        print(f"{'='*70}")
        print(f"{'데이터셋':<12} {'Auth-F1':>10} {'Manip-F1':>10} {'Aigen-F1':>10} {'Macro-F1':>10}")
        for ds, r in all_results.items():
            auth_f1  = r["per_class"].get("authentic",    {}).get("f1", 0.0)
            manip_f1 = r["per_class"].get("manipulated",  {}).get("f1", 0.0)
            aigen_f1 = r["per_class"].get("ai_generated", {}).get("f1", 0.0)
            print(f"{ds:<12} {100*auth_f1:>9.2f}% {100*manip_f1:>9.2f}% {100*aigen_f1:>9.2f}% {100*r['macro_f1']:>9.2f}%")

    # "둘 다 틀리는" 케이스 분석
    if both_wrong_rows:
        print(f"\n{'='*70}")
        print("[둘 다 틀리는 케이스 수정률] MNV2 & CLIP 모두 오분류 → ICWMV 4모델 수정")
        print(f"{'='*70}")
        print(f"{'데이터셋':<12} {'Both Wrong':>12} {'Fixed':>8} {'Fix Rate':>10}")
        for ds, total, fixed, rate in both_wrong_rows:
            print(f"{ds:<12} {total:>12} {fixed:>8} {100*rate:>9.1f}%")

        # 수정된 케이스의 혼동 패턴
        print("\n[수정 케이스 패턴 (true → ICWMV pred)]")
        for ds, r in all_results.items():
            if r["both_wrong_total"] == 0:
                continue
            print(f"\n  {ds}:")
            for pattern, cnt in list(r["both_wrong_details"].items())[:8]:
                print(f"    {pattern}: {cnt}개")

    # 개선량 요약
    print(f"\n{'='*70}")
    print("개선량 요약 (vs MNV2 단독)")
    print(f"{'='*70}")
    for ds in DATASETS:
        mnv2_val = model_rows["MNV2 (단독)"][DATASETS.index(ds)]
        four_val = model_rows[f"ICWMV 4모델 (w={w_spec})"][DATASETS.index(ds)]
        if mnv2_val is not None and four_val is not None:
            delta = (four_val - mnv2_val) * 100
            sign = "+" if delta >= 0 else ""
            print(f"  {ds:<12}: {sign}{delta:.2f}%p  ({100*mnv2_val:.2f}% → {100*four_val:.2f}%)")

    # 결과 저장
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUT_DIR / f"icwmv_4model_wspec{w_spec}_{ts}.json"
    with open(out_path, "w") as f:
        json.dump({
            "description": "ICWMV 4-model consensus (MNV2 + CLIP + SpecM + SpecG)",
            "config":  {"w_spec": w_spec},
            "results": all_results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n결과 저장: {out_path}")


if __name__ == "__main__":
    main()
