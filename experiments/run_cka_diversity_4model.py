#!/usr/bin/env python3
"""
4-model 시스템(MNV2 + CLIP + SpecM-v2 + SpecG) 예측 다양성 분석
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
논문 contribution 강화용: 4-model 시스템이 2-model(MNV2+CLIP) 대비
출력 공간에서 더 다양한 예측을 제공하는지 검증.

분석 항목:
  1. Output-space CKA (linear kernel)
  2. Error Jaccard: 모든 pairwise 에러 집합 Jaccard 지수
  3. 평균 예측 불일치율 (pairwise disagreement rate)
  4. 2-model vs 4-model 다양성 요약

실행 (MAIFS/ 루트에서):
    .venv-qwen/bin/python experiments/run_cka_diversity_4model.py

출력:
    experiments/results/cka_diversity/cka_diversity_4model_{ts}.json
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── 경로 설정 ──────────────────────────────────────────────────────────── #
ROOT = Path(__file__).resolve().parents[1]   # MAIFS/
RESULTS_DIR = ROOT / "experiments" / "results"
OUT_DIR = RESULTS_DIR / "cka_diversity"

DATASETS = ["base", "dsC", "opensdi", "aigenproxy"]

# ── 파일 경로 템플릿 ────────────────────────────────────────────────────── #
MNV2_TS  = "20260319_070725"
CLIP_TS  = "20260319_061834"

MNV2_TMPL  = str(RESULTS_DIR / "backbone_eval") + "/mobilenetv2_dualstream_{ds}_" + MNV2_TS + ".jsonl"
CLIP_TMPL  = str(RESULTS_DIR / "backbone_eval") + "/mobileclip_s2_finetuned_{ds}_" + CLIP_TS + ".jsonl"
SPECM_GLOB = str(RESULTS_DIR / "specialist_eval") + "/specialist_m_v2_{ds}_*.jsonl"
SPECG_GLOB = str(RESULTS_DIR / "specialist_eval") + "/specialist_g_{ds}_*.jsonl"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
#  데이터 로딩                                                                #
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #

def _glob_latest(pattern: str) -> Optional[Path]:
    """glob 패턴에 매칭되는 파일 중 가장 최신 파일 반환."""
    matches = sorted(glob(pattern))
    return Path(matches[-1]) if matches else None


def load_jsonl(path: Path) -> List[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_model_records(ds: str) -> Dict[str, Dict[str, dict]]:
    """
    각 모델의 {image_path: record} 딕셔너리를 반환.
    반환 키: "mnv2", "clip", "specm", "specg"
    """
    files = {
        "mnv2":  Path(MNV2_TMPL.format(ds=ds)),
        "clip":  Path(CLIP_TMPL.format(ds=ds)),
        "specm": _glob_latest(SPECM_GLOB.format(ds=ds)),
        "specg": _glob_latest(SPECG_GLOB.format(ds=ds)),
    }

    result: Dict[str, Dict[str, dict]] = {}
    for model, fpath in files.items():
        if fpath is None or not fpath.exists():
            print(f"  [WARN] {model} 파일 없음: {fpath}", file=sys.stderr)
            result[model] = {}
            continue
        records = load_jsonl(fpath)
        result[model] = {r["image_path"]: r for r in records}
        print(f"  {model:6s}: {len(records):5d} records  <- {fpath.name}")

    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
#  softmax 벡터 추출                                                          #
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
# 클래스 순서: [authentic, manipulated, ai_generated]
_LABEL_TO_IDX = {"authentic": 0, "manipulated": 1, "ai_generated": 2}


def _scores_from_record(record: dict, model: str) -> np.ndarray:
    """
    레코드 하나를 [auth, manip, aigen] shape=(3,) 벡터로 변환.

    모델별 규칙:
      mnv2 / clip : record["scores"] = {"authentic":, "manipulated":, "ai_generated":}
      specm       : record["authentic_score"], record["manip_score"]  → (auth, manip, 0) padding
      specg       : record["authentic_score"], record["aigen_score"]  → (auth, 0, aigen) padding
    """
    if model in ("mnv2", "clip"):
        s = record["scores"]
        return np.array([
            s.get("authentic",    0.0),
            s.get("manipulated",  0.0),
            s.get("ai_generated", 0.0),
        ], dtype=np.float64)

    elif model == "specm":
        auth  = record.get("authentic_score", 0.0)
        manip = record.get("manip_score",     0.0)
        # 합이 1이 되도록 정규화(이미 합산 ≈ 1이지만 padding 후 재정규화)
        total = auth + manip
        if total > 0:
            auth, manip = auth / total, manip / total
        vec = np.array([auth, manip, 0.0], dtype=np.float64)
        return vec

    elif model == "specg":
        auth  = record.get("authentic_score", 0.0)
        aigen = record.get("aigen_score",     0.0)
        total = auth + aigen
        if total > 0:
            auth, aigen = auth / total, aigen / total
        vec = np.array([auth, 0.0, aigen], dtype=np.float64)
        return vec

    else:
        raise ValueError(f"Unknown model: {model}")


def build_score_matrix(
    image_paths: List[str],
    model_data: Dict[str, dict],
    model: str,
) -> np.ndarray:
    """n×3 행렬 반환. 레코드 누락 시 uniform [1/3,1/3,1/3] 사용."""
    mat = np.empty((len(image_paths), 3), dtype=np.float64)
    uniform = np.full(3, 1.0 / 3.0)
    for i, ip in enumerate(image_paths):
        if ip in model_data:
            mat[i] = _scores_from_record(model_data[ip], model)
        else:
            mat[i] = uniform
    return mat


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
#  linear CKA                                                                 #
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #

def _center(K: np.ndarray) -> np.ndarray:
    """Gram 행렬 이중 중심화."""
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Linear CKA between two n×d matrices X, Y.
      CKA(X, Y) = HSIC(X, Y) / sqrt(HSIC(X, X) * HSIC(Y, Y))
    HSIC 는 linear kernel 기반 이중 중심화 추정량 사용.
    반환값: [0, 1] (1에 가까울수록 표현 공간 유사)
    """
    Kx = _center(X @ X.T)
    Ky = _center(Y @ Y.T)
    n  = X.shape[0]
    hsic_xy = np.trace(Kx @ Ky) / (n ** 2)
    hsic_xx = np.trace(Kx @ Kx) / (n ** 2)
    hsic_yy = np.trace(Ky @ Ky) / (n ** 2)
    denom = np.sqrt(hsic_xx * hsic_yy)
    if denom < 1e-12:
        return float("nan")
    return float(hsic_xy / denom)


def compute_cka_matrix(
    model_vectors: Dict[str, np.ndarray],
) -> Dict[str, float]:
    """
    모든 pairwise linear CKA 계산.
    반환: {"A_vs_B": cka_value, ...}
    """
    models = list(model_vectors.keys())
    results: Dict[str, float] = {}
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            a, b = models[i], models[j]
            key = f"{a}_vs_{b}"
            results[key] = linear_cka(model_vectors[a], model_vectors[b])
    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
#  에러 Jaccard + 불일치율                                                    #
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #

def _pred_label(record: dict, model: str) -> str:
    """레코드에서 예측 레이블 반환."""
    return record.get("pred_label", "authentic")


def compute_error_jaccard(
    image_paths: List[str],
    model_data_all: Dict[str, Dict[str, dict]],
    model_names: List[str],
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    각 모델의 오류 집합(인덱스 기준)을 구하고,
    pairwise Jaccard 지수와 불일치율을 반환.
    true_label 은 첫 번째 모델(mnv2) 레코드에서 참조.
    """
    # true_label 맵
    true_labels: Dict[str, str] = {}
    for ip in image_paths:
        for m in model_names:
            rec = model_data_all[m].get(ip)
            if rec:
                true_labels[ip] = rec["true_label"]
                break

    # 각 모델의 에러 집합 (image_path 기준)
    error_sets: Dict[str, set] = {}
    pred_seqs:  Dict[str, List[str]] = {}
    for m in model_names:
        err  = set()
        preds = []
        for ip in image_paths:
            rec = model_data_all[m].get(ip)
            pl  = _pred_label(rec, m) if rec else "authentic"
            tl  = true_labels.get(ip, "authentic")
            preds.append(pl)
            if pl != tl:
                err.add(ip)
        error_sets[m] = err
        pred_seqs[m]  = preds

    # pairwise Jaccard
    jaccard: Dict[str, float] = {}
    disagree: Dict[str, float] = {}
    n = len(image_paths)
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            a, b = model_names[i], model_names[j]
            key = f"{a}_vs_{b}"
            ea, eb = error_sets[a], error_sets[b]
            inter = len(ea & eb)
            union = len(ea | eb)
            jaccard[key] = inter / union if union > 0 else float("nan")
            # 불일치율
            diffs = sum(1 for pa, pb in zip(pred_seqs[a], pred_seqs[b]) if pa != pb)
            disagree[key] = diffs / n if n > 0 else 0.0

    return jaccard, disagree


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
#  데이터셋별 분석                                                             #
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #

def analyze_dataset(ds: str) -> dict:
    """
    단일 데이터셋에 대해 4-model / 2-model 다양성 분석 수행.

    공통 이미지 선택 전략:
      - 4-model CKA: mnv2, clip, specm, specg 모두에 레코드가 있는 이미지
        (SpecM은 authentic/manipulated만 평가하므로 aigenproxy에서는 공통 집합이 작아짐)
      - 2-model CKA: mnv2, clip에 레코드가 있는 이미지 (더 넓은 집합)
    """
    print(f"\n{'='*60}")
    print(f"  Dataset: {ds}")
    print(f"{'='*60}")

    model_data = load_model_records(ds)
    all_models = ["mnv2", "clip", "specm", "specg"]

    # ── 4-model 공통 이미지 집합 ─────────────────────────────────────────
    sets_4 = [set(model_data[m].keys()) for m in all_models if model_data[m]]
    if len(sets_4) < len(all_models):
        print(f"  [WARN] 일부 모델 데이터 누락 — 4-model 분석 제한됨", file=sys.stderr)
    common_4 = set.intersection(*sets_4) if sets_4 else set()
    common_4_list = sorted(common_4)

    # ── 2-model 공통 이미지 집합 ─────────────────────────────────────────
    sets_2 = [set(model_data[m].keys()) for m in ["mnv2", "clip"] if model_data[m]]
    common_2 = set.intersection(*sets_2) if len(sets_2) == 2 else set()
    common_2_list = sorted(common_2)

    print(f"  공통 이미지: 4-model={len(common_4_list)}, 2-model={len(common_2_list)}")

    # ── 4-model score matrices ────────────────────────────────────────────
    vecs_4: Dict[str, np.ndarray] = {}
    for m in all_models:
        if model_data[m]:
            vecs_4[m] = build_score_matrix(common_4_list, model_data[m], m)

    # ── 2-model score matrices ────────────────────────────────────────────
    vecs_2: Dict[str, np.ndarray] = {}
    for m in ["mnv2", "clip"]:
        if model_data[m]:
            vecs_2[m] = build_score_matrix(common_2_list, model_data[m], m)

    # ── CKA 계산 ──────────────────────────────────────────────────────────
    cka_4 = compute_cka_matrix(vecs_4) if len(vecs_4) >= 2 else {}
    cka_2 = compute_cka_matrix(vecs_2) if len(vecs_2) == 2 else {}

    valid_cka_4 = [v for v in cka_4.values() if not np.isnan(v)]
    valid_cka_2 = [v for v in cka_2.values() if not np.isnan(v)]
    mean_cka_4 = float(np.mean(valid_cka_4)) if valid_cka_4 else float("nan")
    mean_cka_2 = float(np.mean(valid_cka_2)) if valid_cka_2 else float("nan")

    # ── Error Jaccard + 불일치율 ──────────────────────────────────────────
    jaccard_4, disagree_4 = compute_error_jaccard(
        common_4_list, model_data, all_models
    )
    jaccard_2, disagree_2 = compute_error_jaccard(
        common_2_list, model_data, ["mnv2", "clip"]
    )

    valid_j4 = [v for v in jaccard_4.values() if not np.isnan(v)]
    valid_j2 = [v for v in jaccard_2.values() if not np.isnan(v)]
    mean_j4 = float(np.mean(valid_j4)) if valid_j4 else float("nan")
    mean_j2 = float(np.mean(valid_j2)) if valid_j2 else float("nan")

    valid_d4 = list(disagree_4.values())
    valid_d2 = list(disagree_2.values())
    mean_d4 = float(np.mean(valid_d4)) if valid_d4 else float("nan")
    mean_d2 = float(np.mean(valid_d2)) if valid_d2 else float("nan")

    # ── 콘솔 출력 ─────────────────────────────────────────────────────────
    print(f"\n  [CKA — pairwise]")
    for k, v in cka_4.items():
        print(f"    4-model  {k:<30s}: {v:.4f}")
    for k, v in cka_2.items():
        print(f"    2-model  {k:<30s}: {v:.4f}")

    print(f"\n  [CKA 요약]")
    print(f"    4-model 평균 pairwise CKA : {mean_cka_4:.4f}")
    print(f"    2-model 평균 pairwise CKA : {mean_cka_2:.4f}")
    delta_cka = mean_cka_4 - mean_cka_2 if not (np.isnan(mean_cka_4) or np.isnan(mean_cka_2)) else float("nan")
    print(f"    ΔCKA (4 − 2)             : {delta_cka:+.4f}  "
          f"({'다양성 감소(낮을수록 다양)' if delta_cka > 0 else '다양성 증가(낮을수록 다양)'})")

    print(f"\n  [Error Jaccard — pairwise]")
    for k, v in jaccard_4.items():
        d = disagree_4.get(k, float("nan"))
        print(f"    4-model  {k:<30s}: Jaccard={v:.4f}  Disagree={d:.4f}")
    for k, v in jaccard_2.items():
        d = disagree_2.get(k, float("nan"))
        print(f"    2-model  {k:<30s}: Jaccard={v:.4f}  Disagree={d:.4f}")

    print(f"\n  [Jaccard / 불일치율 요약]")
    print(f"    4-model 평균 Error Jaccard: {mean_j4:.4f}  (낮을수록 에러 독립적)")
    print(f"    2-model 평균 Error Jaccard: {mean_j2:.4f}")
    print(f"    4-model 평균 불일치율     : {mean_d4:.4f}")
    print(f"    2-model 평균 불일치율     : {mean_d2:.4f}")

    return {
        "dataset": ds,
        "n_images_4model": len(common_4_list),
        "n_images_2model": len(common_2_list),
        "cka_pairwise_4model": cka_4,
        "cka_pairwise_2model": cka_2,
        "mean_cka_4model": mean_cka_4,
        "mean_cka_2model": mean_cka_2,
        "delta_cka": delta_cka,
        "jaccard_pairwise_4model": jaccard_4,
        "jaccard_pairwise_2model": jaccard_2,
        "mean_jaccard_4model": mean_j4,
        "mean_jaccard_2model": mean_j2,
        "disagree_pairwise_4model": disagree_4,
        "disagree_pairwise_2model": disagree_2,
        "mean_disagree_4model": mean_d4,
        "mean_disagree_2model": mean_d2,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
#  집계 요약                                                                  #
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #

def aggregate_summary(results: List[dict]) -> dict:
    """전체 데이터셋에 걸친 매크로 평균 요약."""
    def _macro(key: str) -> float:
        vals = [r[key] for r in results if not np.isnan(r.get(key, float("nan")))]
        return float(np.mean(vals)) if vals else float("nan")

    summary = {
        "macro_mean_cka_4model":     _macro("mean_cka_4model"),
        "macro_mean_cka_2model":     _macro("mean_cka_2model"),
        "macro_delta_cka":           _macro("delta_cka"),
        "macro_mean_jaccard_4model": _macro("mean_jaccard_4model"),
        "macro_mean_jaccard_2model": _macro("mean_jaccard_2model"),
        "macro_mean_disagree_4model":_macro("mean_disagree_4model"),
        "macro_mean_disagree_2model":_macro("mean_disagree_2model"),
    }

    print(f"\n{'='*60}")
    print("  전체 데이터셋 매크로 요약")
    print(f"{'='*60}")
    print(f"  4-model 평균 CKA    : {summary['macro_mean_cka_4model']:.4f}")
    print(f"  2-model 평균 CKA    : {summary['macro_mean_cka_2model']:.4f}")
    print(f"  ΔCKA (4−2)         : {summary['macro_delta_cka']:+.4f}  "
          f"(음수 → 4-model이 더 다양)")
    print(f"  4-model 평균 Jaccard: {summary['macro_mean_jaccard_4model']:.4f}")
    print(f"  2-model 평균 Jaccard: {summary['macro_mean_jaccard_2model']:.4f}")
    print(f"  4-model 평균 불일치 : {summary['macro_mean_disagree_4model']:.4f}")
    print(f"  2-model 평균 불일치 : {summary['macro_mean_disagree_2model']:.4f}")

    diversity_improved = (
        summary["macro_delta_cka"] < 0
        if not np.isnan(summary["macro_delta_cka"])
        else None
    )
    summary["diversity_improved_by_4model"] = diversity_improved
    verdict = "YES — 4-model이 더 다양한 예측 공간 보유" if diversity_improved else \
              ("NO  — CKA 기준 다양성 개선 없음" if diversity_improved is False else "N/A")
    print(f"\n  [결론] 다양성 개선 여부: {verdict}")

    return summary


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
#  메인                                                                       #
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #

def main() -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("CKA 다양성 분석: 4-model vs 2-model")
    print(f"ROOT      : {ROOT}")
    print(f"Datasets  : {DATASETS}")
    print(f"Output dir: {OUT_DIR}")

    ds_results = []
    for ds in DATASETS:
        res = analyze_dataset(ds)
        ds_results.append(res)

    summary = aggregate_summary(ds_results)

    # ── JSON 저장 ──────────────────────────────────────────────────────────
    output = {
        "timestamp": ts,
        "models": {
            "mnv2":  f"mobilenetv2_dualstream_{MNV2_TS}",
            "clip":  f"mobileclip_s2_finetuned_{CLIP_TS}",
            "specm": "specialist_m_v2 (latest glob)",
            "specg": "specialist_g (latest glob)",
        },
        "datasets": DATASETS,
        "per_dataset": ds_results,
        "summary": summary,
    }

    out_path = OUT_DIR / f"cka_diversity_4model_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n결과 저장: {out_path}")


if __name__ == "__main__":
    main()
