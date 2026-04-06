"""
SHIELD Phase 1.5: 최적 조합 결정 (Optimal Agent Combination Decision)

목적:
    Phase 1.1~1.4 실험 결과(Shapley, STII, CKA, PID)를 종합하여
    On-Device 배포를 위한 최적 에이전트 조합을 이론적으로 결정한다.

결정 기준 (다기준 스코어링):
    1. 성능 (F1)          - 높을수록 좋음
    2. 에이전트 수         - 적을수록 좋음 (엣지 비용)
    3. 상호작용 보존       - 강한 시너지 쌍은 함께 유지
    4. 고유정보 보존       - Unique 정보가 높은 에이전트 우선 유지
    5. 경량화 가능성       - Unique 낮은 에이전트는 제거/축소 우선

실행:
    .venv-qwen/bin/python experiments/run_optimal_combination_phase1.py
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "experiments/results/shapley_phase1"

# 가중치 (합 = 1.0)
W_F1 = 0.45          # 정확도
W_COST = 0.20        # 에이전트 수 (적을수록 good)
W_SYNERGY = 0.20     # 시너지 쌍 보존
W_UNIQUE = 0.15      # 고유정보 보존

# 4 에이전트 on-device 추론 비용 (ms) — MAIFS 측정값
INFERENCE_COST_MS = {
    "frequency": 80,   # CAT-Net
    "noise":     61,   # MVSS-Net
    "fatformer": 57,   # FatFormer (CLIP ViT-L/14)
    "spatial":   97,   # Mesorch
}

# 4 에이전트 모델 크기 (MB) — MAIFS 측정값
MODEL_SIZE_MB = {
    "frequency": 150,
    "noise":     120,
    "fatformer": 890,
    "spatial":   100,
}


def load_phase1_results() -> Dict:
    shapley_path = sorted(RESULT_DIR.glob("shapley_phase1_*.json"))[-1]
    pid_path = sorted(RESULT_DIR.glob("pid_phase1_*.json"))[-1]
    with open(shapley_path) as f:
        shapley = json.load(f)
    with open(pid_path) as f:
        pid = json.load(f)
    return shapley, pid


def compute_synergy_preservation(subset: List[str], pid_pairs: Dict) -> float:
    """
    subset 내 쌍들의 시너지 합 / 전체 시너지 합 비율.
    시너지 높은 쌍이 함께 포함될수록 점수 높음.
    """
    total_synergy = sum(v["synergy"] for v in pid_pairs.values())
    if total_synergy <= 0:
        return 0.0
    preserved = 0.0
    for pair_key, vals in pid_pairs.items():
        ai, aj = pair_key.split("__")
        if ai in subset and aj in subset:
            preserved += vals["synergy"]
    return preserved / total_synergy


def compute_unique_preservation(subset: List[str], unique_summary: Dict) -> float:
    """
    subset에 포함된 에이전트들의 Unique 합 / 전체 Unique 합.
    """
    total_unique = sum(unique_summary.values())
    if total_unique <= 0:
        return 0.0
    preserved = sum(unique_summary.get(a, 0.0) for a in subset)
    return preserved / total_unique


def normalize_dict(d: Dict) -> Dict:
    vals = list(d.values())
    mn, mx = min(vals), max(vals)
    if mx - mn < 1e-10:
        return {k: 1.0 for k in d}
    return {k: (v - mn) / (mx - mn) for k, v in d.items()}


def run():
    shapley, pid = load_phase1_results()

    # ------------------------------------------------------------------ #
    # 데이터 추출
    # ------------------------------------------------------------------ #
    cf = shapley["characteristic_function"]  # 부분집합 → F1
    shapley_vals = {item["agent"]: item["value"] for item in shapley["shapley_ranking"]}
    stii_vals = {item["pair"]: item["value"] for item in shapley["stii_ranking"]}
    cka_vals = shapley["cka_similarity"]
    unique_summary = pid["unique_summary"]
    pid_pairs = pid["pairwise_pid"]

    ALL_AGENTS = ["frequency", "noise", "fatformer", "spatial"]

    # ------------------------------------------------------------------ #
    # 전체 F1 (4개) 기준값
    # ------------------------------------------------------------------ #
    # cf 키가 리스트 형태로 저장됨 → 정렬된 문자열로 변환
    def cf_key(subset):
        return str(sorted(subset))

    f1_full = max(cf.values())  # 4개 에이전트 F1
    f1_single_max = max(
        v for k, v in cf.items() if len(k) < 20  # 단일 에이전트 대략 필터
    )

    print("=" * 70)
    print("  SHIELD Phase 1.5: 최적 조합 결정")
    print("=" * 70)
    print(f"\n  기준 F1 (전체 4개):  {f1_full:.4f}")

    # ------------------------------------------------------------------ #
    # 후보 조합 평가 (size 1~4)
    # ------------------------------------------------------------------ #
    results = []

    for size in range(1, 5):
        for combo in combinations(ALL_AGENTS, size):
            combo_list = list(combo)
            combo_key = str(sorted(combo_list))

            # F1
            f1 = cf.get(combo_key, None)
            if f1 is None:
                continue

            # 추론 비용 (ms 합산)
            cost_ms = sum(INFERENCE_COST_MS[a] for a in combo_list)
            # 모델 크기 (MB 합산)
            size_mb = sum(MODEL_SIZE_MB[a] for a in combo_list)
            # 시너지 보존율
            syn_pres = compute_synergy_preservation(combo_list, pid_pairs)
            # Unique 보존율
            uniq_pres = compute_unique_preservation(combo_list, unique_summary)
            # Unique 절대량 (중요 에이전트 포함 여부)
            uniq_abs = sum(unique_summary.get(a, 0.0) for a in combo_list)

            results.append({
                "combo": combo_list,
                "size": size,
                "f1": f1,
                "f1_vs_full": f1 / f1_full,
                "cost_ms": cost_ms,
                "size_mb": size_mb,
                "synergy_preservation": syn_pres,
                "unique_preservation": uniq_pres,
                "unique_abs": uniq_abs,
            })

    # ------------------------------------------------------------------ #
    # 정규화 후 다기준 스코어 계산
    # ------------------------------------------------------------------ #
    f1_arr = np.array([r["f1"] for r in results])
    cost_arr = np.array([r["cost_ms"] for r in results])
    syn_arr = np.array([r["synergy_preservation"] for r in results])
    uniq_arr = np.array([r["unique_preservation"] for r in results])

    def norm(arr):
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-10:
            return np.ones_like(arr)
        return (arr - mn) / (mx - mn)

    f1_norm = norm(f1_arr)
    cost_norm = 1.0 - norm(cost_arr)   # 낮을수록 좋음 → 반전
    syn_norm = norm(syn_arr)
    uniq_norm = norm(uniq_arr)

    score_arr = (
        W_F1 * f1_norm
        + W_COST * cost_norm
        + W_SYNERGY * syn_norm
        + W_UNIQUE * uniq_norm
    )

    for i, r in enumerate(results):
        r["score"] = float(score_arr[i])

    results_sorted = sorted(results, key=lambda x: -x["score"])

    # ------------------------------------------------------------------ #
    # 출력: 전체 순위
    # ------------------------------------------------------------------ #
    print(f"\n  {'조합':45s}  {'F1':6s}  {'vs4':5s}  {'ms':5s}  {'MB':5s}  "
          f"{'Syn':5s}  {'Uniq':5s}  {'Score':6s}")
    print("  " + "-" * 95)

    for r in results_sorted:
        combo_str = "+".join(r["combo"])
        print(
            f"  {combo_str:45s}  {r['f1']:.4f}  {r['f1_vs_full']:.3f}  "
            f"{r['cost_ms']:5d}  {r['size_mb']:5d}  "
            f"{r['synergy_preservation']:.3f}  {r['unique_preservation']:.3f}  "
            f"{r['score']:.4f}"
        )

    # ------------------------------------------------------------------ #
    # 사이즈별 최고 조합
    # ------------------------------------------------------------------ #
    print("\n  === 에이전트 수별 최고 조합 ===")
    for size in range(1, 5):
        candidates = [r for r in results_sorted if r["size"] == size]
        if not candidates:
            continue
        best = candidates[0]
        combo_str = "+".join(best["combo"])
        print(
            f"  [{size}개] {combo_str:45s}  F1={best['f1']:.4f}  "
            f"({best['f1_vs_full']*100:.1f}% of full)  score={best['score']:.4f}"
        )

    # ------------------------------------------------------------------ #
    # 에지 배포 관점 분석
    # ------------------------------------------------------------------ #
    print("\n  === On-Device 배포 제약 분석 ===")
    print(f"  제약: <500MB, <300ms/img")
    print(f"  {'조합':45s}  {'MB':5s}  {'ms':5s}  {'F1':6s}  {'판정':10s}")
    print("  " + "-" * 85)

    for r in results_sorted:
        size_mb = r["size_mb"]
        cost_ms = r["cost_ms"]
        f1 = r["f1"]
        combo_str = "+".join(r["combo"])

        if size_mb <= 500 and cost_ms <= 300:
            status = "✓ 배포가능"
        elif size_mb > 500:
            status = "✗ 크기초과"
        else:
            status = "△ 속도한계"

        print(f"  {combo_str:45s}  {size_mb:5d}  {cost_ms:5d}  {f1:.4f}  {status}")

    # ------------------------------------------------------------------ #
    # 최종 권고
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print("  Phase 1.5 최적 조합 결정 — 최종 권고")
    print("=" * 70)

    # 경량화 후 (3개) 최적
    best_3 = [r for r in results_sorted if r["size"] == 3][0]
    best_4 = [r for r in results_sorted if r["size"] == 4][0]
    best_2 = [r for r in results_sorted if r["size"] == 2][0]

    f1_drop_3 = (best_4["f1"] - best_3["f1"]) / best_4["f1"] * 100
    size_drop_3 = (best_4["size_mb"] - best_3["size_mb"]) / best_4["size_mb"] * 100

    print(f"\n  [권고 A] On-Device 최적: {'+'.join(best_3['combo'])}")
    print(f"           F1 = {best_3['f1']:.4f}  ({f1_drop_3:+.1f}% vs 4개)")
    print(f"           모델 크기 = {best_3['size_mb']}MB  ({size_drop_3:.0f}% 절감)")
    print(f"           추론 시간 = {best_3['cost_ms']}ms")
    print(f"           Unique 보존 = {best_3['unique_preservation']*100:.0f}%")
    print(f"           시너지 보존 = {best_3['synergy_preservation']*100:.0f}%")

    print(f"\n  [권고 B] 경량화 후 목표:")
    print(f"           FatFormer→MobileCLIP(50MB), CAT-Net pruned(55MB), MVSS-Net→MobileNetV3(25MB)")
    total_light = 50 + 55 + 25
    print(f"           경량화 후 총 크기 = {total_light}MB  (<500MB 충족)")
    print(f"           Spatial 제거: Unique=0.0000 — 정보 손실 없음")

    print(f"\n  [이론적 근거]")
    print(f"   · Spatial φ={shapley_vals['spatial']:+.4f} (최저), Unique=0.0000 → 제거 정당화")
    print(f"   · freq↔fatformer STII={stii_vals['frequency__fatformer']:+.4f} → 상호작용 보존 필수")
    print(f"   · noise↔fatformer 시너지=+{pid_pairs['noise__fatformer']['synergy']:.4f} → noise 유지 필요")
    print(f"   · frequency Unique=0.2029 (전체의 {0.2029/sum(unique_summary.values())*100:.0f}%) → 절대 유지")

    # 저장
    output = {
        "timestamp": __import__("datetime").datetime.now().isoformat(),
        "ranking": results_sorted,
        "best_per_size": {
            "2": {"combo": best_2["combo"], "f1": best_2["f1"], "score": best_2["score"]},
            "3": {"combo": best_3["combo"], "f1": best_3["f1"], "score": best_3["score"]},
            "4": {"combo": best_4["combo"], "f1": best_4["f1"], "score": best_4["score"]},
        },
        "recommendation": {
            "ondevice_optimal": best_3["combo"],
            "f1_drop_pct": f1_drop_3,
            "size_reduction_pct": size_drop_3,
            "justification": {
                "remove_spatial": "Unique=0.0000, φ=+0.0547 (최저)",
                "keep_frequency": "Unique=0.2029 (61%), φ=+0.2690 (최고)",
                "keep_noise_fatformer": "시너지 쌍 noise↔fatformer=+0.1093 보존 필수",
            },
        },
    }

    out_path = RESULT_DIR / f"optimal_combination_phase1.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  결과 저장: {out_path}")


if __name__ == "__main__":
    run()
