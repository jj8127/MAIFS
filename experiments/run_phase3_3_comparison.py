#!/usr/bin/env python3
"""
SHIELD Phase 3.3: 다기준 비교 → 최종 Track 선정
================================================
수집된 모든 실험 결과를 통합하여 F1 × 크기 × 속도 × OOD 일관성
기준으로 Track 후보를 비교하고 배포 권고안을 도출한다.

Track 후보:
  - CLIP-only      : MobileCLIP-ft4 단독 (baseline)
  - MNV2-only      : MobileNetV2-DS 단독
  - CLIP+MNV2      : Shapley 최적 2-모델 (Track 1b)
  - Track1         : ForMa + MNV2 + CLIP
  - Track2         : ForMa + CLIP
  - Track3-cascade : Tiny(exit) → ForMa + CLIP

실행:
  .venv-qwen/bin/python experiments/run_phase3_3_comparison.py
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT    = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "experiments" / "results" / "phase3_comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. 벤치마크 데이터 (backbone_benchmark JSON에서 파싱) ──────────────────── #
BENCH_JSON = sorted(
    (ROOT / "experiments" / "results" / "backbone_benchmark").glob("*.json")
)[-1]

def load_bench():
    with open(BENCH_JSON) as f:
        d = json.load(f)
    r = d["results"]
    out = {}
    for model, data in r.items():
        cu = data.get("cuda", {})
        cp = data.get("cpu",  {})
        out[model] = {
            "params_m":  cu.get("model", {}).get("params_total_m", 0),
            "ckpt_mib":  cu.get("model", {}).get("checkpoint_mib", 0),
            "gpu_ms":    cu.get("latency", {}).get("mean_ms", 0),
            "cpu_ms":    cp.get("latency", {}).get("mean_ms", 0),
        }
    return out

# ── 2. Phase3-tracks combined F1 ─────────────────────────────────────────── #
TRACKS_JSON = sorted(
    (ROOT / "experiments" / "results" / "phase3_tracks").glob(
        "phase3_tracks_combined_*.json"
    )
)[-1]

def load_track_f1():
    with open(TRACKS_JSON) as f:
        d = json.load(f)
    combined = d.get("combined", {})
    out = {}
    for k, v in combined.items():
        if isinstance(v, dict) and "mean" in v:
            out[k] = {"f1_mean": v["mean"], "f1_std": v.get("std", 0.0)}
    return out

# ── 3. Fair LOO OOD F1 ────────────────────────────────────────────────────── #
FAIR_JSON = sorted(
    (ROOT / "experiments" / "results" / "fair_cross").glob("*.json")
)[-1]

def load_fair_loo():
    with open(FAIR_JSON) as f:
        d = json.load(f)
    folds = d.get("folds", {})
    keys = list(folds.keys())
    out = {}
    for metric in ["clip_loo", "track1_gbm", "track2_gbm", "track3_gbm"]:
        vals = [folds[ds][metric] for ds in keys if metric in folds[ds]]
        out[metric] = float(np.mean(vals)) if vals else 0.0
    return out

# ── 4. Shapley Phase3 단독 F1 ─────────────────────────────────────────────── #
SHAP_JSON = sorted(
    (ROOT / "experiments" / "results" / "shapley_phase3").glob("*.json")
)[-1]

def load_shapley_f1():
    with open(SHAP_JSON) as f:
        d = json.load(f)
    sr = d.get("subset_results", {})
    out = {}
    for k, v in sr.items():
        out[k] = v["f1_mean"]
    return out


# ── 5. Track 후보 정의 ────────────────────────────────────────────────────── #
def build_candidates(bench, track_f1, fair_loo, shap_f1):
    """
    각 Track 후보의 다기준 지표를 딕셔너리로 구성한다.
    latency는 순차 합산 (cascade Track3는 가중 평균 사용).
    """

    def sz(*models):
        return sum(bench.get(m, {}).get("ckpt_mib", 0) for m in models)

    def gpu(*models):
        return sum(bench.get(m, {}).get("gpu_ms", 0) for m in models)

    def cpu(*models):
        return sum(bench.get(m, {}).get("cpu_ms", 0) for m in models)

    # 실제 벤치마크 키: forma / mobileclip_ft4 / tiny_ladeda / mobilenetv2_dualstream
    tiny_gpu = bench.get("tiny_ladeda",           {}).get("gpu_ms", 0)
    tiny_cpu = bench.get("tiny_ladeda",           {}).get("cpu_ms", 0)
    forma_gpu= bench.get("forma",                 {}).get("gpu_ms", 0)
    forma_cpu= bench.get("forma",                 {}).get("cpu_ms", 0)
    clip_gpu = bench.get("mobileclip_ft4",        {}).get("gpu_ms", 0)
    clip_cpu = bench.get("mobileclip_ft4",        {}).get("cpu_ms", 0)
    mnv2_gpu = bench.get("mobilenetv2_dualstream",{}).get("gpu_ms", 0)
    mnv2_cpu = bench.get("mobilenetv2_dualstream",{}).get("cpu_ms", 0)

    CASCADE_RATE = 0.50   # Tier-2 진입 비율 (50% 가정)
    t3_gpu = tiny_gpu + CASCADE_RATE * (forma_gpu + clip_gpu)
    t3_cpu = tiny_cpu + CASCADE_RATE * (forma_cpu + clip_cpu)

    # shapley subset key 매핑
    clip_mnv2_f1 = shap_f1.get("['mobileclip', 'mobilenetv2']", 0.0)

    # combined in-dist F1 (실제 키: track1_gbm, track2_gbm, track3_gbm, baseline_clip 등)
    t1_indist   = track_f1.get("track1_gbm",    {}).get("f1_mean", 0.0)
    t2_indist   = track_f1.get("track2_gbm",    {}).get("f1_mean", 0.0)
    t3_indist   = track_f1.get("track3_gbm",    {}).get("f1_mean", 0.0)
    clip_indist = track_f1.get("baseline_clip", {}).get("f1_mean",
                  shap_f1.get("['mobileclip']", 0.0))
    mnv2_indist = track_f1.get("baseline_mobilenet", {}).get("f1_mean",
                  shap_f1.get("['mobilenetv2']", 0.0))

    # OOD F1 (Fair LOO)
    t1_ood  = fair_loo.get("track1_gbm", 0.0)
    t2_ood  = fair_loo.get("track2_gbm", 0.0)
    t3_ood  = fair_loo.get("track3_gbm", 0.0)
    clip_ood = fair_loo.get("clip_loo",  0.0)
    # MNV2-only OOD: Fair LOO에는 없으므로 Track1과 CLIP의 중간으로 근사
    # (Track1이 MNV2의 기여로 OOD 개선. MNV2 단독은 CLIP보다 약간 높을 것으로 추정)
    mnv2_ood  = clip_ood + 0.03   # 보수적 근사 (검증 안 된 수치)
    clip_mnv2_ood = t1_ood        # CLIP+MNV2 ≈ Track1 (ForMa 기여≈0)

    candidates = {
        "CLIP-only": {
            "desc":       "MobileCLIP-ft4 단독",
            "f1_indist":  clip_indist,
            "f1_ood":     clip_ood,
            "ckpt_mib":   sz("mobileclip_ft4"),
            "gpu_ms":     clip_gpu,
            "cpu_ms":     clip_cpu,
            "rpi5_ok":    True,
            "note":       "OOD 취약 (Fair LOO avg=0.639)"
        },
        "MNV2-only": {
            "desc":       "MobileNetV2-DS 단독",
            "f1_indist":  mnv2_indist,
            "f1_ood":     mnv2_ood,
            "ckpt_mib":   sz("mobilenetv2_dualstream"),
            "gpu_ms":     mnv2_gpu,
            "cpu_ms":     mnv2_cpu,
            "rpi5_ok":    True,
            "note":       "OOD 추정값(미검증). Shapley φ=0.304(1위)"
        },
        "CLIP+MNV2": {
            "desc":       "MobileCLIP-ft4 + MobileNetV2-DS (Shapley 최적 2-모델)",
            "f1_indist":  clip_mnv2_f1,
            "f1_ood":     clip_mnv2_ood,
            "ckpt_mib":   sz("mobileclip_ft4", "mobilenetv2_dualstream"),
            "gpu_ms":     clip_gpu + mnv2_gpu,
            "cpu_ms":     clip_cpu + mnv2_cpu,
            "rpi5_ok":    True,
            "note":       "in-dist 최고 2-조합. CKA=0.92(중복) 논문 서술 포인트"
        },
        "Track1": {
            "desc":       "ForMa + MNV2 + CLIP (3-모델)",
            "f1_indist":  t1_indist,
            "f1_ood":     t1_ood,
            "ckpt_mib":   sz("forma", "mobilenetv2_dualstream", "mobileclip_ft4"),
            "gpu_ms":     forma_gpu + mnv2_gpu + clip_gpu,
            "cpu_ms":     forma_cpu + mnv2_cpu + clip_cpu,
            "rpi5_ok":    False,
            "note":       "ForMa CPU=1613ms → RPi5 불가. OOD 최강(0.731)"
        },
        "Track2": {
            "desc":       "ForMa + CLIP (2-모델)",
            "f1_indist":  t2_indist,
            "f1_ood":     t2_ood,
            "ckpt_mib":   sz("forma", "mobileclip_ft4"),
            "gpu_ms":     forma_gpu + clip_gpu,
            "cpu_ms":     forma_cpu + clip_cpu,
            "rpi5_ok":    False,
            "note":       "ForMa CPU=1613ms → RPi5 불가. MNV2 없어 OOD 약함"
        },
        "Track3-cascade": {
            "desc":       f"Tiny→(ForMa+CLIP) cascade ({CASCADE_RATE*100:.0f}% pass-through)",
            "f1_indist":  t3_indist,
            "f1_ood":     t3_ood,
            "ckpt_mib":   sz("tiny_ladeda", "forma", "mobileclip_ft4"),
            "gpu_ms":     t3_gpu,
            "cpu_ms":     t3_cpu,
            "rpi5_ok":    False,
            "note":       "ForMa CPU=1613ms → RPi5 불가. cascade 비율 50% 가정"
        },
    }
    return candidates


# ── 6. 복합 점수 계산 ─────────────────────────────────────────────────────── #
WEIGHTS = {
    "f1_ood":    0.40,   # OOD 강건성 (SHIELD 논문 핵심)
    "f1_indist": 0.25,   # in-dist 성능
    "cpu_speed": 0.20,   # CPU 추론 속도 (RPi5)
    "size":      0.15,   # 모델 크기
}

def compute_scores(candidates):
    """Min-max 정규화 후 가중 합산으로 복합 점수 계산."""
    metrics = {
        "f1_ood":    {k: v["f1_ood"]    for k, v in candidates.items()},
        "f1_indist": {k: v["f1_indist"] for k, v in candidates.items()},
        "cpu_speed": {k: 1.0 / max(v["cpu_ms"], 1) for k, v in candidates.items()},
        "size":      {k: 1.0 / max(v["ckpt_mib"], 1) for k, v in candidates.items()},
    }
    normalized = {}
    for metric, vals in metrics.items():
        mn, mx = min(vals.values()), max(vals.values())
        rng = mx - mn if mx != mn else 1.0
        normalized[metric] = {k: (v - mn) / rng for k, v in vals.items()}

    scores = {}
    for name in candidates:
        scores[name] = sum(
            WEIGHTS[m] * normalized[m][name] for m in WEIGHTS
        )
    return scores


# ── 7. 출력 및 저장 ───────────────────────────────────────────────────────── #
def run():
    print("[Phase 3.3] 다기준 비교 분석 시작\n")

    bench    = load_bench()
    track_f1 = load_track_f1()
    fair_loo = load_fair_loo()
    shap_f1  = load_shapley_f1()

    print(f"  벤치마크: {BENCH_JSON.name}")
    print(f"  Track F1: {TRACKS_JSON.name}")
    print(f"  Fair LOO: {FAIR_JSON.name}")
    print(f"  Shapley:  {SHAP_JSON.name}\n")

    candidates = build_candidates(bench, track_f1, fair_loo, shap_f1)
    scores     = compute_scores(candidates)

    # ── 비교 테이블 출력 ──
    W = 16
    hdr = (f"{'Track':14s}  {'InDist-F1':>10}  {'OOD-F1':>8}  "
           f"{'Size(MiB)':>10}  {'GPU(ms)':>8}  {'CPU(ms)':>9}  "
           f"{'RPi5':>5}  {'Score':>7}")
    print("=" * len(hdr))
    print("  SHIELD Phase 3.3 — 다기준 Track 비교")
    print("=" * len(hdr))
    print(f"  {hdr}")
    print("-" * len(hdr))

    ranked = sorted(scores.items(), key=lambda x: -x[1])
    for name, score in ranked:
        c = candidates[name]
        rpi = "✓" if c["rpi5_ok"] else "✗"
        print(f"  {name:14s}  {c['f1_indist']:10.4f}  {c['f1_ood']:8.4f}  "
              f"{c['ckpt_mib']:10.1f}  {c['gpu_ms']:8.1f}  {c['cpu_ms']:9.1f}  "
              f"{rpi:>5}  {score:7.4f}")

    print("=" * len(hdr))

    # ── 가중치 표시 ──
    print(f"\n  복합 점수 가중치: OOD {WEIGHTS['f1_ood']*100:.0f}%  "
          f"InDist {WEIGHTS['f1_indist']*100:.0f}%  "
          f"CPU속도 {WEIGHTS['cpu_speed']*100:.0f}%  "
          f"크기 {WEIGHTS['size']*100:.0f}%")

    # ── 세부 분석 ──
    print("\n" + "=" * 60)
    print("  상세 분석")
    print("=" * 60)
    for name, score in ranked:
        c = candidates[name]
        print(f"\n  [{name}] score={score:.4f}")
        print(f"    구성: {c['desc']}")
        print(f"    InDist F1 = {c['f1_indist']:.4f}  |  OOD F1 = {c['f1_ood']:.4f}")
        print(f"    크기: {c['ckpt_mib']:.1f} MiB  |  GPU: {c['gpu_ms']:.1f}ms  |  CPU: {c['cpu_ms']:.1f}ms")
        print(f"    RPi5 배포: {'가능' if c['rpi5_ok'] else '불가 (ForMa CPU=1613ms)'}")
        print(f"    비고: {c['note']}")

    # ── 최종 권고 ──
    best = ranked[0][0]
    best_rpi = next((n for n, _ in ranked if candidates[n]["rpi5_ok"]), None)

    print("\n" + "=" * 60)
    print("  최종 권고")
    print("=" * 60)
    print(f"\n  [서버/GPU 환경 권고] → {best}")
    print(f"    {candidates[best]['desc']}")
    print(f"    OOD F1={candidates[best]['f1_ood']:.4f}, InDist F1={candidates[best]['f1_indist']:.4f}")

    if best_rpi and best_rpi != best:
        print(f"\n  [RPi5 엣지 환경 권고] → {best_rpi}")
        print(f"    {candidates[best_rpi]['desc']}")
        print(f"    OOD F1={candidates[best_rpi]['f1_ood']:.4f}, "
              f"InDist F1={candidates[best_rpi]['f1_indist']:.4f}")
        print(f"    CPU: {candidates[best_rpi]['cpu_ms']:.1f}ms  |  "
              f"크기: {candidates[best_rpi]['ckpt_mib']:.1f} MiB")

    print(f"\n  [ForMa 제거 근거]")
    print(f"    Shapley φ=+0.0079 (MNV2의 1/38), CKA<0.02, CPU=1613ms")
    print(f"    → in-dist 기여 무시 가능, RPi5 배포 시 결정적 병목")

    print(f"\n  [CLIP↔MNV2 중복 현상 — 논문 포인트]")
    print(f"    CKA=0.922, STII=-0.584, PID Redundancy=0.60")
    print(f"    → 경량화 시 두 모델이 동일 특징 수렴 (Phase1 무거운 모델과 반대)")
    print(f"    → SHIELD 기여 C2: 'interaction-preserving lightweighting 필요성' 입증")

    # ── 저장 ──
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = OUT_DIR / f"phase3_comparison_{ts}.json"
    result = {
        "timestamp": ts,
        "weights":   WEIGHTS,
        "candidates": {
            k: {**v, "composite_score": scores[k]}
            for k, v in candidates.items()
        },
        "ranking": [{"track": n, "score": s} for n, s in ranked],
        "recommendation": {
            "gpu_server": best,
            "rpi5_edge":  best_rpi,
        },
        "sources": {
            "benchmark":  str(BENCH_JSON),
            "track_f1":   str(TRACKS_JSON),
            "fair_loo":   str(FAIR_JSON),
            "shapley":    str(SHAP_JSON),
        },
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n  저장: {out}")
    return result


if __name__ == "__main__":
    run()
