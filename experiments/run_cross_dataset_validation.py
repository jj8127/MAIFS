"""
SHIELD Phase 1 Cross-Dataset Validation
Phase 1 결론(Shapley ranking, Spatial 제거, noise↔fatformer 시너지)이
다양한 데이터셋에서도 일관되게 유지되는지 검증한다.

검증 데이터셋 (기존 JSONL 재사용):
    1. base       : CASIA2 + BigGAN        (1500샘플) ← Phase 1 원본
    2. dsC        : CASIA2 + IMD2020 + BigGAN (900샘플) ← Inpainting 조작 추가
    3. opensdi    : OpenSDID               (900샘플)  ← 완전히 다른 출처
    4. aigenproxy : AI-GenBench proxy      (900샘플)  ← 다양한 AI 생성기

판단 기준:
    - Shapley ranking 일치율 (frequency 1위 유지 여부)
    - Spatial Unique=0 유지 여부 (제거 정당성)
    - noise↔fatformer 시너지 상위권 유지 여부
    → 4개 데이터셋 중 3개 이상 일치 시 결론 확정

실행:
    .venv-qwen/bin/python experiments/run_cross_dataset_validation.py
"""
from __future__ import annotations

import json
import sys
import warnings
from datetime import datetime
from itertools import combinations, permutations
from math import factorial
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.meta.simulator import AGENT_NAMES, VERDICTS, TRUE_LABELS

warnings.filterwarnings("ignore")

ALL_AGENTS: Tuple[str, ...] = tuple(AGENT_NAMES)
VERDICT_IDX = {v: i for i, v in enumerate(VERDICTS)}
LABEL_IDX = {l: i for i, l in enumerate(TRUE_LABELS)}
EPS = 1e-12

# ── 검증 데이터셋 정의 ──────────────────────────────────────────────────── #
DATASETS = {
    "base": {
        "jsonl": "experiments/results/phase2_patha_scale500_gain_predictor/patha_agent_outputs_20260304_080157.jsonl",
        "desc": "CASIA2 + BigGAN (1500, Phase 1 원본)",
    },
    "dsC": {
        "jsonl": "experiments/results/phase2_patha_case3_scale300_dsC/patha_agent_outputs_20260303_105005.jsonl",
        "desc": "CASIA2 + IMD2020(Inpainting) + BigGAN (900)",
    },
    "opensdi": {
        "jsonl": "experiments/results/phase2_patha_case3_opensdi_scale300/patha_agent_outputs_fixed_seed42.jsonl",
        "desc": "OpenSDID 전체 (900, 다른 출처)",
    },
    "aigenproxy": {
        "jsonl": "experiments/results/phase2_patha_case3_aigenproxy_scale300/patha_agent_outputs_fixed_seed42.jsonl",
        "desc": "AI-GenBench proxy (900, 다양한 AI 생성기)",
    },
}

GBM_PARAMS = {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.1,
               "subsample": 0.8, "min_samples_leaf": 5, "random_state": 42}
SEEDS = [300, 301, 302, 303, 304]   # 속도 위해 5 seeds
TRAIN_RATIO, VAL_RATIO = 0.6, 0.2
EMPTY_F1 = 1.0 / len(TRUE_LABELS)


# ── 데이터 로드 ──────────────────────────────────────────────────────────── #
def load_records(path: str) -> List[Dict]:
    recs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


# ── Feature 빌드 ─────────────────────────────────────────────────────────── #
def build_features(records: List[Dict], agents: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for rec in records:
        row = []
        for agent in agents:
            v = rec["agent_verdicts"].get(agent, "uncertain")
            one_hot = [0.0] * len(VERDICTS)
            if v in VERDICT_IDX:
                one_hot[VERDICT_IDX[v]] = 1.0
            row.extend(one_hot)
        X.append(row)
        y.append(LABEL_IDX.get(rec["true_label"], 0))
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


# ── 부분집합 평가 ────────────────────────────────────────────────────────── #
def evaluate_subset(records: List[Dict], agents: List[str], seeds: List[int]) -> float:
    if not agents:
        return EMPTY_F1
    X, y = build_features(records, agents)
    f1s = []
    for seed in seeds:
        X_tr, X_tmp, y_tr, y_tmp = train_test_split(
            X, y, test_size=1 - TRAIN_RATIO, random_state=seed, stratify=y)
        val_size = VAL_RATIO / (1 - TRAIN_RATIO)
        X_val, X_te, y_val, y_te = train_test_split(
            X_tmp, y_tmp, test_size=1 - val_size, random_state=seed, stratify=y_tmp)
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(**{**GBM_PARAMS, "random_state": seed})),
        ])
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_te)
        f1s.append(f1_score(y_te, pred, average="macro", zero_division=0))
    return float(np.mean(f1s))


# ── Shapley 계산 ─────────────────────────────────────────────────────────── #
def compute_shapley(cf: Dict[frozenset, float]) -> Dict[str, float]:
    n = len(ALL_AGENTS)
    shapley = {a: 0.0 for a in ALL_AGENTS}
    for agent in ALL_AGENTS:
        others = [a for a in ALL_AGENTS if a != agent]
        for size in range(n):
            for subset in combinations(others, size):
                s = frozenset(subset)
                s_with = frozenset(subset) | {agent}
                weight = factorial(size) * factorial(n - size - 1) / factorial(n)
                shapley[agent] += weight * (cf[s_with] - cf[s])
    return shapley


# ── STII k=2 ────────────────────────────────────────────────────────────── #
def compute_stii(cf: Dict[frozenset, float]) -> Dict[str, float]:
    n = len(ALL_AGENTS)
    stii = {}
    for ai, aj in combinations(ALL_AGENTS, 2):
        pair = frozenset({ai, aj})
        val = 0.0
        others = [a for a in ALL_AGENTS if a not in pair]
        for size in range(n - 1):
            for subset in combinations(others, size):
                s = frozenset(subset)
                w = factorial(size) * factorial(n - size - 2) / factorial(n - 1)
                delta = (cf[s | pair] - cf[s | {ai}] - cf[s | {aj}] + cf[s])
                val += w * delta
        stii[f"{ai}__{aj}"] = val
    return stii


# ── PID (Williams & Beer Imin) ──────────────────────────────────────────── #
def estimate_joint_pya(records: List[Dict], agent: str) -> np.ndarray:
    n_y, n_a = len(TRUE_LABELS), len(VERDICTS)
    counts = np.zeros((n_y, n_a))
    for rec in records:
        yi = LABEL_IDX.get(rec["true_label"], -1)
        ai = VERDICT_IDX.get(rec["agent_verdicts"].get(agent, "uncertain"), -1)
        if yi >= 0 and ai >= 0:
            counts[yi, ai] += 1
    return counts / max(counts.sum(), 1)


def mi_single(p_ya: np.ndarray) -> float:
    p_y = p_ya.sum(axis=1, keepdims=True)
    p_a = p_ya.sum(axis=0, keepdims=True)
    mask = p_ya > EPS
    return float(max(np.where(mask, p_ya * np.log(p_ya / (p_y * p_a + EPS) + EPS), 0).sum(), 0))


def specific_info(p_ya: np.ndarray) -> np.ndarray:
    p_y = p_ya.sum(axis=1)
    p_a = p_ya.sum(axis=0)
    p_y_given_a = p_ya.T / (p_a[:, None] + EPS)
    p_a_given_y = p_ya / (p_y[:, None] + EPS)
    i_spec = np.zeros(len(TRUE_LABELS))
    for yi in range(len(TRUE_LABELS)):
        if p_y[yi] < EPS:
            continue
        s = 0.0
        for ai in range(len(VERDICTS)):
            pac = p_a_given_y[yi, ai]
            pyc = p_y_given_a[ai, yi]
            if pac > EPS and pyc > EPS and p_y[yi] > EPS:
                s += pac * np.log(pyc / (p_y[yi] + EPS) + EPS)
        i_spec[yi] = s
    return i_spec


def pid_pair(records: List[Dict], ai: str, aj: str) -> Dict:
    p_i = estimate_joint_pya(records, ai)
    p_j = estimate_joint_pya(records, aj)
    p_y = p_i.sum(axis=1)
    mi_i = mi_single(p_i)
    mi_j = mi_single(p_j)
    # joint MI (via independence approximation for speed; exact via 3D tensor is slow)
    # 3D tensor approach
    n_y, n_a = len(TRUE_LABELS), len(VERDICTS)
    counts = np.zeros((n_y, n_a, n_a))
    for rec in records:
        yi = LABEL_IDX.get(rec["true_label"], -1)
        aii = VERDICT_IDX.get(rec["agent_verdicts"].get(ai, "uncertain"), -1)
        aji = VERDICT_IDX.get(rec["agent_verdicts"].get(aj, "uncertain"), -1)
        if yi >= 0 and aii >= 0 and aji >= 0:
            counts[yi, aii, aji] += 1
    p_yab = counts / max(counts.sum(), 1)
    p_y3 = p_yab.sum(axis=(1, 2), keepdims=True)
    p_ab = p_yab.sum(axis=0, keepdims=True)
    mask = p_yab > EPS
    mi_joint = float(max(np.where(mask, p_yab * np.log(p_yab / (p_y3 * p_ab + EPS) + EPS), 0).sum(), 0))
    # Imin redundancy
    is_i = specific_info(p_i)
    is_j = specific_info(p_j)
    red = float(max(np.sum(p_y * np.minimum(is_i, is_j)), 0))
    unique_i = max(mi_i - red, 0.0)
    unique_j = max(mi_j - red, 0.0)
    synergy = mi_joint - mi_i - mi_j + red
    synergy = max(synergy, -abs(mi_joint) * 0.01)
    return {"unique_i": unique_i, "unique_j": unique_j,
            "synergy": synergy, "redundancy": red,
            "mi_i": mi_i, "mi_j": mi_j, "mi_joint": mi_joint}


def compute_unique_summary(records: List[Dict]) -> Dict[str, float]:
    pair_results = {}
    for ai, aj in combinations(ALL_AGENTS, 2):
        pair_results[(ai, aj)] = pid_pair(records, ai, aj)
    unique_summary = {}
    for agent in ALL_AGENTS:
        uniques = []
        for (ai, aj), res in pair_results.items():
            if ai == agent:
                uniques.append(res["unique_i"])
            elif aj == agent:
                uniques.append(res["unique_j"])
        unique_summary[agent] = float(min(uniques)) if uniques else 0.0
    synergy_ranking = sorted(
        [{"pair": f"{ai}__{aj}", "synergy": res["synergy"]}
         for (ai, aj), res in pair_results.items()],
        key=lambda x: -x["synergy"]
    )
    return unique_summary, synergy_ranking, pair_results


# ── 단일 데이터셋 전체 Phase 1 실행 ────────────────────────────────────── #
def run_one_dataset(name: str, info: Dict) -> Dict:
    jsonl_path = ROOT / info["jsonl"]
    print(f"\n{'='*65}")
    print(f"  [{name}] {info['desc']}")
    print(f"{'='*65}")

    records = load_records(str(jsonl_path))
    print(f"  → {len(records)}개 레코드 로드")

    # ── Characteristic function (16 subsets) ──────────────────────────── #
    print("  [Shapley] 16개 부분집합 평가 중...", end="", flush=True)
    cf: Dict[frozenset, float] = {}
    for size in range(len(ALL_AGENTS) + 1):
        for subset in combinations(ALL_AGENTS, size):
            fs = frozenset(subset)
            cf[fs] = evaluate_subset(records, list(subset), SEEDS)
            print(".", end="", flush=True)
    print()

    # ── Shapley values ─────────────────────────────────────────────────── #
    shapley = compute_shapley(cf)
    shapley_ranking = sorted(shapley.items(), key=lambda x: -x[1])
    print("  Shapley ranking:", [(a, f"{v:+.4f}") for a, v in shapley_ranking])

    # ── STII ──────────────────────────────────────────────────────────── #
    stii = compute_stii(cf)
    stii_ranking = sorted(stii.items(), key=lambda x: -abs(x[1]))
    print("  STII top pair:", stii_ranking[0])

    # ── PID ───────────────────────────────────────────────────────────── #
    print("  [PID] 계산 중...", end="", flush=True)
    unique_summary, synergy_ranking, pid_pairs = compute_unique_summary(records)
    print(" 완료")
    print("  Unique:", {a: f"{v:.4f}" for a, v in sorted(unique_summary.items(), key=lambda x: -x[1])})
    print("  Synergy top:", synergy_ranking[0])

    # ── cf를 직렬화 가능한 형태로 변환 ──────────────────────────────────── #
    cf_serializable = {str(sorted(list(k))): v for k, v in cf.items()}

    return {
        "name": name,
        "desc": info["desc"],
        "n_samples": len(records),
        "shapley": dict(shapley),
        "shapley_ranking": [a for a, _ in shapley_ranking],
        "stii": stii,
        "stii_top_pair": stii_ranking[0][0],
        "unique_summary": unique_summary,
        "synergy_ranking": synergy_ranking,
        "cf": cf_serializable,
        "f1_full": cf[frozenset(ALL_AGENTS)],
        "f1_3best": max(
            cf[frozenset(s)] for s in combinations(ALL_AGENTS, 3)
        ),
    }


# ── 일관성 분석 ──────────────────────────────────────────────────────────── #
def analyze_consistency(results: List[Dict]) -> Dict:
    print("\n" + "=" * 65)
    print("  Cross-Dataset 일관성 분석")
    print("=" * 65)

    ds_names = [r["name"] for r in results]

    # 1) frequency 1위 유지 여부
    freq_top = [r["shapley_ranking"][0] == "frequency" for r in results]
    print(f"\n  [C1] frequency Shapley 1위 유지")
    for r, ok in zip(results, freq_top):
        rank = r["shapley_ranking"].index("frequency") + 1
        print(f"    {r['name']:12s}: {r['shapley_ranking']}  → {'✓' if ok else '✗'} ({rank}위)")
    c1_pass = sum(freq_top)

    # 2) spatial Shapley 최하위 유지
    spatial_last = [r["shapley_ranking"][-1] == "spatial" for r in results]
    print(f"\n  [C2] spatial Shapley 최하위 유지")
    for r, ok in zip(results, spatial_last):
        rank = r["shapley_ranking"].index("spatial") + 1
        print(f"    {r['name']:12s}: rank {rank}/4  → {'✓' if ok else '✗'}")
    c2_pass = sum(spatial_last)

    # 3) spatial Unique ≤ 0.01 (사실상 0) 유지
    spatial_unique_zero = [r["unique_summary"]["spatial"] <= 0.01 for r in results]
    print(f"\n  [C3] spatial Unique ≈ 0 (≤0.01) 유지")
    for r, ok in zip(results, spatial_unique_zero):
        val = r["unique_summary"]["spatial"]
        print(f"    {r['name']:12s}: Unique(spatial)={val:.4f}  → {'✓' if ok else '✗'}")
    c3_pass = sum(spatial_unique_zero)

    # 4) noise↔fatformer 시너지 Top-2 유지
    nf_top2 = []
    for r in results:
        top2_pairs = {item["pair"] for item in r["synergy_ranking"][:2]}
        nf_top2.append("noise__fatformer" in top2_pairs or "fatformer__noise" in top2_pairs)
    print(f"\n  [C4] noise↔fatformer 시너지 Top-2 유지")
    for r, ok in zip(results, nf_top2):
        top3 = [item["pair"] for item in r["synergy_ranking"][:3]]
        print(f"    {r['name']:12s}: {top3}  → {'✓' if ok else '✗'}")
    c4_pass = sum(nf_top2)

    # 5) Shapley ranking 전체 순서 일치율 (base와 비교)
    base_ranking = results[0]["shapley_ranking"]
    print(f"\n  [C5] Shapley ranking 전체 순서 (base={base_ranking} 기준)")
    for r in results[1:]:
        match = sum(a == b for a, b in zip(r["shapley_ranking"], base_ranking))
        print(f"    {r['name']:12s}: {r['shapley_ranking']}  → {match}/4 일치")

    # 6) 3-agent 최적 조합 일치 여부
    print(f"\n  [C6] 3개 조합 최고 F1 조합")
    for r in results:
        # 3개 부분집합 중 F1 최고
        best3 = max(
            (k for k in r["cf"] if len(eval(k)) == 3),
            key=lambda k: r["cf"][k]
        )
        print(f"    {r['name']:12s}: best3={best3}  F1={r['cf'][best3]:.4f}  "
              f"(full={r['f1_full']:.4f}, {r['cf'][best3]/r['f1_full']*100:.1f}%)")

    # 종합 판정
    print(f"\n  === 종합 결론 ===")
    checks = [
        ("C1 frequency 1위", c1_pass, len(results)),
        ("C2 spatial 최하위", c2_pass, len(results)),
        ("C3 spatial Unique≈0", c3_pass, len(results)),
        ("C4 noise↔fatformer 시너지 Top2", c4_pass, len(results)),
    ]
    all_pass = True
    for label, passed, total in checks:
        threshold = total * 0.75   # 75% 이상 일치
        ok = passed >= threshold
        all_pass = all_pass and ok
        print(f"  {label:35s}: {passed}/{total}  → {'✓ PASS' if ok else '✗ FAIL'}")

    verdict = "확정" if all_pass else "재검토 필요"
    print(f"\n  최종 판정: Phase 1 결론 [{verdict}]")
    if all_pass:
        print("  → freq+noise+fatformer 3개 조합, Spatial 제거 결론이 다중 데이터셋에서 일관됨")
    else:
        print("  → 일부 데이터셋에서 결론이 달라짐 → 조합 결정 수정 필요")

    return {
        "c1_freq_top": c1_pass, "c2_spatial_last": c2_pass,
        "c3_spatial_unique_zero": c3_pass, "c4_nf_synergy": c4_pass,
        "total_datasets": len(results), "verdict": verdict,
    }


# ── 메인 ─────────────────────────────────────────────────────────────────── #
def run():
    save_dir = ROOT / "experiments/results/shapley_phase1"
    save_dir.mkdir(parents=True, exist_ok=True)

    print("SHIELD Phase 1 Cross-Dataset Validation")
    print(f"검증 데이터셋: {list(DATASETS.keys())}")
    print(f"Seeds: {SEEDS} | Train/Val: {TRAIN_RATIO}/{VAL_RATIO}")

    all_results = []
    for name, info in DATASETS.items():
        jsonl_path = ROOT / info["jsonl"]
        if not jsonl_path.exists():
            print(f"\n  ⚠ SKIP [{name}]: {jsonl_path} 없음")
            continue
        result = run_one_dataset(name, info)
        all_results.append(result)

    if len(all_results) < 2:
        print("\n  ⚠ 검증 가능한 데이터셋이 2개 미만 — 중단")
        return

    consistency = analyze_consistency(all_results)

    # 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = save_dir / f"cross_dataset_validation_{timestamp}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "datasets": list(DATASETS.keys()),
            "results": all_results,
            "consistency": consistency,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n  결과 저장: {out_path}")


if __name__ == "__main__":
    run()
