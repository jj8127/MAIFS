#!/usr/bin/env python3
"""
SHIELD Phase 3.2: 경량 모델 Shapley + STII + PID-proxy 분석
=============================================================
4개 경량 모델(ForMa, MobileCLIP-ft4, MobileNetV2-DS, Tiny-LaDeDa)에 대해
Phase 1과 동일한 Shapley / STII 분석을 재실행하고,
PID-proxy(Redundancy / Synergy 분해)를 추가 계산한다.

분석 구조:
  - 데이터: 4개 백본 JSONL (forma, mobileclip_finetuned, mobilenetv2, tiny) × 4 DS
  - 특징 추출: 모델별 3-dim 예측 스코어 + 쌍별 불일치 + 집계 특징
  - 2^4=16 부분집합 → GBM Macro-F1 = v(S)
  - Shapley value, STII(k=2), CKA, PID-proxy 계산
  - 논문 Figure용 테이블/요약 출력

실행:
  .venv-qwen/bin/python experiments/run_shapley_phase3.py
  .venv-qwen/bin/python experiments/run_shapley_phase3.py --seeds 5 --skip-cka
"""
from __future__ import annotations

import json
import sys
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from itertools import combinations
from math import factorial
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ── 상수 ─────────────────────────────────────────────────────────────────── #
MODELS: Tuple[str, ...] = ("forma", "mobileclip", "mobilenetv2", "tiny")
MODEL_DISPLAY = {
    "forma":      "ForMa",
    "mobileclip": "MobileCLIP-ft4",
    "mobilenetv2":"MobileNetV2-DS",
    "tiny":       "Tiny-LaDeDa",
}
LABELS = ("authentic", "manipulated", "ai_generated")
LABEL_IDX = {l: i for i, l in enumerate(LABELS)}

EMPTY_F1 = 1.0 / 3  # 랜덤 베이스라인

OUT_DIR = ROOT / "experiments" / "results" / "shapley_phase3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 최신 JSONL 경로 매핑 ──────────────────────────────────────────────────── #
BACKBONE_DIR = ROOT / "experiments" / "results" / "backbone_eval"
DS_KEYS = ["base", "dsC", "opensdi", "aigenproxy"]

BACKBONE_PREFIX = {
    "forma":       "forma",
    "mobileclip":  "mobileclip_s2_finetuned",
    "mobilenetv2": "mobilenetv2_dualstream",
    "tiny":        "tiny_ladeda",
}


def latest_jsonl(model_key: str, ds: str) -> Path:
    """해당 모델·데이터셋의 가장 최신 JSONL 반환."""
    prefix = BACKBONE_PREFIX[model_key]
    candidates = sorted(BACKBONE_DIR.glob(f"{prefix}_{ds}_*.jsonl"))
    if not candidates:
        raise FileNotFoundError(f"JSONL 없음: {prefix}_{ds}_*.jsonl")
    return candidates[-1]


# ── 레코드 로드 ───────────────────────────────────────────────────────────── #
@dataclass
class ModelRecord:
    """단일 모델의 한 이미지 예측 결과."""
    image_path: str
    true_label: str
    auth_score:  float
    manip_score: float
    aigen_score: float

    @property
    def pred_label(self) -> str:
        scores = [self.auth_score, self.manip_score, self.aigen_score]
        return LABELS[int(np.argmax(scores))]

    @property
    def confidence(self) -> float:
        return max(self.auth_score, self.manip_score, self.aigen_score)


def _scores_from_record(d: dict, model_key: str) -> Tuple[float, float, float]:
    """
    JSONL 레코드에서 (auth, manip, aigen) 스코어 추출.
    forma / tiny처럼 scores dict가 없으면 verdict+confidence에서 재구성.
    """
    if "scores" in d:
        sc = d["scores"]
        return (
            float(sc.get("authentic", 0.0)),
            float(sc.get("manipulated", 0.0)),
            float(sc.get("ai_generated", 0.0)),
        )
    # binary verdict 모델
    verdict = d.get("verdict", d.get("pred_label", "authentic"))
    conf = float(d.get("confidence", 0.33))
    left = (1.0 - conf) / 2
    if model_key == "forma":
        # authentic vs manipulated only
        if verdict == "authentic":
            return conf, left * 2, 0.0
        else:  # manipulated
            return left * 2, conf, 0.0
    elif model_key == "tiny":
        # authentic vs ai_generated only
        if verdict == "authentic":
            return conf, 0.0, left * 2
        else:  # ai_generated
            return left * 2, 0.0, conf
    else:
        # 알 수 없는 경우 uniform
        return 1/3, 1/3, 1/3


def load_model_records(model_key: str) -> Dict[str, ModelRecord]:
    """4개 데이터셋을 통합해 {image_path: ModelRecord} 반환."""
    recs: Dict[str, ModelRecord] = {}
    for ds in DS_KEYS:
        path = latest_jsonl(model_key, ds)
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                img = d["image_path"]
                lbl = d.get("true_label", "authentic")
                if lbl not in LABEL_IDX:
                    continue
                auth, manip, aigen = _scores_from_record(d, model_key)
                recs[img] = ModelRecord(img, lbl, auth, manip, aigen)
    print(f"  {model_key:12s} → {len(recs):5d}개 레코드  "
          f"({latest_jsonl(model_key, 'base').name})")
    return recs


# ── 부분집합 특징 추출 ─────────────────────────────────────────────────────── #
def extract_subset_features(
    common_imgs: List[str],
    model_data: Dict[str, Dict[str, ModelRecord]],
    subset: Tuple[str, ...],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    주어진 모델 subset의 특징을 추출한다.

    특징 구조:
      per-model  : 3-dim score vector × |subset|
      pairwise   : disagree(1) + conf_diff(1) × C(|S|,2)
      aggregate  : conf_var, verdict_entropy, max_min_gap, unique_verdicts  (4)
    """
    pairs = list(combinations(subset, 2))
    X_rows, y_rows = [], []

    for img in common_imgs:
        row = []
        confs, pred_labels = [], []

        # per-model
        for m in subset:
            rec = model_data[m][img]
            row.append(np.array([rec.auth_score, rec.manip_score, rec.aigen_score],
                                 dtype=np.float32))
            confs.append(rec.confidence)
            pred_labels.append(rec.pred_label)

        # pairwise
        for m1, m2 in pairs:
            r1, r2 = model_data[m1][img], model_data[m2][img]
            disagree = float(r1.pred_label != r2.pred_label)
            diff = abs(r1.confidence - r2.confidence)
            row.append(np.array([disagree, diff], dtype=np.float32))

        # aggregate
        conf_var  = float(np.var(confs)) if len(confs) > 1 else 0.0
        vc: Dict[str, int] = {}
        for v in pred_labels:
            vc[v] = vc.get(v, 0) + 1
        n = len(pred_labels)
        entropy = 0.0
        for cnt in vc.values():
            p = cnt / n
            entropy -= p * np.log(p + 1e-12)
        max_min_gap    = max(confs) - min(confs)
        unique_v       = float(len(set(pred_labels)))
        row.append(np.array([conf_var, entropy, max_min_gap, unique_v],
                             dtype=np.float32))

        X_rows.append(np.concatenate(row))
        y_rows.append(LABEL_IDX[model_data[subset[0]][img].true_label])

    return np.array(X_rows, dtype=np.float32), np.array(y_rows, dtype=np.int64)


# ── 단일 부분집합 평가 ─────────────────────────────────────────────────────── #
def evaluate_subset(
    common_imgs: List[str],
    model_data: Dict[str, Dict[str, ModelRecord]],
    subset: Tuple[str, ...],
    seeds: List[int],
    gbm_params: dict,
) -> Dict:
    X, y = extract_subset_features(common_imgs, model_data, subset)
    f1_scores = []
    for seed in seeds:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y)
        clf = Pipeline([
            ("sc", StandardScaler()),
            ("gbm", GradientBoostingClassifier(**{**gbm_params, "random_state": seed})),
        ])
        clf.fit(X_tr, y_tr)
        f1_scores.append(float(f1_score(y_te, clf.predict(X_te),
                                         average="macro", zero_division=0)))
    return {
        "subset": list(subset),
        "f1_mean": float(np.mean(f1_scores)),
        "f1_std":  float(np.std(f1_scores)),
        "f1_per_seed": f1_scores,
        "feature_dim": X.shape[1],
    }


# ── Shapley value ──────────────────────────────────────────────────────────── #
def compute_shapley(v: Dict[FrozenSet[str], float],
                    models: Tuple[str, ...]) -> Dict[str, float]:
    n = len(models)
    phi = {m: 0.0 for m in models}
    for model in models:
        others = [m for m in models if m != model]
        for size in range(n):
            for combo in combinations(others, size):
                s = frozenset(combo)
                s_with = s | {model}
                marginal = v.get(s_with, EMPTY_F1) - v.get(s, EMPTY_F1)
                weight   = factorial(size) * factorial(n - size - 1) / factorial(n)
                phi[model] += weight * marginal
    return phi


# ── STII (pairwise, k=2) ───────────────────────────────────────────────────── #
def compute_stii(v: Dict[FrozenSet[str], float],
                 models: Tuple[str, ...]) -> Dict[Tuple[str, str], float]:
    n = len(models)
    stii = {}
    for i, mi in enumerate(models):
        for j, mj in enumerate(models):
            if j <= i:
                continue
            others = [m for m in models if m not in {mi, mj}]
            val = 0.0
            for size in range(n - 1):
                for combo in combinations(others, size):
                    s    = frozenset(combo)
                    s_ij = s | {mi, mj}
                    s_i  = s | {mi}
                    s_j  = s | {mj}
                    interaction = (v.get(s_ij, EMPTY_F1) - v.get(s_i, EMPTY_F1)
                                   - v.get(s_j, EMPTY_F1) + v.get(s, EMPTY_F1))
                    weight = factorial(size) * factorial(n - size - 2) / factorial(n - 1)
                    val += weight * interaction
            stii[(mi, mj)] = val
    return stii


# ── CKA (linear kernel) ────────────────────────────────────────────────────── #
def compute_cka(common_imgs: List[str],
                model_data: Dict[str, Dict[str, ModelRecord]],
                models: Tuple[str, ...]) -> Dict[str, float]:
    feats: Dict[str, np.ndarray] = {}
    for m in models:
        X = np.array([[model_data[m][img].auth_score,
                       model_data[m][img].manip_score,
                       model_data[m][img].aigen_score] for img in common_imgs],
                     dtype=np.float32)
        feats[m] = X

    cka_dict = {}
    for m1, m2 in combinations(models, 2):
        K1 = feats[m1] @ feats[m1].T
        K2 = feats[m2] @ feats[m2].T
        n  = K1.shape[0]
        H  = np.eye(n) - np.ones((n, n)) / n
        K1c, K2c = H @ K1 @ H, H @ K2 @ H
        denom = np.sqrt(max(np.trace(K1c @ K1c), 1e-12) *
                        max(np.trace(K2c @ K2c), 1e-12)) / (n - 1) ** 2
        cka = float(np.trace(K1c @ K2c) / (n - 1) ** 2 / denom) if denom > 0 else 0.0
        cka_dict[f"{m1}__{m2}"] = cka
    return cka_dict


# ── PID-proxy (Redundancy / Synergy) ──────────────────────────────────────── #
def compute_pid_proxy(v: Dict[FrozenSet[str], float],
                      models: Tuple[str, ...]) -> Dict[str, Dict]:
    """
    모든 모델 쌍(i,j)에 대해 간단한 PID 분해.

    v0 = v(∅) = EMPTY_F1 (랜덤 베이스라인)
    vi = v({i}) - v0    (단독 기여)
    vj = v({j}) - v0
    vij = v({i,j}) - v0 (쌍 기여)

    Synergy    = max(0,  vij - vi - vj)          → 조합에서만 나오는 정보
    Redundancy = max(0,  vi + vj - vij)           → 두 모델이 공유하는 정보
    Unique_i   = vi - min(vi, Redundancy)         → i만의 고유 정보
    Unique_j   = vj - min(vj, Redundancy)         → j만의 고유 정보
    """
    pid = {}
    for mi, mj in combinations(models, 2):
        vi  = v.get(frozenset({mi}), EMPTY_F1) - EMPTY_F1
        vj  = v.get(frozenset({mj}), EMPTY_F1) - EMPTY_F1
        vij = v.get(frozenset({mi, mj}), EMPTY_F1) - EMPTY_F1
        synergy    = max(0.0, vij - vi - vj)
        redundancy = max(0.0, vi + vj - vij)
        unique_i   = max(0.0, vi - redundancy)
        unique_j   = max(0.0, vj - redundancy)
        total      = max(vij, 1e-9)
        pid[f"{mi}__{mj}"] = {
            "vi":         round(vi,  4),
            "vj":         round(vj,  4),
            "vij":        round(vij, 4),
            "synergy":    round(synergy,    4),
            "redundancy": round(redundancy, 4),
            "unique_i":   round(unique_i,   4),
            "unique_j":   round(unique_j,   4),
            "syn_ratio":  round(synergy    / total, 3),
            "red_ratio":  round(redundancy / total, 3),
        }
    return pid


# ── 메인 ──────────────────────────────────────────────────────────────────── #
def run(n_seeds: int = 10, skip_cka: bool = False):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    gbm_params = dict(n_estimators=200, max_depth=4, learning_rate=0.1,
                      subsample=0.8, min_samples_leaf=5)
    seeds = list(range(300, 300 + n_seeds))

    # ── 데이터 로드 ──
    print("\n[Step 0] 4개 백본 JSONL 로드 (4 DS 통합)")
    model_data: Dict[str, Dict[str, ModelRecord]] = {}
    for m in MODELS:
        model_data[m] = load_model_records(m)

    # 공통 이미지만 사용
    common = sorted(set.intersection(*[set(model_data[m].keys()) for m in MODELS]))
    print(f"\n  공통 이미지: {len(common)}개")
    # 레이블 분포
    label_counts: Dict[str, int] = {}
    for img in common:
        lbl = model_data[MODELS[0]][img].true_label
        label_counts[lbl] = label_counts.get(lbl, 0) + 1
    for lbl, cnt in label_counts.items():
        print(f"    {lbl}: {cnt}")

    # ── Step 1: 16개 부분집합 평가 ──
    print(f"\n[Step 1] 2^4=16 부분집합 평가 ({n_seeds} seeds 각)")
    v: Dict[FrozenSet[str], float] = {frozenset(): EMPTY_F1}
    subset_results = {}

    all_subsets = []
    for size in range(1, len(MODELS) + 1):
        for combo in combinations(MODELS, size):
            all_subsets.append(combo)

    for idx, subset in enumerate(all_subsets, 1):
        key = frozenset(subset)
        display = "{" + ", ".join(MODEL_DISPLAY[m] for m in subset) + "}"
        print(f"  [{idx:2d}/15] {display} ...", end=" ", flush=True)
        res = evaluate_subset(common, model_data, subset, seeds, gbm_params)
        v[key] = res["f1_mean"]
        subset_results[str(sorted(subset))] = res
        print(f"F1={res['f1_mean']:.4f} ± {res['f1_std']:.4f}")

    # ── Step 2: Shapley values ──
    print("\n[Step 2] Shapley value 계산")
    shapley = compute_shapley(v, MODELS)
    shapley_ranking = sorted(shapley.items(), key=lambda x: -x[1])
    for m, val in shapley_ranking:
        print(f"  φ({MODEL_DISPLAY[m]:18s}) = {val:+.4f}")

    # ── Step 3: STII ──
    print("\n[Step 3] STII pairwise interaction 계산")
    stii = compute_stii(v, MODELS)
    stii_ranking = sorted(stii.items(), key=lambda x: -abs(x[1]))
    stii_serial = {f"{mi}__{mj}": val for (mi, mj), val in stii.items()}
    for (mi, mj), val in stii_ranking:
        sign = "↑ 시너지" if val > 0 else "↓ 중복"
        print(f"  STII({MODEL_DISPLAY[mi]:18s}, {MODEL_DISPLAY[mj]:18s}) = {val:+.4f}  {sign}")

    # ── Step 4: CKA ──
    cka_dict = {}
    if not skip_cka:
        print("\n[Step 4] CKA feature similarity 계산 (선형 커널)")
        cka_dict = compute_cka(common, model_data, MODELS)
        for key, val in sorted(cka_dict.items(), key=lambda x: -x[1]):
            m1, m2 = key.split("__")
            print(f"  CKA({MODEL_DISPLAY[m1]:18s}, {MODEL_DISPLAY[m2]:18s}) = {val:.4f}")
    else:
        print("\n[Step 4] CKA skip")

    # ── Step 5: PID-proxy ──
    print("\n[Step 5] PID-proxy (Redundancy / Synergy 분해)")
    pid = compute_pid_proxy(v, MODELS)
    print(f"  {'Pair':38s}  {'Redund':>7}  {'Unique_i':>8}  {'Unique_j':>8}  {'Synergy':>8}")
    for pair_key, vals in pid.items():
        mi, mj = pair_key.split("__")
        label = f"{MODEL_DISPLAY[mi]} × {MODEL_DISPLAY[mj]}"
        print(f"  {label:38s}  {vals['redundancy']:7.4f}  "
              f"{vals['unique_i']:8.4f}  {vals['unique_j']:8.4f}  "
              f"{vals['synergy']:8.4f}")

    # ── 요약 출력 ──
    _print_summary(shapley, stii, pid, v)

    # ── 저장 ──
    out = OUT_DIR / f"shapley_phase3_{ts}.json"
    result = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "models": list(MODELS),
            "n_samples": len(common),
            "label_counts": label_counts,
            "seeds": seeds,
            "gbm_params": gbm_params,
        },
        "characteristic_function": {
            str(sorted(list(k))): float(vv) for k, vv in v.items()
        },
        "shapley_values": {m: float(val) for m, val in shapley.items()},
        "shapley_ranking": [
            {"model": m, "display": MODEL_DISPLAY[m], "value": float(val)}
            for m, val in shapley_ranking
        ],
        "stii_k2": stii_serial,
        "stii_ranking": [
            {"pair": f"{mi}__{mj}", "value": float(val)}
            for (mi, mj), val in stii_ranking
        ],
        "cka_similarity": cka_dict,
        "pid_proxy": pid,
        "subset_results": subset_results,
    }

    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n  저장: {out}")
    return result


def _print_summary(shapley, stii, pid, v):
    print(f"\n{'='*70}")
    print("  SHIELD Phase 3.2 — 경량 모델 Shapley 분석 요약")
    print(f"{'='*70}")

    ranked = sorted(shapley.items(), key=lambda x: -x[1])
    print("\n  [모델 기여도 Shapley 순위]")
    for rank, (m, val) in enumerate(ranked, 1):
        bar = "█" * int(val / max(shapley.values()) * 20)
        print(f"  {rank}. {MODEL_DISPLAY[m]:18s} φ={val:+.4f}  {bar}")

    print("\n  [pairwise STII (시너지/중복)]")
    for (mi, mj), val in sorted(stii.items(), key=lambda x: -abs(x[1])):
        tag = "↑ 시너지" if val > 0 else "↓ 중복"
        print(f"     {MODEL_DISPLAY[mi]:18s} × {MODEL_DISPLAY[mj]:18s}: "
              f"{val:+.4f}  {tag}")

    # 가장 시너지 높은 쌍
    best_syn  = max(stii.items(), key=lambda x: x[1])
    best_redu = min(stii.items(), key=lambda x: x[1])
    print(f"\n  최대 시너지: {MODEL_DISPLAY[best_syn[0][0]]} × "
          f"{MODEL_DISPLAY[best_syn[0][1]]} (STII={best_syn[1]:+.4f})")
    print(f"  최대 중복  : {MODEL_DISPLAY[best_redu[0][0]]} × "
          f"{MODEL_DISPLAY[best_redu[0][1]]} (STII={best_redu[1]:+.4f})")

    # 전체 4모델 앙상블 기여
    full_f1 = v.get(frozenset(MODELS), 0.0)
    clip_f1 = v.get(frozenset({"mobileclip"}), EMPTY_F1)
    print(f"\n  전체 앙상블 F1     : {full_f1:.4f}")
    print(f"  MobileCLIP 단독 F1 : {clip_f1:.4f}  (baseline)")
    print(f"  앙상블 개선        : {full_f1 - clip_f1:+.4f}")
    print(f"{'='*70}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SHIELD Phase 3.2 Shapley 분석")
    parser.add_argument("--seeds", type=int, default=10, help="평가 반복 seed 수")
    parser.add_argument("--skip-cka", action="store_true", help="CKA 계산 스킵")
    args = parser.parse_args()
    run(n_seeds=args.seeds, skip_cka=args.skip_cka)
