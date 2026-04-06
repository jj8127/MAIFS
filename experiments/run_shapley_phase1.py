"""
SHIELD Phase 1: Model Shapley + STII 계산

목적:
    4개 에이전트(frequency, noise, fatformer, spatial)의 가치와 상호작용을
    이론적으로 정량화한다.

방법:
    1) 기존 수집 데이터(precollected JSONL)를 로드
    2) 2⁴=16개 에이전트 부분집합 S 각각에 대해:
       - 해당 에이전트들의 출력만 사용해 DAAC GBM 학습·평가
       - v(S) = mean Macro-F1 (across seeds)
    3) v(S) 값들로 Shapley value 계산
    4) STII(k=2) pairwise interaction 계산
    5) CKA-style 특징 유사도 계산 (단순 feature correlation)
    6) 결과 저장

실행:
    .venv-qwen/bin/python experiments/run_shapley_phase1.py
    .venv-qwen/bin/python experiments/run_shapley_phase1.py experiments/configs/shapley_phase1.yaml
"""
from __future__ import annotations

import json
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations
from math import factorial
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Tuple

import numpy as np
import yaml
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.meta.simulator import AGENT_NAMES, VERDICTS, TRUE_LABELS, SimulatedOutput

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# 상수
# --------------------------------------------------------------------------- #
ALL_AGENTS: Tuple[str, ...] = tuple(AGENT_NAMES)   # (frequency, noise, fatformer, spatial)
N_AGENTS = len(ALL_AGENTS)
AGENT_IDX = {a: i for i, a in enumerate(ALL_AGENTS)}
VERDICT_IDX = {v: i for i, v in enumerate(VERDICTS)}
LABEL_IDX = {l: i for i, l in enumerate(TRUE_LABELS)}

# 빈 집합 결과 (에이전트 없음 = random baseline)
EMPTY_SET_F1 = 1.0 / len(TRUE_LABELS)   # ≈ 0.333


# --------------------------------------------------------------------------- #
# 데이터 로드
# --------------------------------------------------------------------------- #
@dataclass
class Record:
    true_label: str
    agent_verdicts: Dict[str, str]
    agent_confidences: Dict[str, float]

    def to_simulated(self) -> SimulatedOutput:
        return SimulatedOutput(
            true_label=self.true_label,
            agent_verdicts=dict(self.agent_verdicts),
            agent_confidences=dict(self.agent_confidences),
        )


def load_jsonl(path: str) -> List[Record]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            records.append(Record(
                true_label=d["true_label"],
                agent_verdicts=d["agent_verdicts"],
                agent_confidences=d["agent_confidences"],
            ))
    return records


# --------------------------------------------------------------------------- #
# Subset 특징 추출기
# --------------------------------------------------------------------------- #
def extract_subset_features(
    records: List[Record],
    subset: Tuple[str, ...],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    주어진 에이전트 subset의 출력만 사용해 메타 특징을 추출한다.

    특징 구조 (subset agents만 사용):
        per-agent  : 4 verdicts one-hot + 1 confidence = 5 × |subset|
        pairwise   : disagree + conf_diff + conflict = 3 × C(|subset|, 2)
        aggregate  : conf_var, verdict_entropy, max_min_gap,
                     unique_verdicts, majority_ratio = 5
    """
    if not subset:
        raise ValueError("subset이 비어있습니다.")

    pairs = list(combinations(subset, 2))
    X_rows = []
    y_rows = []

    for rec in records:
        row = []

        # --- per-agent features ---
        for agent in subset:
            verdict = rec.agent_verdicts.get(agent, "uncertain")
            one_hot = np.zeros(len(VERDICTS), dtype=np.float32)
            one_hot[VERDICT_IDX[verdict]] = 1.0
            row.append(one_hot)
            row.append(np.array([rec.agent_confidences.get(agent, 0.33)], dtype=np.float32))

        # --- pairwise features ---
        disagree_vec, conf_diff_vec, conflict_vec = [], [], []
        for a1, a2 in pairs:
            v1 = rec.agent_verdicts.get(a1, "uncertain")
            v2 = rec.agent_verdicts.get(a2, "uncertain")
            c1 = rec.agent_confidences.get(a1, 0.33)
            c2 = rec.agent_confidences.get(a2, 0.33)
            is_disagree = float(v1 != v2)
            diff = abs(c1 - c2)
            disagree_vec.append(is_disagree)
            conf_diff_vec.append(diff)
            conflict_vec.append(diff if is_disagree else 0.0)
        if pairs:
            row.append(np.array(disagree_vec, dtype=np.float32))
            row.append(np.array(conf_diff_vec, dtype=np.float32))
            row.append(np.array(conflict_vec, dtype=np.float32))

        # --- aggregate features ---
        confs = [rec.agent_confidences.get(a, 0.33) for a in subset]
        verdicts_here = [rec.agent_verdicts.get(a, "uncertain") for a in subset]

        conf_var = float(np.var(confs))
        # verdict entropy
        verdict_counts: Dict[str, int] = {}
        for v in verdicts_here:
            verdict_counts[v] = verdict_counts.get(v, 0) + 1
        n = len(verdicts_here)
        entropy = 0.0
        for cnt in verdict_counts.values():
            p = cnt / n
            if p > 0:
                entropy -= p * np.log(p + 1e-12)
        max_min_gap = max(confs) - min(confs)
        unique_v = float(len(set(verdicts_here)))
        # majority ratio
        majority_cnt = max(verdict_counts.values()) if verdict_counts else 1
        majority_ratio = majority_cnt / n

        row.append(np.array([conf_var, entropy, max_min_gap, unique_v, majority_ratio], dtype=np.float32))

        X_rows.append(np.concatenate(row))

        # label
        lbl = rec.true_label
        y_rows.append(LABEL_IDX.get(lbl, 0))

    return np.array(X_rows, dtype=np.float32), np.array(y_rows, dtype=np.int64)


# --------------------------------------------------------------------------- #
# 단일 subset 평가
# --------------------------------------------------------------------------- #
def evaluate_subset(
    records: List[Record],
    subset: Tuple[str, ...],
    seeds: List[int],
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    gbm_params: Optional[Dict] = None,
) -> Dict:
    """
    주어진 subset에서 GBM을 학습하고 Macro-F1을 반환한다.
    여러 seed로 반복해 mean ± std를 계산한다.
    """
    if gbm_params is None:
        gbm_params = {
            "n_estimators": 200,
            "max_depth": 4,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "min_samples_leaf": 5,
            "random_state": 42,
        }

    X, y = extract_subset_features(records, subset)
    test_ratio = 1.0 - train_ratio - val_ratio
    f1_scores = []

    for seed in seeds:
        # train / test split (val은 GBM에서는 미사용, 구조 유지용)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_ratio,
            random_state=seed,
            stratify=y,
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train,
            test_size=val_ratio / (train_ratio + val_ratio),
            random_state=seed,
            stratify=y_train,
        )

        gbm = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(**{**gbm_params, "random_state": seed})),
        ])
        gbm.fit(X_train, y_train)
        y_pred = gbm.predict(X_test)

        labels = list(range(len(TRUE_LABELS)))
        f1 = f1_score(y_test, y_pred, labels=labels, average="macro", zero_division=0)
        f1_scores.append(float(f1))

    return {
        "subset": list(subset),
        "f1_mean": float(np.mean(f1_scores)),
        "f1_std": float(np.std(f1_scores)),
        "f1_per_seed": f1_scores,
        "n_seeds": len(seeds),
        "feature_dim": X.shape[1],
    }


# --------------------------------------------------------------------------- #
# Shapley value 계산
# --------------------------------------------------------------------------- #
def compute_shapley_values(
    v: Dict[FrozenSet[str], float],
    agents: Tuple[str, ...],
) -> Dict[str, float]:
    """
    v: 부분집합 → 특성함수값 (Macro-F1)
    반환: 에이전트 → Shapley value
    """
    n = len(agents)
    phi = {a: 0.0 for a in agents}

    for agent in agents:
        for size in range(n):
            # agent를 포함하지 않는 크기=size인 부분집합들
            others = [a for a in agents if a != agent]
            for combo in combinations(others, size):
                s = frozenset(combo)
                s_with = frozenset(combo) | {agent}
                marginal = v.get(s_with, EMPTY_SET_F1) - v.get(s, EMPTY_SET_F1)
                weight = factorial(size) * factorial(n - size - 1) / factorial(n)
                phi[agent] += weight * marginal

    return phi


# --------------------------------------------------------------------------- #
# STII (Shapley-Taylor Interaction Index, k=2)
# --------------------------------------------------------------------------- #
def compute_stii_k2(
    v: Dict[FrozenSet[str], float],
    agents: Tuple[str, ...],
) -> Dict[Tuple[str, str], float]:
    """
    pairwise Shapley-Taylor Interaction Index.
    STII_{ij} = Σ_{S ⊆ N\{i,j}} [weight × (v(S∪{i,j}) - v(S∪{i}) - v(S∪{j}) + v(S))]
    """
    n = len(agents)
    stii = {}

    for i_idx, ai in enumerate(agents):
        for j_idx, aj in enumerate(agents):
            if j_idx <= i_idx:
                continue
            pair = (ai, aj)
            val = 0.0
            others = [a for a in agents if a not in {ai, aj}]

            for size in range(n - 1):  # size of S (excluding i and j)
                for combo in combinations(others, size):
                    s = frozenset(combo)
                    s_ij = s | {ai, aj}
                    s_i = s | {ai}
                    s_j = s | {aj}

                    interaction = (
                        v.get(s_ij, EMPTY_SET_F1)
                        - v.get(s_i, EMPTY_SET_F1)
                        - v.get(s_j, EMPTY_SET_F1)
                        + v.get(s, EMPTY_SET_F1)
                    )
                    weight = factorial(size) * factorial(n - size - 2) / factorial(n - 1)
                    val += weight * interaction

            stii[pair] = val

    return stii


# --------------------------------------------------------------------------- #
# Feature correlation (CKA proxy)
# --------------------------------------------------------------------------- #
def compute_feature_correlation(
    records: List[Record],
    agents: Tuple[str, ...],
    seed: int = 300,
) -> Dict[str, float]:
    """
    각 에이전트 쌍의 특징 벡터 간 피어슨 상관관계 (CKA 근사).
    단일 에이전트 특징을 추출해 cosine similarity를 계산한다.
    """
    # 각 에이전트의 특징 벡터 추출 (단일 에이전트 subset)
    agent_features: Dict[str, np.ndarray] = {}
    for agent in agents:
        X, _ = extract_subset_features(records, (agent,))
        agent_features[agent] = X  # (N, d_agent)

    correlations = {}
    for a1, a2 in combinations(agents, 2):
        X1 = agent_features[a1]
        X2 = agent_features[a2]
        # 차원이 다를 수 있으므로 HSIC 대신 벡터 norm 내적 사용
        # CKA(K1, K2) = HSIC(K1,K2) / sqrt(HSIC(K1,K1) * HSIC(K2,K2))
        # 여기서는 linear kernel 사용
        K1 = X1 @ X1.T
        K2 = X2 @ X2.T
        # 중앙화
        n = K1.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        K1c = H @ K1 @ H
        K2c = H @ K2 @ H
        hsic_12 = np.trace(K1c @ K2c) / (n - 1) ** 2
        hsic_11 = np.trace(K1c @ K1c) / (n - 1) ** 2
        hsic_22 = np.trace(K2c @ K2c) / (n - 1) ** 2
        denom = np.sqrt(max(hsic_11, 1e-12) * max(hsic_22, 1e-12))
        cka = float(hsic_12 / denom) if denom > 0 else 0.0
        correlations[f"{a1}__{a2}"] = cka

    return correlations


# --------------------------------------------------------------------------- #
# 메인 실행
# --------------------------------------------------------------------------- #
def load_config(config_path: Optional[str]) -> Dict:
    if config_path and Path(config_path).exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    # 기본값
    return {
        "jsonl_path": "experiments/results/phase2_patha_scale500_gain_predictor/patha_agent_outputs_20260304_080157.jsonl",
        "seeds": [300, 301, 302, 303, 304, 305, 306, 307, 308, 309],
        "split": {"train_ratio": 0.6, "val_ratio": 0.2},
        "gbm": {
            "n_estimators": 200,
            "max_depth": 4,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "min_samples_leaf": 5,
        },
        "skip_cka": False,
        "output": {"save_dir": "experiments/results/shapley_phase1"},
    }


def run(config_path: Optional[str] = None) -> Dict:
    cfg = load_config(config_path)

    jsonl_path = ROOT / cfg["jsonl_path"]
    seeds = cfg["seeds"]
    train_ratio = cfg["split"]["train_ratio"]
    val_ratio = cfg["split"]["val_ratio"]
    gbm_params = {**cfg.get("gbm", {}), "random_state": 42}
    skip_cka = cfg.get("skip_cka", False)
    save_dir = ROOT / cfg["output"]["save_dir"]
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Shapley Phase 1] 데이터 로드: {jsonl_path}")
    records = load_jsonl(str(jsonl_path))
    print(f"  → {len(records)}개 레코드")

    # ------------------------------------------------------------------ #
    # Step 1: 16개 부분집합 평가
    # ------------------------------------------------------------------ #
    print("\n[Step 1] 16개 부분집합 평가 (각 {n_seeds} seed)".format(n_seeds=len(seeds)))
    v: Dict[FrozenSet[str], float] = {}
    subset_results = {}

    all_subsets = []
    for size in range(1, N_AGENTS + 1):
        for combo in combinations(ALL_AGENTS, size):
            all_subsets.append(combo)

    total = len(all_subsets)
    for idx, subset in enumerate(all_subsets, 1):
        key = frozenset(subset)
        print(f"  [{idx:2d}/{total}] {{{', '.join(subset)}}} ...", end=" ", flush=True)
        result = evaluate_subset(records, subset, seeds, train_ratio, val_ratio, gbm_params)
        v[key] = result["f1_mean"]
        subset_results[str(sorted(subset))] = result
        print(f"F1={result['f1_mean']:.4f} ± {result['f1_std']:.4f}")

    # 빈 집합
    v[frozenset()] = EMPTY_SET_F1

    # ------------------------------------------------------------------ #
    # Step 2: Shapley values
    # ------------------------------------------------------------------ #
    print("\n[Step 2] Shapley value 계산")
    shapley = compute_shapley_values(v, ALL_AGENTS)
    for agent, val in sorted(shapley.items(), key=lambda x: -x[1]):
        print(f"  φ({agent:12s}) = {val:+.4f}")

    # ------------------------------------------------------------------ #
    # Step 3: STII (k=2)
    # ------------------------------------------------------------------ #
    print("\n[Step 3] STII pairwise interaction 계산")
    stii = compute_stii_k2(v, ALL_AGENTS)
    stii_serializable = {f"{a1}__{a2}": val for (a1, a2), val in sorted(stii.items(), key=lambda x: -abs(x[1]))}
    for (a1, a2), val in sorted(stii.items(), key=lambda x: -abs(x[1])):
        print(f"  STII({a1:10s}, {a2:10s}) = {val:+.4f}")

    # ------------------------------------------------------------------ #
    # Step 4: CKA (feature similarity)
    # ------------------------------------------------------------------ #
    cka_results = {}
    if not skip_cka:
        print("\n[Step 4] CKA feature similarity 계산 (선형 커널)")
        print("  (샘플 수가 많으면 시간이 걸릴 수 있습니다...)")
        cka_results = compute_feature_correlation(records, ALL_AGENTS)
        for pair_key, val in sorted(cka_results.items(), key=lambda x: -x[1]):
            a1, a2 = pair_key.split("__")
            print(f"  CKA({a1:10s}, {a2:10s}) = {val:.4f}")
    else:
        print("\n[Step 4] CKA 계산 skip (skip_cka=True)")

    # ------------------------------------------------------------------ #
    # Step 5: 결과 저장
    # ------------------------------------------------------------------ #
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = save_dir / f"shapley_phase1_{timestamp}.json"

    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "jsonl_path": str(jsonl_path),
            "seeds": seeds,
            "n_samples": len(records),
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
        },
        "characteristic_function": {
            str(sorted(list(k))): float(vv) for k, vv in v.items()
        },
        "shapley_values": {k: float(vv) for k, vv in shapley.items()},
        "shapley_ranking": [
            {"agent": a, "value": float(vv)}
            for a, vv in sorted(shapley.items(), key=lambda x: -x[1])
        ],
        "stii_k2": stii_serializable,
        "stii_ranking": [
            {"pair": f"{a1}__{a2}", "value": float(vv)}
            for (a1, a2), vv in sorted(stii.items(), key=lambda x: -abs(x[1]))
        ],
        "cka_similarity": cka_results,
        "subset_results": subset_results,
        "summary": _build_summary(shapley, stii, cka_results),
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n[완료] 결과 저장: {out_path}")
    _print_summary(output["summary"])

    return output


def _build_summary(
    shapley: Dict[str, float],
    stii: Dict[Tuple[str, str], float],
    cka: Dict[str, float],
) -> Dict:
    ranked_agents = sorted(shapley.items(), key=lambda x: -x[1])
    top_interaction = max(stii.items(), key=lambda x: abs(x[1])) if stii else (("?", "?"), 0.0)
    most_similar_pair = max(cka.items(), key=lambda x: x[1]) if cka else ("?", 0.0)

    return {
        "top_agent": ranked_agents[0][0] if ranked_agents else "?",
        "top_agent_shapley": ranked_agents[0][1] if ranked_agents else 0.0,
        "agent_ranking": [a for a, _ in ranked_agents],
        "strongest_interaction_pair": f"{top_interaction[0][0]}__{top_interaction[0][1]}",
        "strongest_interaction_value": top_interaction[1],
        "most_redundant_pair": most_similar_pair[0] if cka else "?",
        "most_redundant_cka": most_similar_pair[1] if cka else 0.0,
    }


def _print_summary(summary: Dict) -> None:
    print("\n" + "=" * 60)
    print("  SHIELD Phase 1 결과 요약")
    print("=" * 60)
    print(f"  에이전트 Shapley 순위: {' > '.join(summary['agent_ranking'])}")
    print(f"  최고 기여 에이전트  : {summary['top_agent']} (φ={summary['top_agent_shapley']:+.4f})")
    print(f"  최강 pairwise 상호작용: {summary['strongest_interaction_pair']} "
          f"(STII={summary['strongest_interaction_value']:+.4f})")
    if summary.get("most_redundant_pair") and summary["most_redundant_pair"] != "?":
        print(f"  최고 중복 에이전트 쌍: {summary['most_redundant_pair']} "
              f"(CKA={summary['most_redundant_cka']:.4f})")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SHIELD Phase 1: Model Shapley + STII")
    parser.add_argument("config", nargs="?", default=None, help="YAML config path (선택)")
    args = parser.parse_args()
    run(args.config)
