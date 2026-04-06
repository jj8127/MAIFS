"""
SHIELD Phase 1.4: PID (Partial Information Decomposition)

목적:
    4개 에이전트 쌍에 대해 고유(Unique) / 중복(Redundant) / 시너지(Synergistic)
    정보를 정량화하여 경량화 및 cascade 설계 근거를 제공한다.

이론:
    Williams & Beer (2010) Imin redundancy measure 기반 PID.

    Y = true label (authentic / manipulated / ai_generated)
    A_i = agent verdict (authentic / manipulated / ai_generated / uncertain)

    pairwise PID(Y; A_i, A_j):
        Specific information: I_spec(y; A_k) = Σ_{a} p(a|y) log[p(y|a) / p(y)]
        Redundancy (Imin): R(Y; A_i, A_j) = Σ_y p(y) · min(I_spec(y; A_i), I_spec(y; A_j))
        Unique(A_i) = MI(Y; A_i) - R
        Unique(A_j) = MI(Y; A_j) - R
        Synergy     = MI(Y; A_i, A_j) - MI(Y; A_i) - MI(Y; A_j) + R

    Co-information (interaction information):
        CoI(Y; A_i; A_j) = MI(Y;A_i) + MI(Y;A_j) - MI(Y;A_i,A_j)
        양수 → 중복 우세 / 음수 → 시너지 우세

    4-agent 분해:
        각 에이전트별 Unique 정보 = MI(Y; A_i) - max over j≠i [R(Y; A_i, A_j)]
        (보수적 추정)

실행:
    .venv-qwen/bin/python experiments/run_pid_phase1.py
    .venv-qwen/bin/python experiments/run_pid_phase1.py experiments/configs/shapley_phase1.yaml
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.meta.simulator import AGENT_NAMES, VERDICTS, TRUE_LABELS

ALL_AGENTS: Tuple[str, ...] = tuple(AGENT_NAMES)
TRUE_LABEL_IDX = {l: i for i, l in enumerate(TRUE_LABELS)}
VERDICT_IDX = {v: i for i, v in enumerate(VERDICTS)}
EPS = 1e-12


# --------------------------------------------------------------------------- #
# 데이터 로드
# --------------------------------------------------------------------------- #
def load_records(path: str) -> List[Dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# --------------------------------------------------------------------------- #
# 경험적 확률 추정
# --------------------------------------------------------------------------- #
def estimate_joint_pya(
    records: List[Dict],
    agent: str,
) -> np.ndarray:
    """
    p(Y=y, A=a) 행렬 추정.
    shape: (|TRUE_LABELS|, |VERDICTS|)
    """
    n_y = len(TRUE_LABELS)
    n_a = len(VERDICTS)
    counts = np.zeros((n_y, n_a), dtype=np.float64)
    for rec in records:
        y_idx = TRUE_LABEL_IDX.get(rec["true_label"], -1)
        a_idx = VERDICT_IDX.get(rec["agent_verdicts"].get(agent, "uncertain"), -1)
        if y_idx >= 0 and a_idx >= 0:
            counts[y_idx, a_idx] += 1
    return counts / max(counts.sum(), 1)


def estimate_joint_pyab(
    records: List[Dict],
    agent_i: str,
    agent_j: str,
) -> np.ndarray:
    """
    p(Y=y, A_i=a, A_j=b) 3차원 텐서.
    shape: (|TRUE_LABELS|, |VERDICTS|, |VERDICTS|)
    """
    n_y = len(TRUE_LABELS)
    n_a = len(VERDICTS)
    counts = np.zeros((n_y, n_a, n_a), dtype=np.float64)
    for rec in records:
        y_idx = TRUE_LABEL_IDX.get(rec["true_label"], -1)
        ai_idx = VERDICT_IDX.get(rec["agent_verdicts"].get(agent_i, "uncertain"), -1)
        aj_idx = VERDICT_IDX.get(rec["agent_verdicts"].get(agent_j, "uncertain"), -1)
        if y_idx >= 0 and ai_idx >= 0 and aj_idx >= 0:
            counts[y_idx, ai_idx, aj_idx] += 1
    return counts / max(counts.sum(), 1)


# --------------------------------------------------------------------------- #
# 정보량 계산
# --------------------------------------------------------------------------- #
def mutual_information_single(p_ya: np.ndarray) -> float:
    """MI(Y; A) = Σ_{y,a} p(y,a) log[p(y,a) / (p(y) p(a))]"""
    p_y = p_ya.sum(axis=1, keepdims=True)   # (n_y, 1)
    p_a = p_ya.sum(axis=0, keepdims=True)   # (1, n_a)
    mask = p_ya > EPS
    mi = np.where(mask, p_ya * np.log(p_ya / (p_y * p_a + EPS) + EPS), 0.0).sum()
    return float(max(mi, 0.0))


def mutual_information_joint(p_yab: np.ndarray) -> float:
    """MI(Y; A_i, A_j) = Σ_{y,a,b} p(y,a,b) log[p(y,a,b) / (p(y) p(a,b))]"""
    p_y = p_yab.sum(axis=(1, 2), keepdims=True)   # (n_y, 1, 1)
    p_ab = p_yab.sum(axis=0, keepdims=True)        # (1, n_a, n_a)
    mask = p_yab > EPS
    mi = np.where(mask, p_yab * np.log(p_yab / (p_y * p_ab + EPS) + EPS), 0.0).sum()
    return float(max(mi, 0.0))


# --------------------------------------------------------------------------- #
# Specific information (Williams & Beer Imin)
# --------------------------------------------------------------------------- #
def specific_information(p_ya: np.ndarray) -> np.ndarray:
    """
    I_spec(y; A) = Σ_a p(a|y) log[p(y|a) / p(y)]
    shape: (n_y,)
    """
    p_y = p_ya.sum(axis=1)                        # (n_y,)
    p_ay = p_ya.T                                  # (n_a, n_y)
    p_a = p_ya.sum(axis=0)                         # (n_a,)
    # p(y|a) = p(y,a) / p(a)
    p_y_given_a = p_ay / (p_a[:, None] + EPS)     # (n_a, n_y)
    # p(a|y) = p(y,a) / p(y)
    p_a_given_y = p_ya / (p_y[:, None] + EPS)     # (n_y, n_a)

    i_spec = np.zeros(len(TRUE_LABELS))
    for y_idx in range(len(TRUE_LABELS)):
        if p_y[y_idx] < EPS:
            continue
        s = 0.0
        for a_idx in range(len(VERDICTS)):
            p_a_cond_y = p_a_given_y[y_idx, a_idx]
            p_y_cond_a = p_y_given_a[a_idx, y_idx]
            p_y_marg = p_y[y_idx]
            if p_a_cond_y > EPS and p_y_cond_a > EPS and p_y_marg > EPS:
                s += p_a_cond_y * np.log(p_y_cond_a / (p_y_marg + EPS) + EPS)
        i_spec[y_idx] = s
    return i_spec


def imin_redundancy(
    i_spec_i: np.ndarray,
    i_spec_j: np.ndarray,
    p_y: np.ndarray,
) -> float:
    """
    Imin(Y; A_i, A_j) = Σ_y p(y) · min(I_spec(y; A_i), I_spec(y; A_j))
    """
    imin_per_y = np.minimum(i_spec_i, i_spec_j)
    return float(np.sum(p_y * imin_per_y))


# --------------------------------------------------------------------------- #
# Pairwise PID
# --------------------------------------------------------------------------- #
def pairwise_pid(
    records: List[Dict],
    agent_i: str,
    agent_j: str,
) -> Dict:
    """
    Y; (A_i, A_j) pairwise PID.
    반환: unique_i, unique_j, redundancy, synergy, co_information, mi_i, mi_j, mi_joint
    """
    p_ya_i = estimate_joint_pya(records, agent_i)
    p_ya_j = estimate_joint_pya(records, agent_j)
    p_yab = estimate_joint_pyab(records, agent_i, agent_j)

    p_y = p_ya_i.sum(axis=1)  # (n_y,)

    mi_i = mutual_information_single(p_ya_i)
    mi_j = mutual_information_single(p_ya_j)
    mi_joint = mutual_information_joint(p_yab)

    # Specific information per label
    is_i = specific_information(p_ya_i)
    is_j = specific_information(p_ya_j)

    # Imin redundancy
    red = imin_redundancy(is_i, is_j, p_y)
    red = max(red, 0.0)

    # Unique information
    unique_i = max(mi_i - red, 0.0)
    unique_j = max(mi_j - red, 0.0)

    # Synergy
    synergy = mi_joint - mi_i - mi_j + red
    # 수치 오차 클리핑
    synergy = max(synergy, -abs(mi_joint) * 0.01)

    # Co-information (interaction information)
    co_info = mi_i + mi_j - mi_joint

    return {
        "mi_i": mi_i,
        "mi_j": mi_j,
        "mi_joint": mi_joint,
        "redundancy": red,
        "unique_i": unique_i,
        "unique_j": unique_j,
        "synergy": synergy,
        "co_information": co_info,
        "specific_info_i": is_i.tolist(),
        "specific_info_j": is_j.tolist(),
    }


# --------------------------------------------------------------------------- #
# 4-agent 요약: 에이전트별 Unique 추정
# --------------------------------------------------------------------------- #
def agent_unique_summary(
    pair_results: Dict[Tuple[str, str], Dict],
    agents: Tuple[str, ...],
) -> Dict[str, float]:
    """
    각 에이전트의 Unique 정보 = min over 모든 파트너 j에 대한 unique_i
    (가장 보수적 추정: 어떤 파트너와도 중복되지 않는 정보량)
    """
    unique_summary = {}
    for agent in agents:
        uniques = []
        for (ai, aj), res in pair_results.items():
            if ai == agent:
                uniques.append(res["unique_i"])
            elif aj == agent:
                uniques.append(res["unique_j"])
        unique_summary[agent] = float(min(uniques)) if uniques else 0.0
    return unique_summary


# --------------------------------------------------------------------------- #
# 메인
# --------------------------------------------------------------------------- #
def load_config(config_path: Optional[str]) -> Dict:
    if config_path and Path(config_path).exists():
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return cfg
    return {
        "jsonl_path": "experiments/results/phase2_patha_scale500_gain_predictor/patha_agent_outputs_20260304_080157.jsonl",
        "output": {"save_dir": "experiments/results/shapley_phase1"},
    }


def run(config_path: Optional[str] = None) -> Dict:
    cfg = load_config(config_path)
    jsonl_path = ROOT / cfg["jsonl_path"]
    save_dir = ROOT / cfg["output"]["save_dir"]
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"[PID Phase 1.4] 데이터 로드: {jsonl_path}")
    records = load_records(str(jsonl_path))
    print(f"  → {len(records)}개 레코드")

    # ------------------------------------------------------------------ #
    # Step 1: 개별 MI
    # ------------------------------------------------------------------ #
    print("\n[Step 1] 에이전트별 MI(Y; A_i)")
    mi_individual = {}
    for agent in ALL_AGENTS:
        p_ya = estimate_joint_pya(records, agent)
        mi = mutual_information_single(p_ya)
        mi_individual[agent] = mi
        print(f"  MI(Y; {agent:12s}) = {mi:.4f} nats")

    # ------------------------------------------------------------------ #
    # Step 2: Pairwise PID (6쌍)
    # ------------------------------------------------------------------ #
    print("\n[Step 2] Pairwise PID (Williams & Beer Imin)")
    print(f"  {'Pair':30s}  {'MI_i':6s}  {'MI_j':6s}  {'MI_joint':8s}  "
          f"{'Redund':8s}  {'Unique_i':8s}  {'Unique_j':8s}  {'Synergy':8s}  {'CoInfo':8s}")
    print("  " + "-" * 100)

    pair_results: Dict[Tuple[str, str], Dict] = {}
    for ai, aj in combinations(ALL_AGENTS, 2):
        res = pairwise_pid(records, ai, aj)
        pair_results[(ai, aj)] = res
        print(
            f"  ({ai:10s}, {aj:10s})  "
            f"{res['mi_i']:6.4f}  {res['mi_j']:6.4f}  {res['mi_joint']:8.4f}  "
            f"{res['redundancy']:8.4f}  {res['unique_i']:8.4f}  {res['unique_j']:8.4f}  "
            f"{res['synergy']:8.4f}  {res['co_information']:8.4f}"
        )

    # ------------------------------------------------------------------ #
    # Step 3: 에이전트별 Unique 요약
    # ------------------------------------------------------------------ #
    print("\n[Step 3] 에이전트별 Unique 정보 (보수적 추정)")
    unique_summary = agent_unique_summary(pair_results, ALL_AGENTS)
    for agent, val in sorted(unique_summary.items(), key=lambda x: -x[1]):
        print(f"  Unique({agent:12s}) = {val:.4f} nats")

    # ------------------------------------------------------------------ #
    # Step 4: 시너지/중복 순위
    # ------------------------------------------------------------------ #
    print("\n[Step 4] Synergy 순위 (높을수록 함께 쓸 때 추가 가치)")
    synergy_sorted = sorted(pair_results.items(), key=lambda x: -x[1]["synergy"])
    for (ai, aj), res in synergy_sorted:
        print(f"  ({ai:10s}, {aj:10s})  synergy={res['synergy']:+.4f}  "
              f"co_info={res['co_information']:+.4f}")

    print("\n[Step 5] Co-Information 순위 (양수→중복, 음수→시너지)")
    co_sorted = sorted(pair_results.items(), key=lambda x: x[1]["co_information"])
    for (ai, aj), res in co_sorted:
        label = "시너지 우세" if res["co_information"] < 0 else "중복 우세"
        print(f"  ({ai:10s}, {aj:10s})  co_info={res['co_information']:+.4f}  [{label}]")

    # ------------------------------------------------------------------ #
    # 저장
    # ------------------------------------------------------------------ #
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = save_dir / f"pid_phase1_{timestamp}.json"

    serializable_pairs = {
        f"{ai}__{aj}": res
        for (ai, aj), res in pair_results.items()
    }

    output = {
        "timestamp": datetime.now().isoformat(),
        "n_samples": len(records),
        "mi_individual": mi_individual,
        "pairwise_pid": serializable_pairs,
        "unique_summary": unique_summary,
        "synergy_ranking": [
            {
                "pair": f"{ai}__{aj}",
                "synergy": res["synergy"],
                "co_information": res["co_information"],
                "redundancy": res["redundancy"],
                "unique_i": res["unique_i"],
                "unique_j": res["unique_j"],
            }
            for (ai, aj), res in synergy_sorted
        ],
        "summary": _build_summary(pair_results, unique_summary, mi_individual),
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n[완료] 결과 저장: {out_path}")
    _print_summary(output["summary"])

    return output


def _build_summary(
    pair_results: Dict,
    unique_summary: Dict,
    mi_individual: Dict,
) -> Dict:
    # 가장 시너지 높은 쌍
    top_synergy_pair = max(pair_results.items(), key=lambda x: x[1]["synergy"])
    # 가장 중복 높은 쌍
    top_redundant_pair = max(pair_results.items(), key=lambda x: x[1]["redundancy"])
    # 가장 Unique 정보 많은 에이전트 (경량화 금지)
    top_unique_agent = max(unique_summary.items(), key=lambda x: x[1])
    # 가장 Unique 정보 적은 에이전트 (경량화 후보)
    bottom_unique_agent = min(unique_summary.items(), key=lambda x: x[1])

    ai, aj = top_synergy_pair[0]
    ri, rj = top_redundant_pair[0]
    return {
        "top_synergy_pair": f"{ai}__{aj}",
        "top_synergy_value": top_synergy_pair[1]["synergy"],
        "top_redundant_pair": f"{ri}__{rj}",
        "top_redundant_value": top_redundant_pair[1]["redundancy"],
        "highest_unique_agent": top_unique_agent[0],
        "highest_unique_value": top_unique_agent[1],
        "lowest_unique_agent": bottom_unique_agent[0],
        "lowest_unique_value": bottom_unique_agent[1],
    }


def _print_summary(summary: Dict) -> None:
    print("\n" + "=" * 65)
    print("  SHIELD Phase 1.4 PID 결과 요약")
    print("=" * 65)
    print(f"  최고 시너지 쌍 : {summary['top_synergy_pair']} "
          f"(synergy={summary['top_synergy_value']:+.4f})")
    print(f"  최고 중복 쌍   : {summary['top_redundant_pair']} "
          f"(redundancy={summary['top_redundant_value']:.4f})")
    print(f"  Unique 최고 에이전트 (제거 불가) : {summary['highest_unique_agent']} "
          f"(unique={summary['highest_unique_value']:.4f})")
    print(f"  Unique 최저 에이전트 (경량화 우선): {summary['lowest_unique_agent']} "
          f"(unique={summary['lowest_unique_value']:.4f})")
    print("=" * 65)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SHIELD Phase 1.4: PID")
    parser.add_argument("config", nargs="?", default=None, help="YAML config path (선택)")
    args = parser.parse_args()
    run(args.config)
