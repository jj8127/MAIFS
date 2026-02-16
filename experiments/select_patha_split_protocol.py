"""
Path A split protocol 후보 비교 및 기본값 추천기.

입력:
    - multiseed/repeated summary json 여러 개
출력:
    - 후보별 핵심 지표 추출
    - 결정 규칙(lexicographic)에 따른 추천 순위

예시:
    .venv-qwen/bin/python experiments/select_patha_split_protocol.py \
      --candidate random25:experiments/results/phase2_patha_scale120_feat_enhanced36_ridge/repeated_random_summary_25runs_300_324_20260216.json \
      --candidate kfold25:experiments/results/phase2_patha_scale120_feat_enhanced36_ridge/fixed_kfold_summary_25runs_5seeds_20260216.json \
      --out experiments/results/phase2_patha_scale120_feat_enhanced36_ridge/split_protocol_selection_20260216.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Select default PathA split protocol from summary candidates")
    p.add_argument(
        "--candidate",
        action="append",
        default=[],
        help="Format: label:path_to_summary_json (repeatable)",
    )
    p.add_argument("--out", type=str, default="", help="Optional output report json path")
    return p.parse_args()


def _load_json(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _parse_candidate_token(token: str) -> Tuple[str, str]:
    t = token.strip()
    if ":" not in t:
        raise ValueError(f"invalid candidate token (missing ':'): {token}")
    label, path = t.split(":", 1)
    label = label.strip()
    path = path.strip()
    if not label or not path:
        raise ValueError(f"invalid candidate token: {token}")
    return label, path


def _extract_metrics(label: str, path: str, summary: Dict[str, Any]) -> Dict[str, Any]:
    agg = summary.get("aggregate", {})
    n_runs = int(summary.get("n_runs", 0))
    pos = int(agg.get("positive_count", 0))
    neg = int(agg.get("negative_count", 0))
    zero = int(agg.get("zero_count", 0))
    directional_margin = int(pos - neg)
    return {
        "label": label,
        "summary_path": path,
        "n_runs": n_runs,
        "f1_diff_mean": float(agg.get("f1_diff_mean", 0.0)),
        "f1_diff_std": float(agg.get("f1_diff_std", 0.0)),
        "positive_count": pos,
        "negative_count": neg,
        "zero_count": zero,
        "directional_margin": directional_margin,
        "sign_test_pvalue": float(agg.get("sign_test_pvalue", 1.0)),
        "pooled_mcnemar_pvalue": float(agg.get("pooled_mcnemar_pvalue", 1.0))
        if agg.get("pooled_mcnemar_pvalue") is not None
        else 1.0,
    }


def _rank_key(row: Dict[str, Any]) -> Tuple[float, float, float, float, float]:
    """
    우선순위:
        1) f1_diff_mean 높을수록 좋음
        2) sign_test_pvalue 낮을수록 좋음
        3) directional_margin(pos-neg) 높을수록 좋음
        4) f1_diff_std 낮을수록 좋음
        5) pooled_mcnemar_pvalue 낮을수록 좋음
    """
    return (
        float(row["f1_diff_mean"]),
        -float(row["sign_test_pvalue"]),
        float(row["directional_margin"]),
        -float(row["f1_diff_std"]),
        -float(row["pooled_mcnemar_pvalue"]),
    )


def _recommend(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not candidates:
        raise ValueError("no candidates")
    ranked = sorted(candidates, key=_rank_key, reverse=True)
    winner = ranked[0]
    return {
        "recommended_label": winner["label"],
        "recommended_summary_path": winner["summary_path"],
        "ranking": [r["label"] for r in ranked],
        "winner_metrics": winner,
        "decision_rule": {
            "type": "lexicographic",
            "order": [
                "max f1_diff_mean",
                "min sign_test_pvalue",
                "max directional_margin (positive_count - negative_count)",
                "min f1_diff_std",
                "min pooled_mcnemar_pvalue",
            ],
        },
    }


def main() -> None:
    args = parse_args()
    if not args.candidate:
        raise ValueError("at least one --candidate is required")

    rows: List[Dict[str, Any]] = []
    for token in args.candidate:
        label, path = _parse_candidate_token(token)
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"candidate summary not found: {path}")
        summary = _load_json(path)
        rows.append(_extract_metrics(label=label, path=path, summary=summary))

    recommendation = _recommend(rows)
    payload = {
        "timestamp": datetime.now().isoformat(),
        "candidates": rows,
        "recommendation": recommendation,
    }

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print("[ProtocolSelect] report file:", out_path)

    print("[ProtocolSelect] ranking:", recommendation["ranking"])
    print("[ProtocolSelect] recommended:", recommendation["recommended_label"])


if __name__ == "__main__":
    main()
