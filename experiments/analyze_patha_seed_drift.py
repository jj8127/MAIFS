"""
Path A 멀티시드 블록 간 드리프트 분석 리포트 생성기.

목적:
    - 두 개의 multiseed summary를 비교해 성능 방향 전환/불안정성 신호를 정리
    - aggregate 지표 + 시드별 결과 + 모델쌍 분포 + (옵션) subgroup 변화까지 한 번에 저장

예시:
    .venv-qwen/bin/python experiments/analyze_patha_seed_drift.py \
      experiments/results/phase2_patha_scale120_feat_enhanced36_ridge/summary_10seeds_42_51_statsv2_20260216.json \
      experiments/results/phase2_patha_scale120_feat_enhanced36_ridge/summary_10seeds_52_61_statsv2_20260216.json \
      --reference-label seeds_42_51 \
      --candidate-label seeds_52_61 \
      --reference-subgroup experiments/results/phase2_patha_scale120_feat_enhanced36_ridge/subgroup_summary_10seeds_statsv2_20260216.json \
      --candidate-subgroup experiments/results/phase2_patha_scale120_feat_enhanced36_ridge/subgroup_summary_10seeds_52_61_statsv2_20260216.json \
      --out experiments/results/phase2_patha_scale120_feat_enhanced36_ridge/seed_drift_42_51_vs_52_61_20260216.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from math import comb
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare two PathA multiseed summaries for seed-block drift")
    p.add_argument("reference_summary", type=str, help="Reference summary json path")
    p.add_argument("candidate_summary", type=str, help="Candidate summary json path")
    p.add_argument("--reference-label", type=str, default="reference", help="Label for reference block")
    p.add_argument("--candidate-label", type=str, default="candidate", help="Label for candidate block")
    p.add_argument("--reference-subgroup", type=str, default="", help="Optional reference subgroup summary json")
    p.add_argument("--candidate-subgroup", type=str, default="", help="Optional candidate subgroup summary json")
    p.add_argument("--out", type=str, default="", help="Optional output report json path")
    return p.parse_args()


def _load_json(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _exact_sign_test_two_sided(pos_count: int, neg_count: int) -> float:
    n = int(pos_count + neg_count)
    if n <= 0:
        return 1.0
    k = int(min(pos_count, neg_count))
    cdf_tail = sum(comb(n, i) for i in range(k + 1)) / float(2**n)
    return float(min(1.0, 2.0 * cdf_tail))


def _extract_sign_stats(summary: Dict[str, Any]) -> Dict[str, Any]:
    agg = summary.get("aggregate", {})
    pos = agg.get("positive_count")
    neg = agg.get("negative_count")
    zero = agg.get("zero_count")
    pval = agg.get("sign_test_pvalue")

    if pos is None or neg is None or zero is None:
        diffs = [float(r.get("f1_diff", 0.0)) for r in summary.get("runs", [])]
        pos = int(sum(1 for d in diffs if d > 0.0))
        neg = int(sum(1 for d in diffs if d < 0.0))
        zero = int(sum(1 for d in diffs if d == 0.0))
    else:
        pos = int(pos)
        neg = int(neg)
        zero = int(zero)

    if pval is None:
        pval = _exact_sign_test_two_sided(pos, neg)
    else:
        pval = float(pval)

    return {
        "positive_count": int(pos),
        "negative_count": int(neg),
        "zero_count": int(zero),
        "sign_test_pvalue": float(pval),
    }


def _extract_f1_diff_stats(summary: Dict[str, Any]) -> Dict[str, Any]:
    diffs = np.array([float(r.get("f1_diff", 0.0)) for r in summary.get("runs", [])], dtype=float)
    if diffs.size == 0:
        return {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "q25": 0.0,
            "median": 0.0,
            "q75": 0.0,
            "max": 0.0,
            "sorted": [],
        }
    return {
        "count": int(diffs.size),
        "mean": float(np.mean(diffs)),
        "std": float(np.std(diffs, ddof=0)),
        "min": float(np.min(diffs)),
        "q25": float(np.quantile(diffs, 0.25)),
        "median": float(np.quantile(diffs, 0.5)),
        "q75": float(np.quantile(diffs, 0.75)),
        "max": float(np.max(diffs)),
        "sorted": [float(x) for x in sorted(diffs.tolist())],
    }


def _model_pair_counts(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    counter = Counter()
    for run in summary.get("runs", []):
        p1 = str(run.get("phase1_best", ""))
        p2 = str(run.get("phase2_best", ""))
        counter[(p1, p2)] += 1

    rows: List[Dict[str, Any]] = []
    for (p1, p2), count in sorted(counter.items(), key=lambda x: (-x[1], x[0][0], x[0][1])):
        rows.append({"phase1_best": p1, "phase2_best": p2, "count": int(count)})
    return rows


def _seed_rows(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for run in sorted(summary.get("runs", []), key=lambda x: int(x.get("seed", 0))):
        rows.append(
            {
                "seed": int(run.get("seed", 0)),
                "f1_diff": float(run.get("f1_diff", 0.0)),
                "phase1_best": str(run.get("phase1_best", "")),
                "phase2_best": str(run.get("phase2_best", "")),
                "mcnemar_pvalue": float(run.get("mcnemar_pvalue", 1.0)),
                "significant": bool(run.get("significant", False)),
            }
        )
    return rows


def describe_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    agg = summary.get("aggregate", {})
    return {
        "n_runs": int(summary.get("n_runs", 0)),
        "seeds": [int(s) for s in summary.get("seeds", [])],
        "aggregate": {
            "f1_diff_mean": float(agg.get("f1_diff_mean", 0.0)),
            "f1_diff_std": float(agg.get("f1_diff_std", 0.0)),
            "f1_diff_min": float(agg.get("f1_diff_min", 0.0)),
            "f1_diff_max": float(agg.get("f1_diff_max", 0.0)),
            "significant_count": int(agg.get("significant_count", 0)),
            "pooled_mcnemar_pvalue": (
                None if agg.get("pooled_mcnemar_pvalue") is None else float(agg.get("pooled_mcnemar_pvalue"))
            ),
        },
        "sign_stats": _extract_sign_stats(summary),
        "f1_diff_distribution": _extract_f1_diff_stats(summary),
        "model_pair_counts": _model_pair_counts(summary),
        "runs": _seed_rows(summary),
    }


def _compare_numeric_metric_maps(
    reference: Dict[str, Any],
    candidate: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for key in sorted(set(reference.keys()) | set(candidate.keys())):
        r = reference.get(key)
        c = candidate.get(key)
        is_num = isinstance(r, (int, float)) and isinstance(c, (int, float))
        out[key] = {
            "reference": r,
            "candidate": c,
            "delta_candidate_minus_reference": float(c - r) if is_num else None,
        }
    return out


def _compare_named_metric_map(
    reference: Dict[str, float],
    candidate: Dict[str, float],
) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for key in sorted(set(reference.keys()) | set(candidate.keys())):
        r = reference.get(key)
        c = candidate.get(key)
        if r is None or c is None:
            out[key] = {
                "reference": r,
                "candidate": c,
                "delta_candidate_minus_reference": None,
                "sign_flip": None,
            }
            continue
        r_f = float(r)
        c_f = float(c)
        out[key] = {
            "reference": r_f,
            "candidate": c_f,
            "delta_candidate_minus_reference": float(c_f - r_f),
            "sign_flip": bool(r_f * c_f < 0.0),
        }
    return out


def compare_subgroup(
    reference_subgroup: Dict[str, Any],
    candidate_subgroup: Dict[str, Any],
) -> Dict[str, Any]:
    ref_delta = reference_subgroup.get("delta_phase2_minus_phase1", {})
    cand_delta = candidate_subgroup.get("delta_phase2_minus_phase1", {})
    ref_sub = ref_delta.get("by_sub_type_macro_f1", {})
    cand_sub = cand_delta.get("by_sub_type_macro_f1", {})
    ref_lbl = ref_delta.get("by_true_label_macro_f1", {})
    cand_lbl = cand_delta.get("by_true_label_macro_f1", {})
    return {
        "by_sub_type_macro_f1": _compare_named_metric_map(ref_sub, cand_sub),
        "by_true_label_macro_f1": _compare_named_metric_map(ref_lbl, cand_lbl),
    }


def _build_risk_flags(reference_desc: Dict[str, Any], candidate_desc: Dict[str, Any]) -> List[str]:
    flags: List[str] = []
    ref_agg = reference_desc.get("aggregate", {})
    cand_agg = candidate_desc.get("aggregate", {})
    ref_sign = reference_desc.get("sign_stats", {})
    cand_sign = candidate_desc.get("sign_stats", {})

    if float(cand_agg.get("f1_diff_mean", 0.0)) < float(ref_agg.get("f1_diff_mean", 0.0)):
        flags.append("candidate_f1_diff_mean_dropped_vs_reference")
    if int(cand_sign.get("positive_count", 0)) < int(cand_sign.get("negative_count", 0)):
        flags.append("candidate_negative_seeds_exceed_positive")
    if float(cand_sign.get("sign_test_pvalue", 1.0)) > float(ref_sign.get("sign_test_pvalue", 1.0)):
        flags.append("candidate_directional_consistency_weaker_than_reference")
    ref_pool = ref_agg.get("pooled_mcnemar_pvalue")
    cand_pool = cand_agg.get("pooled_mcnemar_pvalue")
    if isinstance(ref_pool, (int, float)) and isinstance(cand_pool, (int, float)) and float(cand_pool) > float(ref_pool):
        flags.append("candidate_pooled_mcnemar_weaker_than_reference")
    return flags


def main() -> None:
    args = parse_args()
    ref_summary = _load_json(args.reference_summary)
    cand_summary = _load_json(args.candidate_summary)

    ref_desc = describe_summary(ref_summary)
    cand_desc = describe_summary(cand_summary)

    report: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "reference_label": args.reference_label,
        "candidate_label": args.candidate_label,
        "reference_summary_path": args.reference_summary,
        "candidate_summary_path": args.candidate_summary,
        "reference": ref_desc,
        "candidate": cand_desc,
        "comparison": {
            "aggregate": _compare_numeric_metric_maps(ref_desc["aggregate"], cand_desc["aggregate"]),
            "sign_stats": _compare_numeric_metric_maps(ref_desc["sign_stats"], cand_desc["sign_stats"]),
            "risk_flags": _build_risk_flags(ref_desc, cand_desc),
        },
    }

    if args.reference_subgroup and args.candidate_subgroup:
        ref_sub = _load_json(args.reference_subgroup)
        cand_sub = _load_json(args.candidate_subgroup)
        report["reference_subgroup_path"] = args.reference_subgroup
        report["candidate_subgroup_path"] = args.candidate_subgroup
        report["comparison"]["subgroup"] = compare_subgroup(ref_sub, cand_sub)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print("[SeedDrift] report file:", out_path)

    ref_m = ref_desc["aggregate"]["f1_diff_mean"]
    cand_m = cand_desc["aggregate"]["f1_diff_mean"]
    ref_sign = ref_desc["sign_stats"]
    cand_sign = cand_desc["sign_stats"]
    print(
        "[SeedDrift] f1_diff_mean:",
        f"{args.reference_label}={ref_m:+.6f}",
        f"{args.candidate_label}={cand_m:+.6f}",
        f"delta={cand_m - ref_m:+.6f}",
    )
    print(
        "[SeedDrift] sign stats:",
        f"{args.reference_label}(+/-/0={ref_sign['positive_count']}/{ref_sign['negative_count']}/{ref_sign['zero_count']},"
        f" p={ref_sign['sign_test_pvalue']:.6f})",
        f"{args.candidate_label}(+/-/0={cand_sign['positive_count']}/{cand_sign['negative_count']}/{cand_sign['zero_count']},"
        f" p={cand_sign['sign_test_pvalue']:.6f})",
    )
    print("[SeedDrift] risk flags:", report["comparison"]["risk_flags"])


if __name__ == "__main__":
    main()
