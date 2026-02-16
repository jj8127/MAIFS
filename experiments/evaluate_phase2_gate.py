"""
Path A Phase2 의사결정 게이트 평가기.

목적:
    - 멀티시드 summary 파일을 기준으로 합격/불합격을 자동 판정
    - baseline 대비 개선 여부를 일관된 기준으로 기록

예시:
    .venv-qwen/bin/python experiments/evaluate_phase2_gate.py \
      experiments/results/phase2_patha_scale120_oracle_p15_ls005/summary_10seeds_42_51.json \
      --baseline-summary experiments/results/phase2_patha_scale120/phase2_patha_multiseed_summary_scale120_10seeds_42_51_20260216.json \
      --min-runs 10 \
      --min-f1-diff-mean 0.01 \
      --min-significant-count 3 \
      --min-improvement-over-baseline 0.0 \
      --out experiments/results/phase2_patha_scale120_oracle_p15_ls005/gate_report_seed10.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from math import comb, erfc, sqrt
from pathlib import Path
from typing import Any, Dict, Optional


def _load_json(path: str) -> Dict[str, Any]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate PathA Phase2 gate criteria from summary json")
    p.add_argument("candidate_summary", type=str, help="Candidate multiseed summary json path")
    p.add_argument(
        "--baseline-summary",
        type=str,
        default="",
        help="Optional baseline multiseed summary json path",
    )
    p.add_argument("--min-runs", type=int, default=10, help="Required minimum number of runs")
    p.add_argument(
        "--min-f1-diff-mean",
        type=float,
        default=0.01,
        help="Required minimum mean(Phase2-Phase1) F1",
    )
    p.add_argument(
        "--min-significant-count",
        type=int,
        default=3,
        help="Required minimum number of significant runs",
    )
    p.add_argument(
        "--min-improvement-over-baseline",
        type=float,
        default=0.0,
        help="Required minimum (candidate_mean - baseline_mean)",
    )
    p.add_argument(
        "--min-positive-seed-count",
        type=int,
        default=0,
        help="Required minimum #seeds with positive F1 diff",
    )
    p.add_argument(
        "--max-sign-test-pvalue",
        type=float,
        default=None,
        help="Optional upper bound for exact sign-test p-value",
    )
    p.add_argument(
        "--require-pooled-mcnemar-significant",
        action="store_true",
        help="Require pooled McNemar p < 0.05 (needs run-level b/c counts)",
    )
    p.add_argument(
        "--max-pooled-mcnemar-pvalue",
        type=float,
        default=None,
        help="Optional upper bound for pooled McNemar p-value",
    )
    p.add_argument("--out", type=str, default="", help="Optional output report json path")
    return p.parse_args()


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
        runs = summary.get("runs", [])
        diffs = [float(r.get("f1_diff", 0.0)) for r in runs]
        pos = int(sum(1 for d in diffs if d > 0.0))
        neg = int(sum(1 for d in diffs if d < 0.0))
        zero = int(sum(1 for d in diffs if d == 0.0))
        pval = _exact_sign_test_two_sided(pos, neg)
    else:
        pos = int(pos)
        neg = int(neg)
        zero = int(zero)
        if pval is None:
            pval = _exact_sign_test_two_sided(pos, neg)
        else:
            pval = float(pval)

    return {
        "positive_count": pos,
        "negative_count": neg,
        "zero_count": zero,
        "sign_test_pvalue": float(pval),
    }


def _extract_pooled_mcnemar(summary: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    agg = summary.get("aggregate", {})
    if "pooled_mcnemar_pvalue" in agg:
        return {
            "b": int(agg.get("pooled_mcnemar_b", 0)),
            "c": int(agg.get("pooled_mcnemar_c", 0)),
            "statistic": float(agg.get("pooled_mcnemar_statistic", 0.0)),
            "pvalue": float(agg.get("pooled_mcnemar_pvalue", 1.0)),
            "significant": bool(agg.get("pooled_mcnemar_significant", False)),
            "source": "aggregate",
        }

    runs = summary.get("runs", [])
    b_total = 0
    c_total = 0
    used = 0
    for r in runs:
        b = r.get("mcnemar_b")
        c = r.get("mcnemar_c")
        if b is None or c is None:
            continue
        b_total += int(b)
        c_total += int(c)
        used += 1

    if used == 0:
        return None
    if b_total + c_total == 0:
        return {
            "b": int(b_total),
            "c": int(c_total),
            "statistic": 0.0,
            "pvalue": 1.0,
            "significant": False,
            "source": "runs",
        }

    statistic = (abs(b_total - c_total) - 1) ** 2 / (b_total + c_total)
    pvalue = float(erfc(sqrt(statistic / 2.0)))
    return {
        "b": int(b_total),
        "c": int(c_total),
        "statistic": float(statistic),
        "pvalue": pvalue,
        "significant": bool(pvalue < 0.05),
        "source": "runs",
    }


def evaluate_gate(
    candidate: Dict[str, Any],
    baseline: Optional[Dict[str, Any]] = None,
    *,
    min_runs: int = 10,
    min_f1_diff_mean: float = 0.01,
    min_significant_count: int = 3,
    min_improvement_over_baseline: float = 0.0,
    min_positive_seed_count: int = 0,
    max_sign_test_pvalue: Optional[float] = None,
    require_pooled_mcnemar_significant: bool = False,
    max_pooled_mcnemar_pvalue: Optional[float] = None,
) -> Dict[str, Any]:
    """
    candidate/baseline summary dict를 기준으로 게이트 판정을 수행한다.
    """
    cand_agg = candidate.get("aggregate", {})
    base_agg = baseline.get("aggregate", {}) if baseline is not None else None

    n_runs = int(candidate.get("n_runs", 0))
    cand_mean = float(cand_agg.get("f1_diff_mean", 0.0))
    cand_sig = int(cand_agg.get("significant_count", 0))
    sign_stats = _extract_sign_stats(candidate)
    pooled_mcnemar = _extract_pooled_mcnemar(candidate)

    checks = []
    checks.append(
        {
            "name": "min_runs",
            "pass": n_runs >= int(min_runs),
            "actual": n_runs,
            "required": int(min_runs),
        }
    )
    checks.append(
        {
            "name": "min_f1_diff_mean",
            "pass": cand_mean >= float(min_f1_diff_mean),
            "actual": cand_mean,
            "required": float(min_f1_diff_mean),
        }
    )
    checks.append(
        {
            "name": "min_significant_count",
            "pass": cand_sig >= int(min_significant_count),
            "actual": cand_sig,
            "required": int(min_significant_count),
        }
    )
    if int(min_positive_seed_count) > 0:
        checks.append(
            {
                "name": "min_positive_seed_count",
                "pass": int(sign_stats["positive_count"]) >= int(min_positive_seed_count),
                "actual": int(sign_stats["positive_count"]),
                "required": int(min_positive_seed_count),
            }
        )
    if max_sign_test_pvalue is not None:
        checks.append(
            {
                "name": "max_sign_test_pvalue",
                "pass": float(sign_stats["sign_test_pvalue"]) <= float(max_sign_test_pvalue),
                "actual": float(sign_stats["sign_test_pvalue"]),
                "required": float(max_sign_test_pvalue),
            }
        )

    baseline_delta = None
    if base_agg is not None:
        base_mean = float(base_agg.get("f1_diff_mean", 0.0))
        baseline_delta = cand_mean - base_mean
        checks.append(
            {
                "name": "min_improvement_over_baseline",
                "pass": baseline_delta >= float(min_improvement_over_baseline),
                "actual": baseline_delta,
                "required": float(min_improvement_over_baseline),
            }
        )

    if require_pooled_mcnemar_significant:
        checks.append(
            {
                "name": "require_pooled_mcnemar_significant",
                "pass": bool(pooled_mcnemar and pooled_mcnemar.get("significant", False)),
                "actual": None if pooled_mcnemar is None else bool(pooled_mcnemar.get("significant", False)),
                "required": True,
            }
        )
    if max_pooled_mcnemar_pvalue is not None:
        checks.append(
            {
                "name": "max_pooled_mcnemar_pvalue",
                "pass": bool(
                    pooled_mcnemar is not None
                    and float(pooled_mcnemar.get("pvalue", 1.0)) <= float(max_pooled_mcnemar_pvalue)
                ),
                "actual": None if pooled_mcnemar is None else float(pooled_mcnemar.get("pvalue", 1.0)),
                "required": float(max_pooled_mcnemar_pvalue),
            }
        )

    gate_pass = all(bool(c["pass"]) for c in checks)
    return {
        "criteria": {
            "min_runs": int(min_runs),
            "min_f1_diff_mean": float(min_f1_diff_mean),
            "min_significant_count": int(min_significant_count),
            "min_improvement_over_baseline": float(min_improvement_over_baseline),
            "min_positive_seed_count": int(min_positive_seed_count),
            "max_sign_test_pvalue": max_sign_test_pvalue,
            "require_pooled_mcnemar_significant": bool(require_pooled_mcnemar_significant),
            "max_pooled_mcnemar_pvalue": max_pooled_mcnemar_pvalue,
        },
        "candidate": {
            "n_runs": n_runs,
            "aggregate": cand_agg,
            "sign_stats": sign_stats,
            "pooled_mcnemar": pooled_mcnemar,
        },
        "baseline": {
            "aggregate": base_agg,
        }
        if base_agg is not None
        else None,
        "baseline_delta_f1_diff_mean": baseline_delta,
        "checks": checks,
        "gate_pass": gate_pass,
    }


def main() -> None:
    args = parse_args()
    candidate = _load_json(args.candidate_summary)
    baseline = _load_json(args.baseline_summary) if args.baseline_summary else None
    report = evaluate_gate(
        candidate=candidate,
        baseline=baseline,
        min_runs=int(args.min_runs),
        min_f1_diff_mean=float(args.min_f1_diff_mean),
        min_significant_count=int(args.min_significant_count),
        min_improvement_over_baseline=float(args.min_improvement_over_baseline),
        min_positive_seed_count=int(args.min_positive_seed_count),
        max_sign_test_pvalue=args.max_sign_test_pvalue,
        require_pooled_mcnemar_significant=bool(args.require_pooled_mcnemar_significant),
        max_pooled_mcnemar_pvalue=args.max_pooled_mcnemar_pvalue,
    )
    report = {
        "timestamp": datetime.now().isoformat(),
        "candidate_summary": args.candidate_summary,
        "baseline_summary": args.baseline_summary or None,
        **report,
    }

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print("[Gate] report file:", out_path)

    print("[Gate] pass:", report["gate_pass"])
    for c in report["checks"]:
        status = "PASS" if c["pass"] else "FAIL"
        print(f"  - {status:4} {c['name']}: actual={c['actual']} required={c['required']}")


if __name__ == "__main__":
    main()
