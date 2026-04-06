#!/usr/bin/env python3
"""
Summarize 3-case study metrics across multiple repeated-split summaries.

Input:
  repeated_split_summary_*.json files (one per dataset composition)

Output:
  - JSON report with per-dataset and overall aggregate stats
  - Markdown table for paper-ready quick view
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from math import comb
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
from scipy.stats import ttest_rel, wilcoxon


DEFAULT_SUMMARIES = [
    "experiments/results/phase2_patha_case3_scale300_dsA/repeated_split_summary_10runs_300_309_20260304.json",
    "experiments/results/phase2_patha_case3_scale300_dsB/repeated_split_summary_10runs_300_309_20260304.json",
    "experiments/results/phase2_patha_case3_scale300_dsC/repeated_split_summary_10runs_300_309_20260304.json",
    "experiments/results/phase2_patha_case3_scale300_dsD/repeated_split_summary_10runs_300_309_20260304.json",
]

DEFAULT_LABELS = [
    "DS-A_Au_Tp_BigGANai",
    "DS-B_Nature_Tp_BigGANai",
    "DS-C_Au_IMD_BigGANai",
    "DS-D_Nature_IMD_BigGANai",
]


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _std(vals: Sequence[float]) -> float:
    arr = np.asarray(vals, dtype=float)
    if arr.size <= 1:
        return 0.0
    return float(np.std(arr, ddof=1))


def _exact_sign_test_two_sided(pos_count: int, neg_count: int) -> float:
    n = int(pos_count + neg_count)
    if n <= 0:
        return 1.0
    k = int(min(pos_count, neg_count))
    cdf_tail = sum(comb(n, i) for i in range(k + 1)) / float(2**n)
    return float(min(1.0, 2.0 * cdf_tail))


def _paired_stats(a: Sequence[float], b: Sequence[float]) -> Dict[str, Any]:
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    diff = a_arr - b_arr
    pos = int(np.sum(diff > 0))
    neg = int(np.sum(diff < 0))

    try:
        p_t = float(ttest_rel(a_arr, b_arr, nan_policy="omit").pvalue)
    except Exception:
        p_t = float("nan")
    try:
        p_w = float(wilcoxon(a_arr, b_arr, zero_method="wilcox", alternative="two-sided").pvalue)
    except Exception:
        p_w = float("nan")

    return {
        "mean": float(np.mean(diff)),
        "std": _std(diff),
        "positive_count": pos,
        "negative_count": neg,
        "sign_test_pvalue": _exact_sign_test_two_sided(pos, neg),
        "paired_ttest_pvalue": p_t,
        "wilcoxon_pvalue": p_w,
    }


def _fmt(v: float) -> str:
    return f"{v:.4f}"


def _fmt_p(v: float) -> str:
    if np.isnan(v):
        return "nan"
    if v < 1e-4:
        return f"{v:.2e}"
    return f"{v:.4f}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize case3 across multi-dataset repeated split outputs")
    p.add_argument(
        "--summaries",
        type=str,
        nargs="*",
        default=DEFAULT_SUMMARIES,
        help="repeated_split_summary json files",
    )
    p.add_argument(
        "--labels",
        type=str,
        nargs="*",
        default=DEFAULT_LABELS,
        help="dataset labels (same length as summaries)",
    )
    p.add_argument(
        "--output-json",
        type=str,
        default="experiments/results/phase2_patha_case3_multidata/multi_dataset_case3_comparison_20260304.json",
    )
    p.add_argument(
        "--output-md",
        type=str,
        default="experiments/results/phase2_patha_case3_multidata/multi_dataset_case3_comparison_20260304.md",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    summaries = [Path(x) for x in args.summaries]
    labels = args.labels

    if len(summaries) != len(labels):
        raise ValueError(f"labels length mismatch: {len(labels)} vs summaries {len(summaries)}")

    per_dataset: List[Dict[str, Any]] = []
    all_cobra: List[float] = []
    all_daac: List[float] = []
    all_fusion: List[float] = []

    for label, summary_path in zip(labels, summaries):
        if not summary_path.exists():
            raise FileNotFoundError(f"summary not found: {summary_path}")
        summary = _load_json(summary_path)
        runs = summary.get("runs", [])
        if not runs:
            raise ValueError(f"no runs in summary: {summary_path}")

        cobra_vals: List[float] = []
        daac_vals: List[float] = []
        fusion_vals: List[float] = []

        for r in runs:
            result_path = Path(str(r.get("result_path", "")))
            if not result_path.exists():
                raise FileNotFoundError(f"result path not found: {result_path}")
            payload = _load_json(result_path)
            cases = payload.get("three_case_study", {}).get("cases", {})
            cobra_vals.append(float(cases["cobra_only"]["macro_f1"]))
            daac_vals.append(float(cases["daac_only"]["macro_f1"]))
            fusion_vals.append(float(cases["cobra_plus_daac"]["macro_f1"]))

        all_cobra.extend(cobra_vals)
        all_daac.extend(daac_vals)
        all_fusion.extend(fusion_vals)

        daac_vs_cobra = _paired_stats(daac_vals, cobra_vals)
        fusion_vs_cobra = _paired_stats(fusion_vals, cobra_vals)
        fusion_vs_daac = _paired_stats(fusion_vals, daac_vals)

        per_dataset.append(
            {
                "dataset": label,
                "summary_path": str(summary_path),
                "n_runs": int(len(cobra_vals)),
                "cobra_only_f1_mean": float(np.mean(cobra_vals)),
                "cobra_only_f1_std": _std(cobra_vals),
                "daac_only_f1_mean": float(np.mean(daac_vals)),
                "daac_only_f1_std": _std(daac_vals),
                "cobra_plus_daac_f1_mean": float(np.mean(fusion_vals)),
                "cobra_plus_daac_f1_std": _std(fusion_vals),
                "daac_vs_cobra": daac_vs_cobra,
                "fusion_vs_cobra": fusion_vs_cobra,
                "fusion_vs_daac": fusion_vs_daac,
            }
        )

    overall = {
        "n_total_runs": int(len(all_cobra)),
        "cobra_only_f1_mean": float(np.mean(all_cobra)),
        "cobra_only_f1_std": _std(all_cobra),
        "daac_only_f1_mean": float(np.mean(all_daac)),
        "daac_only_f1_std": _std(all_daac),
        "cobra_plus_daac_f1_mean": float(np.mean(all_fusion)),
        "cobra_plus_daac_f1_std": _std(all_fusion),
        "daac_vs_cobra": _paired_stats(all_daac, all_cobra),
        "fusion_vs_cobra": _paired_stats(all_fusion, all_cobra),
        "fusion_vs_daac": _paired_stats(all_fusion, all_daac),
    }

    report = {
        "timestamp": datetime.now().isoformat(),
        "note": "3-case repeated-split(10 seeds x 4 dataset compositions) aggregate",
        "datasets": per_dataset,
        "overall": overall,
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    md_lines: List[str] = []
    md_lines.append("# Multi-dataset 3-case Comparison (2026-03-04)")
    md_lines.append("")
    md_lines.append("| Dataset | COBRA-only | DAAC-only | COBRA+DAAC | p(DAAC>COBRA, Wilcoxon) | p(Fusion>COBRA, Wilcoxon) | p(Fusion vs DAAC, Wilcoxon) |")
    md_lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for ds in per_dataset:
        md_lines.append(
            "| "
            + ds["dataset"]
            + " | "
            + f"{_fmt(ds['cobra_only_f1_mean'])} ± {_fmt(ds['cobra_only_f1_std'])}"
            + " | "
            + f"{_fmt(ds['daac_only_f1_mean'])} ± {_fmt(ds['daac_only_f1_std'])}"
            + " | "
            + f"{_fmt(ds['cobra_plus_daac_f1_mean'])} ± {_fmt(ds['cobra_plus_daac_f1_std'])}"
            + " | "
            + _fmt_p(ds["daac_vs_cobra"]["wilcoxon_pvalue"])
            + " | "
            + _fmt_p(ds["fusion_vs_cobra"]["wilcoxon_pvalue"])
            + " | "
            + _fmt_p(ds["fusion_vs_daac"]["wilcoxon_pvalue"])
            + " |"
        )

    md_lines.append("| **Overall (40 runs)** | "
                    + f"**{_fmt(overall['cobra_only_f1_mean'])} ± {_fmt(overall['cobra_only_f1_std'])}** | "
                    + f"**{_fmt(overall['daac_only_f1_mean'])} ± {_fmt(overall['daac_only_f1_std'])}** | "
                    + f"**{_fmt(overall['cobra_plus_daac_f1_mean'])} ± {_fmt(overall['cobra_plus_daac_f1_std'])}** | "
                    + f"**{_fmt_p(overall['daac_vs_cobra']['wilcoxon_pvalue'])}** | "
                    + f"**{_fmt_p(overall['fusion_vs_cobra']['wilcoxon_pvalue'])}** | "
                    + f"**{_fmt_p(overall['fusion_vs_daac']['wilcoxon_pvalue'])}** |")
    md_lines.append("")
    md_lines.append("- p-values are paired Wilcoxon signed-rank tests on matched split seeds.")

    output_md = Path(args.output_md)
    output_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"[saved] json: {output_json}")
    print(f"[saved] md:   {output_md}")


if __name__ == "__main__":
    main()
