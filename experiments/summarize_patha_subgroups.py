"""
Path A 멀티시드 결과의 subgroup(sub_type/true_label) 지표 집계.

입력:
    - run_phase2_patha_multiseed.py summary json
    - 각 run의 result_path가 존재해야 함
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate subgroup metrics from PathA multiseed summary")
    p.add_argument("summary", type=str, help="multiseed summary json path")
    p.add_argument("--out", type=str, default="", help="optional output json path")
    return p.parse_args()


def _weighted_means(agg: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for key, v in agg.items():
        n = float(v.get("n_samples_total", 0.0))
        if n <= 0.0:
            continue
        out[key] = {
            "n_samples_total": int(n),
            "macro_f1_weighted_mean": float(v["macro_f1_sum"] / n),
            "balanced_accuracy_weighted_mean": float(v["balanced_sum"] / n),
        }
    return out


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    agg = {
        "phase1_best": {
            "by_sub_type": defaultdict(lambda: {"n_samples_total": 0.0, "macro_f1_sum": 0.0, "balanced_sum": 0.0}),
            "by_true_label": defaultdict(lambda: {"n_samples_total": 0.0, "macro_f1_sum": 0.0, "balanced_sum": 0.0}),
        },
        "phase2_best": {
            "by_sub_type": defaultdict(lambda: {"n_samples_total": 0.0, "macro_f1_sum": 0.0, "balanced_sum": 0.0}),
            "by_true_label": defaultdict(lambda: {"n_samples_total": 0.0, "macro_f1_sum": 0.0, "balanced_sum": 0.0}),
        },
    }

    used_runs = 0
    skipped_runs = 0
    for run in summary.get("runs", []):
        result_path = str(run.get("result_path", "")).strip()
        if not result_path:
            skipped_runs += 1
            continue
        p = Path(result_path)
        if not p.exists():
            skipped_runs += 1
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        subgroup = d.get("subgroup_metrics", {})
        if not subgroup:
            skipped_runs += 1
            continue

        used_runs += 1
        for phase_key in ["phase1_best", "phase2_best"]:
            phase_data = subgroup.get(phase_key, {})
            for scope in ["by_sub_type", "by_true_label"]:
                scope_data = phase_data.get(scope, {})
                for name, metrics in scope_data.items():
                    n = float(metrics.get("n_samples", 0.0))
                    if n <= 0:
                        continue
                    agg_bucket = agg[phase_key][scope][name]
                    agg_bucket["n_samples_total"] += n
                    agg_bucket["macro_f1_sum"] += n * float(metrics.get("macro_f1", 0.0))
                    agg_bucket["balanced_sum"] += n * float(metrics.get("balanced_accuracy", 0.0))

    phase1_sub = _weighted_means(agg["phase1_best"]["by_sub_type"])
    phase2_sub = _weighted_means(agg["phase2_best"]["by_sub_type"])
    phase1_lbl = _weighted_means(agg["phase1_best"]["by_true_label"])
    phase2_lbl = _weighted_means(agg["phase2_best"]["by_true_label"])

    delta_sub = {}
    for k in sorted(set(phase1_sub) | set(phase2_sub)):
        if k in phase1_sub and k in phase2_sub:
            delta_sub[k] = phase2_sub[k]["macro_f1_weighted_mean"] - phase1_sub[k]["macro_f1_weighted_mean"]

    delta_lbl = {}
    for k in sorted(set(phase1_lbl) | set(phase2_lbl)):
        if k in phase1_lbl and k in phase2_lbl:
            delta_lbl[k] = phase2_lbl[k]["macro_f1_weighted_mean"] - phase1_lbl[k]["macro_f1_weighted_mean"]

    report: Dict[str, Any] = {
        "summary_path": str(summary_path),
        "n_runs_in_summary": int(summary.get("n_runs", 0)),
        "used_runs": used_runs,
        "skipped_runs": skipped_runs,
        "phase1_best": {
            "by_sub_type": phase1_sub,
            "by_true_label": phase1_lbl,
        },
        "phase2_best": {
            "by_sub_type": phase2_sub,
            "by_true_label": phase2_lbl,
        },
        "delta_phase2_minus_phase1": {
            "by_sub_type_macro_f1": delta_sub,
            "by_true_label_macro_f1": delta_lbl,
        },
    }

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print("[Subgroup] report file:", out_path)

    print("[Subgroup] used_runs:", used_runs, "skipped_runs:", skipped_runs)
    print("[Subgroup] delta by sub_type:", report["delta_phase2_minus_phase1"]["by_sub_type_macro_f1"])
    print("[Subgroup] delta by true_label:", report["delta_phase2_minus_phase1"]["by_true_label_macro_f1"])


if __name__ == "__main__":
    main()
