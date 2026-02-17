"""
Path A guard 민감도 분석 리포트 생성기.

목적:
    - repeated summary + 개별 run result를 결합해 guard 선택 동작을 정량화
    - 음수 run에서 route rate / raw phase2 gate 상태를 빠르게 역추적

예시:
    .venv-qwen/bin/python experiments/analyze_patha_guard_sensitivity.py \
      experiments/results/phase2_patha_scale120_feat_risk52_oracle_lossaverse_guard_valselect_tunec/fixed_kfold_summary_30runs_6seeds_risk52_oracle_lossaverse_guard_valselect_tunec_20260217.json \
      --out experiments/results/phase2_patha_scale120_feat_risk52_oracle_lossaverse_guard_valselect_tunec/guard_sensitivity_30runs_20260217.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze PathA guard sensitivity from repeated summary")
    p.add_argument("summary", type=str, help="PathA repeated summary json path")
    p.add_argument("--top-k", type=int, default=10, help="Number of worst runs to include")
    p.add_argument("--out", type=str, default="", help="Optional output report json path")
    return p.parse_args()


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_result_path(raw: str) -> Optional[Path]:
    if not raw:
        return None
    p = Path(raw)
    if p.exists():
        return p
    p2 = ROOT / raw
    if p2.exists():
        return p2
    return None


def _to_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _to_bool(v: Any) -> Optional[bool]:
    if isinstance(v, bool):
        return v
    if v is None:
        return None
    if isinstance(v, str):
        t = v.strip().lower()
        if t in {"true", "1", "yes"}:
            return True
        if t in {"false", "0", "no"}:
            return False
    return None


def _numeric_stats(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "q25": None,
            "median": None,
            "q75": None,
            "max": None,
        }
    arr = np.array(values, dtype=float)
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=0)),
        "min": float(np.min(arr)),
        "q25": float(np.quantile(arr, 0.25)),
        "median": float(np.quantile(arr, 0.50)),
        "q75": float(np.quantile(arr, 0.75)),
        "max": float(np.max(arr)),
    }


def _build_selected_row(run: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    router = result.get("router", {})
    guard = router.get("guard", {}) if isinstance(router, dict) else {}
    tuning = guard.get("tuning", {}) if isinstance(guard, dict) else {}

    phase2_best = str(run.get("phase2_best", ""))
    sel = tuning.get(phase2_best, {}) if isinstance(tuning, dict) else {}

    return {
        "split_seed": int(run.get("split_seed", 0)),
        "seed": int(run.get("seed", 0)),
        "test_fold": run.get("test_fold"),
        "val_fold": run.get("val_fold"),
        "phase1_best": str(run.get("phase1_best", "")),
        "phase2_best": phase2_best,
        "f1_diff": float(run.get("f1_diff", 0.0)),
        "mcnemar_pvalue": _to_float(run.get("mcnemar_pvalue")),
        "selection_scope": guard.get("selection_scope"),
        "selected_has_tuning": bool(sel),
        "selected_score_mode": sel.get("score_mode"),
        "selected_threshold": _to_float(sel.get("threshold")),
        "selected_route_rate_test": _to_float(sel.get("phase2_route_rate_test")),
        "selected_route_rate_val": _to_float(sel.get("phase2_route_rate_val")),
        "selected_val_gain_vs_phase1": _to_float(sel.get("val_gain_vs_phase1")),
        "selected_raw_phase2_val_gain": _to_float(sel.get("raw_phase2_val_gain")),
        "selected_raw_phase2_gate_pass": _to_bool(sel.get("raw_phase2_gate_pass")),
        "selected_min_phase2_val_gain": _to_float(sel.get("min_phase2_val_gain")),
        "selected_min_val_gain": _to_float(sel.get("min_val_gain")),
        "selected_max_route_rate": _to_float(sel.get("max_route_rate")),
    }


def _group_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    route = [float(r["selected_route_rate_test"]) for r in rows if r.get("selected_route_rate_test") is not None]
    raw_gain = [float(r["selected_raw_phase2_val_gain"]) for r in rows if r.get("selected_raw_phase2_val_gain") is not None]
    val_gain = [float(r["selected_val_gain_vs_phase1"]) for r in rows if r.get("selected_val_gain_vs_phase1") is not None]
    raw_gate = [bool(r["selected_raw_phase2_gate_pass"]) for r in rows if r.get("selected_raw_phase2_gate_pass") is not None]
    return {
        "n_runs": int(len(rows)),
        "with_selected_tuning": int(sum(1 for r in rows if bool(r.get("selected_has_tuning", False)))),
        "selected_route_rate_test": _numeric_stats(route),
        "selected_raw_phase2_val_gain": _numeric_stats(raw_gain),
        "selected_val_gain_vs_phase1": _numeric_stats(val_gain),
        "selected_raw_phase2_gate_pass_rate": (
            None if not raw_gate else float(sum(1 for x in raw_gate if x) / len(raw_gate))
        ),
    }


def _seed_table(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_seed: Dict[int, List[Dict[str, Any]]] = {}
    for r in rows:
        key = int(r["split_seed"])
        by_seed.setdefault(key, []).append(r)

    out: List[Dict[str, Any]] = []
    for seed in sorted(by_seed):
        rs = by_seed[seed]
        diffs = np.array([float(r["f1_diff"]) for r in rs], dtype=float)
        out.append(
            {
                "split_seed": int(seed),
                "n_runs": int(len(rs)),
                "f1_diff_mean": float(np.mean(diffs)),
                "f1_diff_min": float(np.min(diffs)),
                "f1_diff_max": float(np.max(diffs)),
                "negative_count": int(np.sum(diffs < 0.0)),
                "positive_count": int(np.sum(diffs > 0.0)),
                "zero_count": int(np.sum(diffs == 0.0)),
            }
        )
    return out


def analyze(summary: Dict[str, Any], top_k: int = 10) -> Dict[str, Any]:
    runs = summary.get("runs", [])
    rows: List[Dict[str, Any]] = []
    missing_result_paths: List[str] = []

    for run in runs:
        rp = _resolve_result_path(str(run.get("result_path", "")))
        if rp is None:
            missing_result_paths.append(str(run.get("result_path", "")))
            continue
        result = _load_json(rp)
        rows.append(_build_selected_row(run, result))

    rows_sorted = sorted(rows, key=lambda x: float(x["f1_diff"]))
    neg_rows = [r for r in rows if float(r["f1_diff"]) < 0.0]
    non_neg_rows = [r for r in rows if float(r["f1_diff"]) >= 0.0]

    agg = summary.get("aggregate", {})
    report: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "summary_path": summary.get("summary_path", None),
        "n_runs_summary": int(summary.get("n_runs", len(runs))),
        "n_runs_loaded": int(len(rows)),
        "missing_result_paths": missing_result_paths,
        "aggregate_from_summary": {
            "f1_diff_mean": _to_float(agg.get("f1_diff_mean")),
            "f1_diff_std": _to_float(agg.get("f1_diff_std")),
            "positive_count": agg.get("positive_count"),
            "negative_count": agg.get("negative_count"),
            "zero_count": agg.get("zero_count"),
            "sign_test_pvalue": _to_float(agg.get("sign_test_pvalue")),
            "negative_rate": _to_float(agg.get("negative_rate")),
            "downside_mean": _to_float(agg.get("downside_mean")),
            "cvar_downside": _to_float(agg.get("cvar_downside")),
            "worst_case_loss": _to_float(agg.get("worst_case_loss")),
        },
        "selected_overall": _group_summary(rows),
        "selected_negative_runs": _group_summary(neg_rows),
        "selected_non_negative_runs": _group_summary(non_neg_rows),
        "worst_runs": rows_sorted[: max(1, int(top_k))],
        "best_runs": list(reversed(rows_sorted[-max(1, int(top_k)) :])),
        "split_seed_summary": _seed_table(rows),
    }
    return report


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary)
    summary = _load_json(summary_path)
    report = analyze(summary, top_k=int(args.top_k))
    report["summary_path"] = str(summary_path)

    if args.out:
        out_path = Path(args.out)
    else:
        out_path = summary_path.with_name(f"{summary_path.stem}_guard_sensitivity.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[GuardSensitivity] report file:", out_path)
    agg = report["aggregate_from_summary"]
    print(
        "[GuardSensitivity] aggregate:",
        {
            "f1_diff_mean": agg["f1_diff_mean"],
            "positive_count": agg["positive_count"],
            "negative_count": agg["negative_count"],
            "negative_rate": agg["negative_rate"],
        },
    )


if __name__ == "__main__":
    main()
