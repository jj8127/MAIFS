"""
Path A fixed-kfold summary 변동성 분석 리포트 생성기.

목적:
    - split_seed / test_fold 축으로 f1_diff 변동을 분해
    - 블록(요약 파일) 간 평균 방향성 차이를 비교
    - "초기 양성 블록 vs 후속 음성 블록" 현상을 재현 가능한 형태로 기록

예시:
    .venv-qwen/bin/python experiments/analyze_patha_kfold_variance.py \
      --summary kfold25_a:experiments/results/.../fixed_kfold_summary_25runs_5seeds_20260216.json \
      --summary kfold25_b:experiments/results/.../fixed_kfold_summary_25runs_5seeds_305_309_20260216.json \
      --summary kfold25_c:experiments/results/.../fixed_kfold_summary_25runs_5seeds_310_314_20260216.json \
      --summary kfold75:experiments/results/.../fixed_kfold_summary_75runs_15seeds_300_314_20260216.json \
      --out experiments/results/.../kfold_variance_diagnostics_20260216.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze PathA fixed-kfold variance across split_seed/test_fold")
    p.add_argument(
        "--summary",
        action="append",
        default=[],
        help="Format: label:path_to_summary_json (repeatable)",
    )
    p.add_argument("--out", type=str, default="", help="Optional output report json path")
    return p.parse_args()


def _load_json(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _parse_labeled_path(token: str) -> Tuple[str, str]:
    t = token.strip()
    if ":" not in t:
        raise ValueError(f"invalid --summary token (need label:path): {token}")
    label, path = t.split(":", 1)
    label = label.strip()
    path = path.strip()
    if not label or not path:
        raise ValueError(f"invalid --summary token: {token}")
    return label, path


def _stat_row(values: List[float]) -> Dict[str, Any]:
    arr = np.array(values, dtype=float)
    if arr.size == 0:
        return {
            "n": 0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "positive_count": 0,
            "negative_count": 0,
            "zero_count": 0,
        }
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=0)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "positive_count": int(np.sum(arr > 0.0)),
        "negative_count": int(np.sum(arr < 0.0)),
        "zero_count": int(np.sum(arr == 0.0)),
    }


def _build_risk_flags(global_stats: Dict[str, Any], seed_rows: List[Dict[str, Any]], fold_rows: List[Dict[str, Any]]) -> List[str]:
    flags: List[str] = []

    seed_means = [float(r["mean"]) for r in seed_rows]
    fold_means = [float(r["mean"]) for r in fold_rows]
    g_mean = float(global_stats["mean"])
    g_std = float(global_stats["std"])

    if any(m > 0.0 for m in seed_means) and any(m < 0.0 for m in seed_means):
        flags.append("split_seed_mean_sign_mixed")
    if seed_means and (max(seed_means) - min(seed_means) >= 0.03):
        flags.append("split_seed_mean_range_large")
    if any(m > 0.0 for m in fold_means) and any(m < 0.0 for m in fold_means):
        flags.append("test_fold_mean_sign_mixed")
    if fold_means and (max(fold_means) - min(fold_means) >= 0.01):
        flags.append("test_fold_bias_possible")
    if abs(g_mean) < 0.002 and g_std >= 0.02:
        flags.append("mean_near_zero_with_high_run_variance")
    if int(global_stats["positive_count"]) == int(global_stats["negative_count"]):
        flags.append("positive_negative_balanced")

    return flags


def _summarize_kfold_summary(label: str, summary_path: str, summary: Dict[str, Any]) -> Dict[str, Any]:
    runs = summary.get("runs", [])
    diffs = [float(r.get("f1_diff", 0.0)) for r in runs]
    global_stats = _stat_row(diffs)

    seed_map: Dict[int, List[Dict[str, Any]]] = {}
    fold_map: Dict[int, List[Dict[str, Any]]] = {}
    cell_map: Dict[Tuple[int, int], float] = {}

    for r in runs:
        seed = int(r.get("split_seed", r.get("seed", 0)))
        fold = int(r.get("test_fold", -1))
        diff = float(r.get("f1_diff", 0.0))
        seed_map.setdefault(seed, []).append(r)
        fold_map.setdefault(fold, []).append(r)
        cell_map[(seed, fold)] = diff

    seed_rows: List[Dict[str, Any]] = []
    for seed in sorted(seed_map):
        vals = [float(x.get("f1_diff", 0.0)) for x in seed_map[seed]]
        row = {"split_seed": int(seed), **_stat_row(vals)}
        row["fold_values"] = {
            str(int(x.get("test_fold", -1))): float(x.get("f1_diff", 0.0))
            for x in sorted(seed_map[seed], key=lambda y: int(y.get("test_fold", -1)))
        }
        seed_rows.append(row)

    fold_rows: List[Dict[str, Any]] = []
    for fold in sorted(fold_map):
        vals = [float(x.get("f1_diff", 0.0)) for x in fold_map[fold]]
        row = {"test_fold": int(fold), **_stat_row(vals)}
        row["split_seed_values"] = {
            str(int(x.get("split_seed", x.get("seed", 0)))): float(x.get("f1_diff", 0.0))
            for x in sorted(fold_map[fold], key=lambda y: int(y.get("split_seed", y.get("seed", 0))))
        }
        fold_rows.append(row)

    seed_order = sorted(seed_map)
    fold_order = sorted(fold_map)
    matrix_rows: List[Dict[str, Any]] = []
    for seed in seed_order:
        row: Dict[str, Any] = {"split_seed": int(seed)}
        for fold in fold_order:
            row[f"fold_{fold}"] = None if (seed, fold) not in cell_map else float(cell_map[(seed, fold)])
        matrix_rows.append(row)

    agg = summary.get("aggregate", {})
    f1_mean = float(agg.get("f1_diff_mean", global_stats["mean"]))
    sign_p = agg.get("sign_test_pvalue", None)
    pooled_p = agg.get("pooled_mcnemar_pvalue", None)

    return {
        "label": label,
        "summary_path": summary_path,
        "n_runs": int(summary.get("n_runs", len(runs))),
        "split_strategy": summary.get("split_protocol", {}).get("strategy"),
        "aggregate": {
            "f1_diff_mean": f1_mean,
            "f1_diff_std": float(agg.get("f1_diff_std", global_stats["std"])),
            "sign_test_pvalue": None if sign_p is None else float(sign_p),
            "pooled_mcnemar_pvalue": None if pooled_p is None else float(pooled_p),
            "positive_count": int(agg.get("positive_count", global_stats["positive_count"])),
            "negative_count": int(agg.get("negative_count", global_stats["negative_count"])),
            "zero_count": int(agg.get("zero_count", global_stats["zero_count"])),
        },
        "run_f1_diff_stats": global_stats,
        "split_seed_stats": {
            "rows": seed_rows,
            "mean_of_means": float(np.mean([r["mean"] for r in seed_rows])) if seed_rows else 0.0,
            "std_of_means": float(np.std([r["mean"] for r in seed_rows], ddof=0)) if seed_rows else 0.0,
            "min_mean": float(np.min([r["mean"] for r in seed_rows])) if seed_rows else 0.0,
            "max_mean": float(np.max([r["mean"] for r in seed_rows])) if seed_rows else 0.0,
            "positive_mean_seed_count": int(sum(1 for r in seed_rows if float(r["mean"]) > 0.0)),
            "negative_mean_seed_count": int(sum(1 for r in seed_rows if float(r["mean"]) < 0.0)),
            "zero_mean_seed_count": int(sum(1 for r in seed_rows if float(r["mean"]) == 0.0)),
        },
        "test_fold_stats": {
            "rows": fold_rows,
            "mean_of_means": float(np.mean([r["mean"] for r in fold_rows])) if fold_rows else 0.0,
            "std_of_means": float(np.std([r["mean"] for r in fold_rows], ddof=0)) if fold_rows else 0.0,
            "min_mean": float(np.min([r["mean"] for r in fold_rows])) if fold_rows else 0.0,
            "max_mean": float(np.max([r["mean"] for r in fold_rows])) if fold_rows else 0.0,
        },
        "split_seed_fold_matrix": {
            "fold_order": fold_order,
            "rows": matrix_rows,
        },
        "risk_flags": _build_risk_flags(global_stats=global_stats, seed_rows=seed_rows, fold_rows=fold_rows),
    }


def _compare_summaries(summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    rows = [
        {
            "label": s["label"],
            "summary_path": s["summary_path"],
            "n_runs": s["n_runs"],
            "f1_diff_mean": float(s["aggregate"]["f1_diff_mean"]),
            "f1_diff_std": float(s["aggregate"]["f1_diff_std"]),
            "sign_test_pvalue": s["aggregate"]["sign_test_pvalue"],
            "pooled_mcnemar_pvalue": s["aggregate"]["pooled_mcnemar_pvalue"],
            "positive_count": int(s["aggregate"]["positive_count"]),
            "negative_count": int(s["aggregate"]["negative_count"]),
            "zero_count": int(s["aggregate"]["zero_count"]),
        }
        for s in summaries
    ]
    ranked = sorted(rows, key=lambda x: (x["f1_diff_mean"], -x["f1_diff_std"]), reverse=True)

    deltas = []
    if rows:
        ref = rows[0]
        for cand in rows[1:]:
            deltas.append(
                {
                    "reference_label": ref["label"],
                    "candidate_label": cand["label"],
                    "delta_f1_diff_mean_candidate_minus_reference": float(cand["f1_diff_mean"] - ref["f1_diff_mean"]),
                }
            )

    return {
        "rows": rows,
        "ranking_by_f1_diff_mean": [r["label"] for r in ranked],
        "pairwise_deltas_against_first": deltas,
    }


def main() -> None:
    args = parse_args()
    if not args.summary:
        raise ValueError("at least one --summary is required")

    items: List[Dict[str, Any]] = []
    for token in args.summary:
        label, path = _parse_labeled_path(token)
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"summary not found: {path}")
        summary = _load_json(path)
        items.append(_summarize_kfold_summary(label=label, summary_path=path, summary=summary))

    report = {
        "timestamp": datetime.now().isoformat(),
        "summaries": items,
        "comparison": _compare_summaries(items),
    }

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print("[KFoldVariance] report file:", out_path)

    for s in items:
        agg = s["aggregate"]
        print(
            f"[KFoldVariance] {s['label']}:",
            f"n_runs={s['n_runs']}",
            f"mean={agg['f1_diff_mean']:+.6f}",
            f"sign_p={agg['sign_test_pvalue']}",
            f"pooled_p={agg['pooled_mcnemar_pvalue']}",
            f"+/-/0={agg['positive_count']}/{agg['negative_count']}/{agg['zero_count']}",
        )
        print("  risk flags:", s["risk_flags"])

    ranking = report["comparison"]["ranking_by_f1_diff_mean"]
    print("[KFoldVariance] ranking_by_f1_diff_mean:", ranking)


if __name__ == "__main__":
    main()

