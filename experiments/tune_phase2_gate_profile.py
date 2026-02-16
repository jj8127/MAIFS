"""
Path A Phase2 게이트 임계치(grid) 자동 탐색기.

목적:
    - 여러 summary 후보를 pass/fail 타깃으로 지정
    - 게이트 파라미터 조합을 탐색해 타깃 분리를 최대화
    - 추천 프로파일(임계치)과 각 summary 판정 결과를 JSON으로 저장

예시:
    .venv-qwen/bin/python experiments/tune_phase2_gate_profile.py \
      --config experiments/configs/phase2_patha_scale120_feat_enhanced36_ridge.yaml \
      --summary seed42:experiments/results/phase2_patha_scale120_feat_enhanced36_ridge/summary_10seeds_42_51_statsv2_20260216.json \
      --summary seed52:experiments/results/phase2_patha_scale120_feat_enhanced36_ridge/summary_10seeds_52_61_statsv2_20260216.json \
      --summary random25:experiments/results/phase2_patha_scale120_feat_enhanced36_ridge/repeated_random_summary_25runs_300_324_20260216.json \
      --summary kfold25:experiments/results/phase2_patha_scale120_feat_enhanced36_ridge/fixed_kfold_summary_25runs_5seeds_20260216.json \
      --target-pass seed42,kfold25 \
      --target-fail seed52,random25 \
      --out experiments/results/phase2_patha_scale120_feat_enhanced36_ridge/gate_profile_tuning_20260216.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.evaluate_phase2_gate import evaluate_gate


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Grid-search gate profile thresholds for PathA summaries")
    p.add_argument("--config", type=str, default="", help="Optional config yaml (for baseline_summary)")
    p.add_argument(
        "--summary",
        action="append",
        default=[],
        help="Format: label:path_to_summary_json (repeatable)",
    )
    p.add_argument("--target-pass", type=str, default="", help="Comma-separated labels expected to pass")
    p.add_argument("--target-fail", type=str, default="", help="Comma-separated labels expected to fail")
    p.add_argument(
        "--min-f1-grid",
        type=str,
        default="0.0,0.001,0.002,0.003,0.005,0.008,0.01",
        help="Comma-separated thresholds for min_f1_diff_mean",
    )
    p.add_argument(
        "--min-improve-grid",
        type=str,
        default="-0.01,-0.005,-0.002,-0.001,0.0",
        help="Comma-separated thresholds for min_improvement_over_baseline",
    )
    p.add_argument(
        "--max-sign-grid",
        type=str,
        default="none,0.5,0.4,0.35,0.3,0.25,0.2",
        help="Comma-separated thresholds for max_sign_test_pvalue (use 'none')",
    )
    p.add_argument(
        "--max-pooled-grid",
        type=str,
        default="0.1,0.2,0.3,0.5,0.7,1.0",
        help="Comma-separated thresholds for max_pooled_mcnemar_pvalue",
    )
    p.add_argument("--min-runs", type=int, default=10, help="Fixed min_runs constraint")
    p.add_argument("--out", type=str, default="", help="Optional output json path")
    return p.parse_args()


def _load_json(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_yaml(path: str) -> Dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


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


def _parse_float_list(raw: str) -> List[float]:
    out: List[float] = []
    for x in raw.split(","):
        t = x.strip()
        if not t:
            continue
        out.append(float(t))
    if not out:
        raise ValueError("empty float grid")
    return out


def _parse_optional_float_list(raw: str) -> List[Optional[float]]:
    out: List[Optional[float]] = []
    for x in raw.split(","):
        t = x.strip().lower()
        if not t:
            continue
        if t in {"none", "null"}:
            out.append(None)
        else:
            out.append(float(t))
    if not out:
        raise ValueError("empty optional float grid")
    return out


def _parse_label_set(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def _sort_key(row: Dict[str, Any]) -> Tuple[float, float, float, float, float]:
    max_sign = row["profile"]["max_sign_test_pvalue"]
    max_sign_tie = 1.1 if max_sign is None else float(max_sign)
    return (
        float(row["score"]),
        float(row["profile"]["min_f1_diff_mean"]),
        -max_sign_tie,
        -float(row["profile"]["max_pooled_mcnemar_pvalue"]),
        float(row["profile"]["min_improvement_over_baseline"]),
    )


def main() -> None:
    args = parse_args()
    if not args.summary:
        raise ValueError("at least one --summary is required")

    baseline_summary_path: Optional[str] = None
    if args.config:
        cfg = _load_yaml(args.config)
        b = cfg.get("protocol", {}).get("baseline_summary")
        if isinstance(b, str) and b.strip():
            baseline_summary_path = b.strip()
    baseline = _load_json(baseline_summary_path) if baseline_summary_path else None

    summaries: Dict[str, Dict[str, Any]] = {}
    summary_paths: Dict[str, str] = {}
    for token in args.summary:
        label, path = _parse_labeled_path(token)
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"summary not found: {path}")
        summaries[label] = _load_json(path)
        summary_paths[label] = path

    target_pass = set(_parse_label_set(args.target_pass))
    target_fail = set(_parse_label_set(args.target_fail))
    all_labels = set(summaries.keys())
    if not target_pass and not target_fail:
        raise ValueError("at least one target label is required in --target-pass/--target-fail")
    unknown = (target_pass | target_fail) - all_labels
    if unknown:
        raise ValueError(f"unknown target labels: {sorted(unknown)}; available={sorted(all_labels)}")

    min_f1_grid = _parse_float_list(args.min_f1_grid)
    min_improve_grid = _parse_float_list(args.min_improve_grid)
    max_sign_grid = _parse_optional_float_list(args.max_sign_grid)
    max_pooled_grid = _parse_float_list(args.max_pooled_grid)

    rows: List[Dict[str, Any]] = []
    for min_f1 in min_f1_grid:
        for min_improve in min_improve_grid:
            for max_sign in max_sign_grid:
                for max_pooled in max_pooled_grid:
                    profile = {
                        "min_runs": int(args.min_runs),
                        "min_f1_diff_mean": float(min_f1),
                        "min_significant_count": 0,
                        "min_improvement_over_baseline": float(min_improve),
                        "min_positive_seed_count": 0,
                        "max_sign_test_pvalue": max_sign,
                        "require_pooled_mcnemar_significant": False,
                        "max_pooled_mcnemar_pvalue": float(max_pooled),
                    }
                    by_label: Dict[str, Any] = {}
                    pass_ok = 0
                    fail_ok = 0
                    for label, summary in summaries.items():
                        rep = evaluate_gate(candidate=summary, baseline=baseline, **profile)
                        gate_pass = bool(rep["gate_pass"])
                        by_label[label] = {
                            "gate_pass": gate_pass,
                            "f1_diff_mean": float(rep["candidate"]["aggregate"].get("f1_diff_mean", 0.0)),
                            "sign_test_pvalue": float(rep["candidate"]["sign_stats"].get("sign_test_pvalue", 1.0)),
                            "pooled_mcnemar_pvalue": (
                                None
                                if rep["candidate"]["pooled_mcnemar"] is None
                                else float(rep["candidate"]["pooled_mcnemar"].get("pvalue", 1.0))
                            ),
                        }
                        if label in target_pass and gate_pass:
                            pass_ok += 1
                        if label in target_fail and not gate_pass:
                            fail_ok += 1

                    score = pass_ok + fail_ok
                    row = {
                        "score": int(score),
                        "target_pass_ok": int(pass_ok),
                        "target_fail_ok": int(fail_ok),
                        "target_pass_total": int(len(target_pass)),
                        "target_fail_total": int(len(target_fail)),
                        "all_constraints_satisfied": bool(pass_ok == len(target_pass) and fail_ok == len(target_fail)),
                        "profile": profile,
                        "by_label": by_label,
                    }
                    rows.append(row)

    ranked = sorted(rows, key=_sort_key, reverse=True)
    best = ranked[0]
    all_satisfied = [r for r in ranked if r["all_constraints_satisfied"]]
    best_all_satisfied = all_satisfied[0] if all_satisfied else None

    payload = {
        "timestamp": datetime.now().isoformat(),
        "config": args.config or None,
        "baseline_summary": baseline_summary_path,
        "summary_paths": summary_paths,
        "target_pass": sorted(target_pass),
        "target_fail": sorted(target_fail),
        "grid": {
            "min_f1_diff_mean": min_f1_grid,
            "min_improvement_over_baseline": min_improve_grid,
            "max_sign_test_pvalue": max_sign_grid,
            "max_pooled_mcnemar_pvalue": max_pooled_grid,
            "min_runs": int(args.min_runs),
        },
        "search_size": len(rows),
        "best_overall": best,
        "best_all_constraints_satisfied": best_all_satisfied,
        "top10": ranked[:10],
    }

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print("[GateTune] report file:", out_path)

    print("[GateTune] search_size:", len(rows))
    print("[GateTune] best_overall score:", best["score"])
    print("[GateTune] best_overall profile:", best["profile"])
    if best_all_satisfied is not None:
        print("[GateTune] best_all_constraints_satisfied profile:", best_all_satisfied["profile"])
    else:
        print("[GateTune] no profile satisfies all pass/fail constraints")


if __name__ == "__main__":
    main()
