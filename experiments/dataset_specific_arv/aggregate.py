#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = SCRIPT_DIR.parent
RESULTS_ROOT = EXPERIMENTS_DIR / "results" / "dataset_runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate dataset-specific ARV outputs")
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def now_utc() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def iter_run_dirs(results_root: Path) -> Iterable[Path]:
    if not results_root.exists():
        return []
    for dataset_dir in sorted(results_root.iterdir()):
        if not dataset_dir.is_dir() or dataset_dir.name.startswith("_"):
            continue
        for seed_dir in sorted(dataset_dir.iterdir()):
            if not seed_dir.is_dir():
                continue
            for run_dir in sorted(seed_dir.iterdir()):
                if run_dir.is_dir():
                    yield run_dir


def metric_at(payload: Dict[str, Any], path: Sequence[str]) -> Optional[float]:
    node: Any = payload
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    try:
        return float(node)
    except Exception:
        return None


def first_present(payload: Dict[str, Any], paths: Sequence[Sequence[str]]) -> Optional[float]:
    for path in paths:
        value = metric_at(payload, path)
        if value is not None:
            return value
    return None


def render_markdown_table(rows: Sequence[Dict[str, Any]]) -> str:
    lines: List[str] = []
    if any(bool(row.get("is_smoke")) for row in rows):
        lines.extend([
            "> **주의**: `최종 성능 아님`",
            "> 이 표에는 smoke 검증용 run이 포함될 수 있으므로 최종 벤치마크로 해석하면 안 됩니다.",
            "",
        ])
    headers = [
        "dataset",
        "seed",
        "purpose",
        "run_id",
        "state",
        "train_records",
        "base_strict_f1",
        "stage1_strict_f1",
        "stage2_strict_f1",
        "base_binary_f1",
        "aux_binary_f1",
        "stage2_binary_f1",
        "failed_jobs",
    ]
    lines.extend(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"])
    for row in rows:
        failed = ",".join(job["stage"] for job in row.get("failed_jobs", [])) or "-"
        values = [
            str(row.get("dataset_name", "")),
            str(row.get("seed", "")),
            str(row.get("run_purpose", "")),
            str(row.get("run_id", "")),
            str(row.get("overall_state", "")),
            str(row.get("train_records", "-")),
            fmt(row.get("strict_macro_f1", {}).get("base")),
            fmt(row.get("strict_macro_f1", {}).get("stage1")),
            fmt(row.get("strict_macro_f1", {}).get("stage2")),
            fmt(row.get("binary_macro_f1", {}).get("base")),
            fmt(row.get("binary_macro_f1", {}).get("aux")),
            fmt(row.get("binary_macro_f1", {}).get("stage2")),
            failed,
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def render_failure_table(rows: Sequence[Dict[str, Any]]) -> str:
    failures: List[Dict[str, Any]] = []
    for row in rows:
        for job in row.get("failed_jobs", []):
            failures.append(
                {
                    "dataset": row.get("dataset_name"),
                    "seed": row.get("seed"),
                    "run_id": row.get("run_id"),
                    "stage": job.get("stage"),
                    "job_id": job.get("job_id"),
                    "error": job.get("error"),
                    "stderr_path": job.get("stderr_path"),
                }
            )
    if not failures:
        return "_No failures recorded._\n"

    headers = ["dataset", "seed", "run_id", "stage", "job_id", "error", "stderr_path"]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in failures:
        values = [str(row.get(header, "")) for header in headers]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.4f}"
    except Exception:
        return str(value)


def extract_failed_jobs(run_manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        {
            "job_id": job.get("job_id"),
            "stage": job.get("stage"),
            "error": job.get("error"),
            "stderr_path": job.get("stderr_path"),
        }
        for job in run_manifest.get("jobs", [])
        if job.get("state") == "failed"
    ]


def _train_records(run_manifest: Dict[str, Any], manifest_summary: Dict[str, Any]) -> Optional[int]:
    if run_manifest.get("train_records") is not None:
        try:
            return int(run_manifest["train_records"])
        except Exception:
            pass
    try:
        return int(manifest_summary.get("counts_by_split", {}).get("train", {}).get("n_records"))
    except Exception:
        return None


def render_run_report(row: Dict[str, Any]) -> str:
    dataset_name = str(row.get("dataset_name", "dataset"))
    run_id = str(row.get("run_id", "run"))
    run_purpose = str(row.get("run_purpose", "unknown"))
    is_smoke = bool(row.get("is_smoke"))
    notes = list(row.get("notes", []))
    split_counts = row.get("manifest_summary", {}).get("counts_by_split", {})
    stage2_vs_base = row.get("change_stats", {}).get("stage2_vs_base") or {}
    lines = [f"# {dataset_name} {run_purpose.capitalize()} Run Report", ""]
    if is_smoke:
        lines.extend([
            "> **주의**: `최종 성능 아님`",
            "> 이 보고서는 smoke 검증 결과이므로 최종 성능 판단 근거로 사용하면 안 됩니다.",
            "",
        ])
    lines.extend([
        "**메타데이터**",
        "",
        "| 항목 | 값 |",
        "| --- | --- |",
        f"| `dataset_name` | `{dataset_name}` |",
        f"| `seed` | `{row.get('seed')}` |",
        f"| `run_id` | `{run_id}` |",
        f"| `run_purpose` | `{run_purpose}` |",
        f"| `is_smoke` | `{str(is_smoke).lower()}` |",
        f"| `overall_state` | `{row.get('overall_state')}` |",
        f"| `base_epochs` | `{row.get('base_epochs')}` |",
        f"| `aux_epochs` | `{row.get('aux_epochs')}` |",
        f"| `train_records` | `{row.get('train_records')}` |",
        "",
        "**Split Counts**",
        "",
        "| split | n_records | by_true_label |",
        "| --- | --- | --- |",
    ])
    for split in ("train", "val", "test"):
        entry = split_counts.get(split, {})
        lines.append(f"| `{split}` | `{entry.get('n_records', '-')}` | `{json.dumps(entry.get('by_true_label', {}), ensure_ascii=False)}` |")
    lines.extend([
        "",
        "**Key Metrics**",
        "",
        "| 단계 | strict macro-F1 | binary macro-F1 |",
        "| --- | --- | --- |",
        f"| `base` | `{fmt(row.get('strict_macro_f1', {}).get('base'))}` | `{fmt(row.get('binary_macro_f1', {}).get('base'))}` |",
        f"| `stage1` | `{fmt(row.get('strict_macro_f1', {}).get('stage1'))}` | `{fmt(row.get('binary_macro_f1', {}).get('stage1'))}` |",
        f"| `stage2` | `{fmt(row.get('strict_macro_f1', {}).get('stage2'))}` | `{fmt(row.get('binary_macro_f1', {}).get('stage2'))}` |",
        "",
        "**Stage2 Change Stats**",
        "",
        f"- `base 대비 변경 건수 = {stage2_vs_base.get('n_changed', '-')}`",
        f"- `helpful_change_rate = {stage2_vs_base.get('helpful_change_rate', '-')}`",
        f"- `harmful_change_rate = {stage2_vs_base.get('harmful_change_rate', '-')}`",
    ])
    if notes:
        lines.extend(["", "**Notes**", ""])
        lines.extend([f"- {note}" for note in notes])
    return "\n".join(lines) + "\n"


def summarize_run(run_dir: Path) -> Dict[str, Any]:
    run_manifest = load_json(run_dir / "run_manifest.json") or {}
    manifest_summary = load_json(run_dir / "manifests" / "manifest_summary.json") or {}
    preflight = load_json(run_dir / "preflight" / "preflight_report.json") or {}
    base_summary = load_json(run_dir / "base" / "summaries" / "run_summary.json") or {}
    aux_summary = load_json(run_dir / "aux" / "aux_summary.json") or {}
    stage2_summary = load_json(run_dir / "stage2" / "stage2_summary.json") or {}

    jobs = list(run_manifest.get("jobs", []))
    job_states = Counter(job.get("state", "unknown") for job in jobs)
    blocking_jobs = [job for job in jobs if job.get("stage") != "aggregate"]
    blocking_states = Counter(job.get("state", "unknown") for job in blocking_jobs)
    failed_jobs = extract_failed_jobs(run_manifest)
    overall_state = "failed" if failed_jobs else ("succeeded" if blocking_states and not blocking_states.get("pending") and not blocking_states.get("running") else "partial")

    return {
        "dataset_name": run_manifest.get("dataset_name", run_dir.parent.parent.name),
        "seed": run_manifest.get("seed", run_dir.parent.name),
        "run_id": run_manifest.get("run_id", run_dir.name),
        "run_dir": str(run_dir),
        "run_purpose": run_manifest.get("run_purpose", "unknown"),
        "base_epochs": run_manifest.get("base_epochs"),
        "aux_epochs": run_manifest.get("aux_epochs"),
        "train_records": _train_records(run_manifest, manifest_summary),
        "is_smoke": bool(run_manifest.get("is_smoke", False)),
        "notes": list(run_manifest.get("notes", [])),
        "overall_state": overall_state,
        "state_counts": dict(job_states),
        "failed_jobs": failed_jobs,
        "manifest_summary": manifest_summary,
        "preflight": preflight,
        "base_summary": base_summary,
        "aux_summary": aux_summary,
        "stage2_summary": stage2_summary,
        "strict_macro_f1": {
            "base": first_present(base_summary, [("results", "test", "metrics", "strict_three_class", "macro_f1")]),
            "stage1": first_present(stage2_summary, [("split_metrics", "test", "stage1", "strict_three_class", "macro_f1")]),
            "stage2": first_present(stage2_summary, [("final_test", "metrics", "strict_three_class", "macro_f1"), ("split_metrics", "test", "stage2", "strict_three_class", "macro_f1")]),
        },
        "binary_macro_f1": {
            "base": first_present(base_summary, [("results", "test", "metrics", "binary_auth_vs_edited", "macro_f1")]),
            "aux": first_present(aux_summary, [("splits", "test", "macro_f1"), ("split_results", "test", "metrics", "binary_auth_vs_edited", "macro_f1")]),
            "stage1": first_present(stage2_summary, [("split_metrics", "test", "stage1", "binary_auth_vs_edited", "macro_f1")]),
            "stage2": first_present(stage2_summary, [("final_test", "metrics", "binary_auth_vs_edited", "macro_f1"), ("split_metrics", "test", "stage2", "binary_auth_vs_edited", "macro_f1")]),
        },
        "change_stats": {
            "stage1_vs_base": stage2_summary.get("split_metrics", {}).get("test", {}).get("change_stats_vs_base"),
            "stage2_vs_base": stage2_summary.get("final_test", {}).get("change_stats_vs_base"),
            "stage2_vs_stage1": stage2_summary.get("final_test", {}).get("change_stats_vs_stage1"),
        },
        "best_veto": stage2_summary.get("selection"),
    }


def build_aggregate_report(results_root: Path = RESULTS_ROOT, run_dir: Optional[Path] = None, output_dir: Optional[Path] = None) -> Path:
    run_dirs = [run_dir] if run_dir is not None else list(iter_run_dirs(results_root))
    rows = [summarize_run(path) for path in run_dirs if path is not None and (path / "run_manifest.json").exists()]
    rows.sort(key=lambda row: (str(row.get("dataset_name", "")), str(row.get("seed", "")), str(row.get("run_id", ""))))

    summary = {
        "timestamp": now_utc(),
        "results_root": str(results_root),
        "run_dir": str(run_dir) if run_dir else None,
        "n_runs": len(rows),
        "rows": rows,
        "dataset_counts": dict(Counter(row.get("dataset_name") for row in rows)),
        "overall_counts": dict(Counter(row.get("overall_state") for row in rows)),
        "purpose_counts": dict(Counter(row.get("run_purpose") for row in rows)),
    }

    target_dir = output_dir or ((run_dir / "aggregate") if run_dir else (results_root / "_matrix" / "aggregate"))
    target_dir.mkdir(parents=True, exist_ok=True)
    json_path = target_dir / "matrix_aggregate.json"
    md_path = target_dir / "matrix_aggregate.md"
    failure_path = target_dir / "failure_table.md"

    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown_table(rows), encoding="utf-8")
    failure_path.write_text(render_failure_table(rows), encoding="utf-8")
    if run_dir is not None and len(rows) == 1:
        (target_dir / "run_report.md").write_text(render_run_report(rows[0]), encoding="utf-8")
    return json_path


def main() -> int:
    args = parse_args()
    path = build_aggregate_report(results_root=args.results_root, run_dir=args.run_dir, output_dir=args.output_dir)
    print(f"[aggregate] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
