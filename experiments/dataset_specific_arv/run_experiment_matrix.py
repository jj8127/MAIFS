#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = SCRIPT_DIR.parent
REPO_ROOT = EXPERIMENTS_DIR.parent
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from dataset_specific_arv.aggregate import build_aggregate_report
from dataset_specific_arv.manifests import build_manifest_bundle
from dataset_specific_arv.paths import DATASET_RUNS_ROOT, ensure_dir
from dataset_specific_arv.preflight import check_df40_archives, check_df40_extraction_status, run_preflight
from dataset_specific_arv.registry import DATASET_NAMES, get_dataset_records


DEFAULT_CPU_WORKERS = max(2, (os_cpu_count := (__import__("os").cpu_count() or 4)) // 2)
DEFAULT_SEEDS = "42"
DEFAULT_INIT_MODE = "scratch"
DEFAULT_DF40_TRACK = "extended"
DEFAULT_RUN_PURPOSE = "full"
RUN_PURPOSE_CHOICES = ("smoke", "pilot", "full")
RUN_PURPOSE_PROFILES = {
    "smoke": {
        "base_epochs": 1,
        "aux_epochs": 1,
        "base_batch_size": 32,
        "aux_batch_size": 32,
        "eval_batch_size": 64,
        "num_workers": 0,
        "models": ["logreg"],
        "notes": ["Smoke profile: minimal end-to-end validation."],
    },
    "pilot": {
        "base_epochs": 30,
        "aux_epochs": 20,
        "base_batch_size": 32,
        "aux_batch_size": 32,
        "eval_batch_size": 64,
        "num_workers": 0,
        "models": ["logreg", "xgb"],
        "notes": ["Pilot profile: single-dataset real-performance validation budget."],
    },
    "full": {
        "base_epochs": 30,
        "aux_epochs": 20,
        "base_batch_size": 32,
        "aux_batch_size": 32,
        "eval_batch_size": 64,
        "num_workers": 0,
        "models": ["logreg", "xgb"],
        "notes": ["Full profile: core dataset matrix budget after pilot acceptance."],
    },
}


@dataclass(frozen=True)
class RunBundle:
    dataset_name: str
    seed: int
    run_id: str
    run_dir: Path
    manifests_dir: Path
    preflight_dir: Path
    logs_dir: Path
    base_dir: Path
    aux_dir: Path
    stage2_dir: Path
    aggregate_dir: Path
    source_records_path: Path
    canonical_manifest_path: Path
    split_paths: Dict[str, Path]
    manifest_summary_path: Path
    preflight_report_path: Path


@dataclass
class Job:
    job_id: str
    dataset_name: str
    seed: int
    stage: str
    kind: str
    deps: List[str]
    run_dir: Path
    command: List[str]
    stdout_path: Path
    stderr_path: Path
    artifacts: Dict[str, str] = field(default_factory=dict)
    state: str = "pending"
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    return_code: Optional[int] = None
    error: Optional[str] = None


def now_utc() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def short_hash(payload: str) -> str:
    import hashlib

    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]


def slugify(value: str) -> str:
    parts = []
    for ch in value.lower():
        parts.append(ch if ch.isalnum() else "-")
    slug = "".join(parts).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "run"


def json_dump(path: Path, payload: Mapping[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def jsonl_dump(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-GPU dataset-specific ARV matrix runner")
    parser.add_argument("--datasets", nargs="+", default=list(DATASET_NAMES), choices=list(DATASET_NAMES))
    parser.add_argument("--seeds", type=str, default=DEFAULT_SEEDS)
    parser.add_argument("--gpu-slots", type=int, default=1)
    parser.add_argument("--cpu-workers", type=int, default=DEFAULT_CPU_WORKERS)
    parser.add_argument("--run-purpose", choices=RUN_PURPOSE_CHOICES, default=DEFAULT_RUN_PURPOSE)
    parser.add_argument("--init-mode", choices=("scratch", "imagenet", "checkpoint"), default=DEFAULT_INIT_MODE)
    parser.add_argument("--df40-track", choices=("minimal", "extended"), default=DEFAULT_DF40_TRACK)
    parser.add_argument("--results-root", type=Path, default=DATASET_RUNS_ROOT)
    parser.add_argument("--matrix-root", type=Path, default=DATASET_RUNS_ROOT / "_matrix")
    parser.add_argument("--run-tag", type=str, default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--base-epochs", type=int, default=None)
    parser.add_argument("--aux-epochs", type=int, default=None)
    parser.add_argument("--base-batch-size", type=int, default=None)
    parser.add_argument("--aux-batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--base-lr", type=float, default=2e-4)
    parser.add_argument("--aux-lr", type=float, default=3e-5)
    parser.add_argument("--base-checkpoint", type=Path, default=None)
    parser.add_argument("--aux-checkpoint", type=Path, default=None)
    parser.add_argument("--taus", nargs="+", type=float, default=[0.35, 0.45, 0.55, 0.65])
    parser.add_argument("--pos-weights", nargs="+", type=float, default=[1.0, 2.0, 4.0])
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--weighting-alpha", type=float, default=1.0)
    parser.add_argument("--ai-lock-threshold", type=float, default=0.5)
    parser.add_argument("--notes", nargs="*", default=[])
    return parser.parse_args()


def parse_seeds(raw: str) -> List[int]:
    seeds = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not seeds:
        raise ValueError("At least one seed is required.")
    return seeds


def _group_split_check(rows: Sequence[Mapping[str, Any]], seed: int) -> Dict[str, Any]:
    canonical_rows, _ = build_manifest_bundle(rows, seed=seed)
    group_to_splits: Dict[str, set[str]] = {}
    for row in canonical_rows:
        group_to_splits.setdefault(str(row["group_id"]), set()).add(str(row["split"]))
    leaking = {
        group_id: sorted(list(splits))
        for group_id, splits in group_to_splits.items()
        if len(splits) > 1
    }
    return {
        "ok": not leaking,
        "n_groups": len(group_to_splits),
        "n_leaking_groups": len(leaking),
        "leak_examples": dict(list(leaking.items())[:5]),
    }


def _resolve_df40_gate(seed: int) -> Dict[str, Any]:
    archive_check = check_df40_archives()
    extraction_check = check_df40_extraction_status()
    ready = archive_check.ok and extraction_check.ok
    details: Dict[str, Any] = {
        "ready": False,
        "seed_checked": seed,
        "archive_ok": archive_check.ok,
        "extraction_ok": extraction_check.ok,
        "archive_details": archive_check.details,
        "extraction_details": extraction_check.details,
        "grouped_split_ok": False,
        "grouped_split_details": None,
    }
    if not ready:
        return details

    try:
        rows = get_dataset_records("df40_extended", ensure_extract=False)
        split_check = _group_split_check(rows, seed=seed)
        details["grouped_split_ok"] = split_check["ok"]
        details["grouped_split_details"] = split_check
        details["ready"] = split_check["ok"]
        return details
    except Exception as exc:
        details["grouped_split_details"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        return details


def resolve_run_configuration(args: argparse.Namespace) -> Dict[str, Any]:
    profile = RUN_PURPOSE_PROFILES[args.run_purpose]
    requested_datasets = list(args.datasets)
    base_epochs_overridden = args.base_epochs is not None
    aux_epochs_overridden = args.aux_epochs is not None
    base_batch_overridden = args.base_batch_size is not None
    aux_batch_overridden = args.aux_batch_size is not None
    eval_batch_overridden = args.eval_batch_size is not None
    num_workers_overridden = args.num_workers is not None
    models_overridden = args.models is not None
    base_epochs = args.base_epochs if base_epochs_overridden else profile["base_epochs"]
    aux_epochs = args.aux_epochs if aux_epochs_overridden else profile["aux_epochs"]
    base_batch_size = args.base_batch_size if base_batch_overridden else profile["base_batch_size"]
    aux_batch_size = args.aux_batch_size if aux_batch_overridden else profile["aux_batch_size"]
    eval_batch_size = args.eval_batch_size if eval_batch_overridden else profile["eval_batch_size"]
    num_workers = args.num_workers if num_workers_overridden else profile["num_workers"]
    models = list(args.models) if models_overridden else list(profile["models"])

    notes = list(profile["notes"])
    notes.extend(list(getattr(args, "notes", [])))
    notes.append(f"run_purpose={args.run_purpose}")
    notes.append(f"base_epochs={base_epochs}")
    notes.append(f"aux_epochs={aux_epochs}")
    notes.append(f"base_batch_size={base_batch_size}")
    notes.append(f"aux_batch_size={aux_batch_size}")
    notes.append(f"eval_batch_size={eval_batch_size}")
    notes.append(f"num_workers={num_workers}")
    notes.append(f"models={','.join(models)}")
    notes.append(f"is_smoke={args.run_purpose == 'smoke'}")
    if any([base_epochs_overridden, aux_epochs_overridden, base_batch_overridden, aux_batch_overridden, eval_batch_overridden, num_workers_overridden, models_overridden]):
        override_bits = []
        if base_epochs_overridden:
            override_bits.append(f"base_epochs={args.base_epochs}")
        if aux_epochs_overridden:
            override_bits.append(f"aux_epochs={args.aux_epochs}")
        if base_batch_overridden:
            override_bits.append(f"base_batch_size={args.base_batch_size}")
        if aux_batch_overridden:
            override_bits.append(f"aux_batch_size={args.aux_batch_size}")
        if eval_batch_overridden:
            override_bits.append(f"eval_batch_size={args.eval_batch_size}")
        if num_workers_overridden:
            override_bits.append(f"num_workers={args.num_workers}")
        if models_overridden:
            override_bits.append(f"models={','.join(args.models)}")
        notes.append("explicit overrides applied: " + ", ".join(override_bits))
    else:
        notes.append(f"default {args.run_purpose} profile applied")

    df40_gate = None
    planned_datasets = list(requested_datasets)
    if args.run_purpose == "full" and "df40_extended" in requested_datasets:
        df40_gate = _resolve_df40_gate(seed=parse_seeds(args.seeds)[0])
        if not df40_gate["ready"]:
            planned_datasets = [name for name in requested_datasets if name != "df40_extended"]
            notes.append(
                "df40_extended excluded from the full run because archive/extraction/grouped-split readiness failed."
            )

    if requested_datasets and not planned_datasets:
        raise ValueError("No datasets remain after applying the full-run DF40 readiness gate.")

    args.requested_datasets = requested_datasets
    args.datasets = planned_datasets
    args.base_epochs = base_epochs
    args.aux_epochs = aux_epochs
    args.base_batch_size = base_batch_size
    args.aux_batch_size = aux_batch_size
    args.eval_batch_size = eval_batch_size
    args.num_workers = num_workers
    args.models = models
    args.is_smoke = args.run_purpose == "smoke"
    args.run_notes = notes
    args.df40_gate = df40_gate
    return {
        "requested_datasets": requested_datasets,
        "planned_datasets": planned_datasets,
        "run_purpose": args.run_purpose,
        "base_epochs": base_epochs,
        "aux_epochs": aux_epochs,
        "base_batch_size": base_batch_size,
        "aux_batch_size": aux_batch_size,
        "eval_batch_size": eval_batch_size,
        "num_workers": num_workers,
        "models": models,
        "is_smoke": args.is_smoke,
        "notes": notes,
        "df40_gate": df40_gate,
    }


def build_bundle(results_root: Path, dataset_name: str, seed: int, run_id: str) -> RunBundle:
    run_dir = results_root / dataset_name / str(seed) / run_id
    manifests_dir = run_dir / "manifests"
    preflight_dir = run_dir / "preflight"
    logs_dir = run_dir / "logs"
    return RunBundle(
        dataset_name=dataset_name,
        seed=seed,
        run_id=run_id,
        run_dir=run_dir,
        manifests_dir=manifests_dir,
        preflight_dir=preflight_dir,
        logs_dir=logs_dir,
        base_dir=run_dir / "base",
        aux_dir=run_dir / "aux",
        stage2_dir=run_dir / "stage2",
        aggregate_dir=run_dir / "aggregate",
        source_records_path=manifests_dir / "source_records.jsonl",
        canonical_manifest_path=manifests_dir / "canonical_manifest.jsonl",
        split_paths={
            "train": manifests_dir / "train.jsonl",
            "val": manifests_dir / "val.jsonl",
            "test": manifests_dir / "test.jsonl",
        },
        manifest_summary_path=manifests_dir / "manifest_summary.json",
        preflight_report_path=preflight_dir / "preflight_report.json",
    )


def job_to_json(job: Job) -> Dict[str, Any]:
    return {
        "job_id": job.job_id,
        "dataset_name": job.dataset_name,
        "seed": job.seed,
        "stage": job.stage,
        "kind": job.kind,
        "deps": job.deps,
        "run_dir": str(job.run_dir),
        "command": job.command,
        "stdout_path": str(job.stdout_path),
        "stderr_path": str(job.stderr_path),
        "artifacts": job.artifacts,
        "state": job.state,
        "started_at": job.started_at,
        "finished_at": job.finished_at,
        "return_code": job.return_code,
        "error": job.error,
    }


def read_manifest_metadata(bundle: RunBundle) -> Dict[str, Any]:
    if not bundle.manifest_summary_path.exists():
        return {"train_records": None}
    try:
        payload = json.loads(bundle.manifest_summary_path.read_text(encoding="utf-8"))
    except Exception:
        return {"train_records": None}
    train_records = (
        payload.get("counts_by_split", {})
        .get("train", {})
        .get("n_records")
    )
    return {"train_records": train_records}


def save_run_manifest(bundle: RunBundle, jobs: Mapping[str, Job], args: argparse.Namespace) -> Path:
    manifest_meta = read_manifest_metadata(bundle)
    payload = {
        "timestamp": now_utc(),
        "dataset_name": bundle.dataset_name,
        "seed": bundle.seed,
        "run_id": bundle.run_id,
        "run_dir": str(bundle.run_dir),
        "run_purpose": args.run_purpose,
        "base_epochs": args.base_epochs,
        "aux_epochs": args.aux_epochs,
        "base_batch_size": args.base_batch_size,
        "aux_batch_size": args.aux_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "num_workers": args.num_workers,
        "models": list(args.models),
        "train_records": manifest_meta["train_records"],
        "is_smoke": args.is_smoke,
        "notes": list(getattr(args, "run_notes", [])),
        "requested_datasets": list(getattr(args, "requested_datasets", [bundle.dataset_name])),
        "planned_datasets": list(args.datasets),
        "df40_gate": getattr(args, "df40_gate", None),
        "init_mode": args.init_mode,
        "df40_track": args.df40_track,
        "dry_run": bool(args.dry_run),
        "manifests": {
            "source_records": str(bundle.source_records_path),
            "canonical_manifest": str(bundle.canonical_manifest_path),
            "split_paths": {name: str(path) for name, path in bundle.split_paths.items()},
            "manifest_summary": str(bundle.manifest_summary_path),
            "preflight_report": str(bundle.preflight_report_path),
        },
        "jobs": [job_to_json(job) for job in jobs.values()],
    }
    path = bundle.run_dir / "run_manifest.json"
    json_dump(path, payload)
    return path


def save_matrix_manifest(path: Path, jobs: Mapping[str, Job], args: argparse.Namespace, matrix_id: str) -> None:
    payload = {
        "timestamp": now_utc(),
        "matrix_id": matrix_id,
        "requested_datasets": list(getattr(args, "requested_datasets", list(args.datasets))),
        "datasets": list(args.datasets),
        "seeds": parse_seeds(args.seeds),
        "run_purpose": args.run_purpose,
        "base_epochs": args.base_epochs,
        "aux_epochs": args.aux_epochs,
        "base_batch_size": args.base_batch_size,
        "aux_batch_size": args.aux_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "num_workers": args.num_workers,
        "models": list(args.models),
        "is_smoke": args.is_smoke,
        "notes": list(getattr(args, "run_notes", [])),
        "df40_gate": getattr(args, "df40_gate", None),
        "init_mode": args.init_mode,
        "df40_track": args.df40_track,
        "gpu_slots": args.gpu_slots,
        "cpu_workers": args.cpu_workers,
        "dry_run": bool(args.dry_run),
        "jobs": [job_to_json(job) for job in jobs.values()],
    }
    json_dump(path, payload)


def load_existing_run_manifest(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for item in payload.get("jobs", []):
        job_id = item.get("job_id")
        if job_id:
            out[str(job_id)] = dict(item)
    return out


def apply_resume_state(jobs: Mapping[str, Job], existing: Mapping[str, Dict[str, Any]]) -> None:
    for job in jobs.values():
        prior = existing.get(job.job_id)
        if not prior or prior.get("state") != "succeeded":
            continue
        job.state = "succeeded"
        job.started_at = prior.get("started_at")
        job.finished_at = prior.get("finished_at")
        job.return_code = prior.get("return_code", 0)
        job.artifacts = dict(prior.get("artifacts", {}))


def plan_jobs(bundle: RunBundle, args: argparse.Namespace) -> Dict[str, Job]:
    def jid(stage: str) -> str:
        return f"{bundle.dataset_name}:{bundle.seed}:{stage}"

    def mk(stage: str, kind: str, deps: Sequence[str], command: Sequence[str], artifacts: Mapping[str, str]) -> Job:
        return Job(
            job_id=jid(stage),
            dataset_name=bundle.dataset_name,
            seed=bundle.seed,
            stage=stage,
            kind=kind,
            deps=list(deps),
            run_dir=bundle.run_dir,
            command=list(command),
            stdout_path=bundle.logs_dir / f"{stage}.stdout.log",
            stderr_path=bundle.logs_dir / f"{stage}.stderr.log",
            artifacts=dict(artifacts),
        )

    base_pred_train = bundle.base_dir / "predictions" / "base_train_predictions.jsonl"
    base_pred_val = bundle.base_dir / "predictions" / "base_val_predictions.jsonl"
    base_pred_test = bundle.base_dir / "predictions" / "base_test_predictions.jsonl"
    aux_pred_train = bundle.aux_dir / "aux_predictions_train.jsonl"
    aux_pred_val = bundle.aux_dir / "aux_predictions_val.jsonl"
    aux_pred_test = bundle.aux_dir / "aux_predictions_test.jsonl"

    base_cmd = [
        str(SCRIPT_DIR / "base_trainer.py"),
        "--manifest-train",
        str(bundle.split_paths["train"]),
        "--manifest-val",
        str(bundle.split_paths["val"]),
        "--manifest-test",
        str(bundle.split_paths["test"]),
        "--output-dir",
        str(bundle.base_dir),
        "--seed",
        str(bundle.seed),
        "--epochs",
        str(args.base_epochs),
        "--batch-size",
        str(args.base_batch_size),
        "--lr",
        str(args.base_lr),
        "--num-workers",
        str(args.num_workers),
        "--init-mode",
        args.init_mode,
    ]
    if args.init_mode == "checkpoint" and args.base_checkpoint:
        base_cmd.extend(["--checkpoint", str(args.base_checkpoint)])

    aux_cmd = [
        str(SCRIPT_DIR / "aux_trainer.py"),
        "--dataset-name",
        bundle.dataset_name,
        "--seed",
        str(bundle.seed),
        "--run-id",
        bundle.run_id,
        "--output-root",
        str(args.results_root),
        "--manifest-train",
        str(bundle.split_paths["train"]),
        "--manifest-val",
        str(bundle.split_paths["val"]),
        "--manifest-test",
        str(bundle.split_paths["test"]),
        "--base-preds-train",
        str(base_pred_train),
        "--base-preds-val",
        str(base_pred_val),
        "--base-preds-test",
        str(base_pred_test),
        "--epochs",
        str(args.aux_epochs),
        "--batch-size",
        str(args.aux_batch_size),
        "--eval-batch-size",
        str(args.eval_batch_size),
        "--num-workers",
        str(args.num_workers),
        "--lr",
        str(args.aux_lr),
        "--init-mode",
        args.init_mode,
    ]
    if args.init_mode == "checkpoint" and args.aux_checkpoint:
        aux_cmd.extend(["--init-checkpoint", str(args.aux_checkpoint)])

    stage2_cmd = [
        str(SCRIPT_DIR / "stage2_trainer.py"),
        "--dataset-name",
        bundle.dataset_name,
        "--seed",
        str(bundle.seed),
        "--train-base",
        str(base_pred_train),
        "--train-aux",
        str(aux_pred_train),
        "--val-base",
        str(base_pred_val),
        "--val-aux",
        str(aux_pred_val),
        "--test-base",
        str(base_pred_test),
        "--test-aux",
        str(aux_pred_test),
        "--run-dir",
        str(bundle.stage2_dir),
        "--weighting-alpha",
        str(args.weighting_alpha),
        "--ai-lock-threshold",
        str(args.ai_lock_threshold),
    ]
    if args.taus:
        stage2_cmd.extend(["--taus", *[str(v) for v in args.taus]])
    if args.pos_weights:
        stage2_cmd.extend(["--pos-weights", *[str(v) for v in args.pos_weights]])
    if args.models:
        stage2_cmd.extend(["--models", *list(args.models)])

    aggregate_cmd = [
        str(SCRIPT_DIR / "aggregate.py"),
        "--run-dir",
        str(bundle.run_dir),
        "--output-dir",
        str(bundle.aggregate_dir),
    ]

    jobs = {
        "manifest": mk(
            "manifest",
            "cpu",
            [],
            ["internal", "manifest"],
            {
                "source_records": str(bundle.source_records_path),
                "canonical_manifest": str(bundle.canonical_manifest_path),
                "manifest_summary": str(bundle.manifest_summary_path),
                "train_manifest": str(bundle.split_paths["train"]),
                "val_manifest": str(bundle.split_paths["val"]),
                "test_manifest": str(bundle.split_paths["test"]),
            },
        ),
        "preflight": mk(
            "preflight",
            "cpu",
            [jid("manifest")],
            ["internal", "preflight"],
            {"preflight_report": str(bundle.preflight_report_path)},
        ),
        "base": mk(
            "base",
            "gpu",
            [jid("manifest"), jid("preflight")],
            base_cmd,
            {
                "checkpoint": str(bundle.base_dir / "checkpoints" / "base_best.pth"),
                "summary": str(bundle.base_dir / "summaries" / "run_summary.json"),
                "pred_train": str(base_pred_train),
                "pred_val": str(base_pred_val),
                "pred_test": str(base_pred_test),
            },
        ),
        "aux": mk(
            "aux",
            "gpu",
            [jid("base")],
            aux_cmd,
            {
                "checkpoint": str(bundle.aux_dir / "best.pth"),
                "summary": str(bundle.aux_dir / "aux_summary.json"),
                "pred_train": str(aux_pred_train),
                "pred_val": str(aux_pred_val),
                "pred_test": str(aux_pred_test),
            },
        ),
        "stage2": mk(
            "stage2",
            "cpu",
            [jid("aux")],
            stage2_cmd,
            {
                "model": str(bundle.stage2_dir / "stage2_model.pkl"),
                "summary": str(bundle.stage2_dir / "stage2_summary.json"),
                "pred_test": str(bundle.stage2_dir / "stage2_test_predictions.jsonl"),
            },
        ),
        "aggregate": mk(
            "aggregate",
            "cpu",
            [jid("stage2")],
            aggregate_cmd,
            {
                "aggregate_json": str(bundle.aggregate_dir / "matrix_aggregate.json"),
                "aggregate_md": str(bundle.aggregate_dir / "matrix_aggregate.md"),
                "run_report_md": str(bundle.aggregate_dir / "run_report.md"),
                "failure_md": str(bundle.aggregate_dir / "failure_table.md"),
            },
        ),
    }
    return jobs


def run_manifest_job(bundle: RunBundle) -> Dict[str, Any]:
    ensure_df40_extract = bundle.dataset_name == "df40_extended"
    rows = get_dataset_records(bundle.dataset_name, ensure_extract=ensure_df40_extract)
    canonical_rows, summary = build_manifest_bundle(rows, seed=bundle.seed)
    split_rows = {
        split: [row for row in canonical_rows if row["split"] == split]
        for split in ("train", "val", "test")
    }
    ensure_dir(bundle.run_dir)
    jsonl_dump(bundle.source_records_path, rows)
    jsonl_dump(bundle.canonical_manifest_path, canonical_rows)
    for split, path in bundle.split_paths.items():
        jsonl_dump(path, split_rows[split])
    summary_payload = {
        **summary,
        "timestamp": now_utc(),
        "dataset_name": bundle.dataset_name,
        "seed": bundle.seed,
        "run_id": bundle.run_id,
        "run_dir": str(bundle.run_dir),
        "split_paths": {name: str(path) for name, path in bundle.split_paths.items()},
    }
    json_dump(bundle.manifest_summary_path, summary_payload)
    return summary_payload


def run_preflight_job(bundle: RunBundle) -> Dict[str, Any]:
    report = run_preflight(
        dataset_names=[bundle.dataset_name],
        seed=bundle.seed,
        ensure_df40_extract=(bundle.dataset_name == "df40_extended"),
    ).as_dict()
    report.update(
        {
            "timestamp": now_utc(),
            "dataset_name": bundle.dataset_name,
            "seed": bundle.seed,
            "run_id": bundle.run_id,
            "run_dir": str(bundle.run_dir),
        }
    )
    json_dump(bundle.preflight_report_path, report)
    return report


def execute_subprocess(command: Sequence[str], stdout_path: Path, stderr_path: Path) -> None:
    import subprocess
    import os

    ensure_dir(stdout_path.parent)
    ensure_dir(stderr_path.parent)
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    with stdout_path.open("a", encoding="utf-8") as out_fh, stderr_path.open("a", encoding="utf-8") as err_fh:
        proc = subprocess.run([sys.executable, *command], cwd=str(REPO_ROOT), env=env, stdout=out_fh, stderr=err_fh)
    if proc.returncode != 0:
        raise RuntimeError(f"{Path(command[0]).name} failed with exit code {proc.returncode}")


def execute_job(job: Job, bundle: RunBundle) -> Dict[str, Any]:
    if job.stage == "manifest":
        return run_manifest_job(bundle)
    if job.stage == "preflight":
        return run_preflight_job(bundle)
    execute_subprocess(job.command, job.stdout_path, job.stderr_path)
    return {"ok": True}


def deps_succeeded(job: Job, jobs: Mapping[str, Job]) -> bool:
    return all(jobs[dep].state == "succeeded" for dep in job.deps)


def deps_failed(job: Job, jobs: Mapping[str, Job]) -> bool:
    return any(jobs[dep].state in {"failed", "skipped"} for dep in job.deps)


def mark_failed(job: Job, exc: BaseException) -> None:
    job.state = "failed"
    job.finished_at = now_utc()
    job.return_code = 1
    job.error = f"{type(exc).__name__}: {exc}"


def mark_skipped(job: Job, reason: str) -> None:
    job.state = "skipped"
    job.finished_at = now_utc()
    job.return_code = None
    job.error = reason


def update_job_artifacts(job: Job) -> None:
    existing = {key: path for key, path in job.artifacts.items() if Path(path).exists()}
    if existing:
        job.artifacts = existing


def run_matrix(args: argparse.Namespace) -> Path:
    if args.gpu_slots != 1:
        raise ValueError("This runner currently supports exactly one GPU slot.")

    resolve_run_configuration(args)
    results_root = ensure_dir(Path(args.results_root))
    matrix_root = ensure_dir(Path(args.matrix_root))
    seeds = parse_seeds(args.seeds)
    matrix_id = args.run_tag.strip() or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    matrix_dir = ensure_dir(matrix_root / matrix_id)
    matrix_manifest_path = matrix_dir / "matrix_manifest.json"

    bundles: Dict[str, RunBundle] = {}
    jobs: Dict[str, Job] = {}

    for seed in seeds:
        for dataset_name in args.datasets:
            run_id = f"{slugify(dataset_name)}-{seed}-{datetime.utcnow().strftime('%H%M%S')}-{short_hash(f'{dataset_name}:{seed}:{matrix_id}')}"
            bundle = build_bundle(results_root, dataset_name, seed, run_id)
            bundle_jobs = plan_jobs(bundle, args)
            if args.resume:
                existing = load_existing_run_manifest(bundle.run_dir / "run_manifest.json")
                apply_resume_state(bundle_jobs, existing)
            bundles[str(bundle.run_dir)] = bundle
            for job in bundle_jobs.values():
                jobs[job.job_id] = job
            save_run_manifest(bundle, bundle_jobs, args)

    save_matrix_manifest(matrix_manifest_path, jobs, args, matrix_id)
    if args.dry_run:
        return matrix_manifest_path

    lock = threading.Lock()
    cpu_executor = ThreadPoolExecutor(max_workers=max(1, args.cpu_workers))
    cpu_futures: Dict[Future, str] = {}

    def sync_manifests() -> None:
        with lock:
            save_matrix_manifest(matrix_manifest_path, jobs, args, matrix_id)
            per_run: Dict[str, Dict[str, Job]] = {}
            for job in jobs.values():
                per_run.setdefault(str(job.run_dir), {})[job.stage] = job
            for run_dir, run_jobs in per_run.items():
                save_run_manifest(bundles[run_dir], run_jobs, args)

    try:
        while True:
            progress = False

            for job in jobs.values():
                if job.state == "pending" and deps_failed(job, jobs):
                    mark_skipped(job, "dependency_failed")
                    progress = True

            done = [future for future in cpu_futures if future.done()]
            for future in done:
                job_id = cpu_futures.pop(future)
                job = jobs[job_id]
                try:
                    future.result()
                    if job.state != "failed":
                        job.state = "succeeded"
                        job.return_code = 0
                        update_job_artifacts(job)
                except BaseException as exc:
                    mark_failed(job, exc)
                finally:
                    job.finished_at = now_utc()
                progress = True

            sync_manifests()

            for job in jobs.values():
                if job.state == "pending" and job.kind == "cpu" and deps_succeeded(job, jobs):
                    job.state = "running"
                    job.started_at = now_utc()
                    bundle = bundles[str(job.run_dir)]
                    cpu_futures[cpu_executor.submit(execute_job, job, bundle)] = job.job_id
                    progress = True

            gpu_job = next(
                (job for job in jobs.values() if job.state == "pending" and job.kind == "gpu" and deps_succeeded(job, jobs)),
                None,
            )
            if gpu_job is not None:
                gpu_job.state = "running"
                gpu_job.started_at = now_utc()
                bundle = bundles[str(gpu_job.run_dir)]
                try:
                    execute_job(gpu_job, bundle)
                    gpu_job.state = "succeeded"
                    gpu_job.return_code = 0
                    update_job_artifacts(gpu_job)
                except BaseException as exc:
                    mark_failed(gpu_job, exc)
                finally:
                    gpu_job.finished_at = now_utc()
                    sync_manifests()
                progress = True
                continue

            if not cpu_futures and not any(job.state == "pending" for job in jobs.values()):
                break

            if not progress and cpu_futures:
                wait(list(cpu_futures.keys()), return_when=FIRST_COMPLETED)
                continue

            if not progress:
                break
    finally:
        cpu_executor.shutdown(wait=True, cancel_futures=False)

    sync_manifests()
    for bundle in bundles.values():
        build_aggregate_report(results_root=results_root, run_dir=bundle.run_dir, output_dir=bundle.aggregate_dir)
    build_aggregate_report(results_root=results_root)
    return matrix_manifest_path


def main() -> int:
    args = parse_args()
    try:
        manifest_path = run_matrix(args)
    except Exception as exc:
        print(f"[dataset_specific_arv] failed: {exc}", file=sys.stderr)
        return 1
    print(f"[dataset_specific_arv] matrix manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
