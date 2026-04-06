from __future__ import annotations

import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence

EXPERIMENTS_DIR = Path(__file__).resolve().parents[1]
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from df40_eval_utils import ZIP_SPECS

from .manifests import build_manifest_bundle
from .paths import DATASET_RUNS_ROOT, RESULTS_ROOT, ensure_dir
from .registry import DATASET_NAMES, ensure_df40_extracted_locked, get_dataset_records


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PreflightReport:
    checks: List[CheckResult]

    @property
    def ok(self) -> bool:
        return all(check.ok for check in self.checks)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "checks": [
                {
                    "name": check.name,
                    "ok": check.ok,
                    "message": check.message,
                    "details": check.details,
                }
                for check in self.checks
            ],
        }


def check_gpu() -> CheckResult:
    import torch

    available = torch.cuda.is_available()
    details = {"device_count": torch.cuda.device_count() if available else 0}
    if available:
        try:
            details["device_name"] = torch.cuda.get_device_name(0)
        except Exception:
            details["device_name"] = "unknown"
    return CheckResult(
        name="gpu",
        ok=available,
        message="CUDA available" if available else "CUDA unavailable",
        details=details,
    )


def check_results_root_writable(results_root: Path | None = None) -> CheckResult:
    root = ensure_dir(results_root or DATASET_RUNS_ROOT)
    try:
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=root, prefix=".write_test_", suffix=".tmp", delete=False) as fh:
            fh.write(b"ok")
            temp_path = Path(fh.name)
        temp_path.unlink(missing_ok=True)
        ok = True
        message = "results root writable"
    except Exception as exc:
        ok = False
        message = f"results root not writable: {exc}"
    return CheckResult(name="results_root", ok=ok, message=message, details={"path": str(root)})


def check_dataset_images(dataset_name: str, ensure_df40_extract: bool = False) -> CheckResult:
    try:
        if dataset_name == "df40_extended":
            archive_check = check_df40_archives()
            extraction_check = check_df40_extraction_status()
            if not archive_check.ok or not extraction_check.ok:
                return CheckResult(
                    name=f"dataset_images:{dataset_name}",
                    ok=False,
                    message="DF40 archives or extraction are not ready",
                    details={
                        "archive_ok": archive_check.ok,
                        "extraction_ok": extraction_check.ok,
                        "archive_details": archive_check.details,
                        "extraction_details": extraction_check.details,
                    },
                )
        records = get_dataset_records(dataset_name, ensure_extract=ensure_df40_extract)
        missing = []
        for row in records:
            image_path = Path(row["image_path"])
            if not image_path.is_absolute():
                image_path = Path(__file__).resolve().parents[2] / image_path
            if not image_path.exists():
                missing.append(str(image_path))
        ok = not missing
        message = "all dataset images present" if ok else f"missing {len(missing)} images"
        details = {"n_records": len(records), "missing_examples": missing[:10]}
    except Exception as exc:
        ok = False
        message = f"dataset image check failed: {exc}"
        details = {"dataset_name": dataset_name}
    return CheckResult(name=f"dataset_images:{dataset_name}", ok=ok, message=message, details=details)


def check_df40_archives() -> CheckResult:
    statuses = {}
    for name, spec in ZIP_SPECS.items():
        zip_path = spec["zip_path"]
        statuses[name] = {
            "exists": zip_path.exists(),
            "is_file": zip_path.is_file(),
            "valid_zip": False,
        }
        if zip_path.exists() and zip_path.is_file():
            try:
                import zipfile

                with zipfile.ZipFile(zip_path) as zf:
                    zf.infolist()
                statuses[name]["valid_zip"] = True
            except Exception:
                statuses[name]["valid_zip"] = False
    ok = all(item["exists"] and item["is_file"] and item["valid_zip"] for item in statuses.values())
    message = "all DF40 archives available" if ok else "one or more DF40 archives missing or invalid"
    return CheckResult(name="df40_archives", ok=ok, message=message, details=statuses)


def check_df40_extraction_status() -> CheckResult:
    statuses = {}
    ok = True
    for name, spec in ZIP_SPECS.items():
        extract_dir = spec["extract_dir"]
        done_flag = extract_dir / ".extract_done"
        marker_glob = spec["marker_glob"]
        marker_exists = any(extract_dir.glob(marker_glob))
        entry_ok = extract_dir.exists() and done_flag.exists() and marker_exists
        statuses[name] = {
            "extract_dir": str(extract_dir),
            "done_flag": done_flag.exists(),
            "marker_exists": marker_exists,
            "ready": entry_ok,
        }
        ok = ok and entry_ok
    message = "all DF40 extraction markers present" if ok else "one or more DF40 assets not extracted"
    return CheckResult(name="df40_extraction", ok=ok, message=message, details=statuses)


def check_manifest_feasibility(dataset_name: str, seed: int = 42, ensure_df40_extract: bool = False) -> CheckResult:
    try:
        if dataset_name == "df40_extended":
            archive_check = check_df40_archives()
            extraction_check = check_df40_extraction_status()
            if not archive_check.ok or not extraction_check.ok:
                return CheckResult(
                    name=f"manifest_feasibility:{dataset_name}",
                    ok=False,
                    message="DF40 archives or extraction are not ready",
                    details={
                        "archive_ok": archive_check.ok,
                        "extraction_ok": extraction_check.ok,
                    },
                )
        rows = get_dataset_records(dataset_name, ensure_extract=ensure_df40_extract)
        canonical_rows, summary = build_manifest_bundle(rows, seed=seed)
        ok = bool(canonical_rows) and summary["counts_by_split"]["train"]["n_records"] > 0
        ok = ok and summary["counts_by_split"]["val"]["n_records"] > 0 and summary["counts_by_split"]["test"]["n_records"] > 0
        message = "manifest feasible" if ok else "manifest split check failed"
        details = summary
    except Exception as exc:
        ok = False
        message = f"manifest infeasible: {exc}"
        details = {"dataset_name": dataset_name}
    return CheckResult(name=f"manifest_feasibility:{dataset_name}", ok=ok, message=message, details=details)


def run_preflight(
    dataset_names: Sequence[str] | None = None,
    seed: int = 42,
    ensure_df40_extract: bool = False,
) -> PreflightReport:
    dataset_names = list(dataset_names or DATASET_NAMES)
    checks: List[CheckResult] = [
        check_gpu(),
        check_results_root_writable(),
    ]
    if "df40_extended" in dataset_names or ensure_df40_extract:
        checks.append(check_df40_archives())
        checks.append(check_df40_extraction_status())
    for dataset_name in dataset_names:
        checks.append(check_dataset_images(dataset_name, ensure_df40_extract=ensure_df40_extract))
        checks.append(check_manifest_feasibility(dataset_name, seed=seed, ensure_df40_extract=ensure_df40_extract))
    return PreflightReport(checks=checks)


__all__ = [
    "CheckResult",
    "PreflightReport",
    "check_gpu",
    "check_results_root_writable",
    "check_dataset_images",
    "check_df40_archives",
    "check_df40_extraction_status",
    "check_manifest_feasibility",
    "ensure_df40_extracted_locked",
    "run_preflight",
]
