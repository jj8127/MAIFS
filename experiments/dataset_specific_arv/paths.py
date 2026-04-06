from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PACKAGE_ROOT
EXPERIMENTS_ROOT = REPO_ROOT / "experiments"
RESULTS_ROOT = EXPERIMENTS_ROOT / "results"
DATASETS_ROOT = REPO_ROOT / "datasets"

DF40_ROOT = DATASETS_ROOT / "external_new" / "DF40"
DF40_EXTRACTED_ROOT = DF40_ROOT / "extracted_minimal"

DATASET_RUNS_ROOT = RESULTS_ROOT / "dataset_runs"
MANIFEST_ROOT = RESULTS_ROOT / "dataset_manifests"
PREFLIGHT_ROOT = RESULTS_ROOT / "dataset_preflight"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


__all__ = [
    "PACKAGE_ROOT",
    "REPO_ROOT",
    "EXPERIMENTS_ROOT",
    "RESULTS_ROOT",
    "DATASETS_ROOT",
    "DF40_ROOT",
    "DF40_EXTRACTED_ROOT",
    "DATASET_RUNS_ROOT",
    "MANIFEST_ROOT",
    "PREFLIGHT_ROOT",
    "ensure_dir",
]
