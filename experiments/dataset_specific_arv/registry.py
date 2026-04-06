from __future__ import annotations

import fcntl
import json
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Sequence

EXPERIMENTS_DIR = Path(__file__).resolve().parents[1]
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from df40_eval_utils import (
    DF40_ROOT as DF40_DATA_ROOT,
    ZIP_SPECS,
    ensure_extracted,
    is_valid_zip,
    iter_df40_extended_records,
    iter_df40_records,
)

from .paths import DATASETS_ROOT, DF40_EXTRACTED_ROOT, PACKAGE_ROOT, RESULTS_ROOT, ensure_dir

VALID_LABELS = {"authentic", "manipulated", "ai_generated"}
CANONICAL_SOURCE_KEYS = {"image_path", "true_label", "dataset_name", "group_id", "split_policy", "audit_only"}
DEFAULT_SPLIT_POLICY = "auto"


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    split_policy: str
    source_kind: str
    description: str
    source_root: Path | None = None
    source_glob: str | None = None
    max_per_group: int | None = 8


def _latest_jsonl(paths: Iterable[Path]) -> Path:
    candidates = sorted({p for p in paths if p.exists() and p.is_file()}, key=lambda p: p.as_posix())
    if not candidates:
        raise FileNotFoundError("No JSONL candidates found")
    return candidates[-1]


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _audit_only(raw: Mapping[str, Any], keep_keys: Sequence[str] = ()) -> Dict[str, Any]:
    audit: Dict[str, Any] = {}
    keep = set(CANONICAL_SOURCE_KEYS) | set(keep_keys)
    for key, value in raw.items():
        if key not in keep:
            audit[key] = value
    return audit


def _derive_group_id(dataset_name: str, raw: Mapping[str, Any]) -> str:
    image_path = str(raw.get("image_path", "")).strip()
    sub_type = str(raw.get("sub_type", "")).strip() or "no_subtype"
    base_image_id = str(raw.get("base_image_id", "")).strip()
    source_family = str(raw.get("source_family", "")).strip()
    path = Path(image_path) if image_path else None

    if dataset_name == "df40_extended" and base_image_id:
        return f"{dataset_name}:{source_family or 'df40'}:{base_image_id}"
    if base_image_id:
        return f"{dataset_name}:{sub_type}:{base_image_id}"
    if path is not None:
        parent = path.parent.as_posix() or "root"
        stem = path.stem or "sample"
        return f"{dataset_name}:{sub_type}:{parent}:{stem}"
    return f"{dataset_name}:{sub_type}:{raw.get('true_label', 'unknown')}"


def _make_core_adapter_rows(dataset_name: str, source_path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen_paths: set[str] = set()
    for raw in _read_jsonl(source_path):
        image_path = str(raw.get("image_path", "")).strip()
        true_label = str(raw.get("true_label", "")).strip()
        if not image_path or true_label not in VALID_LABELS:
            continue
        if image_path in seen_paths:
            continue
        seen_paths.add(image_path)
        rows.append(
            {
                "dataset_name": dataset_name,
                "image_path": image_path,
                "true_label": true_label,
                "group_id": _derive_group_id(dataset_name, raw),
                "split_policy": DEFAULT_SPLIT_POLICY,
                "audit_only": {
                    **_audit_only(raw),
                    "source_jsonl": str(source_path.relative_to(PACKAGE_ROOT)),
                },
            }
        )
    rows.sort(key=lambda row: row["image_path"])
    return rows


BASE_SOURCE_DIR = RESULTS_ROOT / "phase2_patha"
DSC_SOURCE_DIR = RESULTS_ROOT / "phase2_patha_case3_scale300_dsC"
OPENDI_SOURCE_DIR = RESULTS_ROOT / "phase2_patha_case3_opensdi_scale300"
AIGENPROXY_SOURCE_DIR = RESULTS_ROOT / "phase2_patha_case3_aigenproxy_scale300"


DATASET_SPECS: Dict[str, DatasetSpec] = {
    "base": DatasetSpec(
        name="base",
        split_policy=DEFAULT_SPLIT_POLICY,
        source_kind="core_jsonl",
        description="PathA base dataset records",
        source_root=BASE_SOURCE_DIR,
        source_glob="patha_agent_outputs*.jsonl",
    ),
    "dsC": DatasetSpec(
        name="dsC",
        split_policy=DEFAULT_SPLIT_POLICY,
        source_kind="core_jsonl",
        description="PathA dsC dataset records",
        source_root=DSC_SOURCE_DIR,
        source_glob="patha_agent_outputs*.jsonl",
    ),
    "opensdi": DatasetSpec(
        name="opensdi",
        split_policy=DEFAULT_SPLIT_POLICY,
        source_kind="core_jsonl",
        description="PathA OpenSDI dataset records",
        source_root=OPENDI_SOURCE_DIR,
        source_glob="patha_agent_outputs*.jsonl",
    ),
    "aigenproxy": DatasetSpec(
        name="aigenproxy",
        split_policy=DEFAULT_SPLIT_POLICY,
        source_kind="core_jsonl",
        description="PathA AI-GenBench proxy dataset records",
        source_root=AIGENPROXY_SOURCE_DIR,
        source_glob="patha_agent_outputs*.jsonl",
    ),
    "df40_extended": DatasetSpec(
        name="df40_extended",
        split_policy="group_stratified",
        source_kind="df40_extended",
        description="DF40 extended local records",
        source_root=DF40_DATA_ROOT,
        max_per_group=8,
    ),
}

DATASET_NAMES = tuple(DATASET_SPECS.keys())


def list_dataset_names() -> List[str]:
    return list(DATASET_NAMES)


def get_dataset_spec(dataset_name: str) -> DatasetSpec:
    try:
        return DATASET_SPECS[dataset_name]
    except KeyError as exc:
        raise KeyError(f"Unknown dataset: {dataset_name}") from exc


def _find_core_source_path(spec: DatasetSpec) -> Path:
    if spec.source_root is None or spec.source_glob is None:
        raise ValueError(f"Dataset spec missing source location: {spec.name}")
    candidates = list(spec.source_root.glob(spec.source_glob))
    if not candidates:
        raise FileNotFoundError(f"No JSONL found for {spec.name} under {spec.source_root}")
    return _latest_jsonl(candidates)


@contextmanager
def _df40_lock(lock_name: str) -> Iterator[None]:
    lock_dir = ensure_dir(DF40_EXTRACTED_ROOT / ".locks")
    lock_path = lock_dir / f"{lock_name}.lock"
    with lock_path.open("w", encoding="utf-8") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def ensure_df40_extracted_locked(name: str) -> Path:
    if name not in ZIP_SPECS:
        raise KeyError(f"Unknown DF40 asset: {name}")
    with _df40_lock(name):
        return ensure_extracted(name)


def load_core_dataset_records(dataset_name: str) -> List[Dict[str, Any]]:
    spec = get_dataset_spec(dataset_name)
    if spec.source_kind != "core_jsonl":
        raise ValueError(f"{dataset_name} is not a core JSONL dataset")
    source_path = _find_core_source_path(spec)
    return _make_core_adapter_rows(dataset_name, source_path)


def load_df40_extended_records(max_per_group: int | None = 8, ensure_extract: bool = True) -> List[Dict[str, Any]]:
    if ensure_extract:
        # Only extract assets that are actually valid locally.
        for asset_name in ZIP_SPECS:
            if is_valid_zip(asset_name):
                ensure_df40_extracted_locked(asset_name)
        rows = iter_df40_extended_records(max_per_group=max_per_group)
    else:
        missing = [name for name, spec in ZIP_SPECS.items() if not spec["extract_dir"].exists()]
        if missing:
            raise FileNotFoundError(f"DF40 extraction missing for: {', '.join(missing)}")
        rows = iter_df40_extended_records(max_per_group=max_per_group)

    adapter_rows: List[Dict[str, Any]] = []
    for raw in rows:
        image_path = str(raw.get("image_path", "")).strip()
        true_label = str(raw.get("true_label", "")).strip()
        if not image_path or true_label not in VALID_LABELS:
            continue
        adapter_rows.append(
            {
                "dataset_name": "df40_extended",
                "image_path": image_path,
                "true_label": true_label,
                "group_id": _derive_group_id("df40_extended", raw),
                "split_policy": "group_stratified",
                "audit_only": _audit_only(raw),
            }
        )
    adapter_rows.sort(key=lambda row: row["image_path"])
    return adapter_rows


def get_dataset_records(dataset_name: str, max_per_group: int | None = None, ensure_extract: bool = True) -> List[Dict[str, Any]]:
    spec = get_dataset_spec(dataset_name)
    if spec.source_kind == "core_jsonl":
        return load_core_dataset_records(dataset_name)
    if dataset_name == "df40_extended":
        return load_df40_extended_records(max_per_group=max_per_group if max_per_group is not None else spec.max_per_group, ensure_extract=ensure_extract)
    raise KeyError(f"Unsupported dataset: {dataset_name}")


__all__ = [
    "VALID_LABELS",
    "CANONICAL_SOURCE_KEYS",
    "DEFAULT_SPLIT_POLICY",
    "DatasetSpec",
    "DATASET_SPECS",
    "DATASET_NAMES",
    "list_dataset_names",
    "get_dataset_spec",
    "load_core_dataset_records",
    "load_df40_extended_records",
    "get_dataset_records",
    "ensure_df40_extracted_locked",
]
