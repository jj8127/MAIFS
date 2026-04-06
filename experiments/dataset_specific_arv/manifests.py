from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Sequence, Tuple

from .paths import MANIFEST_ROOT, ensure_dir

CANONICAL_FIELDS = ("image_path", "true_label", "dataset_name", "group_id", "split", "record_id")
MANIFEST_AUDIT_KEY = "audit_only"
SPLITS = ("train", "val", "test")
LABELS = ("authentic", "manipulated", "ai_generated")


def _stable_digest(*parts: str, length: int = 16) -> str:
    payload = "\n".join(parts).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:length]


def _unit_label_counts(rows: Sequence[Mapping[str, Any]]) -> Counter:
    return Counter(str(row["true_label"]) for row in rows)


def _target_counts(total: int, val_ratio: float, test_ratio: float) -> Dict[str, int]:
    val = int(round(total * val_ratio))
    test = int(round(total * test_ratio))
    train = total - val - test
    if train < 0:
        train = max(total - max(val, 0) - max(test, 0), 0)
    # Keep at least one item in each split when the dataset is large enough.
    ordered = [("train", train), ("val", val), ("test", test)]
    if total >= 3 and any(count == 0 for _, count in ordered):
        ordered = list(ordered)
        for idx, (name, count) in enumerate(ordered):
            if count == 0:
                ordered[idx] = (name, 1)
        overflow = sum(count for _, count in ordered) - total
        while overflow > 0:
            idx = max(range(3), key=lambda i: ordered[i][1])
            name, count = ordered[idx]
            if count > 1:
                ordered[idx] = (name, count - 1)
                overflow -= 1
            else:
                break
    return dict(ordered)


def _choose_split(
    current_totals: Counter,
    current_labels: Dict[str, Counter],
    target_totals: Mapping[str, int],
    target_labels: Mapping[str, Counter],
    unit_size: int,
    unit_label_counts: Mapping[str, int],
) -> str:
    best_split = SPLITS[0]
    best_score = None
    for split in SPLITS:
        total_score = abs((current_totals[split] + unit_size) - target_totals[split])
        label_score = 0
        for label in LABELS:
            label_score += abs((current_labels[split][label] + unit_label_counts.get(label, 0)) - target_labels[split][label])
        spill_penalty = max(0, current_totals[split] + unit_size - target_totals[split]) * 10
        score = total_score + 2 * label_score + spill_penalty
        if best_score is None or score < best_score or (score == best_score and current_totals[split] < current_totals[best_split]):
            best_split = split
            best_score = score
    return best_split


def _assign_units(
    units: Sequence[Tuple[str, List[Mapping[str, Any]]]],
    seed: int,
    val_ratio: float,
    test_ratio: float,
) -> Dict[str, List[Mapping[str, Any]]]:
    total = sum(len(rows) for _, rows in units)
    target_totals = _target_counts(total, val_ratio, test_ratio)
    target_label_totals = {split: Counter() for split in SPLITS}
    for _, rows in units:
        counts = _unit_label_counts(rows)
        for label in LABELS:
            # Proportional targets are computed after split selection for stability.
            pass
    all_label_counts = Counter()
    for _, rows in units:
        all_label_counts.update(_unit_label_counts(rows))
    target_labels = {
        split: Counter({label: int(round(all_label_counts[label] * (target_totals[split] / max(total, 1)))) for label in LABELS})
        for split in SPLITS
    }

    rng = random.Random(seed)
    decorated: List[Tuple[Tuple[int, float, str], str, List[Mapping[str, Any]]]] = []
    for unit_id, rows in units:
        decorated.append(((-len(rows), rng.random(), unit_id), unit_id, list(rows)))
    decorated.sort(key=lambda item: item[0])

    current_totals: Counter = Counter({split: 0 for split in SPLITS})
    current_labels: Dict[str, Counter] = {split: Counter({label: 0 for label in LABELS}) for split in SPLITS}
    assignments: Dict[str, List[Mapping[str, Any]]] = {split: [] for split in SPLITS}

    for _, unit_id, rows in decorated:
        counts = _unit_label_counts(rows)
        split = _choose_split(current_totals, current_labels, target_totals, target_labels, len(rows), counts)
        assignments[split].extend(rows)
        current_totals[split] += len(rows)
        current_labels[split].update(counts)

    # Ensure no split is empty when the dataset contains enough units.
    if total >= 3:
        for split in SPLITS:
            if assignments[split]:
                continue
            donor = max(SPLITS, key=lambda name: len(assignments[name]))
            if len(assignments[donor]) <= 1:
                continue
            moved = assignments[donor].pop()
            assignments[split].append(moved)
    return assignments


def _group_units(rows: Sequence[Mapping[str, Any]]) -> List[Tuple[str, List[Mapping[str, Any]]]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["group_id"])].append(row)
    return [(group_id, grouped[group_id]) for group_id in sorted(grouped.keys())]


def _record_units(rows: Sequence[Mapping[str, Any]]) -> List[Tuple[str, List[Mapping[str, Any]]]]:
    return [(str(idx), [row]) for idx, row in enumerate(rows)]


def split_adapter_rows(
    rows: Sequence[Mapping[str, Any]],
    seed: int = 42,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> List[Dict[str, Any]]:
    if not rows:
        return []
    policy = str(rows[0].get("split_policy", "auto"))
    grouped = _group_units(rows)
    repeated_groups = any(len(unit_rows) > 1 for _, unit_rows in grouped)
    use_group_split = policy == "group_stratified" or (policy == "auto" and repeated_groups)
    units = grouped if use_group_split else _record_units(rows)
    assignments = _assign_units(units, seed=seed, val_ratio=val_ratio, test_ratio=test_ratio)

    split_rows: List[Dict[str, Any]] = []
    for split in SPLITS:
        for row in assignments[split]:
            row_copy = dict(row)
            row_copy["split"] = split
            split_rows.append(row_copy)
    split_rows.sort(key=lambda row: (row["split"], row["dataset_name"], row["image_path"]))
    return split_rows


def canonicalize_adapter_rows(rows: Sequence[Mapping[str, Any]], seed: int = 42, val_ratio: float = 0.15, test_ratio: float = 0.15) -> List[Dict[str, Any]]:
    split_rows = split_adapter_rows(rows, seed=seed, val_ratio=val_ratio, test_ratio=test_ratio)
    canonical_rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(split_rows):
        audit_only = dict(row.get(MANIFEST_AUDIT_KEY, {}))
        for key, value in row.items():
            if key not in CANONICAL_FIELDS and key != MANIFEST_AUDIT_KEY:
                audit_only.setdefault(key, value)
        canonical_rows.append(
            {
                "image_path": str(row["image_path"]),
                "true_label": str(row["true_label"]),
                "dataset_name": str(row["dataset_name"]),
                "group_id": str(row["group_id"]),
                "split": str(row["split"]),
                "record_id": f"{row['dataset_name']}::{row['split']}::{_stable_digest(row['dataset_name'], row['split'], row['group_id'], row['image_path'], str(seed))}",
                MANIFEST_AUDIT_KEY: audit_only,
            }
        )
    return canonical_rows


def validate_manifest_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    seen_ids: set[str] = set()
    for idx, row in enumerate(rows):
        for key in CANONICAL_FIELDS:
            if key not in row:
                raise ValueError(f"Manifest row {idx} missing required field: {key}")
        if row["split"] not in SPLITS:
            raise ValueError(f"Manifest row {idx} has invalid split: {row['split']}")
        if not isinstance(row.get(MANIFEST_AUDIT_KEY), dict):
            raise ValueError(f"Manifest row {idx} audit bucket must be a dict")
        record_id = str(row["record_id"])
        if record_id in seen_ids:
            raise ValueError(f"Duplicate record_id detected: {record_id}")
        seen_ids.add(record_id)


def manifest_summary(rows: Sequence[Mapping[str, Any]], seed: int) -> Dict[str, Any]:
    validate_manifest_rows(rows)
    by_split = {split: [row for row in rows if row["split"] == split] for split in SPLITS}
    summary = {
        "seed": seed,
        "n_records": len(rows),
        "n_unique_groups": len({row["group_id"] for row in rows}),
        "split_policies": sorted({str(row.get(MANIFEST_AUDIT_KEY, {}).get("split_policy", "")) for row in rows}),
        "counts_by_split": {
            split: {
                "n_records": len(split_rows),
                "by_true_label": dict(Counter(row["true_label"] for row in split_rows)),
            }
            for split, split_rows in by_split.items()
        },
        "counts_by_true_label": dict(Counter(row["true_label"] for row in rows)),
    }
    return summary


def build_manifest_bundle(
    rows: Sequence[Mapping[str, Any]],
    seed: int = 42,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    canonical_rows = canonicalize_adapter_rows(rows, seed=seed, val_ratio=val_ratio, test_ratio=test_ratio)
    summary = {
        "seed": seed,
        "n_records": len(canonical_rows),
        "n_unique_groups": len({row["group_id"] for row in canonical_rows}),
        "split_policies": sorted({str(row.get(MANIFEST_AUDIT_KEY, {}).get("split_policy", "")) for row in canonical_rows}),
        "counts_by_split": {
            split: {
                "n_records": sum(1 for row in canonical_rows if row["split"] == split),
                "by_true_label": dict(Counter(row["true_label"] for row in canonical_rows if row["split"] == split)),
            }
            for split in SPLITS
        },
        "counts_by_true_label": dict(Counter(row["true_label"] for row in canonical_rows)),
        "datasets": sorted({row["dataset_name"] for row in canonical_rows}),
        "split_policies": sorted({str(row.get(MANIFEST_AUDIT_KEY, {}).get("split_policy", "")) for row in canonical_rows}),
    }
    validate_manifest_rows(canonical_rows)
    return canonical_rows, summary


def write_manifest_bundle(
    rows: Sequence[Mapping[str, Any]],
    out_dir: Path | None = None,
    manifest_name: str = "manifest.jsonl",
    summary_name: str = "manifest_summary.json",
    seed: int = 42,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Tuple[Path, Path]:
    out_dir = ensure_dir(out_dir or MANIFEST_ROOT)
    canonical_rows, summary = build_manifest_bundle(rows, seed=seed, val_ratio=val_ratio, test_ratio=test_ratio)
    manifest_path = out_dir / manifest_name
    summary_path = out_dir / summary_name
    with manifest_path.open("w", encoding="utf-8") as fh:
        for row in canonical_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    return manifest_path, summary_path


def load_manifest(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if MANIFEST_AUDIT_KEY not in row:
                row[MANIFEST_AUDIT_KEY] = {}
            rows.append(row)
    validate_manifest_rows(rows)
    return rows


__all__ = [
    "CANONICAL_FIELDS",
    "MANIFEST_AUDIT_KEY",
    "split_adapter_rows",
    "canonicalize_adapter_rows",
    "validate_manifest_rows",
    "manifest_summary",
    "build_manifest_bundle",
    "write_manifest_bundle",
    "load_manifest",
]
