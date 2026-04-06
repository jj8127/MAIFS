#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_DIR = ROOT / "experiments"
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

import train_mobilenetv2_dualstream as mnv2_mod


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABELS = ("authentic", "manipulated", "ai_generated")
LABEL_MAP = mnv2_mod.LABEL_MAP
IDX2LABEL = mnv2_mod.IDX2LABEL


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _normalize_repo_relative(path_str: str) -> str:
    p = Path(path_str)
    if not p.is_absolute():
        return str(p)
    try:
        return str(p.relative_to(ROOT))
    except ValueError as exc:
        raise ValueError(f"Manifest image_path must be repo-relative or under {ROOT}: {path_str}") from exc


def _jsonl_records(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                yield json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_no}") from exc


def _extract_audit_only(rec: Dict) -> Dict:
    audit_only = rec.get("audit_only")
    if isinstance(audit_only, dict):
        return dict(audit_only)
    if audit_only is not None:
        return {"value": audit_only}
    known = {"image_path", "true_label", "dataset_name", "group_id", "split", "record_id", "audit_only"}
    return {k: v for k, v in rec.items() if k not in known}


def load_manifest(path: Path, fallback_split: str) -> List[Dict]:
    records: List[Dict] = []
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")

    for idx, rec in enumerate(_jsonl_records(path), start=1):
        image_path = rec.get("image_path")
        true_label = rec.get("true_label")
        if not image_path or true_label not in LABEL_MAP:
            raise ValueError(f"Invalid manifest row in {path}: missing/unsupported image_path or true_label at line {idx}")

        normalized_path = _normalize_repo_relative(str(image_path))
        dataset_name = str(rec.get("dataset_name") or path.stem)
        group_id = str(rec.get("group_id") or rec.get("base_image_id") or Path(normalized_path).stem)
        split = str(rec.get("split") or fallback_split)
        record_id = str(rec.get("record_id") or f"{path.stem}:{idx:08d}")

        records.append(
            {
                "image_path": normalized_path,
                "true_label": str(true_label),
                "dataset_name": dataset_name,
                "group_id": group_id,
                "split": split,
                "record_id": record_id,
                "audit_only": _extract_audit_only(rec),
            }
        )
    return records


def load_manifests(manifest_paths: Sequence[Tuple[str, Path]]) -> Dict[str, List[Dict]]:
    loaded: Dict[str, List[Dict]] = {}
    for split_name, path in manifest_paths:
        loaded[split_name] = load_manifest(path, split_name)
    return loaded


def validate_manifests(split_records: Dict[str, List[Dict]]) -> str:
    all_records = [rec for split in ("train", "val", "test") for rec in split_records.get(split, [])]
    if not all_records:
        raise ValueError("No manifest records loaded.")

    dataset_names = sorted({rec["dataset_name"] for rec in all_records if rec.get("dataset_name")})
    if len(dataset_names) != 1:
        raise ValueError(f"Expected a single dataset_name across train/val/test, got: {dataset_names}")

    seen: Dict[str, str] = {}
    for split_name, records in split_records.items():
        for rec in records:
            rid = rec["record_id"]
            prior = seen.get(rid)
            if prior is not None and prior != split_name:
                raise ValueError(f"record_id collision across splits: {rid} appears in {prior} and {split_name}")
            seen[rid] = split_name

    return dataset_names[0]


def resolve_output_dir(args: argparse.Namespace, dataset_name: str) -> Path:
    if args.output_dir:
        out_dir = Path(args.output_dir)
        if out_dir.exists() and any(out_dir.iterdir()):
            raise FileExistsError(f"Output dir already exists and is not empty: {out_dir}")
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    out_dir = ROOT / "experiments" / "results" / "dataset_runs" / dataset_name / str(args.seed) / run_id
    out_dir.mkdir(parents=True, exist_ok=False)
    return out_dir


def write_jsonl(rows: Sequence[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def confusion_matrix(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> List[List[int]]:
    idx = {label: i for i, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for t, p in zip(y_true, y_pred):
        if t not in idx or p not in idx:
            continue
        matrix[idx[t]][idx[p]] += 1
    return matrix


def classification_metrics(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> Dict:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred length mismatch")
    cm = confusion_matrix(y_true, y_pred, labels)
    total = sum(sum(row) for row in cm)
    correct = sum(cm[i][i] for i in range(len(labels)))

    per_class: Dict[str, Dict[str, float]] = {}
    precisions: List[float] = []
    recalls: List[float] = []
    f1s: List[float] = []
    supports: List[int] = []

    for i, label in enumerate(labels):
        tp = cm[i][i]
        fp = sum(cm[r][i] for r in range(len(labels)) if r != i)
        fn = sum(cm[i][c] for c in range(len(labels)) if c != i)
        support = sum(cm[i])
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        supports.append(support)

    accuracy = correct / total if total else 0.0
    macro_precision = float(np.mean(precisions)) if precisions else 0.0
    macro_recall = float(np.mean(recalls)) if recalls else 0.0
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0
    weighted_f1 = float(np.average(f1s, weights=supports)) if total else 0.0

    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "support": total,
        "per_class": per_class,
        "confusion_matrix": {
            "labels": list(labels),
            "matrix": cm,
        },
    }


def collapse_to_binary(label: str) -> str:
    return "authentic" if label == "authentic" else "edited"


def build_binary_metrics(y_true: Sequence[str], y_pred: Sequence[str]) -> Dict:
    binary_labels = ("authentic", "edited")
    y_true_bin = [collapse_to_binary(label) for label in y_true]
    y_pred_bin = [collapse_to_binary(label) for label in y_pred]
    return classification_metrics(y_true_bin, y_pred_bin, binary_labels)


def prediction_rows(records: Sequence[Dict], probs: np.ndarray, preds: np.ndarray, split_name: str) -> List[Dict]:
    rows: List[Dict] = []
    for rec, prob_vec, pred_idx in zip(records, probs.tolist(), preds.tolist()):
        pred_label = IDX2LABEL[int(pred_idx)]
        row = {
            "record_id": rec["record_id"],
            "image_path": rec["image_path"],
            "dataset_name": rec["dataset_name"],
            "group_id": rec["group_id"],
            "split": split_name,
            "true_label": rec["true_label"],
            "true_label_binary": collapse_to_binary(rec["true_label"]),
            "pred_label": pred_label,
            "pred_label_binary": collapse_to_binary(pred_label),
            "confidence": float(prob_vec[int(pred_idx)]),
            "scores": {IDX2LABEL[i]: float(prob_vec[i]) for i in range(len(prob_vec))},
            "audit_only": rec.get("audit_only", {}),
        }
        rows.append(row)
    return rows


def metrics_for_rows(rows: Sequence[Dict]) -> Dict:
    y_true = [row["true_label"] for row in rows]
    y_pred = [row["pred_label"] for row in rows]
    y_true_bin = [row["true_label_binary"] for row in rows]
    y_pred_bin = [row["pred_label_binary"] for row in rows]
    return {
        "strict_three_class": classification_metrics(y_true, y_pred, LABELS),
        "binary_auth_vs_edited": classification_metrics(y_true_bin, y_pred_bin, ("authentic", "edited")),
    }


def load_checkpoint_state(model: torch.nn.Module, checkpoint_path: Path) -> None:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in state and isinstance(state[key], dict):
                model.load_state_dict(state[key], strict=True)
                return
    if isinstance(state, dict):
        model.load_state_dict(state, strict=True)
        return
    raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")


def build_model(init_mode: str, checkpoint_path: Optional[Path]) -> torch.nn.Module:
    if init_mode == "scratch":
        model = mnv2_mod.DualStreamMobileNetV2(pretrained=False)
    elif init_mode == "imagenet":
        model = mnv2_mod.DualStreamMobileNetV2(pretrained=True)
    elif init_mode == "checkpoint":
        model = mnv2_mod.DualStreamMobileNetV2(pretrained=False)
        if checkpoint_path is None:
            raise ValueError("--checkpoint is required when --init-mode checkpoint is used")
        load_checkpoint_state(model, checkpoint_path)
        return model
    else:
        raise ValueError(f"Unknown init mode: {init_mode}")
    return model


def make_loader(
    records: Sequence[Dict],
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    *,
    eval_mode: bool = False,
) -> DataLoader:
    return DataLoader(
        mnv2_mod.ForensicsDataset(list(records), mnv2_mod.VAL_TRANSFORM if eval_mode else (mnv2_mod.TRAIN_TRANSFORM if shuffle else mnv2_mod.VAL_TRANSFORM)),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(DEVICE.type == "cuda"),
    )


def train_and_eval(args: argparse.Namespace) -> Dict:
    set_seed(args.seed)

    split_records = load_manifests(
        [
            ("train", Path(args.manifest_train)),
            ("val", Path(args.manifest_val)),
            ("test", Path(args.manifest_test)),
        ]
    )
    dataset_name = validate_manifests(split_records)
    out_dir = resolve_output_dir(args, dataset_name)
    checkpoints_dir = out_dir / "checkpoints"
    predictions_dir = out_dir / "predictions"
    summaries_dir = out_dir / "summaries"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    summaries_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "dataset_name": dataset_name,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "num_workers": args.num_workers,
        "init_mode": args.init_mode,
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "manifest_train": str(Path(args.manifest_train)),
        "manifest_val": str(Path(args.manifest_val)),
        "manifest_test": str(Path(args.manifest_test)),
        "device": str(DEVICE),
    }

    write_jsonl([config], summaries_dir / "run_config.jsonl")
    (summaries_dir / "run_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")

    train_loader = make_loader(split_records["train"], args.batch_size, True, args.num_workers)
    train_eval_loader = make_loader(split_records["train"], args.batch_size * 2, False, args.num_workers, eval_mode=True)
    val_loader = make_loader(split_records["val"], args.batch_size * 2, False, args.num_workers, eval_mode=True)
    test_loader = make_loader(split_records["test"], args.batch_size * 2, False, args.num_workers, eval_mode=True)

    model = build_model(args.init_mode, Path(args.checkpoint) if args.checkpoint else None).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs, 1),
        eta_min=args.lr * 0.05,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda"))

    best_path = checkpoints_dir / "base_best.pth"
    best_val_f1 = -1.0
    best_epoch = 0
    history: List[Dict] = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = mnv2_mod.train_epoch(model, train_loader, optimizer, scaler)
        val_stats = mnv2_mod.evaluate(model, val_loader)
        scheduler.step()

        val_probs = val_stats["scores"]
        val_preds = val_stats["preds"]
        val_labels = val_stats["labels"]
        val_rows = prediction_rows(split_records["val"], val_probs, val_preds, "val")
        val_metrics = metrics_for_rows(val_rows)

        epoch_row = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "train_accuracy": float(train_acc),
            "val_strict_macro_f1": float(val_metrics["strict_three_class"]["macro_f1"]),
            "val_strict_accuracy": float(val_metrics["strict_three_class"]["accuracy"]),
            "val_binary_macro_f1": float(val_metrics["binary_auth_vs_edited"]["macro_f1"]),
            "val_binary_accuracy": float(val_metrics["binary_auth_vs_edited"]["accuracy"]),
        }
        history.append(epoch_row)

        improved = epoch_row["val_strict_macro_f1"] > best_val_f1
        if improved:
            best_val_f1 = epoch_row["val_strict_macro_f1"]
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "config": config,
                    "best_val_strict_macro_f1": best_val_f1,
                    "history": history,
                },
                best_path,
            )

        marker = " *" if improved else ""
        print(
            f"[epoch {epoch:03d}/{args.epochs:03d}] "
            f"loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_f1={epoch_row['val_strict_macro_f1']:.4f} "
            f"val_bin_f1={epoch_row['val_binary_macro_f1']:.4f}{marker}"
        )

    if not best_path.exists():
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "epoch": args.epochs,
                "config": config,
                "best_val_strict_macro_f1": best_val_f1,
                "history": history,
            },
            best_path,
        )

    load_checkpoint_state(model, best_path)
    model.eval()

    split_outputs: Dict[str, Dict] = {}
    for split_name, loader in (("train", train_eval_loader), ("val", val_loader), ("test", test_loader)):
        stats = mnv2_mod.evaluate(model, loader)
        probs = stats["scores"]
        preds = stats["preds"]
        rows = prediction_rows(split_records[split_name], probs, preds, split_name)
        metrics = metrics_for_rows(rows)

        pred_path = predictions_dir / f"base_{split_name}_predictions.jsonl"
        write_jsonl(rows, pred_path)
        write_jsonl([metrics], summaries_dir / f"{split_name}_metrics.jsonl")

        split_outputs[split_name] = {
            "metrics": metrics,
            "prediction_jsonl": str(pred_path),
            "n_records": len(rows),
        }

    summary = {
        "dataset_name": dataset_name,
        "device": str(DEVICE),
        "output_dir": str(out_dir),
        "checkpoint": str(best_path),
        "best_epoch": best_epoch,
        "best_val_strict_macro_f1": best_val_f1,
        "history": history,
        "manifests": {
            split: str(Path(getattr(args, f"manifest_{split}"))) for split in ("train", "val", "test")
        },
        "splits": {
            split: {
                "n_records": len(records),
                "n_groups": len({rec["group_id"] for rec in records}),
                "by_true_label": {
                    label: sum(1 for rec in records if rec["true_label"] == label) for label in LABELS
                },
            }
            for split, records in split_records.items()
        },
        "results": split_outputs,
    }

    (summaries_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    write_jsonl([summary], summaries_dir / "run_summary.jsonl")
    write_jsonl(history, summaries_dir / "training_history.jsonl")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dataset-specific scratch ARV base-classifier trainer")
    parser.add_argument("--manifest-train", required=True, type=str)
    parser.add_argument("--manifest-val", required=True, type=str)
    parser.add_argument("--manifest-test", required=True, type=str)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--init-mode", choices=("scratch", "imagenet", "checkpoint"), default="scratch")
    parser.add_argument("--checkpoint", type=str, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = train_and_eval(args)
    print(f"[done] output_dir={summary['output_dir']}")
    print(f"[done] summary={summary['output_dir']}/summaries/run_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
