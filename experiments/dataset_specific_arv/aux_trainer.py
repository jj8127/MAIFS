#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_DIR = ROOT / "experiments"
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

import train_specialist_m_complementary as specm_mod  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_OUTPUT_ROOT = ROOT / "experiments" / "results" / "dataset_runs"
DEFAULT_CHECKPOINT = specm_mod.SPECM_V4
LABELS_2C = ("authentic", "manipulated")
LABEL_TO_IDX = {"authentic": 0, "manipulated": 1}
IDX_TO_LABEL = {0: "authentic", 1: "manipulated"}
VALID_SPLITS = ("train", "val", "test")
VALID_INIT_MODES = ("scratch", "imagenet", "checkpoint")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def write_jsonl(records: Sequence[Mapping[str, Any]], path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def normalize_true_label(label: str) -> str:
    value = str(label).strip().lower()
    if value in {"authentic", "real", "original", "clean"}:
        return "authentic"
    if value in {"manipulated", "edited", "fake", "tampered"}:
        return "manipulated"
    if value in {"ai_generated", "aigen", "generated"}:
        return "ai_generated"
    raise ValueError(f"Unsupported label: {label!r}")


def collapse_to_binary(label: str) -> str:
    return "authentic" if normalize_true_label(label) == "authentic" else "manipulated"


def infer_dataset_name(records: Sequence[Mapping[str, Any]], fallback: str) -> str:
    names = {str(r.get("dataset_name", "")).strip() for r in records if r.get("dataset_name")}
    names.discard("")
    if not names:
        return fallback
    if len(names) == 1:
        return next(iter(names))
    raise ValueError(f"Manifest mixes multiple dataset_name values: {sorted(names)}")


def canonical_record_id(record: Mapping[str, Any], split: str, index: int) -> str:
    if record.get("record_id"):
        return str(record["record_id"])
    image_path = str(record.get("image_path", ""))
    return f"{split}:{index}:{image_path}"


def load_manifest(path: Path, expected_split: str) -> List[Dict[str, Any]]:
    if expected_split not in VALID_SPLITS:
        raise ValueError(f"Unknown expected_split={expected_split!r}")
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")

    raw = read_jsonl(path)
    records: List[Dict[str, Any]] = []
    for idx, record in enumerate(raw):
        if "image_path" not in record or "true_label" not in record:
            raise ValueError(f"Manifest record missing required fields at {path}:{idx}")
        rec = dict(record)
        rec["image_path"] = str(rec["image_path"])
        rec["true_label"] = normalize_true_label(rec["true_label"])
        rec["split"] = str(rec.get("split", expected_split))
        if rec["split"] != expected_split:
            raise ValueError(
                f"Manifest split mismatch in {path}: expected {expected_split}, got {rec['split']}"
            )
        rec["dataset_name"] = str(rec.get("dataset_name", "")).strip() or path.stem
        rec["record_id"] = canonical_record_id(rec, expected_split, idx)
        if "audit_only" in rec and not isinstance(rec["audit_only"], dict):
            rec["audit_only"] = {"value": rec["audit_only"]}
        records.append(rec)
    return records


def assert_unique_record_ids(records_by_split: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
    seen: Dict[str, str] = {}
    for split, records in records_by_split.items():
        for rec in records:
            record_id = str(rec["record_id"])
            if record_id in seen:
                raise ValueError(f"Duplicate record_id across splits: {record_id} ({seen[record_id]} vs {split})")
            seen[record_id] = split


def collect_dataset_name(records_by_split: Mapping[str, Sequence[Mapping[str, Any]]], fallback: str) -> str:
    names = {str(rec.get("dataset_name", "")).strip() for records in records_by_split.values() for rec in records if rec.get("dataset_name")}
    names.discard("")
    if not names:
        return fallback
    if len(names) == 1:
        return next(iter(names))
    raise ValueError(f"Multiple dataset_name values found across splits: {sorted(names)}")


def validate_image_paths(records: Sequence[Mapping[str, Any]]) -> None:
    missing: List[str] = []
    for rec in records:
        img_path = resolve_path(rec["image_path"])
        if not img_path.exists():
            missing.append(str(rec["image_path"]))
    if missing:
        preview = ", ".join(missing[:8])
        extra = "" if len(missing) <= 8 else f" ... (+{len(missing) - 8} more)"
        raise FileNotFoundError(f"Missing images: {preview}{extra}")


def inspect_results_root(path: Path) -> None:
    ensure_dir(path)
    probe = path / ".write_probe"
    probe.write_text("ok\n", encoding="utf-8")
    probe.unlink(missing_ok=True)


def list_df40_archives() -> Dict[str, Path]:
    df40_root = ROOT / "datasets" / "external_new" / "DF40"
    return {
        "ffpp_real": ROOT / "FaceForensics++_real_data_for_DF40.zip",
        "celebdf_real": ROOT / "Celeb-DF-v2_real_data_for_DF40.zip",
        "stylegan2_fake": ROOT / "StyleGAN2.zip",
        "deepfacelab_fake": df40_root / "archives" / "deepfacelab.zip",
        "faceswap_fake": df40_root / "faceswap.zip",
        "fomm_fake": df40_root / "fomm.zip",
        "inswap_fake": df40_root / "inswap.zip",
        "facevid2vid_fake": df40_root / "facevid2vid.zip",
    }


def inspect_df40_state() -> Dict[str, Any]:
    df40_root = ROOT / "datasets" / "external_new" / "DF40"
    extracted = df40_root / "extracted_minimal"
    archive_status: Dict[str, Dict[str, Any]] = {}
    for name, path in list_df40_archives().items():
        archive_status[name] = {"path": str(path), "exists": path.exists()}
    markers = {}
    for name in ("ffpp_real", "celebdf_real", "stylegan2_fake", "deepfacelab_fake", "faceswap_fake", "fomm_fake", "inswap_fake", "facevid2vid_fake"):
        marker = extracted / name / ".extract_done"
        markers[name] = {"path": str(marker), "exists": marker.exists()}
    return {
        "root": str(df40_root),
        "archives": archive_status,
        "markers": markers,
        "metadata_json": str(df40_root / "metadata" / "json" / "DF40_all.json"),
    }


def preflight(dataset_name: str, output_root: Path, manifests: Mapping[str, Path], base_preds: Optional[Mapping[str, Path]] = None) -> Dict[str, Any]:
    checks: Dict[str, Any] = {}
    checks["gpu_available"] = torch.cuda.is_available()
    checks["device"] = str(DEVICE)
    inspect_results_root(output_root)
    checks["output_root"] = {"path": str(output_root), "writable": True}

    dataset_records = {}
    for split, manifest_path in manifests.items():
        records = load_manifest(manifest_path, split)
        validate_image_paths(records)
        dataset_records[split] = records
    assert_unique_record_ids(dataset_records)

    if base_preds is not None:
        checks["base_predictions"] = {}
        for split, pred_path in base_preds.items():
            if not pred_path.exists():
                raise FileNotFoundError(f"Base prediction JSONL not found: {pred_path}")
            checks["base_predictions"][split] = {"path": str(pred_path), "exists": True, "n": len(read_jsonl(pred_path))}

    checks["dataset_name"] = collect_dataset_name(dataset_records, dataset_name)
    checks["split_counts"] = {split: len(records) for split, records in dataset_records.items()}
    checks["record_counts_by_label"] = {
        split: dict(Counter(r["true_label"] for r in records))
        for split, records in dataset_records.items()
    }
    checks["df40"] = inspect_df40_state()
    checks["manifest_ok"] = True
    return checks


def extract_scores(record: Mapping[str, Any]) -> Dict[str, float]:
    candidates: List[Dict[str, float]] = []
    scores = record.get("scores")
    if isinstance(scores, Mapping):
        candidates.append({str(k): float(v) for k, v in scores.items() if isinstance(v, (int, float))})

    direct_keys = {
        "authentic": ("authentic_score", "auth_score", "p_auth", "auth", "score_authentic"),
        "manipulated": ("manip_score", "manipulated_score", "p_manip", "manip", "score_manipulated"),
        "ai_generated": ("ai_generated_score", "aigen_score", "p_aigen", "score_ai_generated"),
    }
    direct: Dict[str, float] = {}
    for label, keys in direct_keys.items():
        for key in keys:
            value = record.get(key)
            if isinstance(value, (int, float)):
                direct[label] = float(value)
                break
    if direct:
        candidates.append(direct)

    if candidates:
        merged: Dict[str, float] = {"authentic": 0.0, "manipulated": 0.0, "ai_generated": 0.0}
        for source in candidates:
            for key, value in source.items():
                norm_key = {
                    "auth": "authentic",
                    "real": "authentic",
                    "original": "authentic",
                    "manip": "manipulated",
                    "edited": "manipulated",
                    "fake": "manipulated",
                    "aigen": "ai_generated",
                    "generated": "ai_generated",
                }.get(key, key)
                if norm_key in merged:
                    merged[norm_key] = max(merged[norm_key], float(value))
        total = sum(max(v, 0.0) for v in merged.values())
        if total <= 0:
            return {"authentic": 1 / 3, "manipulated": 1 / 3, "ai_generated": 1 / 3}
        return {k: max(v, 0.0) / total for k, v in merged.items()}

    pred_label = str(record.get("pred_label", "")).strip().lower()
    confidence = float(record.get("confidence", 0.5))
    if pred_label not in {"authentic", "manipulated", "ai_generated"}:
        return {"authentic": 1 / 3, "manipulated": 1 / 3, "ai_generated": 1 / 3}

    other = max((1.0 - confidence) / 2.0, 0.0)
    probs = {"authentic": other, "manipulated": other, "ai_generated": other}
    probs[pred_label] = confidence
    total = sum(probs.values())
    return {k: v / total for k, v in probs.items()}


def base_confidence(true_label: str, probs: Mapping[str, float]) -> float:
    label = normalize_true_label(true_label)
    if label == "authentic":
        return float(probs.get("authentic", 0.0))
    if label == "manipulated":
        return float(probs.get("manipulated", 0.0))
    return float(1.0 - probs.get("authentic", 0.0))


def compute_weight(confidence: float, gamma: float, w_max: float) -> float:
    return min(max((1.0 - confidence) ** gamma, 0.0), w_max)


def build_weight_map(records: Sequence[Mapping[str, Any]], base_preds: Sequence[Mapping[str, Any]], gamma: float, w_max: float) -> Tuple[Dict[str, float], Dict[str, float]]:
    pred_index: Dict[str, Mapping[str, Any]] = {}
    for pred in base_preds:
        key = str(pred.get("image_path", ""))
        if not key:
            continue
        if key in pred_index:
            raise ValueError(f"Duplicate base prediction for image_path={key}")
        pred_index[key] = pred

    weight_map: Dict[str, float] = {}
    confidence_map: Dict[str, float] = {}
    missing: List[str] = []
    for rec in records:
        image_path = str(rec["image_path"])
        pred = pred_index.get(image_path)
        if pred is None:
            missing.append(image_path)
            continue
        probs = extract_scores(pred)
        c_x = base_confidence(str(rec["true_label"]), probs)
        confidence_map[image_path] = c_x
        weight_map[image_path] = compute_weight(c_x, gamma, w_max)

    if missing:
        preview = ", ".join(missing[:8])
        extra = "" if len(missing) <= 8 else f" ... (+{len(missing) - 8} more)"
        raise ValueError(f"Missing base predictions for {len(missing)} records: {preview}{extra}")

    extras = sorted(set(pred_index) - {str(rec["image_path"]) for rec in records})
    if extras:
        print(f"[aux] warning: {len(extras)} base predictions not matched by manifest; ignored")
    return weight_map, confidence_map


class ManifestAuxDataset(Dataset):
    def __init__(self, records: Sequence[Mapping[str, Any]], weight_map: Optional[Mapping[str, float]] = None, transform: Optional[Any] = None, default_weight: float = 1.0):
        self.records = [dict(r) for r in records]
        self.weight_map = dict(weight_map or {})
        self.transform = transform or specm_mod.VAL_TRANSFORM
        self.default_weight = float(default_weight)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        img_path = resolve_path(rec["image_path"])
        label = LABEL_TO_IDX[collapse_to_binary(rec["true_label"])]
        weight = float(self.weight_map.get(str(rec["image_path"]), self.default_weight))
        try:
            img = Image.open(img_path).convert("RGB")
            tensor = self.transform(img)
        except Exception:
            tensor = torch.zeros(3, 224, 224)
        return (
            tensor,
            label,
            weight,
            str(rec["image_path"]),
            str(rec["true_label"]),
            str(rec["record_id"]),
            str(rec["split"]),
            str(rec.get("dataset_name", "")),
        )


def make_sampler(records: Sequence[Mapping[str, Any]]) -> WeightedRandomSampler:
    labels = [LABEL_TO_IDX[collapse_to_binary(rec["true_label"])] for rec in records]
    counts = Counter(labels)
    weights = [1.0 / max(counts[label], 1) for label in labels]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def prepare_model(init_mode: str, init_checkpoint: Optional[Path]) -> torch.nn.Module:
    if init_mode not in VALID_INIT_MODES:
        raise ValueError(f"Invalid init_mode={init_mode!r}")
    if init_mode == "scratch":
        model = specm_mod.SpecialistM(pretrained=False)
    elif init_mode == "imagenet":
        model = specm_mod.SpecialistM(pretrained=True)
    else:
        model = specm_mod.SpecialistM(pretrained=False)
        ckpt_path = init_checkpoint or DEFAULT_CHECKPOINT
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state, strict=True)
    return model.to(DEVICE)


def weighted_cross_entropy(logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    per_item = F.cross_entropy(logits, labels, reduction="none")
    denom = weights.sum().clamp(min=1e-8)
    return (per_item * weights).sum() / denom


def train_one_epoch(model: torch.nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, scaler: torch.cuda.amp.GradScaler) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    use_amp = DEVICE.type == "cuda"
    for imgs, labels, weights, *_rest in loader:
        imgs = imgs.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        weights = weights.to(DEVICE, dtype=torch.float32, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", enabled=use_amp):
            logits = model(imgs)
            loss = weighted_cross_entropy(logits, labels, weights)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch = int(labels.shape[0])
        total_loss += loss.item() * batch
        correct += int((logits.argmax(1) == labels).sum().item())
        total += batch

    return {
        "loss": total_loss / max(total, 1),
        "accuracy": correct / max(total, 1),
    }


@torch.no_grad()
def predict_split(model: torch.nn.Module, loader: DataLoader) -> List[Dict[str, Any]]:
    model.eval()
    rows: List[Dict[str, Any]] = []
    for imgs, labels, weights, image_paths, true_labels, record_ids, splits, dataset_names in loader:
        imgs = imgs.to(DEVICE, non_blocking=True)
        logits = model(imgs)
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        preds = probs.argmax(axis=1)
        for i in range(len(image_paths)):
            prob_vec = probs[i]
            pred_idx = int(preds[i])
            rows.append(
                {
                    "image_path": image_paths[i],
                    "record_id": record_ids[i],
                    "split": splits[i],
                    "dataset_name": dataset_names[i],
                    "true_label": true_labels[i],
                    "true_label_binary": collapse_to_binary(true_labels[i]),
                    "pred_label": IDX_TO_LABEL[pred_idx],
                    "pred_label_binary": IDX_TO_LABEL[pred_idx],
                    "confidence": float(prob_vec[pred_idx]),
                    "authentic_score": float(prob_vec[0]),
                    "manipulated_score": float(prob_vec[1]),
                    "weight": float(weights[i].item() if hasattr(weights[i], "item") else weights[i]),
                }
            )
    return rows


def compute_metrics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {
            "n": 0,
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "weighted_f1": 0.0,
            "confusion": [[0, 0], [0, 0]],
            "per_class": {},
        }

    y_true = [LABEL_TO_IDX[collapse_to_binary(r["true_label"])] for r in rows]
    y_pred = [LABEL_TO_IDX[collapse_to_binary(r["pred_label"])] for r in rows]
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    return {
        "n": len(rows),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "per_class": {
            IDX_TO_LABEL[i]: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i in range(2)
        },
    }


def summarize_weights(weight_map: Mapping[str, float], confidence_map: Mapping[str, float]) -> Dict[str, Any]:
    weights = np.array(list(weight_map.values()), dtype=np.float32) if weight_map else np.array([], dtype=np.float32)
    confidences = np.array(list(confidence_map.values()), dtype=np.float32) if confidence_map else np.array([], dtype=np.float32)
    if len(weights) == 0:
        return {"n": 0}
    return {
        "n": int(len(weights)),
        "weight_mean": float(weights.mean()),
        "weight_median": float(np.median(weights)),
        "weight_min": float(weights.min()),
        "weight_max": float(weights.max()),
        "confidence_mean": float(confidences.mean()),
        "confidence_median": float(np.median(confidences)),
    }


def build_loader(records: Sequence[Mapping[str, Any]], weight_map: Optional[Mapping[str, float]], batch_size: int, num_workers: int, train: bool) -> DataLoader:
    dataset = ManifestAuxDataset(
        records,
        weight_map=weight_map if weight_map is not None else {},
        transform=specm_mod.TRAIN_TRANSFORM if train else specm_mod.VAL_TRANSFORM,
        default_weight=1.0 if weight_map is None else 0.5,
    )
    sampler = make_sampler(records) if train else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(train and sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=(DEVICE.type == "cuda"),
    )


def load_split_bundle(manifests: Mapping[str, Path], base_preds: Mapping[str, Path]) -> Dict[str, Dict[str, Any]]:
    bundle: Dict[str, Dict[str, Any]] = {}
    for split in VALID_SPLITS:
        bundle[split] = {
            "manifest_path": manifests[split],
            "base_pred_path": base_preds[split],
            "manifest": load_manifest(manifests[split], split),
            "base_preds": read_jsonl(base_preds[split]),
        }
    assert_unique_record_ids({split: bundle[split]["manifest"] for split in VALID_SPLITS})
    return bundle


def save_checkpoint(path: Path, model: torch.nn.Module, history: Sequence[Mapping[str, Any]], best_epoch: int, best_val_f1: float, config: Mapping[str, Any]) -> None:
    ensure_dir(path.parent)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "history": list(history),
            "best_val_macro_f1": float(best_val_f1),
            "best_epoch": int(best_epoch),
            "config": dict(config),
        },
        path,
    )


def train_auxiliary(args: argparse.Namespace) -> Dict[str, Any]:
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = resolve_path(args.output_root)
    run_root = ensure_dir(output_root / args.dataset_name / str(args.seed) / run_id)
    aux_root = ensure_dir(run_root / "aux")

    manifests = {
        "train": resolve_path(args.manifest_train),
        "val": resolve_path(args.manifest_val),
        "test": resolve_path(args.manifest_test),
    }
    base_preds = {
        "train": resolve_path(args.base_preds_train),
        "val": resolve_path(args.base_preds_val),
        "test": resolve_path(args.base_preds_test),
    }

    checks = preflight(args.dataset_name, run_root, manifests, base_preds)
    bundle = load_split_bundle(manifests, base_preds)
    dataset_name = collect_dataset_name({split: bundle[split]["manifest"] for split in VALID_SPLITS}, args.dataset_name)

    train_records = bundle["train"]["manifest"]
    val_records = bundle["val"]["manifest"]
    test_records = bundle["test"]["manifest"]
    train_weight_map, train_confidence_map = build_weight_map(train_records, bundle["train"]["base_preds"], args.gamma, args.w_max)

    train_loader = build_loader(train_records, train_weight_map, args.batch_size, args.num_workers, train=True)
    val_loader = build_loader(val_records, None, args.eval_batch_size, args.num_workers, train=False)
    test_loader = build_loader(test_records, None, args.eval_batch_size, args.num_workers, train=False)

    model = prepare_model(args.init_mode, resolve_path(args.init_checkpoint) if args.init_checkpoint else None)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1), eta_min=args.lr * args.lr_min_factor)
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda"))

    best_val_f1 = -1.0
    best_epoch = 0
    history: List[Dict[str, Any]] = []
    ckpt_path = aux_root / "best.pth"

    for epoch in range(1, args.epochs + 1):
        train_stats = train_one_epoch(model, train_loader, optimizer, scaler)
        val_rows = predict_split(model, val_loader)
        val_metrics = compute_metrics(val_rows)
        scheduler.step()

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_stats["loss"],
                "train_accuracy": train_stats["accuracy"],
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
            }
        )
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            save_checkpoint(
                ckpt_path,
                model,
                history,
                best_epoch,
                best_val_f1,
                {
                    "dataset_name": dataset_name,
                    "seed": args.seed,
                    "init_mode": args.init_mode,
                    "gamma": args.gamma,
                    "w_max": args.w_max,
                    "lr": args.lr,
                    "weight_decay": args.weight_decay,
                    "epochs": args.epochs,
                },
            )

    if not ckpt_path.exists():
        save_checkpoint(
            ckpt_path,
            model,
            history,
            max(best_epoch, 0),
            max(best_val_f1, 0.0),
            {
                "dataset_name": dataset_name,
                "seed": args.seed,
                "init_mode": args.init_mode,
                "gamma": args.gamma,
                "w_max": args.w_max,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "epochs": args.epochs,
            },
        )

    best_ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(best_ckpt["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()

    split_outputs: Dict[str, List[Dict[str, Any]]] = {}
    split_metrics: Dict[str, Dict[str, Any]] = {}
    for split, loader in (("train", build_loader(train_records, None, args.eval_batch_size, args.num_workers, train=False)),
                          ("val", val_loader),
                          ("test", test_loader)):
        rows = predict_split(model, loader)
        split_outputs[split] = rows
        split_metrics[split] = compute_metrics(rows)
        write_jsonl(rows, aux_root / f"aux_predictions_{split}.jsonl")

    summary = {
        "dataset_name": dataset_name,
        "seed": args.seed,
        "run_id": run_id,
        "init_mode": args.init_mode,
        "checkpoint": str(ckpt_path),
        "inputs": {
            "manifests": {split: str(path) for split, path in manifests.items()},
            "base_predictions": {split: str(path) for split, path in base_preds.items()},
        },
        "preflight": checks,
        "history": history,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_f1,
        "weight_stats": summarize_weights(train_weight_map, train_confidence_map),
        "splits": split_metrics,
    }
    write_jsonl([summary], aux_root / "aux_summary.jsonl")
    (aux_root / "aux_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dataset-specific manifest-driven auxiliary trainer")
    parser.add_argument("--dataset-name", type=str, default="dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--manifest-train", type=Path, required=True)
    parser.add_argument("--manifest-val", type=Path, required=True)
    parser.add_argument("--manifest-test", type=Path, required=True)
    parser.add_argument("--base-preds-train", type=Path, required=True)
    parser.add_argument("--base-preds-val", type=Path, required=True)
    parser.add_argument("--base-preds-test", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lr-min-factor", type=float, default=0.05)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--w-max", type=float, default=10.0)
    parser.add_argument("--init-mode", type=str, choices=VALID_INIT_MODES, default="scratch")
    parser.add_argument("--init-checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    set_seed(args.seed)
    summary = train_auxiliary(args)
    print(json.dumps(
        {
            "dataset_name": summary["dataset_name"],
            "run_id": summary["run_id"],
            "checkpoint": summary["checkpoint"],
            "best_val_macro_f1": summary["best_val_macro_f1"],
        },
        ensure_ascii=False,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
