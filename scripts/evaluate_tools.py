#!/usr/bin/env python3
"""
Tool re-evaluation runner.

Runs each tool against available datasets that align with the tool's target task.
Results are saved as JSON for reproducibility.
"""
import argparse
import json
import sys
import time
import os
import multiprocessing as mp
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

# Add repo root to path
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from PIL import Image

from src.tools.base_tool import Verdict
from src.tools.catnet_tool import CATNetAnalysisTool
from src.tools.spatial_tool import SpatialAnalysisTool
from src.tools.fatformer_tool import FatFormerTool

try:
    import torch
except Exception:
    torch = None


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def iter_images(root: Path, max_samples: int = 0) -> List[Path]:
    if not root.exists():
        return []
    files = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    files = sorted(files)
    if max_samples and max_samples > 0:
        files = files[:max_samples]
    return files


def load_image(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def load_mask(path: Path) -> np.ndarray:
    # Masks are grayscale; treat any non-zero as positive.
    mask = Image.open(path).convert("L")
    arr = np.array(mask)
    return (arr > 0).astype(np.uint8)


def align_mask(gt_mask: np.ndarray, pred_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if gt_mask.shape == pred_mask.shape:
        return gt_mask, pred_mask

    # Resize GT to match prediction (nearest for binary mask)
    gt_pil = Image.fromarray((gt_mask * 255).astype(np.uint8))
    gt_resized = gt_pil.resize((pred_mask.shape[1], pred_mask.shape[0]), Image.NEAREST)
    gt_arr = (np.array(gt_resized) > 0).astype(np.uint8)
    return gt_arr, pred_mask


@dataclass
class BinaryCounts:
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0
    uncertain: int = 0
    errors: int = 0

    def add(self, label: int, pred: int, uncertain: bool = False) -> None:
        if uncertain:
            self.uncertain += 1
        if label == 1 and pred == 1:
            self.tp += 1
        elif label == 0 and pred == 1:
            self.fp += 1
        elif label == 0 and pred == 0:
            self.tn += 1
        elif label == 1 and pred == 0:
            self.fn += 1

    def metrics(self) -> Dict[str, float]:
        total = self.tp + self.fp + self.tn + self.fn
        precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0
        recall = self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        accuracy = (self.tp + self.tn) / total if total else 0.0
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "total": total,
            "uncertain": self.uncertain,
            "errors": self.errors,
            "tp": self.tp,
            "fp": self.fp,
            "tn": self.tn,
            "fn": self.fn,
        }


def evaluate_binary_tool(
    tool,
    positive_paths: List[Path],
    negative_paths: List[Path],
    pred_fn: Callable,
    max_samples: int = 0,
) -> Dict[str, float]:
    counts = BinaryCounts()
    times: List[float] = []

    def _run(path: Path, label: int):
        nonlocal counts
        try:
            image = load_image(path)
            start = time.time()
            result = tool(image)
            times.append(time.time() - start)
            pred, is_uncertain = pred_fn(result)
            counts.add(label, pred, uncertain=is_uncertain)
        except Exception:
            counts.errors += 1

    if max_samples and max_samples > 0:
        positive_paths = positive_paths[:max_samples]
        negative_paths = negative_paths[:max_samples]

    for path in positive_paths:
        _run(path, label=1)
    for path in negative_paths:
        _run(path, label=0)

    metrics = counts.metrics()
    metrics["avg_seconds"] = float(np.mean(times)) if times else 0.0
    return metrics


def chunk_list(items: List[Path], chunks: int) -> List[List[Path]]:
    if chunks <= 1:
        return [items]
    buckets = [[] for _ in range(chunks)]
    for idx, item in enumerate(items):
        buckets[idx % chunks].append(item)
    return buckets


def noise_worker(task: Tuple[int, str, List[Path], List[Path]]) -> Dict[str, object]:
    gpu_id, backend, pos_paths, neg_paths = task
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    try:
        import torch
        torch.cuda.set_device(gpu_id)
    except Exception:
        torch = None

    from src.tools.noise_tool import NoiseAnalysisTool

    device = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
    tool = NoiseAnalysisTool(device=device, backend=backend)

    counts = BinaryCounts()
    times: List[float] = []

    def _run(path: Path, label: int):
        try:
            image = load_image(path)
            start = time.time()
            result = tool(image)
            times.append(time.time() - start)
            pred = int(result.verdict in {Verdict.MANIPULATED, Verdict.AI_GENERATED})
            counts.add(label, pred, uncertain=(result.verdict == Verdict.UNCERTAIN))
        except Exception:
            counts.errors += 1

    for path in pos_paths:
        _run(path, label=1)
    for path in neg_paths:
        _run(path, label=0)

    return {
        "tp": counts.tp,
        "fp": counts.fp,
        "tn": counts.tn,
        "fn": counts.fn,
        "uncertain": counts.uncertain,
        "errors": counts.errors,
        "times": times,
    }


def evaluate_noise_parallel(
    backend: str,
    positive_paths: List[Path],
    negative_paths: List[Path],
    workers: int,
) -> Dict[str, float]:
    if workers <= 1:
        tool = __import__("src.tools.noise_tool", fromlist=["NoiseAnalysisTool"]).NoiseAnalysisTool(
            device="cuda", backend=backend
        )

        def noise_pred(result):
            pred = int(result.verdict in {Verdict.MANIPULATED, Verdict.AI_GENERATED})
            return pred, result.verdict == Verdict.UNCERTAIN

        return evaluate_binary_tool(tool, positive_paths, negative_paths, noise_pred, max_samples=0)

    pos_chunks = chunk_list(positive_paths, workers)
    neg_chunks = chunk_list(negative_paths, workers)
    tasks = [(idx, backend, pos_chunks[idx], neg_chunks[idx]) for idx in range(workers)]

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=workers) as pool:
        results = pool.map(noise_worker, tasks)

    counts = BinaryCounts()
    times: List[float] = []
    for res in results:
        counts.tp += res["tp"]
        counts.fp += res["fp"]
        counts.tn += res["tn"]
        counts.fn += res["fn"]
        counts.uncertain += res["uncertain"]
        counts.errors += res["errors"]
        times.extend(res["times"])

    metrics = counts.metrics()
    metrics["avg_seconds"] = float(np.mean(times)) if times else 0.0
    return metrics


def evaluate_spatial_masks(
    tool,
    images_dir: Path,
    masks_dir: Path,
    mask_suffix: str,
    mask_threshold: float = 0.5,
    max_samples: int = 0,
) -> Dict[str, float]:
    image_paths = iter_images(images_dir, max_samples=max_samples)
    ious = []
    f1s = []
    precisions = []
    recalls = []
    times = []
    skipped = 0
    errors = 0

    for image_path in image_paths:
        mask_path = masks_dir / f"{image_path.stem}{mask_suffix}"
        if not mask_path.exists():
            skipped += 1
            continue

        try:
            image = load_image(image_path)
            start = time.time()
            result = tool(image)
            times.append(time.time() - start)

            if result.manipulation_mask is None:
                skipped += 1
                continue

            pred_mask = (result.manipulation_mask >= mask_threshold).astype(np.uint8)
            gt_mask = load_mask(mask_path)
            gt_mask, pred_mask = align_mask(gt_mask, pred_mask)

            tp = int(np.logical_and(pred_mask == 1, gt_mask == 1).sum())
            fp = int(np.logical_and(pred_mask == 1, gt_mask == 0).sum())
            fn = int(np.logical_and(pred_mask == 0, gt_mask == 1).sum())

            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
            iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0

            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
            ious.append(iou)

        except Exception:
            errors += 1

    return {
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        "mean_f1": float(np.mean(f1s)) if f1s else 0.0,
        "mean_precision": float(np.mean(precisions)) if precisions else 0.0,
        "mean_recall": float(np.mean(recalls)) if recalls else 0.0,
        "total": len(image_paths),
        "used": len(ious),
        "skipped": skipped,
        "errors": errors,
        "avg_seconds": float(np.mean(times)) if times else 0.0,
        "mask_threshold": float(mask_threshold),
    }


def clear_cuda_cache() -> None:
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-evaluate MAIFS tools on available datasets")
    parser.add_argument("--max-samples", type=int, default=0, help="max samples per dataset (0 = all)")
    parser.add_argument("--out", type=str, default="", help="output JSON path")
    parser.add_argument("--noise-backend", type=str, default="mvss", help="noise backend: prnu|mvss")
    parser.add_argument("--noise-workers", type=int, default=1, help="noise eval workers (use >1 for multi-GPU)")
    args = parser.parse_args()

    datasets_dir = REPO_ROOT / "datasets"

    paths = {
        "genimage": datasets_dir / "GenImage_subset" / "BigGAN" / "val",
        "casia2": datasets_dir / "CASIA2_subset",
        "imd2020": datasets_dir / "IMD2020_subset" / "IMD2020_Generative_Image_Inpainting_yu2018_01",
        "hinet": REPO_ROOT / "HiNet-main" / "image",
    }

    results: Dict[str, Dict[str, object]] = {
        "run_at": datetime.utcnow().isoformat() + "Z",
        "max_samples": args.max_samples,
        "paths": {k: str(v) for k, v in paths.items()},
        "tools": {},
    }

    # Frequency slot -> CAT-Net compression tool on CASIA2 (Tp vs Au)
    freq_tool = CATNetAnalysisTool()
    ai_paths = iter_images(paths["genimage"] / "ai", args.max_samples)
    real_paths = iter_images(paths["genimage"] / "nature", args.max_samples)
    tp_paths = iter_images(paths["casia2"] / "Tp", args.max_samples)
    au_paths = iter_images(paths["casia2"] / "Au", args.max_samples)

    def freq_pred(result):
        pred = int(result.verdict in {Verdict.MANIPULATED, Verdict.AI_GENERATED})
        return pred, result.verdict == Verdict.UNCERTAIN

    results["tools"]["frequency"] = {
        "dataset": "CASIA2_subset (Tp vs Au) [CAT-Net compression]",
        "counts": {
            "tp": len(tp_paths),
            "au": len(au_paths),
        },
        "metrics": evaluate_binary_tool(freq_tool, tp_paths, au_paths, freq_pred, args.max_samples),
    }

    clear_cuda_cache()

    # Noise tool -> CASIA2 (Tp vs Au)
    results["tools"]["noise"] = {
        "dataset": "CASIA2_subset (Tp vs Au)",
        "backend": args.noise_backend,
        "counts": {
            "tp": len(tp_paths),
            "au": len(au_paths),
        },
        "metrics": evaluate_noise_parallel(
            args.noise_backend,
            tp_paths,
            au_paths,
            max(1, args.noise_workers)
        ),
    }

    clear_cuda_cache()

    # FatFormer tool
    fat_tool = FatFormerTool()
    steg_paths = iter_images(paths["hinet"] / "steg", args.max_samples)
    cover_paths = iter_images(paths["hinet"] / "cover", args.max_samples)

    fat_dataset_name = "HiNet-main/image (steg vs cover)"
    fat_pos_paths = steg_paths
    fat_neg_paths = cover_paths

    # HiNet 샘플이 없으면 GenImage(BigGAN)로 폴백
    if len(fat_pos_paths) == 0 and len(fat_neg_paths) == 0:
        fat_dataset_name = "GenImage_subset/BigGAN/val (ai vs nature)"
        fat_pos_paths = ai_paths
        fat_neg_paths = real_paths

    def fat_pred(result):
        return int(result.verdict == Verdict.AI_GENERATED), result.verdict == Verdict.UNCERTAIN

    results["tools"]["fatformer"] = {
        "dataset": fat_dataset_name,
        "counts": {
            "positive": len(fat_pos_paths),
            "negative": len(fat_neg_paths),
        },
        "metrics": evaluate_binary_tool(fat_tool, fat_pos_paths, fat_neg_paths, fat_pred, args.max_samples),
    }

    clear_cuda_cache()

    # Spatial tool -> IMD2020 masks
    spatial_tool = SpatialAnalysisTool()
    imd_images = paths["imd2020"] / "images"
    imd_masks = paths["imd2020"] / "masks"

    results["tools"]["spatial_imd2020"] = {
        "dataset": "IMD2020_subset (inpainting masks)",
        "metrics": evaluate_spatial_masks(
            spatial_tool,
            imd_images,
            imd_masks,
            mask_suffix="_mask.jpg",
            mask_threshold=float(getattr(spatial_tool, "mask_threshold", 0.5)),
            max_samples=args.max_samples,
        ),
    }

    clear_cuda_cache()

    # Spatial tool -> CASIA2 GT masks
    casia_tp_dir = paths["casia2"] / "Tp"
    casia_gt_dir = paths["casia2"] / "GT"
    results["tools"]["spatial_casia2"] = {
        "dataset": "CASIA2_subset (GT masks)",
        "metrics": evaluate_spatial_masks(
            spatial_tool,
            casia_tp_dir,
            casia_gt_dir,
            mask_suffix="_gt.png",
            mask_threshold=float(getattr(spatial_tool, "mask_threshold", 0.5)),
            max_samples=args.max_samples,
        ),
    }

    out_path = Path(args.out) if args.out else REPO_ROOT / "outputs" / f"tool_reeval_{int(time.time())}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"Saved results: {out_path}")


if __name__ == "__main__":
    main()
