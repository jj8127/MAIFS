#!/usr/bin/env python3
"""
Calibrate tool thresholds using local datasets.

- Frequency: GenImage BigGAN ai vs nature
- Noise: CASIA2 Tp vs Au

Outputs JSON with recommended thresholds.
"""
import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

# Add repo root
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.tools.frequency_tool import FrequencyAnalysisTool
from src.tools.noise_tool import NoiseAnalysisTool
from src.tools.fatformer_tool import FatFormerTool
from src.tools.base_tool import Verdict


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


@dataclass
class Metrics:
    f1: float
    precision: float
    recall: float
    accuracy: float
    balanced_accuracy: float
    tp: int
    fp: int
    tn: int
    fn: int


def compute_metrics(labels: List[int], preds: List[int]) -> Metrics:
    tp = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 1)
    fp = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 1)
    tn = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 0)
    fn = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 0)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    accuracy = (tp + tn) / max(1, (tp + tn + fp + fn))
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    balanced_accuracy = (recall + tnr) / 2.0
    return Metrics(f1, precision, recall, accuracy, balanced_accuracy, tp, fp, tn, fn)


def find_best_threshold(scores: List[float], labels: List[int]) -> Tuple[float, Metrics]:
    best_t = 0.5
    best_metrics = Metrics(0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0)
    for t in np.linspace(0.0, 1.0, 101):
        preds = [1 if s >= t else 0 for s in scores]
        metrics = compute_metrics(labels, preds)
        if metrics.balanced_accuracy > best_metrics.balanced_accuracy:
            best_t = float(t)
            best_metrics = metrics
    return best_t, best_metrics


def calibrate_frequency(max_samples: int) -> dict:
    tool = FrequencyAnalysisTool()
    ai_dir = REPO_ROOT / "datasets" / "GenImage_subset" / "BigGAN" / "val" / "ai"
    real_dir = REPO_ROOT / "datasets" / "GenImage_subset" / "BigGAN" / "val" / "nature"

    ai_paths = iter_images(ai_dir, max_samples)
    real_paths = iter_images(real_dir, max_samples)

    scores = []
    labels = []
    jpeg_flags = []

    for path in ai_paths:
        result = tool(load_image(path))
        scores.append(float(result.evidence.get("ai_generation_score", 0.0)))
        jpeg_flags.append(bool(result.evidence.get("grid_analysis", {}).get("is_likely_jpeg", False)))
        labels.append(1)

    for path in real_paths:
        result = tool(load_image(path))
        scores.append(float(result.evidence.get("ai_generation_score", 0.0)))
        jpeg_flags.append(bool(result.evidence.get("grid_analysis", {}).get("is_likely_jpeg", False)))
        labels.append(0)

    # Search best JPEG penalty (score reduction when JPEG detected).
    best_penalty = 0.0
    best_threshold = 0.5
    best_metrics = Metrics(0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0)
    best_score = -1.0

    for penalty in np.linspace(0.0, 0.3, 31):
        adjusted = [max(0.0, s - penalty) if is_jpeg else s for s, is_jpeg in zip(scores, jpeg_flags)]
        t, metrics = find_best_threshold(adjusted, labels)
        # 단일 지표 과적합 방지를 위해 accuracy/F1/balanced_accuracy를 함께 최적화
        composite_score = (metrics.accuracy + metrics.f1 + metrics.balanced_accuracy) / 3.0
        if composite_score > best_score:
            best_score = composite_score
            best_penalty = float(penalty)
            best_threshold = float(t)
            best_metrics = metrics

    return {
        "ai_threshold": best_threshold,
        "auth_threshold": best_threshold,
        "jpeg_penalty": best_penalty,
        "uncertain_margin": 0.0,
        "counts": {
            "ai": len(ai_paths),
            "nature": len(real_paths),
            "jpeg_detected": int(sum(1 for f in jpeg_flags if f)),
        },
        "metrics": {
            "f1": best_metrics.f1,
            "precision": best_metrics.precision,
            "recall": best_metrics.recall,
            "accuracy": best_metrics.accuracy,
            "balanced_accuracy": best_metrics.balanced_accuracy,
            "tp": best_metrics.tp,
            "fp": best_metrics.fp,
            "tn": best_metrics.tn,
            "fn": best_metrics.fn,
        },
    }


def calibrate_noise(max_samples: int, backend: str) -> dict:
    device = "cuda" if backend == "mvss" else "cpu"
    tool = NoiseAnalysisTool(device=device, backend=backend)
    tp_dir = REPO_ROOT / "datasets" / "CASIA2_subset" / "Tp"
    au_dir = REPO_ROOT / "datasets" / "CASIA2_subset" / "Au"

    tp_paths = iter_images(tp_dir, max_samples)
    au_paths = iter_images(au_dir, max_samples)

    scores = []
    labels = []
    diversity_auth = []

    if backend == "mvss":
        for path in tp_paths:
            result = tool(load_image(path))
            score = float(result.evidence.get("mvss_score", 0.0))
            scores.append(score)
            labels.append(1)

        for path in au_paths:
            result = tool(load_image(path))
            score = float(result.evidence.get("mvss_score", 0.0))
            scores.append(score)
            labels.append(0)

        best_threshold, best_metrics = find_best_threshold(scores, labels)
        return {
            "backend": "mvss",
            "mvss_threshold": best_threshold,
            "mvss_auth_threshold": best_threshold,
            "mvss_uncertain_margin": 0.0,
            "counts": {
                "tp": len(tp_paths),
                "au": len(au_paths),
            },
            "metrics": {
                "f1": best_metrics.f1,
                "precision": best_metrics.precision,
                "recall": best_metrics.recall,
                "accuracy": best_metrics.accuracy,
                "balanced_accuracy": best_metrics.balanced_accuracy,
                "tp": best_metrics.tp,
                "fp": best_metrics.fp,
                "tn": best_metrics.tn,
                "fn": best_metrics.fn,
            },
        }

    for path in tp_paths:
        result = tool(load_image(path))
        manipulation_score = float(result.evidence.get("consistency_analysis", {}).get("manipulation_score", 0.0))
        scores.append(manipulation_score)
        labels.append(1)

    for path in au_paths:
        result = tool(load_image(path))
        manipulation_score = float(result.evidence.get("consistency_analysis", {}).get("manipulation_score", 0.0))
        diversity = float(result.evidence.get("consistency_analysis", {}).get("natural_diversity_score", 0.0))
        scores.append(manipulation_score)
        labels.append(0)
        diversity_auth.append(diversity)

    best_threshold, best_metrics = find_best_threshold(scores, labels)

    diversity_threshold = 0.5
    if diversity_auth:
        diversity_threshold = float(np.percentile(diversity_auth, 25))

    return {
        "backend": "prnu",
        "manipulation_threshold": best_threshold,
        "authentic_diversity_threshold": diversity_threshold,
        "counts": {
            "tp": len(tp_paths),
            "au": len(au_paths),
        },
        "metrics": {
            "f1": best_metrics.f1,
            "precision": best_metrics.precision,
            "recall": best_metrics.recall,
            "accuracy": best_metrics.accuracy,
            "balanced_accuracy": best_metrics.balanced_accuracy,
            "tp": best_metrics.tp,
            "fp": best_metrics.fp,
            "tn": best_metrics.tn,
            "fn": best_metrics.fn,
        },
    }


def calibrate_fatformer(max_samples: int) -> dict:
    """Calibrate FatFormer fake_probability threshold on GenImage BigGAN."""
    tool = FatFormerTool(device="cuda")
    ai_dir = REPO_ROOT / "datasets" / "GenImage_subset" / "BigGAN" / "val" / "ai"
    real_dir = REPO_ROOT / "datasets" / "GenImage_subset" / "BigGAN" / "val" / "nature"

    ai_paths = iter_images(ai_dir, max_samples)
    real_paths = iter_images(real_dir, max_samples)

    scores = []
    labels = []

    for path in ai_paths:
        result = tool(load_image(path))
        scores.append(float(result.evidence.get("fake_probability", 0.0)))
        labels.append(1)

    for path in real_paths:
        result = tool(load_image(path))
        scores.append(float(result.evidence.get("fake_probability", 0.0)))
        labels.append(0)

    best_t = 0.5
    best_metrics = Metrics(0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0)
    for t in np.linspace(0.0, 1.0, 2001):
        preds = [1 if s >= t else 0 for s in scores]
        m = compute_metrics(labels, preds)
        if (m.f1 > best_metrics.f1) or (
            m.f1 == best_metrics.f1 and m.balanced_accuracy > best_metrics.balanced_accuracy
        ):
            best_t = float(t)
            best_metrics = m

    return {
        "ai_threshold": best_t,
        "auth_threshold": best_t,
        "counts": {"ai": len(ai_paths), "nature": len(real_paths)},
        "metrics": {
            "f1": best_metrics.f1,
            "precision": best_metrics.precision,
            "recall": best_metrics.recall,
            "accuracy": best_metrics.accuracy,
            "balanced_accuracy": best_metrics.balanced_accuracy,
            "tp": best_metrics.tp,
            "fp": best_metrics.fp,
            "tn": best_metrics.tn,
            "fn": best_metrics.fn,
        },
    }


def default_spatial_params() -> dict:
    """Spatial defaults from local grid-search (CASIA + IMD trade-off)."""
    return {
        "mask_threshold": 0.4,
        "mvss_weight": 0.5,
        "authentic_ratio_threshold": 0.05,
        "ai_ratio_threshold": 0.8,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate tool thresholds using local datasets")
    parser.add_argument("--max-samples", type=int, default=0, help="max samples per dataset (0 = all)")
    parser.add_argument("--out", type=str, default=str(REPO_ROOT / "configs" / "tool_thresholds.json"))
    parser.add_argument("--noise-backend", type=str, default="prnu", help="noise backend: prnu|mvss")
    args = parser.parse_args()

    data = {
        "calibrated_at": datetime.utcnow().isoformat() + "Z",
        "max_samples": args.max_samples,
        "frequency": calibrate_frequency(args.max_samples),
        "noise": calibrate_noise(args.max_samples, args.noise_backend),
        "fatformer": calibrate_fatformer(args.max_samples),
        "spatial": default_spatial_params(),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Saved thresholds: {out_path}")


if __name__ == "__main__":
    main()
