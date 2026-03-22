#!/usr/bin/env python3
"""
Sweep ICWMV w_spec from saved coral_eval JSONL outputs.

입력:
  - evaluate_variant()가 생성한 `*_mnv2_{ds}.jsonl`
  - evaluate_variant()가 생성한 `*_specm_{ds}.jsonl`

출력:
  - best w_spec와 dataset별 macro-F1 sweep JSON
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

CLASSES = ["authentic", "manipulated", "ai_generated"]
DATASETS = ["base", "dsC", "opensdi", "aigenproxy"]


def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def macro_f1(y_true: list[str], y_pred: list[str]) -> float:
    f1s = []
    for cls in CLASSES:
        tp = sum(t == cls and p == cls for t, p in zip(y_true, y_pred))
        fp = sum(t != cls and p == cls for t, p in zip(y_true, y_pred))
        fn = sum(t == cls and p != cls for t, p in zip(y_true, y_pred))
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)
        if sum(t == cls for t in y_true) > 0:
            f1s.append(f1)
    return float(np.mean(f1s)) if f1s else 0.0


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def resolve_variant_prefix(variant_dir: Path) -> str:
    summaries = sorted(variant_dir.glob("*_summary.json"))
    if summaries:
        data = json.loads(summaries[0].read_text())
        if "variant" in data:
            return str(data["variant"])
    return variant_dir.name


def evaluate_variant_dir(variant_dir: Path, weights: list[float]) -> dict:
    prefix = resolve_variant_prefix(variant_dir)
    results = []
    for w in weights:
        ds_scores = {}
        avg_scores = []
        for ds in DATASETS:
            mnv2_rows = load_jsonl(variant_dir / f"{prefix}_mnv2_{ds}.jsonl")
            specm_rows = load_jsonl(variant_dir / f"{prefix}_specm_{ds}.jsonl")
            specm_map = {row["image_path"]: row for row in specm_rows}

            y_true: list[str] = []
            y_pred: list[str] = []
            for row in mnv2_rows:
                true_label = row["true_label"]
                mnv2 = np.array([row["scores"][c] for c in CLASSES], dtype=np.float32)
                if true_label in {"authentic", "manipulated"}:
                    srow = specm_map[row["image_path"]]
                    specm = np.array([srow["authentic_score"], srow["manip_score"]], dtype=np.float32)
                    auth = (mnv2[0] + w * specm[0]) / (1.0 + w)
                    manip = (mnv2[1] + w * specm[1]) / (1.0 + w)
                    raw = np.array([auth, manip, mnv2[2]], dtype=np.float32)
                    probs = raw / raw.sum()
                    pred = CLASSES[int(np.argmax(probs))]
                else:
                    pred = CLASSES[int(np.argmax(mnv2))]
                y_true.append(true_label)
                y_pred.append(pred)

            score = macro_f1(y_true, y_pred)
            ds_scores[ds] = score
            avg_scores.append(score)

        results.append(
            {
                "w_spec": w,
                "avg_macro_f1": float(np.mean(avg_scores)),
                "per_dataset": ds_scores,
            }
        )
    results.sort(key=lambda row: row["avg_macro_f1"], reverse=True)
    return {"best": results[0], "results": results}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep ICWMV w_spec from saved JSONL outputs")
    parser.add_argument("--variant-dir", type=Path, required=True, help="coral_eval variant output dir")
    parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        default=[0.0, 0.1, 0.2, 0.25, 0.33, 0.4, 0.5, 0.67, 0.75, 1.0, 1.25, 1.5, 2.0],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = evaluate_variant_dir(args.variant_dir, args.weights)
    result["variant_dir"] = str(args.variant_dir)
    out_path = args.variant_dir / f"{args.variant_dir.name}_wspec_sweep_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"saved -> {out_path}")
    print(json.dumps(result["best"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
