#!/usr/bin/env python3
"""
RPi5 deployment-path comparison: current ONNX vs Coral TFLite variants.

목적:
  1. `specm_v4_coral`의 실제 배포 경로 정확도를 4-DS에서 재평가
  2. `mnv2_coral + specm_v4_coral` ICWMV가 현재 ONNX 배포 경로 대비
     얼마나 드리프트하는지 정량화
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inference_rpi5 import OnnxSession, TFLiteSession, icwmv_fuse, load_image, softmax

OUT_DIR = ROOT / "experiments" / "results" / "coral_eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ONNX_Q = ROOT / "weights" / "onnx_quant"
TFLITE_Q = ROOT / "weights" / "tflite"

DATASETS = {
    "base": ROOT / "experiments" / "results" / "phase2_patha_scale500_gain_predictor" / "patha_agent_outputs_20260304_080157.jsonl",
    "dsC": ROOT / "experiments" / "results" / "phase2_patha_case3_scale300_dsC" / "patha_agent_outputs_20260303_105005.jsonl",
    "opensdi": ROOT / "experiments" / "results" / "phase2_patha_case3_opensdi_scale300" / "patha_agent_outputs_fixed_seed42.jsonl",
    "aigenproxy": ROOT / "experiments" / "results" / "phase2_patha_case3_aigenproxy_scale300" / "patha_agent_outputs_fixed_seed42.jsonl",
}

BINARY_LABELS = {"authentic": 0, "manipulated": 1}
CLASSES_3 = ["authentic", "manipulated", "ai_generated"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Coral deployment variants")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASETS),
        choices=list(DATASETS),
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def compute_multiclass_stats(y_true: list[str], y_pred: list[str]) -> dict:
    accuracy = safe_div(sum(int(t == p) for t, p in zip(y_true, y_pred)), len(y_true))
    per_class = {}
    f1s = []
    recalls = []
    for cls in CLASSES_3:
        tp = sum(int(t == cls and p == cls) for t, p in zip(y_true, y_pred))
        fp = sum(int(t != cls and p == cls) for t, p in zip(y_true, y_pred))
        fn = sum(int(t == cls and p != cls) for t, p in zip(y_true, y_pred))
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)
        per_class[cls] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "n": sum(int(t == cls) for t in y_true),
        }
        if per_class[cls]["n"] > 0:
            f1s.append(f1)
            recalls.append(recall)
    return {
        "accuracy": accuracy,
        "macro_f1": float(np.mean(f1s)) if f1s else 0.0,
        "macro_recall": float(np.mean(recalls)) if recalls else 0.0,
        "per_class": per_class,
    }


def compute_binary_stats(y_true: list[int], y_pred: list[int]) -> dict:
    accuracy = safe_div(sum(int(t == p) for t, p in zip(y_true, y_pred)), len(y_true))
    per_class = {}
    recalls = []
    for idx, name in ((0, "authentic"), (1, "manipulated")):
        mask = [i for i, label in enumerate(y_true) if label == idx]
        recall = safe_div(sum(int(y_pred[i] == idx) for i in mask), len(mask))
        per_class[name] = {"recall": recall, "n": len(mask)}
        recalls.append(recall)
    tp = sum(int(t == 1 and p == 1) for t, p in zip(y_true, y_pred))
    fp = sum(int(t == 0 and p == 1) for t, p in zip(y_true, y_pred))
    fn = sum(int(t == 1 and p == 0) for t, p in zip(y_true, y_pred))
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    manip_f1 = safe_div(2 * precision * recall, precision + recall)
    return {
        "accuracy": accuracy,
        "macro_recall": float(np.mean(recalls)) if recalls else 0.0,
        "auth_recall": per_class["authentic"]["recall"],
        "manip_recall": recall,
        "manip_precision": precision,
        "manip_f1": manip_f1,
        "per_class": per_class,
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_session(kind: str, model_path: Path, threads: int):
    if kind == "onnx":
        return OnnxSession(model_path, threads)
    if kind == "tflite":
        return TFLiteSession(model_path, threads=threads, use_edgetpu=False, delegate_path="libedgetpu.so.1")
    raise ValueError(f"unknown session kind: {kind}")


def evaluate_variant(
    name: str,
    mnv2_kind: str,
    mnv2_path: Path,
    specm_kind: str,
    specm_path: Path,
    datasets: list[str],
    threads: int,
) -> dict:
    mnv2_sess = make_session(mnv2_kind, mnv2_path, threads)
    specm_sess = make_session(specm_kind, specm_path, threads)

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    variant_dir = OUT_DIR / f"{name}_{ts}"
    variant_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "variant": name,
        "mnv2_backend": mnv2_kind,
        "mnv2_path": str(mnv2_path),
        "specm_backend": specm_kind,
        "specm_path": str(specm_path),
        "datasets": {},
    }

    for ds in datasets:
        records = load_jsonl(DATASETS[ds])
        spec_rows: list[dict] = []
        pair_rows: list[dict] = []
        mnv2_rows: list[dict] = []

        mnv2_true: list[str] = []
        mnv2_pred: list[str] = []
        pair_true: list[str] = []
        pair_pred: list[str] = []
        spec_true: list[int] = []
        spec_pred: list[int] = []

        for rec in records:
            image_path = ROOT / rec["image_path"]
            x = load_image(image_path)

            mnv2_logits = mnv2_sess.run(x)
            mnv2_probs = softmax(mnv2_logits)
            mnv2_label = CLASSES_3[int(np.argmax(mnv2_probs))]

            mnv2_rows.append({
                "image_path": rec["image_path"],
                "true_label": rec["true_label"],
                "pred_label": mnv2_label,
                "scores": {cls: float(mnv2_probs[i]) for i, cls in enumerate(CLASSES_3)},
                "confidence": float(np.max(mnv2_probs)),
            })
            mnv2_true.append(rec["true_label"])
            mnv2_pred.append(mnv2_label)

            if rec["true_label"] in BINARY_LABELS:
                specm_logits = specm_sess.run(x)
                specm_probs = softmax(specm_logits)
                specm_label = "manipulated" if int(np.argmax(specm_probs)) == 1 else "authentic"
                spec_rows.append({
                    "image_path": rec["image_path"],
                    "true_label": rec["true_label"],
                    "pred_label": specm_label,
                    "authentic_score": float(specm_probs[0]),
                    "manip_score": float(specm_probs[1]),
                    "confidence": float(np.max(specm_probs)),
                })
                spec_true.append(BINARY_LABELS[rec["true_label"]])
                spec_pred.append(int(np.argmax(specm_probs)))

                pair_label, pair_probs = icwmv_fuse(mnv2_probs, specm_probs, w_spec=1.0)
            else:
                pair_probs = mnv2_probs
                pair_label = mnv2_label

            pair_rows.append({
                "image_path": rec["image_path"],
                "true_label": rec["true_label"],
                "pred_label": pair_label,
                "scores": {cls: float(pair_probs[i]) for i, cls in enumerate(CLASSES_3)},
                "confidence": float(np.max(pair_probs)),
            })
            pair_true.append(rec["true_label"])
            pair_pred.append(pair_label)

        mnv2_stats = compute_multiclass_stats(mnv2_true, mnv2_pred)
        pair_stats = compute_multiclass_stats(pair_true, pair_pred)
        spec_stats = compute_binary_stats(spec_true, spec_pred)

        write_jsonl(variant_dir / f"{name}_mnv2_{ds}.jsonl", mnv2_rows)
        write_jsonl(variant_dir / f"{name}_specm_{ds}.jsonl", spec_rows)
        write_jsonl(variant_dir / f"{name}_icwmv_{ds}.jsonl", pair_rows)

        result["datasets"][ds] = {
            "mnv2": mnv2_stats,
            "specm": spec_stats,
            "icwmv": pair_stats,
            "n_total": len(records),
            "n_binary": len(spec_rows),
        }
        print(
            f"[{name}:{ds}] "
            f"mnv2_f1={mnv2_stats['macro_f1']:.4f} | "
            f"specm_f1={spec_stats['manip_f1']:.4f} | "
            f"icwmv_f1={pair_stats['macro_f1']:.4f}"
        )

    icwmv_avg = float(np.mean([result["datasets"][ds]["icwmv"]["macro_f1"] for ds in datasets]))
    specm_avg = float(np.mean([result["datasets"][ds]["specm"]["manip_f1"] for ds in datasets]))
    mnv2_avg = float(np.mean([result["datasets"][ds]["mnv2"]["macro_f1"] for ds in datasets]))
    result["avg"] = {
        "mnv2_macro_f1": mnv2_avg,
        "specm_manip_f1": specm_avg,
        "icwmv_macro_f1": icwmv_avg,
    }

    summary_path = variant_dir / f"{name}_summary.json"
    summary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"[{name}] saved -> {summary_path}")
    return result


def main() -> None:
    args = parse_args()
    variants = [
        (
            "current_onnx",
            "onnx",
            ONNX_Q / "mnv2_int8_dynamic.onnx",
            "onnx",
            ONNX_Q / "specm_v4_int8_dynamic.onnx",
        ),
        (
            "coral_tflite",
            "tflite",
            TFLITE_Q / "mnv2_coral_int8_full.tflite",
            "tflite",
            TFLITE_Q / "specm_v4_coral_int8_full.tflite",
        ),
    ]

    for _, _, mnv2_path, _, specm_path in variants:
        for path in (mnv2_path, specm_path):
            if not path.exists():
                raise FileNotFoundError(path)

    all_results = []
    for name, mnv2_kind, mnv2_path, specm_kind, specm_path in variants:
        all_results.append(
            evaluate_variant(
                name=name,
                mnv2_kind=mnv2_kind,
                mnv2_path=mnv2_path,
                specm_kind=specm_kind,
                specm_path=specm_path,
                datasets=args.datasets,
                threads=args.threads,
            )
        )

    compare = {"timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"), "results": all_results}
    if len(all_results) == 2:
        a, b = all_results
        compare["delta_coral_minus_current"] = {
            "mnv2_macro_f1": b["avg"]["mnv2_macro_f1"] - a["avg"]["mnv2_macro_f1"],
            "specm_manip_f1": b["avg"]["specm_manip_f1"] - a["avg"]["specm_manip_f1"],
            "icwmv_macro_f1": b["avg"]["icwmv_macro_f1"] - a["avg"]["icwmv_macro_f1"],
            "per_dataset_icwmv": {
                ds: b["datasets"][ds]["icwmv"]["macro_f1"] - a["datasets"][ds]["icwmv"]["macro_f1"]
                for ds in args.datasets
            },
            "per_dataset_specm_manip_f1": {
                ds: b["datasets"][ds]["specm"]["manip_f1"] - a["datasets"][ds]["specm"]["manip_f1"]
                for ds in args.datasets
            },
        }
    out_path = OUT_DIR / f"coral_eval_compare_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(compare, ensure_ascii=False, indent=2))
    print(f"\ncompare saved -> {out_path}")


if __name__ == "__main__":
    main()
