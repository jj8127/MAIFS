#!/usr/bin/env python3
"""
Phase 4.10 — MNV2 Coral quantization sweep
==========================================

목적:
  1. `mnv2_coral`의 full INT8 TFLite PTQ 드리프트를 줄일 수 있는지 탐색
  2. `onnx2tf`의 quantization mode / calibration count / IO dtype 조합 비교
  3. 최고 성능 후보를 별도 산출물로 저장하고 Edge TPU compile까지 검증

출력:
  weights/tflite_sweep/*.tflite
  weights/tflite_edgetpu_sweep/*.tflite
  experiments/results/coral_quant_sweep/mnv2_coral_quant_sweep_{ts}.json

실행 예시:
  .venv-edgetpu-export/bin/python experiments/run_mnv2_coral_quant_sweep.py
  .venv-edgetpu-export/bin/python experiments/run_mnv2_coral_quant_sweep.py \
      --calib-counts 64 128 256 --quant-types per-channel per-tensor
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from eval_coral_rpi5_variants import DATASETS, compute_multiclass_stats, load_jsonl, make_session
from inference_rpi5 import CLASSES_3, load_image, softmax
from run_edgetpu_export import (
    MODEL_SPECS,
    build_calibration_npy,
    collect_calibration_images,
    compile_for_edgetpu,
    convert_to_tflite,
    export_coral_onnx,
)

OUT_DIR = ROOT / "experiments" / "results" / "coral_quant_sweep"
TFLITE_SWEEP_DIR = ROOT / "weights" / "tflite_sweep"
EDGE_SWEEP_DIR = ROOT / "weights" / "tflite_edgetpu_sweep"
ONNX_BASELINE = ROOT / "weights" / "onnx_quant" / "mnv2_int8_dynamic.onnx"
TFLITE_BASELINE = ROOT / "weights" / "tflite" / "mnv2_coral_int8_full.tflite"

for path in (OUT_DIR, TFLITE_SWEEP_DIR, EDGE_SWEEP_DIR):
    path.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep MNV2 Coral PTQ settings")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASETS),
        choices=list(DATASETS),
    )
    parser.add_argument(
        "--calib-counts",
        nargs="+",
        type=int,
        default=[64, 128, 256],
        help="calibration image counts to sweep (default: 64 128 256)",
    )
    parser.add_argument(
        "--quant-types",
        nargs="+",
        default=["per-channel", "per-tensor"],
        choices=["per-channel", "per-tensor"],
        help="onnx2tf quantization modes to sweep",
    )
    parser.add_argument(
        "--io-dtypes",
        nargs="+",
        default=["int8"],
        choices=["int8", "uint8"],
        help="input/output quant dtype to sweep together (default: int8)",
    )
    parser.add_argument(
        "--compile-best",
        action="store_true",
        help="best TFLite 후보를 Edge TPU로 compile",
    )
    parser.add_argument(
        "--force-convert",
        action="store_true",
        help="이미 존재하는 sweep 산출물도 다시 생성",
    )
    parser.add_argument(
        "--refresh-onnx",
        action="store_true",
        help="mnv2_coral ONNX를 체크포인트에서 다시 export",
    )
    return parser.parse_args()


def encode_variant_name(quant_type: str, calib_count: int, io_dtype: str) -> str:
    qt = "pc" if quant_type == "per-channel" else "pt"
    return f"mnv2_coral_qsweep_qt{qt}_cal{calib_count:03d}_io{io_dtype}"


def evaluate_variant(name: str, kind: str, model_path: Path, datasets: list[str], threads: int) -> dict:
    session = make_session(kind, model_path, threads)
    result = {
        "variant": name,
        "backend": kind,
        "model_path": str(model_path),
        "datasets": {},
    }

    for ds in datasets:
        records = load_jsonl(DATASETS[ds])
        y_true: list[str] = []
        y_pred: list[str] = []

        for rec in records:
            image_path = ROOT / rec["image_path"]
            x = load_image(image_path)
            probs = softmax(session.run(x))
            label = CLASSES_3[int(np.argmax(probs))]
            y_true.append(rec["true_label"])
            y_pred.append(label)

        stats = compute_multiclass_stats(y_true, y_pred)
        result["datasets"][ds] = stats
        print(f"[{name}:{ds}] macro_f1={stats['macro_f1']:.4f} acc={stats['accuracy']:.4f}")

    result["avg_macro_f1"] = float(np.mean([result["datasets"][ds]["macro_f1"] for ds in datasets]))
    print(f"[{name}] avg_macro_f1={result['avg_macro_f1']:.4f}")
    return result


def main() -> None:
    args = parse_args()
    spec = MODEL_SPECS["mnv2_coral"]
    export_coral_onnx(spec, force=args.refresh_onnx)

    baseline_results: list[dict] = []
    if ONNX_BASELINE.exists():
        baseline_results.append(
            evaluate_variant("current_onnx", "onnx", ONNX_BASELINE, args.datasets, args.threads)
        )
    if TFLITE_BASELINE.exists():
        baseline_results.append(
            evaluate_variant("baseline_coral_tflite", "tflite", TFLITE_BASELINE, args.datasets, args.threads)
        )

    max_calib = max(args.calib_counts)
    calib_images = collect_calibration_images(max_calib)
    candidates: list[dict] = []

    for quant_type in args.quant_types:
        for calib_count in args.calib_counts:
            subset = calib_images[:calib_count]
            calib_path = build_calibration_npy(spec, subset)
            for io_dtype in args.io_dtypes:
                name = encode_variant_name(quant_type, calib_count, io_dtype)
                tflite_path = TFLITE_SWEEP_DIR / f"{name}.tflite"
                entry: dict = {
                    "variant": name,
                    "quant_type": quant_type,
                    "calibration_count": calib_count,
                    "io_dtype": io_dtype,
                    "calibration_npy": str(calib_path),
                    "tflite_path": str(tflite_path),
                    "status": "pending",
                }
                try:
                    if args.force_convert or not tflite_path.exists():
                        converted = convert_to_tflite(
                            spec,
                            calib_path,
                            quant_type,
                            output_path=tflite_path,
                            input_quant_dtype=io_dtype,
                            output_quant_dtype=io_dtype,
                        )
                        entry.update(converted)
                    else:
                        entry["status"] = "reuse"

                    metrics = evaluate_variant(name, "tflite", tflite_path, args.datasets, args.threads)
                    entry["metrics"] = metrics
                    entry["avg_macro_f1"] = metrics["avg_macro_f1"]
                    entry["status"] = "ok"
                except Exception as exc:
                    entry["status"] = "error"
                    entry["error"] = str(exc)
                    print(f"[ERROR] {name}: {exc}")
                candidates.append(entry)

    valid = [entry for entry in candidates if entry.get("status") == "ok"]
    valid.sort(key=lambda item: item["avg_macro_f1"], reverse=True)

    best: dict | None = valid[0] if valid else None
    baseline_onnx = next((item for item in baseline_results if item["variant"] == "current_onnx"), None)
    baseline_tflite = next((item for item in baseline_results if item["variant"] == "baseline_coral_tflite"), None)

    compare: dict = {
        "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": "mnv2_coral",
        "baselines": baseline_results,
        "candidates": candidates,
        "best_candidate": best,
    }

    if best is not None:
        best_metrics = best["metrics"]
        compare["best_delta"] = {}
        if baseline_onnx is not None:
            compare["best_delta"]["vs_current_onnx"] = {
                "avg_macro_f1": best_metrics["avg_macro_f1"] - baseline_onnx["avg_macro_f1"],
                "per_dataset": {
                    ds: best_metrics["datasets"][ds]["macro_f1"] - baseline_onnx["datasets"][ds]["macro_f1"]
                    for ds in args.datasets
                },
            }
        if baseline_tflite is not None:
            compare["best_delta"]["vs_baseline_coral_tflite"] = {
                "avg_macro_f1": best_metrics["avg_macro_f1"] - baseline_tflite["avg_macro_f1"],
                "per_dataset": {
                    ds: best_metrics["datasets"][ds]["macro_f1"] - baseline_tflite["datasets"][ds]["macro_f1"]
                    for ds in args.datasets
                },
            }

        if args.compile_best:
            best_tflite = Path(best["tflite_path"])
            compile_out = EDGE_SWEEP_DIR / f"{best_tflite.stem}_edgetpu.tflite"
            try:
                compiled = compile_for_edgetpu(spec, best_tflite, output_path=compile_out)
                compare["best_candidate_compile"] = compiled
            except Exception as exc:
                compare["best_candidate_compile"] = {"status": "error", "error": str(exc)}

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path = OUT_DIR / f"mnv2_coral_quant_sweep_{ts}.json"
    out_path.write_text(json.dumps(compare, ensure_ascii=False, indent=2))
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
