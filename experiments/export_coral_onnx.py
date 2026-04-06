#!/usr/bin/env python3
"""
Coral-friendly ONNX export helper.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from coral_export_models import export_to_onnx
from coral_export_models import load_mnv2_coral, load_specm_v4_coral, load_specm_v4_coral_ft

ROOT = Path(__file__).resolve().parents[1]
ONNX_DIR = ROOT / "weights" / "onnx"

MODEL_SPECS = {
    "mnv2_coral": {
        "ckpt": ROOT / "weights" / "mobilenetv2_dualstream" / "mobilenetv2_dualstream_best.pth",
        "onnx": ONNX_DIR / "mnv2_coral.onnx",
        "input_name": "image_01",
        "loader": load_mnv2_coral,
        "image_size": 224,
    },
    "specm_v4_coral": {
        "ckpt": ROOT / "weights" / "specialist_m_v4" / "specialist_m_v4_best.pth",
        "onnx": ONNX_DIR / "specm_v4_coral.onnx",
        "input_name": "image_01",
        "loader": load_specm_v4_coral,
        "image_size": 224,
    },
    "specm_v4_coral_ft": {
        "ckpt": ROOT / "weights" / "specialist_m_v4_coral" / "specialist_m_v4_coral_best.pth",
        "onnx": ONNX_DIR / "specm_v4_coral_ft.onnx",
        "input_name": "image_01",
        "loader": load_specm_v4_coral_ft,
        "image_size": 224,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Coral-friendly ONNX models")
    parser.add_argument("--models", nargs="+", default=list(MODEL_SPECS))
    parser.add_argument("--force", action="store_true", help="기존 ONNX 덮어쓰기")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for key in args.models:
        if key not in MODEL_SPECS:
            raise KeyError(f"알 수 없는 model key: {key}")
        spec = MODEL_SPECS[key]
        out_path = spec["onnx"]
        if out_path.exists() and not args.force:
            print(f"[SKIP] {out_path.name} already exists")
            continue
        model = spec["loader"](spec["ckpt"])
        export_to_onnx(model, out_path, spec["input_name"], image_size=spec["image_size"])
        print(f"[OK] {out_path}")


if __name__ == "__main__":
    main()
