#!/usr/bin/env python3
"""
Phase 4 — ONNX -> Full INT8 TFLite -> Edge TPU 컴파일
======================================================

목적:
  1. ONNX 모델을 Coral Edge TPU용 full integer TFLite로 변환
  2. edgetpu_compiler로 `_edgetpu.tflite` 생성
  3. RPi5/Coral 배포용 산출물을 안정적인 파일명으로 저장

출력:
  weights/tflite/{model}_int8_full.tflite
  weights/tflite_edgetpu/{model}_int8_full_edgetpu.tflite
  experiments/results/edgetpu_export/edgetpu_export_{ts}.json

실행:
  python experiments/run_edgetpu_export.py
  python experiments/run_edgetpu_export.py --models mnv2_coral specm_v4_coral
  python experiments/run_edgetpu_export.py --skip-compile
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
ONNX_DIR = ROOT / "weights" / "onnx"
TFLITE_DIR = ROOT / "weights" / "tflite"
EDGE_DIR = ROOT / "weights" / "tflite_edgetpu"
CALIB_DIR = ROOT / "weights" / "tflite_calibration"
OUT_DIR = ROOT / "experiments" / "results" / "edgetpu_export"
REPLACEMENT_DIR = OUT_DIR / "replacement_json"

for path in (TFLITE_DIR, EDGE_DIR, CALIB_DIR, OUT_DIR, REPLACEMENT_DIR):
    path.mkdir(parents=True, exist_ok=True)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass(frozen=True)
class ModelSpec:
    key: str
    onnx_path: Path
    ckpt_path: Path
    input_name: str
    image_size: int
    output_stem: str
    mean: str
    std: str


MODEL_SPECS = {
    "mnv2_coral": ModelSpec(
        key="mnv2_coral",
        onnx_path=ONNX_DIR / "mnv2_coral.onnx",
        ckpt_path=ROOT / "weights" / "mobilenetv2_dualstream" / "mobilenetv2_dualstream_best.pth",
        input_name="image_01",
        image_size=224,
        output_stem="mnv2_coral_int8_full",
        mean="[[[[0.0, 0.0, 0.0]]]]",
        std="[[[[1.0, 1.0, 1.0]]]]",
    ),
    "specm_v4_coral": ModelSpec(
        key="specm_v4_coral",
        onnx_path=ONNX_DIR / "specm_v4_coral.onnx",
        ckpt_path=ROOT / "weights" / "specialist_m_v4" / "specialist_m_v4_best.pth",
        input_name="image_01",
        image_size=224,
        output_stem="specm_v4_coral_int8_full",
        mean="[[[[0.0, 0.0, 0.0]]]]",
        std="[[[[1.0, 1.0, 1.0]]]]",
    ),
    "specm_v4_coral_ft": ModelSpec(
        key="specm_v4_coral_ft",
        onnx_path=ONNX_DIR / "specm_v4_coral_ft.onnx",
        ckpt_path=ROOT / "weights" / "specialist_m_v4_coral" / "specialist_m_v4_coral_best.pth",
        input_name="image_01",
        image_size=224,
        output_stem="specm_v4_coral_ft_int8_full",
        mean="[[[[0.0, 0.0, 0.0]]]]",
        std="[[[[1.0, 1.0, 1.0]]]]",
    ),
}

ALIASES = {
    "mnv2": "mnv2_coral",
    "specm": "specm_v4_coral",
    "specm_v4": "specm_v4_coral",
    "specm_v4_coral_best": "specm_v4_coral_ft",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ONNX -> Full INT8 TFLite -> Edge TPU compile",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["mnv2_coral", "specm_v4_coral"],
        help="변환할 모델 키 (default: mnv2_coral specm_v4_coral)",
    )
    parser.add_argument(
        "--calib-images",
        type=int,
        default=128,
        help="INT8 calibration 이미지 수 (default: 128)",
    )
    parser.add_argument(
        "--quant-type",
        choices=["per-channel", "per-tensor"],
        default="per-channel",
        help="onnx2tf quantization mode (default: per-channel)",
    )
    parser.add_argument(
        "--input-quant-dtype",
        choices=["int8", "uint8", "float32"],
        default="int8",
        help="onnx2tf input quant dtype (default: int8)",
    )
    parser.add_argument(
        "--output-quant-dtype",
        choices=["int8", "uint8", "float32"],
        default="int8",
        help="onnx2tf output quant dtype (default: int8)",
    )
    parser.add_argument(
        "--refresh-onnx",
        action="store_true",
        help="Coral-friendly ONNX를 체크포인트에서 다시 생성",
    )
    parser.add_argument("--skip-convert", action="store_true", help="TFLite 변환 생략")
    parser.add_argument("--skip-compile", action="store_true", help="Edge TPU 컴파일 생략")
    return parser.parse_args()


def resolve_model_keys(raw_keys: Iterable[str]) -> List[str]:
    resolved = []
    for key in raw_keys:
        normalized = ALIASES.get(key, key)
        if normalized not in MODEL_SPECS:
            raise KeyError(f"알 수 없는 model key: {key}")
        resolved.append(normalized)
    return resolved


def image_paths_in(directory: Path) -> List[Path]:
    if not directory.exists():
        return []
    return sorted(
        p for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def collect_calibration_images(limit: int) -> List[Path]:
    source_dirs = [
        ROOT / "datasets" / "CASIA2_subset" / "Au",
        ROOT / "datasets" / "CASIA2_subset" / "Tp",
        ROOT / "datasets" / "OpenSDID_subset" / "authentic",
        ROOT / "datasets" / "OpenSDID_subset" / "manipulated",
        ROOT / "datasets" / "OpenSDID_subset" / "ai_generated",
        ROOT / "datasets" / "GenImage_subset" / "BigGAN" / "val" / "nature",
        ROOT / "datasets" / "GenImage_subset" / "BigGAN" / "val" / "ai",
        ROOT / "datasets" / "AI-GenBench_proxy" / "authentic",
        ROOT / "datasets" / "AI-GenBench_proxy" / "manipulated",
        ROOT / "datasets" / "AI-GenBench_proxy" / "ai_generated",
    ]
    buckets = [image_paths_in(src) for src in source_dirs]
    buckets = [bucket for bucket in buckets if bucket]

    selected: List[Path] = []
    if buckets:
        offset = 0
        while len(selected) < limit:
            progressed = False
            for bucket in buckets:
                if offset < len(bucket):
                    selected.append(bucket[offset])
                    progressed = True
                    if len(selected) >= limit:
                        break
            if not progressed:
                break
            offset += 1

    if len(selected) < limit:
        fallback = image_paths_in(ROOT / "datasets")
        seen = {p.resolve() for p in selected}
        for path in fallback:
            real = path.resolve()
            if real in seen:
                continue
            selected.append(path)
            seen.add(real)
            if len(selected) >= limit:
                break

    if not selected:
        raise RuntimeError("Calibration 이미지를 찾지 못했습니다. datasets/ 구성을 확인하세요.")
    return selected[:limit]


def build_calibration_npy(spec: ModelSpec, images: List[Path]) -> Path:
    out_path = CALIB_DIR / f"{spec.output_stem}_{len(images)}.npy"
    if out_path.exists():
        return out_path

    tensors = []
    for path in images:
        img = Image.open(path).convert("RGB").resize((spec.image_size, spec.image_size), Image.BILINEAR)
        x = np.asarray(img, dtype=np.float32) / 255.0
        tensors.append(x)

    calib = np.stack(tensors, axis=0).astype(np.float32)  # NHWC
    np.save(out_path, calib)
    return out_path


def resolve_onnx2tf_cmd() -> List[str]:
    cmd = shutil.which("onnx2tf")
    if cmd:
        return [cmd]
    return [sys.executable, "-m", "onnx2tf"]


def resolve_compiler_cmd() -> str:
    cmd = shutil.which("edgetpu_compiler")
    if not cmd:
        raise RuntimeError("`edgetpu_compiler`를 찾지 못했습니다. 서버에 컴파일러를 설치하세요.")
    return cmd


def find_generated_tflite(directory: Path) -> Path:
    patterns = [
        "*full_integer_quant.tflite",
        "*integer_quant.tflite",
        "*full_integer_quant_with_int16_act.tflite",
        "*integer_quant_with_int16_act.tflite",
        "*int8*.tflite",
        "*.tflite",
    ]
    for pattern in patterns:
        matches = sorted(directory.rglob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        if matches:
            return matches[0]
    raise RuntimeError(f"TFLite 산출물을 찾지 못했습니다: {directory}")


def validate_tflite_io_dtype(tflite_path: Path) -> dict:
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    input_dtype = np.dtype(inp["dtype"]).name
    output_dtype = np.dtype(out["dtype"]).name
    allowed = {"int8", "uint8"}
    if input_dtype not in allowed or output_dtype not in allowed:
        raise RuntimeError(
            f"Edge TPU 비호환 TFLite dtype: input={input_dtype}, output={output_dtype} "
            f"(expected int8/uint8). 선택된 파일={tflite_path.name}"
        )
    return {
        "input_dtype": input_dtype,
        "output_dtype": output_dtype,
        "input_quant": list(inp.get("quantization", ())),
        "output_quant": list(out.get("quantization", ())),
    }


def python_has_modules(python_exe: str, modules: list[str]) -> bool:
    code = "import " + ", ".join(modules)
    proc = subprocess.run(
        [python_exe, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode == 0


def resolve_export_python() -> str:
    candidates = [
        sys.executable,
        str(ROOT / ".venv-qwen" / "bin" / "python"),
        shutil.which("python3") or "",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        if python_has_modules(candidate, ["torch", "torchvision", "onnx"]):
            return candidate
    raise RuntimeError(
        "Coral-friendly ONNX export용 Python을 찾지 못했습니다. "
        "`torch`, `torchvision`, `onnx`가 있는 환경이 필요합니다."
    )


def export_coral_onnx(spec: ModelSpec, force: bool = False) -> Path:
    if spec.onnx_path.exists() and not force:
        return spec.onnx_path
    if not spec.ckpt_path.exists():
        raise FileNotFoundError(f"체크포인트 없음: {spec.ckpt_path}")
    export_script = ROOT / "experiments" / "export_coral_onnx.py"
    export_python = resolve_export_python()
    cmd = [export_python, str(export_script), "--models", spec.key]
    if force:
        cmd.append("--force")
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Coral-friendly ONNX export 실패({spec.key})\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    return spec.onnx_path


def persist_replacement_json(spec: ModelSpec, stdout: str) -> str | None:
    match = re.search(r"Auto-generated replacement JSON saved to:\s*(\S+)", stdout)
    if not match:
        return None
    src = Path(match.group(1))
    if not src.exists():
        return None
    dst = REPLACEMENT_DIR / f"{spec.key}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    shutil.copy2(src, dst)
    return str(dst)


def convert_to_tflite(
    spec: ModelSpec,
    calib_path: Path,
    quant_type: str,
    output_path: Path | None = None,
    input_quant_dtype: str = "int8",
    output_quant_dtype: str = "int8",
) -> dict:
    if not spec.onnx_path.exists():
        raise FileNotFoundError(f"ONNX 파일 없음: {spec.onnx_path}")

    with tempfile.TemporaryDirectory(prefix=f"{spec.key}_onnx2tf_") as tmp_dir:
        out_dir = Path(tmp_dir) / "export"
        out_dir.mkdir(parents=True, exist_ok=True)

        cmd = resolve_onnx2tf_cmd() + [
            "-i", str(spec.onnx_path),
            "-o", str(out_dir),
            "-b", "1",
            "-oiqt",
            "-qt", quant_type,
            "-iqd", input_quant_dtype,
            "-oqd", output_quant_dtype,
            "-cind", spec.input_name, str(calib_path), spec.mean, spec.std,
        ]
        env = os.environ.copy()
        env["PATH"] = f"{(Path(sys.prefix) / 'bin').resolve()}:{env.get('PATH', '')}"
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)
        if proc.returncode != 0:
            replacement_json = persist_replacement_json(spec, proc.stdout)
            replacement_note = f"\nReplacement JSON: {replacement_json}" if replacement_json else ""
            raise RuntimeError(
                f"onnx2tf 실패({spec.key}){replacement_note}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
            )

        generated = find_generated_tflite(out_dir)
        final_tflite = output_path or (TFLITE_DIR / f"{spec.output_stem}.tflite")
        final_tflite.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(generated, final_tflite)
        dtype_info = validate_tflite_io_dtype(final_tflite)
        return {
            "tflite_path": str(final_tflite),
            "tflite_file_name": generated.name,
            **dtype_info,
            "stdout_tail": proc.stdout.strip().splitlines()[-20:],
            "stderr_tail": proc.stderr.strip().splitlines()[-20:],
        }


def compile_for_edgetpu(spec: ModelSpec, tflite_path: Path, output_path: Path | None = None) -> dict:
    compiler = resolve_compiler_cmd()
    with tempfile.TemporaryDirectory(prefix=f"{spec.key}_edgetpu_") as tmp_dir:
        out_dir = Path(tmp_dir)
        cmd = [compiler, "-s", "-o", str(out_dir), str(tflite_path)]
        env = os.environ.copy()
        env["PATH"] = f"{(Path(sys.prefix) / 'bin').resolve()}:{env.get('PATH', '')}"
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)
        if proc.returncode != 0:
            raise RuntimeError(
                f"edgetpu_compiler 실패({spec.key})\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
            )

        generated = out_dir / f"{tflite_path.stem}_edgetpu.tflite"
        if not generated.exists():
            generated = find_generated_tflite(out_dir)
        final_tpu = output_path or (EDGE_DIR / f"{spec.output_stem}_edgetpu.tflite")
        final_tpu.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(generated, final_tpu)
        return {
            "edgetpu_path": str(final_tpu),
            "stdout_tail": proc.stdout.strip().splitlines()[-20:],
            "stderr_tail": proc.stderr.strip().splitlines()[-20:],
        }


def main() -> None:
    args = parse_args()
    model_keys = resolve_model_keys(args.models)
    calib_images = collect_calibration_images(args.calib_images)

    results = {
        "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "models": {},
        "calibration_images": [str(p.relative_to(ROOT)) for p in calib_images[:16]],
        "calibration_count": len(calib_images),
        "quant_type": args.quant_type,
        "input_quant_dtype": args.input_quant_dtype,
        "output_quant_dtype": args.output_quant_dtype,
    }
    failures = 0

    for key in model_keys:
        spec = MODEL_SPECS[key]
        print(f"[{spec.key}] 변환 시작")
        entry = {
            "onnx_path": str(spec.onnx_path),
            "ckpt_path": str(spec.ckpt_path),
            "calibration_npy": "",
            "status": "pending",
        }
        try:
            onnx_path = export_coral_onnx(spec, force=args.refresh_onnx)
            entry["onnx_path"] = str(onnx_path)
            print(f"  ONNX 준비: {onnx_path.name}")
            calib_path = build_calibration_npy(spec, calib_images)
            entry["calibration_npy"] = str(calib_path)

            if not args.skip_convert:
                converted = convert_to_tflite(
                    spec,
                    calib_path,
                    args.quant_type,
                    input_quant_dtype=args.input_quant_dtype,
                    output_quant_dtype=args.output_quant_dtype,
                )
                entry.update(converted)
                print(f"  TFLite 저장: {converted['tflite_path']}")
            else:
                tflite_path = TFLITE_DIR / f"{spec.output_stem}.tflite"
                if not tflite_path.exists():
                    raise FileNotFoundError(f"--skip-convert 사용 불가: {tflite_path} 없음")
                entry["tflite_path"] = str(tflite_path)

            if not args.skip_compile:
                compiled = compile_for_edgetpu(spec, Path(entry["tflite_path"]))
                entry.update(compiled)
                print(f"  Edge TPU 저장: {compiled['edgetpu_path']}")

            entry["status"] = "ok"
        except Exception as exc:
            failures += 1
            entry["status"] = "error"
            entry["error"] = str(exc)
            print(f"  [ERROR] {exc}")
        results["models"][spec.key] = entry

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path = OUT_DIR / f"edgetpu_export_{ts}.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\n완료: {out_path}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
