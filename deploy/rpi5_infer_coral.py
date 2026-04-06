#!/usr/bin/env python3
"""
SHIELD RPi5 Coral Inference — Edge TPU (Coral USB Accelerator)
==============================================================

MNV2 + SpecM-v4 Edge TPU 컴파일 TFLite 모델로 이미지 위조 탐지를
수행하는 Raspberry Pi 5 + Coral USB Accelerator 전용 독립 스크립트.

⚠ Python 3.9 필수 (tflite-runtime은 3.10+ 미지원)
  python3.9 -m venv .venv-coral
  source .venv-coral/bin/activate
  pip install tflite-runtime Pillow numpy

Coral libedgetpu 설치:
  curl -fsSL https://coral.ai/software/repo/coral-repo.list | sudo tee ...
  sudo apt install libedgetpu1-std
  (또는 setup_rpi5_coral_env.sh 참조)

모델 파일 (서버에서 RPi5로 복사):
  weights/tflite_edgetpu/mnv2_coral_int8_full_edgetpu.tflite
  weights/tflite_edgetpu/specm_v4_coral_ft_int8_full_edgetpu.tflite

  # 또는 sweep-tuned (정확도 우선):
  weights/tflite_edgetpu_sweep/mnv2_coral_qsweep_qtpc_cal064_ioint8_edgetpu.tflite

실행 예시:
  python rpi5_infer_coral.py image.jpg
  python rpi5_infer_coral.py image.jpg --json
  python rpi5_infer_coral.py image.jpg --delegate-path /usr/lib/aarch64-linux-gnu/libedgetpu.so.1
  python rpi5_infer_coral.py image.jpg --mnv2 /path/to/mnv2_edgetpu.tflite
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

# ── 기본 모델 경로 ─────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent  # MAIFS 루트 (있을 경우)

DEFAULT_EDGETPU_DIR      = ROOT / "weights" / "tflite_edgetpu"
DEFAULT_EDGETPU_SWEEP    = ROOT / "weights" / "tflite_edgetpu_sweep"

# MNV2: sweep-tuned 우선, 없으면 기본 coral
DEFAULT_MNV2_CANDIDATES = [
    DEFAULT_EDGETPU_SWEEP / "mnv2_coral_qsweep_qtpc_cal064_ioint8_edgetpu.tflite",
    DEFAULT_EDGETPU_DIR   / "mnv2_coral_int8_full_edgetpu.tflite",
]
# SpecM: coral fine-tuned 우선
DEFAULT_SPECM_CANDIDATES = [
    DEFAULT_EDGETPU_DIR / "specm_v4_coral_ft_int8_full_edgetpu.tflite",
    DEFAULT_EDGETPU_DIR / "specm_v4_coral_int8_full_edgetpu.tflite",
]

CLASSES_3 = ["authentic", "manipulated", "ai_generated"]
CLASSES_2 = ["authentic", "manipulated"]
VERDICT_KO = {"authentic": "정품", "manipulated": "조작", "ai_generated": "AI생성"}

# coral-ft SpecM은 MNV2 대비 약해서 가중치를 낮추는 것이 기본값
DEFAULT_W_SPEC_CORAL_FT = 0.2
DEFAULT_W_SPEC_STD      = 1.0


def resolve_first(candidates: list[Path]) -> Path:
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


# ── 전처리 ────────────────────────────────────────────────────────────────
def load_image(path: Path, size: int = 224) -> np.ndarray:
    """이미지를 [0,1] float32 HWC 텐서로 로드."""
    img = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


# ── 유틸리티 ──────────────────────────────────────────────────────────────
def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def quantize(x: np.ndarray, details: dict) -> np.ndarray:
    dtype = np.dtype(details["dtype"])
    if dtype == np.float32:
        return x.astype(np.float32)
    scale, zero_point = details.get("quantization", (0.0, 0))
    if not scale:
        return x.astype(dtype)
    q = np.clip(np.round(x / scale + zero_point), np.iinfo(dtype).min, np.iinfo(dtype).max)
    return q.astype(dtype)


def dequantize(x: np.ndarray, details: dict) -> np.ndarray:
    dtype = np.dtype(details["dtype"])
    if dtype == np.float32:
        return x.astype(np.float32)
    scale, zero_point = details.get("quantization", (0.0, 0))
    if not scale:
        return x.astype(np.float32)
    return (x.astype(np.float32) - zero_point) * scale


# ── TFLite 런타임 로드 ─────────────────────────────────────────────────────
def load_tflite_runtime():
    """tflite-runtime 우선, 없으면 tensorflow 폴백."""
    try:
        from tflite_runtime.interpreter import Interpreter, load_delegate
        return Interpreter, load_delegate
    except ImportError:
        try:
            import tensorflow as tf
            return tf.lite.Interpreter, tf.lite.experimental.load_delegate
        except ImportError:
            raise RuntimeError(
                "tflite-runtime 미설치.\n"
                "  pip install tflite-runtime  (Python 3.9 venv 권장)\n"
                "  또는 pip install tensorflow"
            )


# ── Edge TPU 세션 ─────────────────────────────────────────────────────────
class CoralModel:
    def __init__(self, model_path: Path, delegate_path: str = "libedgetpu.so.1"):
        Interpreter, load_delegate = load_tflite_runtime()

        try:
            delegate = load_delegate(delegate_path)
            kwargs: dict[str, Any] = {
                "model_path": str(model_path),
                "experimental_delegates": [delegate],
            }
        except ValueError as exc:
            raise RuntimeError(
                f"Edge TPU delegate 로드 실패 ({delegate_path}): {exc}\n"
                "  libedgetpu 미설치 또는 Coral USB가 연결되지 않았을 수 있습니다."
            ) from exc

        self.interpreter = Interpreter(**kwargs)
        self.interpreter.allocate_tensors()
        self._in  = self.interpreter.get_input_details()[0]
        self._out = self.interpreter.get_output_details()[0]

    def infer(self, image_hwc: np.ndarray) -> np.ndarray:
        # TFLite는 NHWC
        x = image_hwc[np.newaxis, ...]
        x = quantize(x, self._in)
        self.interpreter.set_tensor(self._in["index"], x)
        self.interpreter.invoke()
        out = self.interpreter.get_tensor(self._out["index"])[0]
        return dequantize(out, self._out)


# ── ICWMV 융합 ────────────────────────────────────────────────────────────
def icwmv_fuse(
    mnv2_probs: np.ndarray,
    specm_probs: np.ndarray,
    w_spec: float = 0.2,
) -> tuple[str, np.ndarray]:
    """
    auth  = (mnv2[auth]  + w * specm[auth])  / (1 + w)
    manip = (mnv2[manip] + w * specm[manip]) / (1 + w)
    aigen = mnv2[aigen]
    """
    s_auth  = (mnv2_probs[0] + w_spec * specm_probs[0]) / (1.0 + w_spec)
    s_manip = (mnv2_probs[1] + w_spec * specm_probs[1]) / (1.0 + w_spec)
    s_aigen = mnv2_probs[2]

    raw   = np.array([s_auth, s_manip, s_aigen], dtype=np.float32)
    probs = raw / raw.sum()
    label = CLASSES_3[int(np.argmax(probs))]
    return label, probs


# ── 메인 추론 클래스 ──────────────────────────────────────────────────────
class ShieldCoral:
    def __init__(
        self,
        mnv2_path: Path,
        specm_path: Path,
        delegate_path: str = "libedgetpu.so.1",
    ):
        t0 = time.perf_counter()
        self.mnv2  = CoralModel(mnv2_path, delegate_path)
        t1 = time.perf_counter()
        self.specm = CoralModel(specm_path, delegate_path)
        t2 = time.perf_counter()
        self.load_ms = {
            "mnv2_load_ms":  round((t1 - t0) * 1000, 1),
            "specm_load_ms": round((t2 - t1) * 1000, 1),
            "total_load_ms": round((t2 - t0) * 1000, 1),
        }

    def predict(self, image_path: Path, w_spec: float = DEFAULT_W_SPEC_CORAL_FT) -> dict:
        x = load_image(image_path)

        # 단계별 타이밍 (Edge TPU invoke 포함)
        t0 = time.perf_counter()
        mnv2_logits = self.mnv2.infer(x)
        t1 = time.perf_counter()
        specm_logits = self.specm.infer(x)
        t2 = time.perf_counter()

        mnv2_ms  = round((t1 - t0) * 1000, 1)
        specm_ms = round((t2 - t1) * 1000, 1)
        total_ms = round((t2 - t0) * 1000, 1)

        mnv2_probs  = softmax(mnv2_logits)
        specm_probs = softmax(specm_logits)
        verdict, final_probs = icwmv_fuse(mnv2_probs, specm_probs, w_spec)

        return {
            "backend":    "edgetpu_coral",
            "w_spec":     float(w_spec),
            "verdict":    verdict,
            "confidence": float(np.max(final_probs)),
            "scores":     {c: round(float(p), 4) for c, p in zip(CLASSES_3, final_probs)},
            "mnv2_scores":  {c: round(float(p), 4) for c, p in zip(CLASSES_3, mnv2_probs)},
            "specm_scores": {c: round(float(p), 4) for c, p in zip(CLASSES_2, specm_probs)},
            "latency": {
                "mnv2_ms":  mnv2_ms,
                "specm_ms": specm_ms,
                "total_ms": total_ms,
            },
            "load": self.load_ms,
        }


# ── CLI ───────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    default_mnv2  = resolve_first(DEFAULT_MNV2_CANDIDATES)
    default_specm = resolve_first(DEFAULT_SPECM_CANDIDATES)

    p = argparse.ArgumentParser(
        description="SHIELD RPi5 Coral 추론 (Edge TPU + Coral USB Accelerator)"
    )
    p.add_argument("image", type=Path, help="분석할 이미지 경로")
    p.add_argument("--mnv2", type=Path, default=default_mnv2,
                   help=f"MNV2 Edge TPU TFLite 경로 (기본: {default_mnv2.name})")
    p.add_argument("--specm", type=Path, default=default_specm,
                   help=f"SpecM Edge TPU TFLite 경로 (기본: {default_specm.name})")
    p.add_argument("--delegate-path", type=str, default="libedgetpu.so.1",
                   help="Edge TPU delegate .so 경로 (기본: libedgetpu.so.1)")
    p.add_argument("--w-spec", type=float, default=None,
                   help=f"SpecM 융합 가중치 (기본: coral-ft={DEFAULT_W_SPEC_CORAL_FT}, 표준={DEFAULT_W_SPEC_STD})")
    p.add_argument("--json", action="store_true", help="JSON 형식으로 출력")
    return p


def main() -> None:
    args = build_parser().parse_args()

    if not args.image.exists():
        print(f"오류: 이미지를 찾을 수 없습니다 — {args.image}", file=sys.stderr)
        raise SystemExit(1)

    for label, path in [("MNV2", args.mnv2), ("SpecM", args.specm)]:
        if not path.exists():
            print(f"오류: {label} 모델 없음 — {path}", file=sys.stderr)
            print("  서버에서 weights/tflite_edgetpu/ 디렉토리를 복사하세요.", file=sys.stderr)
            raise SystemExit(1)

    # w_spec 자동 결정
    if args.w_spec is not None:
        w_spec = args.w_spec
    elif "coral_ft" in args.specm.name:
        w_spec = DEFAULT_W_SPEC_CORAL_FT
    else:
        w_spec = DEFAULT_W_SPEC_STD

    try:
        model = ShieldCoral(args.mnv2, args.specm, delegate_path=args.delegate_path)
    except RuntimeError as exc:
        print(f"오류: {exc}", file=sys.stderr)
        raise SystemExit(1)

    result = model.predict(args.image, w_spec=w_spec)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    lat   = result["latency"]
    load  = result["load"]
    v     = result["verdict"]
    sc    = result["scores"]
    m     = result["mnv2_scores"]
    s     = result["specm_scores"]

    print(f"백엔드 : {result['backend']}  w_spec={result['w_spec']:.2f}")
    print(f"판정   : {VERDICT_KO.get(v, v)} ({v}, {result['confidence']*100:.1f}%)")
    print(f"  auth={sc['authentic']:.3f}  manip={sc['manipulated']:.3f}  aigen={sc['ai_generated']:.3f}")
    print(f"MNV2   : auth={m['authentic']:.3f}  manip={m['manipulated']:.3f}  aigen={m['ai_generated']:.3f}")
    print(f"SpecM  : auth={s['authentic']:.3f}  manip={s['manipulated']:.3f}")
    print(f"레이턴시: MNV2={lat['mnv2_ms']}ms  SpecM={lat['specm_ms']}ms  합계={lat['total_ms']}ms")
    print(f"모델로드: MNV2={load['mnv2_load_ms']}ms  SpecM={load['specm_load_ms']}ms  합계={load['total_load_ms']}ms")


if __name__ == "__main__":
    main()
