#!/usr/bin/env python3
"""
SHIELD RPi5 Inference — ONNX / TFLite / Edge TPU
================================================

Raspberry Pi 5 단독 실행용 이미지 위조 탐지 스크립트.

지원 백엔드:
  - onnx    : Python 3.10+/onnxruntime (현재 권장 기본값)
  - tflite  : Python 3.9 + tflite-runtime (Coral 없이 CPU 실행)
  - edgetpu : Python 3.9 + tflite-runtime + libedgetpu (Coral USB, 정확도 재검증 중)

모델:
  - MNV2 Dynamic INT8 ONNX / Full INT8 TFLite
  - SpecM-v4 Dynamic INT8 ONNX / Full INT8 TFLite
  - TFLite/Edge TPU 경로는 검증된 tuned `mnv2` 후보가 있으면 먼저 사용
    (없으면 `*_coral` 산출물 fallback)

실행 예시:
  python inference_rpi5.py image.jpg
  python inference_rpi5.py image.jpg --backend onnx
  python inference_rpi5.py image.jpg --backend tflite
  python inference_rpi5.py image.jpg --backend edgetpu --json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
from PIL import Image

# ── 설정 ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
ONNX_Q = ROOT / "weights" / "onnx_quant"
TFLITE_Q = ROOT / "weights" / "tflite"
TFLITE_EDGE = ROOT / "weights" / "tflite_edgetpu"
TFLITE_SWEEP = ROOT / "weights" / "tflite_sweep"
TFLITE_EDGE_SWEEP = ROOT / "weights" / "tflite_edgetpu_sweep"

CLASSES_3 = ["authentic", "manipulated", "ai_generated"]
CLASSES_2 = ["authentic", "manipulated"]

MODEL_CANDIDATES = {
    "onnx": {
        "mnv2": [ONNX_Q / "mnv2_int8_dynamic.onnx"],
        "specm": [ONNX_Q / "specm_v4_int8_dynamic.onnx"],
    },
    "tflite": {
        "mnv2": [
            TFLITE_SWEEP / "mnv2_coral_qsweep_qtpc_cal064_ioint8.tflite",
            TFLITE_Q / "mnv2_coral_int8_full.tflite",
            TFLITE_Q / "mnv2_int8_full.tflite",
        ],
        "specm": [
            TFLITE_Q / "specm_v4_coral_ft_int8_full.tflite",
            TFLITE_Q / "specm_v4_coral_int8_full.tflite",
            TFLITE_Q / "specm_v4_int8_full.tflite",
        ],
    },
    "edgetpu": {
        "mnv2": [
            TFLITE_EDGE_SWEEP / "mnv2_coral_qsweep_qtpc_cal064_ioint8_edgetpu.tflite",
            TFLITE_EDGE / "mnv2_coral_int8_full_edgetpu.tflite",
            TFLITE_EDGE / "mnv2_int8_full_edgetpu.tflite",
        ],
        "specm": [
            TFLITE_EDGE / "specm_v4_coral_ft_int8_full_edgetpu.tflite",
            TFLITE_EDGE / "specm_v4_coral_int8_full_edgetpu.tflite",
            TFLITE_EDGE / "specm_v4_int8_full_edgetpu.tflite",
        ],
    },
}


# ── 전처리 ────────────────────────────────────────────────────────────────
def load_image(path: Path, size: int = 224) -> np.ndarray:
    """이미지를 [0,1] float32 HWC 텐서로 로드."""
    img = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


def as_batched_layout(x_hwc: np.ndarray, expected_shape: Iterable[Any]) -> np.ndarray:
    """
    입력 텐서를 백엔드가 기대하는 레이아웃으로 정렬.

    - ONNX: 대개 NCHW
    - TFLite: 대개 NHWC
    """
    shape = [int(v) if isinstance(v, (int, np.integer)) else v for v in expected_shape]
    if len(shape) != 4:
        raise ValueError(f"지원하지 않는 입력 rank={len(shape)}")

    if shape[-1] in (1, 3):
        return x_hwc[np.newaxis, ...]
    if shape[1] in (1, 3):
        return x_hwc.transpose(2, 0, 1)[np.newaxis, ...]
    return x_hwc[np.newaxis, ...]


# ── 유틸리티 ──────────────────────────────────────────────────────────────
def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def quantize_array(x: np.ndarray, details: dict) -> np.ndarray:
    dtype = np.dtype(details["dtype"])
    if dtype == np.float32:
        return x.astype(np.float32)

    scale, zero_point = details.get("quantization", (0.0, 0))
    if not scale:
        return x.astype(dtype)

    q = np.round(x / scale + zero_point)
    bounds = np.iinfo(dtype)
    q = np.clip(q, bounds.min, bounds.max)
    return q.astype(dtype)


def dequantize_array(x: np.ndarray, details: dict) -> np.ndarray:
    dtype = np.dtype(details["dtype"])
    if dtype == np.float32:
        return x.astype(np.float32)

    scale, zero_point = details.get("quantization", (0.0, 0))
    if not scale:
        return x.astype(np.float32)
    return (x.astype(np.float32) - zero_point) * scale


def import_tflite_runtime():
    try:
        from tflite_runtime.interpreter import Interpreter, load_delegate
        return Interpreter, load_delegate
    except ImportError:
        try:
            import tensorflow as tf
        except ImportError as exc:
            raise RuntimeError(
                "TFLite 백엔드는 `tflite-runtime` 또는 `tensorflow`가 필요합니다. "
                "Coral 경로는 Python 3.9 venv를 권장합니다."
            ) from exc
        return tf.lite.Interpreter, tf.lite.experimental.load_delegate


def can_use_tflite() -> bool:
    try:
        import_tflite_runtime()
        return True
    except RuntimeError:
        return False


def auto_backend_candidates() -> list[str]:
    # Coral 경로는 compile 성공까지는 확보됐지만 정확도 재검증 중이므로,
    # auto는 검증된 ONNX를 우선 사용한다. Coral은 explicit backend로 선택.
    candidates: list[str] = ["onnx"]
    if can_use_tflite():
        if all(resolve_default_model_path("edgetpu", k).exists() for k in ("mnv2", "specm")):
            candidates.append("edgetpu")
        if all(resolve_default_model_path("tflite", k).exists() for k in ("mnv2", "specm")):
            candidates.append("tflite")
    return candidates


def resolve_default_model_path(backend: str, key: str) -> Path:
    candidates = MODEL_CANDIDATES[backend][key]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def resolve_model_path(backend: str, key: str, override: Optional[Path]) -> Path:
    return override if override is not None else resolve_default_model_path(backend, key)


def resolve_default_w_spec(backend: str, specm_path: Path) -> float:
    if backend in {"tflite", "edgetpu"} and "specm_v4_coral_ft" in specm_path.name:
        return 0.2
    return 1.0


# ── 세션 래퍼 ─────────────────────────────────────────────────────────────
class OnnxSession:
    def __init__(self, model_path: Path, threads: int):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        opts.log_severity_level = 3
        self.session = ort.InferenceSession(
            str(model_path), opts, providers=["CPUExecutionProvider"]
        )
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.input_shape = inp.shape

    def run(self, image_hwc: np.ndarray) -> np.ndarray:
        x = as_batched_layout(image_hwc, self.input_shape).astype(np.float32)
        return np.asarray(self.session.run(None, {self.input_name: x})[0][0], dtype=np.float32)


class TFLiteSession:
    def __init__(
        self,
        model_path: Path,
        threads: int,
        use_edgetpu: bool,
        delegate_path: str,
    ):
        Interpreter, load_delegate = import_tflite_runtime()
        kwargs: dict[str, Any] = {"model_path": str(model_path)}
        if use_edgetpu:
            kwargs["experimental_delegates"] = [load_delegate(delegate_path)]
        else:
            kwargs["num_threads"] = threads

        self.interpreter = Interpreter(**kwargs)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()[0]

    def run(self, image_hwc: np.ndarray) -> np.ndarray:
        x = as_batched_layout(image_hwc, self.input_details["shape"]).astype(np.float32)
        x = quantize_array(x, self.input_details)
        self.interpreter.set_tensor(self.input_details["index"], x)
        self.interpreter.invoke()
        out = self.interpreter.get_tensor(self.output_details["index"])[0]
        return dequantize_array(out, self.output_details)


# ── ICWMV 융합 ────────────────────────────────────────────────────────────
def icwmv_fuse(
    mnv2_probs: np.ndarray,   # shape (3,): [auth, manip, aigen]
    specm_probs: np.ndarray,  # shape (2,): [auth, manip]
    w_spec: float = 1.0,
) -> tuple[str, np.ndarray]:
    """
    MNV2(3-class) + SpecM(binary) class-wise weighted average.

    auth  = (mnv2[auth]  + w*specm[auth])  / (1+w)
    manip = (mnv2[manip] + w*specm[manip]) / (1+w)
    aigen = mnv2[aigen]
    """
    s_auth = (mnv2_probs[0] + w_spec * specm_probs[0]) / (1.0 + w_spec)
    s_manip = (mnv2_probs[1] + w_spec * specm_probs[1]) / (1.0 + w_spec)
    s_aigen = mnv2_probs[2]

    raw = np.array([s_auth, s_manip, s_aigen], dtype=np.float32)
    probs = raw / raw.sum()
    label = CLASSES_3[int(np.argmax(probs))]
    return label, probs


# ── 추론 엔진 ─────────────────────────────────────────────────────────────
class ShieldInference:
    def __init__(
        self,
        backend: str,
        mnv2_path: Path,
        specm_path: Path,
        threads: int = 2,
        w_spec: float = 1.0,
        delegate_path: str = "libedgetpu.so.1",
    ):
        self.backend = backend
        self.w_spec = w_spec

        t0 = time.perf_counter()
        if backend == "onnx":
            self.mnv2_sess = OnnxSession(mnv2_path, threads)
            self.specm_sess = OnnxSession(specm_path, threads)
        elif backend in {"tflite", "edgetpu"}:
            use_edgetpu = backend == "edgetpu"
            self.mnv2_sess = TFLiteSession(
                mnv2_path, threads=threads, use_edgetpu=use_edgetpu, delegate_path=delegate_path
            )
            self.specm_sess = TFLiteSession(
                specm_path, threads=threads, use_edgetpu=use_edgetpu, delegate_path=delegate_path
            )
        else:
            raise ValueError(f"지원하지 않는 backend={backend}")
        self.load_ms = (time.perf_counter() - t0) * 1000

    def predict(self, image_path: Path) -> dict:
        x = load_image(image_path)

        t0 = time.perf_counter()
        mnv2_logits = self.mnv2_sess.run(x)
        specm_logits = self.specm_sess.run(x)
        infer_ms = (time.perf_counter() - t0) * 1000

        mnv2_probs = softmax(mnv2_logits)
        specm_probs = softmax(specm_logits)
        verdict, final_probs = icwmv_fuse(mnv2_probs, specm_probs, self.w_spec)

        return {
            "backend": self.backend,
            "w_spec": float(self.w_spec),
            "verdict": verdict,
            "confidence": float(np.max(final_probs)),
            "scores": {c: round(float(p), 4) for c, p in zip(CLASSES_3, final_probs)},
            "mnv2_scores": {c: round(float(p), 4) for c, p in zip(CLASSES_3, mnv2_probs)},
            "specm_scores": {c: round(float(p), 4) for c, p in zip(CLASSES_2, specm_probs)},
            "latency_ms": round(infer_ms, 1),
            "load_ms": round(self.load_ms, 1),
        }


# ── CLI ───────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SHIELD RPi5 이미지 위조 탐지 (ONNX / TFLite / Edge TPU)",
    )
    parser.add_argument("image", type=Path, help="분석할 이미지 경로")
    parser.add_argument(
        "--backend",
        choices=["auto", "onnx", "tflite", "edgetpu"],
        default="auto",
        help="추론 백엔드 선택 (default: auto)",
    )
    parser.add_argument("--mnv2", type=Path, default=None, help="MNV2 모델 경로 override")
    parser.add_argument("--specm", type=Path, default=None, help="SpecM 모델 경로 override")
    parser.add_argument(
        "--threads",
        type=int,
        default=2,
        help="CPU 백엔드 스레드 수 (onnx/tflite only, default: 2)",
    )
    parser.add_argument(
        "--w-spec",
        type=float,
        default=None,
        help="SpecM 가중치 override (기본: onnx=1.0, coral-ft tflite/edgetpu=0.2)",
    )
    parser.add_argument(
        "--delegate-path",
        type=str,
        default="libedgetpu.so.1",
        help="Edge TPU delegate 경로 (default: libedgetpu.so.1)",
    )
    parser.add_argument("--json", action="store_true", help="JSON 형식으로 출력")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if not args.image.exists():
        print(f"오류: 이미지 파일을 찾을 수 없습니다 — {args.image}", file=sys.stderr)
        raise SystemExit(1)

    backends = auto_backend_candidates() if args.backend == "auto" else [args.backend]
    inferer: Optional[ShieldInference] = None
    last_error = ""

    for backend in backends:
        mnv2_path = resolve_model_path(backend, "mnv2", args.mnv2)
        specm_path = resolve_model_path(backend, "specm", args.specm)

        missing = [(label, path) for label, path in [("MNV2", mnv2_path), ("SpecM", specm_path)] if not path.exists()]
        if missing:
            detail = ", ".join(f"{label}={path}" for label, path in missing)
            last_error = f"{backend} 모델 없음: {detail}"
            if args.backend == "auto":
                continue
            print(f"오류: {last_error}", file=sys.stderr)
            raise SystemExit(1)

        try:
            w_spec = args.w_spec if args.w_spec is not None else resolve_default_w_spec(backend, specm_path)
            inferer = ShieldInference(
                backend=backend,
                mnv2_path=mnv2_path,
                specm_path=specm_path,
                threads=args.threads,
                w_spec=w_spec,
                delegate_path=args.delegate_path,
            )
            break
        except Exception as exc:
            last_error = f"{backend} 초기화 실패: {exc}"
            if args.backend != "auto":
                print(f"오류: {last_error}", file=sys.stderr)
                raise SystemExit(1)

    if inferer is None:
        print(f"오류: 사용 가능한 백엔드를 찾지 못했습니다 — {last_error}", file=sys.stderr)
        raise SystemExit(1)

    result = inferer.predict(args.image)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    scores = result["scores"]
    mnv2_scores = result["mnv2_scores"]
    specm_scores = result["specm_scores"]
    verdict = result["verdict"]
    verdict_ko = {
        "authentic": "정품",
        "manipulated": "조작",
        "ai_generated": "AI생성",
    }
    print(f"백엔드: {result['backend']}")
    print(f"w_spec: {result['w_spec']:.2f}")
    print(f"판정: {verdict_ko.get(verdict, verdict)} ({verdict}, {result['confidence'] * 100:.1f}%)")
    print(
        f"  authentic={scores['authentic']:.3f}  "
        f"manipulated={scores['manipulated']:.3f}  "
        f"ai_generated={scores['ai_generated']:.3f}"
    )
    print(
        f"MNV2: authentic={mnv2_scores['authentic']:.3f}  "
        f"manipulated={mnv2_scores['manipulated']:.3f}  "
        f"ai_generated={mnv2_scores['ai_generated']:.3f}"
    )
    print(
        f"SpecM: authentic={specm_scores['authentic']:.3f}  "
        f"manipulated={specm_scores['manipulated']:.3f}"
    )
    print(f"추론: {result['latency_ms']:.0f}ms  |  모델로드: {result['load_ms']:.0f}ms")


if __name__ == "__main__":
    main()
