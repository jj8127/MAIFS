#!/usr/bin/env python3
"""
SHIELD RPi5 CPU Inference — ONNX Runtime (Coral 없이)
=====================================================

MNV2(INT8 dynamic-quant) + SpecM-v4(INT8 dynamic-quant) ONNX 모델로
이미지 위조 탐지를 수행하는 Raspberry Pi 5 전용 독립 스크립트.

의존성:
  pip install onnxruntime Pillow numpy

모델 파일:
  Raspberry_pi5_Experiment/common/models/onnx_quant/mnv2_int8_dynamic.onnx
  Raspberry_pi5_Experiment/common/models/onnx_quant/specm_v4_int8_static.onnx

실행 예시:
  python rpi5_infer_cpu.py image.jpg
  python rpi5_infer_cpu.py image.jpg --threads 4
  python rpi5_infer_cpu.py image.jpg --json
  python rpi5_infer_cpu.py image.jpg --mnv2 /path/to/mnv2.onnx --specm /path/to/specm.onnx
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

# ── 기본 모델 경로 ─────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
COMMON_DIR = SCRIPT_DIR.parent
MODEL_DIR = COMMON_DIR / "models" / "onnx_quant"

DEFAULT_MNV2 = MODEL_DIR / "mnv2_int8_dynamic.onnx"
DEFAULT_SPECM = MODEL_DIR / "specm_v4_int8_static.onnx"

CLASSES_3 = ["authentic", "manipulated", "ai_generated"]
CLASSES_2 = ["authentic", "manipulated"]

VERDICT_KO = {"authentic": "정품", "manipulated": "조작", "ai_generated": "AI생성"}


# ── 전처리 ────────────────────────────────────────────────────────────────
def load_image(path: Path, size: int = 224) -> np.ndarray:
    """이미지를 [0,1] float32 HWC 텐서로 로드."""
    img = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


def to_nchw(x_hwc: np.ndarray) -> np.ndarray:
    """HWC → NCHW (ONNX 표준 레이아웃)."""
    return x_hwc.transpose(2, 0, 1)[np.newaxis, ...]


# ── 유틸리티 ──────────────────────────────────────────────────────────────
def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


# ── ONNX 세션 ─────────────────────────────────────────────────────────────
class OnnxModel:
    def __init__(self, model_path: Path, threads: int):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.log_severity_level = 3

        self.session = ort.InferenceSession(
            str(model_path), opts, providers=["CPUExecutionProvider"]
        )
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        # ONNX 모델 입력 shape 기준으로 레이아웃 결정
        self._is_nchw = inp.shape[1] in (1, 3) if len(inp.shape) == 4 else True

    def infer(self, image_hwc: np.ndarray) -> np.ndarray:
        x = to_nchw(image_hwc) if self._is_nchw else image_hwc[np.newaxis, ...]
        raw = self.session.run(None, {self.input_name: x.astype(np.float32)})[0][0]
        return np.asarray(raw, dtype=np.float32)


# ── ICWMV 융합 ────────────────────────────────────────────────────────────
def icwmv_fuse(
    mnv2_probs: np.ndarray,   # (3,) [auth, manip, aigen]
    specm_probs: np.ndarray,  # (2,) [auth, manip]
    w_spec: float = 1.0,
) -> tuple[str, np.ndarray]:
    """
    Inverse-Confidence Weighted Majority Vote (simple weighted average).

    auth  = (mnv2[auth]  + w * specm[auth])  / (1 + w)
    manip = (mnv2[manip] + w * specm[manip]) / (1 + w)
    aigen = mnv2[aigen]                        (SpecM은 AI 탐지 불가)
    """
    s_auth  = (mnv2_probs[0] + w_spec * specm_probs[0]) / (1.0 + w_spec)
    s_manip = (mnv2_probs[1] + w_spec * specm_probs[1]) / (1.0 + w_spec)
    s_aigen = mnv2_probs[2]

    raw = np.array([s_auth, s_manip, s_aigen], dtype=np.float32)
    probs = raw / raw.sum()
    label = CLASSES_3[int(np.argmax(probs))]
    return label, probs


# ── 메인 추론 클래스 ──────────────────────────────────────────────────────
class ShieldCPU:
    def __init__(self, mnv2_path: Path, specm_path: Path, threads: int = 4):
        t0 = time.perf_counter()
        self.mnv2  = OnnxModel(mnv2_path, threads)
        t1 = time.perf_counter()
        self.specm = OnnxModel(specm_path, threads)
        t2 = time.perf_counter()
        self.load_ms = {
            "mnv2_load_ms":  round((t1 - t0) * 1000, 1),
            "specm_load_ms": round((t2 - t1) * 1000, 1),
            "total_load_ms": round((t2 - t0) * 1000, 1),
        }

    def predict_array(self, x: np.ndarray, w_spec: float = 1.0) -> dict:
        # 단계별 타이밍
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
            "backend":    "onnx_cpu",
            "w_spec":     float(w_spec),
            "verdict":    verdict,
            "confidence": float(np.max(final_probs)),
            "scores":     {c: round(float(p), 4) for c, p in zip(CLASSES_3, final_probs)},
            "mnv2_scores": {c: round(float(p), 4) for c, p in zip(CLASSES_3, mnv2_probs)},
            "specm_scores": {c: round(float(p), 4) for c, p in zip(CLASSES_2, specm_probs)},
            "latency": {
                "mnv2_ms":  mnv2_ms,
                "specm_ms": specm_ms,
                "total_ms": total_ms,
            },
            "load": self.load_ms,
        }

    def predict(self, image_path: Path, w_spec: float = 1.0) -> dict:
        x = load_image(image_path)
        return self.predict_array(x, w_spec=w_spec)

    def benchmark(self, image_path: Path, w_spec: float, warmup: int, runs: int) -> dict:
        x = load_image(image_path)

        for _ in range(max(warmup, 0)):
            self.predict_array(x, w_spec=w_spec)

        results = [self.predict_array(x, w_spec=w_spec) for _ in range(max(runs, 0))]
        if not results:
            raise ValueError("benchmark runs must be >= 1")

        mnv2_vals = [float(r["latency"]["mnv2_ms"]) for r in results]
        specm_vals = [float(r["latency"]["specm_ms"]) for r in results]
        total_vals = [float(r["latency"]["total_ms"]) for r in results]

        def summary(values: list[float]) -> dict:
            return {
                "avg": round(statistics.mean(values), 3),
                "std": round(statistics.pstdev(values), 3),
                "min": round(min(values), 3),
                "max": round(max(values), 3),
            }

        base = results[-1]
        return {
            "backend": "onnx_cpu",
            "mode": "benchmark",
            "w_spec": float(w_spec),
            "warmup_runs_discarded": int(warmup),
            "measured_runs": int(runs),
            "scores": base["scores"],
            "mnv2_scores": base["mnv2_scores"],
            "specm_scores": base["specm_scores"],
            "verdict": base["verdict"],
            "confidence": base["confidence"],
            "load": self.load_ms,
            "summary_ms": {
                "mnv2": summary(mnv2_vals),
                "specm": summary(specm_vals),
                "total": summary(total_vals),
            },
            "all_runs": [
                {
                    "run_index": idx + 1,
                    "latency": row["latency"],
                    "verdict": row["verdict"],
                    "confidence": row["confidence"],
                }
                for idx, row in enumerate(results)
            ],
        }


# ── CLI ───────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="SHIELD RPi5 CPU 추론 (ONNX Runtime, Coral 없음)"
    )
    p.add_argument("image", type=Path, help="분석할 이미지 경로")
    p.add_argument("--mnv2", type=Path, default=DEFAULT_MNV2,
                   help=f"MNV2 ONNX 경로 (기본: {DEFAULT_MNV2})")
    p.add_argument("--specm", type=Path, default=DEFAULT_SPECM,
                   help=f"SpecM ONNX 경로 (기본: {DEFAULT_SPECM})")
    p.add_argument("--threads", type=int, default=4,
                   help="ONNX intra-op 스레드 수 (기본: 4, RPi5 코어 수 권장)")
    p.add_argument("--w-spec", type=float, default=1.0,
                   help="SpecM 융합 가중치 (기본: 1.0)")
    p.add_argument("--benchmark-warmup", type=int, default=0,
                   help="같은 세션에서 먼저 버릴 warm-up 횟수")
    p.add_argument("--benchmark-runs", type=int, default=0,
                   help="같은 세션에서 반복 측정할 횟수")
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
            raise SystemExit(1)

    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        print("오류: onnxruntime 미설치 — `pip install onnxruntime`", file=sys.stderr)
        raise SystemExit(1)

    model = ShieldCPU(args.mnv2, args.specm, threads=args.threads)
    if args.benchmark_runs > 0:
        result = model.benchmark(
            args.image,
            w_spec=args.w_spec,
            warmup=args.benchmark_warmup,
            runs=args.benchmark_runs,
        )
    else:
        result = model.predict(args.image, w_spec=args.w_spec)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.benchmark_runs > 0:
        total = result["summary_ms"]["total"]
        mnv2 = result["summary_ms"]["mnv2"]
        specm = result["summary_ms"]["specm"]
        print(f"백엔드 : {result['backend']}  (threads={args.threads})")
        print(f"반복측정: warmup={args.benchmark_warmup}  measured={args.benchmark_runs}")
        print(f"MNV2   : avg={mnv2['avg']}ms  std={mnv2['std']}ms")
        print(f"SpecM  : avg={specm['avg']}ms  std={specm['std']}ms")
        print(f"합계   : avg={total['avg']}ms  std={total['std']}ms")
        return

    lat = result["latency"]
    load = result["load"]
    verdict = result["verdict"]
    scores = result["scores"]
    m = result["mnv2_scores"]
    s = result["specm_scores"]

    print(f"백엔드 : {result['backend']}  (threads={args.threads})")
    print(f"판정   : {VERDICT_KO.get(verdict, verdict)} ({verdict}, {result['confidence']*100:.1f}%)")
    print(f"  auth={scores['authentic']:.3f}  manip={scores['manipulated']:.3f}  aigen={scores['ai_generated']:.3f}")
    print(f"MNV2   : auth={m['authentic']:.3f}  manip={m['manipulated']:.3f}  aigen={m['ai_generated']:.3f}")
    print(f"SpecM  : auth={s['authentic']:.3f}  manip={s['manipulated']:.3f}")
    print(f"레이턴시: MNV2={lat['mnv2_ms']}ms  SpecM={lat['specm_ms']}ms  합계={lat['total_ms']}ms")
    print(f"모델로드: MNV2={load['mnv2_load_ms']}ms  SpecM={load['specm_load_ms']}ms  합계={load['total_load_ms']}ms")


if __name__ == "__main__":
    main()
