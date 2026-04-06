#!/usr/bin/env python3
"""
Phase 4.2 — ONNX INT8 양자화 (Dynamic + Static PTQ)
=====================================================

목표: SpecG RPi5 추정 800ms → ≤200ms (4× 단축 목표)

전략:
  1. Dynamic Quantization — 가중치 INT8, 활성화 FP32 (캘리브레이션 불필요)
  2. Static Quantization  — 가중치+활성화 INT8 (캘리브레이션 100장 필요)
  3. 크기 비교 + CPU 지연 재측정 + 로짓 정확도 검증

실행:
  .venv-qwen/bin/python experiments/run_quantization.py
  .venv-qwen/bin/python experiments/run_quantization.py --models specg specm
  .venv-qwen/bin/python experiments/run_quantization.py --skip-static  # dynamic only
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

ROOT     = Path(__file__).resolve().parents[1]
ONNX_DIR = ROOT / "weights" / "onnx"
QUANT_DIR = ROOT / "weights" / "onnx_quant"
OUT_DIR  = ROOT / "experiments" / "results" / "onnx_benchmark"
QUANT_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

RPI5_SCALE   = 4.0
RPI5_BUDGET  = 200.0
N_CALIB      = 100   # 캘리브레이션 이미지 수
N_BENCH_RUNS = 50

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1,3,1,1)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1,3,1,1)
CLIP_MEAN     = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32).reshape(1,3,1,1)
CLIP_STD      = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32).reshape(1,3,1,1)

MODEL_SPECS = {
    "mnv2":  {"input_name": "image_01",  "input_size": (1,3,224,224), "norm": "imagenet"},
    "specm": {"input_name": "image_01",  "input_size": (1,3,224,224), "norm": "imagenet"},
    "specg": {"input_name": "image_clip","input_size": (1,3,256,256), "norm": "clip"},
    "clip":  {"input_name": "image_clip","input_size": (1,3,256,256), "norm": "clip"},
}


# ── 전처리 ───────────────────────────────────────────────────────────────────

def preprocess(img_path: Path, size: int, norm: str) -> Optional[np.ndarray]:
    try:
        from PIL import Image
        img = Image.open(img_path).convert("RGB")
        img = img.resize((size, size), Image.BILINEAR)
        x = np.array(img, dtype=np.float32) / 255.0      # [H,W,3]
        x = x.transpose(2, 0, 1)[None]                   # [1,3,H,W]
        if norm == "imagenet":
            x = (x - IMAGENET_MEAN) / IMAGENET_STD
        else:  # clip
            x = (x - CLIP_MEAN) / CLIP_STD
        return x.astype(np.float32)
    except Exception:
        return None


def collect_calib_images(n: int) -> List[Path]:
    """base 데이터셋 JSONL에서 실제 이미지 경로 수집."""
    import glob
    jsonl_files = sorted(glob.glob(
        str(ROOT / "experiments/results/backbone_eval/mobilenetv2_dualstream_base_*.jsonl")))
    if not jsonl_files:
        return []
    paths = []
    with open(jsonl_files[-1]) as f:
        for line in f:
            r = json.loads(line.strip())
            p = ROOT / r["image_path"]
            if p.exists():
                paths.append(p)
            if len(paths) >= n:
                break
    return paths


# ── CalibrationDataReader ────────────────────────────────────────────────────

class ImageCalibDataReader:
    """onnxruntime.quantization.CalibrationDataReader 구현."""

    def __init__(self, images: List[Path], input_name: str,
                 input_size: tuple, norm: str):
        from onnxruntime.quantization import CalibrationDataReader
        self._base_class = CalibrationDataReader
        self.images     = images
        self.input_name = input_name
        self.h = self.w = input_size[-1]
        self.norm       = norm
        self._idx       = 0

    def get_next(self) -> Optional[Dict]:
        if self._idx >= len(self.images):
            return None
        x = preprocess(self.images[self._idx], self.h, self.norm)
        self._idx += 1
        if x is None:
            return self.get_next()
        return {self.input_name: x}

    def rewind(self):
        self._idx = 0


# ── 벤치마크 ─────────────────────────────────────────────────────────────────

def benchmark_onnx(model_path: Path, input_name: str,
                   dummy: np.ndarray, n_warmup=5, n_runs=N_BENCH_RUNS) -> float:
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(model_path), opts,
                                providers=["CPUExecutionProvider"])
    feed = {input_name: dummy}
    for _ in range(n_warmup):
        sess.run(None, feed)
    t0 = time.perf_counter()
    for _ in range(n_runs):
        sess.run(None, feed)
    return (time.perf_counter() - t0) / n_runs * 1000


def check_accuracy(orig_path: Path, quant_path: Path,
                   input_name: str, dummy: np.ndarray) -> Dict:
    """원본 vs 양자화 로짓 차이 (cosine similarity + max abs diff)."""
    import onnxruntime as ort
    def run(p):
        s = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
        return s.run(None, {input_name: dummy})[0]
    try:
        orig  = run(orig_path).flatten()
        quant = run(quant_path).flatten()
        cos   = float(np.dot(orig, quant) /
                      (np.linalg.norm(orig) * np.linalg.norm(quant) + 1e-8))
        mad   = float(np.max(np.abs(orig - quant)))
        return {"cosine_sim": round(cos, 5), "max_abs_diff": round(mad, 5)}
    except Exception as e:
        return {"error": str(e)}


def get_size_mb(path: Path) -> float:
    total = path.stat().st_size if path.exists() else 0
    ext   = path.with_suffix(".onnx.data")
    if ext.exists():
        total += ext.stat().st_size
    return round(total / 1024 / 1024, 1)


# ── 양자화 ────────────────────────────────────────────────────────────────────

def _embed_external_data(src: Path) -> Path:
    """
    외부 데이터(.onnx.data)를 ONNX 모델에 내장 → 단일 파일로 병합.
    양자화 도구들이 external data를 직접 다루지 못하는 문제 우회.
    """
    import onnx
    embed_path = QUANT_DIR / (src.stem + "_embedded.onnx")
    if embed_path.exists():
        return embed_path
    print(f"      외부 데이터 내장 중... (크기 주의)")
    # load_external_data=True → 가중치를 메모리에 로드
    model = onnx.load(str(src), load_external_data=True)
    # 단일 파일로 저장 (external_data 없이)
    onnx.save_model(model, str(embed_path),
                    save_as_external_data=False)
    size_mb = embed_path.stat().st_size / 1024 / 1024
    print(f"      병합 완료: {embed_path.name}  ({size_mb:.1f} MB)")
    return embed_path


def _preprocess_onnx(src: Path) -> Path:
    """
    quant_pre_process: shape inference + optimization.
    외부 데이터가 있으면 먼저 단일 파일로 병합.
    """
    from onnxruntime.quantization import quant_pre_process

    # 외부 데이터 병합
    data_file = src.with_suffix(".onnx.data")
    if data_file.exists():
        src = _embed_external_data(src)

    prep = QUANT_DIR / (src.stem + "_prep.onnx")
    if prep.exists():
        return prep
    try:
        quant_pre_process(str(src), str(prep), skip_symbolic_shape=True)
        return prep
    except Exception as e:
        print(f"      [WARN] pre_process 실패({e}), 병합본 사용")
        return src


def dynamic_quantize(src: Path, dst: Path) -> bool:
    """
    Dynamic quantization: Conv/Gemm/MatMul 가중치를 INT8로.
    LayerNormalization은 shape inference 충돌로 제외.
    """
    from onnxruntime.quantization import quantize_dynamic, QuantType

    # 외부 데이터 내장 (단일 파일)
    data_file = src.with_suffix(".onnx.data")
    if data_file.exists():
        src = _embed_external_data(src)

    # Gemm/MatMul만 양자화 (ConvInteger는 이 ORT 빌드에서 미구현)
    # Conv INT8은 static quant(QLinearConv)에서만 지원됨
    quantize_ops = ["Gemm", "MatMul"]
    try:
        quantize_dynamic(
            model_input=str(src),
            model_output=str(dst),
            op_types_to_quantize=quantize_ops,
            weight_type=QuantType.QInt8,
            per_channel=False,
            reduce_range=False,
        )
        return True
    except Exception as e:
        print(f"      [ERROR] Dynamic quant 실패: {e}")
        return False


def static_quantize(src: Path, dst: Path, calib_reader) -> bool:
    """
    Static quantization (QDQ 포맷): 가중치+활성화 INT8.
    QDQ 포맷 → QLinearConv 사용 → ConvInteger 문제 우회.
    """
    from onnxruntime.quantization import (
        quantize_static, QuantType, QuantFormat, CalibrationMethod,
    )

    # 외부 데이터 내장 (단일 파일)
    data_file = src.with_suffix(".onnx.data")
    if data_file.exists():
        src = _embed_external_data(src)

    try:
        quantize_static(
            model_input=str(src),
            model_output=str(dst),
            calibration_data_reader=calib_reader,
            quant_format=QuantFormat.QDQ,       # QLinearConv 생성
            weight_type=QuantType.QInt8,
            activation_type=QuantType.QInt8,
            calibrate_method=CalibrationMethod.MinMax,
            per_channel=False,
            reduce_range=False,
        )
        return True
    except Exception as e:
        print(f"      [ERROR] Static quant 실패: {e}")
        # fallback: QOperator 포맷 시도
        try:
            quantize_static(
                model_input=str(src),
                model_output=str(dst),
                calibration_data_reader=calib_reader,
                quant_format=QuantFormat.QOperator,
                weight_type=QuantType.QUInt8,
                activation_type=QuantType.QUInt8,
                calibrate_method=CalibrationMethod.MinMax,
                per_channel=False,
                reduce_range=False,
            )
            print(f"      (QOperator fallback 성공)")
            return True
        except Exception as e2:
            print(f"      [ERROR] Static quant fallback도 실패: {e2}")
            return False


# ── 모델별 양자화 실행 ────────────────────────────────────────────────────────

def quantize_model(key: str, skip_static: bool,
                   calib_images: List[Path]) -> Dict:
    spec      = MODEL_SPECS[key]
    orig_path = ONNX_DIR / f"{key}.onnx"
    dyn_path  = QUANT_DIR / f"{key}_int8_dynamic.onnx"
    sta_path  = QUANT_DIR / f"{key}_int8_static.onnx"

    print(f"\n{'='*55}")
    print(f"  [{key.upper()}] INT8 양자화")
    print(f"{'='*55}")

    if not orig_path.exists():
        print(f"  [SKIP] 원본 ONNX 없음: {orig_path.name}")
        return {"key": key, "skip_reason": "no onnx"}

    input_name = spec["input_name"]
    h = w = spec["input_size"][-1]
    dummy = np.random.rand(1, 3, h, w).astype(np.float32)

    # 원본 지연 (기준)
    orig_ms = benchmark_onnx(orig_path, input_name, dummy)
    orig_mb = get_size_mb(orig_path)
    print(f"  원본:    {orig_mb:.1f} MB  |  {orig_ms:.1f} ms  "
          f"|  RPi5 추정 {orig_ms * RPI5_SCALE:.0f} ms")

    result = {
        "key":        key,
        "orig_mb":    orig_mb,
        "orig_ms":    round(orig_ms, 2),
        "orig_rpi5":  round(orig_ms * RPI5_SCALE, 1),
        "dynamic":    {},
        "static":     {},
    }

    # ── Dynamic 양자화 ─────────────────────────────────────────────────────
    print(f"\n  [Dynamic INT8]")
    if not dyn_path.exists():
        print(f"    양자화 실행 중...")
        ok = dynamic_quantize(orig_path, dyn_path)
    else:
        print(f"    기존 파일 재사용: {dyn_path.name}")
        ok = True

    if ok and dyn_path.exists():
        dyn_ms = benchmark_onnx(dyn_path, input_name, dummy)
        dyn_mb = get_size_mb(dyn_path)
        dyn_acc = check_accuracy(orig_path, dyn_path, input_name, dummy)
        dyn_rpi5 = dyn_ms * RPI5_SCALE
        speedup = orig_ms / dyn_ms
        budget_ok = dyn_rpi5 <= RPI5_BUDGET
        budget_str = "✓ OK" if budget_ok else "✗ OVER"
        print(f"    크기:   {orig_mb:.1f} → {dyn_mb:.1f} MB  "
              f"({100*(1-dyn_mb/orig_mb):.0f}% 절감)")
        print(f"    지연:   {orig_ms:.1f} → {dyn_ms:.1f} ms  "
              f"({speedup:.1f}× 가속)")
        print(f"    RPi5:   {dyn_rpi5:.0f} ms  [{budget_str}]")
        print(f"    정확도: cosine={dyn_acc.get('cosine_sim','—')}  "
              f"max_diff={dyn_acc.get('max_abs_diff','—')}")
        result["dynamic"] = {
            "mb": dyn_mb, "ms": round(dyn_ms,2), "rpi5_ms": round(dyn_rpi5,1),
            "speedup": round(speedup,2), "budget_ok": budget_ok,
            **dyn_acc,
        }
    else:
        result["dynamic"]["error"] = "quantization failed"

    # ── Static 양자화 ──────────────────────────────────────────────────────
    if skip_static:
        print(f"\n  [Static INT8] SKIP (--skip-static)")
        return result

    print(f"\n  [Static INT8]  (캘리브레이션 {len(calib_images)}장)")
    if not sta_path.exists():
        reader = ImageCalibDataReader(
            calib_images, input_name, spec["input_size"], spec["norm"])
        print(f"    양자화 실행 중...")
        ok = static_quantize(orig_path, sta_path, reader)
    else:
        print(f"    기존 파일 재사용: {sta_path.name}")
        ok = True

    if ok and sta_path.exists():
        sta_ms = benchmark_onnx(sta_path, input_name, dummy)
        sta_mb = get_size_mb(sta_path)
        sta_acc = check_accuracy(orig_path, sta_path, input_name, dummy)
        sta_rpi5 = sta_ms * RPI5_SCALE
        speedup = orig_ms / sta_ms
        budget_ok = sta_rpi5 <= RPI5_BUDGET
        budget_str = "✓ OK" if budget_ok else "✗ OVER"
        print(f"    크기:   {orig_mb:.1f} → {sta_mb:.1f} MB  "
              f"({100*(1-sta_mb/orig_mb):.0f}% 절감)")
        print(f"    지연:   {orig_ms:.1f} → {sta_ms:.1f} ms  "
              f"({speedup:.1f}× 가속)")
        print(f"    RPi5:   {sta_rpi5:.0f} ms  [{budget_str}]")
        print(f"    정확도: cosine={sta_acc.get('cosine_sim','—')}  "
              f"max_diff={sta_acc.get('max_abs_diff','—')}")
        result["static"] = {
            "mb": sta_mb, "ms": round(sta_ms,2), "rpi5_ms": round(sta_rpi5,1),
            "speedup": round(speedup,2), "budget_ok": budget_ok,
            **sta_acc,
        }
    else:
        result["static"]["error"] = "quantization failed"

    return result


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+",
                        choices=list(MODEL_SPECS),
                        default=list(MODEL_SPECS))
    parser.add_argument("--skip-static", action="store_true")
    args = parser.parse_args()

    print("=" * 65)
    print("Phase 4.2 — INT8 양자화 (Dynamic + Static PTQ)")
    print("=" * 65)
    print(f"  대상 모델:  {args.models}")
    print(f"  출력 경로:  {QUANT_DIR}")
    print(f"  RPi5 예산:  {RPI5_BUDGET} ms  (스케일 {RPI5_SCALE}×)")

    # 캘리브레이션 이미지 수집
    calib_images = []
    if not args.skip_static:
        print(f"\n  캘리브레이션 이미지 수집 중... (N={N_CALIB})")
        calib_images = collect_calib_images(N_CALIB)
        print(f"  수집됨: {len(calib_images)}장")
        if len(calib_images) < 20:
            print("  [경고] 이미지 부족 — static 양자화 품질이 낮을 수 있습니다")

    all_results = []
    for key in args.models:
        r = quantize_model(key, args.skip_static, calib_images)
        all_results.append(r)

    # ── 요약 테이블 ─────────────────────────────────────────────────────────
    print(f"\n\n{'='*75}")
    print("양자화 요약")
    print(f"{'='*75}")
    print(f"{'모델':<8} {'원본':>8} {'Dyn크기':>8} {'DynRPi5':>9} {'DynOK':>6} "
          f"{'Sta크기':>8} {'StaRPi5':>9} {'StaOK':>6}")
    print("-" * 75)

    best_specg_rpi5 = None

    for r in all_results:
        if "skip_reason" in r:
            print(f"{r['key']:<8} SKIP")
            continue
        orig = f"{r['orig_rpi5']:.0f}ms"
        d = r.get("dynamic", {})
        s = r.get("static", {})
        dyn_mb  = f"{d.get('mb',0):.1f}" if "mb"  in d else "—"
        dyn_rpi = f"{d.get('rpi5_ms',0):.0f}" if "rpi5_ms" in d else "—"
        dyn_ok  = "✓" if d.get("budget_ok") else ("✗" if "rpi5_ms" in d else "—")
        sta_mb  = f"{s.get('mb',0):.1f}" if "mb"  in s else "—"
        sta_rpi = f"{s.get('rpi5_ms',0):.0f}" if "rpi5_ms" in s else "—"
        sta_ok  = "✓" if s.get("budget_ok") else ("✗" if "rpi5_ms" in s else "—")
        print(f"{r['key']:<8} {orig:>8} {dyn_mb:>8} {dyn_rpi:>9} {dyn_ok:>6} "
              f"{sta_mb:>8} {sta_rpi:>9} {sta_ok:>6}")

        if r["key"] == "specg":
            best_ms = min(
                d.get("rpi5_ms", 9999),
                s.get("rpi5_ms", 9999),
            )
            if best_ms < 9999:
                best_specg_rpi5 = best_ms

    print("-" * 75)

    # RPi5 배포 시나리오
    print("\n[RPi5 배포 시나리오 분석]")
    mnv2_r = next((r for r in all_results if r["key"] == "mnv2"), {})
    specm_r = next((r for r in all_results if r["key"] == "specm"), {})
    specg_r = next((r for r in all_results if r["key"] == "specg"), {})

    def best_rpi5(r):
        d = r.get("dynamic", {}).get("rpi5_ms", 9999)
        s = r.get("static", {}).get("rpi5_ms", 9999)
        orig = r.get("orig_rpi5", 9999)
        return min(d, s, orig)

    mnv2_ms  = best_rpi5(mnv2_r)
    specm_ms = best_rpi5(specm_r)
    specg_ms = best_rpi5(specg_r)

    s1 = mnv2_ms + specm_ms
    s2 = mnv2_ms + specm_ms + specg_ms

    print(f"  시나리오 A (MNV2+SpecM):        {s1:.0f} ms  "
          f"{'✓' if s1 <= 500 else '✗'}  [3-class+manip, AI-gen 서버]")
    print(f"  시나리오 B (MNV2+SpecM+SpecG):  {s2:.0f} ms  "
          f"{'✓' if s2 <= 500 else '✗'}  [전체 4-class, AI-gen 포함]")
    print(f"  ※ 단일 이미지 RPi5 목표: ≤500ms (sequential 3모델)")

    # 저장
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = OUT_DIR / f"quantization_{ts}.json"
    with open(out, "w") as f:
        json.dump({
            "timestamp":    ts,
            "rpi5_scale":   RPI5_SCALE,
            "rpi5_budget":  RPI5_BUDGET,
            "scenario_a_ms": round(s1, 1),
            "scenario_b_ms": round(s2, 1),
            "results":      all_results,
        }, f, indent=2, ensure_ascii=False)

    # Markdown 리포트
    md = OUT_DIR / f"quantization_{ts}.md"
    lines = [
        "# Phase 4.2 INT8 양자화 결과\n",
        f"생성일: {ts}  |  RPi5 스케일: {RPI5_SCALE}×  |  예산: {RPI5_BUDGET}ms\n",
        "\n## 양자화 결과\n",
        "| 모델 | 원본RPi5 | Dynamic크기 | DynamicRPi5 | Static크기 | StaticRPi5 | 코사인유사도(Dyn) |",
        "|------|---------|-----------|------------|----------|----------|----------------|",
    ]
    for r in all_results:
        if "skip_reason" in r: continue
        d = r.get("dynamic", {}); s = r.get("static", {})
        lines.append(
            f"| {r['key']} | {r.get('orig_rpi5','—')}ms"
            f" | {d.get('mb','—')}MB | {d.get('rpi5_ms','—')}ms"
            f" | {s.get('mb','—')}MB | {s.get('rpi5_ms','—')}ms"
            f" | {d.get('cosine_sim','—')} |"
        )
    lines += [
        f"\n## RPi5 배포 시나리오",
        f"| 시나리오 | 구성 | 추정 지연 | 비고 |",
        f"|---------|------|---------|------|",
        f"| A | MNV2 + SpecM | {s1:.0f} ms | 3-class + manip 강화, AI-gen 서버 |",
        f"| B | MNV2 + SpecM + SpecG | {s2:.0f} ms | 전체 on-device |",
    ]
    with open(md, "w") as f:
        f.write("\n".join(lines))

    print(f"\n  결과 저장: {out.name}")
    print(f"  리포트:   {md.name}")


if __name__ == "__main__":
    main()
