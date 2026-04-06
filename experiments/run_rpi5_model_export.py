#!/usr/bin/env python3
"""
RPi5 배포용 최종 모델 생성 + 정확도 검증
=========================================

대상:
  1. mnv2_int8_static.onnx    — 이미 존재, 정확도만 재검증
  2. specm_v4_int8_static.onnx — 신규 생성 (specm_v4.onnx 기반 static PTQ)

결과:
  weights/onnx_quant/specm_v4_int8_static.onnx
  experiments/results/rpi5_model_export/rpi5_model_export_{ts}.json

실행:
  .venv-qwen/bin/python experiments/run_rpi5_model_export.py
  .venv-qwen/bin/python experiments/run_rpi5_model_export.py --skip-existing
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from tqdm import tqdm

ROOT      = Path(__file__).resolve().parents[1]
ONNX_FP32 = ROOT / "weights" / "onnx"
ONNX_Q    = ROOT / "weights" / "onnx_quant"
DS_ROOT   = ROOT / "datasets"
OUT_DIR   = ROOT / "experiments" / "results" / "rpi5_model_export"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_CALIB = 128   # 캘리브레이션 이미지 수
N_BENCH = 30    # 벤치마크 반복 횟수
N_ACC   = 200   # 정확도 검증 이미지 수


# ── 캘리브레이션 이미지 수집 ────────────────────────────────────────────────

def collect_images(n: int) -> list[Path]:
    """CASIA2 + OpenSDID에서 랜덤하게 n장 수집."""
    import random
    candidates: list[Path] = []
    for subdir in [
        DS_ROOT / "CASIA2_subset" / "Au",
        DS_ROOT / "CASIA2_subset" / "Tp",
        DS_ROOT / "OpenSDID_subset",
    ]:
        if subdir.exists():
            candidates.extend(subdir.rglob("*.jpg"))
            candidates.extend(subdir.rglob("*.png"))
    if not candidates:
        raise RuntimeError(f"캘리브레이션 이미지를 찾을 수 없습니다 (datasets/ 확인)")
    random.seed(42)
    random.shuffle(candidates)
    found = [p for p in candidates if p.exists()][:n]
    print(f"  캘리브레이션 이미지: {len(found)}장 (후보 {len(candidates)}장)")
    return found


def load_raw(path: Path, size: int = 224) -> Optional[np.ndarray]:
    """[0,1] float32 NCHW — 모델 내부에서 ImageNet 정규화 수행."""
    try:
        img = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
        x = np.asarray(img, dtype=np.float32) / 255.0
        return x.transpose(2, 0, 1)[np.newaxis, ...]   # NCHW
    except Exception:
        return None


# ── CalibrationDataReader ─────────────────────────────────────────────────

class RawCalibReader:
    def __init__(self, images: list[Path], input_name: str, size: int = 224):
        self.images = images
        self.input_name = input_name
        self.size = size
        self._idx = 0

    def get_next(self):
        while self._idx < len(self.images):
            x = load_raw(self.images[self._idx], self.size)
            self._idx += 1
            if x is not None:
                return {self.input_name: x}
        return None

    def rewind(self):
        self._idx = 0


# ── Static PTQ ──────────────────────────────────────────────────────────────

def static_quantize(src: Path, dst: Path, calib_images: list[Path], input_name: str) -> bool:
    from onnxruntime.quantization import (
        quantize_static, QuantType, QuantFormat, CalibrationMethod,
    )
    print(f"  Static PTQ: {src.name} → {dst.name}")

    # per_channel=True 시도 → 실패 시 per_channel=False로 재시도
    for per_channel in (True, False):
        reader = RawCalibReader(calib_images, input_name)
        try:
            quantize_static(
                model_input=str(src),
                model_output=str(dst),
                calibration_data_reader=reader,
                quant_format=QuantFormat.QDQ,       # QDQ: 더 넓은 op 호환
                per_channel=per_channel,
                weight_type=QuantType.QInt8,
                activation_type=QuantType.QInt8,
                calibrate_method=CalibrationMethod.MinMax,
            )
            print(f"  ✓ 완료 (per_channel={per_channel}): {dst.stat().st_size/1e6:.1f}MB")
            return True
        except Exception as exc:
            print(f"  per_channel={per_channel} 실패: {exc}")
            if dst.exists():
                dst.unlink()
    return False


# ── 벤치마크 ─────────────────────────────────────────────────────────────────

def bench(model_path: Path, input_name: str, threads: int = 4) -> float:
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.log_severity_level = 3
    sess = ort.InferenceSession(str(model_path), opts,
                                providers=["CPUExecutionProvider"])
    x = np.random.rand(1, 3, 224, 224).astype(np.float32)
    for _ in range(5):  # 워밍업
        sess.run(None, {input_name: x})
    t0 = time.perf_counter()
    for _ in range(N_BENCH):
        sess.run(None, {input_name: x})
    return (time.perf_counter() - t0) / N_BENCH * 1000


# ── 정확도 검증 (LOO-CD 결과 JSONL 재활용) ──────────────────────────────────

def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


LABEL_MAP_3 = ["authentic", "manipulated", "ai_generated"]
LABEL_MAP_2 = ["authentic", "manipulated"]


def eval_model_on_dataset(
    model_path: Path,
    input_name: str,
    jsonl_path: Path,
    n_images: int,
) -> dict:
    """JSONL에서 이미지 경로 + 라벨 읽어서 macro-F1 계산."""
    import onnxruntime as ort
    from sklearn.metrics import f1_score  # type: ignore

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    opts.log_severity_level = 3
    sess = ort.InferenceSession(str(model_path), opts,
                                providers=["CPUExecutionProvider"])

    # 출력 크기로 클래스 맵 결정
    n_out = sess.run(None, {input_name: np.random.rand(1,3,224,224).astype(np.float32)})[0].shape[-1]
    label_map = LABEL_MAP_3 if n_out == 3 else LABEL_MAP_2

    preds, gts = [], []
    count = 0
    with open(jsonl_path) as f:
        for line in f:
            if count >= n_images:
                break
            r = json.loads(line.strip())
            img_p = ROOT / r["image_path"]
            if not img_p.exists():
                continue
            true_label = r.get("true_label", r.get("label", ""))
            # 3-class 모델에서 2-class 라벨 처리: ai_generated → skip
            if n_out == 2 and true_label not in LABEL_MAP_2:
                continue
            x = load_raw(img_p)
            if x is None:
                continue
            logits = sess.run(None, {input_name: x})[0][0]
            pred_idx = int(np.argmax(softmax(logits)))
            preds.append(label_map[pred_idx])
            gts.append(true_label)
            count += 1

    if not preds:
        return {"f1": None, "n": 0}

    f1 = f1_score(gts, preds, average="macro", zero_division=0)
    return {"f1": round(float(f1), 4), "n": len(preds)}


def compare_accuracy(
    fp32_path: Path,
    dyn_path: Optional[Path],
    sta_path: Optional[Path],
    input_name: str,
    label: str,
) -> dict:
    """FP32 vs Dynamic vs Static 정확도 비교 (base 데이터셋 기준)."""
    import glob
    # base 데이터셋 JSONL 찾기
    if "mnv2" in label.lower():
        pattern = str(ROOT / "experiments/results/backbone_eval/mobilenetv2_dualstream_base_20260319_070725.jsonl")
    else:
        pattern = str(ROOT / "experiments/results/specialist_eval/specialist_m_v4_base_*.jsonl")
        matches = sorted(glob.glob(pattern))
        pattern = matches[-1] if matches else ""

    jsonl = Path(pattern)
    if not jsonl.exists():
        print(f"  [경고] JSONL 없음: {jsonl}")
        return {}

    print(f"  정확도 비교 (n={N_ACC})...")
    result = {}
    for name, path in [("fp32", fp32_path), ("dynamic", dyn_path), ("static", sta_path)]:
        if path is None or not path.exists():
            result[name] = "N/A"
            continue
        r = eval_model_on_dataset(path, input_name, jsonl, N_ACC)
        result[name] = r
        print(f"    {name:8s}: F1={r['f1']}  (n={r['n']})")
    return result


# ── 메인 ─────────────────────────────────────────────────────────────────────

MODELS = {
    "mnv2": {
        "fp32":    ONNX_FP32 / "mnv2.onnx",
        "dynamic": ONNX_Q   / "mnv2_int8_dynamic.onnx",
        "static":  ONNX_Q   / "mnv2_int8_static.onnx",
        "input_name": "image_01",
        "need_export": True,    # 기존 static 정확도 저하 → QDQ로 재생성
    },
    "specm_v4": {
        "fp32":    ONNX_FP32 / "specm_v4.onnx",
        "dynamic": ONNX_Q   / "specm_v4_int8_dynamic.onnx",
        "static":  ONNX_Q   / "specm_v4_int8_static.onnx",
        "input_name": "image_01",
        "need_export": True,    # 신규 생성 필요
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-existing", action="store_true",
                        help="이미 존재하는 static 모델 재생성 건너뜀")
    parser.add_argument("--no-accuracy", action="store_true",
                        help="정확도 검증 건너뜀 (속도 우선)")
    args = parser.parse_args()

    calib_images = collect_images(N_CALIB)

    all_results = {}

    for key, spec in MODELS.items():
        print(f"\n{'='*60}")
        print(f"  [{key}]")
        print(f"{'='*60}")

        sta_path: Path = spec["static"]
        fp32_path: Path = spec["fp32"]
        dyn_path: Path = spec["dynamic"]
        input_name: str = spec["input_name"]

        model_result: dict = {
            "fp32_mb":    round(fp32_path.stat().st_size / 1e6, 1) if fp32_path.exists() else None,
            "dynamic_mb": round(dyn_path.stat().st_size / 1e6, 1) if dyn_path.exists() else None,
            "static_mb":  None,
        }

        # Static 모델 생성
        if sta_path.exists() and (args.skip_existing or not spec["need_export"]):
            print(f"  [SKIP] static 이미 존재: {sta_path.name} ({sta_path.stat().st_size/1e6:.1f}MB)")
        else:
            ok = static_quantize(fp32_path, sta_path, calib_images, input_name)
            if not ok:
                print(f"  [ERROR] static 생성 실패 — 건너뜀")
                all_results[key] = model_result
                continue

        if sta_path.exists():
            model_result["static_mb"] = round(sta_path.stat().st_size / 1e6, 1)

        # 벤치마크 (서버 기준, RPi5는 ~4× 느림)
        print("\n  벤치마크 (threads=4, n=30):")
        for name, path in [("fp32", fp32_path), ("dynamic", dyn_path), ("static", sta_path)]:
            if path and path.exists():
                ms = bench(path, input_name)
                model_result[f"{name}_ms"] = round(ms, 1)
                rpi5_est = round(ms * 4, 0)
                print(f"    {name:8s}: {ms:6.1f}ms/img  (RPi5 추정 ~{rpi5_est:.0f}ms)")

        # 정확도 검증
        if not args.no_accuracy:
            acc = compare_accuracy(fp32_path, dyn_path, sta_path, input_name, key)
            model_result["accuracy"] = acc

        all_results[key] = model_result

    # 요약
    print(f"\n{'='*60}")
    print("  요약")
    print(f"{'='*60}")
    for key, r in all_results.items():
        print(f"\n  [{key}]")
        for name in ("fp32", "dynamic", "static"):
            mb  = r.get(f"{name}_mb", "N/A")
            ms  = r.get(f"{name}_ms", "N/A")
            acc_r = r.get("accuracy", {}).get(name, "N/A")
            f1  = acc_r["f1"] if isinstance(acc_r, dict) else acc_r
            print(f"    {name:8s}: {str(mb):>6}MB  {str(ms):>6}ms(서버)  F1={f1}")

    # 권고 출력
    print(f"\n{'='*60}")
    print("  RPi5 권고 모델")
    print(f"{'='*60}")
    print("  CPU 경로:")
    print(f"    MNV2  → {ONNX_Q}/mnv2_int8_static.onnx")
    print(f"    SpecM → {ONNX_Q}/specm_v4_int8_static.onnx")
    print("  Coral 경로:")
    print(f"    MNV2  → weights/tflite_edgetpu_sweep/mnv2_coral_qsweep_qtpc_cal064_ioint8_edgetpu.tflite")
    print(f"    SpecM → weights/tflite_edgetpu/specm_v4_coral_ft_int8_full_edgetpu.tflite")

    # 저장
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUT_DIR / f"rpi5_model_export_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n  저장: {out_path}")


if __name__ == "__main__":
    main()
