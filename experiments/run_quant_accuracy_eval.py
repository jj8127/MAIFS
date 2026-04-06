#!/usr/bin/env python3
"""
Phase 4.2 — 양자화 모델 정확도 평가
=====================================
FP32 원본 vs Dynamic INT8 vs Static INT8 macro-F1 비교 (4개 데이터셋)

실행:
  .venv-qwen/bin/python experiments/run_quant_accuracy_eval.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

ROOT      = Path(__file__).resolve().parents[1]
ONNX_FP32 = ROOT / "weights" / "onnx"
ONNX_Q    = ROOT / "weights" / "onnx_quant"
BE        = ROOT / "experiments" / "results" / "backbone_eval"
SE        = ROOT / "experiments" / "results" / "specialist_eval"
OUT_DIR   = ROOT / "experiments" / "results" / "onnx_benchmark"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(1,3,1,1)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], np.float32).reshape(1,3,1,1)
CLIP_MEAN     = np.array([0.48145466, 0.4578275, 0.40821073], np.float32).reshape(1,3,1,1)
CLIP_STD      = np.array([0.26862954, 0.26130258, 0.27577711], np.float32).reshape(1,3,1,1)

CLASSES_3 = ["authentic", "manipulated", "ai_generated"]
CLASSES_2M = ["authentic", "manipulated"]
CLASSES_2G = ["authentic", "ai_generated"]

DATASETS = ["base", "dsC", "opensdi", "aigenproxy"]

# ── 데이터셋별 원본 eval JSONL 파일 ─────────────────────────────────────────

def get_eval_jsonl(model_prefix: str, ds: str) -> Optional[Path]:
    """backbone_eval 또는 specialist_eval 폴더에서 해당 데이터셋 JSONL 탐색."""
    if model_prefix in ("mnv2",):
        folder = BE
        pattern = f"mobilenetv2_dualstream_{ds}_20260319_070725.jsonl"
    elif model_prefix in ("clip",):
        folder = BE
        pattern = f"mobileclip_s2_finetuned_{ds}_20260319_061834.jsonl"
    elif model_prefix in ("specm",):
        folder = SE
        files = sorted(folder.glob(f"specialist_m_v2_{ds}_*.jsonl"))
        return files[-1] if files else None
    elif model_prefix in ("specg",):
        folder = SE
        files = sorted(folder.glob(f"specialist_g_{ds}_*.jsonl"))
        return files[-1] if files else None
    else:
        return None
    p = folder / pattern
    return p if p.exists() else None


# ── 전처리 ──────────────────────────────────────────────────────────────────

def load_image(path: Path, size: int, norm: str) -> Optional[np.ndarray]:
    try:
        from PIL import Image
        img = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
        x = np.array(img, dtype=np.float32) / 255.0
        x = x.transpose(2, 0, 1)[None]  # [1,3,H,W]
        if norm == "imagenet":
            return (x - IMAGENET_MEAN) / IMAGENET_STD
        elif norm == "clip":
            return (x - CLIP_MEAN) / CLIP_STD
        else:  # norm == "raw": ONNX 모델이 내부에서 정규화 수행 (MNV2, SpecM)
            return x
    except Exception:
        return None


# ── ONNX 추론 세션 ──────────────────────────────────────────────────────────

def make_session(onnx_path: Path):
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(str(onnx_path), opts,
                                providers=["CPUExecutionProvider"])


# ── macro-F1 계산 ────────────────────────────────────────────────────────────

def macro_f1(y_true: List[str], y_pred: List[str]) -> float:
    classes = sorted(set(y_true))
    f1s = []
    for c in classes:
        tp = sum(t == c and p == c for t, p in zip(y_true, y_pred))
        fp = sum(t != c and p == c for t, p in zip(y_true, y_pred))
        fn = sum(t == c and p != c for t, p in zip(y_true, y_pred))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s)) if f1s else 0.0


# ── 단일 모델 평가 ────────────────────────────────────────────────────────────

def eval_model(
    sess,
    input_name: str,
    img_size: int,
    norm: str,
    label_map: List[str],   # 모델 출력 index → label
    records: List[dict],
    desc: str = "",
) -> Tuple[float, float]:
    """macro-F1 + 평균 지연(ms) 반환."""
    y_true, y_pred = [], []
    times = []

    for rec in tqdm(records, desc=desc, leave=False, ncols=80):
        img_path = ROOT / rec["image_path"]
        x = load_image(img_path, img_size, norm)
        if x is None:
            continue
        t0 = time.perf_counter()
        logits = sess.run(None, {input_name: x})[0][0]  # [n_class]
        times.append((time.perf_counter() - t0) * 1000)

        pred_idx = int(np.argmax(logits))
        pred_label = label_map[pred_idx]

        # specialist는 binary → 3-class true_label 매핑
        true = rec["true_label"]
        # specm: authentic/manipulated 외 클래스는 평가에서 제외
        if label_map == CLASSES_2M and true not in CLASSES_2M:
            continue
        if label_map == CLASSES_2G and true not in CLASSES_2G:
            continue

        y_true.append(true)
        y_pred.append(pred_label)

    f1  = macro_f1(y_true, y_pred)
    lat = float(np.mean(times)) if times else 0.0
    return f1, lat


# ── 모델 설정 ────────────────────────────────────────────────────────────────

MODEL_CFG = {
    "mnv2": {
        "input_name": "image_01", "img_size": 224,
        "norm": "raw", "label_map": CLASSES_3,  # ONNX 내부 정규화 ([0,1] 입력)
        "fp32": ONNX_FP32 / "mnv2.onnx",
        "dyn":  ONNX_Q    / "mnv2_int8_dynamic.onnx",
        "sta":  ONNX_Q    / "mnv2_int8_static.onnx",
    },
    "specm": {
        "input_name": "image_01", "img_size": 224,
        "norm": "raw", "label_map": CLASSES_2M,  # ONNX 내부 정규화 ([0,1] 입력)
        "fp32": ONNX_FP32 / "specm.onnx",
        "dyn":  ONNX_Q    / "specm_int8_dynamic.onnx",
        "sta":  ONNX_Q    / "specm_int8_static.onnx",
    },
    "specg": {
        "input_name": "image_clip", "img_size": 256,
        "norm": "clip", "label_map": CLASSES_2G,
        "fp32": ONNX_FP32 / "specg.onnx",
        "dyn":  ONNX_Q    / "specg_int8_dynamic.onnx",
        "sta":  ONNX_Q    / "specg_int8_static.onnx",
    },
    "clip": {
        "input_name": "image_clip", "img_size": 256,
        "norm": "clip", "label_map": CLASSES_3,
        "fp32": ONNX_FP32 / "clip.onnx",
        "dyn":  ONNX_Q    / "clip_int8_dynamic.onnx",
        "sta":  ONNX_Q    / "clip_int8_static.onnx",
    },
}


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Phase 4.2 — 양자화 모델 정확도 평가 (macro-F1)")
    print("=" * 70)

    all_results = {}

    for key, cfg in MODEL_CFG.items():
        print(f"\n{'='*50}")
        print(f"  [{key.upper()}]")
        print(f"{'='*50}")

        variants = {
            "fp32": cfg["fp32"],
            "dyn":  cfg["dyn"],
            "sta":  cfg["sta"],
        }

        # 사용 가능한 variant만 세션 로드
        sessions = {}
        for vname, vpath in variants.items():
            if vpath.exists():
                try:
                    sessions[vname] = make_session(vpath)
                    print(f"  [{vname}] 로드: {vpath.name}")
                except Exception as e:
                    print(f"  [{vname}] 로드 실패: {e}")

        if not sessions:
            print("  사용 가능한 모델 없음")
            continue

        model_results = {vname: {} for vname in sessions}

        for ds in DATASETS:
            jsonl = get_eval_jsonl(key, ds)
            if jsonl is None or not jsonl.exists():
                continue
            records = [json.loads(l) for l in open(jsonl)]

            ds_row = {}
            for vname, sess in sessions.items():
                f1, lat = eval_model(
                    sess, cfg["input_name"], cfg["img_size"],
                    cfg["norm"], cfg["label_map"], records,
                    desc=f"{key}/{vname}/{ds}")
                ds_row[vname] = {"f1": round(f1, 4), "ms": round(lat, 1)}

            # 출력
            parts = []
            for vname, v in ds_row.items():
                parts.append(f"{vname}={100*v['f1']:.1f}%({v['ms']:.0f}ms)")
            print(f"  {ds:12s}: {' | '.join(parts)}")

            for vname, v in ds_row.items():
                model_results[vname][ds] = v

        # avg F1 계산
        for vname in sessions:
            f1s = [model_results[vname][ds]["f1"]
                   for ds in DATASETS if ds in model_results[vname]]
            if f1s:
                model_results[vname]["avg_f1"] = round(float(np.mean(f1s)), 4)

        # Delta 출력
        if "fp32" in model_results and model_results["fp32"]:
            for vname in ("dyn", "sta"):
                if vname not in model_results:
                    continue
                deltas = []
                for ds in DATASETS:
                    if ds in model_results["fp32"] and ds in model_results[vname]:
                        d = model_results[vname][ds]["f1"] - model_results["fp32"][ds]["f1"]
                        deltas.append(d)
                if deltas:
                    avg_d = float(np.mean(deltas))
                    print(f"  [{vname} vs fp32] avg Δ: {100*avg_d:+.2f}%p  "
                          f"avg_f1: {100*model_results[vname].get('avg_f1',0):.2f}%")

        all_results[key] = model_results

    # ── 전체 요약 테이블 ─────────────────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print("전체 요약 — avg macro-F1 (4-DS 평균)")
    print(f"{'='*70}")
    print(f"{'모델':<8} {'FP32':>8} {'Dyn INT8':>10} {'ΔDyn':>7} {'Sta INT8':>10} {'ΔSta':>7}")
    print("-" * 55)

    final = {}
    for key in MODEL_CFG:
        if key not in all_results:
            continue
        r = all_results[key]
        fp32 = r.get("fp32", {}).get("avg_f1", None)
        dyn  = r.get("dyn",  {}).get("avg_f1", None)
        sta  = r.get("sta",  {}).get("avg_f1", None)

        def fmt(v): return f"{100*v:.2f}%" if v is not None else "—"
        def fdelta(v, base):
            if v is None or base is None: return "—"
            return f"{100*(v-base):+.2f}%p"

        print(f"{key:<8} {fmt(fp32):>8} {fmt(dyn):>10} {fdelta(dyn,fp32):>7} "
              f"{fmt(sta):>10} {fdelta(sta,fp32):>7}")
        final[key] = {"fp32": fp32, "dyn": dyn, "sta": sta}

    # RPi5 최적 시나리오
    print(f"\n[RPi5 최적 배포 시나리오]")
    mnv2_best = max(
        (final.get("mnv2",{}).get("fp32") or 0),
        (final.get("mnv2",{}).get("dyn") or 0),
    )
    specm_sta = final.get("specm",{}).get("sta")
    if mnv2_best and specm_sta:
        print(f"  MNV2-FP32: {100*mnv2_best:.2f}%  "
              f"SpecM-Static: {100*specm_sta:.2f}%  "
              f"→ ICWMV 조합 시 추가 개선 기대")

    # 저장
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = OUT_DIR / f"quant_accuracy_{ts}.json"
    with open(out, "w") as f:
        json.dump({"timestamp": ts, "results": all_results}, f,
                  indent=2, ensure_ascii=False)
    print(f"\n결과 저장: {out.name}")


if __name__ == "__main__":
    main()
