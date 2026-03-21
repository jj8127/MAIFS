#!/usr/bin/env python3
"""
Phase 4.5 — WildRF Benchmark
==============================================

SHIELD 배포 모델 (MNV2+SpecM-v4 ICWMV) vs Tiny-LaDeDa (SOTA baseline)
WildRF test set 3개 플랫폼 (reddit / twitter / facebook) 평가.

WildRF 구조 (datasets/WildRF/ 아래에 위치 필요):
  WildRF/test/
    reddit/   0_real/  1_fake/
    twitter/  0_real/  1_fake/
    facebook/ 0_real/  1_fake/

3-class → binary 매핑:
  authentic  → real  (label=0)
  manipulated / ai_generated → fake (label=1)

비교 기준 (논문 보고값):
  Tiny-LaDeDa (WildRF-trained):  mAP=93.7%  (Cavia et al. 2024, arXiv:2406.09398)

실행:
  .venv-qwen/bin/python experiments/run_wildrf_benchmark.py
  .venv-qwen/bin/python experiments/run_wildrf_benchmark.py --wildrf /path/to/WildRF
  .venv-qwen/bin/python experiments/run_wildrf_benchmark.py --skip-tiny   # ICWMV only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

ROOT    = Path(__file__).resolve().parents[1]
ONNX_Q  = ROOT / "weights" / "onnx_quant"
OUT_DIR = ROOT / "experiments" / "results" / "wildrf_benchmark"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PLATFORMS = ["reddit", "twitter", "facebook"]
CLASSES_3 = ["authentic", "manipulated", "ai_generated"]
CLASSES_2 = ["authentic", "manipulated"]


# ── 전처리 ────────────────────────────────────────────────────────────────────

def load_image_onnx(path: Path, size: int = 224) -> np.ndarray:
    """[0,1] float32 NCHW — ONNX 모델 입력 형식."""
    img = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    x   = np.array(img, dtype=np.float32) / 255.0
    return x.transpose(2, 0, 1)[np.newaxis]   # [1,3,H,W]


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


# ── ICWMV 융합 ────────────────────────────────────────────────────────────────

def icwmv_fuse(mnv2_p: np.ndarray, specm_p: np.ndarray, w: float = 1.0) -> np.ndarray:
    """MNV2(3-class) + SpecM(binary auth/manip) → fused 3-class probs."""
    s_auth  = (mnv2_p[0] + w * specm_p[0]) / (1.0 + w)
    s_manip = (mnv2_p[1] + w * specm_p[1]) / (1.0 + w)
    s_aigen = mnv2_p[2]
    raw     = np.array([s_auth, s_manip, s_aigen], dtype=np.float32)
    return raw / raw.sum()


# ── 평가 지표 ─────────────────────────────────────────────────────────────────

def compute_metrics(y_true: list[int], y_score: list[float]) -> dict:
    """binary 평가: acc, mAP(AP), r_acc(real), f_acc(fake), AUC."""
    from sklearn.metrics import (accuracy_score, average_precision_score,
                                  roc_auc_score)
    y_t  = np.array(y_true)
    y_s  = np.array(y_score)
    y_p  = (y_s > 0.5).astype(int)

    acc    = float(accuracy_score(y_t, y_p))
    ap     = float(average_precision_score(y_t, y_s))
    r_acc  = float(accuracy_score(y_t[y_t == 0], y_p[y_t == 0])) if (y_t == 0).any() else 0.0
    f_acc  = float(accuracy_score(y_t[y_t == 1], y_p[y_t == 1])) if (y_t == 1).any() else 0.0
    try:
        auc = float(roc_auc_score(y_t, y_s))
    except Exception:
        auc = 0.0
    return {"acc": acc, "ap": ap, "r_acc": r_acc, "f_acc": f_acc, "auc": auc,
            "n_real": int((y_t == 0).sum()), "n_fake": int((y_t == 1).sum())}


# ── ICWMV 평가 ───────────────────────────────────────────────────────────────

def eval_icwmv(wildrf_dir: Path, w_spec: float, threads: int) -> dict:
    import onnxruntime as ort

    mnv2_path  = ONNX_Q / "mnv2_int8_dynamic.onnx"
    specm_path = ONNX_Q / "specm_v4_int8_dynamic.onnx"
    for p in [mnv2_path, specm_path]:
        if not p.exists():
            print(f"[ERROR] ONNX 파일 없음: {p}")
            sys.exit(1)

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.log_severity_level   = 3
    mnv2_sess  = ort.InferenceSession(str(mnv2_path),  opts, providers=["CPUExecutionProvider"])
    specm_sess = ort.InferenceSession(str(specm_path), opts, providers=["CPUExecutionProvider"])
    print(f"[ICWMV] MNV2 + SpecM-v4 로드 완료 (threads={threads}, w_spec={w_spec})")

    results = {}
    test_dir = wildrf_dir / "test"

    for platform in PLATFORMS:
        plat_dir = test_dir / platform
        if not plat_dir.exists():
            print(f"  [SKIP] {platform} — 디렉토리 없음: {plat_dir}")
            continue

        y_true, y_score = [], []
        n_images = 0
        t0 = time.perf_counter()

        for label_dir, label in [("0_real", 0), ("1_fake", 1)]:
            img_dir = plat_dir / label_dir
            if not img_dir.exists():
                continue
            exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
            img_files = [f for f in img_dir.iterdir() if f.suffix.lower() in exts]

            for img_path in img_files:
                try:
                    x = load_image_onnx(img_path)
                    mnv2_logits  = mnv2_sess.run(None,  {"image_01": x})[0][0]
                    specm_logits = specm_sess.run(None, {"image_01": x})[0][0]
                    mnv2_p  = softmax(mnv2_logits)
                    specm_p = softmax(specm_logits)
                    fused   = icwmv_fuse(mnv2_p, specm_p, w_spec)
                    # binary 점수: P(fake) = P(manip) + P(aigen)
                    fake_score = float(fused[1] + fused[2])
                    y_true.append(label)
                    y_score.append(fake_score)
                    n_images += 1
                except Exception as e:
                    print(f"    [WARN] {img_path.name}: {e}")

        elapsed = time.perf_counter() - t0
        if not y_true:
            print(f"  [{platform}] 이미지 없음, 건너뜀")
            continue

        metrics = compute_metrics(y_true, y_score)
        metrics["n_images"] = n_images
        metrics["elapsed_s"] = round(elapsed, 2)
        metrics["ms_per_image"] = round(elapsed / n_images * 1000, 1) if n_images else 0
        results[platform] = metrics

        print(f"  [{platform:8s}]  n={n_images:4d} | "
              f"acc={metrics['acc']*100:.1f}%  mAP={metrics['ap']*100:.1f}%  "
              f"AUC={metrics['auc']*100:.1f}%  "
              f"r_acc={metrics['r_acc']*100:.1f}%  f_acc={metrics['f_acc']*100:.1f}%  "
              f"({metrics['ms_per_image']:.0f}ms/img)")

    if results:
        avg_ap  = np.mean([v["ap"]  for v in results.values()])
        avg_acc = np.mean([v["acc"] for v in results.values()])
        avg_auc = np.mean([v["auc"] for v in results.values()])
        results["_avg"] = {"ap": float(avg_ap), "acc": float(avg_acc), "auc": float(avg_auc)}
        print(f"\n  [ICWMV 평균]  mAP={avg_ap*100:.2f}%  acc={avg_acc*100:.2f}%  AUC={avg_auc*100:.2f}%")
        print(f"  [비교] Tiny-LaDeDa(WildRF-trained) mAP=93.7%")

    return results


# ── Tiny-LaDeDa 평가 ──────────────────────────────────────────────────────────

def eval_tiny_ladeda(wildrf_dir: Path) -> dict:
    import torch
    import torchvision.transforms as T
    sys.path.insert(0, str(ROOT / "TinyLaDeDa-main"))
    from networks.Tiny_LaDeDa import tiny_ladeda

    ckpt_path = ROOT / "TinyLaDeDa-main" / "weights" / "Tiny_LaDeDa" / "WildRF_Tiny_LaDeDa.pth"
    if not ckpt_path.exists():
        print(f"[WARN] Tiny-LaDeDa 가중치 없음: {ckpt_path} — 건너뜀")
        return {}

    model = tiny_ladeda(preprocess_type="NPR")
    sd = torch.load(ckpt_path, map_location="cpu")
    # 다양한 저장 형식 대응
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    elif isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    model.load_state_dict(sd, strict=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    print(f"[Tiny-LaDeDa] WildRF 가중치 로드 완료 (device={device})")

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
    ])

    results = {}
    test_dir = wildrf_dir / "test"

    for platform in PLATFORMS:
        plat_dir = test_dir / platform
        if not plat_dir.exists():
            continue

        y_true, y_score = [], []
        for label_dir, label in [("0_real", 0), ("1_fake", 1)]:
            img_dir = plat_dir / label_dir
            if not img_dir.exists():
                continue
            exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
            for img_path in [f for f in img_dir.iterdir() if f.suffix.lower() in exts]:
                try:
                    img = Image.open(img_path).convert("RGB")
                    x   = transform(img).unsqueeze(0).to(device)
                    with torch.no_grad():
                        score = torch.sigmoid(model(x)).item()
                    y_true.append(label)
                    y_score.append(score)
                except Exception as e:
                    print(f"    [WARN] {img_path.name}: {e}")

        if not y_true:
            continue

        metrics = compute_metrics(y_true, y_score)
        metrics["n_images"] = len(y_true)
        results[platform] = metrics
        print(f"  [{platform:8s}]  n={len(y_true):4d} | "
              f"acc={metrics['acc']*100:.1f}%  mAP={metrics['ap']*100:.1f}%  "
              f"AUC={metrics['auc']*100:.1f}%  "
              f"r_acc={metrics['r_acc']*100:.1f}%  f_acc={metrics['f_acc']*100:.1f}%")

    if results:
        avg_ap  = np.mean([v["ap"]  for v in results.values()])
        avg_acc = np.mean([v["acc"] for v in results.values()])
        results["_avg"] = {"ap": float(avg_ap), "acc": float(avg_acc)}
        print(f"\n  [Tiny-LaDeDa 평균]  mAP={avg_ap*100:.2f}%  acc={avg_acc*100:.2f}%")

    return results


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 4.5 WildRF Benchmark")
    parser.add_argument("--wildrf",    type=Path, default=ROOT / "datasets" / "WildRF",
                        help="WildRF 데이터셋 루트 경로")
    parser.add_argument("--w-spec",   type=float, default=1.0,
                        help="ICWMV SpecM 가중치 (default: 1.0)")
    parser.add_argument("--threads",  type=int,   default=4,
                        help="ONNX intra-op 스레드 수 (default: 4)")
    parser.add_argument("--skip-tiny", action="store_true",
                        help="Tiny-LaDeDa 평가 건너뜀")
    args = parser.parse_args()

    if not (args.wildrf / "test").exists():
        print(f"[ERROR] WildRF 데이터셋을 찾을 수 없습니다: {args.wildrf}")
        print()
        print("  다운로드 방법:")
        print("  1. https://drive.google.com/file/d/1A0xoL44Yg68ixd-FuIJn2VC4vdZ6M2gn/view")
        print("     에서 WildRF.zip 다운로드")
        print(f"  2. {ROOT}/datasets/ 에 압축 해제 → datasets/WildRF/test/reddit|twitter|facebook")
        print()
        print("  또는 --wildrf 옵션으로 경로 직접 지정:")
        print("  .venv-qwen/bin/python experiments/run_wildrf_benchmark.py --wildrf /path/to/WildRF")
        sys.exit(1)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUT_DIR / f"wildrf_benchmark_{ts}.json"

    print("=" * 60)
    print("Phase 4.5 WildRF Benchmark")
    print(f"  WildRF 경로: {args.wildrf}")
    print(f"  w_spec={args.w_spec}, threads={args.threads}")
    print("=" * 60)

    output = {
        "timestamp": ts,
        "wildrf_root": str(args.wildrf),
        "w_spec": args.w_spec,
        "threads": args.threads,
        "sota_reference": {
            "Tiny-LaDeDa (WildRF-trained)": {"mAP": 0.937, "source": "arXiv:2406.09398"},
        },
    }

    # ICWMV 평가
    print("\n[1/2] SHIELD ICWMV (MNV2 + SpecM-v4) 평가")
    output["icwmv"] = eval_icwmv(args.wildrf, args.w_spec, args.threads)

    # Tiny-LaDeDa 평가
    if not args.skip_tiny:
        print("\n[2/2] Tiny-LaDeDa (WildRF-trained) 평가")
        output["tiny_ladeda"] = eval_tiny_ladeda(args.wildrf)
    else:
        print("\n[2/2] Tiny-LaDeDa 평가 건너뜀 (--skip-tiny)")

    # 최종 비교 출력
    print("\n" + "=" * 60)
    print("최종 비교 (mAP)")
    print("-" * 60)
    if "icwmv" in output and "_avg" in output["icwmv"]:
        icwmv_map = output["icwmv"]["_avg"]["ap"] * 100
        print(f"  SHIELD ICWMV (MNV2+SpecM-v4):         {icwmv_map:.2f}%")
    if "tiny_ladeda" in output and "_avg" in output.get("tiny_ladeda", {}):
        tl_map = output["tiny_ladeda"]["_avg"]["ap"] * 100
        print(f"  Tiny-LaDeDa (WildRF-trained, 실측):   {tl_map:.2f}%")
    print(f"  Tiny-LaDeDa (WildRF-trained, 논문값): 93.70%")
    print("=" * 60)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장: {out_path}")


if __name__ == "__main__":
    main()
