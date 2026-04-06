#!/usr/bin/env python3
"""
ONNX 변환 + CPU 벤치마크 — Phase 4.1 Edge Deployment
=====================================================

대상 모델:
  1. MobileNetV2 dual-stream (5.77M)   — RPi5 1차 권고
  2. MobileCLIP-ft4          (99.4M)   — GPU 서버용
  3. Specialist-M v2          (7.66M)   — binary: auth vs manip
  4. Specialist-G             (35.91M)  — binary: auth vs aigen

출력:
  weights/onnx/{model}.onnx
  experiments/results/onnx_benchmark/onnx_benchmark_{ts}.json / .md

RPi5 추정:
  서버 CPU(EPYC) → RPi5(Cortex-A76 @2.4GHz) 스케일링 인자 = 4.0×
  (MobileNetV2 기준 EPYC ~35ms, RPi5 실측치 ~140ms 경험적 참조)

실행:
  .venv-qwen/bin/python experiments/run_onnx_export.py
  .venv-qwen/bin/python experiments/run_onnx_export.py --models mnv2 specm  # 선택 변환
  .venv-qwen/bin/python experiments/run_onnx_export.py --skip-export         # 벤치마크만
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import importlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT    = Path(__file__).resolve().parents[1]
ONNX_DIR = ROOT / "weights" / "onnx"
OUT_DIR  = ROOT / "experiments" / "results" / "onnx_benchmark"
ONNX_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# RPi5(Cortex-A76) 추정 스케일 팩터 (서버 EPYC CPU 대비)
RPI5_SCALE = 4.0
# 최대 RPi5 허용 지연 (ms)
RPI5_BUDGET_MS = 200.0

# ── 모델 정의 (train 스크립트에서 복사) ─────────────────────────────────────

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
CLIP_MEAN     = [0.48145466, 0.4578275,  0.40821073]
CLIP_STD      = [0.26862954, 0.26130258, 0.27577711]


def _normalize(x: torch.Tensor, mean, std) -> torch.Tensor:
    m = torch.tensor(mean, device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
    s = torch.tensor(std,  device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
    return (x - m) / s


# ── MobileNetV2 dual-stream ─────────────────────────────────────────────────

class _SRMNoise(nn.Module):
    def __init__(self):
        super().__init__()
        f1 = [[0,0,0,0,0],[0,-1,2,-1,0],[0,2,-4,2,0],[0,-1,2,-1,0],[0,0,0,0,0]]
        f2 = [[-1,2,-2,2,-1],[2,-6,8,-6,2],[-2,8,-12,8,-2],[2,-6,8,-6,2],[-1,2,-2,2,-1]]
        f3 = [[0,0,0,0,0],[0,0,0,0,0],[0,1,-2,1,0],[0,0,0,0,0],[0,0,0,0,0]]
        q  = torch.tensor([4.0,12.0,2.0]).view(3,1,1,1)
        k  = torch.tensor([f1,f2,f3], dtype=torch.float32).unsqueeze(1) / q
        self.register_buffer("kernels", k)
    def forward(self, x):
        g = x.mean(dim=1, keepdim=True)
        r = F.conv2d(g, self.kernels, padding=2).clamp(-2.,2.)
        return (r + 2.0) / 4.0


class _MNVBranch(nn.Module):
    def __init__(self, in_ch=3, pretrained=True):
        super().__init__()
        from torchvision import models as tv
        w = tv.MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None
        base = tv.mobilenet_v2(weights=w)
        if in_ch != 3:
            old = base.features[0][0]
            base.features[0][0] = nn.Conv2d(in_ch, old.out_channels,
                kernel_size=old.kernel_size, stride=old.stride,
                padding=old.padding, bias=False)
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)
    def forward(self, x):
        return self.pool(self.features(x)).flatten(1)


class MNV2DualStream(nn.Module):
    """MobileNetV2 dual-stream (RGB + SRM noise), 3-class."""
    def __init__(self, dropout=0.25):
        super().__init__()
        self.noise_extractor = _SRMNoise()
        self.rgb_branch   = _MNVBranch(3, pretrained=True)
        self.noise_branch = _MNVBranch(3, pretrained=True)
        self.head = nn.Sequential(
            nn.LayerNorm(2560),
            nn.Linear(2560, 512), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 3),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,3,H,W] in [0,1]
        rgb   = _normalize(x, IMAGENET_MEAN, IMAGENET_STD)
        noise = self.noise_extractor(x)
        noise = _normalize(noise, IMAGENET_MEAN, IMAGENET_STD)
        return self.head(torch.cat([self.rgb_branch(rgb),
                                    self.noise_branch(noise)], dim=1))


# ── Specialist-M v2 ─────────────────────────────────────────────────────────

class _SRMExt(nn.Module):
    def __init__(self):
        super().__init__()
        f1 = [[0,0,0,0,0],[0,-1,2,-1,0],[0,2,-4,2,0],[0,-1,2,-1,0],[0,0,0,0,0]]
        f2 = [[-1,2,-2,2,-1],[2,-6,8,-6,2],[-2,8,-12,8,-2],[2,-6,8,-6,2],[-1,2,-2,2,-1]]
        f3 = [[0,0,0,0,0],[0,0,0,0,0],[0,1,-2,1,0],[0,0,0,0,0],[0,0,0,0,0]]
        q  = torch.tensor([4.0,12.0,2.0]).view(3,1,1,1)
        k  = torch.tensor([f1,f2,f3], dtype=torch.float32).unsqueeze(1) / q
        self.register_buffer("kernels", k)
    def forward(self, x):
        g = x.mean(dim=1, keepdim=True)
        return ((F.conv2d(g, self.kernels, padding=2).clamp(-2.,2.) + 2.0) / 4.0)


class _DCTExt(nn.Module):
    def forward(self, x):
        g = x.mean(dim=1, keepdim=True)
        lp = F.avg_pool2d(g, kernel_size=8, stride=8)
        lp = F.interpolate(lp, size=g.shape[-2:], mode='nearest')
        return ((g - lp).clamp(-1.,1.) + 1.0) / 2.0


class SpecialistMv2(nn.Module):
    """3-stream (RGB+SRM+DCT), binary: auth vs manip."""
    def __init__(self, dropout=0.3):
        super().__init__()
        self.srm_ext = _SRMExt()
        self.dct_ext = _DCTExt()
        self.rgb_br  = _MNVBranch(3, pretrained=True)
        self.srm_br  = _MNVBranch(3, pretrained=False)
        self.dct_br  = _MNVBranch(1, pretrained=False)
        self.head = nn.Sequential(
            nn.LayerNorm(3840),
            nn.Linear(3840, 256), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,3,H,W] in [0,1]
        rgb = _normalize(x, IMAGENET_MEAN, IMAGENET_STD)
        srm = _normalize(self.srm_ext(x), [.5,.5,.5], [.5,.5,.5])
        dct = _normalize(self.dct_ext(x), [.5], [.5])
        return self.head(torch.cat([self.rgb_br(rgb),
                                    self.srm_br(srm),
                                    self.dct_br(dct)], dim=1))


# ── Specialist-G ──────────────────────────────────────────────────────────

class _PiDResidual(nn.Module):
    def __init__(self, n_bits=4):
        super().__init__()
        M = torch.tensor([[.299,.587,.114],[-.169,-.331,.5],[.5,-.419,-.081]])
        self.register_buffer("M",     M)
        self.register_buffer("M_inv", torch.inverse(M))
        self.n_bins = 2 ** n_bits
    def forward(self, x):
        B,C,H,W = x.shape
        flat = x.permute(0,2,3,1).reshape(-1,3)
        yuv  = flat @ self.M.T
        yuv_q = torch.round(yuv * self.n_bins) / self.n_bins
        rgb_q = yuv_q @ self.M_inv.T
        res   = (flat - rgb_q).abs().reshape(B,H,W,3).permute(0,3,1,2)
        return res.mean(dim=1, keepdim=True).clamp(0,1)


class SpecialistGONNX(nn.Module):
    """
    SpecialistG ONNX용 래퍼: 단일 입력 (CLIP 정규화).
    x_raw는 x에서 자동 역정규화.
    binary: auth vs aigen.
    """
    def __init__(self, clip_encoder: nn.Module, dropout=0.3):
        super().__init__()
        self.clip_encoder = clip_encoder
        self.pid = _PiDResidual(n_bits=4)
        self.pid_branch = nn.Sequential(
            nn.Conv2d(1,16,3,stride=2,padding=1), nn.ReLU(),
            nn.Conv2d(16,32,3,stride=2,padding=1), nn.ReLU(),
            nn.Conv2d(32,64,3,stride=2,padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(576),
            nn.Linear(576, 128), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2),
        )
        # CLIP 역정규화용 상수
        self.register_buffer("clip_mean", torch.tensor(CLIP_MEAN).view(1,3,1,1))
        self.register_buffer("clip_std",  torch.tensor(CLIP_STD).view(1,3,1,1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,3,H,W] CLIP 정규화
        clip_feat = self.clip_encoder(x)
        if isinstance(clip_feat, tuple):
            clip_feat = clip_feat[0]
        if clip_feat.dim() > 2:
            clip_feat = clip_feat.flatten(1)
        # 역정규화 → PiD
        x_raw = (x * self.clip_std + self.clip_mean).clamp(0, 1)
        pid_feat = self.pid_branch(self.pid(x_raw))
        return self.head(torch.cat([clip_feat, pid_feat], dim=1))


# ── MobileCLIP Forensics (ft4) ──────────────────────────────────────────────

class MobileCLIPForensicsONNX(nn.Module):
    """MobileCLIP-ft4 ONNX용: CLIP 정규화 입력 → 3-class logits."""
    def __init__(self, encoder: nn.Module, embed_dim=512, dropout=0.3):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 256), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 3),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(x)
        if isinstance(feat, tuple): feat = feat[0]
        if feat.dim() > 2: feat = feat.flatten(1)
        return self.head(feat)


# ── 모델 로더 ────────────────────────────────────────────────────────────────

def load_mnv2(ckpt: Path) -> nn.Module:
    # 원본 학습 스크립트에서 클래스 임포트 (키 이름 일치 보장)
    sys.path.insert(0, str(ROOT / "experiments"))
    import train_mobilenetv2_dualstream as _m
    importlib.reload(_m)
    model = _m.DualStreamMobileNetV2()
    sd = torch.load(ckpt, map_location="cpu")
    sd = sd.get("model_state_dict", sd)
    model.load_state_dict(sd, strict=True)
    return model.eval()


def load_specm(ckpt: Path) -> nn.Module:
    sys.path.insert(0, str(ROOT / "experiments"))
    import train_specialist_m_v2 as _m
    importlib.reload(_m)
    model = _m.SpecialistM()
    sd = torch.load(ckpt, map_location="cpu")
    sd = sd.get("model_state_dict", sd)
    model.load_state_dict(sd, strict=True)
    return model.eval()


def load_specg(ckpt: Path) -> Optional[nn.Module]:
    """SpecialistG: 원본 학습 스크립트 모델 + ONNX 단일입력 래퍼."""
    try:
        sys.path.insert(0, str(ROOT / "experiments"))
        import train_specialist_g as _g
        importlib.reload(_g)

        ft4_ckpt = ROOT / "weights" / "mobileclip_forensics" / "mobileclip_s2_forensics_ft4.pth"
        model_orig = _g.SpecialistG(clip_ckpt=ft4_ckpt, use_pid=True)
        sd = torch.load(ckpt, map_location="cpu")
        sd = sd.get("model_state_dict", sd)
        model_orig.load_state_dict(sd, strict=False)
        model_orig.eval()

        # ONNX 단일입력 래퍼로 감싸기
        # (SpecialistG.forward(x, x_raw=None) → x_raw 자동 역정규화)
        class _Wrapper(nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.inner(x, x_raw=None)

        return _Wrapper(model_orig).eval()
    except Exception as e:
        print(f"    [WARN] SpecialistG 로드 실패: {e}")
        return None


def load_clip_forensics(ckpt: Path) -> Optional[nn.Module]:
    """MobileCLIP ft4: 원본 학습 스크립트에서 모델 재구성."""
    try:
        import open_clip
        sys.path.insert(0, str(ROOT / "experiments"))
        import finetune_mobileclip as _fc
        importlib.reload(_fc)

        clip_model, _, _ = open_clip.create_model_and_transforms(
            "MobileCLIP-S2", pretrained="datacompdr")
        model = _fc.MobileCLIPForensics(clip_model, finetune_blocks=4)
        sd = torch.load(ckpt, map_location="cpu")
        sd = sd.get("model_state_dict", sd)
        model.load_state_dict(sd, strict=False)
        return model.eval()
    except Exception as e:
        print(f"    [WARN] MobileCLIP ft4 로드 실패: {e}")
        return None


# ── ONNX 변환 ────────────────────────────────────────────────────────────────

def export_onnx(model: nn.Module, dummy: torch.Tensor,
                out_path: Path, input_name: str,
                opset: int = 14) -> bool:
    """
    TorchScript 기반 ONNX 내보내기 (opset 14).
    torch.export 기반(opset 17+)은 onnxruntime quantization과 호환 문제 있음.
    """
    try:
        import onnx
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # dynamo=False: PyTorch 2.x에서 구형 TorchScript 기반 exporter 사용
            # (onnxruntime quantization과의 호환성을 위해 필수)
            torch.onnx.export(
                model, dummy,
                str(out_path),
                input_names=[input_name],
                output_names=["logits"],
                dynamic_axes={input_name: {0: "batch"}, "logits": {0: "batch"}},
                opset_version=opset,
                do_constant_folding=True,
                export_params=True,
                dynamo=False,
            )
        onnx.checker.check_model(str(out_path))
        return True
    except Exception as e:
        print(f"      [ERROR] ONNX 변환 실패: {e}")
        return False


# ── CPU 벤치마크 ──────────────────────────────────────────────────────────────

def benchmark_onnx(onnx_path: Path, dummy_np: np.ndarray,
                   input_name: str,
                   n_warmup: int = 5, n_runs: int = 50) -> Dict:
    try:
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1   # 단일 코어 (RPi5 조건 시뮬레이션)
        opts.inter_op_num_threads = 1
        sess = ort.InferenceSession(str(onnx_path), opts,
                                    providers=["CPUExecutionProvider"])
        feed = {input_name: dummy_np}

        # warmup
        for _ in range(n_warmup):
            sess.run(None, feed)

        # 측정
        t0 = time.perf_counter()
        for _ in range(n_runs):
            sess.run(None, feed)
        elapsed = (time.perf_counter() - t0) / n_runs * 1000  # ms

        return {
            "server_cpu_ms":       round(elapsed, 2),
            "rpi5_estimated_ms":   round(elapsed * RPI5_SCALE, 2),
            "rpi5_budget_ok":      elapsed * RPI5_SCALE <= RPI5_BUDGET_MS,
            "n_runs":              n_runs,
            "threads":             1,
        }
    except Exception as e:
        return {"error": str(e)}


def get_onnx_size_mb(path: Path) -> float:
    """ONNX 파일 + 외부 데이터 파일(.onnx.data) 합계 크기 반환."""
    total = 0
    if path.exists():
        total += path.stat().st_size
    data_file = path.with_suffix(".onnx.data")
    if data_file.exists():
        total += data_file.stat().st_size
    return round(total / 1024 / 1024, 1)


# ── 메인 ─────────────────────────────────────────────────────────────────────

MODEL_SPECS = {
    "mnv2": {
        "name":       "MobileNetV2 dual-stream",
        "ckpt":       ROOT / "weights" / "mobilenetv2_dualstream" / "mobilenetv2_dualstream_best.pth",
        "loader":     load_mnv2,
        "input_size": (1, 3, 224, 224),
        "input_norm": "[0,1]",
        "input_name": "image_01",
        "classes":    3,
        "params_m":   5.77,
    },
    "specm": {
        "name":       "Specialist-M v2",
        "ckpt":       ROOT / "weights" / "specialist_m_v2" / "specialist_m_v2_best.pth",
        "loader":     load_specm,
        "input_size": (1, 3, 224, 224),
        "input_norm": "[0,1]",
        "input_name": "image_01",
        "classes":    2,
        "params_m":   7.66,
    },
    "specg": {
        "name":       "Specialist-G",
        "ckpt":       ROOT / "weights" / "specialist_g" / "specialist_g_best.pth",
        "loader":     load_specg,
        "input_size": (1, 3, 256, 256),
        "input_norm": "CLIP",
        "input_name": "image_clip",
        "classes":    2,
        "params_m":   35.91,
    },
    "clip": {
        "name":       "MobileCLIP-ft4",
        "ckpt":       ROOT / "weights" / "mobileclip_forensics" / "mobileclip_s2_forensics_ft4.pth",
        "loader":     load_clip_forensics,
        "input_size": (1, 3, 256, 256),
        "input_norm": "CLIP",
        "input_name": "image_clip",
        "classes":    3,
        "params_m":   99.4,
    },
}


def run_model(key: str, spec: dict, skip_export: bool) -> dict:
    print(f"\n{'='*55}")
    print(f"  [{key.upper()}] {spec['name']}")
    print(f"{'='*55}")

    onnx_path = ONNX_DIR / f"{key}.onnx"
    result = {
        "key":         key,
        "name":        spec["name"],
        "params_m":    spec["params_m"],
        "classes":     spec["classes"],
        "input_norm":  spec["input_norm"],
        "onnx_path":   str(onnx_path),
        "export_ok":   False,
        "onnx_mb":     0.0,
    }

    # ── 모델 로드 ──
    print(f"  체크포인트: {spec['ckpt'].name}")
    if not spec["ckpt"].exists():
        print(f"  [SKIP] 체크포인트 없음")
        result["skip_reason"] = "checkpoint not found"
        return result

    model = spec["loader"](spec["ckpt"])
    if model is None:
        print(f"  [SKIP] 모델 로드 실패")
        result["skip_reason"] = "model load failed"
        return result

    # 파라미터 수 실측
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    result["params_m_actual"] = round(n_params, 2)
    print(f"  파라미터 수: {n_params:.2f}M")

    # ── ONNX 변환 ──
    if not skip_export or not onnx_path.exists():
        print(f"  ONNX 변환 중... (opset=17)")
        B, C, H, W = spec["input_size"]
        dummy = torch.zeros(B, C, H, W, dtype=torch.float32)
        ok = export_onnx(model, dummy, onnx_path, spec["input_name"])
        result["export_ok"]  = ok
        result["onnx_mb"]    = get_onnx_size_mb(onnx_path)
        if ok:
            print(f"  → {onnx_path.name}  ({result['onnx_mb']:.1f} MB)")
        else:
            return result
    else:
        result["export_ok"] = True
        result["onnx_mb"]   = get_onnx_size_mb(onnx_path)
        print(f"  ONNX 재사용: {onnx_path.name}  ({result['onnx_mb']:.1f} MB)")

    # ── CPU 벤치마크 ──
    print(f"  CPU 벤치마크 (1 thread, batch=1, n=50)...")
    B, C, H, W = spec["input_size"]
    dummy_np = np.random.rand(B, C, H, W).astype(np.float32)
    bench = benchmark_onnx(onnx_path, dummy_np, spec["input_name"])
    result.update(bench)

    if "error" not in bench:
        budget_str = "✓ OK" if bench["rpi5_budget_ok"] else "✗ OVER"
        print(f"  서버 CPU(1T):  {bench['server_cpu_ms']:.1f} ms")
        print(f"  RPi5 추정:     {bench['rpi5_estimated_ms']:.1f} ms  [{budget_str}]")
    else:
        print(f"  벤치마크 오류: {bench['error']}")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+",
                        choices=list(MODEL_SPECS),
                        default=list(MODEL_SPECS),
                        help="변환할 모델 목록 (기본: 전체)")
    parser.add_argument("--skip-export", action="store_true",
                        help="ONNX 변환 생략, 기존 파일로 벤치마크만 실행")
    args = parser.parse_args()

    print("=" * 65)
    print("ONNX 변환 + CPU 벤치마크  — Phase 4.1 Edge Deployment")
    print("=" * 65)
    print(f"  대상 모델: {args.models}")
    print(f"  ONNX 저장: {ONNX_DIR}")
    print(f"  RPi5 스케일 팩터: {RPI5_SCALE}×  (예산: {RPI5_BUDGET_MS}ms)")

    all_results = []
    for key in args.models:
        r = run_model(key, MODEL_SPECS[key], args.skip_export)
        all_results.append(r)

    # ── 요약 테이블 ─────────────────────────────────────────────────────────
    print(f"\n\n{'='*65}")
    print("ONNX 변환 + 벤치마크 요약")
    print(f"{'='*65}")
    hdr = f"{'모델':<22} {'파라미터':>8} {'ONNX MB':>8} {'서버CPU(ms)':>12} {'RPi5추정(ms)':>13} {'예산OK':>8}"
    print(hdr)
    print("-" * 75)

    total_rpi5_4model = 0.0
    total_rpi5_rpi_only = 0.0  # mnv2+specm+specg only (RPi5 배포 대상)

    for r in all_results:
        params = f"{r.get('params_m_actual', r['params_m']):.2f}M"
        onnx_mb = f"{r.get('onnx_mb', 0):.1f}"
        if "error" in r:
            cpu_ms = rpi5_ms = "ERR"
            ok_str = "—"
        elif "skip_reason" in r:
            cpu_ms = rpi5_ms = "SKIP"
            ok_str = "—"
        elif "server_cpu_ms" not in r:
            cpu_ms = rpi5_ms = "—"
            ok_str = "—"
        else:
            cpu_ms = f"{r['server_cpu_ms']:.1f}"
            rpi5_ms = f"{r['rpi5_estimated_ms']:.1f}"
            ok_str = "✓" if r.get("rpi5_budget_ok") else "✗"
            total_rpi5_4model += r.get("rpi5_estimated_ms", 0)
            if r["key"] in ("mnv2", "specm", "specg"):
                total_rpi5_rpi_only += r.get("rpi5_estimated_ms", 0)

        print(f"{r['name']:<22} {params:>8} {onnx_mb:>8} {cpu_ms:>12} {rpi5_ms:>13} {ok_str:>8}")

    print("-" * 75)
    if total_rpi5_4model > 0:
        print(f"  4-model 순차 합계 (RPi5 추정): {total_rpi5_4model:.0f} ms")
        print(f"  RPi5 배포 3-model (MNV2+SpecM+SpecG): {total_rpi5_rpi_only:.0f} ms")
        print(f"  ※ CLIP(399MB)은 서버 전용 — RPi5 배포 제외")

    # ── 저장 ────────────────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_json = OUT_DIR / f"onnx_benchmark_{ts}.json"
    out_md   = OUT_DIR / f"onnx_benchmark_{ts}.md"

    summary = {
        "timestamp":          ts,
        "rpi5_scale_factor":  RPI5_SCALE,
        "rpi5_budget_ms":     RPI5_BUDGET_MS,
        "total_rpi5_4model":  round(total_rpi5_4model, 1),
        "total_rpi5_rpi_only": round(total_rpi5_rpi_only, 1),
        "models": all_results,
    }
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Markdown 리포트
    md_lines = [
        "# ONNX 변환 + RPi5 Edge Deployment 벤치마크\n",
        f"생성일: {ts}  |  RPi5 스케일 팩터: {RPI5_SCALE}×  |  예산: {RPI5_BUDGET_MS}ms\n",
        "\n## 모델별 결과\n",
        "| 모델 | 파라미터 | ONNX MB | 서버CPU(ms) | RPi5추정(ms) | 예산OK |",
        "|------|---------|---------|------------|-------------|--------|",
    ]
    for r in all_results:
        params = f"{r.get('params_m_actual', r['params_m']):.2f}M"
        onnx_mb = f"{r.get('onnx_mb', 0):.1f}"
        cpu_ms = f"{r.get('server_cpu_ms', '—')}"
        rpi5_ms = f"{r.get('rpi5_estimated_ms', '—')}"
        ok_str = "✓" if r.get("rpi5_budget_ok") else ("✗" if "server_cpu_ms" in r else "—")
        md_lines.append(f"| {r['name']} | {params} | {onnx_mb} | {cpu_ms} | {rpi5_ms} | {ok_str} |")

    md_lines += [
        f"\n## 전체 추정 지연",
        f"- **4-model 순차 합계 (RPi5)**: {total_rpi5_4model:.0f} ms",
        f"- **RPi5 배포 3-model** (MNV2+SpecM+SpecG): {total_rpi5_rpi_only:.0f} ms",
        f"- MobileCLIP(~400MB)은 서버 전용 배포 권고",
        f"\n## 비고",
        f"- 스케일 팩터 {RPI5_SCALE}×: EPYC 서버 → RPi5(Cortex-A76 @2.4GHz) 경험적 비율",
        f"- ONNXRuntime 1-thread CPU, batch=1, n=50 평균",
        f"- RPi5 XNNPACK 적용 시 ~1.5× 가속 추가 가능 (onnxruntime-xnnpack)",
    ]

    with open(out_md, "w") as f:
        f.write("\n".join(md_lines))

    print(f"\n  결과 저장: {out_json.name}")
    print(f"  리포트:   {out_md.name}")


if __name__ == "__main__":
    main()
