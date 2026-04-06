#!/usr/bin/env python3
"""
Phase 2.7 — Backbone Latency / Size Benchmark
=============================================

측정 대상:
  - ForMa
  - MobileCLIP-ft4
  - Tiny-LaDeDa
  - MobileNetV2 dual-stream
  - MobileNetV2 proxy (optional legacy reference)

출력:
  - experiments/results/backbone_benchmark/backbone_benchmark_<ts>.json
  - experiments/results/backbone_benchmark/backbone_benchmark_<ts>.md

기준:
  - batch=1
  - 이미지 로드 / 전처리 / host→device 복사 제외, 순수 model forward latency
  - GPU는 synchronize 포함
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "experiments" / "results" / "backbone_benchmark"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FORMA_DIR = ROOT / "ForMa-main"
TINYLA_DIR = ROOT / "TinyLaDeDa-main"

sys.path.insert(0, str(FORMA_DIR))
sys.path.insert(0, str(FORMA_DIR / "models"))
sys.path.insert(0, str(TINYLA_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from finetune_mobileclip import MobileCLIPForensics  # noqa: E402
from train_mobilenetv2_dualstream import DualStreamMobileNetV2  # noqa: E402


CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

FORMA_TRANSFORM = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

MOBILECLIP_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
])

TINYLA_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

MOBILENET_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

MOBILENET_DUALSTREAM_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])


@dataclass
class ModelArtifacts:
    model: nn.Module
    checkpoint_path: Optional[Path]
    input_tensor: torch.Tensor
    input_shape: List[int]
    note: str


def _run_cmd(cmd: List[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True).strip()
    except Exception:
        return ""


def hardware_info() -> Dict[str, object]:
    cpu_name = ""
    for line in _run_cmd(["bash", "-lc", "lscpu | sed -n '1,20p'"]).splitlines():
        if line.startswith("Model name:"):
            cpu_name = line.split(":", 1)[1].strip()
            break

    gpu_name = _run_cmd(
        ["bash", "-lc", "nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1"]
    )
    gpu_mem = _run_cmd(
        ["bash", "-lc", "nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -n 1"]
    )
    gpu_driver = _run_cmd(
        ["bash", "-lc", "nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1"]
    )

    return {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cpu_model": cpu_name,
        "cpu_threads_default": torch.get_num_threads(),
        "cpu_count": os.cpu_count(),
        "gpu_model": gpu_name or None,
        "gpu_memory_total": gpu_mem or None,
        "gpu_driver": gpu_driver or None,
    }


def resolve_forma_weight_path() -> Optional[Path]:
    candidates = [
        FORMA_DIR / "weights" / "ForMa_weights.pth",
        ROOT / "ForMa_weights.pth",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def sample_image_path() -> Path:
    jsonl = ROOT / "experiments" / "results" / "phase2_patha_scale500_gain_predictor" / "patha_agent_outputs_20260304_080157.jsonl"
    with open(jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            img_path = ROOT / rec["image_path"]
            if img_path.exists():
                return img_path
    raise FileNotFoundError("대표 샘플 이미지를 찾지 못했습니다.")


def load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def patch_forma_cuda_backend(model: nn.Module) -> int:
    from functools import partial

    patched = 0
    for module in model.modules():
        if hasattr(module, "forward_corev2") and hasattr(module, "disable_force32"):
            module.forward_core = partial(
                module.forward_corev2,
                force_fp32=(not module.disable_force32),
                selective_scan_backend="mamba",
            )
            patched += 1
    return patched


def load_forma(device: torch.device, image: Image.Image) -> ModelArtifacts:
    from models.vmamba_pixelshuf_modals import Forensic_Vmamba

    model = Forensic_Vmamba().to(device)
    ckpt_path = resolve_forma_weight_path()
    if ckpt_path is None:
        raise FileNotFoundError("ForMa_weights.pth not found")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    note = "ForMa 37.3M, VMamba splicing detector"
    if device.type == "cuda":
        patched = patch_forma_cuda_backend(model)
        note += f", cuda_mamba_backend={patched}"
    model.eval()
    inp = FORMA_TRANSFORM(image).unsqueeze(0).to(device)
    return ModelArtifacts(
        model=model,
        checkpoint_path=ckpt_path,
        input_tensor=inp,
        input_shape=list(inp.shape),
        note=note,
    )


def load_mobileclip_ft4(device: torch.device, image: Image.Image) -> ModelArtifacts:
    import open_clip

    ckpt_path = ROOT / "weights" / "mobileclip_forensics" / "mobileclip_s2_forensics_ft4.pth"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    finetune_blocks = int(ckpt.get("finetune_blocks", 4))

    clip_model, _, _ = open_clip.create_model_and_transforms(
        "MobileCLIP-S2",
        pretrained="datacompdr",
        cache_dir=str(ROOT / "weights" / "mobileclip"),
    )
    model = MobileCLIPForensics(clip_model, finetune_blocks=finetune_blocks).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    inp = MOBILECLIP_TRANSFORM(image).unsqueeze(0).to(device)
    return ModelArtifacts(
        model=model,
        checkpoint_path=ckpt_path,
        input_tensor=inp,
        input_shape=list(inp.shape),
        note=f"MobileCLIP-S2 forensics ft{finetune_blocks}",
    )


def load_tiny_ladeda(device: torch.device, image: Image.Image) -> ModelArtifacts:
    from networks.Tiny_LaDeDa import Bottleneck, TinyLaDeDa

    ckpt_path = TINYLA_DIR / "weights" / "Tiny_LaDeDa" / "WildRF_Tiny_LaDeDa.pth"
    orig_exp = Bottleneck.expansion
    Bottleneck.expansion = 1
    model = TinyLaDeDa(Bottleneck, layer=1, stride=2, kernel=1, preprocess_type="right_diag")
    Bottleneck.expansion = orig_exp
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model = model.to(device).eval()
    inp = TINYLA_TRANSFORM(image).unsqueeze(0).to(device)
    return ModelArtifacts(
        model=model,
        checkpoint_path=ckpt_path,
        input_tensor=inp,
        input_shape=list(inp.shape),
        note="Tiny-LaDeDa WildRF binary screener",
    )


def load_mobilenetv2_proxy(device: torch.device, image: Image.Image) -> ModelArtifacts:
    model = models.mobilenet_v2(weights=None, num_classes=3).to(device).eval()
    inp = MOBILENET_TRANSFORM(image).unsqueeze(0).to(device)
    return ModelArtifacts(
        model=model,
        checkpoint_path=None,
        input_tensor=inp,
        input_shape=list(inp.shape),
        note="Proxy only: torchvision MobileNetV2 single-stream, dual-stream Track1 not implemented",
    )


def load_mobilenetv2_dualstream(device: torch.device, image: Image.Image) -> ModelArtifacts:
    ckpt_path = ROOT / "weights" / "mobilenetv2_dualstream" / "mobilenetv2_dualstream_best.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Dual-stream checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = DualStreamMobileNetV2(pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    inp = MOBILENET_DUALSTREAM_TRANSFORM(image).unsqueeze(0).to(device)
    return ModelArtifacts(
        model=model,
        checkpoint_path=ckpt_path,
        input_tensor=inp,
        input_shape=list(inp.shape),
        note="Track1 MobileNetV2 dual-stream (RGB + SRM residual)",
    )


MODEL_SPECS: Dict[str, Callable[[torch.device, Image.Image], ModelArtifacts]] = {
    "forma": load_forma,
    "mobileclip_ft4": load_mobileclip_ft4,
    "tiny_ladeda": load_tiny_ladeda,
    "mobilenetv2_dualstream": load_mobilenetv2_dualstream,
    "mobilenetv2_proxy": load_mobilenetv2_proxy,
}


def tensor_bytes_from_state_dict(model: nn.Module) -> int:
    total = 0
    for value in model.state_dict().values():
        if torch.is_tensor(value):
            total += value.numel() * value.element_size()
    return total


def mib(num_bytes: Optional[int]) -> Optional[float]:
    if num_bytes is None:
        return None
    return round(num_bytes / (1024 ** 2), 3)


def summarize_model(model: nn.Module, checkpoint_path: Optional[Path]) -> Dict[str, object]:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    state_bytes = tensor_bytes_from_state_dict(model)
    ckpt_bytes = checkpoint_path.stat().st_size if checkpoint_path and checkpoint_path.exists() else None
    return {
        "params_total": int(total_params),
        "params_total_m": round(total_params / 1e6, 4),
        "params_trainable": int(trainable_params),
        "params_trainable_m": round(trainable_params / 1e6, 4),
        "state_dict_mib": mib(state_bytes),
        "checkpoint_mib": mib(ckpt_bytes),
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
    }


def percentile(sorted_vals: List[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    idx = (len(sorted_vals) - 1) * q
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return sorted_vals[lo]
    frac = idx - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def benchmark_forward(
    model: nn.Module,
    input_tensor: torch.Tensor,
    device: torch.device,
    warmup: int,
    min_runs: int,
    max_runs: int,
    min_duration_s: float,
) -> Dict[str, object]:
    times_ms: List[float] = []
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    with torch.inference_mode():
        for _ in range(warmup):
            _ = model(input_tensor)
            if device.type == "cuda":
                torch.cuda.synchronize(device)

        started = time.perf_counter()
        while len(times_ms) < max_runs:
            t0 = time.perf_counter()
            _ = model(input_tensor)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            times_ms.append(dt_ms)
            if len(times_ms) >= min_runs and (time.perf_counter() - started) >= min_duration_s:
                break

    vals = sorted(times_ms)
    mean_ms = statistics.mean(vals)
    median_ms = statistics.median(vals)
    std_ms = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    p95_ms = percentile(vals, 0.95)
    peak_gpu_mib = None
    if device.type == "cuda":
        peak_gpu_mib = round(torch.cuda.max_memory_allocated(device) / (1024 ** 2), 3)

    return {
        "runs": len(vals),
        "warmup_runs": warmup,
        "mean_ms": round(mean_ms, 3),
        "median_ms": round(median_ms, 3),
        "std_ms": round(std_ms, 3),
        "p95_ms": round(p95_ms, 3),
        "min_ms": round(vals[0], 3),
        "max_ms": round(vals[-1], 3),
        "fps_from_mean": round(1000.0 / mean_ms, 3) if mean_ms > 0 else None,
        "peak_gpu_mib": peak_gpu_mib,
    }


def device_benchmark_config(device: torch.device) -> Dict[str, float]:
    if device.type == "cuda":
        return {"warmup": 3, "min_runs": 10, "max_runs": 40, "min_duration_s": 1.0}
    return {"warmup": 1, "min_runs": 3, "max_runs": 12, "min_duration_s": 1.5}


def benchmark_one_model(name: str, loader_fn, image: Image.Image, device: torch.device) -> Dict[str, object]:
    artifacts = loader_fn(device, image)
    model_meta = summarize_model(artifacts.model, artifacts.checkpoint_path)
    bench_cfg = device_benchmark_config(device)
    latency = benchmark_forward(
        artifacts.model,
        artifacts.input_tensor,
        device,
        warmup=int(bench_cfg["warmup"]),
        min_runs=int(bench_cfg["min_runs"]),
        max_runs=int(bench_cfg["max_runs"]),
        min_duration_s=float(bench_cfg["min_duration_s"]),
    )
    result = {
        "name": name,
        "device": device.type,
        "input_shape": artifacts.input_shape,
        "note": artifacts.note,
        "model": model_meta,
        "latency": latency,
    }
    del artifacts
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def render_markdown(results: Dict[str, Dict[str, object]], meta: Dict[str, object]) -> str:
    lines: List[str] = []
    lines.append("# Backbone Benchmark")
    lines.append("")
    lines.append(f"- Timestamp: `{meta['timestamp']}`")
    lines.append(f"- Hardware: CPU `{meta['hardware']['cpu_model']}`, GPU `{meta['hardware'].get('gpu_model')}`")
    lines.append(f"- Torch: `{meta['hardware']['torch_version']}`")
    lines.append("- Batch: `1`")
    lines.append("- Scope: forward-only latency (preprocess / disk I/O / host-device copy excluded)")
    lines.append("")
    lines.append("## Size")
    lines.append("")
    lines.append("| Model | Params (M) | Trainable (M) | State Dict (MiB) | Checkpoint (MiB) | Note |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for name, row in results.items():
        primary = row.get("cuda") or row.get("cpu")
        model = primary["model"]
        note = primary["note"]
        ckpt = model["checkpoint_mib"]
        lines.append(
            f"| {name} | {model['params_total_m']:.4f} | {model['params_trainable_m']:.4f} | "
            f"{model['state_dict_mib']:.3f} | {ckpt if ckpt is not None else 'N/A'} | {note} |"
        )
    lines.append("")
    lines.append("## Latency")
    lines.append("")
    lines.append("| Model | GPU mean / p50 / p95 (ms) | GPU FPS | GPU peak MiB | CPU mean / p50 / p95 (ms) | CPU FPS | Input |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    for name, row in results.items():
        gpu = row.get("cuda", {}).get("latency", {})
        cpu = row.get("cpu", {}).get("latency", {})
        primary = row.get("cuda") or row.get("cpu")
        input_shape = primary["input_shape"]
        gpu_text = f"{gpu.get('mean_ms')} / {gpu.get('median_ms')} / {gpu.get('p95_ms')}" if gpu else "N/A"
        cpu_text = f"{cpu.get('mean_ms')} / {cpu.get('median_ms')} / {cpu.get('p95_ms')}" if cpu else "N/A"
        lines.append(
            f"| {name} | {gpu_text} | {gpu.get('fps_from_mean', 'N/A')} | {gpu.get('peak_gpu_mib', 'N/A')} | "
            f"{cpu_text} | {cpu.get('fps_from_mean', 'N/A')} | `{input_shape}` |"
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- `mobilenetv2_dualstream` is the actual Track-1 noise backbone used by `run_phase3_tracks.py`.")
    lines.append("- `mobilenetv2_proxy` is a legacy single-stream reference kept only for historical comparison.")
    lines.append("- `checkpoint_mib` reflects the on-disk checkpoint file currently used in the repo.")
    lines.append("- `state_dict_mib` reflects raw tensor storage in memory and is more comparable across checkpoint formats.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Phase 2 backbones on GPU and CPU")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODEL_SPECS.keys()),
        default=["forma", "mobileclip_ft4", "tiny_ladeda", "mobilenetv2_dualstream"],
        help="models to benchmark",
    )
    parser.add_argument(
        "--devices",
        nargs="+",
        choices=["cuda", "cpu"],
        default=["cuda", "cpu"],
        help="devices to benchmark",
    )
    args = parser.parse_args()

    sample_path = sample_image_path()
    image = load_rgb(sample_path)
    meta = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "sample_image": str(sample_path),
        "hardware": hardware_info(),
    }

    print("=" * 72)
    print("Phase 2.7 — Backbone Latency / Size Benchmark")
    print("=" * 72)
    print(f"Sample image: {sample_path}")
    print(f"Models: {', '.join(args.models)}")
    print(f"Devices: {', '.join(args.devices)}")

    results: Dict[str, Dict[str, object]] = {}
    for model_name in args.models:
        loader_fn = MODEL_SPECS[model_name]
        results[model_name] = {}
        print(f"\n[{model_name}]")
        for device_name in args.devices:
            if device_name == "cuda" and not torch.cuda.is_available():
                print("  cuda unavailable → skip")
                continue
            device = torch.device(device_name)
            print(f"  - benchmarking on {device_name}...")
            results[model_name][device_name] = benchmark_one_model(model_name, loader_fn, image, device)
            lat = results[model_name][device_name]["latency"]
            print(
                f"    mean={lat['mean_ms']}ms p50={lat['median_ms']}ms "
                f"p95={lat['p95_ms']}ms fps={lat['fps_from_mean']}"
            )

    payload = {
        "meta": meta,
        "results": results,
    }
    ts = meta["timestamp"]
    json_path = OUT_DIR / f"backbone_benchmark_{ts}.json"
    md_path = OUT_DIR / f"backbone_benchmark_{ts}.md"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(render_markdown(results, meta))

    print(f"\nSaved JSON: {json_path}")
    print(f"Saved Markdown: {md_path}")


if __name__ == "__main__":
    main()
