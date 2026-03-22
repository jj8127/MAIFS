#!/usr/bin/env python3
"""
Coral-friendly export model helpers.

기존 학습 체크포인트는 유지한 채, Edge TPU가 싫어하는 연산만
export 전용으로 치환한 모델을 정의한다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def normalize_batch(x: torch.Tensor, mean: Iterable[float], std: Iterable[float]) -> torch.Tensor:
    mean_list = list(mean)
    std_list = list(std)
    if len(mean_list) != x.shape[1]:
        mean_list = mean_list[:1]
        std_list = std_list[:1]
    mean_t = torch.tensor(mean_list, device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
    std_t = torch.tensor(std_list, device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
    return (x - mean_t) / std_t


def make_coral_head(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    """
    기존 Sequential slot을 최대한 유지해 linear weight는 그대로 재사용한다.

    원래:
      0 LayerNorm / 1 Linear / 2 GELU / 3 Dropout / 4 Linear
    변경:
      0 Identity  / 1 Linear / 2 ReLU / 3 Identity / 4 Linear
    """
    return nn.Sequential(
        nn.Identity(),
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(inplace=False),
        nn.Identity(),
        nn.Linear(hidden_dim, out_dim),
    )


class SRMNoiseExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        f1 = [[0, 0, 0, 0, 0],
              [0, -1, 2, -1, 0],
              [0, 2, -4, 2, 0],
              [0, -1, 2, -1, 0],
              [0, 0, 0, 0, 0]]

        f2 = [[-1, 2, -2, 2, -1],
              [2, -6, 8, -6, 2],
              [-2, 8, -12, 8, -2],
              [2, -6, 8, -6, 2],
              [-1, 2, -2, 2, -1]]

        f3 = [[0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0],
              [0, 1, -2, 1, 0],
              [0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0]]

        q = torch.tensor([4.0, 12.0, 2.0]).view(3, 1, 1, 1)
        kernels = torch.tensor([f1, f2, f3], dtype=torch.float32).unsqueeze(1) / q
        self.register_buffer("kernels", kernels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gray = x.mean(dim=1, keepdim=True)
        residual = F.conv2d(gray, self.kernels, padding=2)
        residual = residual.clamp(-2.0, 2.0)
        return (residual + 2.0) / 4.0


class SRMExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        f1 = [[0, 0, 0, 0, 0], [0, -1, 2, -1, 0], [0, 2, -4, 2, 0], [0, -1, 2, -1, 0], [0, 0, 0, 0, 0]]
        f2 = [[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2], [2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]]
        f3 = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, -2, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
        q = torch.tensor([4.0, 12.0, 2.0]).view(3, 1, 1, 1)
        kernels = torch.tensor([f1, f2, f3], dtype=torch.float32).unsqueeze(1) / q
        self.register_buffer("kernels", kernels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gray = x.mean(dim=1, keepdim=True)
        residual = F.conv2d(gray, self.kernels, padding=2).clamp(-2.0, 2.0)
        return (residual + 2.0) / 4.0


class StaticDCTResidualExtractor(nn.Module):
    """
    224x224 입력 전용의 정적 DCT 근사 extractor.

    `F.interpolate(..., size=gray.shape[-2:])` 대신
    상수 `scale_factor=8`을 사용해 동적 Resize shape 생성을 피한다.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gray = x.mean(dim=1, keepdim=True)
        low_pass = F.avg_pool2d(gray, kernel_size=8, stride=8)
        low_pass = F.interpolate(low_pass, scale_factor=8.0, mode="nearest")
        residual = (gray - low_pass).clamp(-1.0, 1.0)
        return (residual + 1.0) / 2.0


class MobileNetBranch(nn.Module):
    def __init__(self, in_channels: int = 3, pretrained: bool = True):
        super().__init__()
        from torchvision import models

        weights = models.MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None
        base = models.mobilenet_v2(weights=weights)
        if in_channels != 3:
            old = base.features[0][0]
            base.features[0][0] = nn.Conv2d(
                in_channels,
                old.out_channels,
                kernel_size=old.kernel_size,
                stride=old.stride,
                padding=old.padding,
                bias=False,
            )
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out_dim = 1280

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.features(x)).flatten(1)


class CoralDualStreamMobileNetV2(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.noise_extractor = SRMNoiseExtractor()
        self.rgb_branch = MobileNetBranch(pretrained=pretrained)
        self.noise_branch = MobileNetBranch(pretrained=pretrained)
        self.head = make_coral_head(2560, 512, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rgb = normalize_batch(x, IMAGENET_MEAN, IMAGENET_STD)
        noise = normalize_batch(self.noise_extractor(x), IMAGENET_MEAN, IMAGENET_STD)
        fused = torch.cat([self.rgb_branch(rgb), self.noise_branch(noise)], dim=1)
        return self.head(fused)


class CoralSpecialistMv4(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.srm_extractor = SRMExtractor()
        self.dct_extractor = StaticDCTResidualExtractor()
        self.rgb_branch = MobileNetBranch(in_channels=3, pretrained=pretrained)
        self.srm_branch = MobileNetBranch(in_channels=3, pretrained=False)
        self.dct_branch = MobileNetBranch(in_channels=1, pretrained=False)
        self.head = make_coral_head(3840, 256, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rgb = normalize_batch(x, IMAGENET_MEAN, IMAGENET_STD)
        srm = normalize_batch(self.srm_extractor(x), [0.5] * 3, [0.5] * 3)
        dct = normalize_batch(self.dct_extractor(x), [0.5], [0.5])
        fused = torch.cat([
            self.rgb_branch(rgb),
            self.srm_branch(srm),
            self.dct_branch(dct),
        ], dim=1)
        return self.head(fused)


def load_checkpoint(path: Path) -> dict:
    state = torch.load(path, map_location="cpu")
    return state.get("model_state_dict", state)


def _validate_partial_load(label: str, missing: list[str], unexpected: list[str]) -> None:
    allowed_missing = set()
    allowed_unexpected = {"head.0.weight", "head.0.bias"}
    if set(missing) != allowed_missing:
        raise RuntimeError(f"{label}: 예상치 못한 missing keys: {missing}")
    if not set(unexpected).issubset(allowed_unexpected):
        raise RuntimeError(f"{label}: 예상치 못한 unexpected keys: {unexpected}")


def load_mnv2_coral(ckpt_path: Path) -> nn.Module:
    model = CoralDualStreamMobileNetV2(pretrained=False)
    missing, unexpected = model.load_state_dict(load_checkpoint(ckpt_path), strict=False)
    _validate_partial_load("mnv2_coral", missing, unexpected)
    return model.eval()


def load_specm_v4_coral(ckpt_path: Path) -> nn.Module:
    model = CoralSpecialistMv4(pretrained=False)
    missing, unexpected = model.load_state_dict(load_checkpoint(ckpt_path), strict=False)
    _validate_partial_load("specm_v4_coral", missing, unexpected)
    return model.eval()


def load_specm_v4_coral_ft(ckpt_path: Path) -> nn.Module:
    model = CoralSpecialistMv4(pretrained=False)
    state = load_checkpoint(ckpt_path)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"specm_v4_coral_ft: checkpoint mismatch missing={missing} unexpected={unexpected}"
        )
    return model.eval()


def export_to_onnx(model: nn.Module, out_path: Path, input_name: str, image_size: int = 224) -> None:
    import onnx

    out_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, 3, image_size, image_size, dtype=torch.float32)
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            str(out_path),
            input_names=[input_name],
            output_names=["logits"],
            dynamic_axes={input_name: {0: "batch"}, "logits": {0: "batch"}},
            opset_version=14,
            do_constant_folding=True,
            export_params=True,
            dynamo=False,
        )
    onnx.checker.check_model(str(out_path))
