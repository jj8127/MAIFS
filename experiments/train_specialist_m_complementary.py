#!/usr/bin/env python3
"""
SpecM-Complementary: Error-Weighted Complementary Training
===========================================================

MNV2(3-class, 동결)의 정답확률로 SpecM 학습 가중치를 계산한다.
  L = mean( w(x) * CE(SpecM(x), y_binary) )
  w(x) = clip( (1 - c(x))^γ, 0, w_max )
  c(x) = P_MNV2(y_true_binary|x)   ← 정답확률 (MSP 아님)

설계 결정사항 (사용자 확인):
  - Dustbin:    ai_generated → SpecM label=1 (manipulated)
                c_dustbin(x) = 1 - P_MNV2(authentic|x)
                              = P_MNV2(manip|x) + P_MNV2(aigen|x)
  - 정답확률:   auth→P_MNV2(auth|x), manip→P_MNV2(manip|x)
  - γ ablation: {1, 2, 3} — --gamma 인수로 지정
  - w_max clipping: {1.0(no-clip), 5.0, 10.0, 20.0} — --w-max 인수
  - Temperature Scaling: --no-ts 플래그로 ablation

2-stage training (MNV2 동결 → SpecM 학습):
  Stage 0: MNV2 inference cache (per-image c(x) 사전계산)
  Stage 1: SpecM fine-tuning with weighted loss

실행:
  # 기본 (γ=2, w_max=10, v4 best에서 resume)
  .venv-qwen/bin/python experiments/train_specialist_m_complementary.py

  # γ ablation
  .venv-qwen/bin/python experiments/train_specialist_m_complementary.py --gamma 1
  .venv-qwen/bin/python experiments/train_specialist_m_complementary.py --gamma 3

  # w_max ablation
  .venv-qwen/bin/python experiments/train_specialist_m_complementary.py --gamma 2 --w-max 5.0
  .venv-qwen/bin/python experiments/train_specialist_m_complementary.py --gamma 2 --w-max 20.0
  .venv-qwen/bin/python experiments/train_specialist_m_complementary.py --gamma 2 --w-max 1.0

  # Temperature Scaling 비교 (--no-ts = TS 없이 기본 c(x) 사용)
  .venv-qwen/bin/python experiments/train_specialist_m_complementary.py --no-ts

  # eval only
  .venv-qwen/bin/python experiments/train_specialist_m_complementary.py \\
      --eval-only weights/specialist_m_complementary/gamma2_wmax10/best.pth
"""

from __future__ import annotations

import argparse
import io
import importlib
import json
import random
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms

warnings.filterwarnings("ignore")

ROOT   = Path(__file__).resolve().parents[1]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sys.path.insert(0, str(ROOT / "experiments"))

BINARY_LABEL  = {"authentic": 0, "manipulated": 1}
IDX2LABEL     = {0: "authentic", 1: "manipulated"}
MNV2_LABEL    = {"authentic": 0, "manipulated": 1, "ai_generated": 2}
IMG_EXTS      = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ── 경로 ──────────────────────────────────────────────────────────────────────
CASIA2_AU         = ROOT / "datasets" / "CASIA2_subset" / "Au"
CASIA2_TP         = ROOT / "datasets" / "CASIA2_subset" / "Tp"
IMD2020_DIR       = ROOT / "datasets" / "IMD2020_subset" / \
                    "IMD2020_Generative_Image_Inpainting_yu2018_01" / "images"
GENIMAGE_NATURE   = ROOT / "datasets" / "GenImage_subset" / "BigGAN" / "val" / "nature"
GENIMAGE_AI       = ROOT / "datasets" / "GenImage_subset" / "BigGAN" / "val" / "ai"
GENIMAGE_AUTH_MAX = 3000
GENIMAGE_AI_MAX   = 2000   # dustbin 샘플 수 (ablation 가능)

MNV2_CKPT   = ROOT / "weights" / "mobilenetv2_dualstream" / "mobilenetv2_dualstream_best.pth"
SPECM_V4    = ROOT / "weights" / "specialist_m_v4" / "specialist_m_v4_best.pth"

DSC_EVAL_JSONL = ROOT / "experiments" / "results" / "backbone_eval" / \
                 "mobilenetv2_dualstream_dsC_20260319_070725.jsonl"

CACHE_DIR = ROOT / "experiments" / "results" / "phase1_ewct"
WEIGHT_CACHE = CACHE_DIR / "mnv2_training_weights_cache.jsonl"

EVAL_JSONL = {
    "base":       ROOT / "experiments" / "results" / "backbone_eval" / "mobilenetv2_dualstream_base_20260319_070725.jsonl",
    "dsC":        ROOT / "experiments" / "results" / "backbone_eval" / "mobilenetv2_dualstream_dsC_20260319_070725.jsonl",
    "opensdi":    ROOT / "experiments" / "results" / "backbone_eval" / "mobilenetv2_dualstream_opensdi_20260319_070725.jsonl",
    "aigenproxy": ROOT / "experiments" / "results" / "backbone_eval" / "mobilenetv2_dualstream_aigenproxy_20260319_070725.jsonl",
}


# ═══════════════════════════════════════════════════════════════════════════════
# 1. MNV2 모델 (동결, 3-class)
# ═══════════════════════════════════════════════════════════════════════════════

def load_mnv2_frozen() -> nn.Module:
    """MNV2 DualStreamMobileNetV2 로드 후 완전 동결"""
    import train_mobilenetv2_dualstream as _m
    importlib.reload(_m)
    model = _m.DualStreamMobileNetV2(pretrained=False)
    sd = torch.load(MNV2_CKPT, map_location="cpu")
    sd = sd.get("model_state_dict", sd)
    model.load_state_dict(sd, strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model.to(DEVICE)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. SpecialistM 아키텍처 (v4와 동일, 2-class)
# ═══════════════════════════════════════════════════════════════════════════════

def normalize_batch(x: torch.Tensor, mean, std) -> torch.Tensor:
    t = torch.tensor(mean if len(mean) == x.shape[1] else mean[:1],
                     device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
    s = torch.tensor(std  if len(std)  == x.shape[1] else std[:1],
                     device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
    return (x - t) / s


class SRMExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        f1 = [[0,0,0,0,0],[0,-1,2,-1,0],[0,2,-4,2,0],[0,-1,2,-1,0],[0,0,0,0,0]]
        f2 = [[-1,2,-2,2,-1],[2,-6,8,-6,2],[-2,8,-12,8,-2],[2,-6,8,-6,2],[-1,2,-2,2,-1]]
        f3 = [[0,0,0,0,0],[0,0,0,0,0],[0,1,-2,1,0],[0,0,0,0,0],[0,0,0,0,0]]
        q  = torch.tensor([4.0, 12.0, 2.0]).view(3, 1, 1, 1)
        k  = torch.tensor([f1, f2, f3], dtype=torch.float32).unsqueeze(1) / q
        self.register_buffer("kernels", k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gray = x.mean(dim=1, keepdim=True)
        res  = F.conv2d(gray, self.kernels, padding=2)
        return ((res.clamp(-2.0, 2.0) + 2.0) / 4.0)


class DCTResidualExtractor(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gray     = x.mean(dim=1, keepdim=True)
        low_pass = F.avg_pool2d(gray, kernel_size=8, stride=8)
        low_pass = F.interpolate(low_pass, size=gray.shape[-2:], mode='nearest')
        residual = (gray - low_pass).clamp(-1.0, 1.0)
        return (residual + 1.0) / 2.0


class MobileNetBranch(nn.Module):
    def __init__(self, in_channels: int = 3, pretrained: bool = True):
        super().__init__()
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None
        base    = models.mobilenet_v2(weights=weights)
        if in_channels != 3:
            old = base.features[0][0]
            base.features[0][0] = nn.Conv2d(
                in_channels, old.out_channels,
                kernel_size=old.kernel_size, stride=old.stride,
                padding=old.padding, bias=False)
        self.features = base.features
        self.pool     = nn.AdaptiveAvgPool2d(1)
        self.out_dim  = 1280

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.features(x)).flatten(1)


class SpecialistM(nn.Module):
    def __init__(self, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        self.srm_extractor = SRMExtractor()
        self.dct_extractor = DCTResidualExtractor()
        self.rgb_branch    = MobileNetBranch(in_channels=3, pretrained=pretrained)
        self.srm_branch    = MobileNetBranch(in_channels=3, pretrained=False)
        self.dct_branch    = MobileNetBranch(in_channels=1, pretrained=False)
        fused = self.rgb_branch.out_dim + self.srm_branch.out_dim + self.dct_branch.out_dim
        self.head = nn.Sequential(
            nn.LayerNorm(fused),
            nn.Linear(fused, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rgb  = normalize_batch(x, IMAGENET_MEAN, IMAGENET_STD)
        srm  = self.srm_extractor(x)
        srm  = normalize_batch(srm, [0.5]*3, [0.5]*3)
        dct  = self.dct_extractor(x)
        dct  = normalize_batch(dct, [0.5], [0.5])
        return self.head(torch.cat([
            self.rgb_branch(rgb),
            self.srm_branch(srm),
            self.dct_branch(dct),
        ], dim=1))


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Temperature Scaling (post-hoc calibration)
# ═══════════════════════════════════════════════════════════════════════════════

class TemperatureScaling(nn.Module):
    """MNV2 softmax calibration을 위한 단일 파라미터 post-hoc TS"""
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature.clamp(min=0.1)


def fit_temperature_scaling(mnv2: nn.Module, calib_records: List[Dict],
                             n_iters: int = 100, lr: float = 0.01) -> float:
    """
    MNV2 calibration: validation set으로 temperature T를 NLL 최소화로 학습.
    Returns: 최적 temperature 값
    """
    ts = TemperatureScaling().to(DEVICE)
    optimizer = torch.optim.LBFGS([ts.temperature], lr=lr, max_iter=n_iters)
    val_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    # calibration 데이터 (val set)에서 logits 수집
    all_logits, all_labels = [], []
    mnv2.eval()
    with torch.no_grad():
        for rec in calib_records:
            path = ROOT / rec["image_path"]
            lbl  = MNV2_LABEL.get(rec["true_label"], -1)
            if lbl < 0:
                continue
            try:
                img    = Image.open(path).convert("RGB")
                tensor = val_tf(img).unsqueeze(0).to(DEVICE)
                logit  = mnv2(tensor).cpu()
                all_logits.append(logit)
                all_labels.append(lbl)
            except Exception:
                continue

    if not all_logits:
        print("  [TS] calibration 데이터 없음 → T=1.0 (no-op)")
        return 1.0

    logits_t = torch.cat(all_logits).to(DEVICE)
    labels_t = torch.tensor(all_labels, dtype=torch.long).to(DEVICE)

    def closure():
        optimizer.zero_grad()
        scaled = ts(logits_t)
        loss   = F.cross_entropy(scaled, labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    T = ts.temperature.item()
    print(f"  [TS] 최적 Temperature: T={T:.4f}")
    return T


# ═══════════════════════════════════════════════════════════════════════════════
# 4. MNV2 confidence 캐시 (학습 데이터 전체에 대해 사전 계산)
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def build_weight_cache(records: List[Dict], mnv2: nn.Module,
                       temperature: float = 1.0,
                       cache_path: Optional[Path] = None,
                       batch_size: int = 64) -> Dict[str, float]:
    """
    학습 데이터 각 이미지에 대해 c(x) = 정답확률 계산 후 캐시.

    c(x) 정의:
      - authentic:  P_MNV2(authentic|x)
      - manipulated: P_MNV2(manipulated|x)
      - ai_generated (dustbin): 1 - P_MNV2(authentic|x)
                                = P_MNV2(manip|x) + P_MNV2(aigen|x)

    Returns: {image_path_str: c_x}
    """
    if cache_path and cache_path.exists():
        print(f"  [캐시] 기존 캐시 로드: {cache_path}")
        cache = {}
        with open(cache_path) as f:
            for line in f:
                obj = json.loads(line)
                cache[obj["image_path"]] = obj["c_x"]
        # 기존 캐시에 없는 레코드 확인
        missing = [r for r in records if r["image_path"] not in cache]
        if not missing:
            print(f"  [캐시] {len(cache)}개 로드 완료 (신규 없음)")
            return cache
        print(f"  [캐시] {len(cache)}개 로드, {len(missing)}개 신규 계산 필요")
        records = missing
    else:
        cache = {}

    val_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    mnv2.eval()
    new_entries = []
    n_fail = 0

    # 배치 단위 처리
    for i in range(0, len(records), batch_size):
        batch_recs = records[i:i + batch_size]
        imgs, valid_recs = [], []
        for rec in batch_recs:
            path = ROOT / rec["image_path"]
            try:
                img = Image.open(path).convert("RGB")
                imgs.append(val_tf(img))
                valid_recs.append(rec)
            except Exception:
                n_fail += 1
                # 실패 시 c_x=0.5 (중립 가중치)
                cache[rec["image_path"]] = 0.5

        if not imgs:
            continue

        tensor = torch.stack(imgs).to(DEVICE)
        logits = mnv2(tensor)
        if temperature != 1.0:
            logits = logits / max(temperature, 0.1)
        probs  = F.softmax(logits, dim=-1).cpu().numpy()  # (B, 3)

        for rec, prob_vec in zip(valid_recs, probs):
            true_label = rec["true_label"]
            if true_label == "authentic":
                c_x = float(prob_vec[0])  # P(auth)
            elif true_label == "manipulated":
                c_x = float(prob_vec[1])  # P(manip)
            else:  # ai_generated → dustbin
                c_x = 1.0 - float(prob_vec[0])  # 1 - P(auth)
            cache[rec["image_path"]] = c_x
            new_entries.append({"image_path": rec["image_path"], "c_x": c_x})

        if (i // batch_size) % 50 == 0:
            pct = min(i + batch_size, len(records)) / len(records) * 100
            print(f"  [캐시 계산] {pct:.0f}% ({i + len(batch_recs)}/{len(records)})", end="\r")

    print(f"\n  [캐시] 완료: {len(new_entries)}개 신규, {n_fail}개 실패(c_x=0.5 사용)")

    if cache_path and new_entries:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "a") as f:
            for entry in new_entries:
                f.write(json.dumps(entry) + "\n")
        print(f"  [캐시] 저장: {cache_path}")

    return cache


def compute_wx(c_x: float, gamma: float, w_max: float) -> float:
    """w(x) = clip((1 - c(x))^γ, 0, w_max)"""
    return min((1.0 - c_x) ** gamma, w_max)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. 데이터셋 및 증강
# ═══════════════════════════════════════════════════════════════════════════════

class RandomJPEGCompression:
    def __init__(self, quality_range=(40, 95), p=0.5):
        self.quality_range = quality_range
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            quality = random.randint(*self.quality_range)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=quality)
            buf.seek(0)
            img = Image.open(buf).convert("RGB")
        return img


class RandomGaussianNoise:
    def __init__(self, std_range=(0.002, 0.015), p=0.4):
        self.std_range = std_range
        self.p = p

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p:
            std = random.uniform(*self.std_range)
            tensor = (tensor + torch.randn_like(tensor) * std).clamp(0.0, 1.0)
        return tensor


TRAIN_TRANSFORM = transforms.Compose([
    RandomJPEGCompression(quality_range=(40, 95), p=0.5),
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.1),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
    transforms.ToTensor(),
    RandomGaussianNoise(std_range=(0.002, 0.015), p=0.4),
    transforms.RandomErasing(p=0.35, scale=(0.02, 0.25), ratio=(0.3, 3.3), value='random'),
])

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])


class WeightedBinaryDataset(Dataset):
    """
    (img, label, w_x) 반환.
    ai_generated → label=1 (dustbin), w_x = (1-c_dustbin)^γ
    """
    def __init__(self, records: List[Dict], weight_cache: Dict[str, float],
                 gamma: float, w_max: float, transform=None):
        self.records      = records
        self.weight_cache = weight_cache
        self.gamma        = gamma
        self.w_max        = w_max
        self.transform    = transform or VAL_TRANSFORM

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec   = self.records[idx]
        path  = ROOT / rec["image_path"]
        # ai_generated → 1 (dustbin), 그 외 binary
        if rec["true_label"] == "ai_generated":
            label = 1  # dustbin
        else:
            label = BINARY_LABEL.get(rec["true_label"], -1)

        c_x = self.weight_cache.get(rec["image_path"], 0.5)
        w_x = compute_wx(c_x, self.gamma, self.w_max)

        try:
            img    = Image.open(path).convert("RGB")
            tensor = self.transform(img)
        except Exception:
            tensor = torch.zeros(3, 224, 224)

        return tensor, label, float(w_x), rec["image_path"]


# ═══════════════════════════════════════════════════════════════════════════════
# 6. 데이터 수집
# ═══════════════════════════════════════════════════════════════════════════════

def get_dsc_eval_imd2020_filenames() -> set:
    if not DSC_EVAL_JSONL.exists():
        return set()
    excluded = set()
    with open(DSC_EVAL_JSONL) as f:
        for line in f:
            r = json.loads(line.strip())
            if "IMD2020" in r.get("image_path", ""):
                excluded.add(Path(r["image_path"]).name)
    return excluded


def scan_training_data(genimage_auth_max: int = GENIMAGE_AUTH_MAX,
                       aigen_max: int = GENIMAGE_AI_MAX) -> List[Dict]:
    """학습 데이터 수집 (SpecM binary + dustbin)"""
    records = []
    excluded_imd = get_dsc_eval_imd2020_filenames()

    au_casia = 0
    for p in sorted(CASIA2_AU.iterdir()):
        if p.suffix.lower() in IMG_EXTS:
            records.append({"image_path": str(p.relative_to(ROOT)),
                             "true_label": "authentic"})
            au_casia += 1

    au_genimg = 0
    if GENIMAGE_NATURE.exists():
        nature_imgs = sorted(p for p in GENIMAGE_NATURE.iterdir()
                             if p.suffix.lower() in IMG_EXTS)
        rng = random.Random(42)
        rng.shuffle(nature_imgs)
        for p in nature_imgs[:genimage_auth_max]:
            records.append({"image_path": str(p.relative_to(ROOT)),
                             "true_label": "authentic"})
            au_genimg += 1

    tp_n = 0
    for p in sorted(CASIA2_TP.iterdir()):
        if p.suffix.lower() in IMG_EXTS:
            records.append({"image_path": str(p.relative_to(ROOT)),
                             "true_label": "manipulated"})
            tp_n += 1

    imd_n = 0
    if IMD2020_DIR.exists():
        for p in sorted(IMD2020_DIR.iterdir()):
            if p.suffix.lower() in IMG_EXTS and p.name not in excluded_imd:
                records.append({"image_path": str(p.relative_to(ROOT)),
                                 "true_label": "manipulated"})
                imd_n += 1

    # Dustbin: BigGAN ai_generated
    ai_n = 0
    if GENIMAGE_AI.exists() and aigen_max > 0:
        ai_imgs = sorted(p for p in GENIMAGE_AI.iterdir()
                         if p.suffix.lower() in IMG_EXTS)
        rng2 = random.Random(42)
        rng2.shuffle(ai_imgs)
        for p in ai_imgs[:aigen_max]:
            records.append({"image_path": str(p.relative_to(ROOT)),
                             "true_label": "ai_generated"})
            ai_n += 1

    total_au    = au_casia + au_genimg
    total_manip = tp_n + imd_n
    print(f"  Authentic   — CASIA2: {au_casia}, GenImage_nature: {au_genimg}  → {total_au}")
    print(f"  Manipulated — CASIA2: {tp_n}, IMD2020(non-eval): {imd_n}       → {total_manip}")
    print(f"  Dustbin(AI) — BigGAN: {ai_n}")
    print(f"  Total: {len(records)}")
    return records


def split_records(records: List[Dict], val_ratio: float = 0.15,
                  seed: int = 42) -> Tuple[List, List]:
    rng  = random.Random(seed)
    data = records[:]
    rng.shuffle(data)
    n    = int(len(data) * val_ratio)
    return data[n:], data[:n]


def make_sampler(records: List[Dict]) -> WeightedRandomSampler:
    """클래스 균형 샘플러 (dustbin은 manipulated로 취급)"""
    labels = []
    for r in records:
        lbl = 1 if r["true_label"] == "ai_generated" else BINARY_LABEL.get(r["true_label"], 1)
        labels.append(lbl)
    class_counts = [labels.count(0), labels.count(1)]
    weights      = [1.0 / max(class_counts[l], 1) for l in labels]
    return WeightedRandomSampler(weights, num_samples=len(records), replacement=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. 손실 함수 & 학습/평가
# ═══════════════════════════════════════════════════════════════════════════════

def complementary_loss(logits: torch.Tensor, labels: torch.Tensor,
                       weights: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    L = sum(w(x) * CE(SpecM(x), y)) / sum(w(x))
    normalize by sum(w) so that batches with small total weight don't explode
    """
    ce      = F.cross_entropy(logits, labels, reduction="none")
    w_total = weights.sum().clamp(min=eps)
    return (weights * ce).sum() / w_total


def train_epoch(model, loader, optimizer, scaler) -> Tuple[float, float]:
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    use_amp = DEVICE.type == "cuda"

    for imgs, labels, wx, _ in loader:
        imgs   = imgs.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        wx     = wx.to(DEVICE, dtype=torch.float32, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", enabled=use_amp):
            logits = model(imgs)
            loss   = complementary_loss(logits, labels, wx)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * len(labels)
        correct    += (logits.argmax(1) == labels).sum().item()
        n          += len(labels)

    return total_loss / max(n, 1), correct / max(n, 1)


@torch.no_grad()
def evaluate_binary(model, loader) -> Dict:
    model.eval()
    all_preds, all_probs, all_labels, all_wx = [], [], [], []

    for imgs, labels, wx, _ in loader:
        imgs  = imgs.to(DEVICE, non_blocking=True)
        probs = F.softmax(model(imgs), dim=-1).cpu()
        all_preds.append(probs.argmax(1))
        all_probs.append(probs)
        all_labels.append(labels)
        all_wx.append(wx)

    probs  = torch.cat(all_probs).numpy()
    preds  = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    wx     = torch.cat(all_wx).numpy()

    valid  = labels != -1
    preds, labels, probs, wx = preds[valid], labels[valid], probs[valid], wx[valid]

    acc = float((preds == labels).mean())
    per_class, recalls = {}, []
    for idx, name in IDX2LABEL.items():
        mask = labels == idx
        if mask.sum() == 0:
            per_class[name] = {"recall": 0.0, "n": 0}
            recalls.append(0.0)
            continue
        rec = float((preds[mask] == idx).mean())
        per_class[name] = {"recall": rec, "n": int(mask.sum())}
        recalls.append(rec)

    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    prec   = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1_m   = 2 * prec * recall / max(prec + recall, 1e-8)

    # weighted loss
    logits_t = torch.tensor(np.log(probs + 1e-8))
    wx_t     = torch.tensor(wx)
    lbl_t    = torch.tensor(labels, dtype=torch.long)
    val_loss = complementary_loss(logits_t, lbl_t, wx_t).item()

    return {
        "accuracy":        acc,
        "macro_recall":    float(np.mean(recalls)),
        "manip_f1":        f1_m,
        "manip_recall":    recall,
        "manip_precision": prec,
        "val_loss":        val_loss,
        "per_class":       per_class,
        "probs":           probs,
        "preds":           preds,
        "labels":          labels,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 8. eval 데이터셋에서 AURC 계산 (MNV2 오분류 교정률)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_mnv2_error_correction_rate(specm_preds: np.ndarray,
                                        specm_scores: np.ndarray,
                                        mnv2_jsonl_path: Path) -> Dict:
    """
    MNV2 오분류 케이스에서 SpecM이 얼마나 교정하는지 측정.
    binary 공간(auth/manip)에서만 계산.
    """
    if not mnv2_jsonl_path.exists():
        return {}

    mnv2_records = []
    with open(mnv2_jsonl_path) as f:
        for line in f:
            r = json.loads(line)
            if r["true_label"] in BINARY_LABEL:
                mnv2_records.append(r)

    if len(mnv2_records) != len(specm_preds):
        return {"error": f"샘플 수 불일치: MNV2={len(mnv2_records)}, SpecM={len(specm_preds)}"}

    n_mnv2_errors, n_corrected = 0, 0
    patterns = {}

    for rec, s_pred, s_score in zip(mnv2_records, specm_preds, specm_scores):
        true_lbl = rec["true_label"]
        mnv2_pred = rec["pred_label"]
        if mnv2_pred == "ai_generated":
            continue  # binary 공간에서는 제외

        if mnv2_pred != true_lbl:
            n_mnv2_errors += 1
            specm_pred_lbl = IDX2LABEL[int(s_pred)]
            pattern = f"{true_lbl}→{mnv2_pred}"
            if pattern not in patterns:
                patterns[pattern] = {"total": 0, "corrected": 0}
            patterns[pattern]["total"] += 1
            if specm_pred_lbl == true_lbl:
                n_corrected += 1
                patterns[pattern]["corrected"] += 1

    correction_rate = n_corrected / max(n_mnv2_errors, 1)
    for p in patterns.values():
        p["rate"] = round(p["corrected"] / max(p["total"], 1), 4)

    return {
        "n_mnv2_errors": n_mnv2_errors,
        "n_corrected":   n_corrected,
        "correction_rate": round(correction_rate, 4),
        "patterns":      patterns,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 9. eval 데이터 로드 (backbone_eval JSONL에서 binary 샘플 추출)
# ═══════════════════════════════════════════════════════════════════════════════

def load_eval_records_binary(jsonl_path: Path) -> List[Dict]:
    """binary 레이블(auth/manip)만 필터링하여 반환"""
    records = []
    if not jsonl_path.exists():
        return records
    with open(jsonl_path) as f:
        for line in f:
            r = json.loads(line)
            if r["true_label"] in BINARY_LABEL:
                records.append({
                    "image_path": r["image_path"],
                    "true_label": r["true_label"],
                })
    return records


# ═══════════════════════════════════════════════════════════════════════════════
# 10. 메인 학습 루프
# ═══════════════════════════════════════════════════════════════════════════════

def run_training(args) -> Path:
    tag = f"gamma{args.gamma}_wmax{args.w_max:.0f}"
    if args.no_ts:
        tag += "_noTS"

    print("=" * 70)
    print(f"SpecM-Complementary Training: {tag}")
    print(f"  γ={args.gamma}, w_max={args.w_max}, TS={'disabled' if args.no_ts else 'enabled'}")
    print(f"  epochs={args.epochs}, batch={args.batch_size}, lr={args.lr}")
    print(f"  resume: {args.resume}")
    print("=" * 70)

    # ── 데이터 수집 ──
    print("\n[1/5] 학습 데이터 수집")
    all_recs = scan_training_data(genimage_auth_max=args.genimage_max,
                                  aigen_max=args.aigen_max)
    train_recs, val_recs = split_records(all_recs, val_ratio=0.15, seed=42)
    print(f"  Train: {len(train_recs)}, Val: {len(val_recs)}")

    # ── MNV2 로드 ──
    print("\n[2/5] MNV2 동결 로드")
    mnv2 = load_mnv2_frozen()
    print(f"  MNV2 체크포인트: {MNV2_CKPT.name}")

    # ── Temperature Scaling ──
    temperature = 1.0
    if not args.no_ts:
        print("\n[3/5] Temperature Scaling 학습")
        temperature = fit_temperature_scaling(mnv2, val_recs[:500])
    else:
        print("\n[3/5] Temperature Scaling 스킵 (--no-ts)")

    # ── MNV2 confidence 캐시 ──
    print(f"\n[4/5] MNV2 confidence 캐시 계산 (T={temperature:.4f})")
    weight_cache = build_weight_cache(
        all_recs, mnv2, temperature=temperature,
        cache_path=WEIGHT_CACHE if not args.no_ts else None,
        batch_size=args.batch_size * 2
    )
    del mnv2
    torch.cuda.empty_cache()

    # w(x) 통계 출력
    train_cx  = [weight_cache.get(r["image_path"], 0.5) for r in train_recs]
    train_wx  = [compute_wx(c, args.gamma, args.w_max) for c in train_cx]
    print(f"  w(x) 분포 — mean={np.mean(train_wx):.4f}, "
          f"median={np.median(train_wx):.4f}, "
          f">0.3: {np.mean(np.array(train_wx) > 0.3):.1%}")

    # ── 데이터로더 ──
    train_ds = WeightedBinaryDataset(
        train_recs, weight_cache, args.gamma, args.w_max, TRAIN_TRANSFORM)
    val_ds   = WeightedBinaryDataset(
        val_recs, weight_cache, args.gamma, args.w_max, VAL_TRANSFORM)
    sampler  = make_sampler(train_recs)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              sampler=sampler, num_workers=4,
                              pin_memory=(DEVICE.type == "cuda"))
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size * 2,
                              shuffle=False, num_workers=4,
                              pin_memory=(DEVICE.type == "cuda"))

    # ── 모델 초기화 ──
    print(f"\n[5/5] SpecM-Complementary 학습 시작")
    model = SpecialistM(pretrained=False).to(DEVICE)
    resume_path = Path(args.resume)
    if resume_path.exists():
        ckpt = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        v4_f1 = ckpt.get("best_manip_f1", 0.0)
        print(f"  Resume: {resume_path.name} (v4 best manip_f1={v4_f1:.4f})")
    else:
        model = SpecialistM(pretrained=True).to(DEVICE)
        print(f"  경고: v4 체크포인트 없음 → ImageNet pretrained에서 시작")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.05)
    scaler    = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda"))

    ckpt_dir  = ROOT / "weights" / "specialist_m_complementary" / tag
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "best.pth"

    best_f1, history = 0.0, []
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, scaler)
        val_stats       = evaluate_binary(model, val_loader)
        scheduler.step()

        f1_m = val_stats["manip_f1"]
        star = " ★" if f1_m > best_f1 else ""
        print(
            f"  Epoch {epoch:02d}/{args.epochs} | "
            f"loss={tr_loss:.4f} | tr_acc={tr_acc:.3f} | "
            f"val_acc={val_stats['accuracy']:.3f} | "
            f"auth_recall={val_stats['per_class']['authentic']['recall']:.3f} | "
            f"manip_recall={val_stats['manip_recall']:.3f} | "
            f"manip_f1={f1_m:.4f}{star}"
        )
        history.append({
            "epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
            "val_manip_f1": f1_m,
            "val_auth_recall": val_stats["per_class"]["authentic"]["recall"],
            "val_manip_recall": val_stats["manip_recall"],
            "val_loss": val_stats["val_loss"],
        })

        if f1_m > best_f1:
            best_f1 = f1_m
            torch.save({
                "model_state_dict": model.state_dict(),
                "history":          history,
                "best_manip_f1":    best_f1,
                "epoch":            epoch,
                "config": {
                    "gamma":       args.gamma,
                    "w_max":       args.w_max,
                    "temperature": temperature,
                    "no_ts":       args.no_ts,
                    "tag":         tag,
                },
            }, ckpt_path)
            print(f"  → 저장: {ckpt_path}")

    if best_f1 == 0.0:
        torch.save({
            "model_state_dict": model.state_dict(),
            "history":          history,
            "best_manip_f1":    history[-1]["val_manip_f1"] if history else 0.0,
            "epoch":            args.epochs,
            "config": {"gamma": args.gamma, "w_max": args.w_max, "tag": tag},
        }, ckpt_path)

    print(f"\n[완료] best manip_f1={best_f1:.4f}, 저장: {ckpt_path}")
    return ckpt_path


# ═══════════════════════════════════════════════════════════════════════════════
# 11. 전체 데이터셋 평가 + MNV2 오분류 교정률
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def eval_all_datasets(ckpt_path: Path, args) -> Dict:
    print(f"\n[전체 데이터셋 평가] {ckpt_path.name}")
    model = SpecialistM(pretrained=False).to(DEVICE)
    ckpt  = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    cfg   = ckpt.get("config", {})

    gamma  = cfg.get("gamma",  args.gamma)
    w_max  = cfg.get("w_max",  args.w_max)
    ts_T   = cfg.get("temperature", 1.0)

    # eval 시 w_x = 1.0 (공정 비교를 위해 eval은 가중치 없이)
    DUMMY_W = 1.0

    out_dir = ROOT / "experiments" / "results" / "specm_complementary_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = cfg.get("tag", f"gamma{gamma}_wmax{w_max:.0f}")

    summary = {}
    for ds_key, jsonl_path in EVAL_JSONL.items():
        recs = load_eval_records_binary(jsonl_path)
        if not recs:
            print(f"  [{ds_key}] 레코드 없음")
            continue

        # eval용 dummy weight_cache (w=1 for all)
        dummy_cache = {r["image_path"]: 0.0 for r in recs}  # c=0 → w=1
        ds = WeightedBinaryDataset(recs, dummy_cache, gamma=1.0, w_max=1.0, transform=VAL_TRANSFORM)
        loader = DataLoader(ds, batch_size=args.batch_size * 2, shuffle=False,
                            num_workers=4, pin_memory=(DEVICE.type == "cuda"))

        stats   = evaluate_binary(model, loader)
        probs   = stats.pop("probs")
        preds   = stats.pop("preds")
        labels  = stats.pop("labels")

        # MNV2 오분류 교정률
        corr = compute_mnv2_error_correction_rate(preds, probs[:, 1], jsonl_path)

        # JSONL 저장
        out_jsonl = out_dir / f"specm_comp_{tag}_{ds_key}_{ts}.jsonl"
        with open(out_jsonl, "w") as f:
            for rec, pred_idx, prob_vec in zip(recs, preds.tolist(), probs.tolist()):
                f.write(json.dumps({
                    "image_path":      rec["image_path"],
                    "true_label":      rec["true_label"],
                    "pred_label":      IDX2LABEL[pred_idx],
                    "manip_score":     float(prob_vec[1]),
                    "authentic_score": float(prob_vec[0]),
                    "confidence":      float(max(prob_vec)),
                }, ensure_ascii=False) + "\n")

        summary[ds_key] = {
            "n":                len(recs),
            "accuracy":         stats["accuracy"],
            "macro_recall":     stats["macro_recall"],
            "auth_recall":      stats["per_class"]["authentic"]["recall"],
            "manip_recall":     stats["manip_recall"],
            "manip_f1":         stats["manip_f1"],
            "per_class":        stats["per_class"],
            "mnv2_error_corr":  corr,
            "jsonl":            str(out_jsonl),
        }

        corr_rate = corr.get("correction_rate", "N/A")
        print(
            f"  [{ds_key}] n={len(recs)} | "
            f"auth_recall={stats['per_class']['authentic']['recall']:.3f} | "
            f"manip_recall={stats['manip_recall']:.3f} | "
            f"manip_f1={stats['manip_f1']:.4f} | "
            f"MNV2_err_corr={corr_rate}"
        )
        if corr.get("patterns"):
            for pat, info in corr["patterns"].items():
                print(f"    {pat}: {info['corrected']}/{info['total']} ({info['rate']:.1%})")

    summary_path = out_dir / f"specm_comp_{tag}_summary_{ts}.json"
    with open(summary_path, "w") as f:
        json.dump({"tag": tag, "config": cfg, "results": summary},
                  f, indent=2, ensure_ascii=False)
    print(f"\n저장: {summary_path}")
    return summary


# ═══════════════════════════════════════════════════════════════════════════════
# 12. CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="SpecM-Complementary: Error-Weighted Training")
    parser.add_argument("--epochs",       type=int,   default=20)
    parser.add_argument("--batch-size",   type=int,   default=64)
    parser.add_argument("--lr",           type=float, default=3e-5)
    parser.add_argument("--gamma",        type=float, default=2.0,
                        help="w(x) exponent γ ∈ {1,2,3} (default: 2)")
    parser.add_argument("--w-max",        type=float, default=10.0,
                        help="w(x) clipping 상한 (1.0=no-clip, default: 10.0)")
    parser.add_argument("--no-ts",        action="store_true",
                        help="Temperature Scaling 미적용 (ablation용)")
    parser.add_argument("--resume",       type=str,   default=str(SPECM_V4),
                        help="resume 체크포인트 (default: v4 best)")
    parser.add_argument("--genimage-max", type=int,   default=GENIMAGE_AUTH_MAX)
    parser.add_argument("--aigen-max",    type=int,   default=GENIMAGE_AI_MAX,
                        help="dustbin(ai_gen) 샘플 수 (0=dustbin 없음)")
    parser.add_argument("--eval-only",    type=str,   default=None,
                        help="학습 없이 평가만 수행")
    args = parser.parse_args()

    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    print(f"Device: {DEVICE}")

    if args.eval_only:
        eval_all_datasets(Path(args.eval_only), args)
    else:
        ckpt = run_training(args)
        eval_all_datasets(ckpt, args)


if __name__ == "__main__":
    main()
