#!/usr/bin/env python3
"""
Specialist-M Sigmoid: v4 checkpoint에서 Independent Sigmoid 재학습
==================================================================

핵심 변경:
  - Softmax(2-class CE) → Independent Sigmoid(2-output BCE)
  - AI-generated 이미지를 target=[0,0]으로 학습 (explicit null-class)
  - 잔여 확률 질량 R = 1 - sigmoid(auth) - sigmoid(manip) 가 ai_generated에 대해
    자연스럽게 높아지도록 학습됨

HEMA에서의 활용:
  - f_residual = R = 1 - P_spec(auth) - P_spec(manip)  ← 진정한 implicit null-class signal
  - f12_new = P(aigen_MNV2) × R                        ← 이론적으로 완전한 cross-modal signal

학습 데이터:
  Authentic (target=[1,0]):   CASIA2 Au (~7491) + GenImage nature (최대 3000)
  Manipulated (target=[0,1]): CASIA2 Tp (~5123) + IMD2020 non-eval (~1710)
  AI-generated (target=[0,0]): GenImage BigGAN/val/ai/ (최대 3000 -- 균형 맞춤)

실행:
  .venv-qwen/bin/python experiments/train_specialist_m_sigmoid.py
  .venv-qwen/bin/python experiments/train_specialist_m_sigmoid.py --eval-only weights/specialist_m_sigmoid/specialist_m_sigmoid_best.pth
"""

from __future__ import annotations

import argparse
import io
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

# target 인코딩: auth=[1,0], manip=[0,1], aigen=[0,0] (explicit null)
LABEL2TARGET = {
    "authentic":   [1.0, 0.0],
    "manipulated": [0.0, 1.0],
    "ai_generated": [0.0, 0.0],  # null-class: 두 specialist 클래스 모두 해당 없음
}
LABEL2IDX   = {"authentic": 0, "manipulated": 1, "ai_generated": 2}
IDX2LABEL   = {0: "authentic", 1: "manipulated", 2: "ai_generated"}

IMG_EXTS      = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ── 경로 ──────────────────────────────────────────────────────────────────
CASIA2_AU        = ROOT / "datasets" / "CASIA2_subset" / "Au"
CASIA2_TP        = ROOT / "datasets" / "CASIA2_subset" / "Tp"
IMD2020_DIR      = ROOT / "datasets" / "IMD2020_subset" / \
                   "IMD2020_Generative_Image_Inpainting_yu2018_01" / "images"
GENIMAGE_NATURE  = ROOT / "datasets" / "GenImage_subset" / "BigGAN" / "val" / "nature"
GENIMAGE_AI      = ROOT / "datasets" / "GenImage_subset" / "BigGAN" / "val" / "ai"
GENIMAGE_AUTH_MAX = 3000
GENIMAGE_AI_MAX   = 3000  # 균형: auth/manip 비슷하게 맞춤

DSC_EVAL_JSONL = ROOT / "experiments" / "results" / "backbone_eval" / \
                 "mobilenetv2_dualstream_dsC_20260319_070725.jsonl"

V4_BEST_CKPT = ROOT / "weights" / "specialist_m_v4" / "specialist_m_v4_best.pth"

EVAL_DATASETS = {
    "base":       "experiments/results/phase2_patha_scale500_gain_predictor/patha_agent_outputs_20260304_080157.jsonl",
    "dsC":        "experiments/results/phase2_patha_case3_scale300_dsC/patha_agent_outputs_20260303_105005.jsonl",
    "opensdi":    "experiments/results/phase2_patha_case3_opensdi_scale300/patha_agent_outputs_fixed_seed42.jsonl",
    "aigenproxy": "experiments/results/phase2_patha_case3_aigenproxy_scale300/patha_agent_outputs_fixed_seed42.jsonl",
}

SE_DIR  = ROOT / "experiments" / "results" / "specialist_eval"


# ── 이미지 전처리 ─────────────────────────────────────────────────────────
class RandomJPEGCompression:
    def __init__(self, quality_range: Tuple[int, int] = (40, 95), p: float = 0.5):
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
    def __init__(self, std_range: Tuple[float, float] = (0.002, 0.015), p: float = 0.4):
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


# ── 모델 (v4와 동일한 backbone, head만 해석 방식 변경) ───────────────────

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


class SpecialistMSigmoid(nn.Module):
    """
    Sigmoid 출력 버전 SpecialistM.
    head의 구조는 v4와 동일하나 loss가 BCE(multi-label).
    추론 시: P(auth) = sigmoid(logit[0]), P(manip) = sigmoid(logit[1])
    잔여 확률 질량: R = 1 - P(auth) - P(manip)  (aigen 이미지에서 높아져야 함)
    """
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
            nn.Linear(256, 2),   # logit[0]=auth, logit[1]=manip
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw logits (2-dim)"""
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

    @torch.no_grad()
    def predict_probs(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            probs: (B, 2) — [P(auth), P(manip)]
            residual: (B,) — R = 1 - P(auth) - P(manip)  (implicit aigen mass)
        """
        logits   = self.forward(x)
        probs    = torch.sigmoid(logits)
        residual = (1.0 - probs[:, 0] - probs[:, 1]).clamp(0.0, 1.0)
        return probs, residual


# ── 손실 함수 ────────────────────────────────────────────────────────────

class SigmoidBCELoss(nn.Module):
    """
    Multi-label BCE 손실 + 잔여 질량 정규화 항.

    auth/manip 이미지에 대해 P(auth) + P(manip) ≤ 1을 유도하는
    soft constraint를 추가해 잔여 질량의 의미를 강화함.
    """
    def __init__(self, residual_reg: float = 0.1):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.residual_reg = residual_reg

    def forward(
        self,
        logits: torch.Tensor,     # (B, 2)
        targets: torch.Tensor,    # (B, 2) — [auth_target, manip_target]
        is_known: torch.Tensor,   # (B,) bool — auth/manip=True, aigen=False
    ) -> torch.Tensor:
        # Main BCE loss
        main_loss = self.bce(logits, targets)

        # Residual reg: auth/manip 이미지에서 P(auth)+P(manip) ≤ 1 강제
        # R = 1 - σ(auth) - σ(manip) > 0 이어야 함 (합 ≤ 1)
        if is_known.any() and self.residual_reg > 0:
            known_probs    = torch.sigmoid(logits[is_known])  # (K, 2)
            prob_sum       = known_probs.sum(dim=1)            # (K,)
            # Penalty if sum > 1 (L2 형태)
            excess         = F.relu(prob_sum - 1.0)
            reg_loss       = (excess ** 2).mean()
            return main_loss + self.residual_reg * reg_loss

        return main_loss


# ── 데이터셋 ─────────────────────────────────────────────────────────────

class SigmoidDataset(Dataset):
    def __init__(self, records: List[Dict], transform=None):
        self.records   = records
        self.transform = transform or VAL_TRANSFORM

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec    = self.records[idx]
        path   = ROOT / rec["image_path"]
        target = torch.tensor(LABEL2TARGET[rec["true_label"]], dtype=torch.float32)
        label  = LABEL2IDX[rec["true_label"]]

        try:
            img    = Image.open(path).convert("RGB")
            tensor = self.transform(img)
        except Exception:
            tensor = torch.zeros(3, 224, 224)

        is_known = rec["true_label"] in ("authentic", "manipulated")
        return tensor, target, label, rec["image_path"], is_known


# ── 데이터 수집 ───────────────────────────────────────────────────────────

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


def get_eval_aigen_filenames() -> set:
    """평가 JSONL에서 사용된 ai_generated 파일명 수집 (학습에서 제외)"""
    excluded = set()
    for rel_path in EVAL_DATASETS.values():
        p = ROOT / rel_path
        if not p.exists():
            continue
        with open(p) as f:
            for line in f:
                r = json.loads(line.strip())
                if r.get("true_label") == "ai_generated":
                    excluded.add(Path(r["image_path"]).name)
    return excluded


def scan_training_data(
    genimage_auth_max: int = GENIMAGE_AUTH_MAX,
    genimage_ai_max:   int = GENIMAGE_AI_MAX,
) -> List[Dict]:
    records = []
    excluded_imd   = get_dsc_eval_imd2020_filenames()
    excluded_aigen = get_eval_aigen_filenames()

    # Authentic — CASIA2
    au_casia = 0
    for p in sorted(CASIA2_AU.iterdir()):
        if p.suffix.lower() in IMG_EXTS:
            records.append({"image_path": str(p.relative_to(ROOT)),
                             "true_label": "authentic"})
            au_casia += 1

    # Authentic — GenImage nature
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

    # Manipulated — CASIA2 Tp
    tp_n = 0
    for p in sorted(CASIA2_TP.iterdir()):
        if p.suffix.lower() in IMG_EXTS:
            records.append({"image_path": str(p.relative_to(ROOT)),
                             "true_label": "manipulated"})
            tp_n += 1

    # Manipulated — IMD2020 (eval 제외)
    imd_n = 0
    if IMD2020_DIR.exists():
        for p in sorted(IMD2020_DIR.iterdir()):
            if p.suffix.lower() in IMG_EXTS and p.name not in excluded_imd:
                records.append({"image_path": str(p.relative_to(ROOT)),
                                 "true_label": "manipulated"})
                imd_n += 1

    # AI-generated — GenImage BigGAN (eval 제외)
    aigen_n = 0
    if GENIMAGE_AI.exists():
        ai_imgs = sorted(p for p in GENIMAGE_AI.iterdir()
                         if p.suffix.lower() in IMG_EXTS
                         and p.name not in excluded_aigen)
        rng = random.Random(42)
        rng.shuffle(ai_imgs)
        for p in ai_imgs[:genimage_ai_max]:
            records.append({"image_path": str(p.relative_to(ROOT)),
                             "true_label": "ai_generated"})
            aigen_n += 1

    total_au    = au_casia + au_genimg
    total_manip = tp_n + imd_n
    print(f"  Authentic — CASIA2: {au_casia}, GenImage_nature: {au_genimg}  → 합계: {total_au}")
    print(f"  Manipulated — CASIA2: {tp_n}, IMD2020(non-eval): {imd_n}  → 합계: {total_manip}")
    print(f"  AI-generated — GenImage BigGAN(non-eval): {aigen_n}")
    return records


def split_records(records: List[Dict], val_ratio: float = 0.15,
                  seed: int = 42) -> Tuple[List, List]:
    rng  = random.Random(seed)
    data = records[:]
    rng.shuffle(data)
    n    = int(len(data) * val_ratio)
    return data[n:], data[:n]


def make_sampler(records: List[Dict]) -> WeightedRandomSampler:
    """3-class balanced sampler"""
    labels = [LABEL2IDX[r["true_label"]] for r in records]
    counts = [labels.count(i) for i in range(3)]
    weights = [1.0 / max(counts[l], 1) for l in labels]
    return WeightedRandomSampler(weights, num_samples=len(records), replacement=True)


# ── 학습 / 평가 ──────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, scaler, criterion) -> Tuple[float, float]:
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    use_amp = DEVICE.type == "cuda"

    for imgs, targets, labels, _, is_known in loader:
        imgs      = imgs.to(DEVICE, non_blocking=True)
        targets   = targets.to(DEVICE, non_blocking=True)
        is_known  = is_known.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", enabled=use_amp):
            logits = model(imgs)
            loss   = criterion(logits, targets, is_known)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Accuracy 계산: argmax on [P(auth), P(manip), R]
        probs_2  = torch.sigmoid(logits).detach().cpu()
        residual = (1.0 - probs_2[:, 0] - probs_2[:, 1]).clamp(0, 1)
        scores_3 = torch.stack([probs_2[:, 0], probs_2[:, 1], residual], dim=1)
        preds    = scores_3.argmax(dim=1)
        correct += (preds == labels).sum().item()
        n       += len(labels)
        total_loss += loss.item() * len(labels)

    return total_loss / max(n, 1), correct / max(n, 1)


@torch.no_grad()
def evaluate_sigmoid(model, loader) -> Dict:
    model.eval()
    all_preds, all_labels = [], []
    auth_probs, manip_probs, residuals = [], [], []

    for imgs, targets, labels, _, _ in loader:
        imgs   = imgs.to(DEVICE, non_blocking=True)
        logits = model(imgs)
        probs  = torch.sigmoid(logits).cpu()
        R      = (1.0 - probs[:, 0] - probs[:, 1]).clamp(0, 1)

        scores_3 = torch.stack([probs[:, 0], probs[:, 1], R], dim=1)
        preds    = scores_3.argmax(dim=1)

        all_preds.append(preds)
        all_labels.append(labels)
        auth_probs.append(probs[:, 0])
        manip_probs.append(probs[:, 1])
        residuals.append(R)

    preds     = torch.cat(all_preds).numpy()
    labels    = torch.cat(all_labels).numpy()
    auth_p    = torch.cat(auth_probs).numpy()
    manip_p   = torch.cat(manip_probs).numpy()
    residual  = torch.cat(residuals).numpy()

    acc = float((preds == labels).mean())

    per_class = {}
    f1_list   = []
    for idx, name in IDX2LABEL.items():
        mask = labels == idx
        n_cls = int(mask.sum())
        if n_cls == 0:
            per_class[name] = {"recall": 0.0, "precision": 0.0, "f1": 0.0, "n": 0}
            continue
        tp   = int(((preds == idx) & (labels == idx)).sum())
        fp   = int(((preds == idx) & (labels != idx)).sum())
        fn   = int(((preds != idx) & (labels == idx)).sum())
        prec = tp / max(tp + fp, 1)
        rec  = tp / max(tp + fn, 1)
        f1   = 2 * prec * rec / max(prec + rec, 1e-8)
        per_class[name] = {"recall": rec, "precision": prec, "f1": f1,
                           "tp": tp, "fp": fp, "fn": fn, "n": n_cls}
        f1_list.append(f1)

    classes_present = [c for c in IDX2LABEL.values() if (labels == LABEL2IDX[c]).any()]
    macro_f1 = float(np.mean([per_class[c]["f1"] for c in classes_present]))

    # 잔여 질량 통계 (클래스별)
    residual_stats = {}
    for idx, name in IDX2LABEL.items():
        mask = labels == idx
        if mask.sum() > 0:
            residual_stats[name] = {
                "mean": float(residual[mask].mean()),
                "median": float(np.median(residual[mask])),
                "std": float(residual[mask].std()),
            }

    return {
        "accuracy":       acc,
        "macro_f1":       macro_f1,
        "per_class":      per_class,
        "residual_stats": residual_stats,
        "mean_auth_prob":   float(auth_p.mean()),
        "mean_manip_prob":  float(manip_p.mean()),
        "mean_residual":    float(residual.mean()),
    }


def load_jsonl_records(rel_path: str) -> List[Dict]:
    path = ROOT / rel_path
    if not path.exists():
        return []
    records = []
    with open(path) as f:
        for line in f:
            r = json.loads(line.strip())
            if r.get("true_label") in LABEL2IDX:
                records.append(r)
    return records


# ── 학습 루프 ─────────────────────────────────────────────────────────────

def run_training(args) -> Path:
    print("\n[학습 데이터 수집]")
    all_records = scan_training_data(
        genimage_auth_max=args.genimage_max,
        genimage_ai_max=args.genimage_ai_max,
    )
    train_recs, val_recs = split_records(all_records, val_ratio=0.15, seed=42)
    print(f"  train={len(train_recs)}, val={len(val_recs)}")

    train_ds = SigmoidDataset(train_recs, TRAIN_TRANSFORM)
    val_ds   = SigmoidDataset(val_recs,   VAL_TRANSFORM)

    sampler    = make_sampler(train_recs)
    train_ld   = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                            num_workers=4, pin_memory=(DEVICE.type == "cuda"))
    val_ld     = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False,
                            num_workers=4, pin_memory=(DEVICE.type == "cuda"))

    # 모델 초기화 + v4 checkpoint 로드
    model = SpecialistMSigmoid(pretrained=False).to(DEVICE)
    if V4_BEST_CKPT.exists():
        ckpt = torch.load(V4_BEST_CKPT, map_location="cpu")
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state, strict=True)
        print(f"\n[Resume] v4 체크포인트 로드: {V4_BEST_CKPT.name}")
    else:
        print(f"\n[Warning] v4 체크포인트 없음 — 랜덤 초기화")

    criterion = SigmoidBCELoss(residual_reg=args.residual_reg)

    # 학습률: backbone 낮게(1e-5), head 높게(lr) — v4 가중치 보존 + head 빠른 적응
    head_params     = list(model.head.parameters())
    backbone_params = [p for p in model.parameters()
                       if not any(p is hp for hp in head_params)]
    optimizer = AdamW([
        {"params": backbone_params, "lr": args.lr * 0.1},
        {"params": head_params,     "lr": args.lr},
    ], weight_decay=1e-4)

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler    = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

    out_dir = ROOT / "weights" / "specialist_m_sigmoid"
    out_dir.mkdir(parents=True, exist_ok=True)

    best_macro_f1 = 0.0
    best_ckpt_path = out_dir / "specialist_m_sigmoid_best.pth"

    print(f"\n[학습 시작] device={DEVICE}, epochs={args.epochs}, "
          f"backbone_lr={args.lr*0.1:.0e}, head_lr={args.lr:.0e}")
    print(f"{'Epoch':>5} {'Train-Loss':>12} {'Train-Acc':>10} "
          f"{'Val-MacroF1':>12} {'Val-Acc':>10} {'R(auth)':>8} {'R(manip)':>9} {'R(aigen)':>9}")
    print("-" * 90)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_ld, optimizer, scaler, criterion)
        val_stats = evaluate_sigmoid(model, val_ld)
        scheduler.step()

        r = val_stats["residual_stats"]
        r_auth  = r.get("authentic",    {}).get("mean", 0.0)
        r_manip = r.get("manipulated",  {}).get("mean", 0.0)
        r_aigen = r.get("ai_generated", {}).get("mean", 0.0)

        improved = ""
        if val_stats["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_stats["macro_f1"]
            torch.save({
                "epoch":              epoch,
                "model_state_dict":   model.state_dict(),
                "macro_f1":           val_stats["macro_f1"],
                "residual_stats":     val_stats["residual_stats"],
            }, best_ckpt_path)
            improved = " ★"

        print(f"{epoch:>5} {train_loss:>12.4f} {100*train_acc:>9.2f}% "
              f"{100*val_stats['macro_f1']:>11.2f}% {100*val_stats['accuracy']:>9.2f}% "
              f"{r_auth:>8.3f} {r_manip:>9.3f} {r_aigen:>9.3f}{improved}")

    print(f"\n[학습 완료] best macro_f1={100*best_macro_f1:.2f}% → {best_ckpt_path}")
    return best_ckpt_path


# ── 전체 데이터셋 평가 + JSONL 저장 ──────────────────────────────────────

def eval_all_datasets(ckpt_path: Path, args) -> Dict:
    print(f"\n[전체 데이터셋 평가] {ckpt_path.name}")

    model = SpecialistMSigmoid(pretrained=False).to(DEVICE)
    ckpt  = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    out_dir = ROOT / "experiments" / "results" / "hema"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    summary = {}

    for ds_key, jsonl_rel in EVAL_DATASETS.items():
        recs = load_jsonl_records(jsonl_rel)
        if not recs:
            print(f"  [{ds_key}] 레코드 없음")
            continue

        loader = DataLoader(
            SigmoidDataset(recs, VAL_TRANSFORM),
            batch_size=args.batch_size * 2, shuffle=False,
            num_workers=4, pin_memory=(DEVICE.type == "cuda"),
        )
        stats = evaluate_sigmoid(model, loader)

        # JSONL 저장 (HEMA 실험에서 사용)
        out_jsonl = out_dir / f"specm_sigmoid_{ds_key}_{ts}.jsonl"
        with open(out_jsonl, "w") as fout:
            for rec in recs:
                path   = ROOT / rec["image_path"]
                try:
                    img    = Image.open(path).convert("RGB")
                    tensor = VAL_TRANSFORM(img).unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        logits = model(tensor)
                        probs  = torch.sigmoid(logits).cpu().numpy()[0]
                        R      = float(max(0.0, 1.0 - probs[0] - probs[1]))
                except Exception as e:
                    probs = np.array([0.5, 0.5])
                    R     = 0.0

                scores_3 = [probs[0], probs[1], R]
                pred_idx  = int(np.argmax(scores_3))

                fout.write(json.dumps({
                    "image_path":      rec["image_path"],
                    "true_label":      rec["true_label"],
                    "pred_label":      IDX2LABEL[pred_idx],
                    "authentic_score": float(probs[0]),
                    "manip_score":     float(probs[1]),
                    "residual":        R,   # ← 핵심: implicit null-class mass
                    "confidence":      float(max(scores_3)),
                }, ensure_ascii=False) + "\n")

        r = stats["residual_stats"]
        auth_f1  = stats["per_class"].get("authentic",    {}).get("f1", 0.0)
        manip_f1 = stats["per_class"].get("manipulated",  {}).get("f1", 0.0)
        aigen_f1 = stats["per_class"].get("ai_generated", {}).get("f1", 0.0)

        summary[ds_key] = {
            "n":         len(recs),
            "accuracy":  stats["accuracy"],
            "macro_f1":  stats["macro_f1"],
            "auth_f1":   auth_f1,
            "manip_f1":  manip_f1,
            "aigen_f1":  aigen_f1,
            "residual_by_class": r,
            "jsonl":     str(out_jsonl),
        }

        print(
            f"  [{ds_key}] n={len(recs)} | "
            f"macro_f1={100*stats['macro_f1']:.2f}% | "
            f"auth={100*auth_f1:.2f}% manip={100*manip_f1:.2f}% aigen={100*aigen_f1:.2f}% | "
            f"R: auth={r.get('authentic',{}).get('mean',0):.3f} "
            f"manip={r.get('manipulated',{}).get('mean',0):.3f} "
            f"aigen={r.get('ai_generated',{}).get('mean',0):.3f}"
        )

    summary_path = out_dir / f"specm_sigmoid_summary_{ts}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n저장: {summary_path}")
    return summary


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SpecM Sigmoid 학습")
    parser.add_argument("--epochs",        type=int,   default=20)
    parser.add_argument("--batch-size",    type=int,   default=64)
    parser.add_argument("--lr",            type=float, default=3e-5,
                        help="head LR (backbone = lr × 0.1)")
    parser.add_argument("--genimage-max",  type=int,   default=GENIMAGE_AUTH_MAX)
    parser.add_argument("--genimage-ai-max", type=int, default=GENIMAGE_AI_MAX)
    parser.add_argument("--residual-reg",  type=float, default=0.1,
                        help="잔여 질량 soft constraint 가중치 (default: 0.1)")
    parser.add_argument("--eval-only",     type=str,   default=None,
                        help="평가만 실행 (체크포인트 경로 지정)")
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    if args.eval_only:
        eval_all_datasets(Path(args.eval_only), args)
    else:
        ckpt = run_training(args)
        eval_all_datasets(ckpt, args)


if __name__ == "__main__":
    main()
