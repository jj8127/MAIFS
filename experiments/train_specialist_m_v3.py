#!/usr/bin/env python3
"""
Specialist-M v3: Manipulation vs Authentic — Authentic OOD 강건화
=================================================================

v2 대비 변경 사항:
  1. Authentic 다양화: GenImage nature 실제 사진 추가 (ImageNet val, 최대 3000장)
     → CASIA2만으로 학습 시 outdoor/nature authentic을 manipulated로 오분류하는
       편향 제거 (openSDI authentic recall v2=11% → v3 목표 ≥60%)
  2. RandomErasing 추가: 조작 영역 패치를 시뮬레이션 (inpainting-style 학습)
     → diffusion 기반 inpainting (openSDI) 탐지력 향상 목표
  3. 나머지 augmentation은 v2와 동일 유지

v3 학습 데이터:
  Authentic: CASIA2 Au (~7491) + GenImage nature (최대 3000장)
  Manipulated: CASIA2 Tp (~5123) + IMD2020 non-eval (~1710)

실행:
  .venv-qwen/bin/python experiments/train_specialist_m_v3.py
  .venv-qwen/bin/python experiments/train_specialist_m_v3.py --epochs 30 --lr 5e-5
  .venv-qwen/bin/python experiments/train_specialist_m_v3.py --eval-only weights/specialist_m_v3/specialist_m_v3_best.pth
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
from typing import Dict, List, Tuple

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

BINARY_LABEL  = {"authentic": 0, "manipulated": 1}
IDX2LABEL     = {0: "authentic", 1: "manipulated"}
IMG_EXTS      = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".jpeg"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ── 경로 설정 ─────────────────────────────────────────────────────────────
CASIA2_AU   = ROOT / "datasets" / "CASIA2_subset" / "Au"
CASIA2_TP   = ROOT / "datasets" / "CASIA2_subset" / "Tp"
IMD2020_DIR = ROOT / "datasets" / "IMD2020_subset" / \
              "IMD2020_Generative_Image_Inpainting_yu2018_01" / "images"
# v3 추가: GenImage nature (ImageNet val real photos, ~6000장)
GENIMAGE_NATURE = ROOT / "datasets" / "GenImage_subset" / "BigGAN" / "val" / "nature"
GENIMAGE_AUTH_MAX = 3000  # authentic 클래스 균형 유지를 위해 cap

DSC_EVAL_JSONL = ROOT / "experiments" / "results" / "backbone_eval" / \
                 "mobilenetv2_dualstream_dsC_20260319_070725.jsonl"

EVAL_DATASETS = {
    "base":       "experiments/results/phase2_patha_scale500_gain_predictor/patha_agent_outputs_20260304_080157.jsonl",
    "dsC":        "experiments/results/phase2_patha_case3_scale300_dsC/patha_agent_outputs_20260303_105005.jsonl",
    "opensdi":    "experiments/results/phase2_patha_case3_opensdi_scale300/patha_agent_outputs_fixed_seed42.jsonl",
    "aigenproxy": "experiments/results/phase2_patha_case3_aigenproxy_scale300/patha_agent_outputs_fixed_seed42.jsonl",
}


# ── 커스텀 Augmentation ──────────────────────────────────────────────────
class RandomJPEGCompression:
    """JPEG 랜덤 재압축 — 다양한 카메라/소셜미디어 압축 조건 시뮬레이션"""
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
    """텐서에 Gaussian 노이즈 추가 — 다양한 카메라 센서 노이즈 시뮬레이션"""
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
    # v3 추가: inpainting-style 패치 학습 (diffusion 조작 영역 시뮬레이션)
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.20), ratio=(0.3, 3.3), value=0),
])

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])


# ── 모델 컴포넌트 (v2와 동일) ───────────────────────────────────────────
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


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float = 0.5):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce   = F.cross_entropy(logits, targets, reduction="none")
        pt   = torch.exp(-ce)
        w    = torch.where(targets == 1,
                           torch.full_like(pt, self.alpha),
                           torch.full_like(pt, 1 - self.alpha))
        return (w * (1 - pt) ** self.gamma * ce).mean()


# ── 데이터 수집 ──────────────────────────────────────────────────────────
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


def scan_training_data(genimage_auth_max: int = GENIMAGE_AUTH_MAX) -> List[Dict]:
    """
    학습 데이터 수집 (v3):
      Authentic: CASIA2 Au + GenImage nature (최대 genimage_auth_max장)
      Manipulated: CASIA2 Tp + IMD2020 (non-eval)
    """
    records = []
    excluded_imd = get_dsc_eval_imd2020_filenames()

    # CASIA2 Authentic
    au_casia = 0
    for p in sorted(CASIA2_AU.iterdir()):
        if p.suffix.lower() in IMG_EXTS:
            records.append({"image_path": str(p.relative_to(ROOT)),
                             "true_label": "authentic"})
            au_casia += 1

    # v3 추가: GenImage nature authentic (ImageNet val real photos)
    au_genimg = 0
    if GENIMAGE_NATURE.exists():
        nature_imgs = sorted(p for p in GENIMAGE_NATURE.iterdir()
                             if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".jpeg"})
        # cap 적용 후 셔플 (재현성)
        rng = random.Random(42)
        rng.shuffle(nature_imgs)
        for p in nature_imgs[:genimage_auth_max]:
            records.append({"image_path": str(p.relative_to(ROOT)),
                             "true_label": "authentic"})
            au_genimg += 1

    # CASIA2 Manipulated
    tp_n = 0
    for p in sorted(CASIA2_TP.iterdir()):
        if p.suffix.lower() in IMG_EXTS:
            records.append({"image_path": str(p.relative_to(ROOT)),
                             "true_label": "manipulated"})
            tp_n += 1

    # IMD2020 Manipulated (non-eval)
    imd_n = 0
    if IMD2020_DIR.exists():
        for p in sorted(IMD2020_DIR.iterdir()):
            if p.suffix.lower() in IMG_EXTS and p.name not in excluded_imd:
                records.append({"image_path": str(p.relative_to(ROOT)),
                                 "true_label": "manipulated"})
                imd_n += 1

    total_au = au_casia + au_genimg
    total_manip = tp_n + imd_n
    print(f"  Authentic — CASIA2: {au_casia}, GenImage_nature: {au_genimg}  → 합계: {total_au}")
    print(f"  Manipulated — CASIA2: {tp_n}, IMD2020(non-eval): {imd_n}  → 합계: {total_manip}")
    return records


def split_records(records: List[Dict], val_ratio: float = 0.15,
                  seed: int = 42) -> Tuple[List, List]:
    rng  = random.Random(seed)
    data = records[:]
    rng.shuffle(data)
    n    = int(len(data) * val_ratio)
    return data[n:], data[:n]


def make_sampler(records: List[Dict]) -> WeightedRandomSampler:
    """클래스 균형 WeightedRandomSampler (authentic:manipulated ≈ 1:1)"""
    labels = [BINARY_LABEL[r["true_label"]] for r in records]
    class_counts = [labels.count(0), labels.count(1)]
    weights = [1.0 / class_counts[l] for l in labels]
    return WeightedRandomSampler(weights, num_samples=len(records), replacement=True)


class BinaryDataset(Dataset):
    def __init__(self, records: List[Dict], transform=None):
        self.records   = records
        self.transform = transform or VAL_TRANSFORM

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec   = self.records[idx]
        path  = ROOT / rec["image_path"]
        label = BINARY_LABEL.get(rec["true_label"], -1)
        try:
            img    = Image.open(path).convert("RGB")
            tensor = self.transform(img)
        except Exception:
            tensor = torch.zeros(3, 224, 224)
        return tensor, label, rec["image_path"]


# ── 학습/평가 ─────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, scaler, criterion) -> Tuple[float, float]:
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    use_amp = DEVICE.type == "cuda"
    for imgs, labels, _ in loader:
        imgs   = imgs.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", enabled=use_amp):
            logits = model(imgs)
            loss   = criterion(logits, labels)
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
    all_preds, all_probs, all_labels = [], [], []
    for imgs, labels, _ in loader:
        imgs    = imgs.to(DEVICE, non_blocking=True)
        probs   = F.softmax(model(imgs), dim=-1).cpu()
        preds   = probs.argmax(1)
        all_probs.append(probs)
        all_preds.append(preds)
        all_labels.append(labels)
    probs  = torch.cat(all_probs).numpy()
    preds  = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    valid  = labels != -1
    preds, labels, probs = preds[valid], labels[valid], probs[valid]
    acc = float((preds == labels).mean())
    per_class, recalls = {}, []
    for idx, name in IDX2LABEL.items():
        mask = labels == idx
        rec  = float((preds[mask] == idx).mean()) if mask.sum() > 0 else 0.0
        per_class[name] = {"recall": rec, "n": int(mask.sum())}
        recalls.append(rec)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    prec   = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1_m   = 2 * prec * recall / max(prec + recall, 1e-8)
    return {
        "accuracy":        acc,
        "macro_recall":    float(np.mean(recalls)),
        "manip_f1":        f1_m,
        "manip_recall":    recall,
        "manip_precision": prec,
        "per_class":       per_class,
        "probs":           probs,
        "preds":           preds,
        "labels":          labels,
    }


def load_jsonl_records(rel_path: str) -> List[Dict]:
    path = ROOT / rel_path
    if not path.exists():
        return []
    records = []
    with open(path) as f:
        for line in f:
            r = json.loads(line.strip())
            if r.get("true_label") in BINARY_LABEL:
                records.append(r)
    return records


# ── 메인 ──────────────────────────────────────────────────────────────────
def run_training(args) -> Path:
    print("=" * 60)
    print("Specialist-M v3: Authentic OOD 강건화")
    print(f"  epochs={args.epochs}, batch={args.batch_size}, "
          f"lr={args.lr}, focal_gamma={args.focal_gamma}, device={DEVICE}")
    print(f"  v3 추가: GenImage nature (최대 {args.genimage_max}장) + RandomErasing")
    print("=" * 60)

    print("\n[데이터 수집] CASIA2 + IMD2020 + GenImage nature")
    all_recs = scan_training_data(genimage_auth_max=args.genimage_max)
    au_n  = sum(1 for r in all_recs if r["true_label"] == "authentic")
    tp_n  = sum(1 for r in all_recs if r["true_label"] == "manipulated")
    print(f"  Total — Authentic: {au_n}, Manipulated: {tp_n}, Total: {len(all_recs)}")

    train_recs, val_recs = split_records(all_recs, val_ratio=0.15, seed=42)
    print(f"  Train: {len(train_recs)}, Val: {len(val_recs)}")

    sampler = make_sampler(train_recs)
    train_loader = DataLoader(BinaryDataset(train_recs, TRAIN_TRANSFORM),
                              batch_size=args.batch_size,
                              sampler=sampler,
                              num_workers=4, pin_memory=(DEVICE.type == "cuda"))
    val_loader   = DataLoader(BinaryDataset(val_recs, VAL_TRANSFORM),
                              batch_size=args.batch_size * 2, shuffle=False,
                              num_workers=4, pin_memory=(DEVICE.type == "cuda"))

    print("\n[모델 초기화]")
    model    = SpecialistM(pretrained=True).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params/1e6:.2f}M")
    print(f"  Augmentation: JPEG(40-95) + GaussianNoise + ColorJitter + RandomErasing(p=0.3)")
    print(f"  Sampler: WeightedRandom (1:1 class balance)")

    criterion = FocalLoss(gamma=args.focal_gamma, alpha=0.55)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.05)
    scaler    = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda"))

    ckpt_dir  = ROOT / "weights" / "specialist_m_v3"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "specialist_m_v3_best.pth"

    best_f1, history = 0.0, []
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, scaler, criterion)
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
            f"manip_f1={f1_m:.3f}{star}"
        )
        history.append({"epoch": epoch, "train_loss": tr_loss,
                         "train_acc": tr_acc, "val_manip_f1": f1_m,
                         "val_auth_recall": val_stats["per_class"]["authentic"]["recall"],
                         "val_manip_recall": val_stats["manip_recall"]})
        if f1_m > best_f1:
            best_f1 = f1_m
            torch.save({"model_state_dict": model.state_dict(),
                        "history": history,
                        "best_manip_f1": best_f1,
                        "epoch": epoch,
                        "version": "v3"}, ckpt_path)
            print(f"  → 저장: {ckpt_path}")

    print(f"\n[완료] best manip_f1={best_f1:.4f}")
    return ckpt_path


@torch.no_grad()
def eval_all_datasets(ckpt_path: Path, args) -> Dict:
    print("\n[전체 데이터셋 평가]")
    model = SpecialistM(pretrained=False).to(DEVICE)
    ckpt  = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    out_dir = ROOT / "experiments" / "results" / "specialist_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    summary = {}
    for ds_key, jsonl_rel in EVAL_DATASETS.items():
        recs = load_jsonl_records(jsonl_rel)
        recs_bin = [r for r in recs if r["true_label"] in BINARY_LABEL]
        if not recs_bin:
            print(f"  [{ds_key}] 유효 레코드 없음")
            continue

        loader = DataLoader(BinaryDataset(recs_bin, VAL_TRANSFORM),
                            batch_size=args.batch_size * 2, shuffle=False,
                            num_workers=4, pin_memory=(DEVICE.type == "cuda"))
        stats  = evaluate_binary(model, loader)
        probs  = stats.pop("probs")
        preds  = stats.pop("preds")
        labels = stats.pop("labels")

        out_jsonl = out_dir / f"specialist_m_v3_{ds_key}_{ts}.jsonl"
        with open(out_jsonl, "w") as f:
            for rec, pred_idx, prob_vec in zip(recs_bin, preds.tolist(), probs.tolist()):
                f.write(json.dumps({
                    "image_path":      rec["image_path"],
                    "true_label":      rec["true_label"],
                    "pred_label":      IDX2LABEL[pred_idx],
                    "manip_score":     float(prob_vec[1]),
                    "authentic_score": float(prob_vec[0]),
                    "confidence":      float(max(prob_vec)),
                }, ensure_ascii=False) + "\n")

        au_rec  = stats["per_class"]["authentic"]["recall"]
        summary[ds_key] = {
            "n_binary":     len(recs_bin),
            "accuracy":     stats["accuracy"],
            "macro_recall": stats["macro_recall"],
            "auth_recall":  au_rec,
            "manip_recall": stats["manip_recall"],
            "manip_f1":     stats["manip_f1"],
            "per_class":    stats["per_class"],
            "jsonl":        str(out_jsonl),
        }
        print(
            f"  [{ds_key}] n={len(recs_bin)} | "
            f"auth_recall={au_rec:.3f} | "
            f"manip_recall={stats['manip_recall']:.3f} | "
            f"manip_f1={stats['manip_f1']:.3f} → {out_jsonl.name}"
        )

    summary_path = out_dir / f"specialist_m_v3_summary_{ts}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n저장: {summary_path}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",       type=int,   default=30)
    parser.add_argument("--batch-size",   type=int,   default=64)
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--focal-gamma",  type=float, default=2.0)
    parser.add_argument("--genimage-max", type=int,   default=GENIMAGE_AUTH_MAX,
                        help="GenImage nature 최대 사용 장수 (default: 3000)")
    parser.add_argument("--eval-only",    type=str,   default=None)
    args = parser.parse_args()

    random.seed(42); np.random.seed(42); torch.manual_seed(42)

    if args.eval_only:
        eval_all_datasets(Path(args.eval_only), args)
    else:
        ckpt = run_training(args)
        eval_all_datasets(ckpt, args)


if __name__ == "__main__":
    main()
