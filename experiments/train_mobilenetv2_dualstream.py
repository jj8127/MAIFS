#!/usr/bin/env python3
"""
MobileNetV2 Dual-Stream Forensics
=================================

Track 1용 경량 noise backbone.
RGB branch + SRM residual branch를 결합해 3-class
(authentic / manipulated / ai_generated) 분류를 수행한다.

실행:
  .venv-qwen/bin/python experiments/train_mobilenetv2_dualstream.py
  .venv-qwen/bin/python experiments/train_mobilenetv2_dualstream.py --epochs 10 --batch-size 64
  .venv-qwen/bin/python experiments/train_mobilenetv2_dualstream.py --eval-only weights/mobilenetv2_dualstream/mobilenetv2_dualstream_best.pth
"""

from __future__ import annotations

import argparse
import json
import random
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
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from finetune_mobileclip import DATASETS, IDX2LABEL, LABEL_MAP, load_all_records, load_records, split_records

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.04),
    transforms.ToTensor(),
])

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])


def normalize_batch(x: torch.Tensor, mean, std) -> torch.Tensor:
    mean_t = torch.tensor(mean, device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
    std_t = torch.tensor(std, device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
    return (x - mean_t) / std_t


class ForensicsDataset(Dataset):
    def __init__(self, records: List[Dict], transform=None):
        self.records = records
        self.transform = transform or VAL_TRANSFORM

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img_path = ROOT / rec["image_path"]
        label = LABEL_MAP[rec["true_label"]]
        try:
            img = Image.open(img_path).convert("RGB")
            tensor = self.transform(img)
        except Exception:
            tensor = torch.zeros(3, 224, 224)
        return tensor, label, rec["image_path"]


class SRMNoiseExtractor(nn.Module):
    """
    간단한 고정 SRM filter bank.
    grayscale residual 3장을 만들어 noise branch 입력으로 사용한다.
    """

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
        # [0,1] 범위로 재매핑 후 ImageNet 정규화에 투입
        residual = (residual + 2.0) / 4.0
        return residual


class MobileNetBranch(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None
        base = models.mobilenet_v2(weights=weights)
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out_dim = 1280

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        feat = self.pool(feat).flatten(1)
        return feat


class DualStreamMobileNetV2(nn.Module):
    def __init__(self, pretrained: bool = True, dropout: float = 0.25):
        super().__init__()
        self.noise_extractor = SRMNoiseExtractor()
        self.rgb_branch = MobileNetBranch(pretrained=pretrained)
        self.noise_branch = MobileNetBranch(pretrained=pretrained)
        fused_dim = self.rgb_branch.out_dim + self.noise_branch.out_dim
        self.head = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rgb = normalize_batch(x, IMAGENET_MEAN, IMAGENET_STD)
        noise = self.noise_extractor(x)
        noise = normalize_batch(noise, IMAGENET_MEAN, IMAGENET_STD)
        rgb_feat = self.rgb_branch(rgb)
        noise_feat = self.noise_branch(noise)
        fused = torch.cat([rgb_feat, noise_feat], dim=1)
        return self.head(fused)


def compute_stats(labels: np.ndarray, preds: np.ndarray) -> Dict:
    acc = float((labels == preds).mean())
    per_class = {}
    recalls = []
    for idx, name in IDX2LABEL.items():
        mask = labels == idx
        if mask.sum() == 0:
            rec = 0.0
        else:
            rec = float((preds[mask] == idx).mean())
        per_class[name] = {"recall": rec, "n": int(mask.sum())}
        recalls.append(rec)
    return {
        "accuracy": acc,
        "macro_recall": float(np.mean(recalls)),
        "per_class": per_class,
    }


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader) -> Dict:
    model.eval()
    all_probs, all_preds, all_labels = [], [], []
    for imgs, labels, _ in loader:
        imgs = imgs.to(DEVICE, non_blocking=True)
        logits = model(imgs)
        probs = F.softmax(logits, dim=-1).cpu()
        preds = probs.argmax(1)
        all_probs.append(probs)
        all_preds.append(preds)
        all_labels.append(labels)
    probs = torch.cat(all_probs, dim=0).numpy()
    preds = torch.cat(all_preds, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()
    stats = compute_stats(labels, preds)
    stats["scores"] = probs
    stats["preds"] = preds
    stats["labels"] = labels
    return stats


def train_epoch(model: nn.Module, loader: DataLoader, optimizer, scaler) -> Tuple[float, float]:
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    use_amp = DEVICE.type == "cuda"
    for imgs, labels, _ in loader:
        imgs = imgs.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", enabled=use_amp):
            logits = model(imgs)
            loss = F.cross_entropy(logits, labels, label_smoothing=0.05)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * len(labels)
        correct += (logits.argmax(1) == labels).sum().item()
        n += len(labels)
    return total_loss / max(n, 1), correct / max(n, 1)


def run_training(args) -> Path:
    print("=" * 60)
    print("MobileNetV2 Dual-Stream Forensics")
    print(f"  epochs={args.epochs}, batch={args.batch_size}, lr={args.lr}, device={DEVICE}")
    print("=" * 60)

    print("\n[데이터 로드]")
    all_recs = load_all_records()
    train_recs, val_recs = split_records(all_recs, val_ratio=0.2, seed=42)
    print(f"  Train: {len(train_recs)}, Val: {len(val_recs)}")

    train_loader = DataLoader(
        ForensicsDataset(train_recs, TRAIN_TRANSFORM),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=(DEVICE.type == "cuda"),
    )
    val_loader = DataLoader(
        ForensicsDataset(val_recs, VAL_TRANSFORM),
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=4,
        pin_memory=(DEVICE.type == "cuda"),
    )

    print("\n[모델 로드]")
    model = DualStreamMobileNetV2(pretrained=True).to(DEVICE)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_total/1e6:.2f}M")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda"))

    ckpt_dir = ROOT / "weights" / "mobilenetv2_dualstream"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "mobilenetv2_dualstream_best.pth"

    best_macro = 0.0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scaler)
        val_stats = evaluate(model, val_loader)
        scheduler.step()
        macro = val_stats["macro_recall"]
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_acc": val_stats["accuracy"],
            "val_macro_recall": macro,
        })
        star = " ★" if macro > best_macro else ""
        print(
            f"  Epoch {epoch:02d}/{args.epochs} | loss={train_loss:.4f} | "
            f"train_acc={train_acc:.3f} | val_acc={val_stats['accuracy']:.3f} | "
            f"val_macro={macro:.3f}{star}"
        )
        if macro > best_macro:
            best_macro = macro
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "history": history,
                    "val_stats": {k: v for k, v in val_stats.items() if k not in {"scores", "preds", "labels"}},
                    "epoch": epoch,
                },
                ckpt_path,
            )
            print(f"  → 체크포인트 저장: {ckpt_path}")

    print(f"\n[완료] best_macro={best_macro:.4f}")
    return ckpt_path


@torch.no_grad()
def eval_all_datasets(ckpt_path: Path, args) -> Dict:
    print("\n[전체 데이터셋 평가]")
    model = DualStreamMobileNetV2(pretrained=False).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    out_dir = ROOT / "experiments" / "results" / "backbone_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    summary = {}
    for ds_key, ds_info in DATASETS.items():
        recs = load_records(ds_info["jsonl"])
        if not recs:
            continue
        loader = DataLoader(
            ForensicsDataset(recs, VAL_TRANSFORM),
            batch_size=args.batch_size * 2,
            shuffle=False,
            num_workers=4,
            pin_memory=(DEVICE.type == "cuda"),
        )
        stats = evaluate(model, loader)
        probs = stats.pop("scores")
        preds = stats.pop("preds")
        labels = stats.pop("labels")

        out_jsonl = out_dir / f"mobilenetv2_dualstream_{ds_key}_{ts}.jsonl"
        with open(out_jsonl, "w", encoding="utf-8") as f:
            for rec, pred_idx, prob_vec in zip(recs, preds.tolist(), probs.tolist()):
                pred_label = IDX2LABEL[pred_idx]
                f.write(
                    json.dumps(
                        {
                            "image_path": rec["image_path"],
                            "true_label": rec["true_label"],
                            "pred_label": pred_label,
                            "confidence": float(prob_vec[pred_idx]),
                            "scores": {IDX2LABEL[i]: float(prob_vec[i]) for i in range(3)},
                            "sub_type": rec.get("sub_type", ""),
                        },
                        ensure_ascii=False,
                    ) + "\n"
                )

        summary[ds_key] = {
            "desc": ds_info["desc"],
            "accuracy": stats["accuracy"],
            "macro_recall": stats["macro_recall"],
            "per_class": stats["per_class"],
            "jsonl": str(out_jsonl),
        }
        print(
            f"  [{ds_key}] acc={stats['accuracy']:.3f}, "
            f"macro_recall={stats['macro_recall']:.3f} -> {out_jsonl.name}"
        )

    summary_path = out_dir / f"mobilenetv2_dualstream_summary_{ts}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n저장: {summary_path}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--eval-only", type=str, default=None)
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    if args.eval_only:
        ckpt_path = Path(args.eval_only)
        eval_all_datasets(ckpt_path, args)
    else:
        ckpt_path = run_training(args)
        eval_all_datasets(ckpt_path, args)


if __name__ == "__main__":
    main()
