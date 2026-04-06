#!/usr/bin/env python3
"""
MobileCLIP-S2 Forensics Fine-tuning
======================================
4개 JSONL 데이터셋으로 MobileCLIP-S2 이미지 인코더를 forensics 도메인에 적응.

전략:
  Stage 1 (default): Linear probe — 인코더 frozen, 3-class linear head만 학습
  Stage 2 (--finetune-blocks N): 마지막 N개 transformer block + head 학습

실행:
  # Linear probe (기본)
  .venv-qwen/bin/python experiments/finetune_mobileclip.py

  # 마지막 4블록 fine-tune
  .venv-qwen/bin/python experiments/finetune_mobileclip.py --finetune-blocks 4

  # cross-dataset 평가 (leave-one-out)
  .venv-qwen/bin/python experiments/finetune_mobileclip.py --cross-dataset
"""

from __future__ import annotations

import argparse
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
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent  # MAIFS/

# ── 데이터셋 정의 ─────────────────────────────────────────────────────────── #
DATASETS: Dict[str, Dict] = {
    "base": {
        "jsonl": "experiments/results/phase2_patha_scale500_gain_predictor/patha_agent_outputs_20260304_080157.jsonl",
        "desc": "CASIA2 + BigGAN (1500)",
    },
    "dsC": {
        "jsonl": "experiments/results/phase2_patha_case3_scale300_dsC/patha_agent_outputs_20260303_105005.jsonl",
        "desc": "CASIA2 + IMD2020 + BigGAN (900)",
    },
    "opensdi": {
        "jsonl": "experiments/results/phase2_patha_case3_opensdi_scale300/patha_agent_outputs_fixed_seed42.jsonl",
        "desc": "OpenSDID (900)",
    },
    "aigenproxy": {
        "jsonl": "experiments/results/phase2_patha_case3_aigenproxy_scale300/patha_agent_outputs_fixed_seed42.jsonl",
        "desc": "AI-GenBench proxy (900)",
    },
}

LABEL_MAP = {"authentic": 0, "manipulated": 1, "ai_generated": 2}
IDX2LABEL = {v: k for k, v in LABEL_MAP.items()}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── 이미지 전처리 ─────────────────────────────────────────────────────────── #
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((288, 288)),
    transforms.RandomCrop(256),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
])

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
])


# ── 데이터셋 클래스 ───────────────────────────────────────────────────────── #
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
            # 이미지 로드 실패 → 검은 이미지
            tensor = torch.zeros(3, 256, 256)
        return tensor, label, rec["image_path"]


def load_records(jsonl_path: str) -> List[Dict]:
    """JSONL 파일에서 레코드 로드 (image_path + true_label만 추출)."""
    recs = []
    path = ROOT / jsonl_path
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if "true_label" not in d or d["true_label"] not in LABEL_MAP:
                continue
            img_path = ROOT / d["image_path"]
            if not img_path.exists():
                continue
            recs.append({"image_path": d["image_path"], "true_label": d["true_label"],
                          "sub_type": d.get("sub_type", "")})
    return recs


def load_all_records(exclude_key: Optional[str] = None) -> List[Dict]:
    """4개 데이터셋 통합 로드 (exclude_key 데이터셋 제외 가능)."""
    all_recs = []
    for key, info in DATASETS.items():
        if key == exclude_key:
            continue
        recs = load_records(info["jsonl"])
        print(f"  [{key}] {len(recs)}개 로드")
        all_recs.extend(recs)
    return all_recs


def split_records(records: List[Dict], val_ratio: float = 0.2,
                  seed: int = 42) -> Tuple[List[Dict], List[Dict]]:
    """Stratified train/val split."""
    rng = random.Random(seed)
    per_class: Dict[str, List] = {}
    for r in records:
        per_class.setdefault(r["true_label"], []).append(r)
    train, val = [], []
    for cls, recs in per_class.items():
        recs = recs[:]
        rng.shuffle(recs)
        n_val = max(1, int(len(recs) * val_ratio))
        val.extend(recs[:n_val])
        train.extend(recs[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


# ── 모델 ──────────────────────────────────────────────────────────────────── #
class ForensicsHead(nn.Module):
    """MobileCLIP-S2 위에 올라가는 3-class forensics head."""
    def __init__(self, embed_dim: int = 512, hidden_dim: int = 256,
                 num_classes: int = 3, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MobileCLIPForensics(nn.Module):
    """MobileCLIP-S2 인코더 + Forensics Head."""
    def __init__(self, clip_model, finetune_blocks: int = 0):
        super().__init__()
        self.clip = clip_model
        self.head = ForensicsHead(embed_dim=512)
        self.finetune_blocks = finetune_blocks
        self._freeze()

    def _freeze(self):
        """인코더 전체 frozen → 마지막 N 블록만 해제."""
        for p in self.clip.parameters():
            p.requires_grad = False
        if self.finetune_blocks > 0:
            # MobileCLIP-S2는 image_encoder.trunk.blocks 형태의 ViT 구조
            blocks = self._get_blocks()
            if blocks is not None:
                n = len(blocks)
                unfreeze_start = max(0, n - self.finetune_blocks)
                for blk in blocks[unfreeze_start:]:
                    for p in blk.parameters():
                        p.requires_grad = True
                # projection/norm도 해제
                for name, m in self.clip.named_modules():
                    if any(k in name for k in ['proj', 'post_layernorm', 'ln_post']):
                        for p in m.parameters():
                            p.requires_grad = True
                print(f"  Unfroze last {self.finetune_blocks} blocks (out of {n})")
            else:
                print("  WARNING: blocks 구조 감지 실패 — 인코더 전체 frozen 유지")

    def _get_blocks(self):
        """MobileCLIP-S2 (FastViT) 내부 블록 시퀀스 탐색.
        구조: model.visual.trunk (FastViT) → trunk.stages[i].blocks
        모든 stage의 블록을 순서대로 flatten하여 반환.
        """
        visual = getattr(self.clip, 'visual', None)
        if visual is None:
            return None
        trunk = getattr(visual, 'trunk', None)
        if trunk is None:
            return None
        # FastViT: trunk.stages[i].blocks
        stages = getattr(trunk, 'stages', None)
        if stages is not None:
            all_blocks = []
            for stage in stages:
                stage_blocks = getattr(stage, 'blocks', None)
                if stage_blocks is not None:
                    all_blocks.extend(list(stage_blocks))
            if all_blocks:
                return all_blocks
        # fallback: trunk.blocks
        blocks = getattr(trunk, 'blocks', None)
        if blocks is not None:
            return list(blocks)
        return None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.clip.encode_image(x)          # (B, 512)
        feat = F.normalize(feat, dim=-1)
        return self.head(feat)                     # (B, 3)

    def trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]


# ── 학습 ──────────────────────────────────────────────────────────────────── #
def train_epoch(model: MobileCLIPForensics, loader: DataLoader,
                optimizer, scheduler, epoch: int) -> float:
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for imgs, labels, _ in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = F.cross_entropy(logits, labels, label_smoothing=0.05)
        loss.backward()
        nn.utils.clip_grad_norm_(model.trainable_params(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(labels)
        correct += (logits.argmax(1) == labels).sum().item()
        n += len(labels)
    scheduler.step()
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model: MobileCLIPForensics, loader: DataLoader) -> Dict:
    model.eval()
    all_preds, all_labels = [], []
    for imgs, labels, _ in loader:
        imgs = imgs.to(DEVICE)
        logits = model(imgs)
        preds = logits.argmax(1).cpu()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    acc = (all_preds == all_labels).mean()
    per_class = {}
    for idx, name in IDX2LABEL.items():
        mask = all_labels == idx
        if mask.sum() == 0:
            per_class[name] = {"recall": 0.0, "n": 0}
            continue
        recall = (all_preds[mask] == idx).mean()
        per_class[name] = {"recall": float(recall), "n": int(mask.sum())}
    macro_recall = np.mean([v["recall"] for v in per_class.values()])
    return {"accuracy": float(acc), "macro_recall": float(macro_recall),
            "per_class": per_class}


def run_training(args) -> Path:
    """메인 학습 루프. 체크포인트 경로 반환."""
    print("\n" + "="*60)
    print(f"MobileCLIP-S2 Forensics Fine-tuning")
    print(f"  finetune_blocks={args.finetune_blocks}, epochs={args.epochs}")
    print(f"  lr={args.lr}, batch={args.batch_size}, device={DEVICE}")
    print("="*60)

    # 데이터 로드
    print("\n[데이터 로드]")
    all_recs = load_all_records()
    train_recs, val_recs = split_records(all_recs, val_ratio=0.2, seed=42)
    print(f"  Train: {len(train_recs)}, Val: {len(val_recs)}")

    train_ds = ForensicsDataset(train_recs, TRAIN_TRANSFORM)
    val_ds = ForensicsDataset(val_recs, VAL_TRANSFORM)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2,
                            shuffle=False, num_workers=4, pin_memory=True)

    # 모델 로드
    print("\n[모델 로드]")
    import open_clip
    clip_model, _, _ = open_clip.create_model_and_transforms(
        "MobileCLIP-S2", pretrained="datacompdr",
        cache_dir=str(ROOT / "weights" / "mobileclip")
    )
    model = MobileCLIPForensics(clip_model,
                                finetune_blocks=args.finetune_blocks).to(DEVICE)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {n_trainable/1e6:.2f}M / Total: {n_total/1e6:.2f}M")

    # Optimizer
    optimizer = AdamW(model.trainable_params(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    # 학습 루프
    print("\n[학습 시작]")
    best_macro = 0.0
    best_epoch = 0
    ckpt_dir = ROOT / "weights" / "mobileclip_forensics"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"mobileclip_s2_forensics_ft{args.finetune_blocks}.pth"

    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, epoch)
        val_stats = evaluate(model, val_loader)
        macro = val_stats["macro_recall"]
        history.append({"epoch": epoch, "train_loss": train_loss,
                         "train_acc": train_acc, "val_acc": val_stats["accuracy"],
                         "val_macro_recall": macro})
        star = " ★" if macro > best_macro else ""
        print(f"  Epoch {epoch:3d}/{args.epochs} | loss={train_loss:.4f} | "
              f"train_acc={train_acc:.3f} | val_acc={val_stats['accuracy']:.3f} | "
              f"val_macro_recall={macro:.3f}{star}")
        for cls, s in val_stats["per_class"].items():
            print(f"    {cls:12s}: recall={s['recall']:.3f}")
        if macro > best_macro:
            best_macro = macro
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "finetune_blocks": args.finetune_blocks,
                "val_stats": val_stats,
                "history": history,
            }, ckpt_path)
            print(f"  → 체크포인트 저장: {ckpt_path}")

    print(f"\n[완료] Best epoch={best_epoch}, best_macro_recall={best_macro:.4f}")
    print(f"  체크포인트: {ckpt_path}")
    return ckpt_path


# ── 파인튜닝 모델로 재평가 ────────────────────────────────────────────────── #
@torch.no_grad()
def eval_all_datasets(ckpt_path: Path, args) -> Dict:
    """4개 데이터셋 전체에 fine-tuned 모델 적용하여 결과 저장."""
    print("\n[Fine-tuned 모델 전체 데이터셋 평가]")

    import open_clip
    clip_model, _, _ = open_clip.create_model_and_transforms(
        "MobileCLIP-S2", pretrained="datacompdr",
        cache_dir=str(ROOT / "weights" / "mobileclip")
    )
    model = MobileCLIPForensics(clip_model, finetune_blocks=args.finetune_blocks).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"  로드: {ckpt_path} (epoch={ckpt['epoch']})")

    out_dir = ROOT / "experiments" / "results" / "backbone_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    all_summary = {}
    for ds_key, ds_info in DATASETS.items():
        recs = load_records(ds_info["jsonl"])
        if not recs:
            continue
        ds = ForensicsDataset(recs, VAL_TRANSFORM)
        loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4)

        out_recs = []
        all_preds, all_labels = [], []
        for imgs, labels, paths in loader:
            imgs = imgs.to(DEVICE)
            logits = model(imgs)
            probs = F.softmax(logits, dim=-1).cpu()
            preds = probs.argmax(1)
            for i in range(len(labels)):
                out_recs.append({
                    "image_path": paths[i],
                    "true_label": IDX2LABEL[labels[i].item()],
                    "pred_label": IDX2LABEL[preds[i].item()],
                    "confidence": probs[i][preds[i]].item(),
                    "scores": {IDX2LABEL[j]: probs[i][j].item() for j in range(3)},
                })
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

        # JSONL 저장
        out_jsonl = out_dir / f"mobileclip_s2_finetuned_{ds_key}_{ts}.jsonl"
        with open(out_jsonl, "w") as f:
            for r in out_recs:
                f.write(json.dumps(r) + "\n")

        # 통계
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        acc = (all_preds == all_labels).mean()
        per_class = {}
        for idx, name in IDX2LABEL.items():
            mask = all_labels == idx
            if mask.sum() == 0:
                per_class[name] = {"recall": 0.0, "n": 0}
                continue
            recall = (all_preds[mask] == idx).mean()
            per_class[name] = {"recall": float(recall), "n": int(mask.sum())}
        macro = np.mean([v["recall"] for v in per_class.values()])
        all_summary[ds_key] = {
            "desc": ds_info["desc"], "n": len(recs),
            "accuracy": float(acc), "macro_recall": float(macro),
            "per_class": per_class,
        }
        print(f"\n  [{ds_key}] acc={acc:.3f}, macro_recall={macro:.3f}")
        for cls, s in per_class.items():
            print(f"    {cls:12s}: recall={s['recall']:.3f}  (n={s['n']})")
        print(f"    → {out_jsonl.name}")

    # 요약 JSON 저장
    summary_path = out_dir / f"mobileclip_s2_finetuned_summary_{ts}.json"
    with open(summary_path, "w") as f:
        json.dump(all_summary, f, indent=2, ensure_ascii=False)
    print(f"\n  요약 저장: {summary_path}")
    return all_summary


# ── Cross-dataset 평가 ────────────────────────────────────────────────────── #
def run_cross_dataset(args):
    """Leave-one-out cross-dataset 평가: 3개로 학습, 1개로 테스트."""
    import open_clip
    print("\n[Cross-Dataset Evaluation (Leave-One-Out)]")
    results = {}
    for held_out in DATASETS.keys():
        print(f"\n  Held-out: {held_out}")
        # 학습 데이터
        train_recs = load_all_records(exclude_key=held_out)
        train_recs, val_recs = split_records(train_recs, val_ratio=0.15, seed=42)
        # 테스트 데이터 (held-out 전체)
        test_recs = load_records(DATASETS[held_out]["jsonl"])

        train_ds = ForensicsDataset(train_recs, TRAIN_TRANSFORM)
        val_ds = ForensicsDataset(val_recs, VAL_TRANSFORM)
        test_ds = ForensicsDataset(test_recs, VAL_TRANSFORM)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4)

        clip_model, _, _ = open_clip.create_model_and_transforms(
            "MobileCLIP-S2", pretrained="datacompdr",
            cache_dir=str(ROOT / "weights" / "mobileclip")
        )
        model = MobileCLIPForensics(clip_model,
                                    finetune_blocks=args.finetune_blocks).to(DEVICE)
        optimizer = AdamW(model.trainable_params(), lr=args.lr, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

        best_macro, best_state = 0.0, None
        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, epoch)
            val_stats = evaluate(model, val_loader)
            macro = val_stats["macro_recall"]
            if macro > best_macro:
                best_macro = macro
                best_state = {k: v.clone() if hasattr(v, 'clone') else v
                              for k, v in model.state_dict().items()}
            if epoch % 5 == 0:
                print(f"    ep{epoch}: loss={train_loss:.4f} val_macro={macro:.3f}")

        if best_state:
            model.load_state_dict(best_state)
        test_stats = evaluate(model, test_loader)
        results[held_out] = test_stats
        print(f"  [{held_out}] test_acc={test_stats['accuracy']:.3f}, "
              f"test_macro={test_stats['macro_recall']:.3f}")
        for cls, s in test_stats["per_class"].items():
            print(f"    {cls:12s}: recall={s['recall']:.3f}")

    print("\n[Cross-Dataset 요약]")
    for k, v in results.items():
        print(f"  {k:12s}: acc={v['accuracy']:.3f}, macro={v['macro_recall']:.3f}")
    return results


# ── Main ─────────────────────────────────────────────────────────────────── #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--finetune-blocks", type=int, default=0,
                        help="마지막 N개 블록 unfreeze (0=linear probe)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--cross-dataset", action="store_true",
                        help="Leave-one-out cross-dataset 평가")
    parser.add_argument("--eval-only", type=str, default=None,
                        help="기존 체크포인트 경로로 평가만 실행")
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)

    if args.eval_only:
        ckpt_path = Path(args.eval_only)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        args.finetune_blocks = ckpt.get("finetune_blocks", 0)
        eval_all_datasets(ckpt_path, args)
    elif args.cross_dataset:
        run_cross_dataset(args)
    else:
        ckpt_path = run_training(args)
        eval_all_datasets(ckpt_path, args)


if __name__ == "__main__":
    main()
