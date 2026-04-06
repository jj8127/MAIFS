#!/usr/bin/env python3
"""
Specialist-M: Manipulation vs Authentic Binary Classifier
=========================================================

"둘 다 틀리는" 에러 분석에서 발견된 핵심 실패 모드:
  - manipulated → authentic (미세 조작을 진본으로 오인)

해결 전략:
  1. Binary task (authentic=0, manipulated=1) — 3-class feature dilution 없음
  2. CASIA2 전용 학습 — 미세 조작 특화
  3. RGB + SRM(3ch) + DCT-approx(1ch) = 7ch 입력 — 물리 흔적 최대 포착
  4. Focal Loss — hard example(미세 조작) 집중 학습
  5. 출력: binary softmax + manipulation score (ICWMV 합의에서 가중치로 사용)

실행:
  .venv-qwen/bin/python experiments/train_specialist_m.py
  .venv-qwen/bin/python experiments/train_specialist_m.py --epochs 20
  .venv-qwen/bin/python experiments/train_specialist_m.py --eval-only weights/specialist_m/specialist_m_best.pth
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

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

BINARY_LABEL = {"authentic": 0, "manipulated": 1}
IDX2LABEL    = {0: "authentic", 1: "manipulated"}

TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.1),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),
    transforms.ToTensor(),
])

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

# ── 데이터셋 경로 ──────────────────────────────────────────────────────────
CASIA2_AU = ROOT / "datasets" / "CASIA2_subset" / "Au"
CASIA2_TP = ROOT / "datasets" / "CASIA2_subset" / "Tp"

# 4개 평가 데이터셋 (JSONL 기반, finetune_mobileclip 재사용)
EVAL_DATASETS = {
    "base":       "experiments/results/phase2_patha_scale500_gain_predictor/patha_agent_outputs_20260304_080157.jsonl",
    "dsC":        "experiments/results/phase2_patha_case3_scale300_dsC/patha_agent_outputs_20260303_105005.jsonl",
    "opensdi":    "experiments/results/phase2_patha_case3_opensdi_scale300/patha_agent_outputs_fixed_seed42.jsonl",
    "aigenproxy": "experiments/results/phase2_patha_case3_aigenproxy_scale300/patha_agent_outputs_fixed_seed42.jsonl",
}


# ── 모델 컴포넌트 ──────────────────────────────────────────────────────────
def normalize_batch(x: torch.Tensor, mean, std) -> torch.Tensor:
    t = torch.tensor(mean if len(mean) == x.shape[1] else mean[:1],
                     device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
    s = torch.tensor(std  if len(std)  == x.shape[1] else std[:1],
                     device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
    return (x - t) / s


class SRMExtractor(nn.Module):
    """3개 고정 SRM 필터 → 3ch noise residual."""
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
        return ((res.clamp(-2.0, 2.0) + 2.0) / 4.0)  # → [0,1]


class DCTResidualExtractor(nn.Module):
    """
    8x8 블록 DCT 근사 잔차 (1ch).
    저주파(DC) 성분 제거 후 남는 고주파 에너지.
    JPEG 조작 영역에서 블록 경계 불일치가 강하게 나타남.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gray     = x.mean(dim=1, keepdim=True)                        # [B,1,H,W]
        low_pass = F.avg_pool2d(gray, kernel_size=8, stride=8)        # [B,1,H/8,W/8]
        low_pass = F.interpolate(low_pass, size=gray.shape[-2:],
                                 mode='nearest')                       # [B,1,H,W]
        residual = (gray - low_pass).clamp(-1.0, 1.0)
        return (residual + 1.0) / 2.0                                  # → [0,1]


class MobileNetBranch(nn.Module):
    def __init__(self, in_channels: int = 3, pretrained: bool = True):
        super().__init__()
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None
        base    = models.mobilenet_v2(weights=weights)
        if in_channels != 3:
            # 첫 레이어 in_channels 수정
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
    """
    Binary: authentic(0) vs manipulated(1)
    입력: RGB(3) + SRM(3) + DCT-approx(1) = 7ch → 3개 branch
    """
    def __init__(self, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        self.srm_extractor = SRMExtractor()
        self.dct_extractor = DCTResidualExtractor()
        self.rgb_branch    = MobileNetBranch(in_channels=3, pretrained=pretrained)
        self.srm_branch    = MobileNetBranch(in_channels=3, pretrained=False)
        self.dct_branch    = MobileNetBranch(in_channels=1, pretrained=False)
        fused = self.rgb_branch.out_dim + self.srm_branch.out_dim + self.dct_branch.out_dim  # 3840
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
        f_rgb = self.rgb_branch(rgb)
        f_srm = self.srm_branch(srm)
        f_dct = self.dct_branch(dct)
        return self.head(torch.cat([f_rgb, f_srm, f_dct], dim=1))


# ── Focal Loss ─────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """
    감마가 높을수록 hard example(미세 조작)에 더 집중.
    """
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
        loss = w * (1 - pt) ** self.gamma * ce
        return loss.mean()


# ── 데이터 로드 ────────────────────────────────────────────────────────────
IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def scan_casia2() -> List[Dict]:
    """CASIA2 Au + Tp 직접 스캔 → binary record list."""
    records = []
    for p in sorted(CASIA2_AU.iterdir()):
        if p.suffix.lower() in IMG_EXTS:
            records.append({"image_path": str(p.relative_to(ROOT)),
                             "true_label": "authentic"})
    for p in sorted(CASIA2_TP.iterdir()):
        if p.suffix.lower() in IMG_EXTS:
            records.append({"image_path": str(p.relative_to(ROOT)),
                             "true_label": "manipulated"})
    return records


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


def split_records(records: List[Dict], val_ratio: float = 0.15,
                  seed: int = 42) -> Tuple[List, List]:
    rng  = random.Random(seed)
    data = records[:]
    rng.shuffle(data)
    n    = int(len(data) * val_ratio)
    return data[n:], data[:n]


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
        logits  = model(imgs)
        probs   = F.softmax(logits, dim=-1).cpu()
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
    # 클래스별 recall
    per_class = {}
    recalls   = []
    for idx, name in IDX2LABEL.items():
        mask = labels == idx
        rec  = float((preds[mask] == idx).mean()) if mask.sum() > 0 else 0.0
        per_class[name] = {"recall": rec, "n": int(mask.sum())}
        recalls.append(rec)
    # F1 for manipulated
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    prec   = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1_m   = 2 * prec * recall / max(prec + recall, 1e-8)
    return {
        "accuracy":       acc,
        "macro_recall":   float(np.mean(recalls)),
        "manip_f1":       f1_m,
        "manip_recall":   recall,
        "manip_precision": prec,
        "per_class":      per_class,
        "probs":          probs,
        "preds":          preds,
        "labels":         labels,
    }


# ── 메인 ──────────────────────────────────────────────────────────────────
def run_training(args) -> Path:
    print("=" * 60)
    print("Specialist-M: Manipulation vs Authentic")
    print(f"  epochs={args.epochs}, batch={args.batch_size}, "
          f"lr={args.lr}, focal_gamma={args.focal_gamma}, device={DEVICE}")
    print("=" * 60)

    print("\n[데이터 로드] CASIA2 전용 (Au + Tp)")
    all_recs = scan_casia2()
    au_n  = sum(1 for r in all_recs if r["true_label"] == "authentic")
    tp_n  = sum(1 for r in all_recs if r["true_label"] == "manipulated")
    print(f"  Authentic: {au_n}, Manipulated: {tp_n}, Total: {len(all_recs)}")

    train_recs, val_recs = split_records(all_recs, val_ratio=0.15, seed=42)
    print(f"  Train: {len(train_recs)}, Val: {len(val_recs)}")

    train_loader = DataLoader(BinaryDataset(train_recs, TRAIN_TRANSFORM),
                              batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=(DEVICE.type == "cuda"))
    val_loader   = DataLoader(BinaryDataset(val_recs, VAL_TRANSFORM),
                              batch_size=args.batch_size * 2, shuffle=False,
                              num_workers=4, pin_memory=(DEVICE.type == "cuda"))

    print("\n[모델 초기화]")
    model    = SpecialistM(pretrained=True).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params/1e6:.2f}M")

    criterion = FocalLoss(gamma=args.focal_gamma, alpha=0.6)  # alpha=0.6: manip 중시
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.05)
    scaler    = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda"))

    ckpt_dir  = ROOT / "weights" / "specialist_m"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "specialist_m_best.pth"

    best_f1   = 0.0
    history   = []
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
            f"manip_recall={val_stats['manip_recall']:.3f} | "
            f"manip_f1={f1_m:.3f}{star}"
        )
        history.append({"epoch": epoch, "train_loss": tr_loss,
                         "train_acc": tr_acc, "val_manip_f1": f1_m,
                         "val_manip_recall": val_stats["manip_recall"]})
        if f1_m > best_f1:
            best_f1 = f1_m
            torch.save({"model_state_dict": model.state_dict(),
                        "history": history,
                        "best_manip_f1": best_f1,
                        "epoch": epoch}, ckpt_path)
            print(f"  → 저장: {ckpt_path}")

    print(f"\n[완료] best manip_f1={best_f1:.4f}")
    return ckpt_path


@torch.no_grad()
def eval_all_datasets(ckpt_path: Path, args) -> Dict:
    """4개 데이터셋 평가. binary 관점(authentic/manipulated만 유효, ai_gen 제외)."""
    print("\n[전체 데이터셋 평가]")
    model = SpecialistM(pretrained=False).to(DEVICE)
    ckpt  = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    out_dir = ROOT / "experiments" / "results" / "specialist_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")

    summary = {}
    for ds_key, jsonl_rel in EVAL_DATASETS.items():
        recs = load_jsonl_records(jsonl_rel)
        # authentic + manipulated 만 평가 (binary specialist이므로)
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

        # JSONL 저장 (manipulation score 포함)
        out_jsonl = out_dir / f"specialist_m_{ds_key}_{ts}.jsonl"
        valid_recs = [r for r in recs_bin if r["true_label"] in BINARY_LABEL]
        with open(out_jsonl, "w") as f:
            for rec, pred_idx, prob_vec in zip(valid_recs, preds.tolist(), probs.tolist()):
                f.write(json.dumps({
                    "image_path":        rec["image_path"],
                    "true_label":        rec["true_label"],
                    "pred_label":        IDX2LABEL[pred_idx],
                    "manip_score":       float(prob_vec[1]),  # manipulation confidence
                    "authentic_score":   float(prob_vec[0]),
                    "confidence":        float(max(prob_vec)),
                }, ensure_ascii=False) + "\n")

        summary[ds_key] = {
            "n_binary": len(valid_recs),
            "accuracy": stats["accuracy"],
            "macro_recall": stats["macro_recall"],
            "manip_recall": stats["manip_recall"],
            "manip_f1": stats["manip_f1"],
            "per_class": stats["per_class"],
            "jsonl": str(out_jsonl),
        }
        print(
            f"  [{ds_key}] n={len(valid_recs)} | "
            f"acc={stats['accuracy']:.3f} | "
            f"manip_recall={stats['manip_recall']:.3f} | "
            f"manip_f1={stats['manip_f1']:.3f} → {out_jsonl.name}"
        )

    summary_path = out_dir / f"specialist_m_summary_{ts}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n저장: {summary_path}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",      type=int,   default=20)
    parser.add_argument("--batch-size",  type=int,   default=64)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--eval-only",   type=str,   default=None)
    args = parser.parse_args()

    random.seed(42); np.random.seed(42); torch.manual_seed(42)

    if args.eval_only:
        eval_all_datasets(Path(args.eval_only), args)
    else:
        ckpt = run_training(args)
        eval_all_datasets(ckpt, args)


if __name__ == "__main__":
    main()
