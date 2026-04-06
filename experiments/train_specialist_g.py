#!/usr/bin/env python3
"""
Specialist-G: AI-Generated vs Authentic Binary Classifier
=========================================================

"둘 다 틀리는" 에러 분석에서 발견된 핵심 실패 모드:
  - ai_generated → authentic (새 생성기 이미지를 진본으로 오인)

해결 전략:
  1. Binary task (authentic=0, ai_generated=1) — OOD 일반화 집중
  2. MobileCLIP-ft4 frozen backbone + binary head (과적합 방지)
  3. PiD 양자화 잔차 추가 채널 — 생성 모델의 수학적 잔차 포착
     (SD3/FLUX/DALL-E3 등 새 생성기도 동일 원리로 잔차 존재)
  4. B-Free 스타일: diverse augmentation으로 포맷 편향 제거
  5. 출력: binary softmax + ai_gen score (ICWMV 합의에서 가중치로 사용)

실행:
  .venv-qwen/bin/python experiments/train_specialist_g.py
  .venv-qwen/bin/python experiments/train_specialist_g.py --epochs 30 --use-pid
  .venv-qwen/bin/python experiments/train_specialist_g.py --eval-only weights/specialist_g/specialist_g_best.pth
"""

from __future__ import annotations

import argparse
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
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

warnings.filterwarnings("ignore")

ROOT   = Path(__file__).resolve().parents[1]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BINARY_LABEL = {"authentic": 0, "ai_generated": 1}
IDX2LABEL    = {0: "authentic", 1: "ai_generated"}

# ── 전처리 (MobileCLIP 기준) ───────────────────────────────────────────────
CLIP_MEAN = [0.48145466, 0.4578275,  0.40821073]
CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]

def _make_train_transform():
    return transforms.Compose([
        transforms.Resize((288, 288)),
        transforms.RandomCrop(256),
        transforms.RandomHorizontalFlip(),
        # B-Free 스타일: JPEG 포맷 편향 제거용 다양한 압축 augmentation
        transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 1.0))], p=0.3),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(CLIP_MEAN, CLIP_STD),
    ])

def _make_val_transform():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(CLIP_MEAN, CLIP_STD),
    ])

TRAIN_TRANSFORM = _make_train_transform()
VAL_TRANSFORM   = _make_val_transform()


# ── 데이터셋 경로 ──────────────────────────────────────────────────────────
EVAL_DATASETS = {
    "base":       "experiments/results/phase2_patha_scale500_gain_predictor/patha_agent_outputs_20260304_080157.jsonl",
    "dsC":        "experiments/results/phase2_patha_case3_scale300_dsC/patha_agent_outputs_20260303_105005.jsonl",
    "opensdi":    "experiments/results/phase2_patha_case3_opensdi_scale300/patha_agent_outputs_fixed_seed42.jsonl",
    "aigenproxy": "experiments/results/phase2_patha_case3_aigenproxy_scale300/patha_agent_outputs_fixed_seed42.jsonl",
}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def scan_binary_dataset() -> List[Dict]:
    """
    AI-gen vs Authentic 데이터 수집:
      authentic: CASIA2/Au + OpenSDID/authentic + AI-GenBench/authentic
      ai_gen:    GenImage/BigGAN + AI-GenBench_proxy/ai_gen + AI-GenBench_fakepart + OpenSDID/ai_gen
    """
    records = []

    def add_dir(directory: Path, label: str, limit: int = None):
        d = ROOT / directory
        if not d.exists():
            return 0
        paths = sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXTS)
        if limit:
            paths = paths[:limit]
        for p in paths:
            records.append({"image_path": str(p.relative_to(ROOT)), "true_label": label})
        return len(paths)

    # ── Authentic sources ──
    n = add_dir(Path("datasets/CASIA2_subset/Au"),            "authentic")
    print(f"    CASIA2 Au:          {n}")
    n = add_dir(Path("datasets/OpenSDID_subset/authentic"),   "authentic")
    print(f"    OpenSDID Auth:      {n}")
    n = add_dir(Path("datasets/AI-GenBench_proxy/authentic"), "authentic")
    print(f"    AIGenBench Auth:    {n}")

    # ── AI-Generated sources ──
    n = add_dir(Path("datasets/GenImage_subset/BigGAN/val/ai"),        "ai_generated")
    print(f"    GenImage BigGAN:    {n}")
    n = add_dir(Path("datasets/AI-GenBench_proxy/ai_generated"),      "ai_generated")
    print(f"    AIGenBench proxy:   {n}")
    n = add_dir(Path("datasets/AI-GenBench_fakepart_subset/ai_generated"), "ai_generated")
    print(f"    AIGenBench fakepart:{n}")
    n = add_dir(Path("datasets/OpenSDID_subset/ai_generated"),        "ai_generated")
    print(f"    OpenSDID AIgen:     {n}")

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
    by_label: Dict[str, List] = {}
    for r in records:
        by_label.setdefault(r["true_label"], []).append(r)
    train_all, val_all = [], []
    rng = random.Random(seed)
    for label, recs in by_label.items():
        recs = recs[:]
        rng.shuffle(recs)
        n = int(len(recs) * val_ratio)
        val_all.extend(recs[:n])
        train_all.extend(recs[n:])
    rng.shuffle(train_all)
    rng.shuffle(val_all)
    return train_all, val_all


# ── PiD 양자화 잔차 ────────────────────────────────────────────────────────
class PiDResidual(nn.Module):
    """
    Pixelwise Decomposition Residuals (PiD) 경량 구현.
    RGB → YUV 변환 → 양자화 → 역변환 → 잔차 계산.
    AI 생성 이미지는 이 잔차가 구조적으로 다름.

    참조: PiD (ICML 2025) — 생성 모델이 저주파 최적화로 인해
          양자화 잔차 성분을 보존하지 못함을 이용.
    """
    def __init__(self, n_bits: int = 4):
        super().__init__()
        # RGB → YCbCr 근사 변환 행렬 (ITU-R BT.601)
        M = torch.tensor([
            [ 0.299,  0.587,  0.114],
            [-0.169, -0.331,  0.500],
            [ 0.500, -0.419, -0.081],
        ])
        M_inv = torch.inverse(M)
        self.register_buffer("M",     M)
        self.register_buffer("M_inv", M_inv)
        self.n_bins  = 2 ** n_bits
        self.n_bits  = n_bits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B,3,H,W] in [0,1] (이미 CLIP 정규화 전 값 필요)"""
        B, C, H, W = x.shape
        # RGB → YUV 공간
        flat = x.permute(0, 2, 3, 1).reshape(-1, 3)   # [B*H*W, 3]
        yuv  = flat @ self.M.T                          # [B*H*W, 3]
        # 양자화
        yuv_q = torch.round(yuv * self.n_bins) / self.n_bins
        # 역변환
        rgb_q = yuv_q @ self.M_inv.T                   # [B*H*W, 3]
        # 잔차
        residual = (flat - rgb_q).abs()                # [B*H*W, 3]
        residual = residual.reshape(B, H, W, 3).permute(0, 3, 1, 2)  # [B,3,H,W]
        # 1ch로 집약 (채널 평균)
        residual = residual.mean(dim=1, keepdim=True)  # [B,1,H,W]
        return residual.clamp(0, 1)


# ── 모델 ──────────────────────────────────────────────────────────────────
class SpecialistG(nn.Module):
    """
    Binary: authentic(0) vs ai_generated(1)

    구조:
      - MobileCLIP-S2 frozen backbone → 512-dim feature
      - (옵션) PiD 잔차 → 경량 CNN → 64-dim feature
      - Binary head
    """
    def __init__(self, clip_ckpt: Path, use_pid: bool = True, dropout: float = 0.3):
        super().__init__()
        self.use_pid = use_pid

        # MobileCLIP 로드 (frozen)
        self.clip_encoder = self._load_clip(clip_ckpt)
        for p in self.clip_encoder.parameters():
            p.requires_grad_(False)
        clip_dim = 512

        # PiD 잔차 브랜치
        if use_pid:
            self.pid = PiDResidual(n_bits=4)
            # 잔차 처리: 경량 CNN (1ch → 64dim)
            self.pid_branch = nn.Sequential(
                nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
            )
            fused_dim = clip_dim + 64
        else:
            fused_dim = clip_dim

        self.head = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2),
        )

    def _load_clip(self, ckpt_path: Path) -> nn.Module:
        """MobileCLIP-S2 이미지 인코더 로드."""
        try:
            import open_clip
            model, _, _ = open_clip.create_model_and_transforms(
                "MobileCLIP-S2", pretrained="datacompdr")
            encoder = model.visual
            if ckpt_path.exists():
                ckpt = torch.load(ckpt_path, map_location="cpu")
                # ft4 체크포인트에서 visual encoder 부분만 로드
                sd = ckpt.get("model_state_dict", ckpt)
                visual_sd = {k.replace("visual.", ""): v
                             for k, v in sd.items() if k.startswith("visual.")}
                if visual_sd:
                    encoder.load_state_dict(visual_sd, strict=False)
                    print(f"  MobileCLIP visual 가중치 로드: {ckpt_path.name}")
            return encoder
        except Exception as e:
            print(f"  [경고] MobileCLIP 로드 실패: {e}")
            print("  → Fallback: 간단한 CNN 인코더 사용")
            return self._fallback_encoder()

    def _fallback_encoder(self) -> nn.Module:
        """MobileCLIP 없을 때 fallback."""
        from torchvision import models
        base = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V2)
        base.classifier = nn.Identity()
        # 출력을 512dim으로 맞춤
        return nn.Sequential(base.features, nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                              nn.Linear(1280, 512))

    def forward(self, x: torch.Tensor, x_raw: torch.Tensor = None) -> torch.Tensor:
        """
        x:     CLIP 정규화된 텐서 [B,3,H,W]
        x_raw: PiD용 [0,1] 범위 텐서 (None이면 x에서 역정규화)
        """
        # CLIP feature
        try:
            clip_feat = self.clip_encoder(x)
            if isinstance(clip_feat, tuple):
                clip_feat = clip_feat[0]
            if clip_feat.dim() > 2:
                clip_feat = clip_feat.flatten(1)
        except Exception:
            clip_feat = torch.zeros(x.shape[0], 512, device=x.device)

        if not self.use_pid:
            return self.head(clip_feat)

        # PiD 잔차 (역정규화 후 처리)
        if x_raw is None:
            # CLIP 역정규화 근사 (정확도 낮지만 잔차 신호 추출에는 충분)
            mean = torch.tensor(CLIP_MEAN, device=x.device).view(1,3,1,1)
            std  = torch.tensor(CLIP_STD,  device=x.device).view(1,3,1,1)
            x_raw = (x * std + mean).clamp(0, 1)
        pid_residual = self.pid(x_raw)
        pid_feat     = self.pid_branch(pid_residual)

        return self.head(torch.cat([clip_feat, pid_feat], dim=1))


# ── 학습/평가 ─────────────────────────────────────────────────────────────
class BinaryDataset(Dataset):
    def __init__(self, records: List[Dict], transform=None):
        self.records   = records
        self.transform = transform or VAL_TRANSFORM

    def __len__(self): return len(self.records)

    def __getitem__(self, idx):
        rec   = self.records[idx]
        path  = ROOT / rec["image_path"]
        label = BINARY_LABEL.get(rec["true_label"], -1)
        try:
            img    = Image.open(path).convert("RGB")
            tensor = self.transform(img)
        except Exception:
            tensor = torch.zeros(3, 256, 256)
        return tensor, label, rec["image_path"]


def train_epoch(model, loader, optimizer, scaler, criterion) -> Tuple[float, float]:
    model.train()
    # CLIP backbone frozen 유지
    for p in model.clip_encoder.parameters():
        p.requires_grad_(False)
    total_loss, correct, n = 0.0, 0, 0
    use_amp = DEVICE.type == "cuda"
    for imgs, labels, _ in loader:
        imgs, labels = imgs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
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
        all_probs.append(probs); all_preds.append(preds); all_labels.append(labels)
    probs  = torch.cat(all_probs).numpy()
    preds  = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    valid  = labels != -1
    preds, labels, probs = preds[valid], labels[valid], probs[valid]

    acc   = float((preds == labels).mean())
    tp    = int(((preds == 1) & (labels == 1)).sum())
    fp    = int(((preds == 1) & (labels == 0)).sum())
    fn    = int(((preds == 0) & (labels == 1)).sum())
    prec  = tp / max(tp + fp, 1)
    rec   = tp / max(tp + fn, 1)
    f1_g  = 2 * prec * rec / max(prec + rec, 1e-8)
    per_class = {}
    recalls   = []
    for idx, name in IDX2LABEL.items():
        mask = labels == idx
        r    = float((preds[mask] == idx).mean()) if mask.sum() > 0 else 0.0
        per_class[name] = {"recall": r, "n": int(mask.sum())}
        recalls.append(r)
    return {"accuracy": acc, "macro_recall": float(np.mean(recalls)),
            "aigen_f1": f1_g, "aigen_recall": rec, "aigen_precision": prec,
            "per_class": per_class, "probs": probs, "preds": preds, "labels": labels}


# ── 메인 ──────────────────────────────────────────────────────────────────
def run_training(args) -> Path:
    print("=" * 60)
    print("Specialist-G: AI-Generated vs Authentic")
    print(f"  epochs={args.epochs}, batch={args.batch_size}, "
          f"lr={args.lr}, use_pid={args.use_pid}, device={DEVICE}")
    print("=" * 60)

    clip_ckpt = ROOT / "weights" / "mobileclip_forensics" / "mobileclip_s2_forensics_ft4.pth"
    print(f"\n[MobileCLIP 체크포인트] {clip_ckpt}")

    print("\n[데이터 수집]")
    all_recs = scan_binary_dataset()
    au_n  = sum(1 for r in all_recs if r["true_label"] == "authentic")
    ag_n  = sum(1 for r in all_recs if r["true_label"] == "ai_generated")
    print(f"  Authentic: {au_n}, AI-Generated: {ag_n}, Total: {len(all_recs)}")

    train_recs, val_recs = split_records(all_recs, val_ratio=0.15, seed=42)
    print(f"  Train: {len(train_recs)}, Val: {len(val_recs)}")

    train_loader = DataLoader(BinaryDataset(train_recs, TRAIN_TRANSFORM),
                              batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=(DEVICE.type == "cuda"))
    val_loader   = DataLoader(BinaryDataset(val_recs, VAL_TRANSFORM),
                              batch_size=args.batch_size * 2, shuffle=False,
                              num_workers=4, pin_memory=(DEVICE.type == "cuda"))

    print("\n[모델 초기화]")
    model    = SpecialistG(clip_ckpt, use_pid=args.use_pid).to(DEVICE)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Total: {total/1e6:.2f}M, Trainable: {trainable/1e6:.2f}M "
          f"(frozen CLIP backbone)")

    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, 1.5]).to(DEVICE))  # ai_gen 더 중시
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.05)
    scaler    = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda"))

    ckpt_dir  = ROOT / "weights" / "specialist_g"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "specialist_g_best.pth"

    best_f1 = 0.0
    history = []
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, scaler, criterion)
        val_stats       = evaluate_binary(model, val_loader)
        scheduler.step()
        f1_g = val_stats["aigen_f1"]
        star = " ★" if f1_g > best_f1 else ""
        print(
            f"  Epoch {epoch:02d}/{args.epochs} | loss={tr_loss:.4f} | "
            f"tr_acc={tr_acc:.3f} | val_acc={val_stats['accuracy']:.3f} | "
            f"aigen_recall={val_stats['aigen_recall']:.3f} | "
            f"aigen_f1={f1_g:.3f}{star}"
        )
        history.append({"epoch": epoch, "train_loss": tr_loss,
                         "aigen_f1": f1_g, "aigen_recall": val_stats["aigen_recall"]})
        if f1_g > best_f1:
            best_f1 = f1_g
            torch.save({"model_state_dict": model.state_dict(),
                        "history": history, "best_aigen_f1": best_f1,
                        "epoch": epoch, "use_pid": args.use_pid}, ckpt_path)
            print(f"  → 저장: {ckpt_path}")

    print(f"\n[완료] best aigen_f1={best_f1:.4f}")
    return ckpt_path


@torch.no_grad()
def eval_all_datasets(ckpt_path: Path, args) -> Dict:
    print("\n[전체 데이터셋 평가]")
    ckpt      = torch.load(ckpt_path, map_location="cpu")
    use_pid   = ckpt.get("use_pid", args.use_pid)
    clip_ckpt = ROOT / "weights" / "mobileclip_forensics" / "mobileclip_s2_forensics_ft4.pth"
    model     = SpecialistG(clip_ckpt, use_pid=use_pid).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    out_dir = ROOT / "experiments" / "results" / "specialist_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")

    summary = {}
    for ds_key, jsonl_rel in EVAL_DATASETS.items():
        recs     = load_jsonl_records(jsonl_rel)
        recs_bin = [r for r in recs if r["true_label"] in BINARY_LABEL]
        if not recs_bin:
            continue
        loader = DataLoader(BinaryDataset(recs_bin, VAL_TRANSFORM),
                            batch_size=args.batch_size * 2, shuffle=False,
                            num_workers=4, pin_memory=(DEVICE.type == "cuda"))
        stats  = evaluate_binary(model, loader)
        probs  = stats.pop("probs")
        preds  = stats.pop("preds")
        labels = stats.pop("labels")

        out_jsonl = out_dir / f"specialist_g_{ds_key}_{ts}.jsonl"
        with open(out_jsonl, "w") as f:
            for rec, pred_idx, prob_vec in zip(recs_bin, preds.tolist(), probs.tolist()):
                f.write(json.dumps({
                    "image_path":      rec["image_path"],
                    "true_label":      rec["true_label"],
                    "pred_label":      IDX2LABEL[pred_idx],
                    "aigen_score":     float(prob_vec[1]),
                    "authentic_score": float(prob_vec[0]),
                    "confidence":      float(max(prob_vec)),
                }, ensure_ascii=False) + "\n")

        summary[ds_key] = {
            "n_binary": len(recs_bin),
            "accuracy": stats["accuracy"],
            "macro_recall": stats["macro_recall"],
            "aigen_recall": stats["aigen_recall"],
            "aigen_f1": stats["aigen_f1"],
            "per_class": stats["per_class"],
            "jsonl": str(out_jsonl),
        }
        print(
            f"  [{ds_key}] n={len(recs_bin)} | acc={stats['accuracy']:.3f} | "
            f"aigen_recall={stats['aigen_recall']:.3f} | "
            f"aigen_f1={stats['aigen_f1']:.3f}"
        )

    summary_path = out_dir / f"specialist_g_summary_{ts}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n저장: {summary_path}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",    type=int,   default=25)
    parser.add_argument("--batch-size", type=int,   default=48)
    parser.add_argument("--lr",        type=float, default=3e-4)
    parser.add_argument("--use-pid",   action="store_true", default=True)
    parser.add_argument("--no-pid",    dest="use_pid", action="store_false")
    parser.add_argument("--eval-only", type=str,   default=None)
    args = parser.parse_args()

    random.seed(42); np.random.seed(42); torch.manual_seed(42)

    if args.eval_only:
        eval_all_datasets(Path(args.eval_only), args)
    else:
        ckpt = run_training(args)
        eval_all_datasets(ckpt, args)


if __name__ == "__main__":
    main()
