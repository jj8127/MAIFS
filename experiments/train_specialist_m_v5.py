#!/usr/bin/env python3
"""
Specialist-M v5: PiD + DCT 2-branch (RGB/SRM 완전 제거)
=========================================================

Phase 4.6 Embedding CKA 분석 결과 기반 재설계:
  - 중복 주범 제거: RGB branch (mnv2_rgb↔specm_rgb CKA=0.656) → 제거
  - 중복 제거: SRM branch (mnv2_noise↔specm_srm CKA=0.564) → 제거
  - 독립 채택 ①: DCT residual branch (CKA=0.239) → 유지
  - 독립 채택 ②: PiD branch (CKA=0.001) → 확장 (64d→1280d, MobileNetV2 기반)

아키텍처:
  PiD branch  : PiDResidual(n_bits=4) → MobileNetBranch(in_ch=1, pretrained=False) → 1280d
  DCT branch  : DCTResidualExtractor   → MobileNetBranch(in_ch=1, pretrained=False) → 1280d
  fused       : cat([pid, dct]) = 2560d
  head        : LayerNorm → Linear(2560→256) → GELU → Dropout → Linear(256→2)

v4 대비 변경:
  - RGB branch 제거 (-1280d)
  - SRM branch 제거 (-1280d, SRM 필터 계열 중복)
  - PiD branch 추가 (+1280d, YUV 양자화 잔차 — MNV2와 거의 독립)
  - fused dim: 3840d → 2560d (33% 감소)
  - 예상 CKA vs MNV2: 0.392 → ~0.1

데이터 (v4와 동일):
  Authentic:   CASIA2 Au (~7491) + GenImage_nature (최대 3000장)
  Manipulated: CASIA2 Tp (~5123) + IMD2020 non-eval (~1710장)

실행:
  .venv-qwen/bin/python experiments/train_specialist_m_v5.py
  .venv-qwen/bin/python experiments/train_specialist_m_v5.py --epochs 30 --lr 1e-4
  .venv-qwen/bin/python experiments/train_specialist_m_v5.py --eval-only weights/specialist_m_v5/specialist_m_v5_best.pth
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
LABEL_MAP_3   = {"authentic": 0, "manipulated": 1, "ai_generated": 2}
IMG_EXTS      = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ── 경로 ─────────────────────────────────────────────────────────────────
CASIA2_AU   = ROOT / "datasets" / "CASIA2_subset" / "Au"
CASIA2_TP   = ROOT / "datasets" / "CASIA2_subset" / "Tp"
IMD2020_DIR = ROOT / "datasets" / "IMD2020_subset" / \
              "IMD2020_Generative_Image_Inpainting_yu2018_01" / "images"
GENIMAGE_NATURE   = ROOT / "datasets" / "GenImage_subset" / "BigGAN" / "val" / "nature"
GENIMAGE_AUTH_MAX = 3000

EVAL_DATASETS = {
    "base":       "experiments/results/phase2_patha_scale500_gain_predictor/patha_agent_outputs_20260304_080157.jsonl",
    "dsC":        "experiments/results/phase2_patha_case3_scale300_dsC/patha_agent_outputs_20260303_105005.jsonl",
    "opensdi":    "experiments/results/phase2_patha_case3_opensdi_scale300/patha_agent_outputs_fixed_seed42.jsonl",
    "aigenproxy": "experiments/results/phase2_patha_case3_aigenproxy_scale300/patha_agent_outputs_fixed_seed42.jsonl",
}


# ── Augmentation ──────────────────────────────────────────────────────────
class RandomJPEGCompression:
    def __init__(self, quality_range=(40, 95), p=0.5):
        self.quality_range = quality_range
        self.p = p

    def __call__(self, img):
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

    def __call__(self, tensor):
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


# ── 모델 컴포넌트 ─────────────────────────────────────────────────────────
def normalize_batch(x: torch.Tensor, mean, std) -> torch.Tensor:
    t = torch.tensor(mean if len(mean) == x.shape[1] else mean[:1],
                     device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
    s = torch.tensor(std  if len(std)  == x.shape[1] else std[:1],
                     device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
    return (x - t) / s


class DCTResidualExtractor(nn.Module):
    """저주파 제거 DCT 잔차: 고주파(국소 패턴) 추출"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gray     = x.mean(dim=1, keepdim=True)
        low_pass = F.avg_pool2d(gray, kernel_size=8, stride=8)
        low_pass = F.interpolate(low_pass, size=gray.shape[-2:], mode='nearest')
        residual = (gray - low_pass).clamp(-1.0, 1.0)
        return (residual + 1.0) / 2.0  # [0,1] 범위


class PiDResidual(nn.Module):
    """
    Pixelwise Decomposition Residuals.
    RGB → YUV → 양자화 → 역변환 → 잔차.
    AI 생성 이미지는 잔차 패턴이 자연 이미지와 다름.
    MNV2와의 CKA = 0.001 (완전 독립적 표현).
    """
    def __init__(self, n_bits: int = 4):
        super().__init__()
        M = torch.tensor([
            [ 0.299,  0.587,  0.114],
            [-0.169, -0.331,  0.500],
            [ 0.500, -0.419, -0.081],
        ])
        self.register_buffer("M",     M)
        self.register_buffer("M_inv", torch.inverse(M))
        self.n_bins = 2 ** n_bits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B,3,H,W] in [0,1]"""
        B, C, H, W = x.shape
        flat  = x.permute(0, 2, 3, 1).reshape(-1, 3)
        yuv   = flat @ self.M.T
        yuv_q = torch.round(yuv * self.n_bins) / self.n_bins
        rgb_q = yuv_q @ self.M_inv.T
        res   = (flat - rgb_q).abs().reshape(B, H, W, 3).permute(0, 3, 1, 2)
        return res.mean(dim=1, keepdim=True).clamp(0, 1)  # [B,1,H,W]


class MobileNetBranch(nn.Module):
    """MobileNetV2 기반 단일 브랜치 (1280-dim 출력)"""
    def __init__(self, in_channels: int = 1, pretrained: bool = False):
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


class SpecialistMv5(nn.Module):
    """
    SpecM-v5: PiD + DCT 2-branch
    ==============================
    MNV2와의 embedding CKA 목표: ~0.1 (v4 0.392 대비 대폭 감소)

    브랜치 독립성 (Phase 4.6 측정 기준):
      DCT  : CKA vs MNV2 = 0.239
      PiD  : CKA vs MNV2 = 0.001 (거의 완전 독립)
    """
    def __init__(self, dropout: float = 0.3):
        super().__init__()
        self.pid_extractor = PiDResidual(n_bits=4)
        self.dct_extractor = DCTResidualExtractor()

        # PiD branch: 1ch YUV 잔차 → MobileNetV2 (pretrained=False, no RGB prior)
        self.pid_branch = MobileNetBranch(in_channels=1, pretrained=False)
        # DCT branch: 1ch 고주파 잔차 → MobileNetV2 (pretrained=False)
        self.dct_branch = MobileNetBranch(in_channels=1, pretrained=False)

        fused = self.pid_branch.out_dim + self.dct_branch.out_dim  # 2560
        self.head = nn.Sequential(
            nn.LayerNorm(fused),
            nn.Linear(fused, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # PiD: [0,1] 범위 입력 필요 (정규화 전)
        pid  = self.pid_extractor(x)                              # [B,1,H,W]
        pid  = normalize_batch(pid, [0.5], [0.5])

        # DCT residual
        dct  = self.dct_extractor(x)                              # [B,1,H,W]
        dct  = normalize_batch(dct, [0.5], [0.5])

        return self.head(torch.cat([
            self.pid_branch(pid),
            self.dct_branch(dct),
        ], dim=1))

    def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """GAP 후 fused embedding (2560-dim), head 전"""
        pid = normalize_batch(self.pid_extractor(x), [0.5], [0.5])
        dct = normalize_batch(self.dct_extractor(x), [0.5], [0.5])
        return torch.cat([self.pid_branch(pid), self.dct_branch(dct)], dim=1)


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float = 0.6):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce   = F.cross_entropy(logits, targets, reduction="none")
        pt   = torch.exp(-ce)
        w    = torch.where(targets == 1,
                           torch.full_like(ce, self.alpha),
                           torch.full_like(ce, 1 - self.alpha))
        return (w * (1 - pt) ** self.gamma * ce).mean()


# ── 데이터셋 ─────────────────────────────────────────────────────────────
def collect_training_data() -> Tuple[List[Tuple[Path, int]], List[Tuple[Path, int]]]:
    auth_paths, manip_paths = [], []

    # CASIA2 Au
    if CASIA2_AU.exists():
        for p in CASIA2_AU.iterdir():
            if p.suffix.lower() in IMG_EXTS:
                auth_paths.append((p, 0))
    print(f"  CASIA2 Au: {len(auth_paths)}")

    # GenImage_nature (authentic 보강)
    gen_auth = []
    if GENIMAGE_NATURE.exists():
        for p in GENIMAGE_NATURE.iterdir():
            if p.suffix.lower() in IMG_EXTS:
                gen_auth.append((p, 0))
        random.shuffle(gen_auth)
        gen_auth = gen_auth[:GENIMAGE_AUTH_MAX]
    auth_paths.extend(gen_auth)
    print(f"  GenImage_nature: {len(gen_auth)} (+authentic)")

    # CASIA2 Tp
    if CASIA2_TP.exists():
        for p in CASIA2_TP.iterdir():
            if p.suffix.lower() in IMG_EXTS:
                manip_paths.append((p, 1))
    print(f"  CASIA2 Tp: {len(manip_paths)}")

    # IMD2020 (manipulated 보강)
    imd_manip = []
    if IMD2020_DIR.exists():
        all_imd = [p for p in IMD2020_DIR.iterdir() if p.suffix.lower() in IMG_EXTS]
        random.shuffle(all_imd)
        imd_manip = [(p, 1) for p in all_imd[:1710]]
    manip_paths.extend(imd_manip)
    print(f"  IMD2020: {len(imd_manip)} (+manipulated)")

    print(f"  총: auth={len(auth_paths)}, manip={len(manip_paths)}")
    return auth_paths, manip_paths


class TrainDataset(Dataset):
    def __init__(self, samples: List[Tuple[Path, int]], transform=None):
        self.samples   = samples
        self.transform = transform or TRAIN_TRANSFORM

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img    = Image.open(path).convert("RGB")
            tensor = self.transform(img)
        except Exception:
            tensor = torch.zeros(3, 224, 224)
        return tensor, label


class EvalDataset(Dataset):
    def __init__(self, records: List[Dict], transform=None):
        self.records   = records
        self.transform = transform or VAL_TRANSFORM

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        path = ROOT / rec["image_path"]
        label = LABEL_MAP_3[rec["true_label"]]
        try:
            img    = Image.open(path).convert("RGB")
            tensor = self.transform(img)
        except Exception:
            tensor = torch.zeros(3, 224, 224)
        return tensor, label, rec["image_path"]


def load_eval_records(jsonl_path: str) -> List[Dict]:
    records = []
    full = ROOT / jsonl_path
    if not full.exists():
        return records
    with open(full) as f:
        for line in f:
            rec = json.loads(line.strip())
            if rec.get("true_label") not in LABEL_MAP_3:
                continue
            if (ROOT / rec["image_path"]).exists():
                records.append(rec)
    return records


# ── 학습 유틸 ─────────────────────────────────────────────────────────────
def compute_metrics(labels: np.ndarray, preds: np.ndarray) -> Dict:
    acc = float((labels == preds).mean())
    per = {}
    recalls = []
    for idx, name in IDX2LABEL.items():
        mask = labels == idx
        if mask.sum() == 0:
            per[name] = {"recall": 0.0, "n": 0}
            recalls.append(0.0)
        else:
            r = float((preds[mask] == idx).mean())
            per[name] = {"recall": round(r, 4), "n": int(mask.sum())}
            recalls.append(r)
    tp  = ((preds == 1) & (labels == 1)).sum()
    fp  = ((preds == 1) & (labels == 0)).sum()
    fn  = ((preds == 0) & (labels == 1)).sum()
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {
        "acc": round(acc, 4),
        "manip_f1": round(f1, 4),
        "manip_prec": round(prec, 4),
        "manip_rec": round(rec, 4),
        "macro_recall": round(float(np.mean(recalls)), 4),
        "per_class": per,
    }


def train_epoch(model, loader, optimizer, scaler, criterion) -> Tuple[float, float]:
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    use_amp = DEVICE.type == "cuda"
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=use_amp):
            out  = model(imgs)
            loss = criterion(out, labels)
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total_loss += loss.item() * len(labels)
        correct    += (out.argmax(1) == labels).sum().item()
        n          += len(labels)
    return total_loss / n, correct / n


@torch.no_grad()
def eval_epoch(model, loader) -> Tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss, all_labels, all_preds = 0.0, [], []
    criterion = FocalLoss()
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        out  = model(imgs)
        loss = criterion(out, labels)
        total_loss += loss.item() * len(labels)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(out.argmax(1).cpu().numpy())
    return (total_loss / len(all_labels),
            np.array(all_labels),
            np.array(all_preds))


# ── 4-DS 평가 (eval-only 또는 학습 중 전체 평가) ────────────────────────
@torch.no_grad()
def eval_4ds(model, ds_name: str, jsonl_path: str) -> Dict:
    records = load_eval_records(jsonl_path)
    if not records:
        return {}
    # manipulated/authentic만 추출 (binary 모델)
    binary_recs = [r for r in records if r["true_label"] in ("authentic", "manipulated")]
    if not binary_recs:
        return {}
    ds = EvalDataset(binary_recs, VAL_TRANSFORM)
    loader = DataLoader(ds, batch_size=64, shuffle=False,
                        num_workers=2, pin_memory=True)
    model.eval()
    all_labels, all_preds, all_scores = [], [], []
    for imgs, labels, _ in loader:
        imgs = imgs.to(DEVICE)
        out  = model(imgs)
        prob = torch.softmax(out, dim=1)
        all_labels.extend(labels.numpy())
        all_preds.extend(out.argmax(1).cpu().numpy())
        all_scores.extend(prob[:, 1].cpu().numpy())
    labels_arr = np.array(all_labels)
    preds_arr  = np.array(all_preds)
    scores_arr = np.array(all_scores)
    m = compute_metrics(labels_arr, preds_arr)
    m["n"] = len(records)
    m["n_binary"] = len(binary_recs)
    return m


# ── 메인 ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int,   default=64)
    parser.add_argument("--dropout",    type=float, default=0.3)
    parser.add_argument("--eval-only",  type=str,   default="",
                        help="체크포인트 경로 지정 시 평가만 수행")
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f"Device: {DEVICE}")
    print(f"SpecM-v5: PiD(1280d) + DCT(1280d) → fused(2560d)")
    print(f"v4 대비: RGB/SRM 제거, PiD 추가. 예상 CKA vs MNV2: ~0.1")

    # ── 모델 ──
    model = SpecialistMv5(dropout=args.dropout).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {n_params:.2f}M")

    # ── 저장 경로 ──
    save_dir = ROOT / "weights" / "specialist_m_v5"
    save_dir.mkdir(parents=True, exist_ok=True)
    result_dir = ROOT / "experiments" / "results" / "specialist_eval"
    result_dir.mkdir(parents=True, exist_ok=True)

    # ── eval-only 모드 ──
    if args.eval_only:
        ckpt_path = Path(args.eval_only)
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt))
        print(f"\n체크포인트 로드: {ckpt_path}")
        for ds_name, ds_path in EVAL_DATASETS.items():
            m = eval_4ds(model, ds_name, ds_path)
            if m:
                print(f"  {ds_name}: manip_f1={m['manip_f1']:.4f}, "
                      f"manip_rec={m['manip_rec']:.4f}, "
                      f"auth_rec={m['per_class'].get('authentic',{}).get('recall',0):.4f}")
        return

    # ── 학습 데이터 수집 ──
    print("\n[데이터 수집]")
    auth_paths, manip_paths = collect_training_data()

    # 학습/검증 분리 (9:1)
    random.shuffle(auth_paths)
    random.shuffle(manip_paths)
    val_n_auth  = max(1, len(auth_paths)  // 10)
    val_n_manip = max(1, len(manip_paths) // 10)

    val_samples   = auth_paths[:val_n_auth] + manip_paths[:val_n_manip]
    train_samples = auth_paths[val_n_auth:] + manip_paths[val_n_manip:]
    random.shuffle(train_samples)

    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}")

    # WeightedRandomSampler (클래스 불균형 처리)
    train_labels   = [s[1] for s in train_samples]
    class_counts   = [train_labels.count(c) for c in range(2)]
    class_weights  = [1.0 / max(c, 1) for c in class_counts]
    sample_weights = [class_weights[l] for l in train_labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(train_samples), replacement=True)

    train_loader = DataLoader(
        TrainDataset(train_samples, TRAIN_TRANSFORM),
        batch_size=args.batch_size, sampler=sampler,
        num_workers=4, pin_memory=True)
    val_loader = DataLoader(
        TrainDataset(val_samples, VAL_TRANSFORM),
        batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

    # ── 학습 준비 ──
    criterion = FocalLoss(gamma=2.0, alpha=0.6)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler    = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda"))

    best_manip_f1 = 0.0
    history = []

    print(f"\n[학습 시작] epochs={args.epochs}, lr={args.lr}, batch={args.batch_size}")
    print(f"{'Ep':>4} {'TrainLoss':>10} {'ValLoss':>9} {'ManipF1':>8} {'AuthRec':>8} {'ManipRec':>9} {'Best':>5}")
    print("-" * 60)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scaler, criterion)
        val_loss, val_labels, val_preds = eval_epoch(model, val_loader)
        m = compute_metrics(val_labels, val_preds)
        scheduler.step()

        manip_f1  = m["manip_f1"]
        auth_rec  = m["per_class"].get("authentic",  {}).get("recall", 0)
        manip_rec = m["per_class"].get("manipulated", {}).get("recall", 0)

        is_best = manip_f1 > best_manip_f1
        if is_best:
            best_manip_f1 = manip_f1
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "best_manip_f1": best_manip_f1,
                "history": history,
                "args": vars(args),
                "arch": "SpecialistMv5_PiD+DCT_2branch",
            }, save_dir / "specialist_m_v5_best.pth")

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            **m,
        })

        print(f"{epoch:>4} {train_loss:>10.4f} {val_loss:>9.4f} "
              f"{manip_f1:>8.4f} {auth_rec:>8.4f} {manip_rec:>9.4f} "
              f"{'★' if is_best else ' ':>5}")

    # ── 4-DS 전체 평가 ──
    print(f"\n[4-DS 평가] best manip_f1={best_manip_f1:.4f}")
    ckpt = torch.load(save_dir / "specialist_m_v5_best.pth",
                      map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ds_results = {}
    f1_list = []
    for ds_name, ds_path in EVAL_DATASETS.items():
        m = eval_4ds(model, ds_name, ds_path)
        if m:
            ds_results[ds_name] = m
            f1_list.append(m["manip_f1"])
            print(f"  {ds_name}: manip_f1={m['manip_f1']:.4f}, "
                  f"auth_rec={m['per_class'].get('authentic',{}).get('recall',0):.4f}, "
                  f"manip_rec={m['per_class'].get('manipulated',{}).get('recall',0):.4f}")
    avg_f1 = np.mean(f1_list) if f1_list else 0.0
    print(f"  4-DS avg manip_f1={avg_f1:.4f}")

    # 결과 저장
    out = {
        "arch": "SpecialistMv5_PiD+DCT_2branch",
        "timestamp": ts,
        "device": str(DEVICE),
        "args": vars(args),
        "n_params_M": round(n_params, 2),
        "val_best_manip_f1": best_manip_f1,
        "4ds_avg_manip_f1": round(avg_f1, 4),
        "datasets": ds_results,
        "history": history,
        "design_note": {
            "removed": ["RGB branch (CKA=0.322, mnv2_rgb↔specm_rgb=0.656)",
                        "SRM branch (CKA=0.563, mnv2_noise↔specm_srm=0.564)"],
            "added":   ["PiD branch (CKA=0.001 vs MNV2, YUV quantization residual)"],
            "kept":    ["DCT branch (CKA=0.239 vs MNV2)"],
            "fused_dim": "2560 (v4 3840 대비 -33%)",
        },
    }
    out_path = result_dir / f"specialist_m_v5_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n결과 저장: {out_path}")
    print(f"체크포인트: {save_dir / 'specialist_m_v5_best.pth'}")


if __name__ == "__main__":
    main()
