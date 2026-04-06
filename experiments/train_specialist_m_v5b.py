#!/usr/bin/env python3
"""
Specialist-M v5b: MobileCLIP-S2 (frozen) + SRM lightweight CNN
================================================================

CKA 측정 결과 기반 아키텍처 결정:
  MobileCLIP-S2-ft4  CKA=0.028  ★★★ MNV2와 거의 완전 독립
  SpecM v1~v4        CKA=0.725  ✗   RGB branch 공유로 중복 높음

설계 원칙:
  - MobileCLIP-S2 frozen backbone (512d): 의미론적/전역 특징
    → MNV2(CNN+SRM)와 완전히 다른 representational basis
  - SRM lightweight CNN (128d): 노이즈 불일치 → 조작 흔적 포착
    → MobileNetV2 대신 소형 3-layer CNN 사용 (MNV2와 구조적 차별화)
  - fused: 512 + 128 = 640d → binary head

  MNV2 대비 예상 CKA: ~0.03~0.05 (MobileCLIP dominance)
  v4(0.725) 대비 대폭 감소 → Meyen (2021) 이론적 상한 개선 기대

파라미터:
  MobileCLIP-S2:  ~12M (frozen, gradient 없음)
  SRM CNN:        ~0.5M (학습)
  Head:           ~0.1M (학습)
  학습 파라미터:  ~0.6M (v4 7.66M 대비 -92%)

데이터 (v4와 동일):
  Authentic:   CASIA2 Au + GenImage_nature (최대 3000장)
  Manipulated: CASIA2 Tp + IMD2020 non-eval (~1710장)

실행:
  .venv-qwen/bin/python experiments/train_specialist_m_v5b.py
  .venv-qwen/bin/python experiments/train_specialist_m_v5b.py --epochs 40 --lr 3e-4
  .venv-qwen/bin/python experiments/train_specialist_m_v5b.py --eval-only weights/specialist_m_v5b/specialist_m_v5b_best.pth
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
from torchvision import transforms

warnings.filterwarnings("ignore")

ROOT   = Path(__file__).resolve().parents[1]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLIP_MEAN = [0.48145466, 0.4578275,  0.40821073]
CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]
LABEL_MAP     = {"authentic": 0, "manipulated": 1, "ai_generated": 2}
BINARY_LABEL  = {"authentic": 0, "manipulated": 1}
IMG_EXTS      = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

# ── 경로 ─────────────────────────────────────────────────────────────────
CASIA2_AU         = ROOT / "datasets" / "CASIA2_subset" / "Au"
CASIA2_TP         = ROOT / "datasets" / "CASIA2_subset" / "Tp"
IMD2020_DIR       = ROOT / "datasets" / "IMD2020_subset" / \
                    "IMD2020_Generative_Image_Inpainting_yu2018_01" / "images"
GENIMAGE_NATURE   = ROOT / "datasets" / "GenImage_subset" / "BigGAN" / "val" / "nature"
MOBILECLIP_CKPT   = ROOT / "weights" / "mobileclip_forensics" / "mobileclip_s2_forensics_ft4.pth"
GENIMAGE_AUTH_MAX = 3000

EVAL_DATASETS = {
    "base":       "experiments/results/phase2_patha_scale500_gain_predictor/patha_agent_outputs_20260304_080157.jsonl",
    "dsC":        "experiments/results/phase2_patha_case3_scale300_dsC/patha_agent_outputs_20260303_105005.jsonl",
    "opensdi":    "experiments/results/phase2_patha_case3_opensdi_scale300/patha_agent_outputs_fixed_seed42.jsonl",
    "aigenproxy": "experiments/results/phase2_patha_case3_aigenproxy_scale300/patha_agent_outputs_fixed_seed42.jsonl",
}


# ── 전처리 ────────────────────────────────────────────────────────────────
class RandomJPEGCompression:
    def __init__(self, quality_range=(40, 95), p=0.5):
        self.quality_range = quality_range
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            q = random.randint(*self.quality_range)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=q)
            buf.seek(0)
            img = Image.open(buf).convert("RGB")
        return img


class RandomGaussianNoise:
    def __init__(self, std_range=(0.002, 0.015), p=0.4):
        self.std_range = std_range
        self.p = p

    def __call__(self, t):
        if random.random() < self.p:
            std = random.uniform(*self.std_range)
            t = (t + torch.randn_like(t) * std).clamp(0, 1)
        return t


TRAIN_TRANSFORM = transforms.Compose([
    RandomJPEGCompression(quality_range=(40, 95), p=0.5),
    transforms.Resize((288, 288)),
    transforms.RandomCrop(256),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.1),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
    transforms.ToTensor(),
    RandomGaussianNoise(std_range=(0.002, 0.015), p=0.4),
    transforms.RandomErasing(p=0.35, scale=(0.02, 0.25), ratio=(0.3, 3.3), value='random'),
])

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
])


# ── 모델 ─────────────────────────────────────────────────────────────────
def _normalize(x: torch.Tensor, mean, std) -> torch.Tensor:
    m = torch.tensor(mean, device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
    s = torch.tensor(std,  device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
    return (x - m) / s


class SRMExtractor(nn.Module):
    """SRM 고정 필터 (3-kernel): 노이즈 불일치 패턴 추출"""
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
        return (res.clamp(-2.0, 2.0) + 2.0) / 4.0   # [B, 3, H, W] in [0,1]


class SRMLightCNN(nn.Module):
    """
    SRM 잔차용 경량 CNN (MobileNetV2 대신 사용).
    MNV2 noise branch와 구조적으로 다름 → 낮은 CKA 유지.
    3ch SRM 잔차 → 128-dim 특징
    """
    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            # stage 1: 3 → 32
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            # stage 2: 32 → 64 (depthwise-separable)
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, groups=32, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            # stage 3: 64 → 128 (depthwise-separable)
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, groups=64, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            # stage 4: 128 → out_dim
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, groups=128, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, out_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_dim), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).flatten(1)


def load_mobileclip_encoder(ckpt_path: Path) -> Optional[nn.Module]:
    """MobileCLIP-S2 visual encoder 로드 (ft4 fine-tuned)."""
    try:
        import open_clip
        model, _, _ = open_clip.create_model_and_transforms(
            "MobileCLIP-S2", pretrained="datacompdr")
        encoder = model.visual
        if ckpt_path.exists():
            sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            sd = sd.get("model_state_dict", sd)
            visual_sd = {k.replace("visual.", ""): v
                         for k, v in sd.items() if k.startswith("visual.")}
            if visual_sd:
                encoder.load_state_dict(visual_sd, strict=False)
                print(f"  MobileCLIP-ft4 visual 가중치 로드: {ckpt_path.name}")
            else:
                print(f"  [경고] ft4 체크포인트에 visual 키 없음. datacompdr 가중치 사용.")
        else:
            print(f"  [경고] ft4 체크포인트 없음. datacompdr 가중치 사용.")
        return encoder.eval()
    except Exception as e:
        print(f"  [에러] MobileCLIP 로드 실패: {e}")
        return None


class SpecialistMv5b(nn.Module):
    """
    SpecM-v5b: MobileCLIP(frozen, 512d) + SRM-LightCNN(128d)
    ==========================================================
    MNV2 대비 CKA 목표: ~0.03~0.05 (v4 0.725 → 대폭 감소)

    설계 근거:
      - MobileCLIP: 대조학습 기반 의미론적 표현
                    → CNN 귀납 편향 없이 전역 패턴 포착
                    → MNV2(CNN 기반)와 근본적으로 다른 표현 공간
      - SRM-LightCNN: 고주파 노이즈 불일치 잔차
                    → MobileNetV2 대신 경량 depthwise-sep CNN 사용
                    → MNV2 noise branch와 구조적 차별화
    """
    def __init__(self, clip_encoder: nn.Module, dropout: float = 0.3):
        super().__init__()
        # MobileCLIP: frozen (gradient 없음)
        self.clip_encoder = clip_encoder
        for p in self.clip_encoder.parameters():
            p.requires_grad_(False)

        # SRM 고정 필터 + 경량 CNN
        self.srm         = SRMExtractor()
        self.srm_cnn     = SRMLightCNN(out_dim=128)

        fused = 512 + 128  # 640
        self.head = nn.Sequential(
            nn.LayerNorm(fused),
            nn.Linear(fused, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 3, 256, 256], NOT normalized (raw [0,1] ToTensor output)"""
        # CLIP branch (normalize for CLIP)
        x_clip     = _normalize(x, CLIP_MEAN, CLIP_STD)
        clip_feat  = self.clip_encoder(x_clip)            # (B, 512)

        # SRM branch (normalize SRM output to ~[-1,1])
        srm_out    = self.srm(x)                          # (B, 3, H, W) [0,1]
        srm_norm   = _normalize(srm_out, [0.5]*3, [0.5]*3)
        srm_feat   = self.srm_cnn(srm_norm)               # (B, 128)

        return self.head(torch.cat([clip_feat, srm_feat], dim=1))

    def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """GAP 후 fused embedding (640d), head 전"""
        x_clip    = _normalize(x, CLIP_MEAN, CLIP_STD)
        clip_feat = self.clip_encoder(x_clip)
        srm_out   = self.srm(x)
        srm_norm  = _normalize(srm_out, [0.5]*3, [0.5]*3)
        srm_feat  = self.srm_cnn(srm_norm)
        return torch.cat([clip_feat, srm_feat], dim=1)


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float = 0.6):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        w  = torch.where(targets == 1,
                         torch.full_like(ce, self.alpha),
                         torch.full_like(ce, 1 - self.alpha))
        return (w * (1 - pt) ** self.gamma * ce).mean()


# ── 데이터셋 ─────────────────────────────────────────────────────────────
def collect_training_data() -> Tuple[List, List]:
    auth_paths, manip_paths = [], []

    if CASIA2_AU.exists():
        for p in CASIA2_AU.iterdir():
            if p.suffix.lower() in IMG_EXTS:
                auth_paths.append((p, 0))
    print(f"  CASIA2 Au: {len(auth_paths)}")

    gen_auth = []
    if GENIMAGE_NATURE.exists():
        for p in GENIMAGE_NATURE.iterdir():
            if p.suffix.lower() in IMG_EXTS:
                gen_auth.append((p, 0))
        random.shuffle(gen_auth)
        gen_auth = gen_auth[:GENIMAGE_AUTH_MAX]
    auth_paths.extend(gen_auth)
    print(f"  GenImage_nature: {len(gen_auth)}")

    if CASIA2_TP.exists():
        for p in CASIA2_TP.iterdir():
            if p.suffix.lower() in IMG_EXTS:
                manip_paths.append((p, 1))
    print(f"  CASIA2 Tp: {len(manip_paths)}")

    imd = []
    if IMD2020_DIR.exists():
        all_imd = [p for p in IMD2020_DIR.iterdir() if p.suffix.lower() in IMG_EXTS]
        random.shuffle(all_imd)
        imd = [(p, 1) for p in all_imd[:1710]]
    manip_paths.extend(imd)
    print(f"  IMD2020: {len(imd)}")
    print(f"  총: auth={len(auth_paths)}, manip={len(manip_paths)}")
    return auth_paths, manip_paths


class TrainDataset(Dataset):
    def __init__(self, samples: List, transform=None):
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
            tensor = torch.zeros(3, 256, 256)
        return tensor, label


class EvalDataset(Dataset):
    def __init__(self, records: List[Dict], transform=None):
        self.records   = records
        self.transform = transform or VAL_TRANSFORM

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec   = self.records[idx]
        path  = ROOT / rec["image_path"]
        label = LABEL_MAP[rec["true_label"]]
        try:
            img    = Image.open(path).convert("RGB")
            tensor = self.transform(img)
        except Exception:
            tensor = torch.zeros(3, 256, 256)
        return tensor, label, rec["image_path"]


def load_eval_records(jsonl_path: str) -> List[Dict]:
    records = []
    full = ROOT / jsonl_path
    if not full.exists():
        return records
    with open(full) as f:
        for line in f:
            rec = json.loads(line.strip())
            if rec.get("true_label") not in LABEL_MAP:
                continue
            if (ROOT / rec["image_path"]).exists():
                records.append(rec)
    return records


# ── 학습 유틸 ─────────────────────────────────────────────────────────────
def compute_metrics(labels: np.ndarray, preds: np.ndarray) -> Dict:
    acc  = float((labels == preds).mean())
    per, recalls = {}, []
    for idx, name in {0: "authentic", 1: "manipulated"}.items():
        mask = labels == idx
        if mask.sum() == 0:
            per[name] = {"recall": 0.0, "n": 0}
            recalls.append(0.0)
        else:
            r = float((preds[mask] == idx).mean())
            per[name] = {"recall": round(r, 4), "n": int(mask.sum())}
            recalls.append(r)
    tp   = int(((preds == 1) & (labels == 1)).sum())
    fp   = int(((preds == 1) & (labels == 0)).sum())
    fn   = int(((preds == 0) & (labels == 1)).sum())
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


def train_epoch(model, loader, optimizer, scaler, criterion):
    model.train()
    # SRM은 항상 eval (고정 필터)
    model.srm.eval()
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
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
        total_loss += loss.item() * len(labels)
        correct    += (out.argmax(1) == labels).sum().item()
        n          += len(labels)
    return total_loss / n, correct / n


@torch.no_grad()
def eval_epoch(model, loader):
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
    return total_loss / len(all_labels), np.array(all_labels), np.array(all_preds)


@torch.no_grad()
def eval_4ds(model, ds_name: str, jsonl_path: str) -> Dict:
    records = load_eval_records(jsonl_path)
    binary  = [r for r in records if r["true_label"] in ("authentic", "manipulated")]
    if not binary:
        return {}
    ds     = EvalDataset(binary, VAL_TRANSFORM)
    loader = DataLoader(ds, batch_size=64, shuffle=False,
                        num_workers=2, pin_memory=True)
    model.eval()
    all_labels, all_preds = [], []
    for imgs, labels, _ in loader:
        out = model(imgs.to(DEVICE))
        all_labels.extend(labels.numpy())
        all_preds.extend(out.argmax(1).cpu().numpy())
    m    = compute_metrics(np.array(all_labels), np.array(all_preds))
    m["n"] = len(records)
    m["n_binary"] = len(binary)
    return m


# ── 메인 ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=40)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int,   default=64)
    parser.add_argument("--dropout",    type=float, default=0.3)
    parser.add_argument("--eval-only",  type=str,   default="")
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f"Device: {DEVICE}")
    print(f"SpecM-v5b: MobileCLIP(frozen,512d) + SRM-LightCNN(128d) → 640d")
    print(f"목적: MNV2와 CKA~0.03 유지하면서 manipulation 탐지 성능 최대화\n")

    # ── MobileCLIP 로드 ──
    print("[모델 로드] MobileCLIP-S2-ft4...")
    clip_enc = load_mobileclip_encoder(MOBILECLIP_CKPT)
    if clip_enc is None:
        print("MobileCLIP 로드 실패. open_clip 설치 필요: pip install open-clip-torch")
        sys.exit(1)

    model = SpecialistMv5b(clip_encoder=clip_enc.to(DEVICE),
                           dropout=args.dropout).to(DEVICE)

    total_params    = sum(p.numel() for p in model.parameters()) / 1e6
    trainable_params = sum(p.numel() for p in model.parameters()
                           if p.requires_grad) / 1e6
    print(f"  총 파라미터:    {total_params:.2f}M")
    print(f"  학습 파라미터:  {trainable_params:.2f}M "
          f"(MobileCLIP frozen)\n")

    save_dir   = ROOT / "weights" / "specialist_m_v5b"
    result_dir = ROOT / "experiments" / "results" / "specialist_eval"
    save_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    # ── eval-only 모드 ──
    if args.eval_only:
        ckpt = torch.load(args.eval_only, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
        print(f"체크포인트 로드: {args.eval_only}")
        for ds_name, ds_path in EVAL_DATASETS.items():
            m = eval_4ds(model, ds_name, ds_path)
            if m:
                auth_rec  = m["per_class"].get("authentic",   {}).get("recall", 0)
                manip_rec = m["per_class"].get("manipulated", {}).get("recall", 0)
                print(f"  {ds_name}: manip_f1={m['manip_f1']:.4f}, "
                      f"auth_rec={auth_rec:.4f}, manip_rec={manip_rec:.4f}")
        return

    # ── 데이터 수집 ──
    print("[데이터 수집]")
    auth_paths, manip_paths = collect_training_data()
    random.shuffle(auth_paths)
    random.shuffle(manip_paths)
    val_n_auth  = max(1, len(auth_paths)  // 10)
    val_n_manip = max(1, len(manip_paths) // 10)
    val_samples   = auth_paths[:val_n_auth] + manip_paths[:val_n_manip]
    train_samples = auth_paths[val_n_auth:] + manip_paths[val_n_manip:]
    random.shuffle(train_samples)
    print(f"  Train: {len(train_samples)}, Val: {len(val_samples)}\n")

    # WeightedRandomSampler
    train_labels  = [s[1] for s in train_samples]
    cls_counts    = [train_labels.count(c) for c in range(2)]
    cls_weights   = [1.0 / max(c, 1) for c in cls_counts]
    samp_weights  = [cls_weights[l] for l in train_labels]
    sampler = WeightedRandomSampler(samp_weights, len(train_samples), replacement=True)

    train_loader = DataLoader(
        TrainDataset(train_samples, TRAIN_TRANSFORM),
        batch_size=args.batch_size, sampler=sampler,
        num_workers=4, pin_memory=True)
    val_loader = DataLoader(
        TrainDataset(val_samples, VAL_TRANSFORM),
        batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

    # ── 학습 준비 ──
    # MobileCLIP frozen → srm_cnn + head만 최적화
    trainable = [p for p in model.parameters() if p.requires_grad]
    criterion = FocalLoss(gamma=2.0, alpha=0.6)
    optimizer = AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler    = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda"))

    best_manip_f1 = 0.0
    history = []

    print(f"[학습 시작] epochs={args.epochs}, lr={args.lr}, "
          f"batch={args.batch_size}, trainable={trainable_params:.2f}M")
    print(f"{'Ep':>4} {'TrLoss':>9} {'ValLoss':>9} "
          f"{'ManipF1':>8} {'AuthRec':>8} {'ManipRec':>9} {'Best':>5}")
    print("-" * 60)

    for epoch in range(1, args.epochs + 1):
        train_loss, _ = train_epoch(model, train_loader, optimizer, scaler, criterion)
        val_loss, val_labels, val_preds = eval_epoch(model, val_loader)
        m = compute_metrics(val_labels, val_preds)
        scheduler.step()

        f1        = m["manip_f1"]
        auth_rec  = m["per_class"].get("authentic",   {}).get("recall", 0)
        manip_rec = m["per_class"].get("manipulated", {}).get("recall", 0)
        is_best   = f1 > best_manip_f1

        if is_best:
            best_manip_f1 = f1
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "best_manip_f1": best_manip_f1,
                "history": history,
                "args": vars(args),
                "arch": "SpecialistMv5b_MobileCLIP+SRM_LightCNN",
            }, save_dir / "specialist_m_v5b_best.pth")

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            **m,
        })
        print(f"{epoch:>4} {train_loss:>9.4f} {val_loss:>9.4f} "
              f"{f1:>8.4f} {auth_rec:>8.4f} {manip_rec:>9.4f} "
              f"{'★' if is_best else '':>5}")

    # ── 4-DS 평가 ──
    print(f"\n[4-DS 평가] best manip_f1={best_manip_f1:.4f}")
    ckpt = torch.load(save_dir / "specialist_m_v5b_best.pth",
                      map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ds_results, f1_list = {}, []
    for ds_name, ds_path in EVAL_DATASETS.items():
        m = eval_4ds(model, ds_name, ds_path)
        if m:
            ds_results[ds_name] = m
            f1_list.append(m["manip_f1"])
            auth_rec  = m["per_class"].get("authentic",   {}).get("recall", 0)
            manip_rec = m["per_class"].get("manipulated", {}).get("recall", 0)
            print(f"  {ds_name}: manip_f1={m['manip_f1']:.4f}, "
                  f"auth_rec={auth_rec:.4f}, manip_rec={manip_rec:.4f}")
    avg_f1 = float(np.mean(f1_list)) if f1_list else 0.0
    print(f"  4-DS avg manip_f1={avg_f1:.4f}")
    print(f"  비교: SpecM-v4 avg_f1=0.8165, v5b={avg_f1:.4f}")

    out = {
        "arch": "SpecialistMv5b_MobileCLIP+SRM_LightCNN",
        "timestamp": ts,
        "device": str(DEVICE),
        "args": vars(args),
        "total_params_M": round(total_params, 2),
        "trainable_params_M": round(trainable_params, 2),
        "val_best_manip_f1": best_manip_f1,
        "4ds_avg_manip_f1": round(avg_f1, 4),
        "datasets": ds_results,
        "history": history,
        "design_note": {
            "cka_vs_mnv2_expected": "~0.03~0.05 (MobileCLIP dominance)",
            "clip_branch": "MobileCLIP-S2-ft4 frozen, 512d, CKA=0.028 vs MNV2",
            "srm_branch":  "SRM fixed filter + lightweight depthwise-sep CNN, 128d",
            "fused_dim":   640,
            "v4_comparison": {
                "v4_arch": "RGB+SRM+DCT MNV2 3-branch, 3840d",
                "v4_cka_vs_mnv2": 0.7249,
                "v5b_cka_vs_mnv2": "TBD (expected ~0.03~0.05)",
            },
        },
    }
    out_path = result_dir / f"specialist_m_v5b_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n결과 저장: {out_path}")
    print(f"체크포인트: {save_dir / 'specialist_m_v5b_best.pth'}")


if __name__ == "__main__":
    main()
