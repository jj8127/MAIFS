#!/usr/bin/env python3
"""
Specialist-M v5b Fine-tune: MobileCLIP unfreeze + differential LR
==================================================================

v5b frozen 결과: val manip_f1=0.7517, 4-DS avg=0.8062
병목: MobileCLIP frozen → forensics 도메인 적응 불가 (0.21M 학습)

개선:
  - MobileCLIP 전체 unfreeze (differential LR)
    * MobileCLIP trunk:  lr = 1e-5  (사전학습 파괴 방지)
    * SRM CNN + head:    lr = 1e-4  (빠른 적응)
  - v5b best checkpoint에서 resume
  - 20 epoch 추가 학습

CKA 안전성:
  MobileCLIP(ViT 계열) vs MNV2(CNN) → 구조적 inductive bias 차이가 크므로
  같은 forensics 데이터 fine-tuning 후에도 CKA는 낮게 유지될 것으로 예상.

실행:
  .venv-qwen/bin/python experiments/train_specialist_m_v5b_ft.py
  .venv-qwen/bin/python experiments/train_specialist_m_v5b_ft.py --epochs 30 --lr-clip 5e-6
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
from torchvision import transforms

warnings.filterwarnings("ignore")

ROOT   = Path(__file__).resolve().parents[1]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLIP_MEAN = [0.48145466, 0.4578275,  0.40821073]
CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]
LABEL_MAP     = {"authentic": 0, "manipulated": 1, "ai_generated": 2}
IMG_EXTS      = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

CASIA2_AU       = ROOT / "datasets" / "CASIA2_subset" / "Au"
CASIA2_TP       = ROOT / "datasets" / "CASIA2_subset" / "Tp"
IMD2020_DIR     = ROOT / "datasets" / "IMD2020_subset" / \
                  "IMD2020_Generative_Image_Inpainting_yu2018_01" / "images"
GENIMAGE_NATURE = ROOT / "datasets" / "GenImage_subset" / "BigGAN" / "val" / "nature"
MOBILECLIP_CKPT = ROOT / "weights" / "mobileclip_forensics" / "mobileclip_s2_forensics_ft4.pth"
V5B_CKPT        = ROOT / "weights" / "specialist_m_v5b" / "specialist_m_v5b_best.pth"
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
            t = (t + torch.randn_like(t) * random.uniform(*self.std_range)).clamp(0, 1)
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


# ── 모델 (v5b와 동일) ────────────────────────────────────────────────────
def _normalize(x, mean, std):
    m = torch.tensor(mean, device=x.device, dtype=x.dtype).view(1,-1,1,1)
    s = torch.tensor(std,  device=x.device, dtype=x.dtype).view(1,-1,1,1)
    return (x - m) / s

class SRMExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        f1 = [[0,0,0,0,0],[0,-1,2,-1,0],[0,2,-4,2,0],[0,-1,2,-1,0],[0,0,0,0,0]]
        f2 = [[-1,2,-2,2,-1],[2,-6,8,-6,2],[-2,8,-12,8,-2],[2,-6,8,-6,2],[-1,2,-2,2,-1]]
        f3 = [[0,0,0,0,0],[0,0,0,0,0],[0,1,-2,1,0],[0,0,0,0,0],[0,0,0,0,0]]
        q = torch.tensor([4.,12.,2.]).view(3,1,1,1)
        k = torch.tensor([f1,f2,f3], dtype=torch.float32).unsqueeze(1) / q
        self.register_buffer("kernels", k)
    def forward(self, x):
        gray = x.mean(dim=1, keepdim=True)
        return (F.conv2d(gray, self.kernels, padding=2).clamp(-2.,2.) + 2.) / 4.

class SRMLightCNN(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3,32,3,stride=2,padding=1,bias=False), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32,32,3,stride=2,padding=1,groups=32,bias=False), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32,64,1,bias=False), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64,64,3,stride=2,padding=1,groups=64,bias=False), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64,128,1,bias=False), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128,128,3,stride=2,padding=1,groups=128,bias=False), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128,out_dim,1,bias=False), nn.BatchNorm2d(out_dim), nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1))
    def forward(self, x):
        return self.net(x).flatten(1)

class SpecialistMv5b(nn.Module):
    def __init__(self, clip_encoder, dropout=0.3):
        super().__init__()
        self.clip_encoder = clip_encoder
        self.srm     = SRMExtractor()
        self.srm_cnn = SRMLightCNN(out_dim=128)
        self.head = nn.Sequential(
            nn.LayerNorm(640), nn.Linear(640,256), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(256,64), nn.GELU(),
            nn.Dropout(dropout*0.5), nn.Linear(64,2))

    def forward(self, x):
        x_clip    = _normalize(x, CLIP_MEAN, CLIP_STD)
        clip_feat = self.clip_encoder(x_clip)
        srm_norm  = _normalize(self.srm(x), [0.5]*3, [0.5]*3)
        srm_feat  = self.srm_cnn(srm_norm)
        return self.head(torch.cat([clip_feat, srm_feat], dim=1))

    def extract_embedding(self, x):
        x_clip    = _normalize(x, CLIP_MEAN, CLIP_STD)
        clip_feat = self.clip_encoder(x_clip)
        srm_norm  = _normalize(self.srm(x), [0.5]*3, [0.5]*3)
        srm_feat  = self.srm_cnn(srm_norm)
        return torch.cat([clip_feat, srm_feat], dim=1)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.6):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        w  = torch.where(targets==1,
                         torch.full_like(ce, self.alpha),
                         torch.full_like(ce, 1-self.alpha))
        return (w * (1-pt)**self.gamma * ce).mean()


# ── 데이터 ────────────────────────────────────────────────────────────────
def collect_data():
    auth, manip = [], []
    if CASIA2_AU.exists():
        auth += [(p,0) for p in CASIA2_AU.iterdir() if p.suffix.lower() in IMG_EXTS]
    gen = []
    if GENIMAGE_NATURE.exists():
        gen = [(p,0) for p in GENIMAGE_NATURE.iterdir() if p.suffix.lower() in IMG_EXTS]
        random.shuffle(gen); gen = gen[:GENIMAGE_AUTH_MAX]
    auth += gen
    if CASIA2_TP.exists():
        manip += [(p,1) for p in CASIA2_TP.iterdir() if p.suffix.lower() in IMG_EXTS]
    if IMD2020_DIR.exists():
        imd = [(p,1) for p in IMD2020_DIR.iterdir() if p.suffix.lower() in IMG_EXTS]
        random.shuffle(imd); manip += imd[:1710]
    print(f"  auth={len(auth)}, manip={len(manip)}")
    return auth, manip

class TrainDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples   = samples
        self.transform = transform or TRAIN_TRANSFORM
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            tensor = self.transform(Image.open(path).convert("RGB"))
        except Exception:
            tensor = torch.zeros(3,256,256)
        return tensor, label

class EvalDataset(Dataset):
    def __init__(self, records):
        self.records = records
    def __len__(self): return len(self.records)
    def __getitem__(self, idx):
        rec   = self.records[idx]
        path  = ROOT / rec["image_path"]
        label = LABEL_MAP[rec["true_label"]]
        try:
            tensor = VAL_TRANSFORM(Image.open(path).convert("RGB"))
        except Exception:
            tensor = torch.zeros(3,256,256)
        return tensor, label, rec["image_path"]

def load_eval_records(jsonl_path):
    records = []
    full = ROOT / jsonl_path
    if not full.exists(): return records
    with open(full) as f:
        for line in f:
            rec = json.loads(line.strip())
            if rec.get("true_label") not in LABEL_MAP: continue
            if (ROOT / rec["image_path"]).exists():
                records.append(rec)
    return records


# ── 학습/평가 유틸 ────────────────────────────────────────────────────────
def compute_metrics(labels, preds):
    acc  = float((labels == preds).mean())
    per, recalls = {}, []
    for idx, name in {0:"authentic",1:"manipulated"}.items():
        mask = labels == idx
        if not mask.any():
            per[name] = {"recall":0.,"n":0}; recalls.append(0.); continue
        r = float((preds[mask] == idx).mean())
        per[name] = {"recall":round(r,4),"n":int(mask.sum())}
        recalls.append(r)
    tp = int(((preds==1)&(labels==1)).sum())
    fp = int(((preds==1)&(labels==0)).sum())
    fn = int(((preds==0)&(labels==1)).sum())
    prec = tp/(tp+fp) if tp+fp>0 else 0.
    rec  = tp/(tp+fn) if tp+fn>0 else 0.
    f1   = 2*prec*rec/(prec+rec) if prec+rec>0 else 0.
    return {"acc":round(acc,4),"manip_f1":round(f1,4),
            "manip_prec":round(prec,4),"manip_rec":round(rec,4),
            "macro_recall":round(float(np.mean(recalls)),4),"per_class":per}

def train_epoch(model, loader, optimizer, scaler, criterion):
    model.train(); model.srm.eval()
    total_loss, correct, n = 0., 0, 0
    use_amp = DEVICE.type == "cuda"
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=use_amp):
            loss = criterion(model(imgs), labels)
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total_loss += loss.item()*len(labels); n += len(labels)
    return total_loss/n

@torch.no_grad()
def eval_epoch(model, loader):
    model.eval()
    total_loss, all_labels, all_preds = 0., [], []
    crit = FocalLoss()
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        out  = model(imgs)
        total_loss += crit(out, labels).item()*len(labels)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(out.argmax(1).cpu().numpy())
    return total_loss/len(all_labels), np.array(all_labels), np.array(all_preds)

@torch.no_grad()
def eval_4ds(model):
    results = {}
    f1_list = []
    for ds_name, ds_path in EVAL_DATASETS.items():
        records = load_eval_records(ds_path)
        binary  = [r for r in records if r["true_label"] in ("authentic","manipulated")]
        if not binary: continue
        loader = DataLoader(EvalDataset(binary), batch_size=64,
                            shuffle=False, num_workers=2, pin_memory=True)
        model.eval()
        all_l, all_p = [], []
        for imgs, labels, _ in loader:
            out = model(imgs.to(DEVICE))
            all_l.extend(labels.numpy()); all_p.extend(out.argmax(1).cpu().numpy())
        m = compute_metrics(np.array(all_l), np.array(all_p))
        m["n_binary"] = len(binary)
        results[ds_name] = m
        f1_list.append(m["manip_f1"])
    avg = float(np.mean(f1_list)) if f1_list else 0.
    return results, avg


# ── 메인 ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",   type=int,   default=20)
    parser.add_argument("--lr-clip",  type=float, default=1e-5,
                        help="MobileCLIP backbone learning rate (작게 유지)")
    parser.add_argument("--lr-head",  type=float, default=1e-4,
                        help="SRM CNN + head learning rate")
    parser.add_argument("--dropout",  type=float, default=0.3)
    parser.add_argument("--seed",     type=int,   default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f"Device: {DEVICE}")
    print(f"SpecM-v5b fine-tune: MobileCLIP UNFREEZE (lr={args.lr_clip}) + "
          f"head (lr={args.lr_head})")
    print(f"Resume: {V5B_CKPT}\n")

    # ── MobileCLIP 로드 ──
    try:
        import open_clip
        clip_model, _, _ = open_clip.create_model_and_transforms(
            "MobileCLIP-S2", pretrained="datacompdr")
        encoder = clip_model.visual
        if MOBILECLIP_CKPT.exists():
            sd = torch.load(MOBILECLIP_CKPT, map_location="cpu", weights_only=False)
            sd = sd.get("model_state_dict", sd)
            vis_sd = {k.replace("visual.",""):v for k,v in sd.items()
                      if k.startswith("visual.")}
            if vis_sd:
                encoder.load_state_dict(vis_sd, strict=False)
    except Exception as e:
        print(f"MobileCLIP 로드 실패: {e}"); sys.exit(1)

    model = SpecialistMv5b(clip_encoder=encoder.to(DEVICE),
                           dropout=args.dropout).to(DEVICE)

    # v5b best 가중치 로드
    if not V5B_CKPT.exists():
        print(f"v5b 체크포인트 없음: {V5B_CKPT}"); sys.exit(1)
    ckpt = torch.load(V5B_CKPT, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
    prev_best = ckpt.get("best_manip_f1", 0.0)
    print(f"  v5b best 로드 완료 (prev best_manip_f1={prev_best:.4f})")

    # MobileCLIP 전체 unfreeze
    for p in model.clip_encoder.parameters():
        p.requires_grad_(True)

    total_p    = sum(p.numel() for p in model.parameters()) / 1e6
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"  총 파라미터: {total_p:.2f}M, 학습 파라미터: {trainable_p:.2f}M "
          f"(frozen→unfreeze)\n")

    save_dir   = ROOT / "weights" / "specialist_m_v5b_ft"
    result_dir = ROOT / "experiments" / "results" / "specialist_eval"
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── 데이터 ──
    print("[데이터 수집]")
    auth_paths, manip_paths = collect_data()
    random.shuffle(auth_paths); random.shuffle(manip_paths)
    val_n_auth  = max(1, len(auth_paths)  // 10)
    val_n_manip = max(1, len(manip_paths) // 10)
    val_samples   = auth_paths[:val_n_auth] + manip_paths[:val_n_manip]
    train_samples = auth_paths[val_n_auth:] + manip_paths[val_n_manip:]
    random.shuffle(train_samples)
    print(f"  Train={len(train_samples)}, Val={len(val_samples)}\n")

    train_labels = [s[1] for s in train_samples]
    cls_w = [1./max(train_labels.count(c),1) for c in range(2)]
    sampler = WeightedRandomSampler(
        [cls_w[l] for l in train_labels], len(train_samples), replacement=True)
    train_loader = DataLoader(TrainDataset(train_samples), batch_size=64,
                              sampler=sampler, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(TrainDataset(val_samples, VAL_TRANSFORM),
                              batch_size=64, shuffle=False,
                              num_workers=2, pin_memory=True)

    # ── Differential LR optimizer ──
    clip_params = list(model.clip_encoder.parameters())
    other_params = (list(model.srm_cnn.parameters()) +
                    list(model.head.parameters()))
    optimizer = AdamW([
        {"params": clip_params,  "lr": args.lr_clip,  "weight_decay": 1e-5},
        {"params": other_params, "lr": args.lr_head,  "weight_decay": 1e-4},
    ])
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)
    scaler    = torch.amp.GradScaler("cuda", enabled=(DEVICE.type=="cuda"))
    criterion = FocalLoss(gamma=2.0, alpha=0.6)

    best_manip_f1 = prev_best
    history = []

    print(f"[학습 시작] epochs={args.epochs}, "
          f"lr_clip={args.lr_clip}, lr_head={args.lr_head}")
    print(f"{'Ep':>4} {'TrLoss':>9} {'ValLoss':>9} "
          f"{'ManipF1':>8} {'AuthRec':>8} {'ManipRec':>9} {'Best':>5}")
    print("-" * 60)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, scaler, criterion)
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
                "arch": "SpecialistMv5b_FT_MobileCLIPUnfrozen",
            }, save_dir / "specialist_m_v5b_ft_best.pth")

        history.append({"epoch":epoch,"train_loss":round(train_loss,4),
                        "val_loss":round(val_loss,4),**m})
        print(f"{epoch:>4} {train_loss:>9.4f} {val_loss:>9.4f} "
              f"{f1:>8.4f} {auth_rec:>8.4f} {manip_rec:>9.4f} "
              f"{'★' if is_best else '':>5}")

    # ── 4-DS 평가 ──
    print(f"\n[4-DS 평가] best manip_f1={best_manip_f1:.4f}")
    ckpt = torch.load(save_dir / "specialist_m_v5b_ft_best.pth",
                      map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    ds_results, avg_f1 = eval_4ds(model)

    for ds_name, m in ds_results.items():
        auth_rec  = m["per_class"].get("authentic",   {}).get("recall", 0)
        manip_rec = m["per_class"].get("manipulated", {}).get("recall", 0)
        print(f"  {ds_name}: manip_f1={m['manip_f1']:.4f}, "
              f"auth_rec={auth_rec:.4f}, manip_rec={manip_rec:.4f}")
    print(f"  4-DS avg manip_f1={avg_f1:.4f}")
    print(f"  비교: v5b frozen={prev_best:.4f}(val) / 0.8062(4-DS avg), "
          f"v5b_ft={avg_f1:.4f}(4-DS avg)")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = {
        "arch": "SpecialistMv5b_FT_MobileCLIPUnfrozen",
        "timestamp": ts,
        "args": vars(args),
        "total_params_M": round(total_p, 2),
        "trainable_params_M": round(trainable_p, 2),
        "val_best_manip_f1": best_manip_f1,
        "4ds_avg_manip_f1": round(avg_f1, 4),
        "datasets": ds_results,
        "history": history,
    }
    out_path = result_dir / f"specialist_m_v5b_ft_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n결과 저장: {out_path}")
    print(f"체크포인트: {save_dir / 'specialist_m_v5b_ft_best.pth'}")


if __name__ == "__main__":
    main()
