#!/usr/bin/env python3
"""
모든 가용 모델 vs MNV2 Embedding CKA 비교
==========================================

MNV2와 가장 독립적인 모델/브랜치를 찾아 SpecM-v5 설계의 근거를 마련한다.

비교 대상:
  1. SpecM-v4 (fused 3840-dim)
  2. SpecM-v4 개별 브랜치: RGB(1280), SRM(1280), DCT(1280)
  3. SpecG (CLIP+PiD, 576-dim)
  4. SpecG 개별 브랜치: CLIP(512), PiD(64)
  5. MNV2 자체 브랜치: RGB(1280), Noise/SRM(1280)

기준: MNV2 fused embedding (2560-dim)

실행:
  .venv-qwen/bin/python experiments/run_cka_all_vs_mnv2.py
"""

from __future__ import annotations

import json
import warnings
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

warnings.filterwarnings("ignore")

ROOT   = Path(__file__).resolve().parents[1]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
CLIP_MEAN     = [0.48145466, 0.4578275,  0.40821073]
CLIP_STD      = [0.26862954, 0.26130258, 0.27577711]
LABEL_MAP     = {"authentic": 0, "manipulated": 1, "ai_generated": 2}
IDX2LABEL     = {v: k for k, v in LABEL_MAP.items()}

# ── 체크포인트 ───────────────────────────────────────────────────────────
MNV2_CKPT  = ROOT / "weights" / "mobilenetv2_dualstream" / "mobilenetv2_dualstream_best.pth"
SPECM_CKPT = ROOT / "weights" / "specialist_m_v4" / "specialist_m_v4_best.pth"
SPECG_CKPT = ROOT / "weights" / "specialist_g" / "specialist_g_best.pth"
CLIP_CKPT  = ROOT / "weights" / "mobileclip_forensics" / "mobileclip_s2_forensics_ft4.pth"

EVAL_DATASETS = {
    "base":       "experiments/results/phase2_patha_scale500_gain_predictor/patha_agent_outputs_20260304_080157.jsonl",
    "dsC":        "experiments/results/phase2_patha_case3_scale300_dsC/patha_agent_outputs_20260303_105005.jsonl",
    "opensdi":    "experiments/results/phase2_patha_case3_opensdi_scale300/patha_agent_outputs_fixed_seed42.jsonl",
    "aigenproxy": "experiments/results/phase2_patha_case3_aigenproxy_scale300/patha_agent_outputs_fixed_seed42.jsonl",
}

VAL_TRANSFORM_224 = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

VAL_TRANSFORM_256 = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
])


# ── 데이터 ───────────────────────────────────────────────────────────────
def load_records(jsonl_path: str) -> List[Dict]:
    records = []
    full_path = ROOT / jsonl_path
    if not full_path.exists():
        return records
    with open(full_path, "r") as f:
        for line in f:
            rec = json.loads(line.strip())
            if "true_label" not in rec or rec["true_label"] not in LABEL_MAP:
                continue
            if (ROOT / rec["image_path"]).exists():
                records.append(rec)
    return records


class DualResDataset(Dataset):
    """224와 256 크기를 동시에 반환"""
    def __init__(self, records: List[Dict]):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        path = ROOT / rec["image_path"]
        label = LABEL_MAP[rec["true_label"]]
        try:
            img = Image.open(path).convert("RGB")
            t224 = VAL_TRANSFORM_224(img)
            t256 = VAL_TRANSFORM_256(img)
        except Exception:
            t224 = torch.zeros(3, 224, 224)
            t256 = torch.zeros(3, 256, 256)
        return t224, t256, label


# ── 공통 컴포넌트 ────────────────────────────────────────────────────────
def normalize_batch(x, mean, std):
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
        q = torch.tensor([4.0, 12.0, 2.0]).view(3, 1, 1, 1)
        k = torch.tensor([f1, f2, f3], dtype=torch.float32).unsqueeze(1) / q
        self.register_buffer("kernels", k)

    def forward(self, x):
        gray = x.mean(dim=1, keepdim=True)
        res = F.conv2d(gray, self.kernels, padding=2)
        return (res.clamp(-2.0, 2.0) + 2.0) / 4.0


class DCTResidualExtractor(nn.Module):
    def forward(self, x):
        gray = x.mean(dim=1, keepdim=True)
        lp = F.avg_pool2d(gray, kernel_size=8, stride=8)
        lp = F.interpolate(lp, size=gray.shape[-2:], mode='nearest')
        return ((gray - lp).clamp(-1, 1) + 1) / 2


class PiDResidual(nn.Module):
    def __init__(self, n_bits=4):
        super().__init__()
        M = torch.tensor([[0.299,0.587,0.114],[-0.169,-0.331,0.500],[0.500,-0.419,-0.081]])
        self.register_buffer("M", M)
        self.register_buffer("M_inv", torch.inverse(M))
        self.n_bins = 2 ** n_bits

    def forward(self, x):
        B, C, H, W = x.shape
        flat = x.permute(0,2,3,1).reshape(-1,3)
        yuv = flat @ self.M.T
        yuv_q = torch.round(yuv * self.n_bins) / self.n_bins
        rgb_q = yuv_q @ self.M_inv.T
        residual = (flat - rgb_q).abs().reshape(B,H,W,3).permute(0,3,1,2)
        return residual.mean(dim=1, keepdim=True).clamp(0,1)


class MobileNetBranch(nn.Module):
    def __init__(self, in_channels=3, pretrained=True):
        super().__init__()
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None
        base = models.mobilenet_v2(weights=weights)
        if in_channels != 3:
            old = base.features[0][0]
            base.features[0][0] = nn.Conv2d(in_channels, old.out_channels,
                kernel_size=old.kernel_size, stride=old.stride, padding=old.padding, bias=False)
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out_dim = 1280

    def forward(self, x):
        return self.pool(self.features(x)).flatten(1)


# ── MNV2 ─────────────────────────────────────────────────────────────────
class DualStreamMobileNetV2(nn.Module):
    def __init__(self, pretrained=True, dropout=0.25):
        super().__init__()
        self.noise_extractor = SRMExtractor()
        self.rgb_branch = MobileNetBranch(in_channels=3, pretrained=pretrained)
        self.noise_branch = MobileNetBranch(in_channels=3, pretrained=pretrained)
        fused_dim = 2560
        self.head = nn.Sequential(
            nn.LayerNorm(fused_dim), nn.Linear(fused_dim, 512),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(512, 3))

    def extract_embedding(self, x):
        rgb = normalize_batch(x, IMAGENET_MEAN, IMAGENET_STD)
        noise = self.noise_extractor(x)
        noise = normalize_batch(noise, IMAGENET_MEAN, IMAGENET_STD)
        return torch.cat([self.rgb_branch(rgb), self.noise_branch(noise)], dim=1)

    def extract_branches(self, x):
        rgb = normalize_batch(x, IMAGENET_MEAN, IMAGENET_STD)
        noise = self.noise_extractor(x)
        noise = normalize_batch(noise, IMAGENET_MEAN, IMAGENET_STD)
        return {
            "mnv2_rgb":   self.rgb_branch(rgb),
            "mnv2_noise": self.noise_branch(noise),
        }


# ── SpecM-v4 ─────────────────────────────────────────────────────────────
class SpecialistM(nn.Module):
    def __init__(self, pretrained=True, dropout=0.3):
        super().__init__()
        self.srm_extractor = SRMExtractor()
        self.dct_extractor = DCTResidualExtractor()
        self.rgb_branch = MobileNetBranch(in_channels=3, pretrained=pretrained)
        self.srm_branch = MobileNetBranch(in_channels=3, pretrained=False)
        self.dct_branch = MobileNetBranch(in_channels=1, pretrained=False)
        fused = 3840
        self.head = nn.Sequential(
            nn.LayerNorm(fused), nn.Linear(fused, 256),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(256, 2))

    def extract_embedding(self, x):
        rgb = normalize_batch(x, IMAGENET_MEAN, IMAGENET_STD)
        srm = normalize_batch(self.srm_extractor(x), [0.5]*3, [0.5]*3)
        dct = normalize_batch(self.dct_extractor(x), [0.5], [0.5])
        return torch.cat([self.rgb_branch(rgb), self.srm_branch(srm),
                          self.dct_branch(dct)], dim=1)

    def extract_branches(self, x):
        rgb = normalize_batch(x, IMAGENET_MEAN, IMAGENET_STD)
        srm = normalize_batch(self.srm_extractor(x), [0.5]*3, [0.5]*3)
        dct = normalize_batch(self.dct_extractor(x), [0.5], [0.5])
        return {
            "specm_rgb": self.rgb_branch(rgb),
            "specm_srm": self.srm_branch(srm),
            "specm_dct": self.dct_branch(dct),
        }

    def extract_srm_dct_only(self, x):
        """RGB 없이 SRM+DCT만 (2560-dim)"""
        srm = normalize_batch(self.srm_extractor(x), [0.5]*3, [0.5]*3)
        dct = normalize_batch(self.dct_extractor(x), [0.5], [0.5])
        return torch.cat([self.srm_branch(srm), self.dct_branch(dct)], dim=1)


# ── SpecG (CLIP + PiD) ──────────────────────────────────────────────────
class SpecialistG(nn.Module):
    def __init__(self, clip_ckpt: Path, use_pid=True, dropout=0.3):
        super().__init__()
        self.use_pid = use_pid
        self.clip_encoder = self._load_clip(clip_ckpt)
        for p in self.clip_encoder.parameters():
            p.requires_grad_(False)
        clip_dim = 512

        if use_pid:
            self.pid = PiDResidual(n_bits=4)
            self.pid_branch = nn.Sequential(
                nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool2d(1), nn.Flatten())
            fused_dim = clip_dim + 64
        else:
            fused_dim = clip_dim

        self.head = nn.Sequential(
            nn.LayerNorm(fused_dim), nn.Linear(fused_dim, 128),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(128, 2))

    def _load_clip(self, ckpt_path):
        try:
            import open_clip
            model, _, _ = open_clip.create_model_and_transforms("MobileCLIP-S2", pretrained="datacompdr")
            encoder = model.visual
            if ckpt_path.exists():
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                sd = ckpt.get("model_state_dict", ckpt)
                visual_sd = {k.replace("visual.", ""): v for k, v in sd.items() if k.startswith("visual.")}
                if visual_sd:
                    encoder.load_state_dict(visual_sd, strict=False)
            return encoder
        except Exception as e:
            print(f"  [경고] MobileCLIP 로드 실패: {e}, fallback CNN 사용")
            base = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V2)
            return nn.Sequential(base.features, nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(1280, 512))

    def extract_embedding(self, x_clip):
        """fused embedding (576 or 512-dim)"""
        try:
            clip_feat = self.clip_encoder(x_clip)
            if isinstance(clip_feat, tuple):
                clip_feat = clip_feat[0]
            if clip_feat.dim() > 2:
                clip_feat = clip_feat.flatten(1)
        except Exception:
            clip_feat = torch.zeros(x_clip.shape[0], 512, device=x_clip.device)

        if not self.use_pid:
            return clip_feat

        mean = torch.tensor(CLIP_MEAN, device=x_clip.device).view(1,3,1,1)
        std  = torch.tensor(CLIP_STD,  device=x_clip.device).view(1,3,1,1)
        x_raw = (x_clip * std + mean).clamp(0, 1)
        pid_feat = self.pid_branch(self.pid(x_raw))
        return torch.cat([clip_feat, pid_feat], dim=1)

    def extract_branches(self, x_clip):
        try:
            clip_feat = self.clip_encoder(x_clip)
            if isinstance(clip_feat, tuple):
                clip_feat = clip_feat[0]
            if clip_feat.dim() > 2:
                clip_feat = clip_feat.flatten(1)
        except Exception:
            clip_feat = torch.zeros(x_clip.shape[0], 512, device=x_clip.device)

        result = {"specg_clip": clip_feat}
        if self.use_pid:
            mean = torch.tensor(CLIP_MEAN, device=x_clip.device).view(1,3,1,1)
            std  = torch.tensor(CLIP_STD,  device=x_clip.device).view(1,3,1,1)
            x_raw = (x_clip * std + mean).clamp(0, 1)
            result["specg_pid"] = self.pid_branch(self.pid(x_raw))
        return result


# ── Unbiased CKA ─────────────────────────────────────────────────────────
def unbiased_hsic(K, L):
    n = K.shape[0]
    if n < 5:
        return float('nan')
    ones = np.ones(n)
    t1 = np.trace(K @ L)
    t2 = (ones @ K @ ones) * (ones @ L @ ones) / ((n-1) * (n-2))
    t3 = 2.0 / (n-2) * (ones @ K @ L @ ones)
    return (t1 + t2 - t3) / (n * (n-3))


def compute_cka(X, Y):
    Kx = X @ X.T; Ky = Y @ Y.T
    np.fill_diagonal(Kx, 0.0); np.fill_diagonal(Ky, 0.0)
    hxy = unbiased_hsic(Kx, Ky)
    hxx = unbiased_hsic(Kx, Kx)
    hyy = unbiased_hsic(Ky, Ky)
    d = np.sqrt(max(hxx, 0) * max(hyy, 0))
    return float(hxy / d) if d > 1e-12 else 0.0


# ── 메인 ─────────────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    print(f"Device: {DEVICE}")

    # ── 모델 로드 ──
    print("\n[1/3] 모델 로드...")

    model_mnv2 = DualStreamMobileNetV2(pretrained=False).to(DEVICE)
    sd = torch.load(MNV2_CKPT, map_location=DEVICE, weights_only=False)
    model_mnv2.load_state_dict(sd.get("model_state_dict", sd))
    model_mnv2.eval()
    print(f"  MNV2 로드 완료")

    model_specm = SpecialistM(pretrained=False).to(DEVICE)
    sd = torch.load(SPECM_CKPT, map_location=DEVICE, weights_only=False)
    model_specm.load_state_dict(sd.get("model_state_dict", sd))
    model_specm.eval()
    print(f"  SpecM-v4 로드 완료")

    specg_available = False
    model_specg = None
    try:
        model_specg = SpecialistG(clip_ckpt=CLIP_CKPT, use_pid=True).to(DEVICE)
        sd = torch.load(SPECG_CKPT, map_location=DEVICE, weights_only=False)
        model_specg.load_state_dict(sd.get("model_state_dict", sd))
        model_specg.eval()
        specg_available = True
        print(f"  SpecG 로드 완료 (CLIP + PiD)")
    except Exception as e:
        print(f"  SpecG 로드 실패: {e}")

    # ── 데이터 로드 ──
    print("\n[2/3] 데이터 로드...")
    all_records = []
    for ds_name, ds_path in EVAL_DATASETS.items():
        recs = load_records(ds_path)
        print(f"  {ds_name}: {len(recs)} records")
        all_records.extend(recs)
    print(f"  총: {len(all_records)} records")

    dataset = DualResDataset(all_records)
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=False, num_workers=2, pin_memory=True)

    # ── Embedding 추출 ──
    print("\n[3/3] Embedding 추출...")
    embs = {k: [] for k in [
        "mnv2_fused", "mnv2_rgb", "mnv2_noise",
        "specm_fused", "specm_rgb", "specm_srm", "specm_dct", "specm_srm_dct",
    ]}
    if specg_available:
        for k in ["specg_fused", "specg_clip", "specg_pid"]:
            embs[k] = []
    labels_all = []

    with torch.no_grad():
        for i, (img224, img256, labels) in enumerate(loader):
            img224 = img224.to(DEVICE)
            img256 = img256.to(DEVICE)

            # MNV2 (224)
            embs["mnv2_fused"].append(model_mnv2.extract_embedding(img224).cpu().numpy())
            branches = model_mnv2.extract_branches(img224)
            embs["mnv2_rgb"].append(branches["mnv2_rgb"].cpu().numpy())
            embs["mnv2_noise"].append(branches["mnv2_noise"].cpu().numpy())

            # SpecM (224)
            embs["specm_fused"].append(model_specm.extract_embedding(img224).cpu().numpy())
            branches = model_specm.extract_branches(img224)
            embs["specm_rgb"].append(branches["specm_rgb"].cpu().numpy())
            embs["specm_srm"].append(branches["specm_srm"].cpu().numpy())
            embs["specm_dct"].append(branches["specm_dct"].cpu().numpy())
            embs["specm_srm_dct"].append(model_specm.extract_srm_dct_only(img224).cpu().numpy())

            # SpecG (256, CLIP normalized)
            if specg_available:
                clip_norm = normalize_batch(img256, CLIP_MEAN, CLIP_STD)
                embs["specg_fused"].append(model_specg.extract_embedding(clip_norm).cpu().numpy())
                branches = model_specg.extract_branches(clip_norm)
                embs["specg_clip"].append(branches["specg_clip"].cpu().numpy())
                if "specg_pid" in branches:
                    embs["specg_pid"].append(branches["specg_pid"].cpu().numpy())

            labels_all.append(labels.numpy())

            if (i + 1) % 20 == 0:
                print(f"    batch {i+1}/{len(loader)}")

    # Concatenate
    for k in embs:
        if embs[k]:
            embs[k] = np.concatenate(embs[k], axis=0)
    labels_all = np.concatenate(labels_all, axis=0)

    # ── CKA 계산: 모든 것 vs MNV2 fused ──
    print(f"\n{'='*70}")
    print(f"  MNV2 fused (2560-dim) vs 각 모델/브랜치 — Unbiased Linear CKA")
    print(f"{'='*70}")

    ref = embs["mnv2_fused"]
    results = OrderedDict()

    targets = [
        ("specm_fused",   "SpecM-v4 fused (3840d)"),
        ("specm_rgb",     "SpecM-v4 RGB branch (1280d)"),
        ("specm_srm",     "SpecM-v4 SRM branch (1280d)"),
        ("specm_dct",     "SpecM-v4 DCT branch (1280d)"),
        ("specm_srm_dct", "SpecM-v4 SRM+DCT only (2560d)"),
    ]
    if specg_available:
        targets.extend([
            ("specg_fused",  "SpecG fused (576d)"),
            ("specg_clip",   "SpecG CLIP branch (512d)"),
            ("specg_pid",    "SpecG PiD branch (64d)"),
        ])
    # Self-branches for reference
    targets.extend([
        ("mnv2_rgb",   "[ref] MNV2 RGB branch (1280d)"),
        ("mnv2_noise", "[ref] MNV2 Noise/SRM branch (1280d)"),
    ])

    for key, desc in targets:
        if key not in embs or not isinstance(embs[key], np.ndarray):
            continue
        cka = compute_cka(ref, embs[key])
        dim = embs[key].shape[1]

        # 클래스별
        cls_cka = {}
        for ci, cn in IDX2LABEL.items():
            mask = labels_all == ci
            if mask.sum() >= 5:
                cls_cka[cn] = round(compute_cka(ref[mask], embs[key][mask]), 4)

        results[key] = {
            "description": desc,
            "dim": dim,
            "cka_vs_mnv2": round(cka, 4),
            "per_class": cls_cka,
        }
        print(f"\n  {desc}")
        print(f"    Overall CKA = {cka:.4f}")
        for cn, cv in cls_cka.items():
            print(f"    {cn:15s}: {cv:.4f}")

    # ── 크로스 브랜치 CKA 매트릭스 (관심 쌍) ──
    print(f"\n{'='*70}")
    print(f"  크로스 브랜치 CKA 매트릭스 (관심 쌍)")
    print(f"{'='*70}")

    cross_pairs = [
        ("mnv2_rgb", "mnv2_noise"),
        ("mnv2_rgb", "specm_srm"),
        ("mnv2_rgb", "specm_dct"),
        ("mnv2_noise", "specm_srm"),
        ("mnv2_noise", "specm_dct"),
        ("specm_srm", "specm_dct"),
    ]
    if specg_available:
        cross_pairs.extend([
            ("mnv2_rgb", "specg_clip"),
            ("mnv2_rgb", "specg_pid"),
            ("mnv2_noise", "specg_clip"),
            ("mnv2_noise", "specg_pid"),
            ("specg_clip", "specg_pid"),
            ("specm_srm", "specg_pid"),
            ("specm_dct", "specg_pid"),
        ])

    cross_results = {}
    for a, b in cross_pairs:
        if a in embs and b in embs and isinstance(embs[a], np.ndarray) and isinstance(embs[b], np.ndarray):
            cka = compute_cka(embs[a], embs[b])
            cross_results[f"{a}↔{b}"] = round(cka, 4)
            print(f"  {a:15s} ↔ {b:15s}: {cka:.4f}")

    # ── 순위 (MNV2와 가장 독립적인 표현) ──
    print(f"\n{'='*70}")
    print(f"  MNV2와의 독립성 순위 (CKA 낮을수록 독립적)")
    print(f"{'='*70}")

    sorted_results = sorted(results.items(), key=lambda x: x[1]["cka_vs_mnv2"])
    for rank, (key, info) in enumerate(sorted_results, 1):
        marker = " ★" if info["cka_vs_mnv2"] < 0.2 else ""
        print(f"  {rank}. CKA={info['cka_vs_mnv2']:.4f}  {info['description']}{marker}")

    # ── 저장 ──
    out_dir = ROOT / "experiments" / "results" / "cka_embedding"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"cka_all_vs_mnv2_{ts}.json"

    save_data = {
        "meta": {
            "timestamp": ts,
            "device": str(DEVICE),
            "n_samples": len(labels_all),
            "method": "unbiased_linear_cka",
        },
        "cka_vs_mnv2_fused": results,
        "cross_branch_cka": cross_results,
        "ranking": [
            {"rank": i+1, "key": k, "cka": v["cka_vs_mnv2"], "desc": v["description"]}
            for i, (k, v) in enumerate(sorted_results)
        ],
    }
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f"\n결과 저장: {out_path}")


if __name__ == "__main__":
    main()
