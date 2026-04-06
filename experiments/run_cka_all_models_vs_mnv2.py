#!/usr/bin/env python3
"""
프로젝트 전체 모델 vs MNV2 Embedding CKA 비교
================================================

SHIELD Phase 4.6: SpecM-v5 기반 모델 선정을 위해
우리가 학습/사용한 모든 모델 중 MNV2와 가장 독립적인 것을 찾는다.

측정 대상 (경량 — embedding 직접 추출):
  A. MobileCLIP-S2-ft4  : open_clip visual encoder, 512-dim
  B. SpecG              : MobileCLIP + PiD, 576-dim (or 512-dim)
  C. SpecM-v4           : RGB+SRM+DCT, 3840-dim
  D. SpecM-v1           : RGB+SRM+DCT (초기 버전), 3840-dim

측정 대상 (Phase 1 heavy — JSONL 예측값 기반 prediction-space CKA):
  E. CAT-Net            : JPEG 주파수 분석
  F. Mesorch            : ViT 공간 분석
  G. FatFormer          : CLIP AI-gen 탐지

결과 해석:
  CKA < 0.2  → 매우 독립적 → SpecM-v5 기반 후보 1순위
  0.2~0.4    → 독립적      → 2순위
  0.4~0.6    → 중간        → 3순위
  > 0.6      → 중복 높음   → 부적합

실행:
  .venv-qwen/bin/python experiments/run_cka_all_models_vs_mnv2.py
  .venv-qwen/bin/python experiments/run_cka_all_models_vs_mnv2.py --max-samples 200
"""

from __future__ import annotations

import json
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

EVAL_DATASETS = {
    "base":       "experiments/results/phase2_patha_scale500_gain_predictor/patha_agent_outputs_20260304_080157.jsonl",
    "dsC":        "experiments/results/phase2_patha_case3_scale300_dsC/patha_agent_outputs_20260303_105005.jsonl",
    "opensdi":    "experiments/results/phase2_patha_case3_opensdi_scale300/patha_agent_outputs_fixed_seed42.jsonl",
    "aigenproxy": "experiments/results/phase2_patha_case3_aigenproxy_scale300/patha_agent_outputs_fixed_seed42.jsonl",
}

# Phase 1 heavy 모델 평가 결과 경로 (prediction-space CKA용)
BACKBONE_EVAL_DIR = ROOT / "experiments" / "results" / "backbone_eval"


# ── Unbiased Linear CKA ──────────────────────────────────────────────────
def unbiased_hsic(K: np.ndarray, L: np.ndarray) -> float:
    n = K.shape[0]
    if n < 5:
        return 0.0
    ones  = np.ones(n)
    term1 = np.trace(K @ L)
    term2 = (ones @ K @ ones) * (ones @ L @ ones) / ((n-1) * (n-2))
    term3 = 2.0 / (n - 2) * (ones @ K @ L @ ones)
    return (term1 + term2 - term3) / (n * (n - 3))


def compute_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Unbiased Linear CKA (Nguyen et al. 2021)."""
    Kx = X @ X.T
    Ky = Y @ Y.T
    np.fill_diagonal(Kx, 0.0)
    np.fill_diagonal(Ky, 0.0)
    hsic_xy = unbiased_hsic(Kx, Ky)
    hsic_xx = unbiased_hsic(Kx, Kx)
    hsic_yy = unbiased_hsic(Ky, Ky)
    denom = np.sqrt(max(hsic_xx, 0.0) * max(hsic_yy, 0.0))
    return float(hsic_xy / denom) if denom > 1e-12 else 0.0


# ── 데이터셋 ─────────────────────────────────────────────────────────────
def load_records(max_samples: int = 0) -> List[Dict]:
    records = []
    for ds_path in EVAL_DATASETS.values():
        full = ROOT / ds_path
        if not full.exists():
            continue
        with open(full) as f:
            for line in f:
                rec = json.loads(line.strip())
                if rec.get("true_label") not in LABEL_MAP:
                    continue
                if (ROOT / rec["image_path"]).exists():
                    records.append(rec)
    if max_samples > 0:
        # 클래스 균형 맞춰서 샘플링
        by_class: Dict[str, List] = {}
        for r in records:
            by_class.setdefault(r["true_label"], []).append(r)
        per = max_samples // 3
        sampled = []
        for cls_recs in by_class.values():
            sampled.extend(cls_recs[:per])
        records = sampled
    return records


class ImageDataset(Dataset):
    def __init__(self, records: List[Dict], transform):
        self.records   = records
        self.transform = transform

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
            h = 224 if "224" in str(self.transform) else 256
            tensor = torch.zeros(3, h, h)
        return tensor, label, rec["image_path"]


# ── 모델 정의 ─────────────────────────────────────────────────────────────
def _normalize(x, mean, std):
    m = torch.tensor(mean, device=x.device, dtype=x.dtype).view(1,-1,1,1)
    s = torch.tensor(std,  device=x.device, dtype=x.dtype).view(1,-1,1,1)
    return (x - m) / s


class SRMNoiseExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        f1 = [[0,0,0,0,0],[0,-1,2,-1,0],[0,2,-4,2,0],[0,-1,2,-1,0],[0,0,0,0,0]]
        f2 = [[-1,2,-2,2,-1],[2,-6,8,-6,2],[-2,8,-12,8,-2],[2,-6,8,-6,2],[-1,2,-2,2,-1]]
        f3 = [[0,0,0,0,0],[0,0,0,0,0],[0,1,-2,1,0],[0,0,0,0,0],[0,0,0,0,0]]
        q  = torch.tensor([4.0, 12.0, 2.0]).view(3,1,1,1)
        k  = torch.tensor([f1, f2, f3], dtype=torch.float32).unsqueeze(1) / q
        self.register_buffer("kernels", k)

    def forward(self, x):
        gray = x.mean(dim=1, keepdim=True)
        res  = F.conv2d(gray, self.kernels, padding=2)
        return (res.clamp(-2.0, 2.0) + 2.0) / 4.0


class DCTExtractor(nn.Module):
    def forward(self, x):
        gray     = x.mean(dim=1, keepdim=True)
        low      = F.avg_pool2d(gray, 8, 8)
        low      = F.interpolate(low, size=gray.shape[-2:], mode='nearest')
        return ((gray - low).clamp(-1,1) + 1) / 2


class MobileNetBranch(nn.Module):
    def __init__(self, in_channels=3, pretrained=True):
        super().__init__()
        w = models.MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None
        b = models.mobilenet_v2(weights=w)
        if in_channels != 3:
            old = b.features[0][0]
            b.features[0][0] = nn.Conv2d(in_channels, old.out_channels,
                kernel_size=old.kernel_size, stride=old.stride,
                padding=old.padding, bias=False)
        self.features = b.features
        self.pool     = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        return self.pool(self.features(x)).flatten(1)


# MNV2 (DualStreamMobileNetV2) — 임베딩 추출용
class MNV2Embedder(nn.Module):
    def __init__(self):
        super().__init__()
        self.srm = SRMNoiseExtractor()
        self.rgb_b   = MobileNetBranch(3,  True)
        self.noise_b = MobileNetBranch(3,  True)
        self.head = nn.Sequential(
            nn.LayerNorm(2560), nn.Linear(2560,512), nn.GELU(),
            nn.Dropout(0.25), nn.Linear(512,3))

    def embed(self, x):
        rgb   = _normalize(x, IMAGENET_MEAN, IMAGENET_STD)
        noise = _normalize(self.srm(x), IMAGENET_MEAN, IMAGENET_STD)
        return torch.cat([self.rgb_b(rgb), self.noise_b(noise)], dim=1)  # 2560


# SpecM-v4 (RGB+SRM+DCT) — 임베딩 추출용
class SpecMEmbedder(nn.Module):
    def __init__(self):
        super().__init__()
        self.srm = SRMNoiseExtractor()
        self.dct = DCTExtractor()
        self.rgb_b = MobileNetBranch(3, True)
        self.srm_b = MobileNetBranch(3, False)
        self.dct_b = MobileNetBranch(1, False)
        self.head  = nn.Sequential(
            nn.LayerNorm(3840), nn.Linear(3840,256), nn.GELU(),
            nn.Dropout(0.3), nn.Linear(256,2))

    def embed(self, x):
        rgb = _normalize(x, IMAGENET_MEAN, IMAGENET_STD)
        srm = _normalize(self.srm(x), [0.5]*3, [0.5]*3)
        dct = _normalize(self.dct(x), [0.5],   [0.5])
        return torch.cat([self.rgb_b(rgb), self.srm_b(srm), self.dct_b(dct)], 1)  # 3840


# SpecG — MobileCLIP 임베딩 추출용 (train_specialist_g.py SpecialistG 구조 그대로)
class SpecGEmbedder(nn.Module):
    def __init__(self, clip_encoder, use_pid=False):
        super().__init__()
        self.clip_encoder = clip_encoder  # attribute name must match checkpoint
        self.use_pid = use_pid
        if use_pid:
            M = torch.tensor([
                [ 0.299,  0.587,  0.114],
                [-0.169, -0.331,  0.500],
                [ 0.500, -0.419, -0.081],
            ])
            # register as sub-module to match checkpoint keys "pid.M", "pid.M_inv"
            class _PiD(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.register_buffer("M",     M)
                    self.register_buffer("M_inv", torch.inverse(M))
                    self.n_bins = 16
                def forward(self, x):
                    B,C,H,W = x.shape
                    flat  = x.permute(0,2,3,1).reshape(-1,3)
                    yuv   = flat @ self.M.T
                    yuv_q = torch.round(yuv * self.n_bins) / self.n_bins
                    rgb_q = yuv_q @ self.M_inv.T
                    res   = (flat - rgb_q).abs().reshape(B,H,W,3).permute(0,3,1,2)
                    return res.mean(1, keepdim=True).clamp(0,1)
            self.pid = _PiD()
            self.pid_branch = nn.Sequential(
                nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool2d(1), nn.Flatten())
            fused = 576
        else:
            fused = 512
        self.head = nn.Sequential(
            nn.LayerNorm(fused), nn.Linear(fused, 128), nn.GELU(),
            nn.Dropout(0.3), nn.Linear(128, 2))
        self.fused_dim = fused

    def embed(self, x_clip):
        """x_clip: CLIP-normalized [B,3,256,256]"""
        clip_feat = self.clip_encoder(x_clip)  # (B, 512)
        if self.use_pid:
            mean = torch.tensor(CLIP_MEAN, device=x_clip.device).view(1,-1,1,1)
            std  = torch.tensor(CLIP_STD,  device=x_clip.device).view(1,-1,1,1)
            x01  = (x_clip * std + mean).clamp(0, 1)
            pid_feat = self.pid_branch(self.pid(x01))  # (B, 64)
            return torch.cat([clip_feat, pid_feat], dim=1)  # (B, 576)
        return clip_feat  # (B, 512)


# ── 모델 로더 ─────────────────────────────────────────────────────────────
def load_mnv2() -> Optional[MNV2Embedder]:
    ckpt_path = ROOT / "weights" / "mobilenetv2_dualstream" / "mobilenetv2_dualstream_best.pth"
    if not ckpt_path.exists():
        print(f"  [SKIP] MNV2 체크포인트 없음: {ckpt_path}")
        return None
    m = MNV2Embedder().to(DEVICE)
    sd = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    m.load_state_dict(sd.get("model_state_dict", sd), strict=False)
    m.eval()
    return m


def load_specm(version: str) -> Optional[SpecMEmbedder]:
    name_map = {
        "v1": "specialist_m/specialist_m_best.pth",
        "v2": "specialist_m_v2/specialist_m_v2_best.pth",
        "v3": "specialist_m_v3/specialist_m_v3_best.pth",
        "v4": "specialist_m_v4/specialist_m_v4_best.pth",
    }
    ckpt_path = ROOT / "weights" / name_map[version]
    if not ckpt_path.exists():
        print(f"  [SKIP] SpecM-{version} 없음")
        return None
    m = SpecMEmbedder().to(DEVICE)
    sd = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    m.load_state_dict(sd.get("model_state_dict", sd), strict=False)
    m.eval()
    return m


def load_mobileclip_encoder():
    """open_clip MobileCLIP-S2 visual encoder, ft4 fine-tuned."""
    try:
        import open_clip
        model, _, _ = open_clip.create_model_and_transforms(
            "MobileCLIP-S2", pretrained="datacompdr")
        encoder = model.visual
        # ft4 가중치로 visual 부분 업데이트
        ft4_path = ROOT / "weights" / "mobileclip_forensics" / "mobileclip_s2_forensics_ft4.pth"
        if ft4_path.exists():
            sd = torch.load(ft4_path, map_location="cpu", weights_only=False)
            sd = sd.get("model_state_dict", sd)
            visual_sd = {k.replace("visual.", ""): v
                         for k, v in sd.items() if k.startswith("visual.")}
            if visual_sd:
                encoder.load_state_dict(visual_sd, strict=False)
                print(f"  MobileCLIP ft4 visual 가중치 로드")
        encoder = encoder.to(DEVICE).eval()
        return encoder
    except Exception as e:
        print(f"  [SKIP] MobileCLIP 로드 실패: {e}")
        return None


def load_specg() -> Optional[SpecGEmbedder]:
    ckpt_path = ROOT / "weights" / "specialist_g" / "specialist_g_best.pth"
    if not ckpt_path.exists():
        return None
    encoder = load_mobileclip_encoder()
    if encoder is None:
        return None
    # use_pid 여부를 체크포인트 키로 판단
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = sd.get("model_state_dict", sd)
    use_pid = any("pid_branch" in k or "pid_conv" in k for k in sd.keys())
    m = SpecGEmbedder(clip_encoder=encoder, use_pid=use_pid).to(DEVICE)
    m.load_state_dict(sd, strict=False)
    m.eval()
    return m


# ── 임베딩 추출 ──────────────────────────────────────────────────────────
@torch.no_grad()
def extract_mnv2_embeddings(model: MNV2Embedder,
                             records: List[Dict],
                             batch_size: int = 64) -> np.ndarray:
    transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    ds     = ImageDataset(records, transform)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=2, pin_memory=True)
    embs   = []
    for imgs, _, _ in loader:
        embs.append(model.embed(imgs.to(DEVICE)).cpu().numpy())
    return np.concatenate(embs, axis=0)


@torch.no_grad()
def extract_specm_embeddings(model: SpecMEmbedder,
                              records: List[Dict],
                              batch_size: int = 64) -> np.ndarray:
    transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    ds     = ImageDataset(records, transform)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=2, pin_memory=True)
    embs   = []
    for imgs, _, _ in loader:
        embs.append(model.embed(imgs.to(DEVICE)).cpu().numpy())
    return np.concatenate(embs, axis=0)


@torch.no_grad()
def extract_clip_embeddings(model: SpecGEmbedder,
                             records: List[Dict],
                             batch_size: int = 64) -> np.ndarray:
    transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(CLIP_MEAN, CLIP_STD),
    ])
    ds     = ImageDataset(records, transform)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=2, pin_memory=True)
    embs   = []
    for imgs, _, _ in loader:
        embs.append(model.embed(imgs.to(DEVICE)).cpu().numpy())
    return np.concatenate(embs, axis=0)


# ── Phase 1 heavy 모델: JSONL 예측 기반 CKA ─────────────────────────────
def load_backbone_predictions(model_key: str, records: List[Dict]) -> Optional[np.ndarray]:
    """
    backbone_eval JSONL에서 예측 softmax 벡터를 로드.
    MNV2 예측과 동일한 이미지 순서 보장.
    model_key: "catnet", "mesorch", "fatformer" 등
    """
    # backbone_eval 디렉토리에서 해당 모델 결과 찾기
    eval_dir = BACKBONE_EVAL_DIR
    if not eval_dir.exists():
        return None

    # model_key와 매칭되는 jsonl 파일 찾기
    candidates = list(eval_dir.glob(f"*{model_key}*.jsonl"))
    if not candidates:
        candidates = list(eval_dir.glob(f"*{model_key}*.json"))
    if not candidates:
        return None

    jsonl_path = sorted(candidates)[-1]  # 가장 최신

    # image_path → scores 매핑
    pred_map: Dict[str, np.ndarray] = {}
    with open(jsonl_path) as f:
        for line in f:
            try:
                rec = json.loads(line.strip())
                img_path = rec.get("image_path", "")
                scores = rec.get("scores", {})
                if scores:
                    vec = np.array([
                        scores.get("authentic", 0.0),
                        scores.get("manipulated", 0.0),
                        scores.get("ai_generated", 0.0),
                    ], dtype=np.float32)
                    pred_map[img_path] = vec
            except Exception:
                continue

    if not pred_map:
        return None

    # records 순서대로 정렬
    vecs = []
    valid_mask = []
    for r in records:
        ip = r["image_path"]
        if ip in pred_map:
            vecs.append(pred_map[ip])
            valid_mask.append(True)
        else:
            valid_mask.append(False)

    if len(vecs) < 10:
        return None

    return np.array(vecs), valid_mask


# ── 메인 ─────────────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size",  type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=400,
                        help="총 샘플 수 (0=전체, 권장 400~800)")
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    print(f"목적: 프로젝트 전체 모델 중 MNV2와 가장 독립적인 모델 찾기\n")

    # ── 데이터 로드 ──
    records = load_records(args.max_samples)
    print(f"평가 이미지: {len(records)}장")
    label_dist = {}
    for r in records:
        label_dist[r["true_label"]] = label_dist.get(r["true_label"], 0) + 1
    print(f"  분포: {label_dist}\n")

    if len(records) < 10:
        print("샘플 부족. EVAL_DATASETS 경로 확인 필요.")
        return

    # ── MNV2 임베딩 추출 (기준) ──
    print("[1/6] MNV2 임베딩 추출...")
    mnv2_model = load_mnv2()
    if mnv2_model is None:
        print("MNV2 로드 실패. 종료.")
        return
    mnv2_emb = extract_mnv2_embeddings(mnv2_model, records, args.batch_size)
    print(f"  MNV2 embedding: {mnv2_emb.shape}")

    results = {}

    # ── 경량 모델 측정 ──
    # SpecM 버전들
    print("\n[2/6] SpecM 버전별 CKA...")
    for ver in ["v1", "v2", "v3", "v4"]:
        m = load_specm(ver)
        if m is None:
            continue
        emb = extract_specm_embeddings(m, records, args.batch_size)
        cka = compute_cka(mnv2_emb, emb)
        print(f"  SpecM-{ver}: CKA={cka:.4f}  dim={emb.shape[1]}")
        results[f"SpecM-{ver}"] = {
            "cka": round(cka, 4), "dim": emb.shape[1],
            "arch": "RGB+SRM+DCT (MNV2 3-branch)",
        }
        del m

    # MobileCLIP ft4
    print("\n[3/6] MobileCLIP-S2-ft4 CKA...")
    clip_enc = load_mobileclip_encoder()
    if clip_enc is not None:
        # MobileCLIP raw encoder (head 없이)
        class ClipOnlyEmbedder(nn.Module):
            def __init__(self, enc):
                super().__init__()
                self.enc = enc
            def embed(self, x):
                return self.enc(x)

        clip_model = ClipOnlyEmbedder(clip_enc).to(DEVICE)
        transform_clip = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize(CLIP_MEAN, CLIP_STD),
        ])
        ds_clip = ImageDataset(records, transform_clip)
        loader_clip = DataLoader(ds_clip, batch_size=args.batch_size,
                                  num_workers=2, pin_memory=True)
        clip_embs = []
        with torch.no_grad():
            for imgs, _, _ in loader_clip:
                clip_embs.append(clip_enc(imgs.to(DEVICE)).cpu().numpy())
        clip_emb = np.concatenate(clip_embs, axis=0)
        cka_clip = compute_cka(mnv2_emb, clip_emb)
        print(f"  MobileCLIP-S2-ft4: CKA={cka_clip:.4f}  dim={clip_emb.shape[1]}")
        results["MobileCLIP-S2-ft4"] = {
            "cka": round(cka_clip, 4), "dim": clip_emb.shape[1],
            "arch": "CLIP ViT (contrastive pretrain + forensics ft4)",
        }

    # SpecG (MobileCLIP + PiD)
    print("\n[4/6] SpecG CKA...")
    specg = load_specg()
    if specg is not None:
        emb_g = extract_clip_embeddings(specg, records, args.batch_size)
        cka_g = compute_cka(mnv2_emb, emb_g)
        print(f"  SpecG: CKA={cka_g:.4f}  dim={emb_g.shape[1]}")
        results["SpecG"] = {
            "cka": round(cka_g, 4), "dim": emb_g.shape[1],
            "arch": "MobileCLIP + PiD residual",
        }
        del specg

    # ── Phase 1 heavy 모델: prediction-space CKA ──
    print("\n[5/6] Phase 1 heavy 모델 (prediction-space CKA)...")

    # MNV2 자신의 예측 추출 (logit 공간에서 비교)
    mnv2_logits = []
    mnv2_model.eval()
    transform_224 = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    ds_224 = ImageDataset(records, transform_224)
    loader_224 = DataLoader(ds_224, batch_size=args.batch_size, num_workers=2, pin_memory=True)
    with torch.no_grad():
        for imgs, _, _ in loader_224:
            imgs_n = _normalize(imgs.to(DEVICE), IMAGENET_MEAN, IMAGENET_STD)
            noise  = mnv2_model.srm(imgs.to(DEVICE))
            noise_n = _normalize(noise, IMAGENET_MEAN, IMAGENET_STD)
            fused  = torch.cat([mnv2_model.rgb_b(imgs_n), mnv2_model.noise_b(noise_n)], 1)
            logits = mnv2_model.head(fused)
            mnv2_logits.append(torch.softmax(logits, dim=1).cpu().numpy())
    mnv2_pred = np.concatenate(mnv2_logits, axis=0)  # (N, 3)

    # backbone_eval 디렉토리에서 Phase 1 모델 결과 로드
    heavy_models = {
        "catnet":    "CAT-Net (JPEG freq)",
        "mesorch":   "Mesorch (ViT spatial)",
        "fatformer": "FatFormer (CLIP AI-gen)",
    }
    for key, desc in heavy_models.items():
        result = load_backbone_predictions(key, records)
        if result is None:
            print(f"  [{desc}] JSONL 없음 — 스킵")
            continue
        pred_arr, valid_mask = result
        # 유효한 샘플만 비교
        mnv2_valid = mnv2_pred[valid_mask]
        if len(pred_arr) < 10 or len(mnv2_valid) != len(pred_arr):
            print(f"  [{desc}] 샘플 매칭 실패 — 스킵")
            continue
        cka_pred = compute_cka(mnv2_valid, pred_arr)
        print(f"  {desc}: CKA={cka_pred:.4f}  (prediction-space, n={len(pred_arr)})")
        results[desc] = {
            "cka": round(cka_pred, 4),
            "dim": 3,
            "arch": f"{desc} — prediction-space CKA (소프트맥스 출력)",
            "note": "embedding이 아닌 예측 공간 비교 — embedding CKA보다 보수적",
        }

    # ── 결과 요약 및 순위 ──
    print("\n" + "="*65)
    print("결과 요약: MNV2와의 CKA (낮을수록 독립적 → SpecM-v5 기반 후보)")
    print("="*65)
    ranked = sorted(results.items(), key=lambda x: x[1]["cka"])
    print(f"  {'순위':>4}  {'모델':>22}  {'CKA':>7}  {'dim':>6}  해석")
    print(f"  {'-'*60}")
    for rank, (name, info) in enumerate(ranked, 1):
        cka = info["cka"]
        dim = info["dim"]
        if cka < 0.2:
            interp = "★★★ 매우 독립적"
        elif cka < 0.4:
            interp = "★★  독립적"
        elif cka < 0.6:
            interp = "★   중간"
        else:
            interp = "✗   중복 높음"
        print(f"  {rank:>4}. {name:>22}  {cka:>7.4f}  {dim:>6}  {interp}")

    print("\n→ 1순위 모델을 SpecM-v5의 백본으로 채택 권장")
    if ranked:
        best_name, best_info = ranked[0]
        print(f"  채택 후보: {best_name}")
        print(f"  CKA vs MNV2: {best_info['cka']:.4f}")
        print(f"  아키텍처: {best_info['arch']}")

    # ── 결과 저장 ──
    out_dir = ROOT / "experiments" / "results" / "cka_embedding"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = {
        "timestamp": ts,
        "device": str(DEVICE),
        "n_samples": len(records),
        "label_dist": label_dist,
        "method": "unbiased_linear_cka (Nguyen et al. 2021)",
        "reference_model": "MNV2 (DualStreamMobileNetV2, 2560-dim)",
        "ranked": [
            {"rank": r+1, "model": n, **i}
            for r, (n, i) in enumerate(ranked)
        ],
        "all_results": {n: i for n, i in results.items()},
    }
    out_path = out_dir / f"cka_all_models_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n결과 저장: {out_path}")


if __name__ == "__main__":
    main()
