#!/usr/bin/env python3
"""
MNV2 ↔ SpecM-v4 Embedding-level CKA 측정
==========================================

두 모델의 penultimate embedding (GAP 후, classifier head 전)에서
Unbiased Linear CKA를 계산하여 표현 중복도를 측정한다.

측정 대상:
  - MNV2 (DualStreamMobileNetV2): fused embedding 2560-dim
  - SpecM-v4 (SpecialistM): fused embedding 3840-dim

Unbiased CKA (Nguyen et al. 2021):
  - Gram 행렬 대각 성분을 0으로 설정하여 미니배치 편향 해소
  - 배치 크기 ≥ 5 필요 (n < 5이면 분모 불안정)

실행:
  .venv-qwen/bin/python experiments/run_cka_embedding_mnv2_specm.py
  .venv-qwen/bin/python experiments/run_cka_embedding_mnv2_specm.py --batch-size 128
"""

from __future__ import annotations

import json
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
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
LABEL_MAP     = {"authentic": 0, "manipulated": 1, "ai_generated": 2}
IDX2LABEL     = {v: k for k, v in LABEL_MAP.items()}

# ── 체크포인트 경로 ──────────────────────────────────────────────────────
MNV2_CKPT   = ROOT / "weights" / "mobilenetv2_dualstream" / "mobilenetv2_dualstream_best.pth"
SPECM_CKPT  = ROOT / "weights" / "specialist_m_v4" / "specialist_m_v4_best.pth"

# ── 평가 데이터셋 ────────────────────────────────────────────────────────
EVAL_DATASETS = {
    "base":       "experiments/results/phase2_patha_scale500_gain_predictor/patha_agent_outputs_20260304_080157.jsonl",
    "dsC":        "experiments/results/phase2_patha_case3_scale300_dsC/patha_agent_outputs_20260303_105005.jsonl",
    "opensdi":    "experiments/results/phase2_patha_case3_opensdi_scale300/patha_agent_outputs_fixed_seed42.jsonl",
    "aigenproxy": "experiments/results/phase2_patha_case3_aigenproxy_scale300/patha_agent_outputs_fixed_seed42.jsonl",
}

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])


# ── 데이터셋 로더 ────────────────────────────────────────────────────────
def load_records(jsonl_path: str) -> List[Dict]:
    """JSONL에서 image_path + true_label 레코드 로드"""
    records = []
    full_path = ROOT / jsonl_path
    if not full_path.exists():
        print(f"  [WARN] {jsonl_path} 없음, 스킵")
        return records
    with open(full_path, "r") as f:
        for line in f:
            rec = json.loads(line.strip())
            if "true_label" not in rec or rec["true_label"] not in LABEL_MAP:
                continue
            img_path = ROOT / rec["image_path"]
            if img_path.exists():
                records.append(rec)
    return records


class EvalDataset(Dataset):
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
        return tensor, label


# ── 모델 정의 (embedding 추출 모드 포함) ─────────────────────────────────
def normalize_batch(x: torch.Tensor, mean, std) -> torch.Tensor:
    t = torch.tensor(mean if len(mean) == x.shape[1] else mean[:1],
                     device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
    s = torch.tensor(std  if len(std)  == x.shape[1] else std[:1],
                     device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
    return (x - t) / s


# --- MNV2 ---
class SRMNoiseExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        f1 = [[0,0,0,0,0],[0,-1,2,-1,0],[0,2,-4,2,0],[0,-1,2,-1,0],[0,0,0,0,0]]
        f2 = [[-1,2,-2,2,-1],[2,-6,8,-6,2],[-2,8,-12,8,-2],[2,-6,8,-6,2],[-1,2,-2,2,-1]]
        f3 = [[0,0,0,0,0],[0,0,0,0,0],[0,1,-2,1,0],[0,0,0,0,0],[0,0,0,0,0]]
        q = torch.tensor([4.0, 12.0, 2.0]).view(3, 1, 1, 1)
        k = torch.tensor([f1, f2, f3], dtype=torch.float32).unsqueeze(1) / q
        self.register_buffer("kernels", k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gray = x.mean(dim=1, keepdim=True)
        res = F.conv2d(gray, self.kernels, padding=2)
        return (res.clamp(-2.0, 2.0) + 2.0) / 4.0


class MobileNetBranch(nn.Module):
    def __init__(self, in_channels: int = 3, pretrained: bool = True):
        super().__init__()
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None
        base = models.mobilenet_v2(weights=weights)
        if in_channels != 3:
            old = base.features[0][0]
            base.features[0][0] = nn.Conv2d(
                in_channels, old.out_channels,
                kernel_size=old.kernel_size, stride=old.stride,
                padding=old.padding, bias=False)
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out_dim = 1280

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.features(x)).flatten(1)


class DualStreamMobileNetV2(nn.Module):
    def __init__(self, pretrained: bool = True, dropout: float = 0.25):
        super().__init__()
        self.noise_extractor = SRMNoiseExtractor()
        self.rgb_branch = MobileNetBranch(in_channels=3, pretrained=pretrained)
        self.noise_branch = MobileNetBranch(in_channels=3, pretrained=pretrained)
        fused_dim = self.rgb_branch.out_dim + self.noise_branch.out_dim  # 2560
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

    def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """GAP 후 fused embedding (2560-dim) 반환, head 적용 전"""
        rgb = normalize_batch(x, IMAGENET_MEAN, IMAGENET_STD)
        noise = self.noise_extractor(x)
        noise = normalize_batch(noise, IMAGENET_MEAN, IMAGENET_STD)
        rgb_feat = self.rgb_branch(rgb)
        noise_feat = self.noise_branch(noise)
        return torch.cat([rgb_feat, noise_feat], dim=1)


# --- SpecM-v4 ---
class SRMExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        f1 = [[0,0,0,0,0],[0,-1,2,-1,0],[0,2,-4,2,0],[0,-1,2,-1,0],[0,0,0,0,0]]
        f2 = [[-1,2,-2,2,-1],[2,-6,8,-6,2],[-2,8,-12,8,-2],[2,-6,8,-6,2],[-1,2,-2,2,-1]]
        f3 = [[0,0,0,0,0],[0,0,0,0,0],[0,1,-2,1,0],[0,0,0,0,0],[0,0,0,0,0]]
        q = torch.tensor([4.0, 12.0, 2.0]).view(3, 1, 1, 1)
        k = torch.tensor([f1, f2, f3], dtype=torch.float32).unsqueeze(1) / q
        self.register_buffer("kernels", k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gray = x.mean(dim=1, keepdim=True)
        res = F.conv2d(gray, self.kernels, padding=2)
        return (res.clamp(-2.0, 2.0) + 2.0) / 4.0


class DCTResidualExtractor(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gray = x.mean(dim=1, keepdim=True)
        low_pass = F.avg_pool2d(gray, kernel_size=8, stride=8)
        low_pass = F.interpolate(low_pass, size=gray.shape[-2:], mode='nearest')
        residual = (gray - low_pass).clamp(-1.0, 1.0)
        return (residual + 1.0) / 2.0


class SpecialistM(nn.Module):
    def __init__(self, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        self.srm_extractor = SRMExtractor()
        self.dct_extractor = DCTResidualExtractor()
        self.rgb_branch = MobileNetBranch(in_channels=3, pretrained=pretrained)
        self.srm_branch = MobileNetBranch(in_channels=3, pretrained=False)
        self.dct_branch = MobileNetBranch(in_channels=1, pretrained=False)
        fused = self.rgb_branch.out_dim + self.srm_branch.out_dim + self.dct_branch.out_dim  # 3840
        self.head = nn.Sequential(
            nn.LayerNorm(fused),
            nn.Linear(fused, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rgb = normalize_batch(x, IMAGENET_MEAN, IMAGENET_STD)
        srm = self.srm_extractor(x)
        srm = normalize_batch(srm, [0.5]*3, [0.5]*3)
        dct = self.dct_extractor(x)
        dct = normalize_batch(dct, [0.5], [0.5])
        return self.head(torch.cat([
            self.rgb_branch(rgb),
            self.srm_branch(srm),
            self.dct_branch(dct),
        ], dim=1))

    def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """GAP 후 fused embedding (3840-dim) 반환, head 적용 전"""
        rgb = normalize_batch(x, IMAGENET_MEAN, IMAGENET_STD)
        srm = self.srm_extractor(x)
        srm = normalize_batch(srm, [0.5]*3, [0.5]*3)
        dct = self.dct_extractor(x)
        dct = normalize_batch(dct, [0.5], [0.5])
        return torch.cat([
            self.rgb_branch(rgb),
            self.srm_branch(srm),
            self.dct_branch(dct),
        ], dim=1)


# ── Unbiased Linear CKA (Nguyen et al. 2021) ────────────────────────────
def unbiased_hsic(K: np.ndarray, L: np.ndarray) -> float:
    """
    Unbiased HSIC estimator.
    K, L: n×n Gram 행렬 (대각 이미 0으로 설정됨)
    """
    n = K.shape[0]
    if n < 5:
        raise ValueError(f"n={n} < 5: unbiased HSIC는 n≥5 필요")
    ones = np.ones(n)
    # trace(KL)
    term1 = np.trace(K @ L)
    # 1^T K 1 · 1^T L 1
    term2 = (ones @ K @ ones) * (ones @ L @ ones) / ((n-1) * (n-2))
    # 2/(n-2) · 1^T KL 1
    term3 = 2.0 / (n - 2) * (ones @ K @ L @ ones)
    return (term1 + term2 - term3) / (n * (n - 3))


def compute_unbiased_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Unbiased Linear CKA.
    X: (n, d1), Y: (n, d2) — n개 샘플의 embedding 행렬
    """
    # Linear kernel Gram matrices
    Kx = X @ X.T
    Ky = Y @ Y.T
    # 대각 성분 제거 (unbiased estimator 핵심)
    np.fill_diagonal(Kx, 0.0)
    np.fill_diagonal(Ky, 0.0)

    hsic_xy = unbiased_hsic(Kx, Ky)
    hsic_xx = unbiased_hsic(Kx, Kx)
    hsic_yy = unbiased_hsic(Ky, Ky)

    denom = np.sqrt(max(hsic_xx, 0.0) * max(hsic_yy, 0.0))
    if denom < 1e-12:
        return 0.0
    return float(hsic_xy / denom)


def compute_naive_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """비교용 Naive (biased) Linear CKA"""
    Kx = X @ X.T
    Ky = Y @ Y.T
    # center
    n = Kx.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    Kxc = H @ Kx @ H
    Kyc = H @ Ky @ H
    hsic_xy = np.trace(Kxc @ Kyc) / (n - 1)**2
    hsic_xx = np.trace(Kxc @ Kxc) / (n - 1)**2
    hsic_yy = np.trace(Kyc @ Kyc) / (n - 1)**2
    denom = np.sqrt(max(hsic_xx, 0.0) * max(hsic_yy, 0.0))
    if denom < 1e-12:
        return 0.0
    return float(hsic_xy / denom)


# ── Embedding 추출 ───────────────────────────────────────────────────────
@torch.no_grad()
def extract_embeddings(
    model_mnv2: DualStreamMobileNetV2,
    model_specm: SpecialistM,
    dataloader: DataLoader,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """두 모델의 embedding을 동시에 추출"""
    mnv2_embs, specm_embs, labels_all = [], [], []

    for batch_imgs, batch_labels in dataloader:
        batch_imgs = batch_imgs.to(DEVICE)
        emb_mnv2 = model_mnv2.extract_embedding(batch_imgs)   # (B, 2560)
        emb_specm = model_specm.extract_embedding(batch_imgs)  # (B, 3840)
        mnv2_embs.append(emb_mnv2.cpu().numpy())
        specm_embs.append(emb_specm.cpu().numpy())
        labels_all.append(batch_labels.numpy())

    return (
        np.concatenate(mnv2_embs, axis=0),
        np.concatenate(specm_embs, axis=0),
        np.concatenate(labels_all, axis=0),
    )


# ── 브랜치별 CKA 분석 ───────────────────────────────────────────────────
@torch.no_grad()
def extract_branch_embeddings(
    model_mnv2: DualStreamMobileNetV2,
    model_specm: SpecialistM,
    dataloader: DataLoader,
) -> Dict[str, np.ndarray]:
    """개별 브랜치 embedding 추출 (세밀한 중복도 분석용)"""
    results = {k: [] for k in [
        "mnv2_rgb", "mnv2_noise",
        "specm_rgb", "specm_srm", "specm_dct",
    ]}

    for batch_imgs, _ in dataloader:
        batch_imgs = batch_imgs.to(DEVICE)

        # MNV2 branches
        rgb = normalize_batch(batch_imgs, IMAGENET_MEAN, IMAGENET_STD)
        noise = model_mnv2.noise_extractor(batch_imgs)
        noise = normalize_batch(noise, IMAGENET_MEAN, IMAGENET_STD)
        results["mnv2_rgb"].append(model_mnv2.rgb_branch(rgb).cpu().numpy())
        results["mnv2_noise"].append(model_mnv2.noise_branch(noise).cpu().numpy())

        # SpecM branches
        rgb_s = normalize_batch(batch_imgs, IMAGENET_MEAN, IMAGENET_STD)
        srm = model_specm.srm_extractor(batch_imgs)
        srm = normalize_batch(srm, [0.5]*3, [0.5]*3)
        dct = model_specm.dct_extractor(batch_imgs)
        dct = normalize_batch(dct, [0.5], [0.5])
        results["specm_rgb"].append(model_specm.rgb_branch(rgb_s).cpu().numpy())
        results["specm_srm"].append(model_specm.srm_branch(srm).cpu().numpy())
        results["specm_dct"].append(model_specm.dct_branch(dct).cpu().numpy())

    return {k: np.concatenate(v, axis=0) for k, v in results.items()}


# ── 메인 ─────────────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=0,
                        help="데이터셋별 최대 샘플 수 (0=전부)")
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    print(f"MNV2 ckpt:  {MNV2_CKPT}")
    print(f"SpecM ckpt: {SPECM_CKPT}")

    # ── 모델 로드 ──
    print("\n[1/4] 모델 로드...")
    model_mnv2 = DualStreamMobileNetV2(pretrained=False).to(DEVICE)
    ckpt = torch.load(MNV2_CKPT, map_location=DEVICE, weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    model_mnv2.load_state_dict(sd)
    model_mnv2.eval()
    print(f"  MNV2 로드 완료 (embedding dim: 2560)")

    model_specm = SpecialistM(pretrained=False).to(DEVICE)
    ckpt = torch.load(SPECM_CKPT, map_location=DEVICE, weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    model_specm.load_state_dict(sd)
    model_specm.eval()
    print(f"  SpecM-v4 로드 완료 (embedding dim: 3840)")

    # ── 데이터셋별 CKA 측정 ──
    all_results = {}
    all_mnv2_embs = []
    all_specm_embs = []
    all_labels = []

    print(f"\n[2/4] 데이터셋별 embedding 추출 & CKA 계산...")
    for ds_name, ds_path in EVAL_DATASETS.items():
        records = load_records(ds_path)
        if not records:
            print(f"  {ds_name}: 0 records, 스킵")
            continue
        if args.max_samples > 0:
            records = records[:args.max_samples]

        dataset = EvalDataset(records)
        loader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=2, pin_memory=True)

        mnv2_emb, specm_emb, labels = extract_embeddings(
            model_mnv2, model_specm, loader)

        # Unbiased CKA (전체)
        cka_unbiased = compute_unbiased_cka(mnv2_emb, specm_emb)
        cka_naive = compute_naive_cka(mnv2_emb, specm_emb)

        # 클래스별 CKA
        class_cka = {}
        for cls_idx, cls_name in IDX2LABEL.items():
            mask = labels == cls_idx
            if mask.sum() >= 5:
                class_cka[cls_name] = {
                    "unbiased": compute_unbiased_cka(mnv2_emb[mask], specm_emb[mask]),
                    "naive": compute_naive_cka(mnv2_emb[mask], specm_emb[mask]),
                    "n": int(mask.sum()),
                }

        ds_result = {
            "n_samples": len(records),
            "cka_unbiased": round(cka_unbiased, 6),
            "cka_naive": round(cka_naive, 6),
            "per_class_cka": class_cka,
        }
        all_results[ds_name] = ds_result
        all_mnv2_embs.append(mnv2_emb)
        all_specm_embs.append(specm_emb)
        all_labels.append(labels)

        print(f"  {ds_name}: n={len(records)}, "
              f"CKA(unbiased)={cka_unbiased:.4f}, CKA(naive)={cka_naive:.4f}")
        for cls_name, cls_info in class_cka.items():
            print(f"    {cls_name}: CKA={cls_info['unbiased']:.4f} (n={cls_info['n']})")

    # ── 전체 통합 CKA ──
    if all_mnv2_embs:
        print(f"\n[3/4] 전체 통합 CKA 계산...")
        combined_mnv2 = np.concatenate(all_mnv2_embs, axis=0)
        combined_specm = np.concatenate(all_specm_embs, axis=0)
        combined_labels = np.concatenate(all_labels, axis=0)

        overall_cka = compute_unbiased_cka(combined_mnv2, combined_specm)
        overall_naive = compute_naive_cka(combined_mnv2, combined_specm)
        print(f"  전체: n={len(combined_mnv2)}, "
              f"CKA(unbiased)={overall_cka:.4f}, CKA(naive)={overall_naive:.4f}")

        all_results["overall"] = {
            "n_samples": len(combined_mnv2),
            "cka_unbiased": round(overall_cka, 6),
            "cka_naive": round(overall_naive, 6),
        }

        # 전체 클래스별
        overall_class_cka = {}
        for cls_idx, cls_name in IDX2LABEL.items():
            mask = combined_labels == cls_idx
            if mask.sum() >= 5:
                overall_class_cka[cls_name] = {
                    "unbiased": round(compute_unbiased_cka(
                        combined_mnv2[mask], combined_specm[mask]), 6),
                    "n": int(mask.sum()),
                }
                print(f"    {cls_name}: CKA={overall_class_cka[cls_name]['unbiased']:.4f} "
                      f"(n={overall_class_cka[cls_name]['n']})")
        all_results["overall"]["per_class_cka"] = overall_class_cka

    # ── 브랜치별 CKA 매트릭스 ──
    print(f"\n[4/4] 브랜치별 CKA 매트릭스 계산...")
    if all_mnv2_embs:
        # 전체 데이터에서 브랜치 embedding 추출
        all_records = []
        for ds_path in EVAL_DATASETS.values():
            all_records.extend(load_records(ds_path))
        if args.max_samples > 0:
            all_records = all_records[:args.max_samples * 4]

        dataset_all = EvalDataset(all_records)
        loader_all = DataLoader(dataset_all, batch_size=args.batch_size,
                                shuffle=False, num_workers=2, pin_memory=True)
        branch_embs = extract_branch_embeddings(model_mnv2, model_specm, loader_all)

        branch_names = list(branch_embs.keys())
        n_branches = len(branch_names)
        cka_matrix = {}

        print(f"\n  Branch CKA Matrix (Unbiased):")
        print(f"  {'':>15s}", end="")
        for name in branch_names:
            print(f"  {name:>12s}", end="")
        print()

        for i, name_i in enumerate(branch_names):
            cka_matrix[name_i] = {}
            print(f"  {name_i:>15s}", end="")
            for j, name_j in enumerate(branch_names):
                if i <= j:
                    cka_val = compute_unbiased_cka(
                        branch_embs[name_i], branch_embs[name_j])
                else:
                    cka_val = cka_matrix[name_j][name_i]  # symmetric
                cka_matrix[name_i][name_j] = round(cka_val, 4)
                print(f"  {cka_val:>12.4f}", end="")
            print()

        all_results["branch_cka_matrix"] = cka_matrix

        # MNV2↔SpecM 크로스 브랜치 요약
        cross_pairs = [
            ("mnv2_rgb", "specm_rgb"),
            ("mnv2_rgb", "specm_srm"),
            ("mnv2_rgb", "specm_dct"),
            ("mnv2_noise", "specm_rgb"),
            ("mnv2_noise", "specm_srm"),
            ("mnv2_noise", "specm_dct"),
        ]
        print(f"\n  Cross-model branch CKA:")
        for a, b in cross_pairs:
            val = cka_matrix[a][b]
            print(f"    {a} ↔ {b}: {val:.4f}")

    # ── 결과 저장 ──
    out_dir = ROOT / "experiments" / "results" / "cka_embedding"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"cka_mnv2_specm_v4_{ts}.json"

    all_results["meta"] = {
        "timestamp": ts,
        "device": str(DEVICE),
        "mnv2_ckpt": str(MNV2_CKPT),
        "specm_ckpt": str(SPECM_CKPT),
        "batch_size": args.batch_size,
        "max_samples": args.max_samples,
        "method": "unbiased_linear_cka (Nguyen et al. 2021)",
        "embedding_dim_mnv2": 2560,
        "embedding_dim_specm": 3840,
    }

    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n결과 저장: {out_path}")

    # ── 해석 가이드 ──
    if "overall" in all_results:
        cka = all_results["overall"]["cka_unbiased"]
        print(f"\n{'='*60}")
        print(f"MNV2 ↔ SpecM-v4 Overall Unbiased CKA = {cka:.4f}")
        if cka < 0.2:
            print("→ 매우 낮음: 두 모델이 거의 독립적인 표현 학습. CKA reg 불필요할 수 있음.")
        elif cka < 0.4:
            print("→ 낮음: 적당한 독립성. CKA reg로 추가 분리 가능하나 효과 제한적.")
        elif cka < 0.6:
            print("→ 중간: 상당한 중복. CKA reg 적용 가치 있음.")
        elif cka < 0.8:
            print("→ 높음: 많은 중복. CKA reg 또는 입력 분리(Track A) 강력 권장.")
        else:
            print("→ 매우 높음: 거의 동일한 표현. 근본적 아키텍처 변경 필요.")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
