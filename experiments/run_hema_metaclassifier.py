#!/usr/bin/env python3
"""
HEMA (Heterogeneous Expert Meta-Aggregation) PoC 실험
=====================================================

RPi5 2-model 시스템 (MNV2 + SpecM-v4) 에서:
1. SpecM-v4를 전체 이미지(ai_generated 포함)에 대해 추론
2. 13-dim 메타 피처 추출
3. XGBoost 메타 분류기 학습 (Stratified K-Fold)
4. Confidence-gated cascade: ICWMV → HEMA
5. Ablation: f12 (cross-model implicit signal) 제거 시 성능

실행:
    .venv-qwen/bin/python experiments/run_hema_metaclassifier.py
"""

from __future__ import annotations

import argparse
import json
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

warnings.filterwarnings("ignore")

ROOT    = Path(__file__).resolve().parents[1]
DEVICE  = "cpu"  # meta-classifier는 CPU로 충분

CLASSES   = ["authentic", "manipulated", "ai_generated"]
CLS2IDX   = {c: i for i, c in enumerate(CLASSES)}
IDX2CLS   = {i: c for i, c in enumerate(CLASSES)}

BE_DIR  = ROOT / "experiments" / "results" / "backbone_eval"
SE_DIR  = ROOT / "experiments" / "results" / "specialist_eval"
OUT_DIR = ROOT / "experiments" / "results" / "hema"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = ["base", "dsC", "opensdi", "aigenproxy"]

# ── JSONL 파일 경로 ──────────────────────────────────────────────────────── #

MNV2_FILES = {
    ds: BE_DIR / f"mobilenetv2_dualstream_{ds}_20260319_070725.jsonl"
    for ds in DATASETS
}

SPECM_FILES = {
    ds: SE_DIR / f"specialist_m_v4_{ds}_20260320_034133.jsonl"
    for ds in DATASETS
}

# 원본 평가용 JSONL (이미지 경로 + true_label 참조)
EVAL_DATASETS = {
    "base":       "experiments/results/phase2_patha_scale500_gain_predictor/patha_agent_outputs_20260304_080157.jsonl",
    "dsC":        "experiments/results/phase2_patha_case3_scale300_dsC/patha_agent_outputs_20260303_105005.jsonl",
    "opensdi":    "experiments/results/phase2_patha_case3_opensdi_scale300/patha_agent_outputs_fixed_seed42.jsonl",
    "aigenproxy": "experiments/results/phase2_patha_case3_aigenproxy_scale300/patha_agent_outputs_fixed_seed42.jsonl",
}


# ── Step 0: SpecM-v4 전체 추론 (ai_generated 포함) ─────────────────────── #

def run_specm_full_inference() -> Dict[str, Path]:
    """기존 SpecM-v4 JSONL에 없는 ai_generated 이미지에 대해 추론 수행"""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F_torch
    from PIL import Image
    from torchvision import models, transforms

    gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[SpecM-v4 Full Inference] device={gpu}")

    # SpecialistM 모델 정의 (train_specialist_m_v4.py와 동일)
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD  = [0.229, 0.224, 0.225]

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
            q  = torch.tensor([4.0, 12.0, 2.0]).view(3, 1, 1, 1)
            k  = torch.tensor([f1, f2, f3], dtype=torch.float32).unsqueeze(1) / q
            self.register_buffer("kernels", k)

        def forward(self, x):
            gray = x.mean(dim=1, keepdim=True)
            res  = F_torch.conv2d(gray, self.kernels, padding=2)
            return ((res.clamp(-2.0, 2.0) + 2.0) / 4.0)

    class DCTResidualExtractor(nn.Module):
        def forward(self, x):
            gray = x.mean(dim=1, keepdim=True)
            lp   = F_torch.avg_pool2d(gray, kernel_size=8, stride=8)
            lp   = F_torch.interpolate(lp, size=gray.shape[-2:], mode='nearest')
            res  = (gray - lp).clamp(-1.0, 1.0)
            return (res + 1.0) / 2.0

    class MobileNetBranch(nn.Module):
        def __init__(self, in_channels=3, pretrained=False):
            super().__init__()
            base = models.mobilenet_v2(weights=None)
            if in_channels != 3:
                old = base.features[0][0]
                base.features[0][0] = nn.Conv2d(
                    in_channels, old.out_channels,
                    kernel_size=old.kernel_size, stride=old.stride,
                    padding=old.padding, bias=False)
            self.features = base.features
            self.pool     = nn.AdaptiveAvgPool2d(1)
            self.out_dim  = 1280

        def forward(self, x):
            return self.pool(self.features(x)).flatten(1)

    class SpecialistM(nn.Module):
        def __init__(self, pretrained=False, dropout=0.3):
            super().__init__()
            self.srm_extractor = SRMExtractor()
            self.dct_extractor = DCTResidualExtractor()
            self.rgb_branch    = MobileNetBranch(in_channels=3, pretrained=pretrained)
            self.srm_branch    = MobileNetBranch(in_channels=3, pretrained=False)
            self.dct_branch    = MobileNetBranch(in_channels=1, pretrained=False)
            fused = self.rgb_branch.out_dim + self.srm_branch.out_dim + self.dct_branch.out_dim
            self.head = nn.Sequential(
                nn.LayerNorm(fused),
                nn.Linear(fused, 256),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(256, 2),
            )

        def forward(self, x):
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

    # 체크포인트 로드
    ckpt_path = ROOT / "weights" / "specialist_m_v4" / "specialist_m_v4_best.pth"
    model = SpecialistM(pretrained=False).to(gpu)
    ckpt  = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"  체크포인트 로드: {ckpt_path.name}")

    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    # 기존 SpecM JSONL에서 이미 추론된 키 수집
    full_outputs = {}
    for ds in DATASETS:
        specm_path = SPECM_FILES[ds]
        existing = {}
        if specm_path and specm_path.exists():
            with open(specm_path) as f:
                for line in f:
                    r = json.loads(line.strip())
                    existing[r["image_path"]] = r

        # 원본 JSONL에서 전체 이미지 목록 가져오기
        eval_jsonl = ROOT / EVAL_DATASETS[ds]
        all_records = []
        with open(eval_jsonl) as f:
            for line in f:
                r = json.loads(line.strip())
                all_records.append(r)

        # 누락된 이미지 (주로 ai_generated) 추론
        missing = [r for r in all_records if r["image_path"] not in existing]
        print(f"  [{ds}] 기존={len(existing)}, 누락={len(missing)}, 전체={len(all_records)}")

        if missing:
            for r in missing:
                img_path = ROOT / r["image_path"]
                try:
                    img = Image.open(img_path).convert("RGB")
                    tensor = val_transform(img).unsqueeze(0).to(gpu)
                    with torch.no_grad():
                        logits = model(tensor)
                        probs  = F_torch.softmax(logits, dim=-1).cpu().numpy()[0]
                    existing[r["image_path"]] = {
                        "image_path":      r["image_path"],
                        "true_label":      r["true_label"],
                        "pred_label":      "manipulated" if probs[1] > probs[0] else "authentic",
                        "manip_score":     float(probs[1]),
                        "authentic_score": float(probs[0]),
                        "confidence":      float(max(probs)),
                    }
                except Exception as e:
                    print(f"    오류: {r['image_path']} → {e}")

        # 전체 결과를 새 JSONL로 저장
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = OUT_DIR / f"specm_v4_full_{ds}_{ts}.jsonl"
        with open(out_path, "w") as f:
            for r in all_records:
                if r["image_path"] in existing:
                    f.write(json.dumps(existing[r["image_path"]], ensure_ascii=False) + "\n")
        full_outputs[ds] = out_path
        print(f"    → {out_path.name} ({len(existing)}건)")

    return full_outputs


# ── Step 1: 메타 피처 추출 (13-dim) ─────────────────────────────────────── #

def entropy(probs: np.ndarray) -> float:
    """Shannon entropy (nats), safe for zeros"""
    p = probs[probs > 0]
    return float(-np.sum(p * np.log(p)))


def margin(probs: np.ndarray) -> float:
    """top1 - top2 probability gap"""
    sorted_p = np.sort(probs)[::-1]
    return float(sorted_p[0] - sorted_p[1]) if len(sorted_p) >= 2 else float(sorted_p[0])


def extract_meta_features(
    mnv2: dict,
    specm: dict,
    include_f12: bool = True,
) -> np.ndarray:
    """
    13-dim 메타 피처 추출 (MNV2 + SpecM-v4)

    From MNV2 (3-class softmax):
      f1: P(auth)
      f2: P(manip)
      f3: P(aigen)
      f4: entropy(MNV2)
      f5: margin(MNV2) — top1 - top2 gap

    From SpecM-v4 (2-class softmax):
      f6: P_spec(auth)
      f7: P_spec(manip)
      f8: entropy(SpecM)
      f9: |P_spec(auth) - 0.5| — 결정 경계 거리

    Cross-model meta-features:
      f10: P(manip_MNV2) - P(manip_SpecM)     — 조작 판정 불일치도
      f11: P(auth_MNV2) - P(auth_SpecM)       — 진본 판정 불일치도
      f12: P(aigen_MNV2) × (1 - P(manip_SpecM)) — implicit null-class signal
      f13: entropy(MNV2) × entropy(SpecM)     — 양쪽 불확실 지표
    """
    # MNV2 확률
    p_auth  = mnv2["scores"]["authentic"]
    p_manip = mnv2["scores"]["manipulated"]
    p_aigen = mnv2["scores"]["ai_generated"]
    mnv2_probs = np.array([p_auth, p_manip, p_aigen])

    # SpecM 확률
    sp_auth  = specm["authentic_score"]
    sp_manip = specm["manip_score"]
    specm_probs = np.array([sp_auth, sp_manip])

    # Individual features
    f1 = p_auth
    f2 = p_manip
    f3 = p_aigen
    f4 = entropy(mnv2_probs)
    f5 = margin(mnv2_probs)

    f6 = sp_auth
    f7 = sp_manip
    f8 = entropy(specm_probs)
    f9 = abs(sp_auth - 0.5)

    # Cross-model features
    f10 = p_manip - sp_manip
    f11 = p_auth - sp_auth
    f12_val = p_aigen * (1.0 - sp_manip)  # implicit null-class signal
    f13 = entropy(mnv2_probs) * entropy(specm_probs)

    if include_f12:
        return np.array([f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12_val, f13])
    else:
        return np.array([f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f13])


FEATURE_NAMES_FULL = [
    "mnv2_auth", "mnv2_manip", "mnv2_aigen",
    "mnv2_entropy", "mnv2_margin",
    "specm_auth", "specm_manip",
    "specm_entropy", "specm_boundary_dist",
    "cross_manip_diff", "cross_auth_diff",
    "cross_implicit_nullclass", "cross_uncertainty_product",
]

FEATURE_NAMES_NO_F12 = [n for n in FEATURE_NAMES_FULL if n != "cross_implicit_nullclass"]


def extract_meta_features_sigmoid(
    mnv2: dict,
    specm_sig: dict,
    include_f12: bool = True,
) -> np.ndarray:
    """
    14-dim 메타 피처 (Sigmoid SpecM 버전).

    Sigmoid SpecM의 핵심 추가 필드:
      specm_sig["residual"] = R = 1 - P(auth) - P(manip)  (진짜 잔여 질량)

    From MNV2 (3-class softmax):
      f1: P(auth), f2: P(manip), f3: P(aigen)
      f4: entropy(MNV2), f5: margin(MNV2)

    From SpecM-Sigmoid:
      f6: P_spec(auth), f7: P_spec(manip)
      f8: R = residual (implicit aigen mass) ← 핵심 신규 피처
      f9: entropy over [P(auth), P(manip), R]  ← 3-way entropy
      f10: |P_spec(auth) - 0.5| (결정 경계 거리)

    Cross-model:
      f11: P(manip_MNV2) - P(manip_SpecM)
      f12: P(aigen_MNV2) × R  ← 진짜 implicit null-class signal
      f13: P(auth_MNV2) - P(auth_SpecM)
      f14: entropy(MNV2) × R  ← MNV2 불확실 & SpecM 미할당 = 어려운 샘플
    """
    p_auth  = mnv2["scores"]["authentic"]
    p_manip = mnv2["scores"]["manipulated"]
    p_aigen = mnv2["scores"]["ai_generated"]
    mnv2_probs = np.array([p_auth, p_manip, p_aigen])

    sp_auth  = specm_sig["authentic_score"]
    sp_manip = specm_sig["manip_score"]
    R        = float(specm_sig.get("residual", max(0.0, 1.0 - sp_auth - sp_manip)))
    specm_3  = np.array([sp_auth, sp_manip, R])

    f1, f2, f3 = p_auth, p_manip, p_aigen
    f4 = entropy(mnv2_probs)
    f5 = margin(mnv2_probs)

    f6  = sp_auth
    f7  = sp_manip
    f8  = R                              # 진짜 잔여 질량
    f9  = entropy(specm_3 + 1e-9)       # 3-way entropy (auth, manip, residual)
    f10 = abs(sp_auth - 0.5)

    f11 = p_manip - sp_manip
    f12_val = p_aigen * R               # 진짜 cross-modal implicit signal
    f13 = p_auth - sp_auth
    f14 = entropy(mnv2_probs) * R       # MNV2 불확실 & SpecM 미할당

    if include_f12:
        return np.array([f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12_val, f13, f14])
    else:
        return np.array([f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f13, f14])


FEATURE_NAMES_SIGMOID_FULL = [
    "mnv2_auth", "mnv2_manip", "mnv2_aigen",
    "mnv2_entropy", "mnv2_margin",
    "specm_auth", "specm_manip",
    "specm_residual",           # ← 핵심 신규 (진짜 잔여 질량)
    "specm_3way_entropy",
    "specm_boundary_dist",
    "cross_manip_diff",
    "cross_aigen_times_residual",  # ← 진짜 f12
    "cross_auth_diff",
    "cross_mnv2entropy_times_residual",
]

FEATURE_NAMES_SIGMOID_NO_F12 = [n for n in FEATURE_NAMES_SIGMOID_FULL
                                  if n != "cross_aigen_times_residual"]


# ── Step 2: 데이터 구성 ─────────────────────────────────────────────────── #

def load_jsonl(path: Path) -> Dict[str, dict]:
    records = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line.strip())
            records[r["image_path"]] = r
    return records


def build_dataset(
    mnv2_path: Path,
    specm_path: Path,
    include_f12: bool = True,
    sigmoid_mode: bool = False,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """MNV2 + SpecM JSONL → (X, y, image_paths)"""
    mnv2_recs  = load_jsonl(mnv2_path)
    specm_recs = load_jsonl(specm_path)

    common = sorted(set(mnv2_recs) & set(specm_recs))
    X_list, y_list, paths = [], [], []

    for key in common:
        mnv2  = mnv2_recs[key]
        specm = specm_recs[key]
        true_label = mnv2["true_label"]
        if true_label not in CLS2IDX:
            continue

        if sigmoid_mode:
            feat = extract_meta_features_sigmoid(mnv2, specm, include_f12=include_f12)
        else:
            feat = extract_meta_features(mnv2, specm, include_f12=include_f12)
        X_list.append(feat)
        y_list.append(CLS2IDX[true_label])
        paths.append(key)

    return np.array(X_list), np.array(y_list), paths


# ── Step 3: ICWMV 2-model baseline (MNV2 + SpecM only) ──────────────────── #

def icwmv_2model_predict(
    mnv2: dict,
    specm: dict,
    w_spec: float = 1.0,
) -> Tuple[str, np.ndarray, float]:
    """
    RPi5 2-model ICWMV (MNV2 + SpecM-v4):
      S(auth)  = (MNV2_auth + w_spec * SpecM_auth) / (1 + w_spec)
      S(manip) = (MNV2_manip + w_spec * SpecM_manip) / (1 + w_spec)
      S(aigen) = MNV2_aigen / 1  (SpecM 기여 없음)
      → renormalize → argmax
    """
    s_auth  = (mnv2["scores"]["authentic"]  + w_spec * specm["authentic_score"]) / (1 + w_spec)
    s_manip = (mnv2["scores"]["manipulated"] + w_spec * specm["manip_score"])     / (1 + w_spec)
    s_aigen = mnv2["scores"]["ai_generated"]  # SpecM은 aigen에 기여 안 함

    raw   = np.array([s_auth, s_manip, s_aigen])
    probs = raw / raw.sum()
    conf  = float(probs.max())
    pred  = CLASSES[int(np.argmax(probs))]
    return pred, probs, conf


# ── Step 4: 평가 함수 ───────────────────────────────────────────────────── #

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Macro-F1, accuracy, per-class metrics"""
    per_class = {}
    for idx, cls in enumerate(CLASSES):
        tp = int(((y_pred == idx) & (y_true == idx)).sum())
        fp = int(((y_pred == idx) & (y_true != idx)).sum())
        fn = int(((y_pred != idx) & (y_true == idx)).sum())
        prec = tp / max(tp + fp, 1)
        rec  = tp / max(tp + fn, 1)
        f1   = 2 * prec * rec / max(prec + rec, 1e-8)
        per_class[cls] = {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn}

    classes_present = [c for c in CLASSES if (y_true == CLS2IDX[c]).any()]
    macro_f1  = float(np.mean([per_class[c]["f1"] for c in classes_present]))
    accuracy  = float((y_true == y_pred).mean())

    return {"accuracy": accuracy, "macro_f1": macro_f1, "per_class": per_class}


# ── Step 5: 메인 실험 ───────────────────────────────────────────────────── #

def run_experiment(
    specm_full_files: Dict[str, Path],
    w_spec: float = 1.0,
    sigmoid_mode: bool = False,
    experiment_label: str = "PoC (Softmax proxy)",
):
    from sklearn.model_selection import StratifiedKFold
    try:
        from xgboost import XGBClassifier
        print("[Meta-classifier] XGBoost 사용")
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier
        XGBClassifier = None
        print("[Meta-classifier] sklearn GradientBoosting 사용 (xgboost 미설치)")

    feat_names = FEATURE_NAMES_SIGMOID_FULL if sigmoid_mode else FEATURE_NAMES_FULL
    n_feats    = len(feat_names)
    specm_desc = "SpecM-Sigmoid (BCE + null-class)" if sigmoid_mode else "SpecM-v4 (Softmax proxy)"

    print(f"\n{'='*70}")
    print(f"HEMA 실험: {experiment_label}")
    print(f"{'='*70}")
    print(f"  모델: MNV2 (3-class) + {specm_desc}")
    print(f"  메타 피처: {n_feats}-dim")
    print(f"  ICWMV w_spec: {w_spec}")

    all_results = {}

    for ds in DATASETS:
        mnv2_path  = MNV2_FILES[ds]
        specm_path = specm_full_files[ds]

        if not mnv2_path.exists() or not specm_path.exists():
            print(f"\n  [{ds}] 파일 없음, 스킵")
            continue

        # 데이터 빌드
        X_full, y, paths = build_dataset(mnv2_path, specm_path, include_f12=True,  sigmoid_mode=sigmoid_mode)
        X_no12, _, _     = build_dataset(mnv2_path, specm_path, include_f12=False, sigmoid_mode=sigmoid_mode)

        n_samples = len(y)
        label_dist = {CLASSES[i]: int((y == i).sum()) for i in range(3)}
        print(f"\n{'─'*70}")
        print(f"[{ds}] n={n_samples}, 분포={label_dist}")

        # ── Baseline: MNV2 단독 ──
        mnv2_recs  = load_jsonl(mnv2_path)
        specm_recs = load_jsonl(specm_path)
        common     = sorted(set(mnv2_recs) & set(specm_recs))
        common     = [k for k in common if mnv2_recs[k]["true_label"] in CLS2IDX]

        mnv2_preds = np.array([CLS2IDX[mnv2_recs[k]["pred_label"]] for k in common])
        mnv2_true  = np.array([CLS2IDX[mnv2_recs[k]["true_label"]]  for k in common])
        mnv2_metrics = compute_metrics(mnv2_true, mnv2_preds)

        # ── Baseline: ICWMV 2-model ──
        icwmv_preds = []
        icwmv_confs = []
        for k in common:
            pred, probs, conf = icwmv_2model_predict(mnv2_recs[k], specm_recs[k], w_spec)
            icwmv_preds.append(CLS2IDX[pred])
            icwmv_confs.append(conf)
        icwmv_preds = np.array(icwmv_preds)
        icwmv_confs = np.array(icwmv_confs)
        icwmv_metrics = compute_metrics(y, icwmv_preds)

        # ── HEMA: Stratified 5-Fold CV ──
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        hema_preds_full = np.zeros(n_samples, dtype=int)
        hema_preds_no12 = np.zeros(n_samples, dtype=int)
        hema_probs_full = np.zeros((n_samples, 3))
        fold_importances = []

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_full, y)):
            X_tr, X_te = X_full[train_idx], X_full[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            if XGBClassifier is not None:
                clf = XGBClassifier(
                    max_depth=3,
                    n_estimators=50,
                    learning_rate=0.1,
                    objective="multi:softprob",
                    num_class=3,
                    random_state=42,
                    verbosity=0,
                    use_label_encoder=False,
                )
            else:
                clf = GradientBoostingClassifier(
                    max_depth=3,
                    n_estimators=50,
                    learning_rate=0.1,
                    random_state=42,
                )

            clf.fit(X_tr, y_tr)
            hema_preds_full[test_idx] = clf.predict(X_te)
            if hasattr(clf, 'predict_proba'):
                hema_probs_full[test_idx] = clf.predict_proba(X_te)

            # Feature importance
            if hasattr(clf, 'feature_importances_'):
                fold_importances.append(clf.feature_importances_)

            # Ablation: no f12
            X_tr_no12 = X_no12[train_idx]
            X_te_no12 = X_no12[test_idx]
            if XGBClassifier is not None:
                clf_no12 = XGBClassifier(
                    max_depth=3, n_estimators=50, learning_rate=0.1,
                    objective="multi:softprob", num_class=3,
                    random_state=42, verbosity=0, use_label_encoder=False,
                )
            else:
                clf_no12 = GradientBoostingClassifier(
                    max_depth=3, n_estimators=50, learning_rate=0.1, random_state=42,
                )
            clf_no12.fit(X_tr_no12, y_tr)
            hema_preds_no12[test_idx] = clf_no12.predict(X_te_no12)

        hema_full_metrics = compute_metrics(y, hema_preds_full)
        hema_no12_metrics = compute_metrics(y, hema_preds_no12)

        # ── Cascade: ICWMV(high-conf) → HEMA(low-conf) ──
        cascade_results = {}
        for tau in [0.5, 0.6, 0.7, 0.8, 0.9]:
            cascade_preds = np.copy(icwmv_preds)
            low_conf_mask = icwmv_confs < tau
            n_low = int(low_conf_mask.sum())
            cascade_preds[low_conf_mask] = hema_preds_full[low_conf_mask]
            cascade_metrics = compute_metrics(y, cascade_preds)
            cascade_results[str(tau)] = {
                "tau": tau,
                "n_low_conf": n_low,
                "pct_low_conf": round(100 * n_low / n_samples, 1),
                **cascade_metrics,
            }

        # ── Feature importance 평균 ──
        avg_importance = {}
        if fold_importances:
            mean_imp = np.mean(fold_importances, axis=0)
            for name, imp in sorted(zip(feat_names, mean_imp), key=lambda x: -x[1]):
                avg_importance[name] = round(float(imp), 4)

        # ── 출력 ──
        print(f"\n  {'Method':<30} {'Macro-F1':>10} {'Accuracy':>10} {'Auth-F1':>10} {'Manip-F1':>10} {'Aigen-F1':>10}")
        print(f"  {'-'*80}")

        for name, m in [
            ("MNV2 (단독)", mnv2_metrics),
            ("ICWMV 2-model", icwmv_metrics),
            ("HEMA 13-dim", hema_full_metrics),
            ("HEMA no-f12 (ablation)", hema_no12_metrics),
        ]:
            auth_f1  = m["per_class"]["authentic"]["f1"]
            manip_f1 = m["per_class"]["manipulated"]["f1"]
            aigen_f1 = m["per_class"]["ai_generated"]["f1"]
            print(f"  {name:<30} {100*m['macro_f1']:>9.2f}% {100*m['accuracy']:>9.2f}% "
                  f"{100*auth_f1:>9.2f}% {100*manip_f1:>9.2f}% {100*aigen_f1:>9.2f}%")

        # Best cascade
        best_tau = max(cascade_results.values(), key=lambda x: x["macro_f1"])
        print(f"\n  Best Cascade (τ={best_tau['tau']}): "
              f"F1={100*best_tau['macro_f1']:.2f}%, "
              f"low-conf={best_tau['pct_low_conf']}% routed to HEMA")

        # Feature importance
        if avg_importance:
            print(f"\n  Feature Importance (Top-5):")
            for i, (name, imp) in enumerate(list(avg_importance.items())[:5]):
                print(f"    {i+1}. {name}: {imp:.4f}")

        # f12 ablation delta
        delta_f12 = hema_full_metrics["macro_f1"] - hema_no12_metrics["macro_f1"]
        print(f"\n  f12 (implicit null-class) ablation: "
              f"Δmacro-F1 = {100*delta_f12:+.2f}%p")

        all_results[ds] = {
            "n_samples": n_samples,
            "label_dist": label_dist,
            "mnv2_solo": mnv2_metrics,
            "icwmv_2model": icwmv_metrics,
            "hema_full": hema_full_metrics,
            "hema_no_f12": hema_no12_metrics,
            "cascade": cascade_results,
            "feature_importance": avg_importance,
            "f12_ablation_delta": round(delta_f12, 6),
        }

    # ── 전체 요약 ──
    print(f"\n{'='*70}")
    print("전체 요약")
    print(f"{'='*70}")
    print(f"\n  {'Dataset':<14} {'MNV2':>8} {'ICWMV':>8} {'HEMA':>8} {'HEMA-f12':>10} {'Δ(HEMA-ICWMV)':>14} {'Δ(f12 ablation)':>16}")
    print(f"  {'-'*80}")
    for ds, r in all_results.items():
        mnv2_f1  = r["mnv2_solo"]["macro_f1"]
        icwmv_f1 = r["icwmv_2model"]["macro_f1"]
        hema_f1  = r["hema_full"]["macro_f1"]
        no12_f1  = r["hema_no_f12"]["macro_f1"]
        delta    = hema_f1 - icwmv_f1
        delta12  = r["f12_ablation_delta"]
        print(f"  {ds:<14} {100*mnv2_f1:>7.2f}% {100*icwmv_f1:>7.2f}% {100*hema_f1:>7.2f}% "
              f"{100*no12_f1:>9.2f}% {100*delta:>+13.2f}%p {100*delta12:>+15.2f}%p")

    # 평균
    if all_results:
        avg_mnv2  = np.mean([r["mnv2_solo"]["macro_f1"] for r in all_results.values()])
        avg_icwmv = np.mean([r["icwmv_2model"]["macro_f1"] for r in all_results.values()])
        avg_hema  = np.mean([r["hema_full"]["macro_f1"] for r in all_results.values()])
        avg_no12  = np.mean([r["hema_no_f12"]["macro_f1"] for r in all_results.values()])
        avg_d     = avg_hema - avg_icwmv
        avg_d12   = avg_hema - avg_no12
        print(f"  {'AVERAGE':<14} {100*avg_mnv2:>7.2f}% {100*avg_icwmv:>7.2f}% {100*avg_hema:>7.2f}% "
              f"{100*avg_no12:>9.2f}% {100*avg_d:>+13.2f}%p {100*avg_d12:>+15.2f}%p")

    # 결과 저장
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUT_DIR / f"hema_poc_results_{ts}.json"
    with open(out_path, "w") as f:
        json.dump({
            "description": "HEMA PoC: MNV2 + SpecM-v4 meta-classifier",
            "config": {"w_spec": w_spec, "meta_features": 13, "folds": 5},
            "results": all_results,
        }, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n결과 저장: {out_path}")

    return all_results


# ── Main ─────────────────────────────────────────────────────────────────── #

def main():
    parser = argparse.ArgumentParser(description="HEMA Meta-Classifier 실험")
    parser.add_argument("--w-spec", type=float, default=1.0,
                        help="ICWMV specialist 가중치 (default: 1.0)")
    parser.add_argument("--skip-inference", action="store_true",
                        help="SpecM 추론 건너뛰기 (기존 full JSONL 사용)")
    parser.add_argument("--sigmoid", action="store_true",
                        help="Sigmoid SpecM JSONL 사용 (specm_sigmoid_*.jsonl)")
    parser.add_argument("--compare", action="store_true",
                        help="Softmax proxy와 Sigmoid 결과 나란히 비교")
    args = parser.parse_args()

    if args.sigmoid or args.compare:
        # Sigmoid 모드: specm_sigmoid_*.jsonl 탐색
        sigmoid_files = {}
        for ds in DATASETS:
            candidates = sorted(OUT_DIR.glob(f"specm_sigmoid_{ds}_*.jsonl"))
            if candidates:
                sigmoid_files[ds] = candidates[-1]
                print(f"[Sigmoid/{ds}] {candidates[-1].name}")
            else:
                print(f"[Sigmoid/{ds}] JSONL 없음 — train_specialist_m_sigmoid.py 먼저 실행 필요")
                sigmoid_files = {}
                break

    if not args.sigmoid:
        # Softmax proxy (기본) 경로
        if args.skip_inference:
            specm_files = {}
            for ds in DATASETS:
                candidates = sorted(OUT_DIR.glob(f"specm_v4_full_{ds}_*.jsonl"))
                if candidates:
                    specm_files[ds] = candidates[-1]
                else:
                    args.skip_inference = False
                    break

        if not args.skip_inference:
            specm_files = run_specm_full_inference()

        poc_results = run_experiment(
            specm_files, w_spec=args.w_spec,
            sigmoid_mode=False,
            experiment_label="Path A — Softmax Proxy (PoC)",
        )

    if args.sigmoid and sigmoid_files:
        run_experiment(
            sigmoid_files, w_spec=args.w_spec,
            sigmoid_mode=True,
            experiment_label="Path B — Sigmoid + Explicit Null-Class (True Residual)",
        )

    if args.compare and sigmoid_files:
        # Softmax proxy도 같이 실행해서 비교
        if args.skip_inference or OUT_DIR.glob("specm_v4_full_base_*.jsonl"):
            softmax_files = {}
            for ds in DATASETS:
                candidates = sorted(OUT_DIR.glob(f"specm_v4_full_{ds}_*.jsonl"))
                if candidates:
                    softmax_files[ds] = candidates[-1]
            if len(softmax_files) == len(DATASETS):
                print("\n" + "="*70)
                print("비교: Softmax Proxy vs Sigmoid Residual")
                print("="*70)
                poc_r = run_experiment(
                    softmax_files, w_spec=args.w_spec,
                    sigmoid_mode=False,
                    experiment_label="Softmax Proxy",
                )
                sig_r = run_experiment(
                    sigmoid_files, w_spec=args.w_spec,
                    sigmoid_mode=True,
                    experiment_label="Sigmoid Residual",
                )

                print(f"\n{'='*70}")
                print("Delta: Sigmoid - Softmax (HEMA macro-F1)")
                print(f"{'='*70}")
                for ds in DATASETS:
                    if ds in poc_r and ds in sig_r:
                        d = sig_r[ds]["hema_full"]["macro_f1"] - poc_r[ds]["hema_full"]["macro_f1"]
                        print(f"  {ds:<14}: {100*d:+.2f}%p")


if __name__ == "__main__":
    main()
