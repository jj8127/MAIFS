#!/usr/bin/env python3
"""
Fair Leave-One-Out Cross-Dataset 평가
======================================
MobileCLIP-ft4도 각 fold별로 3개 데이터셋만 사용해 재파인튜닝,
held-out 1개로 평가 → Track GBM과 공정한 OOD 비교.

평가 구조 (held-out 데이터셋 X):
  - MobileCLIP-LOO: {base,dsC,opensdi,aigenproxy}\X 로 파인튜닝 → X에서 평가
  - Track2-GBM:     동일 3개 데이터셋의 ForMa+CLIP 특징으로 학습 → X에서 평가
  - Track3-GBM:     동일 3개 데이터셋의 Tiny+ForMa+CLIP 특징으로 학습 → X에서 평가
  - Track1-GBM:     동일 3개 데이터셋의 ForMa+MNV2+CLIP 특징으로 학습 → X에서 평가
  (Track GBM 특징의 MobileCLIP 부분은 full-model 예측 사용 — GBM에 소폭 유리)

실행:
  .venv-qwen/bin/python experiments/run_fair_cross.py
  .venv-qwen/bin/python experiments/run_fair_cross.py --epochs 15  # 빠르게
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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "experiments"))

from finetune_mobileclip import (
    ForensicsDataset, ForensicsHead, MobileCLIPForensics,
    LABEL_MAP, IDX2LABEL, TRAIN_TRANSFORM, VAL_TRANSFORM,
    load_records, DATASETS, evaluate
)
from run_phase3_tracks import (
    load_dataset, extract_track1, extract_track2, extract_track3,
    latest_backbone_jsonl, load_backbone, BACKBONE_PREFIX
)

OUT_DIR = ROOT / "experiments" / "results" / "fair_cross"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]


# ── MobileCLIP LOO 재파인튜닝 ─────────────────────────────────────────────── #
def retrain_mobileclip(train_recs: List[Dict], epochs: int,
                       lr: float = 2e-5, batch: int = 32) -> MobileCLIPForensics:
    """3개 데이터셋 레코드로 MobileCLIP last-4-block 재파인튜닝."""
    import open_clip
    clip_model, _, _ = open_clip.create_model_and_transforms(
        "MobileCLIP-S2", pretrained="datacompdr",
        cache_dir=str(ROOT / "weights" / "mobileclip")
    )
    model = MobileCLIPForensics(clip_model, finetune_blocks=4).to(DEVICE)

    train_ds = ForensicsDataset(train_recs, TRAIN_TRANSFORM)
    loader   = DataLoader(train_ds, batch_size=batch, shuffle=True,
                          num_workers=4, pin_memory=True)
    optimizer = AdamW(model.trainable_params(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    best_state, best_loss = None, float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, n = 0.0, 0
        for imgs, labels, _ in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(imgs), labels, label_smoothing=0.05)
            loss.backward()
            nn.utils.clip_grad_norm_(model.trainable_params(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(labels)
            n += len(labels)
        scheduler.step()
        avg_loss = total_loss / n
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        print(f"      ep{epoch:2d}/{epochs} loss={avg_loss:.4f}", flush=True)

    model.load_state_dict(best_state)
    model.eval()
    return model


@torch.no_grad()
def clip_predict_records(model: MobileCLIPForensics,
                         recs: List[Dict]) -> List[Dict]:
    """LOO-retrained MobileCLIP으로 레코드 예측. pred_label, confidence, scores 추가."""
    ds     = ForensicsDataset(recs, VAL_TRANSFORM)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4)
    results = []
    for imgs, labels, paths in loader:
        imgs   = imgs.to(DEVICE)
        logits = model(imgs)
        probs  = F.softmax(logits, dim=-1).cpu()
        preds  = probs.argmax(1)
        for i in range(len(labels)):
            results.append({
                "image_path": paths[i],
                "true_label": IDX2LABEL[labels[i].item()],
                "pred_label": IDX2LABEL[preds[i].item()],
                "confidence": probs[i][preds[i]].item(),
                "scores": {IDX2LABEL[j]: probs[i][j].item() for j in range(3)},
            })
    return results


# ── Track 특징 구성 (LOO-clip 예측 반영) ─────────────────────────────────── #
def build_track_feats(ds: str,
                      loo_clip_map: Dict[str, Dict]) -> Tuple[np.ndarray, ...]:
    """
    forma / mobilenet / tiny는 기존 full-model JSONL,
    clip은 LOO-retrained 예측으로 교체하여 특징 추출.
    반환: (feat_t1, feat_t2, feat_t3, labels)
    """
    forma_map   = load_backbone(latest_backbone_jsonl("forma", ds))
    mnv2_map    = load_backbone(latest_backbone_jsonl("mobilenetv2_dualstream", ds))
    tiny_map    = load_backbone(latest_backbone_jsonl("tiny_ladeda", ds))

    common = set(forma_map) & set(mnv2_map) & set(tiny_map) & set(loo_clip_map)
    t1_list, t2_list, t3_list, y_list = [], [], [], []

    for img in sorted(common):
        f_rec = forma_map[img]
        m_rec = mnv2_map[img]
        t_rec = tiny_map[img]
        c_rec = loo_clip_map[img]
        lbl   = f_rec.get("true_label", "authentic")
        if lbl not in LABEL_MAP:
            continue
        t1_list.append(extract_track1(f_rec, m_rec, c_rec))
        t2_list.append(extract_track2(f_rec, c_rec))
        t3_list.append(extract_track3(f_rec, c_rec, t_rec))
        y_list.append(LABEL_MAP[lbl])

    return (np.array(t1_list, dtype=np.float32),
            np.array(t2_list, dtype=np.float32),
            np.array(t3_list, dtype=np.float32),
            np.array(y_list,  dtype=np.int64))


def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


# ── 메인 LOO 루프 ────────────────────────────────────────────────────────── #
def run(epochs: int = 20):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results = {"epochs": epochs, "timestamp": ts, "folds": {}}

    ds_keys = list(DATASETS.keys())

    for held in ds_keys:
        train_keys = [d for d in ds_keys if d != held]
        print(f"\n{'='*60}")
        print(f"  Held-out: {held}  |  Train: {train_keys}")
        print(f"{'='*60}")

        # ── 학습 레코드 수집 ──
        train_recs = []
        for ds in train_keys:
            r = load_records(DATASETS[ds]["jsonl"])
            train_recs.extend(r)
        test_recs = load_records(DATASETS[held]["jsonl"])
        print(f"  Train {len(train_recs)}개, Test {len(test_recs)}개")

        # ── MobileCLIP LOO 재파인튜닝 ──
        print(f"\n  [MobileCLIP LOO 파인튜닝 — {epochs} epochs]")
        loo_model = retrain_mobileclip(train_recs, epochs=epochs)

        # ── LOO 예측 (train + test 모두) ──
        print(f"  [LOO 예측]")
        train_clip_preds = clip_predict_records(loo_model, train_recs)
        test_clip_preds  = clip_predict_records(loo_model, test_recs)

        # 단독 성능
        loo_test_labels = [LABEL_MAP[r["true_label"]] for r in test_clip_preds]
        loo_test_preds  = [LABEL_MAP[r["pred_label"]] for r in test_clip_preds]
        clip_loo_f1     = macro_f1(loo_test_labels, loo_test_preds)

        # ── Track 특징 구성 (LOO-clip 적용) ──
        print(f"  [Track 특징 구성]")
        # train: 3개 데이터셋 합쳐서 clip을 LOO 예측으로 교체
        train_clip_map = {r["image_path"]: r for r in train_clip_preds}
        test_clip_map  = {r["image_path"]: r for r in test_clip_preds}

        t1_train_list, t2_train_list, t3_train_list, y_train_list = [], [], [], []
        for ds in train_keys:
            ft1, ft2, ft3, yl = build_track_feats(ds, train_clip_map)
            t1_train_list.append(ft1); t2_train_list.append(ft2)
            t3_train_list.append(ft3); y_train_list.append(yl)

        ft1_train = np.concatenate(t1_train_list)
        ft2_train = np.concatenate(t2_train_list)
        ft3_train = np.concatenate(t3_train_list)
        y_train   = np.concatenate(y_train_list)

        ft1_test, ft2_test, ft3_test, y_test = build_track_feats(held, test_clip_map)

        # ── GBM 학습/평가 ──
        print(f"  [Track GBM 학습/평가]")
        gbm_t1 = GradientBoostingClassifier(n_estimators=250, max_depth=4,
                                            learning_rate=0.05, random_state=42)
        gbm_t1.fit(ft1_train, y_train)
        t1_f1 = macro_f1(y_test, gbm_t1.predict(ft1_test))

        gbm_t2 = GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                            learning_rate=0.05, random_state=42)
        gbm_t2.fit(ft2_train, y_train)
        t2_f1 = macro_f1(y_test, gbm_t2.predict(ft2_test))

        gbm_t3 = GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                            learning_rate=0.05, random_state=42)
        gbm_t3.fit(ft3_train, y_train)
        t3_f1 = macro_f1(y_test, gbm_t3.predict(ft3_test))

        fold_res = {
            "clip_loo":   round(clip_loo_f1, 4),
            "track1_gbm": round(t1_f1, 4),
            "track2_gbm": round(t2_f1, 4),
            "track3_gbm": round(t3_f1, 4),
        }
        all_results["folds"][held] = fold_res

        print(f"\n  결과 [{held}]:")
        best = max(fold_res.values())
        for k, v in fold_res.items():
            star = " ★" if v == best else ""
            print(f"    {k:15s}: {v:.4f}{star}")

    # ── 전체 요약 ──
    print(f"\n\n{'='*60}")
    print(f"  Fair LOO 최종 비교 ({epochs} epoch 재파인튜닝)")
    print(f"{'='*60}")
    keys = ["clip_loo", "track1_gbm", "track2_gbm", "track3_gbm"]
    for k in keys:
        vals = [all_results["folds"][d][k] for d in ds_keys]
        print(f"  {k:15s}: " + "  ".join(f"{v:.4f}" for v in vals) +
              f"  → avg {np.mean(vals):.4f}")

    print(f"\n  [vs clip_loo 델타]")
    clip_vals = [all_results["folds"][d]["clip_loo"] for d in ds_keys]
    for k in ["track1_gbm", "track2_gbm", "track3_gbm"]:
        vals = [all_results["folds"][d][k] for d in ds_keys]
        deltas = [v - c for v, c in zip(vals, clip_vals)]
        print(f"  {k:15s}: " + "  ".join(f"{d:+.4f}" for d in deltas) +
              f"  → avg Δ {np.mean(deltas):+.4f}")

    out = OUT_DIR / f"fair_cross_{ts}.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n  저장: {out}")
    return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    run(epochs=args.epochs)
