#!/usr/bin/env python3
"""
Specialist-M v4 Coral fine-tuning
=================================

목적:
  - Edge TPU friendly `specm_v4_coral` 구조를 직접 학습해
    export-only 치환에서 생긴 정확도 손실을 줄인다.

전략:
  1. v4 best checkpoint에서 branch / linear weight를 최대한 재사용
  2. warmup 단계에서는 head만 학습해 LayerNorm+GELU -> ReLU head 변화를 먼저 적응
  3. 이후 전체 모델을 low-lr backbone + high-lr head로 미세조정

실행:
  .venv-qwen/bin/python experiments/train_specialist_m_v4_coral.py
  .venv-qwen/bin/python experiments/train_specialist_m_v4_coral.py --epochs 12 --batch-size 96
  .venv-qwen/bin/python experiments/train_specialist_m_v4_coral.py --eval-only weights/specialist_m_v4_coral/specialist_m_v4_coral_best.pth
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from coral_export_models import CoralSpecialistMv4
from train_specialist_m_v4 import (
    DEVICE,
    ROOT,
    BinaryDataset,
    EVAL_DATASETS,
    FocalLoss,
    TRAIN_TRANSFORM,
    VAL_TRANSFORM,
    evaluate_binary,
    load_jsonl_records,
    make_sampler,
    scan_training_data,
    split_records,
)

V4_BEST_CKPT = ROOT / "weights" / "specialist_m_v4" / "specialist_m_v4_best.pth"


def load_v4_into_coral(model: CoralSpecialistMv4, ckpt_path: Path) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    allowed_missing = set()
    allowed_unexpected = {"head.0.weight", "head.0.bias"}
    bad_missing = sorted(set(missing) - allowed_missing)
    bad_unexpected = sorted(set(unexpected) - allowed_unexpected)
    if bad_missing or bad_unexpected:
        raise RuntimeError(
            f"Unexpected partial load mismatch | missing={bad_missing} unexpected={bad_unexpected}"
        )
    return ckpt


def set_requires_grad(model: CoralSpecialistMv4, head_only: bool) -> None:
    for module in (
        model.srm_extractor,
        model.dct_extractor,
        model.rgb_branch,
        model.srm_branch,
        model.dct_branch,
    ):
        for param in module.parameters():
            param.requires_grad_(not head_only)
    for param in model.head.parameters():
        param.requires_grad_(True)


def make_loader(records, batch_size: int, train: bool):
    return DataLoader(
        BinaryDataset(records, TRAIN_TRANSFORM if train else VAL_TRANSFORM),
        batch_size=batch_size,
        shuffle=False if train else False,
        sampler=make_sampler(records) if train else None,
        num_workers=4,
        pin_memory=(DEVICE.type == "cuda"),
    )


def train_epoch(model, loader, optimizer, scaler, criterion) -> tuple[float, float]:
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    use_amp = DEVICE.type == "cuda"
    for imgs, labels, _ in loader:
        imgs = imgs.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", enabled=use_amp):
            logits = model(imgs)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * len(labels)
        correct += (logits.argmax(1) == labels).sum().item()
        n += len(labels)
    return total_loss / max(n, 1), correct / max(n, 1)


def build_optimizer(model: CoralSpecialistMv4, lr_backbone: float, lr_head: float) -> AdamW:
    backbone_params = []
    for module in (model.rgb_branch, model.srm_branch, model.dct_branch):
        backbone_params.extend([p for p in module.parameters() if p.requires_grad])
    head_params = [p for p in model.head.parameters() if p.requires_grad]
    return AdamW(
        [
            {"params": backbone_params, "lr": lr_backbone},
            {"params": head_params, "lr": lr_head},
        ],
        weight_decay=1e-4,
    )


def run_training(args) -> Path:
    print("=" * 60)
    print("Specialist-M v4 Coral fine-tune")
    print(
        f"  epochs={args.epochs}, batch={args.batch_size}, "
        f"warmup={args.warmup_head_epochs}, lr_backbone={args.lr_backbone}, "
        f"lr_head={args.lr_head}, device={DEVICE}"
    )
    print("=" * 60)

    all_records = scan_training_data(genimage_auth_max=args.genimage_max)
    train_records, val_records = split_records(all_records, val_ratio=0.15, seed=42)
    print(f"  Train={len(train_records)}  Val={len(val_records)}")

    train_loader = make_loader(train_records, args.batch_size, train=True)
    val_loader = make_loader(val_records, args.batch_size * 2, train=False)

    model = CoralSpecialistMv4(pretrained=False).to(DEVICE)
    ckpt_meta = load_v4_into_coral(model, Path(args.resume))
    print(
        f"  Resume={Path(args.resume).name} "
        f"(v4 manip_f1={ckpt_meta.get('best_manip_f1', 0.0):.4f})"
    )

    criterion = FocalLoss(gamma=args.focal_gamma, alpha=args.focal_alpha)
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda"))

    save_dir = ROOT / "weights" / "specialist_m_v4_coral"
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / "specialist_m_v4_coral_best.pth"

    best_f1 = 0.0
    history: list[dict] = []

    # Stage 1: head-only warmup
    if args.warmup_head_epochs > 0:
        set_requires_grad(model, head_only=True)
        optimizer = AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr_head,
            weight_decay=1e-4,
        )
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=max(args.warmup_head_epochs, 1),
            eta_min=args.lr_head * 0.1,
        )
        print("\n[Stage 1] head-only warmup")
        for epoch in range(1, args.warmup_head_epochs + 1):
            tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, scaler, criterion)
            val_stats = evaluate_binary(model, val_loader)
            scheduler.step()
            f1_m = val_stats["manip_f1"]
            star = " ★" if f1_m > best_f1 else ""
            print(
                f"  Warmup {epoch:02d}/{args.warmup_head_epochs} | "
                f"loss={tr_loss:.4f} | tr_acc={tr_acc:.3f} | "
                f"val_acc={val_stats['accuracy']:.3f} | "
                f"auth_recall={val_stats['per_class']['authentic']['recall']:.3f} | "
                f"manip_recall={val_stats['manip_recall']:.3f} | "
                f"manip_f1={f1_m:.3f}{star}"
            )
            history.append(
                {
                    "stage": "warmup",
                    "epoch": epoch,
                    "train_loss": tr_loss,
                    "train_acc": tr_acc,
                    "val_manip_f1": f1_m,
                }
            )
            if f1_m > best_f1:
                best_f1 = f1_m
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "history": history,
                        "best_manip_f1": best_f1,
                        "epoch": epoch,
                        "version": "v4_coral",
                        "resumed_from": str(args.resume),
                    },
                    ckpt_path,
                )

    # Stage 2: full fine-tune
    full_epochs = max(args.epochs - args.warmup_head_epochs, 0)
    if full_epochs > 0:
        print("\n[Stage 2] full fine-tune")
        set_requires_grad(model, head_only=False)
        optimizer = build_optimizer(model, args.lr_backbone, args.lr_head)
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=full_epochs,
            eta_min=args.lr_backbone * 0.1,
        )
        for epoch in range(1, full_epochs + 1):
            tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, scaler, criterion)
            val_stats = evaluate_binary(model, val_loader)
            scheduler.step()
            f1_m = val_stats["manip_f1"]
            star = " ★" if f1_m > best_f1 else ""
            print(
                f"  FineTune {epoch:02d}/{full_epochs} | "
                f"loss={tr_loss:.4f} | tr_acc={tr_acc:.3f} | "
                f"val_acc={val_stats['accuracy']:.3f} | "
                f"auth_recall={val_stats['per_class']['authentic']['recall']:.3f} | "
                f"manip_recall={val_stats['manip_recall']:.3f} | "
                f"manip_f1={f1_m:.3f}{star}"
            )
            history.append(
                {
                    "stage": "finetune",
                    "epoch": epoch,
                    "train_loss": tr_loss,
                    "train_acc": tr_acc,
                    "val_manip_f1": f1_m,
                }
            )
            if f1_m > best_f1:
                best_f1 = f1_m
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "history": history,
                        "best_manip_f1": best_f1,
                        "epoch": epoch,
                        "version": "v4_coral",
                        "resumed_from": str(args.resume),
                    },
                    ckpt_path,
                )

    if not ckpt_path.exists():
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "history": history,
                "best_manip_f1": best_f1,
                "epoch": args.epochs,
                "version": "v4_coral",
                "resumed_from": str(args.resume),
            },
            ckpt_path,
        )
    print(f"\n[완료] best manip_f1={best_f1:.4f} -> {ckpt_path}")
    return ckpt_path


@torch.no_grad()
def eval_all_datasets(ckpt_path: Path, args) -> dict:
    print("\n[전체 데이터셋 평가]")
    model = CoralSpecialistMv4(pretrained=False).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    out_dir = ROOT / "experiments" / "results" / "specialist_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    summary = {}
    manip_f1s = []
    for ds_key, jsonl_rel in EVAL_DATASETS.items():
        records = load_jsonl_records(jsonl_rel)
        records = [r for r in records if r["true_label"] in {"authentic", "manipulated"}]
        loader = DataLoader(
            BinaryDataset(records, VAL_TRANSFORM),
            batch_size=args.batch_size * 2,
            shuffle=False,
            num_workers=4,
            pin_memory=(DEVICE.type == "cuda"),
        )
        stats = evaluate_binary(model, loader)
        probs = stats.pop("probs")
        preds = stats.pop("preds")
        stats.pop("labels")
        manip_f1s.append(stats["manip_f1"])

        out_jsonl = out_dir / f"specialist_m_v4_coral_{ds_key}_{ts}.jsonl"
        with open(out_jsonl, "w") as f:
            for rec, pred_idx, prob_vec in zip(records, preds.tolist(), probs.tolist()):
                f.write(
                    json.dumps(
                        {
                            "image_path": rec["image_path"],
                            "true_label": rec["true_label"],
                            "pred_label": "manipulated" if pred_idx == 1 else "authentic",
                            "manip_score": float(prob_vec[1]),
                            "authentic_score": float(prob_vec[0]),
                            "confidence": float(max(prob_vec)),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        summary[ds_key] = {
            "n_binary": len(records),
            "accuracy": stats["accuracy"],
            "macro_recall": stats["macro_recall"],
            "auth_recall": stats["per_class"]["authentic"]["recall"],
            "manip_recall": stats["manip_recall"],
            "manip_f1": stats["manip_f1"],
            "per_class": stats["per_class"],
            "jsonl": str(out_jsonl),
        }
        print(
            f"  [{ds_key}] auth_recall={summary[ds_key]['auth_recall']:.3f} | "
            f"manip_recall={summary[ds_key]['manip_recall']:.3f} | "
            f"manip_f1={summary[ds_key]['manip_f1']:.3f}"
        )

    summary["avg_manip_f1"] = float(np.mean(manip_f1s)) if manip_f1s else 0.0
    summary["checkpoint"] = str(ckpt_path)
    summary_path = out_dir / f"specialist_m_v4_coral_summary_{ts}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n저장: {summary_path}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="SpecM-v4 Coral fine-tune")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--warmup-head-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--lr-backbone", type=float, default=1e-5)
    parser.add_argument("--lr-head", type=float, default=2e-4)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--focal-alpha", type=float, default=0.6)
    parser.add_argument("--genimage-max", type=int, default=3000)
    parser.add_argument("--resume", type=str, default=str(V4_BEST_CKPT))
    parser.add_argument("--eval-only", type=str, default=None)
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    if args.eval_only:
        eval_all_datasets(Path(args.eval_only), args)
    else:
        ckpt = run_training(args)
        eval_all_datasets(ckpt, args)


if __name__ == "__main__":
    main()
