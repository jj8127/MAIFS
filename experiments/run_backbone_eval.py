#!/usr/bin/env python3
"""
Phase 2 — Backbone 단독 평가 스크립트
ForMa / MobileCLIP-S2 / Tiny-LaDeDa 각각을 4개 JSONL 데이터셋에서 실행하여
image-level verdict + confidence 를 저장합니다.

실행 (MAIFS/ 루트에서):
    .venv-qwen/bin/python experiments/run_backbone_eval.py
"""

from __future__ import annotations

import json
import sys
import os
import warnings
import argparse
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ── 경로 설정 ───────────────────────────────────────────────────────────── #
ROOT = Path(__file__).parent.parent  # MAIFS/
FORMA_DIR = ROOT / "ForMa-main"
TINYLA_DIR = ROOT / "TinyLaDeDa-main"

sys.path.insert(0, str(FORMA_DIR / "models"))
sys.path.insert(0, str(FORMA_DIR))
sys.path.insert(0, str(TINYLA_DIR))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASETS: Dict[str, Dict] = {
    "base": {
        "jsonl": "experiments/results/phase2_patha_scale500_gain_predictor/patha_agent_outputs_20260304_080157.jsonl",
        "desc": "CASIA2 + BigGAN (1500)",
    },
    "dsC": {
        "jsonl": "experiments/results/phase2_patha_case3_scale300_dsC/patha_agent_outputs_20260303_105005.jsonl",
        "desc": "CASIA2 + IMD2020 + BigGAN (900)",
    },
    "opensdi": {
        "jsonl": "experiments/results/phase2_patha_case3_opensdi_scale300/patha_agent_outputs_fixed_seed42.jsonl",
        "desc": "OpenSDID (900)",
    },
    "aigenproxy": {
        "jsonl": "experiments/results/phase2_patha_case3_aigenproxy_scale300/patha_agent_outputs_fixed_seed42.jsonl",
        "desc": "AI-GenBench proxy (900)",
    },
}

OUT_DIR = ROOT / "experiments" / "results" / "backbone_eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 이미지 전처리 ────────────────────────────────────────────────────────── #
FORMA_TRANSFORM = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

TINYLA_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

CLIP_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                         std=[0.26862954, 0.26130258, 0.27577711]),
])


# ── ForMa 로딩 ──────────────────────────────────────────────────────────── #
def resolve_forma_weight_path() -> Optional[Path]:
    candidates = [
        FORMA_DIR / "weights" / "ForMa_weights.pth",
        ROOT / "ForMa_weights.pth",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def load_forma() -> Optional[torch.nn.Module]:
    print("[ForMa] 로딩 중...", flush=True)
    from models.vmamba_pixelshuf_modals import Forensic_Vmamba
    model = Forensic_Vmamba().to(DEVICE)
    weight_path = resolve_forma_weight_path()
    if weight_path is not None:
        ckpt = torch.load(weight_path, map_location="cpu")
        state = ckpt.get("model_state_dict", ckpt)
        missing = model.load_state_dict(state, strict=False)
        print(
            f"[ForMa] 가중치 로드 완료: {weight_path}. "
            f"missing: {len(missing.missing_keys)}, unexpected: {len(missing.unexpected_keys)}"
        )
    else:
        print("[ForMa] ⚠ 사전학습 가중치 없음 — 평가에서 제외합니다.")
        return None

    if DEVICE.type == "cuda":
        patched = 0
        for module in model.modules():
            if hasattr(module, "forward_corev2") and hasattr(module, "disable_force32"):
                module.forward_core = partial(
                    module.forward_corev2,
                    force_fp32=(not module.disable_force32),
                    selective_scan_backend="mamba",
                )
                patched += 1
        print(f"[ForMa] CUDA mamba backend 적용: {patched} modules")

    model.eval()
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[ForMa] {params:.1f}M params, device={DEVICE}")
    return model


def forma_predict(model: torch.nn.Module, img_tensor: torch.Tensor,
                  manip_threshold: float = 0.15) -> Tuple[str, float]:
    """
    ForMa output: (1, 2, H/4, W/4) — class 0=authentic, class 1=manipulated
    manip_threshold: manipulated pixel ratio threshold
    """
    with torch.no_grad():
        out = model(img_tensor.to(DEVICE))  # (1, 2, h, w)
    prob = F.softmax(out, dim=1)  # (1, 2, h, w)
    manip_prob = prob[0, 1]  # (h, w) manipulated probability
    manip_ratio = (manip_prob > 0.5).float().mean().item()
    manip_conf = manip_prob.mean().item()
    # ForMa는 splicing detection → "manipulated" or "authentic" (ai_gen 감지 불가)
    if manip_ratio > manip_threshold:
        verdict = "manipulated"
        confidence = min(0.5 + manip_ratio * 2, 0.99)
    else:
        verdict = "authentic"
        confidence = min(0.5 + (manip_threshold - manip_ratio) * 3, 0.99)
    return verdict, float(confidence)


def forma_predict_batch(
    model: torch.nn.Module,
    img_tensors: torch.Tensor,
    manip_threshold: float = 0.15,
) -> List[Tuple[str, float]]:
    with torch.no_grad():
        out = model(img_tensors.to(DEVICE))  # (B, 2, h, w)
    prob = F.softmax(out, dim=1)[:, 1]  # (B, h, w)
    manip_ratio = (prob > 0.5).float().mean(dim=(1, 2))
    results: List[Tuple[str, float]] = []
    for ratio in manip_ratio.tolist():
        if ratio > manip_threshold:
            verdict = "manipulated"
            confidence = min(0.5 + ratio * 2, 0.99)
        else:
            verdict = "authentic"
            confidence = min(0.5 + (manip_threshold - ratio) * 3, 0.99)
        results.append((verdict, float(confidence)))
    return results


# ── MobileCLIP-S2 로딩 ──────────────────────────────────────────────────── #
def load_mobileclip() -> object:
    print("[MobileCLIP-S2] 로딩 중...", flush=True)
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        "MobileCLIP-S2", pretrained="datacompdr",
        cache_dir=str(ROOT / "weights" / "mobileclip")
    )
    model = model.to(DEVICE)
    model.eval()
    tokenizer = open_clip.get_tokenizer("MobileCLIP-S2")
    # 텍스트 프롬프트 — 3-class 분류
    PROMPTS = {
        "authentic": [
            "a real authentic photograph taken by a camera",
            "a genuine unmodified photograph",
            "a real photo without any digital manipulation",
        ],
        "manipulated": [
            "a digitally manipulated photograph with altered regions",
            "a spliced or copy-moved image forgery",
            "a photograph with forged or tampered regions",
        ],
        "ai_generated": [
            "an image generated by artificial intelligence",
            "a synthetic AI generated image",
            "a fake image created by a generative AI model",
        ],
    }
    # 텍스트 피처 사전 계산
    text_feats = {}
    with torch.no_grad():
        for cls, prompts in PROMPTS.items():
            tokens = tokenizer(prompts).to(DEVICE)
            feats = model.encode_text(tokens)
            feats = F.normalize(feats, dim=-1).mean(dim=0)
            text_feats[cls] = F.normalize(feats, dim=0)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[MobileCLIP-S2] {params:.1f}M params, device={DEVICE}")
    return model, tokenizer, text_feats


def mobileclip_predict(model, tokenizer, text_feats: Dict,
                       img_tensor: torch.Tensor) -> Tuple[str, float]:
    """Zero-shot 3-class classification via text similarity."""
    with torch.no_grad():
        img_feat = model.encode_image(img_tensor.to(DEVICE))
        img_feat = F.normalize(img_feat, dim=-1)  # (1, D)
    scores = {}
    for cls, tf in text_feats.items():
        scores[cls] = (img_feat @ tf.unsqueeze(-1)).squeeze().item()
    # softmax over scores
    keys = list(scores.keys())
    vals = torch.tensor([scores[k] for k in keys])
    probs = F.softmax(vals / 0.07, dim=0)  # temperature=0.07
    best_idx = probs.argmax().item()
    verdict = keys[best_idx]
    confidence = probs[best_idx].item()
    return verdict, float(confidence)


# ── Tiny-LaDeDa 로딩 ────────────────────────────────────────────────────── #
def load_tinylada() -> torch.nn.Module:
    print("[Tiny-LaDeDa] 로딩 중...", flush=True)
    from networks.Tiny_LaDeDa import Bottleneck, TinyLaDeDa
    orig_exp = Bottleneck.expansion
    Bottleneck.expansion = 1
    model = TinyLaDeDa(Bottleneck, layer=1, stride=2, kernel=1,
                       preprocess_type="right_diag")
    Bottleneck.expansion = orig_exp
    weight_path = TINYLA_DIR / "weights" / "Tiny_LaDeDa" / "WildRF_Tiny_LaDeDa.pth"
    ckpt = torch.load(weight_path, map_location="cpu")
    model.load_state_dict(ckpt)
    model = model.to(DEVICE)
    model.eval()
    params = sum(p.numel() for p in model.parameters()) / 1e3
    print(f"[Tiny-LaDeDa] {params:.1f}K params (WildRF), device={DEVICE}")
    return model


def tinylada_predict(model: torch.nn.Module,
                     img_tensor: torch.Tensor) -> Tuple[str, float]:
    """
    Binary real/fake → map fake → manipulated|ai_generated (ambiguous)
    WildRF: trained on real-world social media fakes (mostly AI-gen heavy)
    """
    with torch.no_grad():
        logit = model(img_tensor.to(DEVICE))  # (1, 1)
    prob_fake = torch.sigmoid(logit).squeeze().item()
    if prob_fake > 0.5:
        # WildRF는 AI-gen 비중이 높아 fake=ai_generated로 우선 매핑
        verdict = "ai_generated"
        confidence = prob_fake
    else:
        verdict = "authentic"
        confidence = 1.0 - prob_fake
    return verdict, float(confidence)


# ── 이미지 로드 ──────────────────────────────────────────────────────────── #
def load_image(image_path: str) -> Image.Image:
    full = ROOT / image_path
    if not full.exists():
        raise FileNotFoundError(f"이미지 없음: {full}")
    return Image.open(full).convert("RGB")


# ── 단일 백본 평가 ───────────────────────────────────────────────────────── #
def evaluate_backbone(
    backbone_name: str,
    predict_fn: Callable[[torch.Tensor], Tuple[str, float]],
    transform: transforms.Compose,
    records: List[Dict],
    dataset_name: str,
    batch_predict_fn: Optional[Callable[[torch.Tensor], List[Tuple[str, float]]]] = None,
    batch_size: int = 1,
) -> List[Dict]:
    results = []
    correct = 0
    total = 0
    errors = 0
    if batch_predict_fn is not None and batch_size > 1:
        for start in range(0, len(records), batch_size):
            if start % 100 == 0:
                print(f"  [{backbone_name}] {dataset_name}: {start}/{len(records)}", flush=True)
            batch_records = records[start:start + batch_size]
            valid_records: List[Dict] = []
            tensors: List[torch.Tensor] = []
            for rec in batch_records:
                try:
                    img = load_image(rec["image_path"])
                    tensors.append(transform(img))
                    valid_records.append(rec)
                except FileNotFoundError:
                    errors += 1
                except Exception as e:
                    errors += 1
                    if errors <= 3:
                        print(f"  [{backbone_name}] 오류 ({rec['image_path']}): {e}")
            if not tensors:
                continue
            try:
                preds = batch_predict_fn(torch.stack(tensors, dim=0))
            except Exception as e:
                errors += len(valid_records)
                if errors <= 3:
                    print(f"  [{backbone_name}] 배치 오류 ({dataset_name}, start={start}): {e}")
                continue
            for rec, (verdict, conf) in zip(valid_records, preds):
                true_label = rec["true_label"]
                result = {
                    "image_path": rec["image_path"],
                    "true_label": true_label,
                    "sub_type": rec.get("sub_type", ""),
                    "verdict": verdict,
                    "confidence": conf,
                    "correct": verdict == true_label,
                }
                results.append(result)
                if verdict == true_label:
                    correct += 1
                total += 1
    else:
        for i, rec in enumerate(records):
            if i % 100 == 0:
                print(f"  [{backbone_name}] {dataset_name}: {i}/{len(records)}", flush=True)
            try:
                img = load_image(rec["image_path"])
                tensor = transform(img).unsqueeze(0)  # (1, C, H, W)
                verdict, conf = predict_fn(tensor)
                true_label = rec["true_label"]
                result = {
                    "image_path": rec["image_path"],
                    "true_label": true_label,
                    "sub_type": rec.get("sub_type", ""),
                    "verdict": verdict,
                    "confidence": conf,
                    "correct": verdict == true_label,
                }
                results.append(result)
                if verdict == true_label:
                    correct += 1
                total += 1
            except FileNotFoundError:
                errors += 1
            except Exception as e:
                errors += 1
                if errors <= 3:
                    print(f"  [{backbone_name}] 오류 ({rec['image_path']}): {e}")
    acc = correct / total if total > 0 else 0
    print(f"  [{backbone_name}] {dataset_name}: acc={acc:.3f} ({correct}/{total}), errors={errors}")
    return results


# ── 통계 계산 ────────────────────────────────────────────────────────────── #
def compute_stats(results: List[Dict]) -> Dict:
    from collections import defaultdict
    labels = ["authentic", "manipulated", "ai_generated"]
    by_true = defaultdict(list)
    for r in results:
        by_true[r["true_label"]].append(r)
    stats: Dict = {"overall": {}, "per_class": {}}
    correct_total = sum(1 for r in results if r["correct"])
    stats["overall"]["accuracy"] = correct_total / len(results) if results else 0
    stats["overall"]["n"] = len(results)
    # per-class recall
    for lbl in labels:
        recs = by_true.get(lbl, [])
        if not recs:
            continue
        recall = sum(1 for r in recs if r["correct"]) / len(recs)
        verdict_dist = {}
        for r in recs:
            verdict_dist[r["verdict"]] = verdict_dist.get(r["verdict"], 0) + 1
        stats["per_class"][lbl] = {
            "n": len(recs),
            "recall": recall,
            "verdict_dist": verdict_dist,
            "avg_conf": float(np.mean([r["confidence"] for r in recs])),
        }
    return stats


# ── 메인 ─────────────────────────────────────────────────────────────────── #
def main():
    parser = argparse.ArgumentParser(description="Phase 2 backbone solo evaluation")
    parser.add_argument(
        "--backbone",
        choices=["all", "forma", "mobileclip_s2", "tiny_ladeda"],
        default="all",
        help="evaluate only the selected backbone",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Phase 2 — Backbone 단독 평가")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    backbones = {}

    if args.backbone in ("all", "forma"):
        forma = load_forma()
        if forma is not None:
            backbones["forma"] = {
                "predict_fn": lambda t: forma_predict(forma, t),
                "transform": FORMA_TRANSFORM,
                "batch_predict_fn": lambda t: forma_predict_batch(forma, t),
                "batch_size": 8,
            }

    if args.backbone in ("all", "mobileclip_s2"):
        mobileclip_model, tokenizer, text_feats = load_mobileclip()
        backbones["mobileclip_s2"] = {
            "predict_fn": lambda t: mobileclip_predict(mobileclip_model, tokenizer, text_feats, t),
            "transform": CLIP_TRANSFORM,
            "batch_predict_fn": None,
            "batch_size": 1,
        }

    if args.backbone in ("all", "tiny_ladeda"):
        tinylada = load_tinylada()
        backbones["tiny_ladeda"] = {
            "predict_fn": lambda t: tinylada_predict(tinylada, t),
            "transform": TINYLA_TRANSFORM,
            "batch_predict_fn": None,
            "batch_size": 1,
        }

    if not backbones:
        raise RuntimeError("평가 가능한 backbone이 없습니다.")

    all_summary: Dict = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for ds_name, ds_info in DATASETS.items():
        jsonl_path = ROOT / ds_info["jsonl"]
        if not jsonl_path.exists():
            print(f"\n[스킵] {ds_name}: {jsonl_path} 없음")
            continue

        print(f"\n{'='*50}")
        print(f"데이터셋: {ds_name} — {ds_info['desc']}")
        print(f"{'='*50}")

        # 기존 JSONL 로드
        records = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        print(f"샘플 수: {len(records)}")

        ds_summary: Dict = {"desc": ds_info["desc"], "n": len(records), "backbones": {}}

        for bb_name, backbone_cfg in backbones.items():
            results = evaluate_backbone(
                bb_name,
                backbone_cfg["predict_fn"],
                backbone_cfg["transform"],
                records,
                ds_name,
                batch_predict_fn=backbone_cfg["batch_predict_fn"],
                batch_size=backbone_cfg["batch_size"],
            )

            # JSONL 저장
            out_path = OUT_DIR / f"{bb_name}_{ds_name}_{timestamp}.jsonl"
            with open(out_path, "w", encoding="utf-8") as f:
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

            stats = compute_stats(results)
            ds_summary["backbones"][bb_name] = stats
            print(f"  → 저장: {out_path.name} | acc={stats['overall']['accuracy']:.3f}")

        all_summary[ds_name] = ds_summary

    # 종합 요약 저장
    summary_path = OUT_DIR / f"backbone_eval_summary_{timestamp}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_summary, f, ensure_ascii=False, indent=2)
    print(f"\n종합 요약 저장: {summary_path}")

    # 표 출력
    print("\n" + "=" * 70)
    print("Backbone 단독 평가 결과 요약 (Overall Accuracy)")
    print("=" * 70)
    print(f"{'Dataset':<12} {'ForMa':>10} {'MobileCLIP-S2':>14} {'Tiny-LaDeDa':>12}")
    print("-" * 70)
    for ds_name, ds_info in all_summary.items():
        bbs = ds_info.get("backbones", {})
        forma_acc = bbs.get("forma", {}).get("overall", {}).get("accuracy", float("nan"))
        clip_acc = bbs.get("mobileclip_s2", {}).get("overall", {}).get("accuracy", float("nan"))
        tiny_acc = bbs.get("tiny_ladeda", {}).get("overall", {}).get("accuracy", float("nan"))
        print(f"{ds_name:<12} {forma_acc:>10.3f} {clip_acc:>14.3f} {tiny_acc:>12.3f}")
    print("=" * 70)

    # Per-class recall 표
    print("\nPer-Class Recall (authentic / manipulated / ai_generated)")
    for ds_name, ds_info in all_summary.items():
        print(f"\n[{ds_name}]")
        bbs = ds_info.get("backbones", {})
        for bb_name in ["forma", "mobileclip_s2", "tiny_ladeda"]:
            pc = bbs.get(bb_name, {}).get("per_class", {})
            recalls = {cls: pc.get(cls, {}).get("recall", float("nan")) for cls in ["authentic", "manipulated", "ai_generated"]}
            print(f"  {bb_name:<18}: auth={recalls['authentic']:.3f}, manip={recalls['manipulated']:.3f}, ai_gen={recalls['ai_generated']:.3f}")


if __name__ == "__main__":
    main()
