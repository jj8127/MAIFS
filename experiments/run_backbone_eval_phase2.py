"""
SHIELD Phase 2 Step 1: 새 백본 단독 평가 (Backbone Solo Evaluation)

목적:
    ForMa(VMamba), MobileCLIP-S2, Tiny-LaDeDa 3개 신규 백본을
    기존 4개 JSONL 데이터셋에 대해 단독 추론 후 성능 측정.
    결과는 Phase 3 (3-Track 비교 실험)의 기준선으로 활용.

백본별 역할:
    - ForMa       : 픽셀 수준 조작 탐지 (VMamba 37M) → authentic/manipulated
    - MobileCLIP-S2: 제로샷 CLIP 분류 (99M) → authentic/manipulated/ai_generated
    - Tiny-LaDeDa : 초경량 Real/Fake 분류 (1.3K) → authentic vs (manipulated|ai_generated)

실행:
    .venv-qwen/bin/python experiments/run_backbone_eval_phase2.py
    .venv-qwen/bin/python experiments/run_backbone_eval_phase2.py --backbone forma
    .venv-qwen/bin/python experiments/run_backbone_eval_phase2.py --backbone mobileclip
    .venv-qwen/bin/python experiments/run_backbone_eval_phase2.py --backbone tinylad
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "experiments/results/backbone_eval_phase2"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ──────────────────────────────────────────────────────────────────────
# 데이터셋 JSONL 파일 목록 (기존 Phase 1 검증 데이터 재사용)
# ──────────────────────────────────────────────────────────────────────
JSONL_FILES = {
    "base":       ROOT / "experiments/results/phase2_patha_scale500_gain_predictor/patha_agent_outputs_20260304_080157.jsonl",
    "dsC":        ROOT / "experiments/results/phase2_crossdataset_dsC/patha_agent_outputs_dsC.jsonl",
    "opensdi":    ROOT / "experiments/results/phase2_crossdataset_opensdi/patha_agent_outputs_opensdi.jsonl",
    "aigenproxy": ROOT / "experiments/results/phase2_crossdataset_aigenproxy/patha_agent_outputs_aigenproxy.jsonl",
}

# ──────────────────────────────────────────────────────────────────────
# 이미지 전처리 (공통 유틸)
# ──────────────────────────────────────────────────────────────────────
_basic_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

_forma_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

_clip_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                         std=[0.26862954, 0.26130258, 0.27577711]),
])

_tiny_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def load_image(path: str) -> Image.Image:
    img = Image.open(ROOT / path).convert("RGB")
    return img


# ──────────────────────────────────────────────────────────────────────
# ForMa 백본
# ──────────────────────────────────────────────────────────────────────
class ForMaBackbone:
    """
    VMamba 기반 픽셀 수준 조작 탐지.
    출력: authentic / manipulated (ai_generated 탐지 불가)
    """
    MANIP_THRESHOLD = 0.05  # 조작 픽셀 비율 임계값

    def __init__(self):
        self.model = None
        self.name = "forma"

    def load(self) -> bool:
        forma_dir = ROOT / "ForMa-main"
        if not forma_dir.exists():
            print(f"[ForMa] 디렉토리 없음: {forma_dir}")
            return False
        sys.path.insert(0, str(forma_dir))
        sys.path.insert(0, str(forma_dir / "models"))
        try:
            from models.vmamba_pixelshuf_modals import Forensic_Vmamba
            model = Forensic_Vmamba().to(DEVICE)
            weights_path = forma_dir / "weights" / "ForMa_weights.pth"
            if weights_path.exists():
                ckpt = torch.load(weights_path, map_location="cpu")
                sd = ckpt.get("model_state_dict", ckpt)
                missing, unexpected = model.load_state_dict(sd, strict=False)
                if missing:
                    print(f"[ForMa] 누락 레이어: {len(missing)}개")
                print(f"[ForMa] 가중치 로드 완료: {weights_path.name}")
            else:
                print(f"[ForMa] ⚠ 가중치 없음({weights_path.name}), 랜덤 초기화로 실행")
            model.eval()
            self.model = model
            return True
        except Exception as e:
            print(f"[ForMa] 로드 실패: {e}")
            return False

    @torch.no_grad()
    def predict(self, img: Image.Image) -> Dict:
        t0 = time.perf_counter()
        x = _forma_transform(img).unsqueeze(0).to(DEVICE)
        out = self.model(x)  # (1, 2, H/4, W/4)
        # softmax → 조작 클래스 확률맵
        prob_map = F.softmax(out, dim=1)[0, 1].cpu().numpy()  # (H/4, W/4)
        manip_ratio = float((prob_map > 0.5).mean())
        confidence = float(prob_map.mean())
        if manip_ratio >= self.MANIP_THRESHOLD:
            verdict = "manipulated"
            conf = min(0.95, 0.5 + manip_ratio * 2)
        else:
            verdict = "authentic"
            conf = min(0.95, 0.5 + (self.MANIP_THRESHOLD - manip_ratio) * 5)
        elapsed = time.perf_counter() - t0
        return {
            "verdict": verdict,
            "confidence": round(conf, 4),
            "manip_ratio": round(manip_ratio, 4),
            "mean_prob": round(confidence, 4),
            "time_ms": round(elapsed * 1000, 1),
        }


# ──────────────────────────────────────────────────────────────────────
# MobileCLIP-S2 백본 (Zero-shot)
# ──────────────────────────────────────────────────────────────────────
class MobileCLIPBackbone:
    """
    Apple MobileCLIP-S2 제로샷 분류.
    텍스트 프롬프트: authentic / manipulated / AI generated
    출력: authentic / manipulated / ai_generated
    """
    PROMPTS = {
        "authentic":    ["a real photograph", "an authentic photo", "a genuine image"],
        "manipulated":  ["a manipulated photo", "a forged image", "a spliced photograph"],
        "ai_generated": ["an AI generated image", "a synthetic image created by AI",
                         "a deepfake or GAN generated photo"],
    }

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.text_features = None
        self.name = "mobileclip"

    def load(self) -> bool:
        try:
            import open_clip
            weights_path = ROOT / "weights/mobileclip/mobileclip_s2_openclip.pt"
            model, _, preprocess = open_clip.create_model_and_transforms(
                "MobileCLIP-S2", pretrained="datacompdr",
                cache_dir=str(ROOT / "weights/mobileclip")
            )
            model = model.to(DEVICE).eval()
            tokenizer = open_clip.get_tokenizer("MobileCLIP-S2")

            # 클래스별 텍스트 피처 사전 계산
            text_features = {}
            for cls, prompts in self.PROMPTS.items():
                tokens = tokenizer(prompts).to(DEVICE)
                with torch.no_grad():
                    feats = model.encode_text(tokens)
                    feats = F.normalize(feats, dim=-1)
                    text_features[cls] = feats.mean(0)  # 프롬프트 평균

            self.model = model
            self.tokenizer = tokenizer
            self.text_features = text_features
            print("[MobileCLIP] 로드 완료. 텍스트 피처 3클래스 인코딩 완료.")
            return True
        except Exception as e:
            print(f"[MobileCLIP] 로드 실패: {e}")
            return False

    @torch.no_grad()
    def predict(self, img: Image.Image) -> Dict:
        t0 = time.perf_counter()
        x = _clip_transform(img).unsqueeze(0).to(DEVICE)
        img_feat = self.model.encode_image(x)
        img_feat = F.normalize(img_feat, dim=-1)[0]  # (D,)

        scores = {}
        for cls, txt_feat in self.text_features.items():
            scores[cls] = float(img_feat @ txt_feat)

        # softmax over scores
        score_arr = torch.tensor([scores["authentic"], scores["manipulated"], scores["ai_generated"]])
        probs = F.softmax(score_arr * 100, dim=0).numpy()  # temperature scaling

        classes = ["authentic", "manipulated", "ai_generated"]
        pred_idx = int(probs.argmax())
        verdict = classes[pred_idx]
        confidence = float(probs[pred_idx])

        elapsed = time.perf_counter() - t0
        return {
            "verdict": verdict,
            "confidence": round(confidence, 4),
            "scores": {c: round(float(s), 4) for c, s in scores.items()},
            "probs": {c: round(float(p), 4) for c, p in zip(classes, probs)},
            "time_ms": round(elapsed * 1000, 1),
        }


# ──────────────────────────────────────────────────────────────────────
# Tiny-LaDeDa 백본
# ──────────────────────────────────────────────────────────────────────
class TinyLaDeDaBackbone:
    """
    4-Conv 초경량 Real/Fake 이진 분류기.
    출력: authentic (real=1) vs uncertain/manipulated (fake=0)
    AI-generated vs Manipulated 구분 불가 → fake를 "uncertain"으로 표시
    WildRF 가중치 사용 (실세계 분포)
    """
    FAKE_THRESHOLD = 0.5

    def __init__(self):
        self.model = None
        self.name = "tinylad"

    def load(self) -> bool:
        ladeda_dir = ROOT / "TinyLaDeDa-main"
        if not ladeda_dir.exists():
            print(f"[TinyLaDeDa] 디렉토리 없음: {ladeda_dir}")
            return False
        sys.path.insert(0, str(ladeda_dir))
        try:
            from networks.Tiny_LaDeDa import Bottleneck, TinyLaDeDa
            # expansion=1로 체크포인트와 일치 (WildRF/ForenSynth 학습 버전)
            orig_exp = Bottleneck.expansion
            Bottleneck.expansion = 1
            model = TinyLaDeDa(Bottleneck, layer=1, stride=2, kernel=1,
                                preprocess_type="right_diag")
            Bottleneck.expansion = orig_exp  # 복원

            weights_path = ladeda_dir / "weights" / "Tiny_LaDeDa" / "WildRF_Tiny_LaDeDa.pth"
            if weights_path.exists():
                ckpt = torch.load(weights_path, map_location="cpu")
                model.load_state_dict(ckpt)
                print(f"[TinyLaDeDa] WildRF 가중치 로드 완료")
            else:
                print(f"[TinyLaDeDa] ⚠ 가중치 없음, 랜덤 초기화로 실행")

            model = model.to(DEVICE).eval()
            self.model = model
            return True
        except Exception as e:
            print(f"[TinyLaDeDa] 로드 실패: {e}")
            import traceback; traceback.print_exc()
            return False

    @torch.no_grad()
    def predict(self, img: Image.Image) -> Dict:
        t0 = time.perf_counter()
        x = _tiny_transform(img).unsqueeze(0).to(DEVICE)
        logit = self.model(x)[0, 0]  # scalar
        prob_real = float(torch.sigmoid(logit))

        # WildRF: 1=real, 0=fake
        if prob_real >= self.FAKE_THRESHOLD:
            verdict = "authentic"
            confidence = prob_real
        else:
            # fake이지만 manipulated vs ai_generated 구분 불가
            verdict = "uncertain"
            confidence = 1.0 - prob_real

        elapsed = time.perf_counter() - t0
        return {
            "verdict": verdict,
            "confidence": round(confidence, 4),
            "prob_real": round(prob_real, 4),
            "time_ms": round(elapsed * 1000, 1),
        }


# ──────────────────────────────────────────────────────────────────────
# 평가 함수
# ──────────────────────────────────────────────────────────────────────
VERDICT_MAP = {
    "authentic":    "authentic",
    "manipulated":  "manipulated",
    "ai_generated": "ai_generated",
    "uncertain":    "uncertain",
}


def evaluate_backbone(backbone, jsonl_path: Path, dataset_name: str, max_samples: int = 500) -> Dict:
    if not jsonl_path.exists():
        print(f"  ⚠ JSONL 없음: {jsonl_path}")
        return {}

    records = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if max_samples and len(records) > max_samples:
        np.random.seed(42)
        idx = np.random.choice(len(records), max_samples, replace=False)
        records = [records[i] for i in idx]

    print(f"\n  [{backbone.name}] {dataset_name}: {len(records)}장 평가 중...")

    results = []
    errors = 0
    t_total = time.perf_counter()

    for i, rec in enumerate(records):
        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{len(records)}...", flush=True)
        try:
            img = load_image(rec["image_path"])
            pred = backbone.predict(img)
            results.append({
                "image_path": rec["image_path"],
                "true_label": rec["true_label"],
                "sub_type": rec.get("sub_type", ""),
                "verdict": pred["verdict"],
                "confidence": pred["confidence"],
                "details": pred,
            })
        except Exception as e:
            errors += 1
            results.append({
                "image_path": rec.get("image_path", ""),
                "true_label": rec.get("true_label", ""),
                "sub_type": rec.get("sub_type", ""),
                "verdict": "uncertain",
                "confidence": 0.3,
                "details": {"error": str(e)},
            })

    elapsed_total = time.perf_counter() - t_total

    # ── 지표 계산 ──────────────────────────────────────────────
    labels = ["authentic", "manipulated", "ai_generated", "uncertain"]
    metrics = compute_metrics(results, labels)
    metrics["dataset"] = dataset_name
    metrics["n_samples"] = len(results)
    metrics["n_errors"] = errors
    metrics["total_time_s"] = round(elapsed_total, 2)
    metrics["avg_ms_per_image"] = round(elapsed_total / len(results) * 1000, 1)

    print(f"    → Macro-F1={metrics['macro_f1']:.4f}  Acc={metrics['accuracy']:.4f}  "
          f"errors={errors}  avg={metrics['avg_ms_per_image']}ms/img")

    return {"meta": metrics, "records": results}


def compute_metrics(results: List[Dict], labels: List[str]) -> Dict:
    """Macro-F1, per-class F1, accuracy 계산."""
    true_labels = [r["true_label"] for r in results]
    pred_labels = [r["verdict"] for r in results]

    # 3-class only (authentic/manipulated/ai_generated)
    eval_labels = ["authentic", "manipulated", "ai_generated"]

    correct = sum(t == p for t, p in zip(true_labels, pred_labels))
    accuracy = correct / len(true_labels) if true_labels else 0.0

    per_class = {}
    for cls in eval_labels:
        tp = sum(t == cls and p == cls for t, p in zip(true_labels, pred_labels))
        fp = sum(t != cls and p == cls for t, p in zip(true_labels, pred_labels))
        fn = sum(t == cls and p != cls for t, p in zip(true_labels, pred_labels))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class[cls] = {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}

    macro_f1 = np.mean([per_class[c]["f1"] for c in eval_labels])

    return {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(float(macro_f1), 4),
        "per_class": per_class,
    }


# ──────────────────────────────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", choices=["forma", "mobileclip", "tinylad", "all"],
                        default="all", help="평가할 백본 (기본: all)")
    parser.add_argument("--dataset", choices=list(JSONL_FILES.keys()) + ["all"],
                        default="all", help="평가 데이터셋 (기본: all)")
    parser.add_argument("--max_samples", type=int, default=500,
                        help="데이터셋별 최대 샘플 수 (기본: 500)")
    args = parser.parse_args()

    print("=" * 70)
    print("  SHIELD Phase 2 Step 1: 신규 백본 단독 평가")
    print("=" * 70)
    print(f"  Device: {DEVICE}")
    print(f"  Backbone: {args.backbone}  |  Dataset: {args.dataset}  |  Max: {args.max_samples}")

    # ── 백본 선택 ──────────────────────────────────────────────
    backbone_classes = {
        "forma":      ForMaBackbone,
        "mobileclip": MobileCLIPBackbone,
        "tinylad":    TinyLaDeDaBackbone,
    }
    if args.backbone == "all":
        selected_backbones = list(backbone_classes.keys())
    else:
        selected_backbones = [args.backbone]

    # ── 데이터셋 선택 ──────────────────────────────────────────
    if args.dataset == "all":
        selected_datasets = list(JSONL_FILES.items())
    else:
        selected_datasets = [(args.dataset, JSONL_FILES[args.dataset])]

    # ── 평가 루프 ──────────────────────────────────────────────
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results = {}

    for bname in selected_backbones:
        print(f"\n{'─'*60}")
        print(f"  백본: {bname}")
        print(f"{'─'*60}")

        backbone = backbone_classes[bname]()
        loaded = backbone.load()
        if not loaded:
            print(f"  ✗ {bname} 로드 실패, 건너뜀")
            continue

        bname_results = {}
        for dname, jpath in selected_datasets:
            data = evaluate_backbone(backbone, jpath, dname, args.max_samples)
            if data:
                bname_results[dname] = data

        all_results[bname] = bname_results

        # 백본별 저장
        out_path = RESULT_DIR / f"{bname}_eval_{timestamp}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(bname_results, f, indent=2, ensure_ascii=False)
        print(f"\n  저장: {out_path}")

    # ── 전체 요약 ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  요약 (Macro-F1)")
    print("=" * 70)
    header = f"  {'백본':15s}" + "".join(f"  {d[:8]:10s}" for d in JSONL_FILES)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for bname, bres in all_results.items():
        row = f"  {bname:15s}"
        for dname in JSONL_FILES:
            if dname in bres and "meta" in bres[dname]:
                row += f"  {bres[dname]['meta']['macro_f1']:.4f}    "
            else:
                row += f"  {'N/A':10s}"
        print(row)

    # 전체 저장
    summary_path = RESULT_DIR / f"backbone_eval_summary_{timestamp}.json"
    summary = {
        "timestamp": timestamp,
        "config": vars(args),
        "results": {
            b: {d: v["meta"] for d, v in bres.items()}
            for b, bres in all_results.items()
        }
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n  요약 저장: {summary_path}")


if __name__ == "__main__":
    main()
