#!/usr/bin/env python3
"""
ICWMV Backbone Transfer / Strength Ladder Evaluation
====================================================

목표:
  1. ICWMV가 MNV2 전용인지, 다른 generalist backbone에도 전이되는지 확인
  2. strong vs weak operating point에서 ICWMV gain이 얼마나 달라지는지 비교

비교 조합:
  - mnv2_nofinetune    : ImageNet-pretrained dual-stream + random forensic head
  - mnv2_strong        : strong MNV2 checkpoint
  - mnv2_weak          : weaker MNV2 checkpoint
  - clip_ft4_strong    : fine-tuned MobileCLIP-S2
  - clip_zeroshot_weak : zero-shot MobileCLIP-S2 (no fine-tuning)

모든 조합은 동일한 SpecM-v4와 결합하며, baseline generalist 성능과
ICWMV 결합 성능을 함께 보고한다.
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
BE_DIR = ROOT / "experiments" / "results" / "backbone_eval"
SPECM_DIR = ROOT / "experiments" / "results" / "specm_eval"
OUT_DIR = ROOT / "experiments" / "results" / "icwmv_backbone_transfer"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = ["base", "dsC", "opensdi", "aigenproxy"]
CLASSES = ["authentic", "manipulated", "ai_generated"]
CLS2IDX = {"authentic": 0, "manipulated": 1, "ai_generated": 2}
IDX2CLS = {0: "authentic", 1: "manipulated", 2: "ai_generated"}

BACKBONES = {
    "mnv2_nofinetune": {
        "kind": "mnv2_nofinetune_cache",
        "prefix": "mobilenetv2_dualstream_nofinetune",
        "label": "MNV2 ImageNet no-finetune",
    },
    "mnv2_strong": {
        "kind": "cached_jsonl",
        "prefix": "mobilenetv2_dualstream",
        "timestamp": "20260319_070725",
        "label": "MNV2 strong",
    },
    "mnv2_weak": {
        "kind": "cached_jsonl",
        "prefix": "mobilenetv2_dualstream",
        "timestamp": "20260319_064748",
        "label": "MNV2 weak",
    },
    "clip_ft4_strong": {
        "kind": "cached_jsonl",
        "prefix": "mobileclip_s2_finetuned",
        "timestamp": "20260319_061834",
        "label": "MobileCLIP-ft4 strong",
    },
    "clip_zeroshot_weak": {
        "kind": "clip_zeroshot_cache",
        "prefix": "mobileclip_s2_zeroshot_scored",
        "label": "MobileCLIP zero-shot weak",
    },
}


def load_jsonl(path: Path) -> List[Dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def latest_timestamp(prefix: str, ds: str) -> str | None:
    candidates = sorted(BE_DIR.glob(f"{prefix}_{ds}_*.jsonl"))
    if not candidates:
        return None

    marker = f"{prefix}_{ds}_"
    def key_fn(path: Path) -> str:
        stem = path.stem
        return stem[len(marker):]

    return max(candidates, key=key_fn).stem[len(marker):]


def find_specm_jsonl(ds_name: str) -> Path:
    candidates = sorted(SPECM_DIR.glob(f"specm_v4_{ds_name}_*.jsonl"))
    if not candidates:
        raise FileNotFoundError(f"Missing SpecM-v4 JSONL for {ds_name}")
    return candidates[-1]


def align_records(gen_recs: List[Dict], specm_recs: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    specm_map = {r["image_path"]: r for r in specm_recs}
    aligned_g, aligned_s = [], []
    for g in gen_recs:
        s = specm_map.get(g["image_path"])
        if s:
            aligned_g.append(g)
            aligned_s.append(s)
    return aligned_g, aligned_s


def macro_f1_from_preds(preds: np.ndarray, recs: List[Dict]) -> float:
    labels = np.array([CLS2IDX[r["true_label"]] for r in recs], dtype=np.int64)
    present = sorted(set(labels.tolist()))
    f1s: List[float] = []
    for c in present:
        tp = int(((preds == c) & (labels == c)).sum())
        fp = int(((preds == c) & (labels != c)).sum())
        fn = int(((preds != c) & (labels == c)).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        f1s.append(float(f1))
    return float(np.mean(f1s))


def binary_error_correction(preds: np.ndarray, gen_recs: List[Dict]) -> Dict:
    n_err = 0
    n_corr = 0
    patterns: Dict[str, Dict[str, float]] = {}

    for i, g in enumerate(gen_recs):
        if g["true_label"] == "ai_generated" or g["pred_label"] == "ai_generated":
            continue
        if g["pred_label"] == g["true_label"]:
            continue

        n_err += 1
        pat = f"{g['true_label']}→{g['pred_label']}"
        entry = patterns.setdefault(pat, {"total": 0, "corrected": 0})
        entry["total"] += 1
        if IDX2CLS[int(preds[i])] == g["true_label"]:
            n_corr += 1
            entry["corrected"] += 1

    for entry in patterns.values():
        entry["rate"] = round(entry["corrected"] / max(entry["total"], 1), 4)

    return {
        "n_errors": int(n_err),
        "n_corrected": int(n_corr),
        "rate": round(n_corr / max(n_err, 1), 4),
        "patterns": patterns,
    }


def fuse_icwmv(gen_recs: List[Dict], specm_recs: List[Dict]) -> np.ndarray:
    preds: List[int] = []
    for g, s in zip(gen_recs, specm_recs):
        p_aigen = float(g["scores"].get("ai_generated", 0.0))
        if g["pred_label"] == "ai_generated" or p_aigen > 0.5:
            preds.append(CLS2IDX["ai_generated"])
            continue

        p_auth = float(g["scores"]["authentic"])
        p_manip = float(g["scores"]["manipulated"])
        total = p_auth + p_manip
        if total > 0:
            p_auth /= total
            p_manip /= total

        w_gen = 1.0 / max(float(g["confidence"]), 1e-3)
        w_spec = 1.0 / max(float(s["confidence"]), 1e-3)
        combined = (
            w_gen * np.array([p_auth, p_manip, 0.0], dtype=np.float32)
            + w_spec * np.array(
                [float(s["authentic_score"]), float(s["manip_score"]), 0.0],
                dtype=np.float32,
            )
        )
        preds.append(int(combined[:2].argmax()))
    return np.array(preds, dtype=np.int64)


def normalize_generalist_records(records: List[Dict]) -> List[Dict]:
    normalized = []
    for r in records:
        pred_label = r.get("pred_label", r.get("verdict"))
        if pred_label is None:
            raise KeyError("generalist record is missing pred_label/verdict")
        scores = r.get("scores")
        if not isinstance(scores, dict):
            raise KeyError("generalist record is missing scores dict")
        normalized.append({
            "image_path": r["image_path"],
            "true_label": r["true_label"],
            "pred_label": pred_label,
            "confidence": float(r["confidence"]),
            "scores": {
                "authentic": float(scores.get("authentic", 0.0)),
                "manipulated": float(scores.get("manipulated", 0.0)),
                "ai_generated": float(scores.get("ai_generated", 0.0)),
            },
        })
    return normalized


def ensure_clip_zeroshot_cache() -> str:
    existing = {ds: latest_timestamp("mobileclip_s2_zeroshot_scored", ds) for ds in DATASETS}
    if all(existing.values()):
        timestamps = {ts for ts in existing.values() if ts}
        if len(timestamps) == 1:
            return timestamps.pop()

    import torch
    import torch.nn.functional as F
    import open_clip
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        ),
    ])
    prompts = {
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

    class EvalDataset(Dataset):
        def __init__(self, records: List[Dict]):
            self.records = records

        def __len__(self) -> int:
            return len(self.records)

        def __getitem__(self, idx: int):
            rec = self.records[idx]
            img = Image.open(ROOT / rec["image_path"]).convert("RGB")
            return transform(img), rec["image_path"], rec["true_label"]

    print("[clip_zeroshot] loading zero-shot MobileCLIP-S2...")
    model, _, _ = open_clip.create_model_and_transforms(
        "MobileCLIP-S2",
        pretrained="datacompdr",
        cache_dir=str(ROOT / "weights" / "mobileclip"),
    )
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer("MobileCLIP-S2")

    with torch.no_grad():
        text_features = []
        for cls in CLASSES:
            tokens = tokenizer(prompts[cls]).to(device)
            feats = model.encode_text(tokens)
            feats = F.normalize(feats, dim=-1).mean(dim=0)
            text_features.append(F.normalize(feats, dim=0))
        text_features = torch.stack(text_features, dim=0)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    for ds in DATASETS:
        seed_path = BE_DIR / f"mobileclip_s2_{ds}_20260319_055541.jsonl"
        if not seed_path.exists():
            raise FileNotFoundError(f"Missing seed weak MobileCLIP JSONL: {seed_path}")
        seed_records = load_jsonl(seed_path)
        loader = DataLoader(EvalDataset(seed_records), batch_size=64, shuffle=False, num_workers=4)
        out_records: List[Dict] = []
        for imgs, paths, true_labels in loader:
            imgs = imgs.to(device)
            with torch.no_grad():
                img_feat = model.encode_image(imgs)
                img_feat = F.normalize(img_feat, dim=-1)
                logits = img_feat @ text_features.t()
                probs = F.softmax(logits / 0.07, dim=-1).cpu().numpy()

            for i in range(len(paths)):
                pred_idx = int(probs[i].argmax())
                out_records.append({
                    "image_path": paths[i],
                    "true_label": true_labels[i],
                    "pred_label": CLASSES[pred_idx],
                    "confidence": float(probs[i][pred_idx]),
                    "scores": {
                        CLASSES[j]: float(probs[i][j]) for j in range(3)
                    },
                })

        out_path = BE_DIR / f"mobileclip_s2_zeroshot_scored_{ds}_{ts}.jsonl"
        with open(out_path, "w") as f:
            for row in out_records:
                f.write(json.dumps(row) + "\n")
        print(f"[clip_zeroshot] saved {out_path.name} ({len(out_records)} rows)")
    return ts


def ensure_mnv2_nofinetune_cache() -> str:
    existing = {ds: latest_timestamp("mobilenetv2_dualstream_nofinetune", ds) for ds in DATASETS}
    if all(existing.values()):
        timestamps = {ts for ts in existing.values() if ts}
        if len(timestamps) == 1:
            return timestamps.pop()

    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    from PIL import Image
    from torchvision import transforms

    from train_mobilenetv2_dualstream import DualStreamMobileNetV2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    class EvalDataset(Dataset):
        def __init__(self, records: List[Dict]):
            self.records = records

        def __len__(self) -> int:
            return len(self.records)

        def __getitem__(self, idx: int):
            rec = self.records[idx]
            img = Image.open(ROOT / rec["image_path"]).convert("RGB")
            return transform(img), rec["image_path"], rec["true_label"], rec.get("sub_type", "")

    print("[mnv2_nofinetune] loading ImageNet-pretrained dual-stream with random head...")
    torch.manual_seed(42)
    np.random.seed(42)
    model = DualStreamMobileNetV2(pretrained=True).to(device).eval()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    for ds in DATASETS:
        seed_path = BE_DIR / f"mobilenetv2_dualstream_{ds}_20260319_064748.jsonl"
        if not seed_path.exists():
            raise FileNotFoundError(f"Missing seed MNV2 JSONL: {seed_path}")
        seed_records = load_jsonl(seed_path)
        loader = DataLoader(EvalDataset(seed_records), batch_size=64, shuffle=False, num_workers=4)
        out_records: List[Dict] = []
        for imgs, paths, true_labels, sub_types in loader:
            imgs = imgs.to(device)
            with torch.no_grad():
                probs = F.softmax(model(imgs), dim=-1).cpu().numpy()
            for i in range(len(paths)):
                pred_idx = int(probs[i].argmax())
                out_records.append({
                    "image_path": paths[i],
                    "true_label": true_labels[i],
                    "pred_label": CLASSES[pred_idx],
                    "confidence": float(probs[i][pred_idx]),
                    "scores": {
                        CLASSES[j]: float(probs[i][j]) for j in range(3)
                    },
                    "sub_type": sub_types[i],
                })

        out_path = BE_DIR / f"mobilenetv2_dualstream_nofinetune_{ds}_{ts}.jsonl"
        with open(out_path, "w") as f:
            for row in out_records:
                f.write(json.dumps(row) + "\n")
        print(f"[mnv2_nofinetune] saved {out_path.name} ({len(out_records)} rows)")
    return ts


def load_backbone_records(backbone_key: str) -> Dict[str, List[Dict]]:
    cfg = BACKBONES[backbone_key]
    if cfg["kind"] == "cached_jsonl":
        ts = cfg["timestamp"]
    elif cfg["kind"] == "clip_zeroshot_cache":
        ts = ensure_clip_zeroshot_cache()
    elif cfg["kind"] == "mnv2_nofinetune_cache":
        ts = ensure_mnv2_nofinetune_cache()
    else:
        raise ValueError(f"Unsupported backbone kind: {cfg['kind']}")

    out = {}
    for ds in DATASETS:
        path = BE_DIR / f"{cfg['prefix']}_{ds}_{ts}.jsonl"
        out[ds] = normalize_generalist_records(load_jsonl(path))
    return out


def evaluate_backbone(backbone_key: str) -> Dict:
    label = BACKBONES[backbone_key]["label"]
    gen_data = load_backbone_records(backbone_key)

    baseline_by_ds = {}
    icwmv_by_ds = {}
    baseline_vals = []
    icwmv_vals = []
    corr_vals = []

    for ds in DATASETS:
        specm = load_jsonl(find_specm_jsonl(ds))
        gen_aligned, specm_aligned = align_records(gen_data[ds], specm)
        baseline_preds = np.array([CLS2IDX[r["pred_label"]] for r in gen_aligned], dtype=np.int64)
        baseline_f1 = macro_f1_from_preds(baseline_preds, gen_aligned)
        icwmv_preds = fuse_icwmv(gen_aligned, specm_aligned)
        icwmv_f1 = macro_f1_from_preds(icwmv_preds, gen_aligned)
        corr = binary_error_correction(icwmv_preds, gen_aligned)

        baseline_by_ds[ds] = round(baseline_f1, 4)
        icwmv_by_ds[ds] = {
            "macro_f1": round(icwmv_f1, 4),
            "correction_rate": corr["rate"],
            "n_common": len(gen_aligned),
            "err_corr": corr,
        }
        baseline_vals.append(baseline_f1)
        icwmv_vals.append(icwmv_f1)
        corr_vals.append(corr["rate"])

    return {
        "label": label,
        "baseline": {
            "avg_macro_f1": round(float(np.mean(baseline_vals)), 4),
            "per_dataset": baseline_by_ds,
        },
        "icwmv_specm_v4": {
            "avg_macro_f1": round(float(np.mean(icwmv_vals)), 4),
            "avg_correction_rate": round(float(np.mean(corr_vals)), 4),
            "delta_vs_baseline": round(float(np.mean(icwmv_vals) - np.mean(baseline_vals)), 4),
            "per_dataset": icwmv_by_ds,
        },
    }


def print_summary(results: Dict[str, Dict]) -> None:
    print("\n" + "=" * 94)
    print("  ICWMV Backbone Transfer / Strength Ladder")
    print("=" * 94)
    print(f"  {'Backbone':26s} | {'Baseline':>8s} | {'ICWMV':>8s} | {'ΔF1':>7s} | {'Corr':>7s}")
    print(f"  {'-' * 88}")
    for key, res in results.items():
        base = res["baseline"]["avg_macro_f1"]
        fused = res["icwmv_specm_v4"]["avg_macro_f1"]
        delta = res["icwmv_specm_v4"]["delta_vs_baseline"]
        corr = res["icwmv_specm_v4"]["avg_correction_rate"]
        print(f"  {res['label']:26s} | {base:8.4f} | {fused:8.4f} | {delta:+7.4f} | {corr:7.4f}")

    print("\n  [Family comparison]")
    families = [
        ("MNV2", "mnv2_strong", "mnv2_weak"),
        ("MobileCLIP", "clip_ft4_strong", "clip_zeroshot_weak"),
    ]
    for family, strong_key, weak_key in families:
        strong = results[strong_key]["icwmv_specm_v4"]
        weak = results[weak_key]["icwmv_specm_v4"]
        print(
            f"  {family:10s}: strong={strong['avg_macro_f1']:.4f} "
            f"weak={weak['avg_macro_f1']:.4f}  "
            f"Δstrong-weak={strong['avg_macro_f1'] - weak['avg_macro_f1']:+.4f}"
        )


def main() -> None:
    results = {}
    for backbone_key in BACKBONES:
        print(f"\n[run] {backbone_key}")
        results[backbone_key] = evaluate_backbone(backbone_key)

    print_summary(results)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUT_DIR / f"icwmv_backbone_transfer_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  saved: {out_path}")


if __name__ == "__main__":
    main()
