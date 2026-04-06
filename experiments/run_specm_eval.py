#!/usr/bin/env python3
"""
SpecM 계열 모델 inference 실행기
================================
backbone_eval JSONL의 image_path를 이용해 SpecM-v4 또는 SpecM-Comp를 실행하고
manip_score, authentic_score, confidence를 저장한다.

실행:
  # SpecM-v4 전체 4개 데이터셋
  .venv-qwen/bin/python experiments/run_specm_eval.py --model v4

  # SpecM-Comp γ=1 전체 4개 데이터셋
  .venv-qwen/bin/python experiments/run_specm_eval.py --model comp_g1

  # 특정 데이터셋만
  .venv-qwen/bin/python experiments/run_specm_eval.py --model v4 --datasets base dsC
"""

from __future__ import annotations
import argparse, json, sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ROOT   = Path(__file__).resolve().parents[1]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sys.path.insert(0, str(ROOT / "experiments"))

BE_DIR  = ROOT / "experiments" / "results" / "backbone_eval"
TS      = "20260319_070725"
DATASETS = ["base", "dsC", "opensdi", "aigenproxy"]

CKPTS = {
    "v4":        ROOT / "weights" / "specialist_m_v4" / "specialist_m_v4_best.pth",
    "comp_g1":   ROOT / "weights" / "specialist_m_complementary" / "gamma1.0_wmax10" / "best.pth",
    "comp_g2":   ROOT / "weights" / "specialist_m_complementary" / "gamma2.0_wmax10" / "best.pth",
    "comp_g3":   ROOT / "weights" / "specialist_m_complementary" / "gamma3.0_wmax10" / "best.pth",
    "comp_wmax1":  ROOT / "weights" / "specialist_m_complementary" / "gamma1.0_wmax1" / "best.pth",
    "comp_wmax5":  ROOT / "weights" / "specialist_m_complementary" / "gamma1.0_wmax5" / "best.pth",
    "comp_wmax20": ROOT / "weights" / "specialist_m_complementary" / "gamma1.0_wmax20" / "best.pth",
    "comp_noTS":   ROOT / "weights" / "specialist_m_complementary" / "gamma1.0_wmax10_noTS" / "best.pth",
}

IDX2LABEL = {0: "authentic", 1: "manipulated"}
VAL_TF = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])


def load_specm(ckpt_path: Path):
    from train_specialist_m_complementary import SpecialistM
    model = SpecialistM(pretrained=False).to(DEVICE)
    sd = torch.load(ckpt_path, map_location="cpu")
    sd = sd.get("model_state_dict", sd)
    model.load_state_dict(sd)
    model.eval()
    return model


class ImagePathDataset(Dataset):
    def __init__(self, records):
        self.records = records
    def __len__(self): return len(self.records)
    def __getitem__(self, idx):
        rec = self.records[idx]
        path = ROOT / rec["image_path"]
        try:
            img = Image.open(path).convert("RGB")
            tensor = VAL_TF(img)
        except Exception:
            tensor = torch.zeros(3, 224, 224)
        return tensor, rec["image_path"], rec["true_label"]


@torch.no_grad()
def run_inference(model, records, batch_size=128):
    ds = ImagePathDataset(records)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=(DEVICE.type == "cuda"))
    results = []
    for imgs, paths, true_labels in loader:
        imgs  = imgs.to(DEVICE)
        probs = F.softmax(model(imgs), dim=-1).cpu().numpy()
        for path, true_label, prob in zip(paths, true_labels, probs):
            pred_idx = int(prob.argmax())
            results.append({
                "image_path":      path,
                "true_label":      true_label,
                "pred_label":      IDX2LABEL[pred_idx],
                "manip_score":     float(prob[1]),
                "authentic_score": float(prob[0]),
                "confidence":      float(prob.max()),
            })
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    type=str, required=True,
                        choices=list(CKPTS.keys()),
                        help="모델 키 (v4, comp_g1, comp_g2, ...)")
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    ckpt_path = CKPTS[args.model]
    if not ckpt_path.exists():
        print(f"체크포인트 없음: {ckpt_path}")
        return

    print(f"모델: {args.model} → {ckpt_path.name}")
    model = load_specm(ckpt_path)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    out_dir = ROOT / "experiments" / "results" / "specm_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for ds_name in args.datasets:
        jsonl_path = BE_DIR / f"mobilenetv2_dualstream_{ds_name}_{TS}.jsonl"
        if not jsonl_path.exists():
            print(f"  [{ds_name}] backbone_eval JSONL 없음"); continue

        records = []
        with open(jsonl_path) as f:
            for line in f:
                r = json.loads(line)
                records.append({"image_path": r["image_path"], "true_label": r["true_label"]})

        results = run_inference(model, records, args.batch_size)

        # 저장
        out_path = out_dir / f"specm_{args.model}_{ds_name}_{ts}.jsonl"
        with open(out_path, "w") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        # 메트릭 계산
        binary  = [r for r in results if r["true_label"] in ("authentic", "manipulated")]
        correct = sum(1 for r in binary if r["pred_label"] == r["true_label"])
        acc     = correct / len(binary) if binary else 0

        tp = sum(1 for r in binary if r["true_label"] == "manipulated" and r["pred_label"] == "manipulated")
        fp = sum(1 for r in binary if r["true_label"] != "manipulated" and r["pred_label"] == "manipulated")
        fn = sum(1 for r in binary if r["true_label"] == "manipulated" and r["pred_label"] != "manipulated")
        prec = tp / max(tp+fp, 1)
        rec  = tp / max(tp+fn, 1)
        f1   = 2*prec*rec / max(prec+rec, 1e-8)

        summary[ds_name] = {
            "n": len(results), "n_binary": len(binary),
            "accuracy": round(acc, 4),
            "manip_f1": round(f1, 4),
            "manip_recall": round(rec, 4),
            "manip_precision": round(prec, 4),
            "jsonl": str(out_path),
        }
        print(f"  [{ds_name}] n={len(results)} | acc={acc:.4f} | "
              f"manip_f1={f1:.4f} | recall={rec:.4f} | prec={prec:.4f}")

    sum_path = out_dir / f"specm_{args.model}_summary_{ts}.json"
    with open(sum_path, "w") as f:
        json.dump({"model": args.model, "results": summary}, f, indent=2)
    print(f"\n저장: {sum_path}")


if __name__ == "__main__":
    main()
