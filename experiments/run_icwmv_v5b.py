#!/usr/bin/env python3
"""
SpecM-v5b JSONL 생성 + ICWMV 평가
=====================================

1단계: v5b 모델로 4-DS 추론 → JSONL (v4와 동일 포맷)
2단계: MNV2 + SpecM-v5b (2-model) ICWMV 평가
3단계: v4 결과와 직접 비교

실행:
  .venv-qwen/bin/python experiments/run_icwmv_v5b.py
"""

from __future__ import annotations

import json
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

warnings.filterwarnings("ignore")

ROOT    = Path(__file__).resolve().parents[1]
DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SE      = ROOT / "experiments" / "results" / "specialist_eval"
BE      = ROOT / "experiments" / "results" / "backbone_eval"
OUT_DIR = ROOT / "experiments" / "results" / "icwmv"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLIP_MEAN = [0.48145466, 0.4578275,  0.40821073]
CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]
LABEL_MAP = {"authentic": 0, "manipulated": 1, "ai_generated": 2}
CLASSES   = ["authentic", "manipulated", "ai_generated"]

EVAL_DATASETS = {
    "base":       "experiments/results/phase2_patha_scale500_gain_predictor/patha_agent_outputs_20260304_080157.jsonl",
    "dsC":        "experiments/results/phase2_patha_case3_scale300_dsC/patha_agent_outputs_20260303_105005.jsonl",
    "opensdi":    "experiments/results/phase2_patha_case3_opensdi_scale300/patha_agent_outputs_fixed_seed42.jsonl",
    "aigenproxy": "experiments/results/phase2_patha_case3_aigenproxy_scale300/patha_agent_outputs_fixed_seed42.jsonl",
}

MNV2_FILES = {
    ds: BE / f"mobilenetv2_dualstream_{ds}_20260319_070725.jsonl"
    for ds in EVAL_DATASETS
}


# ── 모델 정의 (train_specialist_m_v5b.py와 동일) ─────────────────────────
def _normalize(x, mean, std):
    m = torch.tensor(mean, device=x.device, dtype=x.dtype).view(1,-1,1,1)
    s = torch.tensor(std,  device=x.device, dtype=x.dtype).view(1,-1,1,1)
    return (x - m) / s


class SRMExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        f1 = [[0,0,0,0,0],[0,-1,2,-1,0],[0,2,-4,2,0],[0,-1,2,-1,0],[0,0,0,0,0]]
        f2 = [[-1,2,-2,2,-1],[2,-6,8,-6,2],[-2,8,-12,8,-2],[2,-6,8,-6,2],[-1,2,-2,2,-1]]
        f3 = [[0,0,0,0,0],[0,0,0,0,0],[0,1,-2,1,0],[0,0,0,0,0],[0,0,0,0,0]]
        q  = torch.tensor([4.0,12.0,2.0]).view(3,1,1,1)
        k  = torch.tensor([f1,f2,f3], dtype=torch.float32).unsqueeze(1) / q
        self.register_buffer("kernels", k)
    def forward(self, x):
        gray = x.mean(dim=1, keepdim=True)
        res  = F.conv2d(gray, self.kernels, padding=2)
        return (res.clamp(-2.,2.) + 2.) / 4.


class SRMLightCNN(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3,32,3,stride=2,padding=1,bias=False), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32,32,3,stride=2,padding=1,groups=32,bias=False), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32,64,1,bias=False), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64,64,3,stride=2,padding=1,groups=64,bias=False), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64,128,1,bias=False), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128,128,3,stride=2,padding=1,groups=128,bias=False), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128,out_dim,1,bias=False), nn.BatchNorm2d(out_dim), nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1))
        self.out_dim = out_dim
    def forward(self, x):
        return self.net(x).flatten(1)


class SpecialistMv5b(nn.Module):
    def __init__(self, clip_encoder, dropout=0.3):
        super().__init__()
        self.clip_encoder = clip_encoder
        for p in self.clip_encoder.parameters():
            p.requires_grad_(False)
        self.srm     = SRMExtractor()
        self.srm_cnn = SRMLightCNN(out_dim=128)
        fused = 640
        self.head = nn.Sequential(
            nn.LayerNorm(fused), nn.Linear(fused,256), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(256,64), nn.GELU(),
            nn.Dropout(dropout*0.5), nn.Linear(64,2))

    def forward(self, x):
        x_clip    = _normalize(x, CLIP_MEAN, CLIP_STD)
        clip_feat = self.clip_encoder(x_clip)
        srm_out   = self.srm(x)
        srm_norm  = _normalize(srm_out, [0.5]*3, [0.5]*3)
        srm_feat  = self.srm_cnn(srm_norm)
        return self.head(torch.cat([clip_feat, srm_feat], dim=1))


def load_v5b_model():
    ckpt_path = ROOT / "weights" / "specialist_m_v5b" / "specialist_m_v5b_best.pth"
    try:
        import open_clip
        model_clip, _, _ = open_clip.create_model_and_transforms(
            "MobileCLIP-S2", pretrained="datacompdr")
        encoder = model_clip.visual

        # ft4 가중치 시도
        ft4_path = ROOT / "weights" / "mobileclip_forensics" / "mobileclip_s2_forensics_ft4.pth"
        if ft4_path.exists():
            sd_ft4 = torch.load(ft4_path, map_location="cpu", weights_only=False)
            sd_ft4 = sd_ft4.get("model_state_dict", sd_ft4)
            vis_sd = {k.replace("visual.", ""): v
                      for k, v in sd_ft4.items() if k.startswith("visual.")}
            if vis_sd:
                encoder.load_state_dict(vis_sd, strict=False)
    except Exception as e:
        print(f"  [에러] MobileCLIP 로드 실패: {e}")
        return None

    model = SpecialistMv5b(clip_encoder=encoder.to(DEVICE)).to(DEVICE)
    ckpt  = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
    model.eval()
    print(f"  v5b 로드 완료: {ckpt_path.name}")
    return model


# ── 데이터셋 ─────────────────────────────────────────────────────────────
VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
])

class EvalDataset(Dataset):
    def __init__(self, records):
        self.records = records
    def __len__(self):
        return len(self.records)
    def __getitem__(self, idx):
        rec   = self.records[idx]
        path  = ROOT / rec["image_path"]
        label = LABEL_MAP[rec["true_label"]]
        try:
            img    = Image.open(path).convert("RGB")
            tensor = VAL_TRANSFORM(img)
        except Exception:
            tensor = torch.zeros(3,256,256)
        return tensor, label, rec["image_path"], rec["true_label"]


def load_records(jsonl_path: str) -> List[Dict]:
    records = []
    full = ROOT / jsonl_path
    if not full.exists():
        return records
    with open(full) as f:
        for line in f:
            rec = json.loads(line.strip())
            if rec.get("true_label") not in LABEL_MAP:
                continue
            if (ROOT / rec["image_path"]).exists():
                records.append(rec)
    return records


# ── v5b 추론 → JSONL ─────────────────────────────────────────────────────
@torch.no_grad()
def infer_v5b(model, records: List[Dict], ts: str) -> Tuple[Path, Dict]:
    ds    = EvalDataset(records)
    loader = DataLoader(ds, batch_size=64, shuffle=False,
                        num_workers=2, pin_memory=True)
    results = []
    for imgs, labels, paths, true_labels in loader:
        imgs = imgs.to(DEVICE)
        out  = model(imgs)
        prob = torch.softmax(out, dim=1)  # (B, 2): [auth, manip]
        for i in range(len(paths)):
            auth_s  = float(prob[i, 0])
            manip_s = float(prob[i, 1])
            pred    = "manipulated" if manip_s > auth_s else "authentic"
            results.append({
                "image_path":      paths[i],
                "true_label":      true_labels[i],
                "pred_label":      pred,
                "authentic_score": auth_s,
                "manip_score":     manip_s,
                "confidence":      max(auth_s, manip_s),
            })
    # 경로별 dict
    rec_map = {r["image_path"]: r for r in results}
    return results, rec_map


# ── ICWMV 퓨전 ───────────────────────────────────────────────────────────
def fuse_mnv2_specm(mnv2: dict, spec_m: dict, w_spec: float = 1.0):
    """MNV2(3-class) + SpecM-v5b(binary) 2-model ICWMV"""
    w_mnv2 = 1.0

    auth_sum  = w_mnv2 * mnv2["scores"]["authentic"]
    manip_sum = w_mnv2 * mnv2["scores"]["manipulated"]
    aigen_sum = w_mnv2 * mnv2["scores"]["ai_generated"]
    auth_w = manip_w = aigen_w = w_mnv2

    # SpecM-v5b: auth/manip 기여 (aigen 없음)
    auth_sum  += w_spec * spec_m["authentic_score"]
    manip_sum += w_spec * spec_m["manip_score"]
    auth_w    += w_spec
    manip_w   += w_spec

    s = np.array([auth_sum/auth_w, manip_sum/manip_w, aigen_sum/aigen_w])
    probs = s / s.sum()
    return CLASSES[int(np.argmax(probs))], probs


def eval_icwmv(ds_name: str, v5b_map: Dict, w_spec: float) -> Optional[Dict]:
    mnv2_path = MNV2_FILES[ds_name]
    if not mnv2_path or not mnv2_path.exists():
        return None

    mnv2_recs = {}
    with open(mnv2_path) as f:
        for line in f:
            r = json.loads(line.strip())
            mnv2_recs[r["image_path"]] = r

    common = sorted(set(mnv2_recs) & set(v5b_map))
    if not common:
        return None

    y_true, y_pred_fused, y_pred_mnv2, y_pred_v5b = [], [], [], []
    for key in common:
        true = mnv2_recs[key]["true_label"]
        pred_fused, _ = fuse_mnv2_specm(mnv2_recs[key], v5b_map[key], w_spec)
        y_true.append(true)
        y_pred_fused.append(pred_fused)
        y_pred_mnv2.append(mnv2_recs[key]["pred_label"])
        y_pred_v5b.append("manipulated" if v5b_map[key]["manip_score"] > 0.5 else "authentic")

    def metrics(yt, yp):
        n      = len(yt)
        acc    = sum(a == b for a, b in zip(yt, yp)) / n
        per    = {}
        recall_list = []
        for cls in CLASSES:
            mask = [t == cls for t in yt]
            if sum(mask) == 0:
                per[cls] = 0.0
                continue
            r = sum(a == b for a, b, m in zip(yt, yp, mask) if m) / sum(mask)
            per[cls] = round(r, 4)
            recall_list.append(r)
        macro = np.mean(recall_list) if recall_list else 0.0
        return {"acc": round(acc, 4), "macro_recall": round(float(macro), 4), "per_class": per, "n": n}

    return {
        "fused":  metrics(y_true, y_pred_fused),
        "mnv2":   metrics(y_true, y_pred_mnv2),
        "v5b":    metrics(y_true, y_pred_v5b),
        "n":      len(common),
    }


# ── 메인 ─────────────────────────────────────────────────────────────────
def main():
    print(f"Device: {DEVICE}")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. v5b 모델 로드
    print("\n[1/3] SpecM-v5b 모델 로드...")
    model = load_v5b_model()
    if model is None:
        return

    # 2. 4-DS 추론 → JSONL 저장
    print("\n[2/3] 4-DS 추론 (v5b)...")
    all_v5b_maps = {}
    for ds_name, ds_path in EVAL_DATASETS.items():
        records = load_records(ds_path)
        if not records:
            print(f"  {ds_name}: 레코드 없음")
            continue
        results, rec_map = infer_v5b(model, records, ts)
        all_v5b_maps[ds_name] = rec_map

        # JSONL 저장 (v4와 동일 포맷)
        out_path = SE / f"specialist_m_v5b_{ds_name}_{ts}.jsonl"
        with open(out_path, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        print(f"  {ds_name}: {len(results)}장 → {out_path.name}")

    # 3. ICWMV 평가 (w_spec grid: 0.5, 1.0, 2.0)
    print("\n[3/3] ICWMV 평가 (MNV2 + SpecM-v5b)...")

    # v4 best 결과 참조용 (비교 baseline)
    V4_ICWMV_AVG = 0.9658  # AGENTS.md 기록값

    best_w    = 1.0
    best_avg  = 0.0
    all_runs  = {}

    for w_spec in [0.5, 1.0, 2.0, 3.0]:
        ds_macros = []
        ds_results = {}
        for ds_name in EVAL_DATASETS:
            if ds_name not in all_v5b_maps:
                continue
            r = eval_icwmv(ds_name, all_v5b_maps[ds_name], w_spec)
            if r is None:
                continue
            ds_results[ds_name] = r
            ds_macros.append(r["fused"]["macro_recall"])

        avg = float(np.mean(ds_macros)) if ds_macros else 0.0
        all_runs[f"w{w_spec}"] = {"avg_macro_recall": round(avg, 4), "datasets": ds_results}

        if avg > best_avg:
            best_avg = avg
            best_w   = w_spec

        print(f"\n  [w_spec={w_spec}] 4-DS avg macro_recall = {avg:.4f}")
        for ds_name, r in ds_results.items():
            fused = r["fused"]
            mnv2  = r["mnv2"]
            print(f"    {ds_name}: fused={fused['macro_recall']:.4f} "
                  f"(mnv2={mnv2['macro_recall']:.4f}) "
                  f"[auth={fused['per_class'].get('authentic',0):.3f} "
                  f"manip={fused['per_class'].get('manipulated',0):.3f} "
                  f"aigen={fused['per_class'].get('ai_generated',0):.3f}]")

    print(f"\n{'='*65}")
    print(f"최적 w_spec = {best_w}")
    print(f"MNV2 + SpecM-v5b ICWMV avg macro_recall = {best_avg:.4f}")
    print(f"비교: MNV2 + SpecM-v4 ICWMV avg           = {V4_ICWMV_AVG:.4f}")
    delta = best_avg - V4_ICWMV_AVG
    mark  = "▲" if delta >= 0 else "▼"
    print(f"차이: {mark} {abs(delta):.4f} ({'개선' if delta >= 0 else '하락'})")
    print(f"{'='*65}")

    # 결과 저장
    out = {
        "timestamp": ts,
        "arch": "MNV2 + SpecM-v5b (MobileCLIP+SRM_LightCNN)",
        "best_w_spec": best_w,
        "best_avg_macro_recall": round(best_avg, 4),
        "v4_icwmv_avg_reference": V4_ICWMV_AVG,
        "delta_vs_v4": round(best_avg - V4_ICWMV_AVG, 4),
        "runs": all_runs,
    }
    out_path = OUT_DIR / f"icwmv_v5b_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n결과 저장: {out_path}")


if __name__ == "__main__":
    main()
