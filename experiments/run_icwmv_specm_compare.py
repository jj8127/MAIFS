#!/usr/bin/env python3
"""
ICWMV + SpecM 모델 비교 (LOO-CD)
==================================
EWCT 학습 SpecM이 ICWMV 성능을 올리는가?
  ICWMV + v4       (baseline SpecM)
  ICWMV + comp_g1  (EWCT γ=1)
  ICWMV + comp_noTS(EWCT γ=1, no-TS)

두 MNV2 버전에서 모두 실행:
  강한 MNV2: 20260319_070725 (avg F1≈0.958)
  약한 MNV2: 20260319_064748 (avg F1≈0.842)
"""
from __future__ import annotations
import json, warnings
import numpy as np
from collections import defaultdict
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT     = Path(__file__).resolve().parents[1]
BE_DIR   = ROOT / "experiments" / "results" / "backbone_eval"
COMP_DIR = ROOT / "experiments" / "results" / "specm_complementary_eval"
SPECM_DIR= ROOT / "experiments" / "results" / "specm_eval"

DATASETS = ["base", "dsC", "opensdi", "aigenproxy"]
CLS2IDX  = {"authentic":0, "manipulated":1, "ai_generated":2}
IDX2CLS  = {0:"authentic", 1:"manipulated", 2:"ai_generated"}

def load_mnv2(ds, ts):
    with open(BE_DIR / f"mobilenetv2_dualstream_{ds}_{ts}.jsonl") as f:
        return [json.loads(l) for l in f]

def find_specm(sm, ds):
    tag_map = {"comp_g1":"gamma1.0_wmax10","comp_noTS":"gamma1.0_wmax10_noTS","v4":None}
    if sm == "v4":
        c = sorted(SPECM_DIR.glob(f"specm_v4_{ds}_*.jsonl"))
    else:
        c = sorted(COMP_DIR.glob(f"specm_comp_{tag_map[sm]}_{ds}_*.jsonl"))
    return c[-1] if c else None

def load_specm(sm, ds):
    p = find_specm(sm, ds)
    if not p: return None
    with open(p) as f: return [json.loads(l) for l in f]

def align(mrecs, srecs):
    sm = {r["image_path"]: r for r in srecs}
    am, as_ = [], []
    for m in mrecs:
        s = sm.get(m["image_path"])
        if s: am.append(m); as_.append(s)
    return am, as_

def fuse_icwmv(mrecs, srecs):
    preds = []
    for m, s in zip(mrecs, srecs):
        p_aigen = m["scores"].get("ai_generated", 0.0)
        if m["pred_label"] == "ai_generated" or p_aigen > 0.5:
            preds.append(CLS2IDX["ai_generated"]); continue
        pa = m["scores"]["authentic"]; pm = m["scores"]["manipulated"]
        tot = pa + pm
        if tot > 0: pa /= tot; pm /= tot
        w_m = 1.0 / max(m["confidence"], 1e-3)
        w_s = 1.0 / max(s["confidence"], 1e-3)
        comb = w_m * np.array([pa,pm,0.]) + w_s * np.array([s["authentic_score"],s["manip_score"],0.])
        preds.append(int(comb[:2].argmax()))
    return np.array(preds)

def eval_preds(preds, mrecs):
    labels = np.array([CLS2IDX[m["true_label"]] for m in mrecs])
    present = sorted(set(labels.tolist()))
    f1s = []
    for c in present:
        tp=((preds==c)&(labels==c)).sum(); fp=((preds==c)&(labels!=c)).sum()
        fn=((preds!=c)&(labels==c)).sum(); pr=tp/max(tp+fp,1); rc=tp/max(tp+fn,1)
        f1s.append(2*pr*rc/max(pr+rc,1e-8))
    n_err=n_corr=0
    patterns=defaultdict(lambda:{"total":0,"corrected":0})
    for i,m in enumerate(mrecs):
        if m["true_label"]=="ai_generated" or m["pred_label"]=="ai_generated": continue
        if m["pred_label"]!=m["true_label"]:
            n_err+=1; pat=f"{m['true_label']}→{m['pred_label']}"; patterns[pat]["total"]+=1
            if IDX2CLS.get(int(preds[i]))==m["true_label"]: n_corr+=1; patterns[pat]["corrected"]+=1
    for p in patterns.values(): p["rate"]=round(p["corrected"]/max(p["total"],1),4)
    return {"macro_f1":round(float(np.mean(f1s)),4),
            "err_corr":{"n_errors":n_err,"n_corrected":n_corr,
                        "rate":round(n_corr/max(n_err,1),4),"patterns":dict(patterns)}}

MNV2_VERSIONS = {
    "strong (070725)": "20260319_070725",
    "weak   (064748)": "20260319_064748",
}
SPECM_MODELS = ["v4", "comp_g1", "comp_noTS"]

all_out = {}
for mnv2_label, ts in MNV2_VERSIONS.items():
    print(f"\n{'='*68}")
    print(f"  MNV2: {mnv2_label}")
    print(f"{'='*68}")

    # MNV2-only baseline
    mnv2_data = {ds: load_mnv2(ds, ts) for ds in DATASETS}
    mnv2_f1s = []
    for ds in DATASETS:
        recs = mnv2_data[ds]
        labels = np.array([CLS2IDX[m["true_label"]] for m in recs])
        preds  = np.array([CLS2IDX[m["pred_label"]]  for m in recs])
        present = sorted(set(labels.tolist()))
        f1s = []
        for c in present:
            tp=((preds==c)&(labels==c)).sum(); fp=((preds==c)&(labels!=c)).sum()
            fn=((preds!=c)&(labels==c)).sum(); pr=tp/max(tp+fp,1); rc=tp/max(tp+fn,1)
            f1s.append(2*pr*rc/max(pr+rc,1e-8))
        mnv2_f1s.append(np.mean(f1s))
    mnv2_avg = round(float(np.mean(mnv2_f1s)), 4)
    print(f"  MNV2-only baseline: avg_F1={mnv2_avg:.4f}")
    print()

    print(f"  {'Method':28s} | avg_F1 | avg_corr | ΔF1 vs MNV2 | per-DS F1")
    print(f"  {'─'*85}")

    specm_results = {}
    for sm in SPECM_MODELS:
        ds_results = {}
        avail = []
        for ds in DATASETS:
            srecs = load_specm(sm, ds)
            if srecs:
                am, as_ = align(mnv2_data[ds], srecs)
                if am:
                    ds_results[ds] = (am, as_)
                    avail.append(ds)

        fold_res = {}
        for test_ds in avail:
            am_te, as_te = ds_results[test_ds]
            preds = fuse_icwmv(am_te, as_te)
            fold_res[test_ds] = eval_preds(preds, am_te)

        avg_f1   = round(float(np.mean([r["macro_f1"] for r in fold_res.values()])), 4)
        avg_corr = round(float(np.mean([r["err_corr"]["rate"] for r in fold_res.values()])), 4)
        delta    = round(avg_f1 - mnv2_avg, 4)
        per_ds   = " | ".join(f"{ds}:{fold_res[ds]['macro_f1']:.4f}" for ds in avail if ds in fold_res)
        label    = f"ICWMV + {sm}"
        print(f"  {label:28s} | {avg_f1:.4f} | {avg_corr:.3f}      | {delta:+.4f}      | {per_ds}")
        specm_results[sm] = {"avg_f1": avg_f1, "avg_corr": avg_corr, "delta": delta}

    # comp_noTS vs v4 차이
    if "v4" in specm_results and "comp_noTS" in specm_results:
        df1 = round(specm_results["comp_noTS"]["avg_f1"] - specm_results["v4"]["avg_f1"], 4)
        dc  = round(specm_results["comp_noTS"]["avg_corr"] - specm_results["v4"]["avg_corr"], 3)
        print(f"\n  [EWCT 효과] comp_noTS vs v4: ΔF1={df1:+.4f}, Δcorr={dc:+.3f}")

    all_out[mnv2_label] = {"mnv2_baseline": mnv2_avg, **specm_results}

# 최종 요약
print(f"\n\n{'='*68}")
print("  전체 요약 — EWCT 기여도")
print(f"{'='*68}")
print(f"  {'':30s} | strong MNV2        | weak MNV2")
print(f"  {'':30s} | F1    corr  ΔF1   | F1    corr  ΔF1")
print(f"  {'─'*75}")
for sm in ["v4","comp_g1","comp_noTS"]:
    s = all_out.get("strong (070725)",{}).get(sm,{})
    w = all_out.get("weak   (064748)",{}).get(sm,{})
    label = f"ICWMV + {sm}"
    print(f"  {label:30s} | {s.get('avg_f1','?'):.4f} {s.get('avg_corr','?'):.3f} {s.get('delta','?'):+.4f} | "
          f"{w.get('avg_f1','?'):.4f} {w.get('avg_corr','?'):.3f} {w.get('delta','?'):+.4f}")

# 저장
ts_now = datetime.now().strftime("%Y%m%d_%H%M%S")
out_dir = ROOT / "experiments" / "results" / "icwmv_specm_compare"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / f"icwmv_specm_{ts_now}.json"
with open(out_path, "w") as f:
    json.dump(all_out, f, indent=2, ensure_ascii=False)
print(f"\n  저장: {out_path}")
