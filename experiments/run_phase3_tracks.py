#!/usr/bin/env python3
"""
Phase 3.1 — 3-Track 앙상블 JSONL 구성 + DAAC 메타 분류기 재학습
=================================================================

Track 1: ForMa + MobileNetV2 dual-stream + MobileCLIP-ft4 (3축)
Track 2: ForMa + MobileCLIP-ft4 (2축)
Track 3: Tiny-LaDeDa → (fake면) ForMa + MobileCLIP → DAAC cascade

각 Track별:
  1) 이미지별 백본 출력 병합 → 불일치 특징 추출 → JSONL 저장
  2) DAAC 메타 분류기(GBM/LogReg) 학습/평가 (10 seed × 80/20 split)
  3) 단일 백본 / rule baseline 비교

실행:
  .venv-qwen/bin/python experiments/run_phase3_tracks.py
  .venv-qwen/bin/python experiments/run_phase3_tracks.py --protocol cross
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

ROOT = Path(__file__).parent.parent  # MAIFS/
BACKBONE_DIR = ROOT / "experiments" / "results" / "backbone_eval"
OUT_DIR = ROOT / "experiments" / "results" / "phase3_tracks"
TRACK_JSONL_DIR = OUT_DIR / "jsonl"
OUT_DIR.mkdir(parents=True, exist_ok=True)
TRACK_JSONL_DIR.mkdir(parents=True, exist_ok=True)

LABEL_MAP = {"authentic": 0, "manipulated": 1, "ai_generated": 2}
IDX2LABEL = {v: k for k, v in LABEL_MAP.items()}
VERDICTS = ["authentic", "manipulated", "ai_generated"]

DATASETS: Dict[str, Dict] = {
    "base": {"desc": "CASIA2 + BigGAN (1500)"},
    "dsC": {"desc": "CASIA2 + IMD2020 + BigGAN (900)"},
    "opensdi": {"desc": "OpenSDID (900)"},
    "aigenproxy": {"desc": "AI-GenBench proxy (900)"},
}

BACKBONE_PREFIX = {
    "forma": "forma",
    "mobilenetv2_dualstream": "mobilenetv2_dualstream",
    "mobileclip_finetuned": "mobileclip_s2_finetuned",
    "tiny_ladeda": "tiny_ladeda",
}

TRACK1_FEATURE_NAMES = [
    "forma_auth",
    "forma_manip",
    "forma_aigen",
    "forma_conf",
    "mobilenet_auth_v",
    "mobilenet_manip_v",
    "mobilenet_aigen_v",
    "mobilenet_auth_s",
    "mobilenet_manip_s",
    "mobilenet_aigen_s",
    "mobilenet_conf",
    "clip_auth_v",
    "clip_manip_v",
    "clip_aigen_v",
    "clip_auth_s",
    "clip_manip_s",
    "clip_aigen_s",
    "clip_conf",
    "forma_vs_mobilenet_disagree",
    "forma_vs_clip_disagree",
    "mobilenet_vs_clip_disagree",
    "unanimous",
    "majority_auth",
    "majority_manip",
    "majority_aigen",
    "forma_auth_mobilenet_manip_clip_aigen",
    "forma_auth_mobilenet_aigen_clip_manip",
    "forma_manip_mobilenet_auth_clip_aigen",
    "forma_manip_mobilenet_aigen_clip_auth",
    "mobilenet_clip_agree_fake",
    "mean_conf",
    "std_conf",
]

TRACK2_FEATURE_NAMES = [
    "forma_auth",
    "forma_manip",
    "forma_aigen",
    "forma_conf",
    "clip_auth_v",
    "clip_manip_v",
    "clip_aigen_v",
    "clip_auth_s",
    "clip_manip_s",
    "clip_aigen_s",
    "clip_conf",
    "disagree",
    "forma_auth_clip_aigen",
    "forma_auth_clip_manip",
    "forma_manip_clip_aigen",
    "forma_manip_clip_auth",
    "conf_product",
    "conf_abs_diff",
]

TRACK3_FEATURE_NAMES = TRACK2_FEATURE_NAMES + [
    "tiny_auth",
    "tiny_manip",
    "tiny_aigen",
    "tiny_conf",
    "tiny_vs_forma_disagree",
    "tiny_vs_clip_disagree",
    "cascade_triggered",
]


def latest_backbone_jsonl(backbone: str, ds: str) -> Path:
    prefix = BACKBONE_PREFIX[backbone]
    candidates = sorted(BACKBONE_DIR.glob(f"{prefix}_{ds}_*.jsonl"))
    if not candidates:
        raise FileNotFoundError(f"Missing backbone JSONL: {prefix}_{ds}_*.jsonl")

    def ts_key(path: Path) -> str:
        stem = path.stem
        marker = f"{prefix}_{ds}_"
        return stem[len(marker):]

    return max(candidates, key=ts_key)


def load_backbone(path: Path) -> Dict[str, Dict]:
    recs = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            recs[d["image_path"]] = d
    return recs


def load_dataset(ds: str) -> Tuple[Dict, Dict, Dict, Dict]:
    forma = load_backbone(latest_backbone_jsonl("forma", ds))
    mobilenet = load_backbone(latest_backbone_jsonl("mobilenetv2_dualstream", ds))
    clip = load_backbone(latest_backbone_jsonl("mobileclip_finetuned", ds))
    tiny = load_backbone(latest_backbone_jsonl("tiny_ladeda", ds))
    return forma, mobilenet, clip, tiny


def verdict_onehot(verdict: str, classes: List[str] = VERDICTS) -> List[float]:
    return [1.0 if verdict == cls else 0.0 for cls in classes]


def record_verdict(rec: Dict) -> str:
    return rec.get("pred_label", rec.get("verdict", "authentic"))


def record_conf(rec: Dict) -> float:
    return float(rec.get("confidence", 0.5))


def multiclass_scores(rec: Dict) -> List[float]:
    scores = rec.get("scores")
    if isinstance(scores, dict):
        return [float(scores.get(cls, 0.0)) for cls in VERDICTS]

    verdict = record_verdict(rec)
    conf = record_conf(rec)
    if verdict == "authentic":
        return [conf, 1.0 - conf, 0.0]
    if verdict == "manipulated":
        return [1.0 - conf, conf, 0.0]
    return [1.0 - conf, 0.0, conf]


def cobra_predict(forma_v: str, clip_v: str, forma_c: float, clip_c: float) -> str:
    if forma_v == clip_v:
        return forma_v
    if clip_v == "ai_generated":
        return "ai_generated"
    return clip_v if clip_c >= forma_c else forma_v


def triad_weighted_vote(forma_rec: Dict, mobilenet_rec: Dict, clip_rec: Dict) -> Tuple[str, float]:
    scores = np.array(multiclass_scores(forma_rec), dtype=np.float32)
    scores += np.array(multiclass_scores(mobilenet_rec), dtype=np.float32)
    scores += np.array(multiclass_scores(clip_rec), dtype=np.float32)
    pred_idx = int(scores.argmax())
    total = float(scores.sum())
    conf = float(scores[pred_idx] / total) if total > 0 else 0.0
    return IDX2LABEL[pred_idx], conf


def cascade_verdict_track3(tiny_rec: Dict, forma_rec: Dict, clip_rec: Dict) -> Tuple[str, float]:
    """
    Track 3 rule baseline:
      - Tiny가 confident authentic면 바로 authentic fast-path
      - 그 외(fake 경로) ForMa + MobileCLIP rule consensus
    """
    tiny_v = record_verdict(tiny_rec)
    tiny_c = record_conf(tiny_rec)
    forma_v = record_verdict(forma_rec)
    forma_c = record_conf(forma_rec)
    clip_v = record_verdict(clip_rec)
    clip_c = record_conf(clip_rec)

    if tiny_v == "authentic" and tiny_c >= 0.90:
        return "authentic", tiny_c

    if clip_v == "ai_generated":
        if forma_v == "manipulated" and forma_c >= max(0.70, clip_c):
            return "manipulated", forma_c
        return "ai_generated", clip_c

    if forma_v == clip_v:
        return forma_v, (forma_c + clip_c) / 2.0

    if forma_v == "manipulated" and forma_c >= 0.70:
        return "manipulated", forma_c
    return clip_v, clip_c


def extract_track1(forma_rec: Dict, mobilenet_rec: Dict, clip_rec: Dict) -> np.ndarray:
    fv = record_verdict(forma_rec)
    fc = record_conf(forma_rec)
    mv = record_verdict(mobilenet_rec)
    mc = record_conf(mobilenet_rec)
    ms = multiclass_scores(mobilenet_rec)
    cv = record_verdict(clip_rec)
    cc = record_conf(clip_rec)
    cs = multiclass_scores(clip_rec)

    votes = [fv, mv, cv]
    counts = Counter(votes)
    feat = (
        verdict_onehot(fv)
        + [fc]
        + verdict_onehot(mv)
        + ms
        + [mc]
        + verdict_onehot(cv)
        + cs
        + [cc]
        + [float(fv != mv)]
        + [float(fv != cv)]
        + [float(mv != cv)]
        + [float(len(counts) == 1)]
        + [float(counts["authentic"] >= 2)]
        + [float(counts["manipulated"] >= 2)]
        + [float(counts["ai_generated"] >= 2)]
        + [float(fv == "authentic" and mv == "manipulated" and cv == "ai_generated")]
        + [float(fv == "authentic" and mv == "ai_generated" and cv == "manipulated")]
        + [float(fv == "manipulated" and mv == "authentic" and cv == "ai_generated")]
        + [float(fv == "manipulated" and mv == "ai_generated" and cv == "authentic")]
        + [float(mv == cv and mv != "authentic")]
        + [float(np.mean([fc, mc, cc]))]
        + [float(np.std([fc, mc, cc]))]
    )
    return np.array(feat, dtype=np.float32)


def extract_track2(forma_rec: Dict, clip_rec: Dict) -> np.ndarray:
    fv = record_verdict(forma_rec)
    fc = record_conf(forma_rec)
    cv = record_verdict(clip_rec)
    cc = record_conf(clip_rec)
    cs = multiclass_scores(clip_rec)

    feat = (
        verdict_onehot(fv)
        + [fc]
        + verdict_onehot(cv)
        + cs
        + [cc]
        + [float(fv != cv)]
        + [float(fv == "authentic" and cv == "ai_generated")]
        + [float(fv == "authentic" and cv == "manipulated")]
        + [float(fv == "manipulated" and cv == "ai_generated")]
        + [float(fv == "manipulated" and cv == "authentic")]
        + [fc * cc]
        + [abs(fc - cc)]
    )
    return np.array(feat, dtype=np.float32)


def extract_track3(forma_rec: Dict, clip_rec: Dict, tiny_rec: Dict) -> np.ndarray:
    tv = record_verdict(tiny_rec)
    tc = record_conf(tiny_rec)
    fv = record_verdict(forma_rec)
    cv = record_verdict(clip_rec)
    cascade_triggered = float(tv != "authentic")

    extra = (
        verdict_onehot(tv)
        + [tc]
        + [float(tv != fv)]
        + [float(tv != cv)]
        + [cascade_triggered]
    )
    return np.concatenate([extract_track2(forma_rec, clip_rec), np.array(extra, dtype=np.float32)])


def feature_dict(names: List[str], values: List[float]) -> Dict[str, float]:
    return {name: float(values[i]) for i, name in enumerate(names)}


def build_records(ds: str) -> List[Dict]:
    forma, mobilenet, clip, tiny = load_dataset(ds)
    common = set(forma) & set(mobilenet) & set(clip) & set(tiny)
    records = []
    for img_path in sorted(common):
        forma_rec = forma[img_path]
        mobilenet_rec = mobilenet[img_path]
        clip_rec = clip[img_path]
        tiny_rec = tiny[img_path]
        true_label = forma_rec.get("true_label", "authentic")
        if true_label not in LABEL_MAP:
            continue

        triad_v, triad_c = triad_weighted_vote(forma_rec, mobilenet_rec, clip_rec)
        cascade_v, cascade_c = cascade_verdict_track3(tiny_rec, forma_rec, clip_rec)

        track1_feat = extract_track1(forma_rec, mobilenet_rec, clip_rec).tolist()
        track2_feat = extract_track2(forma_rec, clip_rec).tolist()
        track3_feat = extract_track3(forma_rec, clip_rec, tiny_rec).tolist()

        records.append(
            {
                "image_path": img_path,
                "true_label": true_label,
                "label_idx": LABEL_MAP[true_label],
                "sub_type": forma_rec.get("sub_type", ""),
                "forma": {
                    "verdict": record_verdict(forma_rec),
                    "conf": record_conf(forma_rec),
                },
                "mobilenet": {
                    "verdict": record_verdict(mobilenet_rec),
                    "conf": record_conf(mobilenet_rec),
                    "scores": multiclass_scores(mobilenet_rec),
                },
                "clip": {
                    "verdict": record_verdict(clip_rec),
                    "conf": record_conf(clip_rec),
                    "scores": multiclass_scores(clip_rec),
                },
                "tiny": {
                    "verdict": record_verdict(tiny_rec),
                    "conf": record_conf(tiny_rec),
                },
                "feat_t1": track1_feat,
                "feat_t2": track2_feat,
                "feat_t3": track3_feat,
                "triad_rule": {"verdict": triad_v, "conf": triad_c},
                "cobra_rule": {
                    "verdict": cobra_predict(
                        record_verdict(forma_rec),
                        record_verdict(clip_rec),
                        record_conf(forma_rec),
                        record_conf(clip_rec),
                    ),
                    "conf": max(record_conf(forma_rec), record_conf(clip_rec)),
                },
                "cascade_rule": {"verdict": cascade_v, "conf": cascade_c},
            }
        )
    return records


def write_track_jsonls(ds: str, records: List[Dict], ts: str) -> Dict[str, str]:
    outputs = {}
    specs = [
        ("track1", "feat_t1", TRACK1_FEATURE_NAMES, ["forma", "mobilenet", "clip"], "triad_rule"),
        ("track2", "feat_t2", TRACK2_FEATURE_NAMES, ["forma", "clip"], "cobra_rule"),
        ("track3", "feat_t3", TRACK3_FEATURE_NAMES, ["tiny", "forma", "clip"], "cascade_rule"),
    ]

    for track_name, feat_key, feat_names, agent_keys, rule_key in specs:
        out_path = TRACK_JSONL_DIR / f"{track_name}_{ds}_{ts}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for rec in records:
                payload = {
                    "image_path": rec["image_path"],
                    "true_label": rec["true_label"],
                    "label_idx": rec["label_idx"],
                    "sub_type": rec["sub_type"],
                    "track": track_name,
                    "agents": {key: rec[key] for key in agent_keys},
                    "features": feature_dict(feat_names, rec[feat_key]),
                    "rule_baseline": rec[rule_key],
                }
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        outputs[track_name] = str(out_path)
    return outputs


def macro_f1(y_true: List[int], y_pred: List[int]) -> float:
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def per_class_recall(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    out = {}
    for idx, name in IDX2LABEL.items():
        mask = y_true == idx
        out[name] = float((y_pred[mask] == idx).mean()) if mask.sum() else 0.0
    return out


def run_evaluation(records: List[Dict], n_seeds: int = 10, val_ratio: float = 0.2) -> Dict:
    feats_t1 = np.array([r["feat_t1"] for r in records], dtype=np.float32)
    feats_t2 = np.array([r["feat_t2"] for r in records], dtype=np.float32)
    feats_t3 = np.array([r["feat_t3"] for r in records], dtype=np.float32)
    labels = np.array([r["label_idx"] for r in records], dtype=np.int64)

    forma_preds = np.array([LABEL_MAP[r["forma"]["verdict"]] for r in records], dtype=np.int64)
    mobilenet_preds = np.array([LABEL_MAP[r["mobilenet"]["verdict"]] for r in records], dtype=np.int64)
    clip_preds = np.array([LABEL_MAP[r["clip"]["verdict"]] for r in records], dtype=np.int64)
    tiny_preds = np.array([LABEL_MAP[r["tiny"]["verdict"]] for r in records], dtype=np.int64)
    cobra_preds = np.array([LABEL_MAP[r["cobra_rule"]["verdict"]] for r in records], dtype=np.int64)
    triad_preds = np.array([LABEL_MAP[r["triad_rule"]["verdict"]] for r in records], dtype=np.int64)
    cascade_preds = np.array([LABEL_MAP[r["cascade_rule"]["verdict"]] for r in records], dtype=np.int64)

    results = defaultdict(list)
    base_indices = list(range(len(records)))

    for seed in range(n_seeds):
        indices = base_indices.copy()
        random.Random(seed).shuffle(indices)
        n_val = int(len(indices) * val_ratio)
        val_idx = np.array(indices[:n_val], dtype=np.int64)
        train_idx = np.array(indices[n_val:], dtype=np.int64)

        y_train = labels[train_idx]
        y_val = labels[val_idx]

        gbm_t1 = GradientBoostingClassifier(
            n_estimators=250, max_depth=4, learning_rate=0.05, random_state=seed
        )
        gbm_t1.fit(feats_t1[train_idx], y_train)
        results["track1_gbm"].append(macro_f1(y_val, gbm_t1.predict(feats_t1[val_idx])))

        lr_t1 = LogisticRegression(max_iter=1000, random_state=seed, C=1.0)
        lr_t1.fit(feats_t1[train_idx], y_train)
        results["track1_lr"].append(macro_f1(y_val, lr_t1.predict(feats_t1[val_idx])))

        gbm_t2 = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05, random_state=seed
        )
        gbm_t2.fit(feats_t2[train_idx], y_train)
        results["track2_gbm"].append(macro_f1(y_val, gbm_t2.predict(feats_t2[val_idx])))

        lr_t2 = LogisticRegression(max_iter=1000, random_state=seed, C=1.0)
        lr_t2.fit(feats_t2[train_idx], y_train)
        results["track2_lr"].append(macro_f1(y_val, lr_t2.predict(feats_t2[val_idx])))

        gbm_t3 = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05, random_state=seed
        )
        gbm_t3.fit(feats_t3[train_idx], y_train)
        results["track3_gbm"].append(macro_f1(y_val, gbm_t3.predict(feats_t3[val_idx])))

        lr_t3 = LogisticRegression(max_iter=1000, random_state=seed, C=1.0)
        lr_t3.fit(feats_t3[train_idx], y_train)
        results["track3_lr"].append(macro_f1(y_val, lr_t3.predict(feats_t3[val_idx])))

        results["baseline_forma"].append(macro_f1(y_val, forma_preds[val_idx]))
        results["baseline_mobilenet"].append(macro_f1(y_val, mobilenet_preds[val_idx]))
        results["baseline_clip"].append(macro_f1(y_val, clip_preds[val_idx]))
        results["baseline_tiny"].append(macro_f1(y_val, tiny_preds[val_idx]))
        results["baseline_cobra"].append(macro_f1(y_val, cobra_preds[val_idx]))
        results["baseline_triad_rule"].append(macro_f1(y_val, triad_preds[val_idx]))
        results["baseline_cascade_rule"].append(macro_f1(y_val, cascade_preds[val_idx]))

    summary = {}
    for key, vals in results.items():
        summary[key] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "vals": [round(v, 4) for v in vals],
        }

    gbm_t1_full = GradientBoostingClassifier(
        n_estimators=250, max_depth=4, learning_rate=0.05, random_state=42
    )
    gbm_t1_full.fit(feats_t1, labels)
    summary["track1_gbm_full_recall"] = per_class_recall(labels, gbm_t1_full.predict(feats_t1))
    summary["track1_feature_importance"] = {
        name: round(float(val), 4)
        for name, val in zip(TRACK1_FEATURE_NAMES, gbm_t1_full.feature_importances_)
    }

    gbm_t2_full = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42
    )
    gbm_t2_full.fit(feats_t2, labels)
    summary["track2_gbm_full_recall"] = per_class_recall(labels, gbm_t2_full.predict(feats_t2))
    summary["track2_feature_importance"] = {
        name: round(float(val), 4)
        for name, val in zip(TRACK2_FEATURE_NAMES, gbm_t2_full.feature_importances_)
    }

    gbm_t3_full = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42
    )
    gbm_t3_full.fit(feats_t3, labels)
    summary["track3_gbm_full_recall"] = per_class_recall(labels, gbm_t3_full.predict(feats_t3))

    return summary


def run_cross_dataset() -> Dict:
    print("\n[Cross-Dataset Leave-One-Out 평가]")
    all_data = {}
    for ds in DATASETS:
        print(f"  로드: {ds}")
        all_data[ds] = build_records(ds)

    results = {}
    for held in DATASETS:
        train_recs = []
        for ds, recs in all_data.items():
            if ds != held:
                train_recs.extend(recs)
        test_recs = all_data[held]

        ft1_train = np.array([r["feat_t1"] for r in train_recs], dtype=np.float32)
        ft1_test = np.array([r["feat_t1"] for r in test_recs], dtype=np.float32)
        ft2_train = np.array([r["feat_t2"] for r in train_recs], dtype=np.float32)
        ft2_test = np.array([r["feat_t2"] for r in test_recs], dtype=np.float32)
        ft3_train = np.array([r["feat_t3"] for r in train_recs], dtype=np.float32)
        ft3_test = np.array([r["feat_t3"] for r in test_recs], dtype=np.float32)
        y_train = np.array([r["label_idx"] for r in train_recs], dtype=np.int64)
        y_test = np.array([r["label_idx"] for r in test_recs], dtype=np.int64)

        gbm_t1 = GradientBoostingClassifier(
            n_estimators=250, max_depth=4, learning_rate=0.05, random_state=42
        )
        gbm_t1.fit(ft1_train, y_train)

        gbm_t2 = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42
        )
        gbm_t2.fit(ft2_train, y_train)

        gbm_t3 = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42
        )
        gbm_t3.fit(ft3_train, y_train)

        mobilenet_preds = np.array([LABEL_MAP[r["mobilenet"]["verdict"]] for r in test_recs], dtype=np.int64)
        clip_preds = np.array([LABEL_MAP[r["clip"]["verdict"]] for r in test_recs], dtype=np.int64)
        triad_preds = np.array([LABEL_MAP[r["triad_rule"]["verdict"]] for r in test_recs], dtype=np.int64)
        cobra_preds = np.array([LABEL_MAP[r["cobra_rule"]["verdict"]] for r in test_recs], dtype=np.int64)
        cascade_preds = np.array([LABEL_MAP[r["cascade_rule"]["verdict"]] for r in test_recs], dtype=np.int64)

        results[held] = {
            "track1_gbm": macro_f1(y_test, gbm_t1.predict(ft1_test)),
            "track2_gbm": macro_f1(y_test, gbm_t2.predict(ft2_test)),
            "track3_gbm": macro_f1(y_test, gbm_t3.predict(ft3_test)),
            "mobilenet_only": macro_f1(y_test, mobilenet_preds),
            "clip_only": macro_f1(y_test, clip_preds),
            "triad_rule": macro_f1(y_test, triad_preds),
            "cobra": macro_f1(y_test, cobra_preds),
            "cascade_rule": macro_f1(y_test, cascade_preds),
        }
        print(
            f"  [{held}] T1={results[held]['track1_gbm']:.4f}, "
            f"T2={results[held]['track2_gbm']:.4f}, "
            f"T3={results[held]['track3_gbm']:.4f}, "
            f"MNV={results[held]['mobilenet_only']:.4f}, "
            f"CLIP={results[held]['clip_only']:.4f}"
        )

    return results


def print_summary(ds: str, summary: Dict):
    desc = DATASETS[ds]["desc"] if ds in DATASETS else "4개 데이터셋 통합"
    print(f"\n{'=' * 60}")
    print(f"  [{ds}] {desc}")
    print(f"{'=' * 60}")

    order = [
        "track1_gbm",
        "track1_lr",
        "track2_gbm",
        "track2_lr",
        "track3_gbm",
        "track3_lr",
        "baseline_triad_rule",
        "baseline_cobra",
        "baseline_cascade_rule",
        "baseline_clip",
        "baseline_mobilenet",
        "baseline_forma",
        "baseline_tiny",
    ]
    labels_nice = {
        "track1_gbm": "Track1 GBM      (ForMa+MNV2+CLIP)",
        "track1_lr": "Track1 LogReg   (ForMa+MNV2+CLIP)",
        "track2_gbm": "Track2 GBM      (ForMa+CLIP)",
        "track2_lr": "Track2 LogReg   (ForMa+CLIP)",
        "track3_gbm": "Track3 GBM      (Tiny+ForMa+CLIP)",
        "track3_lr": "Track3 LogReg   (Tiny+ForMa+CLIP)",
        "baseline_triad_rule": "Triad rule      (weighted vote)",
        "baseline_cobra": "COBRA baseline  (ForMa+CLIP)",
        "baseline_cascade_rule": "Cascade rule    (Tiny→Tier2)",
        "baseline_clip": "MobileCLIP-ft4  (single)",
        "baseline_mobilenet": "MobileNetV2-DS  (single)",
        "baseline_forma": "ForMa           (single)",
        "baseline_tiny": "Tiny-LaDeDa     (single)",
    }
    best_mean = max(summary[k]["mean"] for k in order if k in summary)
    for key in order:
        if key not in summary:
            continue
        star = " ★" if abs(summary[key]["mean"] - best_mean) < 1e-6 else ""
        print(
            f"  {labels_nice[key]:40s}: "
            f"{summary[key]['mean']:.4f} ± {summary[key]['std']:.4f}{star}"
        )

    print("\n  Track1 GBM feature importance (top 5):")
    for name, val in sorted(summary.get("track1_feature_importance", {}).items(), key=lambda x: -x[1])[:5]:
        print(f"    {name:32s}: {val:.4f}")

    print("\n  Track2 GBM feature importance (top 5):")
    for name, val in sorted(summary.get("track2_feature_importance", {}).items(), key=lambda x: -x[1])[:5]:
        print(f"    {name:32s}: {val:.4f}")

    print("\n  Full-data recall:")
    for tag in ["track1_gbm_full_recall", "track2_gbm_full_recall", "track3_gbm_full_recall"]:
        if tag not in summary:
            continue
        print(f"    {tag}:")
        for cls, rec in summary[tag].items():
            print(f"      {cls:12s}: {rec:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        choices=["per_ds", "combined", "cross"],
        default="combined",
        help="per_ds=데이터셋별 독립 평가, combined=4개 통합, cross=leave-one-out",
    )
    parser.add_argument("--seeds", type=int, default=10)
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results = {
        "protocol": args.protocol,
        "timestamp": ts,
        "datasets": {},
        "track_jsonl": {},
        "feature_schemas": {
            "track1": TRACK1_FEATURE_NAMES,
            "track2": TRACK2_FEATURE_NAMES,
            "track3": TRACK3_FEATURE_NAMES,
        },
    }

    if args.protocol == "cross":
        for ds in DATASETS:
            recs = build_records(ds)
            all_results["track_jsonl"][ds] = write_track_jsonls(ds, recs, ts)
        all_results["cross_dataset"] = run_cross_dataset()
        out = OUT_DIR / f"phase3_cross_{ts}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\n저장: {out}")
        return

    if args.protocol == "combined":
        print("\n[전체 4개 데이터셋 통합 로드]")
        all_recs = []
        for ds in DATASETS:
            print(f"  {ds}...")
            recs = build_records(ds)
            all_results["track_jsonl"][ds] = write_track_jsonls(ds, recs, ts)
            all_recs.extend(recs)
            all_results["datasets"][ds] = {"n": len(recs)}
        print(f"  총 {len(all_recs)}개")
        print(f"\n[평가 실행 — {args.seeds} seeds]")
        summary = run_evaluation(all_recs, n_seeds=args.seeds)
        print_summary("combined", summary)
        all_results["combined"] = summary
    else:
        for ds in DATASETS:
            print(f"\n[{ds}] 로드...")
            recs = build_records(ds)
            all_results["track_jsonl"][ds] = write_track_jsonls(ds, recs, ts)
            print(f"  {len(recs)}개 → 평가")
            summary = run_evaluation(recs, n_seeds=args.seeds)
            print_summary(ds, summary)
            all_results["datasets"][ds] = summary

    out = OUT_DIR / f"phase3_tracks_{args.protocol}_{ts}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n\n결과 저장: {out}")

    if args.protocol == "combined":
        s = all_results["combined"]
        print("\n" + "=" * 60)
        print("  최종 비교 (combined, 10-seed macro-F1 mean ± std)")
        print("=" * 60)
        print(f"  Track1-GBM   : {s['track1_gbm']['mean']:.4f} ± {s['track1_gbm']['std']:.4f}")
        print(f"  Track2-GBM   : {s['track2_gbm']['mean']:.4f} ± {s['track2_gbm']['std']:.4f}")
        print(f"  Track3-GBM   : {s['track3_gbm']['mean']:.4f} ± {s['track3_gbm']['std']:.4f}")
        print(f"  Triad-rule   : {s['baseline_triad_rule']['mean']:.4f} ± {s['baseline_triad_rule']['std']:.4f}")
        print(f"  COBRA        : {s['baseline_cobra']['mean']:.4f} ± {s['baseline_cobra']['std']:.4f}")
        print(f"  Cascade-rule : {s['baseline_cascade_rule']['mean']:.4f} ± {s['baseline_cascade_rule']['std']:.4f}")
        print(f"  MobileCLIP   : {s['baseline_clip']['mean']:.4f} ± {s['baseline_clip']['std']:.4f}")
        print(f"  MobileNetV2  : {s['baseline_mobilenet']['mean']:.4f} ± {s['baseline_mobilenet']['std']:.4f}")
        print(f"\n  Track1-GBM vs MobileCLIP: Δ={s['track1_gbm']['mean'] - s['baseline_clip']['mean']:+.4f}")
        print(f"  Track2-GBM vs MobileCLIP: Δ={s['track2_gbm']['mean'] - s['baseline_clip']['mean']:+.4f}")
        print(f"  Track3-GBM vs MobileCLIP: Δ={s['track3_gbm']['mean'] - s['baseline_clip']['mean']:+.4f}")


if __name__ == "__main__":
    main()
