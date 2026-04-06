#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import pickle
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LABELS = ("authentic", "manipulated", "ai_generated")
IDX2LABEL = {0: "authentic", 1: "manipulated", 2: "ai_generated"}
LABEL2IDX = {label: idx for idx, label in IDX2LABEL.items()}
BINARY_LABELS = ("authentic", "edited")

FEATURE_NAMES = [
    "base_auth_prob",
    "base_manip_prob",
    "base_ai_prob",
    "base_confidence",
    "base_binary_confidence",
    "base_binary_margin",
    "base_ai_margin",
    "aux_auth_prob",
    "aux_manip_prob",
    "aux_confidence",
    "aux_binary_confidence",
    "aux_binary_margin",
    "stage1_auth_prob",
    "stage1_manip_prob",
    "stage1_ai_prob",
    "stage1_confidence",
    "stage1_binary_confidence",
    "stage1_binary_margin",
    "base_vs_aux_disagree",
    "base_vs_stage1_disagree",
    "aux_vs_stage1_disagree",
    "ai_lock_flag",
    "base_to_stage1_auth_to_auth",
    "base_to_stage1_auth_to_manip",
    "base_to_stage1_auth_to_ai",
    "base_to_stage1_manip_to_auth",
    "base_to_stage1_manip_to_manip",
    "base_to_stage1_manip_to_ai",
    "base_to_stage1_ai_to_auth",
    "base_to_stage1_ai_to_manip",
    "base_to_stage1_ai_to_ai",
]


@dataclass(frozen=True)
class Stage2Config:
    model_key: str
    tau: float
    pos_weight: float
    weighting_alpha: float = 1.0
    ai_lock_threshold: float = 0.5


@dataclass
class Stage2Split:
    base_rows: List[Dict[str, Any]]
    aux_rows: List[Dict[str, Any]]


class ConstantKeepModel:
    def __init__(self, prob_keep: float):
        self.prob_keep = float(prob_keep)
        self._feature_names = FEATURE_NAMES

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        p = np.full((len(x),), self.prob_keep, dtype=np.float32)
        return np.stack([1.0 - p, p], axis=1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(rows: Iterable[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _index_by_image_path(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    indexed: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        image_path = row.get("image_path")
        if not image_path:
            continue
        if image_path in indexed:
            raise ValueError(f"Duplicate image_path found: {image_path}")
        indexed[image_path] = row
    return indexed


def align_rows(base_rows: Sequence[Dict[str, Any]], aux_rows: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    aux_map = _index_by_image_path(aux_rows)
    aligned_base: List[Dict[str, Any]] = []
    aligned_aux: List[Dict[str, Any]] = []
    missing = 0
    for row in base_rows:
        image_path = row.get("image_path")
        aux_row = aux_map.get(image_path)
        if aux_row is None:
            missing += 1
            continue
        aligned_base.append(row)
        aligned_aux.append(aux_row)
    if missing:
        print(f"  [align] skipped {missing} base rows without matching aux row")
    return aligned_base, aligned_aux


def _normalize_pair(a: float, b: float) -> Tuple[float, float]:
    total = float(a + b)
    if total <= 0.0:
        return 0.5, 0.5
    return float(a / total), float(b / total)


def _softmax_like_triplet(scores: Dict[str, Any], pred_label: Optional[str] = None, confidence: Optional[float] = None) -> Tuple[float, float, float]:
    if isinstance(scores, dict) and scores:
        auth = float(scores.get("authentic", 0.0))
        manip = float(scores.get("manipulated", 0.0))
        aigen = float(scores.get("ai_generated", 0.0))
        total = auth + manip + aigen
        if total > 0.0:
            return auth / total, manip / total, aigen / total

    pred_label = pred_label or "authentic"
    confidence = 1.0 if confidence is None else float(confidence)
    confidence = max(min(confidence, 1.0), 0.0)
    rest = max(1.0 - confidence, 0.0)
    if pred_label not in LABEL2IDX:
        pred_label = "authentic"
    if pred_label == "authentic":
        return confidence, rest * 0.5, rest * 0.5
    if pred_label == "manipulated":
        return rest * 0.5, confidence, rest * 0.5
    return rest * 0.5, rest * 0.5, confidence


def _base_triplet(row: Dict[str, Any]) -> Tuple[float, float, float]:
    scores = row.get("scores", {})
    return _softmax_like_triplet(scores, row.get("pred_label"), row.get("confidence"))


def _aux_pair(row: Dict[str, Any]) -> Tuple[float, float]:
    if "authentic_score" in row or "manip_score" in row:
        auth = float(row.get("authentic_score", 0.0))
        manip = float(row.get("manip_score", 0.0))
        return _normalize_pair(auth, manip)
    scores = row.get("scores", {})
    auth = float(scores.get("authentic", 0.0)) if isinstance(scores, dict) else 0.0
    manip = float(scores.get("manipulated", 0.0)) if isinstance(scores, dict) else 0.0
    if auth == 0.0 and manip == 0.0:
        pred_label = row.get("pred_label", "authentic")
        conf = row.get("confidence", 1.0)
        if pred_label == "manipulated":
            return _normalize_pair(1.0 - float(conf), float(conf))
        return _normalize_pair(float(conf), 1.0 - float(conf))
    return _normalize_pair(auth, manip)


def _binary_probs_from_triplet(auth: float, manip: float) -> Tuple[float, float]:
    return _normalize_pair(auth, manip)


def _pred_idx_from_probs(probs: Sequence[float]) -> int:
    return int(np.argmax(np.asarray(probs, dtype=np.float32)))


def is_ai_lock(base_row: Dict[str, Any], ai_lock_threshold: float = 0.5) -> bool:
    auth, manip, aigen = _base_triplet(base_row)
    pred_label = base_row.get("pred_label", "")
    return pred_label == "ai_generated" or float(aigen) >= ai_lock_threshold


def base_binary_prediction(base_row: Dict[str, Any]) -> int:
    auth, manip, _ = _base_triplet(base_row)
    auth_b, manip_b = _binary_probs_from_triplet(auth, manip)
    return _pred_idx_from_probs((auth_b, manip_b))


def stage1_binary_probs(base_row: Dict[str, Any], aux_row: Dict[str, Any], weighting_alpha: float = 1.0, ai_lock_threshold: float = 0.5) -> Tuple[float, float, float]:
    base_auth, base_manip, base_ai = _base_triplet(base_row)
    if is_ai_lock(base_row, ai_lock_threshold=ai_lock_threshold):
        return 0.0, 0.0, 1.0

    aux_auth, aux_manip = _aux_pair(aux_row)
    base_auth_b, base_manip_b = _binary_probs_from_triplet(base_auth, base_manip)
    base_conf = max(base_auth_b, base_manip_b)
    aux_conf = max(aux_auth, aux_manip)
    if weighting_alpha <= 0.0:
        w_base = 1.0
        w_aux = 1.0
    else:
        w_base = 1.0 / max(base_conf, 1e-3) ** weighting_alpha
        w_aux = 1.0 / max(aux_conf, 1e-3) ** weighting_alpha
    combined_auth = w_base * base_auth_b + w_aux * aux_auth
    combined_manip = w_base * base_manip_b + w_aux * aux_manip
    total = combined_auth + combined_manip
    if total <= 0.0:
        return 0.5, 0.5, 0.0
    return combined_auth / total, combined_manip / total, 0.0


def stage1_pred_idx(base_row: Dict[str, Any], aux_row: Dict[str, Any], weighting_alpha: float = 1.0, ai_lock_threshold: float = 0.5) -> int:
    if is_ai_lock(base_row, ai_lock_threshold=ai_lock_threshold):
        return LABEL2IDX["ai_generated"]
    auth, manip, _ = stage1_binary_probs(base_row, aux_row, weighting_alpha=weighting_alpha, ai_lock_threshold=ai_lock_threshold)
    return LABEL2IDX["manipulated"] if manip >= auth else LABEL2IDX["authentic"]


def _transition_one_hot(base_idx: int, stage1_idx: int) -> List[float]:
    vec = [0.0] * 9
    vec[base_idx * 3 + stage1_idx] = 1.0
    return vec


def build_feature_vector(base_row: Dict[str, Any], aux_row: Dict[str, Any], weighting_alpha: float = 1.0, ai_lock_threshold: float = 0.5) -> Tuple[np.ndarray, Dict[str, Any]]:
    base_auth, base_manip, base_ai = _base_triplet(base_row)
    base_auth_b, base_manip_b = _binary_probs_from_triplet(base_auth, base_manip)
    base_conf = max(base_auth, base_manip, base_ai)
    base_bin_conf = max(base_auth_b, base_manip_b)
    base_bin_margin = base_manip_b - base_auth_b
    base_ai_margin = base_ai - max(base_auth, base_manip)

    aux_auth, aux_manip = _aux_pair(aux_row)
    aux_conf = max(aux_auth, aux_manip)
    aux_bin_conf = aux_conf
    aux_bin_margin = aux_manip - aux_auth

    stage1_auth, stage1_manip, stage1_ai = stage1_binary_probs(
        base_row,
        aux_row,
        weighting_alpha=weighting_alpha,
        ai_lock_threshold=ai_lock_threshold,
    )
    stage1_conf = max(stage1_auth, stage1_manip, stage1_ai)
    stage1_bin_conf = max(stage1_auth, stage1_manip)
    stage1_bin_margin = stage1_manip - stage1_auth

    base_idx = LABEL2IDX.get(base_row.get("pred_label", "authentic"), LABEL2IDX["authentic"])
    aux_idx = 1 if float(aux_manip) >= float(aux_auth) else 0
    stage1_idx = stage1_pred_idx(
        base_row,
        aux_row,
        weighting_alpha=weighting_alpha,
        ai_lock_threshold=ai_lock_threshold,
    )

    features = [
        base_auth,
        base_manip,
        base_ai,
        base_conf,
        base_bin_conf,
        base_bin_margin,
        base_ai_margin,
        aux_auth,
        aux_manip,
        aux_conf,
        aux_bin_conf,
        aux_bin_margin,
        stage1_auth,
        stage1_manip,
        stage1_ai,
        stage1_conf,
        stage1_bin_conf,
        stage1_bin_margin,
        float(base_idx != aux_idx),
        float(base_idx != stage1_idx),
        float(aux_idx != stage1_idx),
        float(is_ai_lock(base_row, ai_lock_threshold=ai_lock_threshold)),
        *_transition_one_hot(base_idx, stage1_idx),
    ]
    audit = {
        "base_pred_label": base_row.get("pred_label", ""),
        "aux_pred_label": aux_row.get("pred_label", ""),
        "stage1_pred_label": IDX2LABEL[stage1_idx],
    }
    return np.asarray(features, dtype=np.float32), audit


def collapse_to_binary(label: str) -> str:
    return "authentic" if label == "authentic" else "edited"


def _class_metrics(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> Dict[str, Any]:
    counts_true = Counter(y_true)
    counts_pred = Counter(y_pred)
    per_class: Dict[str, Dict[str, float]] = {}
    f1s: List[float] = []
    correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    for label in labels:
        tp = sum(1 for a, b in zip(y_true, y_pred) if a == label and b == label)
        fp = sum(1 for a, b in zip(y_true, y_pred) if a != label and b == label)
        fn = sum(1 for a, b in zip(y_true, y_pred) if a == label and b != label)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        f1s.append(f1)
        per_class[label] = {
            "support": int(counts_true.get(label, 0)),
            "predicted": int(counts_pred.get(label, 0)),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "precision": round(float(precision), 6),
            "recall": round(float(recall), 6),
            "f1": round(float(f1), 6),
        }
    matrix = {true: {pred: 0 for pred in labels} for true in labels}
    for true_label, pred_label in zip(y_true, y_pred):
        if true_label in matrix and pred_label in matrix[true_label]:
            matrix[true_label][pred_label] += 1
    return {
        "n": int(len(y_true)),
        "accuracy": round(float(correct / max(len(y_true), 1)), 6),
        "macro_f1": round(float(np.mean(f1s) if f1s else 0.0), 6),
        "per_class": per_class,
        "confusion_matrix": matrix,
    }


def _binary_metrics(y_true: Sequence[str], y_pred: Sequence[str]) -> Dict[str, Any]:
    return _class_metrics(y_true, y_pred, BINARY_LABELS)


def _change_stats(base_true: Sequence[str], base_pred: Sequence[str], new_pred: Sequence[str]) -> Dict[str, Any]:
    n_changed = 0
    n_helpful = 0
    n_harmful = 0
    by_direction: Counter[str] = Counter()
    for t, b, n in zip(base_true, base_pred, new_pred):
        if b == n:
            continue
        n_changed += 1
        by_direction[f"{b}->{n}"] += 1
        if b != t and n == t:
            n_helpful += 1
        elif b == t and n != t:
            n_harmful += 1
    return {
        "n_changed": int(n_changed),
        "n_helpful": int(n_helpful),
        "n_harmful": int(n_harmful),
        "helpful_change_rate": round(n_helpful / max(n_changed, 1), 6),
        "harmful_change_rate": round(n_harmful / max(n_changed, 1), 6),
        "change_directions": dict(by_direction),
    }


def _stage2_labels(base_rows: Sequence[Dict[str, Any]], stage1_preds: Sequence[str]) -> Tuple[List[str], List[str]]:
    y_true: List[str] = []
    y_pred: List[str] = []
    for base_row, stage1_pred in zip(base_rows, stage1_preds):
        true_label = str(base_row.get("true_label", ""))
        if true_label == "ai_generated":
            continue
        if stage1_pred == base_row.get("pred_label", ""):
            continue
        y_true.append("keep" if stage1_pred == true_label else "revert")
        y_pred.append("keep")  # placeholder for shape parity
    return y_true, y_pred


def build_override_dataset(
    base_rows: Sequence[Dict[str, Any]],
    aux_rows: Sequence[Dict[str, Any]],
    pos_weight: float,
    weighting_alpha: float = 1.0,
    ai_lock_threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]], Dict[str, int]]:
    features: List[np.ndarray] = []
    labels: List[int] = []
    weights: List[float] = []
    audits: List[Dict[str, Any]] = []
    stats: Counter[str] = Counter()

    for base_row, aux_row in zip(base_rows, aux_rows):
        true_label = str(base_row.get("true_label", ""))
        if true_label == "ai_generated":
            stats["skip_true_ai_generated"] += 1
            continue
        if is_ai_lock(base_row, ai_lock_threshold=ai_lock_threshold):
            stats["skip_ai_lock"] += 1
            continue
        stage1_idx = stage1_pred_idx(
            base_row,
            aux_row,
            weighting_alpha=weighting_alpha,
            ai_lock_threshold=ai_lock_threshold,
        )
        base_idx = LABEL2IDX.get(str(base_row.get("pred_label", "authentic")), LABEL2IDX["authentic"])
        if stage1_idx == base_idx:
            stats["skip_no_override"] += 1
            continue

        feat, audit = build_feature_vector(
            base_row,
            aux_row,
            weighting_alpha=weighting_alpha,
            ai_lock_threshold=ai_lock_threshold,
        )
        keep_override = int(IDX2LABEL[stage1_idx] == true_label)
        features.append(feat)
        labels.append(keep_override)
        weights.append(float(pos_weight if keep_override else 1.0))
        audits.append(
            {
                "image_path": base_row.get("image_path", ""),
                "true_label": true_label,
                "base_pred_label": base_row.get("pred_label", ""),
                "aux_pred_label": aux_row.get("pred_label", ""),
                "stage1_pred_label": IDX2LABEL[stage1_idx],
                "label": "keep" if keep_override else "revert",
                **audit,
            }
        )
        stats["override_candidates"] += 1
        if keep_override:
            stats["beneficial_overrides"] += 1
        else:
            stats["harmful_overrides"] += 1

    if not features:
        return (
            np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.float32),
            audits,
            dict(stats),
        )

    return (
        np.vstack(features),
        np.asarray(labels, dtype=np.int64),
        np.asarray(weights, dtype=np.float32),
        audits,
        dict(stats),
    )


def train_veto_model(x_tr: np.ndarray, y_tr: np.ndarray, sample_weight: np.ndarray, model_key: str):
    if len(x_tr) == 0:
        return ConstantKeepModel(prob_keep=1.0)

    uniq = sorted(set(y_tr.tolist()))
    if len(uniq) == 1:
        return ConstantKeepModel(prob_keep=float(uniq[0]))

    if model_key == "logreg":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=42)),
        ])
        clf.fit(x_tr, y_tr, clf__sample_weight=sample_weight)
        clf._feature_names = FEATURE_NAMES  # type: ignore[attr-defined]
        return clf

    normalized_model_key = "xgb_depth2" if model_key == "xgb" else model_key

    if normalized_model_key in {"xgb_stump", "xgb_depth2"}:
        import xgboost as xgb

        max_depth = 1 if normalized_model_key == "xgb_stump" else 2
        clf = xgb.XGBClassifier(
            n_estimators=120,
            max_depth=max_depth,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=3.0,
            min_child_weight=4,
            eval_metric="logloss",
            objective="binary:logistic",
            tree_method="hist",
            n_jobs=1,
            random_state=42,
            verbosity=0,
        )
        clf.fit(x_tr, y_tr, sample_weight=sample_weight)
        clf._feature_names = FEATURE_NAMES  # type: ignore[attr-defined]
        return clf

    raise ValueError(f"Unknown veto model: {model_key}")


def _keep_prob(model: Any, feature: np.ndarray) -> float:
    x = np.asarray(feature, dtype=np.float32).reshape(1, -1)
    proba = model.predict_proba(x)
    return float(proba[0, 1])


def apply_stage2(
    base_rows: Sequence[Dict[str, Any]],
    aux_rows: Sequence[Dict[str, Any]],
    model: Any,
    tau: float,
    weighting_alpha: float = 1.0,
    ai_lock_threshold: float = 0.5,
) -> Tuple[List[int], List[Dict[str, Any]], Dict[str, int]]:
    preds: List[int] = []
    out_rows: List[Dict[str, Any]] = []
    actions: Counter[str] = Counter()

    for base_row, aux_row in zip(base_rows, aux_rows):
        base_idx = LABEL2IDX.get(str(base_row.get("pred_label", "authentic")), LABEL2IDX["authentic"])
        true_label = str(base_row.get("true_label", ""))
        if is_ai_lock(base_row, ai_lock_threshold=ai_lock_threshold):
            pred_idx = LABEL2IDX["ai_generated"]
            action = "ai_lock"
            p_keep = 1.0
            stage1_idx = pred_idx
            feature = np.zeros((len(FEATURE_NAMES),), dtype=np.float32)
        else:
            feature, audit = build_feature_vector(
                base_row,
                aux_row,
                weighting_alpha=weighting_alpha,
                ai_lock_threshold=ai_lock_threshold,
            )
            stage1_idx = LABEL2IDX.get(audit["stage1_pred_label"], LABEL2IDX["authentic"])
            if stage1_idx == base_idx:
                pred_idx = base_idx
                action = "no_override"
                p_keep = 1.0
            else:
                p_keep = _keep_prob(model, feature)
                if p_keep >= tau:
                    pred_idx = stage1_idx
                    action = "keep"
                else:
                    pred_idx = base_idx
                    action = "revert"
        preds.append(pred_idx)
        actions[action] += 1
        stage1_label = IDX2LABEL[stage1_idx]
        out_rows.append(
            {
                "image_path": base_row.get("image_path", ""),
                "true_label": true_label,
                "base_pred_label": base_row.get("pred_label", ""),
                "aux_pred_label": aux_row.get("pred_label", ""),
                "stage1_pred_label": stage1_label,
                "stage2_pred_label": IDX2LABEL[pred_idx],
                "action": action,
                "p_keep": float(p_keep),
                "base_scores": base_row.get("scores", {}),
                "aux_scores": {
                    "authentic": float(aux_row.get("authentic_score", 0.0)),
                    "manipulated": float(aux_row.get("manip_score", 0.0)),
                },
            }
        )
    return preds, out_rows, dict(actions)


def evaluate_rows(rows: Sequence[Dict[str, Any]], pred_key: str) -> Dict[str, Any]:
    y_true_strict = [str(r.get("true_label", "")) for r in rows]
    y_pred_strict = [str(r.get(pred_key, "")) for r in rows]
    y_true_binary = [collapse_to_binary(y) for y in y_true_strict]
    y_pred_binary = [collapse_to_binary(y) for y in y_pred_strict]
    return {
        "strict_three_class": _class_metrics(y_true_strict, y_pred_strict, LABELS),
        "binary_auth_vs_edited": _binary_metrics(y_true_binary, y_pred_binary),
    }


def _score_candidate(metrics: Dict[str, Any], change_stats: Dict[str, Any]) -> Tuple[float, float, float, float, float]:
    return (
        float(metrics["strict_three_class"]["macro_f1"]),
        float(metrics["binary_auth_vs_edited"]["macro_f1"]),
        float(change_stats["helpful_change_rate"]),
        -float(change_stats["harmful_change_rate"]),
        float(metrics["binary_auth_vs_edited"]["accuracy"]),
    )


def tune_stage2(
    train_base: Sequence[Dict[str, Any]],
    train_aux: Sequence[Dict[str, Any]],
    val_base: Sequence[Dict[str, Any]],
    val_aux: Sequence[Dict[str, Any]],
    taus: Sequence[float],
    pos_weights: Sequence[float],
    model_keys: Sequence[str],
    weighting_alpha: float = 1.0,
    ai_lock_threshold: float = 0.5,
) -> Dict[str, Any]:
    best: Optional[Dict[str, Any]] = None

    for pos_weight in pos_weights:
        x_tr, y_tr, w_tr, train_audit, train_stats = build_override_dataset(
            train_base,
            train_aux,
            pos_weight=pos_weight,
            weighting_alpha=weighting_alpha,
            ai_lock_threshold=ai_lock_threshold,
        )
        if len(x_tr) == 0:
            continue

        for model_key in model_keys:
            model = train_veto_model(x_tr, y_tr, w_tr, model_key)
            for tau in taus:
                val_preds, val_rows, val_actions = apply_stage2(
                    val_base,
                    val_aux,
                    model,
                    tau=tau,
                    weighting_alpha=weighting_alpha,
                    ai_lock_threshold=ai_lock_threshold,
                )
                val_metrics = evaluate_rows(val_rows, "stage2_pred_label")
                change_stats = _change_stats(
                    [r.get("true_label", "") for r in val_base],
                    [r.get("pred_label", "") for r in val_base],
                    [r["stage2_pred_label"] for r in val_rows],
                )
                cand = {
                    "tau": float(tau),
                    "pos_weight": float(pos_weight),
                    "model_key": model_key,
                    "train_override_candidates": int(train_stats.get("override_candidates", 0)),
                    "train_keep_rate": round(
                        float(train_stats.get("beneficial_overrides", 0)) / max(float(train_stats.get("override_candidates", 0)), 1.0),
                        6,
                    ),
                    "val_metrics": val_metrics,
                    "val_change_stats": change_stats,
                    "val_actions": val_actions,
                    "val_score": _score_candidate(val_metrics, change_stats),
                }
                if best is None or cand["val_score"] > best["val_score"]:
                    best = cand

    if best is None:
        best = {
            "tau": 0.5,
            "pos_weight": 1.0,
            "model_key": "logreg",
            "train_override_candidates": 0,
            "train_keep_rate": 1.0,
            "val_metrics": evaluate_rows([], "stage2_pred_label"),
            "val_change_stats": {
                "n_changed": 0,
                "n_helpful": 0,
                "n_harmful": 0,
                "helpful_change_rate": 0.0,
                "harmful_change_rate": 0.0,
                "change_directions": {},
            },
            "val_actions": {},
            "val_score": (0.0, 0.0, 0.0, 0.0, 0.0),
        }
    return best


def run_stage2_pipeline(
    train_base_path: Path,
    train_aux_path: Path,
    val_base_path: Path,
    val_aux_path: Path,
    test_base_path: Path,
    test_aux_path: Path,
    run_dir: Path,
    *,
    taus: Sequence[float],
    pos_weights: Sequence[float],
    model_keys: Sequence[str],
    weighting_alpha: float = 1.0,
    ai_lock_threshold: float = 0.5,
    seed: int = 42,
) -> Dict[str, Any]:
    set_seed(seed)
    run_dir.mkdir(parents=True, exist_ok=True)

    train_base_raw = load_jsonl(train_base_path)
    train_aux_raw = load_jsonl(train_aux_path)
    val_base_raw = load_jsonl(val_base_path)
    val_aux_raw = load_jsonl(val_aux_path)
    test_base_raw = load_jsonl(test_base_path)
    test_aux_raw = load_jsonl(test_aux_path)

    train_base, train_aux = align_rows(train_base_raw, train_aux_raw)
    val_base, val_aux = align_rows(val_base_raw, val_aux_raw)
    test_base, test_aux = align_rows(test_base_raw, test_aux_raw)

    selection = tune_stage2(
        train_base,
        train_aux,
        val_base,
        val_aux,
        taus=taus,
        pos_weights=pos_weights,
        model_keys=model_keys,
        weighting_alpha=weighting_alpha,
        ai_lock_threshold=ai_lock_threshold,
    )

    trainval_base = list(train_base) + list(val_base)
    trainval_aux = list(train_aux) + list(val_aux)
    x_tv, y_tv, w_tv, trainval_audit, trainval_stats = build_override_dataset(
        trainval_base,
        trainval_aux,
        pos_weight=float(selection["pos_weight"]),
        weighting_alpha=weighting_alpha,
        ai_lock_threshold=ai_lock_threshold,
    )
    final_model = train_veto_model(x_tv, y_tv, w_tv, str(selection["model_key"]))
    final_model_path = run_dir / "stage2_model.pkl"
    with final_model_path.open("wb") as fh:
        pickle.dump(final_model, fh)

    train_preds, train_rows, train_actions = apply_stage2(
        train_base,
        train_aux,
        final_model,
        tau=float(selection["tau"]),
        weighting_alpha=weighting_alpha,
        ai_lock_threshold=ai_lock_threshold,
    )
    val_preds, val_rows, val_actions = apply_stage2(
        val_base,
        val_aux,
        final_model,
        tau=float(selection["tau"]),
        weighting_alpha=weighting_alpha,
        ai_lock_threshold=ai_lock_threshold,
    )
    test_preds, test_rows, test_actions = apply_stage2(
        test_base,
        test_aux,
        final_model,
        tau=float(selection["tau"]),
        weighting_alpha=weighting_alpha,
        ai_lock_threshold=ai_lock_threshold,
    )

    write_jsonl(train_rows, run_dir / "stage2_train_predictions.jsonl")
    write_jsonl(val_rows, run_dir / "stage2_val_predictions.jsonl")
    write_jsonl(test_rows, run_dir / "stage2_test_predictions.jsonl")

    train_base_eval = evaluate_rows(
        [
            {
                "true_label": row.get("true_label", ""),
                "stage2_pred_label": row.get("pred_label", ""),
            }
            for row in train_base
        ],
        "stage2_pred_label",
    )
    val_base_eval = evaluate_rows(
        [
            {
                "true_label": row.get("true_label", ""),
                "stage2_pred_label": row.get("pred_label", ""),
            }
            for row in val_base
        ],
        "stage2_pred_label",
    )
    test_base_eval = evaluate_rows(
        [
            {
                "true_label": row.get("true_label", ""),
                "stage2_pred_label": row.get("pred_label", ""),
            }
            for row in test_base
        ],
        "stage2_pred_label",
    )

    base_test_metrics = evaluate_rows(
        [{"true_label": row.get("true_label", ""), "stage2_pred_label": row.get("pred_label", "")} for row in test_base],
        "stage2_pred_label",
    )

    base_rows_test = [r for r in test_base]
    base_pred_test = [str(r.get("pred_label", "")) for r in test_base]
    stage1_pred_test = [r["stage1_pred_label"] for r in test_rows]
    stage2_pred_test = [r["stage2_pred_label"] for r in test_rows]
    true_test = [str(r.get("true_label", "")) for r in test_base]

    summary = {
        "run_dir": str(run_dir),
        "seed": int(seed),
        "input_paths": {
            "train_base": str(train_base_path),
            "train_aux": str(train_aux_path),
            "val_base": str(val_base_path),
            "val_aux": str(val_aux_path),
            "test_base": str(test_base_path),
            "test_aux": str(test_aux_path),
        },
        "feature_names": FEATURE_NAMES,
        "selection": {
            "tau": float(selection["tau"]),
            "pos_weight": float(selection["pos_weight"]),
            "model_key": str(selection["model_key"]),
            "train_override_candidates": int(selection["train_override_candidates"]),
            "train_keep_rate": float(selection["train_keep_rate"]),
            "val_metrics": selection["val_metrics"],
            "val_change_stats": selection["val_change_stats"],
            "val_actions": selection["val_actions"],
        },
        "trainval_stats": trainval_stats,
        "split_metrics": {
            "train": {
                "base": evaluate_rows(
                    [{"true_label": row.get("true_label", ""), "stage2_pred_label": row.get("pred_label", "")} for row in train_base],
                    "stage2_pred_label",
                ),
                "stage1": evaluate_rows(train_rows, "stage1_pred_label"),
                "stage2": evaluate_rows(train_rows, "stage2_pred_label"),
                "change_stats_vs_base": _change_stats(
                    [row.get("true_label", "") for row in train_base],
                    [row.get("pred_label", "") for row in train_base],
                    [row["stage2_pred_label"] for row in train_rows],
                ),
                "change_stats_vs_stage1": _change_stats(
                    [row.get("true_label", "") for row in train_base],
                    [row.get("stage1_pred_label", "") for row in train_rows],
                    [row["stage2_pred_label"] for row in train_rows],
                ),
            },
            "val": {
                "base": evaluate_rows(
                    [{"true_label": row.get("true_label", ""), "stage2_pred_label": row.get("pred_label", "")} for row in val_base],
                    "stage2_pred_label",
                ),
                "stage1": evaluate_rows(val_rows, "stage1_pred_label"),
                "stage2": evaluate_rows(val_rows, "stage2_pred_label"),
                "change_stats_vs_base": _change_stats(
                    [row.get("true_label", "") for row in val_base],
                    [row.get("pred_label", "") for row in val_base],
                    [row["stage2_pred_label"] for row in val_rows],
                ),
                "change_stats_vs_stage1": _change_stats(
                    [row.get("true_label", "") for row in val_base],
                    [row.get("stage1_pred_label", "") for row in val_rows],
                    [row["stage2_pred_label"] for row in val_rows],
                ),
            },
            "test": {
                "base": evaluate_rows(
                    [{"true_label": row.get("true_label", ""), "stage2_pred_label": row.get("pred_label", "")} for row in test_base],
                    "stage2_pred_label",
                ),
                "stage1": evaluate_rows(test_rows, "stage1_pred_label"),
                "stage2": evaluate_rows(test_rows, "stage2_pred_label"),
                "change_stats_vs_base": _change_stats(true_test, base_pred_test, stage2_pred_test),
                "change_stats_vs_stage1": _change_stats(true_test, stage1_pred_test, stage2_pred_test),
            },
        },
        "final_test": {
            "metrics": evaluate_rows(test_rows, "stage2_pred_label"),
            "change_stats_vs_base": _change_stats(true_test, base_pred_test, stage2_pred_test),
            "change_stats_vs_stage1": _change_stats(true_test, stage1_pred_test, stage2_pred_test),
            "actions": test_actions,
            "confusion_summary": {
                "strict": _class_metrics(true_test, stage2_pred_test, LABELS),
                "binary": _binary_metrics([collapse_to_binary(x) for x in true_test], [collapse_to_binary(x) for x in stage2_pred_test]),
            },
        },
    }

    write_json(summary, run_dir / "stage2_summary.json")
    return summary


def _default_run_dir(output_root: Path, dataset_name: str, seed: int) -> Path:
    run_id = f"stage2_{dataset_name}_seed{seed}"
    return output_root / dataset_name / str(seed) / run_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dataset-specific Stage 2 trainer/evaluator.")
    parser.add_argument("--train-base", type=Path, required=True)
    parser.add_argument("--train-aux", type=Path, required=True)
    parser.add_argument("--val-base", type=Path, required=True)
    parser.add_argument("--val-aux", type=Path, required=True)
    parser.add_argument("--test-base", type=Path, required=True)
    parser.add_argument("--test-aux", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=ROOT / "experiments" / "results" / "dataset_runs")
    parser.add_argument("--dataset-name", type=str, default="dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--taus", nargs="+", type=float, default=[0.35, 0.45, 0.55, 0.65])
    parser.add_argument("--pos-weights", nargs="+", type=float, default=[1.0, 2.0, 4.0])
    parser.add_argument("--models", nargs="+", default=["logreg", "xgb"])
    parser.add_argument("--weighting-alpha", type=float, default=1.0)
    parser.add_argument("--ai-lock-threshold", type=float, default=0.5)
    parser.add_argument("--run-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir or _default_run_dir(args.output_root, args.dataset_name, args.seed)
    summary = run_stage2_pipeline(
        train_base_path=args.train_base,
        train_aux_path=args.train_aux,
        val_base_path=args.val_base,
        val_aux_path=args.val_aux,
        test_base_path=args.test_base,
        test_aux_path=args.test_aux,
        run_dir=run_dir,
        taus=args.taus,
        pos_weights=args.pos_weights,
        model_keys=args.models,
        weighting_alpha=args.weighting_alpha,
        ai_lock_threshold=args.ai_lock_threshold,
        seed=args.seed,
    )
    print(json.dumps(
        {
            "run_dir": summary["run_dir"],
            "selection": summary["selection"],
            "final_test": summary["final_test"]["metrics"],
        },
        ensure_ascii=False,
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
