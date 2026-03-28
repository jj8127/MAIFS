#!/usr/bin/env python3
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
COMMON_DIR = SCRIPT_DIR.parent
MODEL_DIR = COMMON_DIR / "models" / "arv_stage2"
MANIFEST_PATH = MODEL_DIR / "manifest.json"

CLASSES_3 = ["authentic", "manipulated", "ai_generated"]
CLASSES_2 = ["authentic", "manipulated"]

SUBTYPE_VOCAB = [
    "casia_au",
    "casia_tp",
    "biggan",
    "imd2020_inpainting",
    "opensdi_real",
    "opensdi_partial_fake",
    "opensdi_entire_fake",
    "aigen_proxy_real",
    "aigen_proxy_manipulated",
    "aigen_proxy_ai_generated",
]
SUBTYPE2IDX = {name: idx for idx, name in enumerate(SUBTYPE_VOCAB)}


def mnv2_binary_probs(scores3: dict[str, float]) -> tuple[float, float]:
    p_auth = float(scores3["authentic"])
    p_manip = float(scores3["manipulated"])
    total = p_auth + p_manip
    if total <= 0.0:
        return 0.5, 0.5
    return p_auth / total, p_manip / total


def weighted_binary_scores(
    base_scores3: dict[str, float],
    aux_scores2: dict[str, float],
    base_conf: float,
    aux_conf: float,
) -> np.ndarray:
    p_auth_bin, p_manip_bin = mnv2_binary_probs(base_scores3)
    mnv2_scores = np.array([p_auth_bin, p_manip_bin], dtype=np.float32)
    specm_scores = np.array(
        [float(aux_scores2["authentic"]), float(aux_scores2["manipulated"])],
        dtype=np.float32,
    )
    w_m = 1.0 / max(float(base_conf), 1e-3)
    w_s = 1.0 / max(float(aux_conf), 1e-3)
    combined = w_m * mnv2_scores + w_s * specm_scores
    total = float(combined.sum())
    if total > 0.0:
        combined /= total
    return combined


def subtype_context_features(sub_type: str) -> list[float]:
    token = str(sub_type).strip().lower()
    exact = [0.0] * len(SUBTYPE_VOCAB)
    idx = SUBTYPE2IDX.get(token)
    if idx is not None:
        exact[idx] = 1.0
    family = [
        1.0 if token.startswith("casia") else 0.0,
        1.0 if token == "biggan" else 0.0,
        1.0 if "inpaint" in token or token.startswith("imd2020") else 0.0,
        1.0 if token.startswith("opensdi") else 0.0,
        1.0 if token.startswith("aigen_proxy") else 0.0,
        1.0 if not token else 0.0,
    ]
    return exact + family


def is_non_casia(sub_type: str) -> bool:
    token = str(sub_type).strip().lower()
    return bool(token) and not token.startswith("casia")


def is_ood_manip_family(sub_type: str) -> bool:
    token = str(sub_type).strip().lower()
    return token in {
        "imd2020_inpainting",
        "aigen_proxy_manipulated",
        "opensdi_partial_fake",
        "opensdi_entire_fake",
    }


def build_richer_feature(
    base_scores3: dict[str, float],
    aux_scores2: dict[str, float],
    base_conf: float,
    aux_conf: float,
    sub_type: str = "",
) -> list[float]:
    p_auth = float(base_scores3["authentic"])
    p_manip = float(base_scores3["manipulated"])
    p_aigen = float(base_scores3.get("ai_generated", 0.0))
    p_auth_bin, p_manip_bin = mnv2_binary_probs(base_scores3)
    specm_a = float(aux_scores2["authentic"])
    specm_m = float(aux_scores2["manipulated"])
    ic_auth, ic_manip = weighted_binary_scores(base_scores3, aux_scores2, base_conf, aux_conf)
    w_m = 1.0 / max(float(base_conf), 1e-3)
    w_s = 1.0 / max(float(aux_conf), 1e-3)
    ai_margin = p_aigen - max(p_auth, p_manip)
    mnv2_pred = 1 if p_manip_bin >= p_auth_bin else 0
    ic_pred = 1 if ic_manip >= ic_auth else 0

    base = [
        p_auth_bin,
        p_manip_bin,
        specm_a,
        specm_m,
        float(ic_auth),
        float(ic_manip),
        float(ic_manip - ic_auth),
        float(specm_m - p_manip_bin),
        float(specm_a - p_auth_bin),
        float(ai_margin),
        float(base_conf),
        float(aux_conf),
        float(w_s / max(w_m, 1e-6)),
        float(mnv2_pred),
        float(ic_pred),
        float(abs(specm_m - p_manip_bin)),
    ] + subtype_context_features(sub_type)

    mnv2_margin = abs(p_manip_bin - p_auth_bin)
    specm_margin = abs(specm_m - specm_a)
    ic_margin = abs(float(ic_manip) - float(ic_auth))
    conf_gap = float(aux_conf) - float(base_conf)
    conf_prod = float(aux_conf) * float(base_conf)
    toward_auth = 1.0 if mnv2_pred == 1 and ic_pred == 0 else 0.0
    toward_manip = 1.0 if mnv2_pred == 0 and ic_pred == 1 else 0.0

    richer = base + [
        float(mnv2_margin),
        float(specm_margin),
        float(ic_margin),
        float(specm_margin - mnv2_margin),
        float(conf_gap),
        float(conf_prod),
        float(mnv2_pred == 0),
        float(mnv2_pred == 1),
        float(ic_pred == 0),
        float(ic_pred == 1),
        float(toward_auth),
        float(toward_manip),
        float(is_non_casia(sub_type)),
        float(is_ood_manip_family(sub_type)),
    ]
    return richer


def is_ai_lock(base_scores3: dict[str, float]) -> bool:
    p_aigen = float(base_scores3.get("ai_generated", 0.0))
    pred = max(base_scores3, key=base_scores3.get)
    return pred == "ai_generated" or p_aigen > 0.5


@dataclass
class ARVDecision:
    base_bin_label: str
    stage1_bin_label: str
    final_label: str
    action: str
    override_present: bool
    keep_prob: float | None
    tau: float | None
    feature_len: int
    feature_ms: float
    predict_ms: float
    total_ms: float


class ARVStage2Runtime:
    def __init__(self, manifest_path: Path = MANIFEST_PATH):
        import xgboost as xgb

        self.xgb = xgb
        self.manifest_path = manifest_path
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.model_dir = manifest_path.parent
        self._cache: dict[str, Any] = {}

    @property
    def model_keys(self) -> list[str]:
        return list(self.manifest["models"].keys())

    def model_meta(self, model_key: str) -> dict[str, Any]:
        return self.manifest["models"][model_key]

    def load_model(self, model_key: str):
        if model_key in self._cache:
            return self._cache[model_key]
        booster = self.xgb.Booster()
        booster.load_model(str(self.model_dir / self.model_meta(model_key)["path"]))
        self._cache[model_key] = booster
        return booster

    def keep_prob(self, model_key: str, feature: list[float]) -> float:
        booster = self.load_model(model_key)
        x = np.asarray([feature], dtype=np.float32)
        dmat = self.xgb.DMatrix(x)
        pred = booster.predict(dmat)
        return float(pred[0])

    def decide(
        self,
        model_key: str,
        base_scores3: dict[str, float],
        aux_scores2: dict[str, float],
        base_conf: float,
        aux_conf: float,
        sub_type: str = "",
        force_stage2: bool = False,
    ) -> ARVDecision:
        if is_ai_lock(base_scores3):
            return ARVDecision(
                base_bin_label="ai_generated",
                stage1_bin_label="ai_generated",
                final_label="ai_generated",
                action="ai_lock",
                override_present=False,
                keep_prob=None,
                tau=None,
                feature_len=0,
                feature_ms=0.0,
                predict_ms=0.0,
                total_ms=0.0,
            )

        p_auth_bin, p_manip_bin = mnv2_binary_probs(base_scores3)
        base_bin_label = "manipulated" if p_manip_bin >= p_auth_bin else "authentic"
        ic_scores = weighted_binary_scores(base_scores3, aux_scores2, base_conf, aux_conf)
        stage1_bin_label = "manipulated" if float(ic_scores[1]) >= float(ic_scores[0]) else "authentic"
        override_present = stage1_bin_label != base_bin_label

        if not override_present and not force_stage2:
            return ARVDecision(
                base_bin_label=base_bin_label,
                stage1_bin_label=stage1_bin_label,
                final_label=stage1_bin_label,
                action="no_override_keep_stage1",
                override_present=False,
                keep_prob=None,
                tau=None,
                feature_len=0,
                feature_ms=0.0,
                predict_ms=0.0,
                total_ms=0.0,
            )

        t0 = time.perf_counter()
        feature = build_richer_feature(
            base_scores3=base_scores3,
            aux_scores2=aux_scores2,
            base_conf=base_conf,
            aux_conf=aux_conf,
            sub_type=sub_type,
        )
        t1 = time.perf_counter()
        keep_prob = self.keep_prob(model_key, feature)
        t2 = time.perf_counter()

        tau = float(self.model_meta(model_key)["tau"])
        if keep_prob >= tau:
            final_label = stage1_bin_label
            action = "keep_change" if override_present else "forced_keep_no_override"
        else:
            final_label = base_bin_label
            action = "revert_to_base" if override_present else "forced_revert_no_override"

        return ARVDecision(
            base_bin_label=base_bin_label,
            stage1_bin_label=stage1_bin_label,
            final_label=final_label,
            action=action,
            override_present=override_present,
            keep_prob=float(keep_prob),
            tau=tau,
            feature_len=len(feature),
            feature_ms=(t1 - t0) * 1000.0,
            predict_ms=(t2 - t1) * 1000.0,
            total_ms=(t2 - t0) * 1000.0,
        )
