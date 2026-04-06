"""
Agent trust score SSOT (Single Source of Truth)

목적:
    - 런타임(maifs.py, manager_agent.py)과 실험(baselines.py, experiment configs)이
      동일한 소스에서 trust score를 참조한다.
    - settings.py 임포트 side effect(디렉토리 생성)를 피하기 위해 분리된 모듈.
    - I/O 없음, torch/numpy import 없음, __post_init__ 없음.

Override 우선순위:
    1. 실험 YAML cobra.trust_scores (있으면 override)
    2. DEFAULT_TRUST (없으면 기본값)

마이그레이션 정책:
    - 1단계(현재): _DEPRECATED_KEYS는 DeprecationWarning + 조용히 drop
    - 2단계(다음 릴리스): _DEPRECATED_KEYS를 제거하고 unknown과 동일하게 ValueError
"""

import warnings
from typing import Dict, Optional

# 런타임에서 실제로 사용되는 에이전트 키
VALID_AGENT_KEYS: frozenset = frozenset({"frequency", "noise", "fatformer", "spatial"})

# 레거시 키 — 현재 settings.py:244에 "semantic"이 남아 있음
# 1단계에서는 경고 후 drop, 2단계에서 ValueError로 격상
_DEPRECATED_KEYS: frozenset = frozenset({"semantic"})

# 실측 F1 기반 기본값 토론 결과:
# 현재는 기존 동작과의 호환을 위해 원래 값 유지.
# 실측 기반 재설정은 trust-ssot 완료 후 별도 PR에서 수행.
DEFAULT_TRUST: dict = {
    "frequency": 0.85,
    "noise": 0.80,
    "fatformer": 0.85,
    "spatial": 0.85,
}


DEFAULT_TRUST_METRIC_WEIGHTS: dict = {
    "macro_f1": 0.45,
    "balanced_accuracy": 0.45,
    "calibration": 0.10,  # calibration = 1 - ECE
}


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _normalize_metric_weights(metric_weights: Optional[Dict[str, float]]) -> Dict[str, float]:
    """메트릭 가중치 정규화."""
    if not metric_weights:
        return dict(DEFAULT_TRUST_METRIC_WEIGHTS)

    raw = {
        "macro_f1": float(metric_weights.get("macro_f1", 0.0)),
        "balanced_accuracy": float(metric_weights.get("balanced_accuracy", 0.0)),
        "calibration": float(metric_weights.get("calibration", 0.0)),
    }
    total = sum(max(0.0, v) for v in raw.values())
    if total <= 1e-10:
        return dict(DEFAULT_TRUST_METRIC_WEIGHTS)
    return {k: max(0.0, v) / total for k, v in raw.items()}


def _validate_agent_keys(keys: set, label: str) -> None:
    """trust 입력 키 검증 (deprecated는 경고 후 허용)."""
    deprecated_present = keys & _DEPRECATED_KEYS
    if deprecated_present:
        warnings.warn(
            f"Deprecated agent trust keys in {label}: {sorted(deprecated_present)}. "
            f"These keys are ignored. Valid keys: {sorted(VALID_AGENT_KEYS)}",
            DeprecationWarning,
            stacklevel=3,
        )

    unknown = keys - VALID_AGENT_KEYS - _DEPRECATED_KEYS
    if unknown:
        raise ValueError(
            f"Unknown agent keys in {label}: {sorted(unknown)}. "
            f"Valid keys: {sorted(VALID_AGENT_KEYS)}"
        )


def derive_trust_from_metrics(
    agent_metrics: Optional[Dict[str, Dict[str, float]]] = None,
    metric_weights: Optional[Dict[str, float]] = None,
    default_ece: float = 0.25,
    min_trust: float = 0.10,
    max_trust: float = 0.95,
) -> dict:
    """
    에이전트 성능 메트릭에서 trust score를 파생한다.

    기대 입력 형식:
        {
            "frequency": {"macro_f1": 0.82, "balanced_accuracy": 0.84, "ece": 0.07},
            "noise": {...},
            ...
        }
    """
    base = dict(DEFAULT_TRUST)
    if not agent_metrics:
        return base

    if not isinstance(agent_metrics, dict):
        raise TypeError(
            f"agent_metrics must be dict[str, dict[str, float]] or None, got {type(agent_metrics)}"
        )

    _validate_agent_keys(set(agent_metrics.keys()), "agent_metrics")
    weights = _normalize_metric_weights(metric_weights)
    lo, hi = float(min_trust), float(max_trust)
    if lo > hi:
        lo, hi = hi, lo

    for agent in VALID_AGENT_KEYS:
        metrics = agent_metrics.get(agent)
        if not isinstance(metrics, dict):
            continue

        # 누락 메트릭은 기존 기본 trust를 보수적 대체값으로 사용
        f1 = _clip01(metrics.get("macro_f1", base[agent]))
        bal = _clip01(metrics.get("balanced_accuracy", base[agent]))
        ece = float(metrics.get("ece", default_ece))
        if 1.0 < ece <= 100.0:
            ece /= 100.0
        cal = _clip01(1.0 - ece)

        score = (
            weights["macro_f1"] * f1
            + weights["balanced_accuracy"] * bal
            + weights["calibration"] * cal
        )
        base[agent] = max(lo, min(hi, float(score)))

    return base


def resolve_trust(
    yaml_override: "dict | None" = None,
    metrics_override: Optional[Dict[str, Dict[str, float]]] = None,
    metric_weights: Optional[Dict[str, float]] = None,
) -> dict:
    """
    Trust score 결정. override 시 부분 merge + 키 검증.

    Args:
        yaml_override: 실험 YAML cobra.trust_scores 값.
                       None이면 DEFAULT_TRUST 반환.
                       부분 override 가능 — 명시한 키만 교체됨.

    Returns:
        검증된 trust score dict (VALID_AGENT_KEYS 기준, float 보장)

    Raises:
        ValueError: VALID_AGENT_KEYS와 _DEPRECATED_KEYS 모두에 없는 미등록 키 포함 시
    """
    base = derive_trust_from_metrics(
        metrics_override,
        metric_weights=metric_weights,
    )

    if yaml_override is None:
        return base

    if not isinstance(yaml_override, dict):
        raise TypeError(f"yaml_override must be dict or None, got {type(yaml_override)}")

    _validate_agent_keys(set(yaml_override.keys()), "yaml_override")

    # deprecated 제외, valid만 부분 merge
    base.update({k: float(v) for k, v in yaml_override.items() if k in VALID_AGENT_KEYS})
    return base
