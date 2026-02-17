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


def resolve_trust(yaml_override: "dict | None" = None) -> dict:
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
    base = dict(DEFAULT_TRUST)

    if yaml_override is None:
        return base

    if not isinstance(yaml_override, dict):
        raise TypeError(f"yaml_override must be dict or None, got {type(yaml_override)}")

    # 1단계: deprecated 키 — 경고 후 조용히 제거
    deprecated_present = set(yaml_override) & _DEPRECATED_KEYS
    if deprecated_present:
        warnings.warn(
            f"Deprecated agent trust keys will be removed in a future release: "
            f"{sorted(deprecated_present)}. These keys are ignored. "
            f"Valid keys: {sorted(VALID_AGENT_KEYS)}",
            DeprecationWarning,
            stacklevel=2,
        )

    # 완전히 미등록 키 — 즉시 hard fail
    unknown = set(yaml_override) - VALID_AGENT_KEYS - _DEPRECATED_KEYS
    if unknown:
        raise ValueError(
            f"Unknown agent keys in trust override: {sorted(unknown)}. "
            f"Valid keys: {sorted(VALID_AGENT_KEYS)}"
        )

    # deprecated 제외, valid만 부분 merge
    base.update({k: float(v) for k, v in yaml_override.items() if k in VALID_AGENT_KEYS})
    return base
