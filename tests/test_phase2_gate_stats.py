"""
Phase2 게이트 통계 보강 로직 단위 테스트.
"""

from __future__ import annotations

import math

from experiments.evaluate_phase2_gate import (
    _extract_pooled_mcnemar,
    _extract_sign_stats,
    evaluate_gate,
)
from experiments.evaluate_phase2_gate_profiles import _parse_profiles
from experiments.evaluate_phase2_gate_profiles import _profiles_from_config
from experiments.evaluate_phase2_gate_profiles import _resolve_profile_names


def test_extract_sign_stats_from_runs_fallback():
    summary = {
        "n_runs": 5,
        "runs": [
            {"f1_diff": 0.02},
            {"f1_diff": 0.01},
            {"f1_diff": 0.03},
            {"f1_diff": -0.01},
            {"f1_diff": 0.0},
        ],
        "aggregate": {},
    }
    stats = _extract_sign_stats(summary)
    assert stats["positive_count"] == 3
    assert stats["negative_count"] == 1
    assert stats["zero_count"] == 1
    assert math.isclose(stats["sign_test_pvalue"], 0.625, rel_tol=1e-9, abs_tol=1e-12)


def test_extract_sign_stats_prefers_aggregate():
    summary = {
        "n_runs": 4,
        "runs": [{"f1_diff": -1.0}],  # aggregate 우선 사용 확인용 노이즈
        "aggregate": {
            "positive_count": 8,
            "negative_count": 1,
            "zero_count": 1,
            "sign_test_pvalue": 0.0390625,
        },
    }
    stats = _extract_sign_stats(summary)
    assert stats["positive_count"] == 8
    assert stats["negative_count"] == 1
    assert stats["zero_count"] == 1
    assert math.isclose(stats["sign_test_pvalue"], 0.0390625, rel_tol=1e-9, abs_tol=1e-12)


def test_extract_pooled_mcnemar_from_runs():
    summary = {
        "aggregate": {},
        "runs": [
            {"mcnemar_b": 10, "mcnemar_c": 14},
            {"mcnemar_b": 3, "mcnemar_c": 7},
        ],
    }
    pooled = _extract_pooled_mcnemar(summary)
    assert pooled is not None
    assert pooled["source"] == "runs"
    assert pooled["b"] == 13
    assert pooled["c"] == 21
    assert math.isclose(pooled["statistic"], 1.4411764705882353, rel_tol=1e-9, abs_tol=1e-12)
    assert math.isclose(pooled["pvalue"], 0.22994905679421346, rel_tol=1e-9, abs_tol=1e-12)
    assert pooled["significant"] is False


def test_evaluate_gate_sign_driven_passes():
    candidate = {
        "n_runs": 10,
        "aggregate": {
            "f1_diff_mean": 0.02,
            "significant_count": 0,
        },
        "runs": [{"f1_diff": d} for d in ([0.02] * 8 + [-0.01, 0.0])],
    }
    baseline = {
        "aggregate": {
            "f1_diff_mean": 0.003,
        }
    }
    report = evaluate_gate(
        candidate=candidate,
        baseline=baseline,
        min_runs=10,
        min_f1_diff_mean=0.01,
        min_significant_count=0,
        min_improvement_over_baseline=0.0,
        min_positive_seed_count=7,
        max_sign_test_pvalue=0.05,
    )
    assert report["gate_pass"] is True
    assert report["candidate"]["sign_stats"]["positive_count"] == 8
    assert math.isclose(
        report["candidate"]["sign_stats"]["sign_test_pvalue"],
        0.0390625,
        rel_tol=1e-9,
        abs_tol=1e-12,
    )


def test_evaluate_gate_can_require_pooled_mcnemar_significance():
    candidate = {
        "n_runs": 3,
        "aggregate": {
            "f1_diff_mean": 0.02,
            "significant_count": 0,
        },
        "runs": [
            {"f1_diff": 0.02, "mcnemar_b": 10, "mcnemar_c": 14},
            {"f1_diff": 0.03, "mcnemar_b": 3, "mcnemar_c": 7},
            {"f1_diff": 0.01, "mcnemar_b": 1, "mcnemar_c": 1},
        ],
    }
    report = evaluate_gate(
        candidate=candidate,
        baseline=None,
        min_runs=3,
        min_f1_diff_mean=0.0,
        min_significant_count=0,
        require_pooled_mcnemar_significant=True,
    )
    assert report["candidate"]["pooled_mcnemar"] is not None
    assert report["candidate"]["pooled_mcnemar"]["significant"] is False
    assert report["gate_pass"] is False


def test_evaluate_gate_downside_constraints_fail_when_losses_are_large():
    candidate = {
        "n_runs": 4,
        "aggregate": {
            "f1_diff_mean": -0.01,
            "significant_count": 0,
        },
        "runs": [
            {"f1_diff": 0.01},
            {"f1_diff": -0.02},
            {"f1_diff": -0.03},
            {"f1_diff": 0.0},
        ],
    }
    report = evaluate_gate(
        candidate=candidate,
        baseline=None,
        min_runs=4,
        min_f1_diff_mean=-1.0,
        min_significant_count=0,
        max_negative_rate=0.25,
        max_downside_mean=0.01,
        max_cvar_downside=0.025,
        max_worst_case_loss=0.04,
        downside_cvar_alpha=0.25,
    )
    assert report["gate_pass"] is False
    checks = {c["name"]: c for c in report["checks"]}
    assert checks["max_negative_rate"]["pass"] is False
    assert math.isclose(checks["max_negative_rate"]["actual"], 0.5, rel_tol=1e-9, abs_tol=1e-12)


def test_evaluate_gate_downside_constraints_can_pass():
    candidate = {
        "n_runs": 4,
        "aggregate": {
            "f1_diff_mean": 0.01125,
            "significant_count": 0,
        },
        "runs": [
            {"f1_diff": 0.03},
            {"f1_diff": 0.02},
            {"f1_diff": -0.005},
            {"f1_diff": 0.0},
        ],
    }
    report = evaluate_gate(
        candidate=candidate,
        baseline=None,
        min_runs=4,
        min_f1_diff_mean=0.0,
        min_significant_count=0,
        max_negative_rate=0.4,
        max_downside_mean=0.002,
        max_cvar_downside=0.006,
        max_worst_case_loss=0.01,
        downside_cvar_alpha=0.25,
    )
    assert report["gate_pass"] is True
    downside = report["candidate"]["downside_stats"]
    assert math.isclose(downside["negative_rate"], 0.25, rel_tol=1e-9, abs_tol=1e-12)
    assert math.isclose(downside["downside_mean"], 0.00125, rel_tol=1e-9, abs_tol=1e-12)
    assert math.isclose(downside["cvar_downside"], 0.005, rel_tol=1e-9, abs_tol=1e-12)
    assert math.isclose(downside["worst_case_loss"], 0.005, rel_tol=1e-9, abs_tol=1e-12)


def test_parse_profiles_success():
    assert _parse_profiles("strict,sign_driven") == ["strict", "sign_driven"]


def test_parse_profiles_unknown_raises():
    try:
        _parse_profiles("strict,unknown_profile")
    except ValueError as e:
        assert "Unknown profiles" in str(e)
    else:
        raise AssertionError("Expected ValueError for unknown profile")


def test_resolve_profile_names_auto_with_active():
    presets = {
        "strict": {"min_runs": 10},
        "pooled_relaxed": {"min_runs": 10},
    }
    names = _resolve_profile_names(raw_profiles="auto", profile_presets=presets, active_profile="pooled_relaxed")
    assert names == ["pooled_relaxed"]


def test_resolve_profile_names_auto_without_active_uses_all():
    presets = {
        "strict": {"min_runs": 10},
        "sign_driven": {"min_runs": 10},
        "pooled_relaxed": {"min_runs": 10},
    }
    names = _resolve_profile_names(raw_profiles="auto", profile_presets=presets, active_profile=None)
    assert names == ["strict", "sign_driven", "pooled_relaxed"]


def test_resolve_profile_names_auto_invalid_active_raises():
    presets = {
        "strict": {"min_runs": 10},
    }
    try:
        _resolve_profile_names(raw_profiles="auto", profile_presets=presets, active_profile="pooled_relaxed")
    except ValueError as e:
        assert "active_gate_profile" in str(e)
    else:
        raise AssertionError("Expected ValueError for invalid active profile")


def test_profiles_from_config_gate_profiles_priority():
    cfg = {
        "protocol": {
            "gate": {"min_runs": 999},
            "gate_profiles": {
                "strict": {
                    "min_runs": 10,
                    "min_f1_diff_mean": 0.01,
                    "min_significant_count": 3,
                },
                "sign_driven": {
                    "min_runs": 10,
                    "min_f1_diff_mean": 0.01,
                    "min_significant_count": 0,
                    "min_positive_seed_count": 7,
                    "max_sign_test_pvalue": 0.05,
                },
            },
        }
    }
    presets = _profiles_from_config(cfg)
    assert sorted(presets.keys()) == ["sign_driven", "strict"]
    assert presets["strict"]["min_runs"] == 10
    assert presets["sign_driven"]["min_positive_seed_count"] == 7
    assert math.isclose(presets["sign_driven"]["max_sign_test_pvalue"], 0.05, rel_tol=1e-9, abs_tol=1e-12)


def test_profiles_from_config_fallback_to_gate():
    cfg = {
        "protocol": {
            "gate": {
                "min_runs": 11,
                "min_f1_diff_mean": 0.02,
                "min_significant_count": 1,
                "min_improvement_over_baseline": 0.0,
            }
        }
    }
    presets = _profiles_from_config(cfg)
    assert list(presets.keys()) == ["strict"]
    assert presets["strict"]["min_runs"] == 11
    assert math.isclose(presets["strict"]["min_f1_diff_mean"], 0.02, rel_tol=1e-9, abs_tol=1e-12)
