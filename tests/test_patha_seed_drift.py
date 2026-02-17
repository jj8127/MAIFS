"""
Path A seed-block drift 분석기 단위 테스트.
"""

from __future__ import annotations

import math

from experiments.analyze_patha_seed_drift import _extract_sign_stats
from experiments.analyze_patha_seed_drift import compare_subgroup
from experiments.analyze_patha_seed_drift import describe_summary


def test_extract_sign_stats_fallback_from_runs():
    summary = {
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


def test_describe_summary_includes_pair_counts_and_seed_rows():
    summary = {
        "n_runs": 3,
        "seeds": [42, 43, 44],
        "aggregate": {
            "f1_diff_mean": 0.01,
            "f1_diff_std": 0.02,
            "f1_diff_min": -0.01,
            "f1_diff_max": 0.03,
            "significant_count": 0,
            "positive_count": 2,
            "negative_count": 1,
            "zero_count": 0,
            "sign_test_pvalue": 1.0,
        },
        "runs": [
            {
                "seed": 43,
                "f1_diff": -0.01,
                "phase1_best": "mlp",
                "phase2_best": "gradient_boosting",
                "mcnemar_pvalue": 1.0,
                "significant": False,
            },
            {
                "seed": 42,
                "f1_diff": 0.01,
                "phase1_best": "mlp",
                "phase2_best": "gradient_boosting",
                "mcnemar_pvalue": 0.7,
                "significant": False,
            },
            {
                "seed": 44,
                "f1_diff": 0.03,
                "phase1_best": "logistic_regression",
                "phase2_best": "mlp",
                "mcnemar_pvalue": 0.9,
                "significant": False,
            },
        ],
    }
    desc = describe_summary(summary)
    assert desc["n_runs"] == 3
    assert desc["f1_diff_distribution"]["count"] == 3
    assert desc["f1_diff_distribution"]["sorted"] == [-0.01, 0.01, 0.03]
    assert desc["model_pair_counts"][0] == {
        "phase1_best": "mlp",
        "phase2_best": "gradient_boosting",
        "count": 2,
    }
    assert [r["seed"] for r in desc["runs"]] == [42, 43, 44]


def test_compare_subgroup_detects_sign_flip():
    reference = {
        "delta_phase2_minus_phase1": {
            "by_sub_type_macro_f1": {"biggan": 0.01, "casia_tp": 0.02},
            "by_true_label_macro_f1": {"ai_generated": 0.01, "manipulated": 0.02},
        }
    }
    candidate = {
        "delta_phase2_minus_phase1": {
            "by_sub_type_macro_f1": {"biggan": -0.02, "casia_tp": 0.01},
            "by_true_label_macro_f1": {"ai_generated": -0.02, "manipulated": 0.01},
        }
    }
    cmp_res = compare_subgroup(reference, candidate)
    assert cmp_res["by_sub_type_macro_f1"]["biggan"]["sign_flip"] is True
    assert cmp_res["by_sub_type_macro_f1"]["casia_tp"]["sign_flip"] is False
    assert math.isclose(
        cmp_res["by_sub_type_macro_f1"]["casia_tp"]["delta_candidate_minus_reference"],
        -0.01,
        rel_tol=1e-9,
        abs_tol=1e-12,
    )
