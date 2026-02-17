"""
Path A split protocol 선택기 단위 테스트.
"""

from __future__ import annotations

from experiments.select_patha_split_protocol import _parse_candidate_token
from experiments.select_patha_split_protocol import _recommend


def test_parse_candidate_token():
    label, path = _parse_candidate_token("kfold25:experiments/results/x.json")
    assert label == "kfold25"
    assert path == "experiments/results/x.json"


def test_recommend_prefers_higher_f1_then_lower_sign_p():
    candidates = [
        {
            "label": "random25",
            "summary_path": "a.json",
            "n_runs": 25,
            "f1_diff_mean": -0.0039,
            "f1_diff_std": 0.0224,
            "positive_count": 8,
            "negative_count": 12,
            "zero_count": 5,
            "directional_margin": -4,
            "sign_test_pvalue": 0.5034,
            "pooled_mcnemar_pvalue": 0.4996,
        },
        {
            "label": "kfold25",
            "summary_path": "b.json",
            "n_runs": 25,
            "f1_diff_mean": 0.0032,
            "f1_diff_std": 0.0181,
            "positive_count": 14,
            "negative_count": 8,
            "zero_count": 3,
            "directional_margin": 6,
            "sign_test_pvalue": 0.2863,
            "pooled_mcnemar_pvalue": 0.6906,
        },
    ]
    rec = _recommend(candidates)
    assert rec["recommended_label"] == "kfold25"
    assert rec["ranking"] == ["kfold25", "random25"]


def test_recommend_loss_averse_prefers_lower_downside_risk():
    candidates = [
        {
            "label": "mean_high_risk",
            "summary_path": "a.json",
            "n_runs": 25,
            "f1_diff_mean": 0.004,
            "f1_diff_std": 0.03,
            "positive_count": 13,
            "negative_count": 12,
            "zero_count": 0,
            "directional_margin": 1,
            "sign_test_pvalue": 0.8,
            "pooled_mcnemar_pvalue": 0.8,
            "negative_rate": 0.48,
            "downside_mean": 0.011,
            "cvar_downside": 0.03,
            "worst_case_loss": 0.05,
        },
        {
            "label": "mean_mid_safe",
            "summary_path": "b.json",
            "n_runs": 25,
            "f1_diff_mean": 0.001,
            "f1_diff_std": 0.01,
            "positive_count": 12,
            "negative_count": 10,
            "zero_count": 3,
            "directional_margin": 2,
            "sign_test_pvalue": 0.3,
            "pooled_mcnemar_pvalue": 0.6,
            "negative_rate": 0.30,
            "downside_mean": 0.004,
            "cvar_downside": 0.012,
            "worst_case_loss": 0.02,
        },
    ]
    rec = _recommend(candidates, objective="loss_averse")
    assert rec["recommended_label"] == "mean_mid_safe"
    assert rec["ranking"] == ["mean_mid_safe", "mean_high_risk"]
