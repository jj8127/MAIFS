"""
Path A fixed-kfold 변동성 분석기 단위 테스트.
"""

from __future__ import annotations

from experiments.analyze_patha_kfold_variance import _parse_labeled_path
from experiments.analyze_patha_kfold_variance import _stat_row
from experiments.analyze_patha_kfold_variance import _summarize_kfold_summary


def test_parse_labeled_path():
    label, path = _parse_labeled_path("kfold25:experiments/results/x.json")
    assert label == "kfold25"
    assert path == "experiments/results/x.json"


def test_stat_row_counts_signs():
    row = _stat_row([0.02, -0.01, 0.0, 0.03])
    assert row["n"] == 4
    assert row["positive_count"] == 2
    assert row["negative_count"] == 1
    assert row["zero_count"] == 1


def test_summarize_kfold_summary_seed_and_fold_stats():
    summary = {
        "n_runs": 4,
        "split_protocol": {"strategy": "kfold"},
        "runs": [
            {"split_seed": 300, "test_fold": 0, "f1_diff": 0.02},
            {"split_seed": 300, "test_fold": 1, "f1_diff": -0.01},
            {"split_seed": 301, "test_fold": 0, "f1_diff": 0.01},
            {"split_seed": 301, "test_fold": 1, "f1_diff": -0.02},
        ],
        "aggregate": {
            "f1_diff_mean": 0.0,
            "f1_diff_std": 0.015,
            "positive_count": 2,
            "negative_count": 2,
            "zero_count": 0,
            "sign_test_pvalue": 1.0,
            "pooled_mcnemar_pvalue": 0.5,
        },
    }
    rep = _summarize_kfold_summary(label="kfold4", summary_path="x.json", summary=summary)
    assert rep["n_runs"] == 4
    assert rep["split_strategy"] == "kfold"
    assert len(rep["split_seed_stats"]["rows"]) == 2
    assert len(rep["test_fold_stats"]["rows"]) == 2
    assert "split_seed_mean_sign_mixed" in rep["risk_flags"]
    assert "test_fold_mean_sign_mixed" in rep["risk_flags"]
    assert rep["aggregate"]["positive_count"] == 2
    assert rep["aggregate"]["negative_count"] == 2

