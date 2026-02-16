"""
Path A gate tuning 유틸 단위 테스트.
"""

from __future__ import annotations

from experiments.tune_phase2_gate_profile import _parse_labeled_path
from experiments.tune_phase2_gate_profile import _parse_optional_float_list
from experiments.tune_phase2_gate_profile import _sort_key


def test_parse_labeled_path():
    label, path = _parse_labeled_path("kfold25:experiments/results/x.json")
    assert label == "kfold25"
    assert path == "experiments/results/x.json"


def test_parse_optional_float_list():
    vals = _parse_optional_float_list("none,0.5,0.3")
    assert vals == [None, 0.5, 0.3]


def test_sort_key_prefers_higher_score_then_stricter_profile():
    a = {
        "score": 4,
        "profile": {
            "min_f1_diff_mean": 0.002,
            "max_sign_test_pvalue": 0.35,
            "max_pooled_mcnemar_pvalue": 0.7,
            "min_improvement_over_baseline": -0.001,
        },
    }
    b = {
        "score": 4,
        "profile": {
            "min_f1_diff_mean": 0.001,
            "max_sign_test_pvalue": 0.35,
            "max_pooled_mcnemar_pvalue": 0.7,
            "min_improvement_over_baseline": -0.001,
        },
    }
    assert _sort_key(a) > _sort_key(b)
