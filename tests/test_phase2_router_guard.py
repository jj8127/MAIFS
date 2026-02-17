"""
Phase2 router guard fallback 로직 단위 테스트.
"""

from __future__ import annotations

import math

import numpy as np

from experiments.run_phase2_patha import (
    _apply_router_guard,
    _apply_non_regression_veto,
    _router_guard_score,
    _select_best_models_for_comparison,
    _select_guard_threshold,
)


def test_router_guard_score_modes_are_finite_and_monotonic_on_peaky_weights():
    weights = np.array(
        [
            [0.95, 0.03, 0.01, 0.01],  # high confidence
            [0.40, 0.30, 0.20, 0.10],  # medium
            [0.26, 0.25, 0.25, 0.24],  # near-uniform
        ],
        dtype=float,
    )

    for mode in ["max_weight", "neg_entropy", "top2_margin", "max_minus_entropy"]:
        s = _router_guard_score(weights, mode=mode)
        assert s.shape == (3,)
        assert np.isfinite(s).all()
        assert s[0] > s[2]


def test_select_guard_threshold_prefers_conservative_route_when_f1_ties():
    y_val = np.array([0, 0, 1, 1], dtype=int)
    p1_val = np.array([0, 0, 1, 1], dtype=int)  # perfect
    p2_val = np.array([0, 1, 1, 0], dtype=int)
    scores = np.array([0.90, 0.80, 0.20, 0.10], dtype=float)
    thresholds = [0.0, 0.5, 0.85, float("inf")]

    selection = _select_guard_threshold(
        y_val=y_val,
        phase1_val_pred=p1_val,
        phase2_val_pred=p2_val,
        scores_val=scores,
        thresholds=thresholds,
        min_val_gain=0.0,
    )
    assert math.isinf(selection["threshold"])
    assert math.isclose(selection["val_macro_f1"], 1.0, rel_tol=1e-9, abs_tol=1e-12)
    assert math.isclose(selection["phase2_route_rate"], 0.0, rel_tol=1e-9, abs_tol=1e-12)


def test_select_guard_threshold_chooses_mixed_policy_when_it_improves_val_f1():
    y_val = np.array([0, 0, 1, 1], dtype=int)
    p1_val = np.array([0, 1, 1, 0], dtype=int)
    p2_val = np.array([0, 0, 0, 1], dtype=int)
    scores = np.array([0.90, 0.80, 0.20, 0.10], dtype=float)
    thresholds = [0.0, 0.5, float("inf")]

    selection = _select_guard_threshold(
        y_val=y_val,
        phase1_val_pred=p1_val,
        phase2_val_pred=p2_val,
        scores_val=scores,
        thresholds=thresholds,
        min_val_gain=0.0,
    )
    assert math.isclose(selection["threshold"], 0.5, rel_tol=1e-9, abs_tol=1e-12)
    assert selection["val_gain_vs_phase1"] > 0.0

    y_mixed, route_rate = _apply_router_guard(
        phase1_pred=p1_val,
        phase2_pred=p2_val,
        scores=scores,
        threshold=selection["threshold"],
    )
    assert math.isclose(route_rate, 0.5, rel_tol=1e-9, abs_tol=1e-12)
    assert np.array_equal(y_mixed, np.array([0, 0, 1, 0], dtype=int))


def test_select_guard_threshold_respects_max_route_rate_constraint():
    y_val = np.array([0, 0, 1, 1], dtype=int)
    p1_val = np.array([0, 1, 1, 0], dtype=int)
    p2_val = np.array([0, 0, 0, 1], dtype=int)
    scores = np.array([0.90, 0.80, 0.20, 0.10], dtype=float)
    thresholds = [0.0, 0.5, float("inf")]

    unconstrained = _select_guard_threshold(
        y_val=y_val,
        phase1_val_pred=p1_val,
        phase2_val_pred=p2_val,
        scores_val=scores,
        thresholds=thresholds,
        min_val_gain=0.0,
    )
    assert math.isclose(unconstrained["threshold"], 0.5, rel_tol=1e-9, abs_tol=1e-12)
    assert math.isclose(unconstrained["phase2_route_rate"], 0.5, rel_tol=1e-9, abs_tol=1e-12)

    constrained = _select_guard_threshold(
        y_val=y_val,
        phase1_val_pred=p1_val,
        phase2_val_pred=p2_val,
        scores_val=scores,
        thresholds=thresholds,
        min_val_gain=0.0,
        max_route_rate=0.3,
    )
    assert math.isinf(constrained["threshold"])
    assert math.isclose(constrained["phase2_route_rate"], 0.0, rel_tol=1e-9, abs_tol=1e-12)


def test_select_guard_threshold_blocks_routing_when_raw_phase2_val_is_worse():
    y_val = np.array([0, 0, 1, 1], dtype=int)
    p1_val = np.array([0, 0, 1, 1], dtype=int)
    p2_val = np.array([0, 1, 1, 0], dtype=int)
    scores = np.array([0.90, 0.80, 0.20, 0.10], dtype=float)
    thresholds = [0.0, 0.5, float("inf")]

    blocked = _select_guard_threshold(
        y_val=y_val,
        phase1_val_pred=p1_val,
        phase2_val_pred=p2_val,
        scores_val=scores,
        thresholds=thresholds,
        min_val_gain=0.0,
        min_phase2_val_gain=0.0,
    )
    assert blocked["raw_phase2_gate_pass"] is False
    assert math.isinf(blocked["threshold"])
    assert math.isclose(blocked["phase2_route_rate"], 0.0, rel_tol=1e-9, abs_tol=1e-12)
    assert blocked["raw_phase2_val_gain"] < 0.0


def test_apply_non_regression_veto_switches_to_phase1_fallback_on_regression():
    phase1_results = {"gradient_boosting": {"macro_f1": 0.81}}
    phase2_results = {"hybrid_guard_gradient_boosting": {"macro_f1": 0.77}}
    phase1_preds = {"gradient_boosting": np.array([0, 1, 2], dtype=int)}
    phase2_preds = {"hybrid_guard_gradient_boosting": np.array([0, 0, 2], dtype=int)}

    best_p2, info = _apply_non_regression_veto(
        phase1_results=phase1_results,
        phase1_preds=phase1_preds,
        phase2_results=phase2_results,
        phase2_preds=phase2_preds,
        best_p1="gradient_boosting",
        best_p2="hybrid_guard_gradient_boosting",
        tolerance=0.0,
    )

    assert info["applied"] is True
    assert best_p2 == "phase1_fallback_gradient_boosting"
    assert "phase1_fallback_gradient_boosting" in phase2_results
    assert np.array_equal(phase2_preds["phase1_fallback_gradient_boosting"], phase1_preds["gradient_boosting"])


def test_apply_non_regression_veto_keeps_phase2_when_not_regressing():
    phase1_results = {"logistic_regression": {"macro_f1": 0.80}}
    phase2_results = {"hybrid_guard_logistic_regression": {"macro_f1": 0.82}}
    phase1_preds = {"logistic_regression": np.array([0, 1, 2], dtype=int)}
    phase2_preds = {"hybrid_guard_logistic_regression": np.array([0, 1, 2], dtype=int)}

    best_p2, info = _apply_non_regression_veto(
        phase1_results=phase1_results,
        phase1_preds=phase1_preds,
        phase2_results=phase2_results,
        phase2_preds=phase2_preds,
        best_p1="logistic_regression",
        best_p2="hybrid_guard_logistic_regression",
        tolerance=0.0,
    )

    assert info["applied"] is False
    assert best_p2 == "hybrid_guard_logistic_regression"


def test_select_best_models_for_comparison_uses_val_results_and_hybrid_scope():
    phase1_val = {
        "logistic_regression": {"macro_f1": 0.70},
        "gradient_boosting": {"macro_f1": 0.76},
        "mlp": {"macro_f1": 0.74},
    }
    phase2_val = {
        "logistic_regression": {"macro_f1": 0.80},  # would win under all scope
        "hybrid_guard_logistic_regression": {"macro_f1": 0.75},
        "hybrid_guard_gradient_boosting": {"macro_f1": 0.78},  # should win under hybrid_only
    }

    best_p1, best_p2, selected, resolved_scope = _select_best_models_for_comparison(
        phase1_val_results=phase1_val,
        phase2_val_results=phase2_val,
        guard_enabled=True,
        selection_scope="hybrid_only",
        model_key_prefix="hybrid_guard",
    )

    assert best_p1 == "gradient_boosting"
    assert best_p2 == "hybrid_guard_gradient_boosting"
    assert resolved_scope == "hybrid_only"
    assert set(selected.keys()) == {"hybrid_guard_logistic_regression", "hybrid_guard_gradient_boosting"}


def test_select_best_models_for_comparison_falls_back_when_no_hybrid_candidate():
    phase1_val = {"logistic_regression": {"macro_f1": 0.70}}
    phase2_val = {"logistic_regression": {"macro_f1": 0.72}}

    _best_p1, best_p2, selected, resolved_scope = _select_best_models_for_comparison(
        phase1_val_results=phase1_val,
        phase2_val_results=phase2_val,
        guard_enabled=True,
        selection_scope="hybrid_only",
        model_key_prefix="hybrid_guard",
    )

    assert best_p2 == "logistic_regression"
    assert resolved_scope == "all"
    assert set(selected.keys()) == {"logistic_regression"}
