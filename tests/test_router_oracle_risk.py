"""
Path A oracle risk-aware target 동작 테스트.
"""

from __future__ import annotations

import numpy as np

from src.meta.collector import build_empirical_simulator
from src.meta.router import OracleWeightComputer, OracleWeightConfig
from src.meta.simulator import SimulatedOutput


def _train_sample(label: str, verdict: str) -> SimulatedOutput:
    return SimulatedOutput(
        true_label=label,
        sub_type=f"{label}_stub",
        agent_verdicts={
            "frequency": verdict,
            "noise": verdict,
            "fatformer": verdict,
            "spatial": verdict,
        },
        agent_confidences={
            "frequency": 0.7,
            "noise": 0.7,
            "fatformer": 0.7,
            "spatial": 0.7,
        },
    )


def _train_sample_custom(label: str, verdicts: dict[str, str]) -> SimulatedOutput:
    return SimulatedOutput(
        true_label=label,
        sub_type=f"{label}_stub",
        agent_verdicts=dict(verdicts),
        agent_confidences={
            "frequency": 0.7,
            "noise": 0.7,
            "fatformer": 0.7,
            "spatial": 0.7,
        },
    )


def _entropy(w: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(np.asarray(w, dtype=np.float64), eps, 1.0)
    p /= float(p.sum())
    return float(-np.sum(p * np.log(p)))


def test_oracle_risk_adaptive_smoothing_reduces_peak_weight_on_high_risk_sample():
    train_data = []
    for label in ["authentic", "manipulated", "ai_generated"]:
        train_data.append(
            _train_sample_custom(
                label,
                {
                    "frequency": label,
                    "noise": label,
                    "fatformer": "uncertain",
                    "spatial": "uncertain",
                },
            )
        )
        train_data.append(
            _train_sample_custom(
                label,
                {
                    "frequency": label,
                    "noise": label,
                    "fatformer": "manipulated" if label != "manipulated" else "authentic",
                    "spatial": "manipulated" if label != "manipulated" else "authentic",
                },
            )
        )
        train_data.append(
            _train_sample_custom(
                label,
                {
                    "frequency": label,
                    "noise": "uncertain",
                    "fatformer": "ai_generated" if label != "ai_generated" else "authentic",
                    "spatial": "uncertain",
                },
            )
        )

    sim = build_empirical_simulator(train_data, seed=42)

    sample_high_risk = SimulatedOutput(
        true_label="authentic",
        sub_type="authentic_stub",
        agent_verdicts={
            "frequency": "authentic",
            "noise": "manipulated",
            "fatformer": "ai_generated",
            "spatial": "uncertain",
        },
        agent_confidences={
            "frequency": 0.95,
            "noise": 0.35,
            "fatformer": 0.80,
            "spatial": 0.40,
        },
    )

    base = OracleWeightComputer(
        simulator=sim,
        config=OracleWeightConfig(
            power=1.5,
            label_smoothing=0.0,
            adaptive_smoothing_scale=0.0,
        ),
    )
    risk = OracleWeightComputer(
        simulator=sim,
        config=OracleWeightConfig(
            power=1.5,
            label_smoothing=0.0,
            margin_power=0.0,
            wrong_peak_power=0.0,
            adaptive_smoothing_scale=0.30,
            adaptive_smoothing_max=0.35,
        ),
    )

    w_base = base.compute_single(sample_high_risk)
    w_risk = risk.compute_single(sample_high_risk)

    assert np.max(w_risk) < np.max(w_base)
    assert _entropy(w_risk) > _entropy(w_base)


def test_oracle_risk_adaptive_smoothing_keeps_low_risk_sample_near_base():
    train_data = []
    for label in ["authentic", "manipulated", "ai_generated"]:
        train_data.append(_train_sample(label, label))
        train_data.append(_train_sample(label, label))
        train_data.append(_train_sample(label, "uncertain"))

    sim = build_empirical_simulator(train_data, seed=7)

    sample_low_risk = SimulatedOutput(
        true_label="manipulated",
        sub_type="manipulated_stub",
        agent_verdicts={
            "frequency": "manipulated",
            "noise": "manipulated",
            "fatformer": "manipulated",
            "spatial": "manipulated",
        },
        agent_confidences={
            "frequency": 0.8,
            "noise": 0.8,
            "fatformer": 0.8,
            "spatial": 0.8,
        },
    )

    base = OracleWeightComputer(
        simulator=sim,
        config=OracleWeightConfig(
            power=1.5,
            label_smoothing=0.0,
            adaptive_smoothing_scale=0.0,
        ),
    )
    risk = OracleWeightComputer(
        simulator=sim,
        config=OracleWeightConfig(
            power=1.5,
            label_smoothing=0.0,
            margin_power=1.0,
            wrong_peak_power=1.0,
            adaptive_smoothing_scale=0.30,
            adaptive_smoothing_max=0.35,
        ),
    )

    w_base = base.compute_single(sample_low_risk)
    w_risk = risk.compute_single(sample_low_risk)

    assert np.allclose(w_risk, w_base, atol=1e-8)


def test_oracle_loss_averse_mode_penalizes_majority_disagreement_agent():
    train_data = []
    for label in ["authentic", "manipulated", "ai_generated"]:
        train_data.append(_train_sample(label, label))
        train_data.append(_train_sample(label, label))
        train_data.append(_train_sample(label, label))

    sim = build_empirical_simulator(train_data, seed=17)
    sample = SimulatedOutput(
        true_label="authentic",
        sub_type="authentic_stub",
        agent_verdicts={
            "frequency": "authentic",
            "noise": "authentic",
            "fatformer": "manipulated",  # majority와 불일치
            "spatial": "authentic",
        },
        agent_confidences={
            "frequency": 0.8,
            "noise": 0.8,
            "fatformer": 0.8,
            "spatial": 0.8,
        },
    )

    base = OracleWeightComputer(
        simulator=sim,
        config=OracleWeightConfig(
            power=1.5,
            target_mode="accuracy",
        ),
    )
    loss_averse = OracleWeightComputer(
        simulator=sim,
        config=OracleWeightConfig(
            power=1.5,
            target_mode="loss_averse",
            majority_agreement_power=2.0,
            disagreement_penalty=0.2,
        ),
    )

    w_base = base.compute_single(sample)
    w_loss = loss_averse.compute_single(sample)
    idx_fatformer = 2  # AGENT_NAMES = [frequency, noise, fatformer, spatial]

    assert w_loss[idx_fatformer] < w_base[idx_fatformer]
    assert np.isfinite(w_loss).all()


def test_oracle_target_mode_invalid_raises_value_error():
    train_data = [_train_sample("authentic", "authentic"), _train_sample("manipulated", "manipulated")]
    sim = build_empirical_simulator(train_data, seed=123)
    sample = SimulatedOutput(
        true_label="authentic",
        sub_type="authentic_stub",
        agent_verdicts={
            "frequency": "authentic",
            "noise": "authentic",
            "fatformer": "authentic",
            "spatial": "authentic",
        },
        agent_confidences={
            "frequency": 0.8,
            "noise": 0.8,
            "fatformer": 0.8,
            "spatial": 0.8,
        },
    )

    oracle = OracleWeightComputer(simulator=sim, config=OracleWeightConfig(target_mode="unsupported_mode"))
    try:
        oracle.compute_single(sample)
    except ValueError as e:
        assert "target_mode" in str(e)
    else:
        raise AssertionError("expected ValueError for unsupported target_mode")
