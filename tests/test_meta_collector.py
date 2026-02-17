"""
Path A collector 유틸리티 단위 테스트.
"""

import numpy as np

from src.meta.collector import (
    AgentOutputCollector,
    CollectedRecord,
    build_empirical_simulator,
    build_proxy_image_features,
)
from src.meta.simulator import AGENT_NAMES, SimulatedOutput


def _make_sample(label: str, offset: int = 0) -> SimulatedOutput:
    # 라벨별 verdict 패턴을 조금씩 다르게 구성
    if label == "authentic":
        verdicts = {
            "frequency": "authentic",
            "noise": "authentic",
            "fatformer": "authentic",
            "spatial": "manipulated" if offset % 2 else "authentic",
        }
    elif label == "manipulated":
        verdicts = {
            "frequency": "manipulated",
            "noise": "manipulated",
            "fatformer": "authentic",
            "spatial": "manipulated",
        }
    else:
        verdicts = {
            "frequency": "manipulated",
            "noise": "authentic",
            "fatformer": "ai_generated",
            "spatial": "manipulated",
        }

    confidences = {
        "frequency": 0.7,
        "noise": 0.65,
        "fatformer": 0.9 if label == "ai_generated" else 0.4,
        "spatial": 0.6,
    }
    return SimulatedOutput(
        true_label=label,
        sub_type=f"{label}_stub",
        agent_verdicts=verdicts,
        agent_confidences=confidences,
    )


def test_proxy_feature_shape_and_range():
    samples = [
        _make_sample("authentic"),
        _make_sample("manipulated"),
        _make_sample("ai_generated"),
    ]
    X = build_proxy_image_features(samples)
    # 4 agents * (4 verdict one-hot + 1 confidence) = 20
    assert X.shape == (3, 20)
    assert np.isfinite(X).all()
    assert (X[:, -1] >= 0.0).all() and (X[:, -1] <= 1.0).all()


def test_proxy_feature_risk52_shape_and_finite():
    samples = [
        _make_sample("authentic"),
        _make_sample("manipulated"),
        _make_sample("ai_generated"),
    ]
    X = build_proxy_image_features(samples, feature_set="risk52")
    assert X.shape == (3, 52)
    assert np.isfinite(X).all()
    # risk52 마지막 16개는 리스크 통계: 불확실성 관련 값이 모두 0 이상이어야 한다.
    assert (X[:, -16:] >= 0.0).all()


def test_stratified_split_preserves_label_counts():
    samples = []
    for label in ["authentic", "manipulated", "ai_generated"]:
        for i in range(10):
            samples.append(_make_sample(label, offset=i))

    train, val, test = AgentOutputCollector.stratified_split(
        samples,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        seed=42,
    )
    assert len(train) + len(val) + len(test) == len(samples)

    def _counts(ds):
        out = {}
        for s in ds:
            out[s.true_label] = out.get(s.true_label, 0) + 1
        return out

    c_train = _counts(train)
    c_val = _counts(val)
    c_test = _counts(test)
    for label in ["authentic", "manipulated", "ai_generated"]:
        assert c_train.get(label, 0) > 0
        assert c_val.get(label, 0) > 0
        assert c_test.get(label, 0) > 0
        assert c_train.get(label, 0) + c_val.get(label, 0) + c_test.get(label, 0) == 10


def test_stratified_kfold_split_preserves_label_counts():
    samples = []
    for label in ["authentic", "manipulated", "ai_generated"]:
        for i in range(10):
            samples.append(_make_sample(label, offset=i))

    train, val, test = AgentOutputCollector.stratified_kfold_split(
        samples,
        k_folds=5,
        test_fold=0,
        val_fold=1,
        seed=42,
    )
    assert len(train) + len(val) + len(test) == len(samples)

    def _counts(ds):
        out = {}
        for s in ds:
            out[s.true_label] = out.get(s.true_label, 0) + 1
        return out

    c_train = _counts(train)
    c_val = _counts(val)
    c_test = _counts(test)
    for label in ["authentic", "manipulated", "ai_generated"]:
        assert c_train.get(label, 0) == 6
        assert c_val.get(label, 0) == 2
        assert c_test.get(label, 0) == 2
        assert c_train.get(label, 0) + c_val.get(label, 0) + c_test.get(label, 0) == 10


def test_empirical_simulator_confusion_rows_normalized():
    samples = []
    for label in ["authentic", "manipulated", "ai_generated"]:
        for i in range(12):
            samples.append(_make_sample(label, offset=i))

    sim = build_empirical_simulator(samples, seed=0)
    for agent in AGENT_NAMES:
        prof = sim.profiles[agent]
        for true_label in ["authentic", "manipulated", "ai_generated"]:
            row = prof.confusion_matrices[true_label]
            assert row.shape == (4,)
            assert np.isfinite(row).all()
            assert np.all(row >= 0.0)
            assert np.isclose(float(row.sum()), 1.0, atol=1e-6)


def test_save_and_load_jsonl_roundtrip(tmp_path):
    samples = [
        _make_sample("authentic", offset=0),
        _make_sample("manipulated", offset=1),
        _make_sample("ai_generated", offset=2),
    ]
    records = []
    for i, s in enumerate(samples):
        records.append(
            CollectedRecord(
                image_path=f"/tmp/fake_{i}.jpg",
                true_label=s.true_label,
                sub_type=s.sub_type,
                agent_verdicts=dict(s.agent_verdicts),
                agent_confidences=dict(s.agent_confidences),
                evidence_digest={a: {"backend": "stub"} for a in AGENT_NAMES},
            )
        )

    out_path = tmp_path / "collector_roundtrip.jsonl"
    AgentOutputCollector.save_jsonl(records, out_path)
    loaded_samples, loaded_records = AgentOutputCollector.load_jsonl(out_path)

    assert len(loaded_samples) == 3
    assert len(loaded_records) == 3
    assert loaded_samples[0].true_label == samples[0].true_label
    assert loaded_samples[1].agent_verdicts == samples[1].agent_verdicts
    assert loaded_samples[2].agent_confidences == samples[2].agent_confidences
    assert loaded_records[0].image_path == "/tmp/fake_0.jpg"
