"""
DAAC-GBM vs COBRA Inference Latency Benchmark

에이전트 출력이 주어진 상태에서 합의 계층의 샘플당 inference time을 측정.
- COBRA (drwa): trust 기반 규칙 집계
- DAAC-GBM: 43-dim 메타 특징 추출 + XGBoost predict

결과: 샘플당 평균/중앙값 latency (µs 단위)
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
import yaml

from src.meta.baselines import COBRABaseline
from src.meta.collector import AgentOutputCollector
from src.meta.features import MetaFeatureExtractor
from src.meta.simulator import SimulatedOutput
from src.meta.trainer import MetaTrainer

JSONL_PATH = "experiments/results/phase2_patha_scale500_gain_predictor/patha_agent_outputs_20260304_080157.jsonl"
CONFIG_PATH = "experiments/configs/paper_final.yaml"
N_WARMUP = 20       # warmup 제외
N_REPEAT = 10       # 측정 반복 횟수 (전체 test set × N_REPEAT)
SEED = 300


def load_samples(jsonl_path: str):
    samples = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            samples.append(SimulatedOutput(
                true_label=d["true_label"],
                agent_verdicts=d["agent_verdicts"],
                agent_confidences=d["agent_confidences"],
                sub_type=d.get("sub_type"),
            ))
    return samples


def split_data(samples, seed=SEED):
    collector = AgentOutputCollector.__new__(AgentOutputCollector)
    train_idx, val_idx, test_idx = collector.stratified_split(
        samples,
        train_ratio=0.6, val_ratio=0.2, test_ratio=0.2,
        seed=seed, return_indices=True,
    )
    return (
        [samples[i] for i in train_idx],
        [samples[i] for i in val_idx],
        [samples[i] for i in test_idx],
    )


def measure_cobra_batch(test_data, trust_scores, n_warmup=N_WARMUP, n_repeat=N_REPEAT):
    """COBRA drwa: 전체 배치 단위로 반복 측정 → 샘플당 latency 산출"""
    cobra = COBRABaseline(trust_scores, algorithm="drwa")
    n = len(test_data)

    for _ in range(n_warmup):
        cobra.predict(test_data)

    times_per_sample = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        cobra.predict(test_data)
        elapsed_us = (time.perf_counter() - t0) * 1e6
        times_per_sample.append(elapsed_us / n)

    return np.array(times_per_sample)


def measure_daac_gbm_batch(test_data, trainer, n_warmup=N_WARMUP, n_repeat=N_REPEAT):
    """DAAC-GBM: 배치 특징 추출 + XGBoost predict → 샘플당 latency 산출"""
    ext = MetaFeatureExtractor()
    n = len(test_data)
    X_te, _ = ext.extract_dataset(test_data)

    for _ in range(n_warmup):
        trainer.predict("gradient_boosting", X_te)

    times_feat = []
    times_pred = []
    times_total = []

    for _ in range(n_repeat):
        t0 = time.perf_counter()
        X, _ = ext.extract_dataset(test_data)
        t1 = time.perf_counter()
        trainer.predict("gradient_boosting", X)
        t2 = time.perf_counter()

        times_feat.append((t1 - t0) * 1e6 / n)
        times_pred.append((t2 - t1) * 1e6 / n)
        times_total.append((t2 - t0) * 1e6 / n)

    return np.array(times_feat), np.array(times_pred), np.array(times_total)


def report(label, arr):
    print(f"  {label}:")
    print(f"    mean   = {arr.mean():.2f} µs")
    print(f"    median = {np.median(arr):.2f} µs")
    print(f"    p95    = {np.percentile(arr, 95):.2f} µs")
    print(f"    std    = {arr.std():.2f} µs")
    return {"mean": float(arr.mean()), "median": float(np.median(arr)),
            "p95": float(np.percentile(arr, 95)), "std": float(arr.std())}


def main():
    print("=== DAAC Inference Latency Benchmark ===\n")

    # 데이터 로드
    samples = load_samples(JSONL_PATH)
    print(f"총 샘플: {len(samples)}")

    train_data, val_data, test_data = split_data(samples)
    print(f"test 샘플: {len(test_data)}\n")

    # trust scores 로드
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    trust_scores = cfg.get("trust_scores", {})

    # DAAC-GBM 학습 (train+val)
    print("DAAC-GBM 학습 중...")
    ext = MetaFeatureExtractor()
    X_tr, y_tr = ext.extract_dataset(train_data + val_data)
    trainer = MetaTrainer()
    trainer.train("gradient_boosting", X_tr, y_tr)
    print("학습 완료\n")

    # ── 측정 ──────────────────────────────────────────────────────
    print(f"측정: warmup={N_WARMUP}, repeat={N_REPEAT}× batch({len(test_data)} samples) → 샘플당 µs\n")

    print("[COBRA drwa]")
    cobra_times = measure_cobra_batch(test_data, trust_scores)
    cobra_stats = report("per-sample", cobra_times)

    print("\n[DAAC-GBM]")
    feat_times, pred_times, total_times = measure_daac_gbm_batch(test_data, trainer)
    feat_stats  = report("feat extraction (per-sample)", feat_times)
    pred_stats  = report("GBM predict (per-sample)", pred_times)
    total_stats = report("total feat+pred (per-sample)", total_times)

    # ── 요약 ──────────────────────────────────────────────────────
    print("\n" + "="*50)
    print("요약 (샘플당 평균 latency)")
    print("="*50)
    print(f"  COBRA drwa        : {cobra_stats['mean']:7.2f} µs  (median {cobra_stats['median']:.2f})")
    print(f"  DAAC-GBM total    : {total_stats['mean']:7.2f} µs  (median {total_stats['median']:.2f})")
    print(f"    └ feat extract  : {feat_stats['mean']:7.2f} µs")
    print(f"    └ GBM predict   : {pred_stats['mean']:7.2f} µs")
    ratio = total_stats['mean'] / cobra_stats['mean'] if cobra_stats['mean'] > 0 else float('inf')
    print(f"  DAAC/COBRA ratio  : {ratio:.2f}x")

    # JSON 저장
    result = {
        "n_test_samples": len(test_data),
        "n_warmup": N_WARMUP,
        "n_repeat": N_REPEAT,
        "cobra_drwa_us": cobra_stats,
        "daac_gbm_feat_us": feat_stats,
        "daac_gbm_pred_us": pred_stats,
        "daac_gbm_total_us": total_stats,
        "daac_over_cobra_ratio": ratio,
    }
    out_path = Path("experiments/results/speed_benchmark.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n결과 저장: {out_path}")


if __name__ == "__main__":
    main()
