"""
개별 에이전트 성능 평가 (vs DAAC-GBM 비교)

각 에이전트(frequency/noise/fatformer/spatial)의 단독 판정 성능을
DAAC 메타 실험과 동일한 10 seed × 60/20/20 split 조건에서 평가.

Output:
  Table 1 — 개별 에이전트 vs 앙상블 Macro-F1 (mean ± std, Wilcoxon p)
  Table 2 — 개별 에이전트 Per-class F1
  JSON    — experiments/results/agent_eval/agent_eval_{ts}.json

Usage:
  .venv-qwen/bin/python experiments/run_agent_eval.py experiments/configs/paper_final.yaml
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import yaml
from scipy.stats import wilcoxon
from sklearn.metrics import f1_score

from src.meta.baselines import COBRABaseline
from src.meta.collector import AgentOutputCollector
from src.meta.features import MetaFeatureExtractor
from src.meta.simulator import SimulatedOutput
from src.meta.trainer import MetaTrainer

CLASSES = ["authentic", "manipulated", "ai_generated"]
AGENT_NAMES = ["frequency", "noise", "fatformer", "spatial"]
LABEL_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


def _split(
    samples: List[SimulatedOutput],
    seed: int,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
) -> Tuple[List, List, List]:
    collector = AgentOutputCollector.__new__(AgentOutputCollector)
    train_idx, val_idx, test_idx = collector.stratified_split(
        samples,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        return_indices=True,
    )
    return (
        [samples[i] for i in train_idx],
        [samples[i] for i in val_idx],
        [samples[i] for i in test_idx],
    )


def _agent_predict(samples: List[SimulatedOutput], agent: str) -> np.ndarray:
    """단일 에이전트의 verdict를 class index로 변환."""
    preds = []
    for s in samples:
        verdict = s.agent_verdicts.get(agent, "authentic")
        preds.append(LABEL_TO_IDX.get(verdict, 0))
    return np.array(preds)


def _perclass(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    pf = f1_score(y_true, y_pred, average=None, zero_division=0, labels=[0, 1, 2])
    return {cls: float(pf[i]) for i, cls in enumerate(CLASSES)}


def _run_seed(
    samples: List[SimulatedOutput],
    seed: int,
    trust_scores: Dict[str, float],
    split_cfg: Dict,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    train_s, val_s, test_s = _split(
        samples, seed,
        train_ratio=split_cfg.get("train_ratio", 0.6),
        val_ratio=split_cfg.get("val_ratio", 0.2),
        test_ratio=split_cfg.get("test_ratio", 0.2),
    )

    extractor = MetaFeatureExtractor()
    X_train, y_train = extractor.extract_dataset(train_s)
    X_test, y_test = extractor.extract_dataset(test_s)

    # DAAC-GBM (비교 기준)
    trainer = MetaTrainer()
    X_val, y_val = extractor.extract_dataset(val_s)
    trainer.train_all(X_train, y_train, X_val, y_val)
    daac_pred = trainer.predict("gradient_boosting", X_test)
    daac_f1 = float(f1_score(y_test, daac_pred, average="macro", zero_division=0))

    # COBRA 베이스라인
    cobra = COBRABaseline(trust_scores=trust_scores)
    cobra_pred = cobra.predict(test_s)
    cobra_f1 = float(f1_score(y_test, cobra_pred, average="macro", zero_division=0))

    results: Dict[str, float] = {
        "cobra": cobra_f1,
        "daac_gradient_boosting": daac_f1,
    }
    perclass: Dict[str, Dict[str, float]] = {
        "cobra": _perclass(y_test, cobra_pred),
        "daac_gradient_boosting": _perclass(y_test, daac_pred),
    }

    # 개별 에이전트 평가
    for agent in AGENT_NAMES:
        pred = _agent_predict(test_s, agent)
        f1 = float(f1_score(y_test, pred, average="macro", zero_division=0))
        results[f"agent_{agent}"] = f1
        perclass[f"agent_{agent}"] = _perclass(y_test, pred)

    return results, {"perclass": perclass}


def _aggregate(all_results: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    keys = list(all_results[0].keys())
    agg = {}
    for k in keys:
        vals = [r[k] for r in all_results]
        agg[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return agg


def _wilcoxon(all_results: List[Dict[str, float]], ref: str = "daac_gradient_boosting") -> Dict[str, float]:
    ref_runs = [r[ref] for r in all_results]
    pvalues: Dict[str, float] = {}
    for key in all_results[0]:
        if key == ref:
            pvalues[key] = float("nan")
            continue
        vals = [r[key] for r in all_results]
        try:
            _, p = wilcoxon(vals, ref_runs, alternative="two-sided")
            pvalues[key] = float(p)
        except Exception:
            pvalues[key] = float("nan")
    return pvalues


def _print_table1(agg: Dict, pvalues: Dict) -> str:
    ref = "daac_gradient_boosting"
    order = ["agent_frequency", "agent_noise", "agent_fatformer", "agent_spatial",
             "cobra", "daac_gradient_boosting"]
    labels = {
        "agent_frequency": "Frequency (단독)",
        "agent_noise": "Noise (단독)",
        "agent_fatformer": "FatFormer (단독)",
        "agent_spatial": "Spatial (단독)",
        "cobra": "COBRA (앙상블)",
        "daac_gradient_boosting": "DAAC-GBM (앙상블)",
    }
    lines = [
        "\n" + "=" * 70,
        "Table 1: 개별 에이전트 vs DAAC-GBM (Macro-F1, 10 seeds)",
        "=" * 70,
        f"{'Method':<28} {'F1 mean±std':>14}  {'p vs DAAC-GBM':>14}",
        "-" * 70,
    ]
    for k in order:
        if k not in agg:
            continue
        m, s = agg[k]["mean"], agg[k]["std"]
        p = pvalues.get(k, float("nan"))
        p_str = "—" if k == ref else (f"{p:.4f}" + ("**" if p < 0.01 else ("*" if p < 0.05 else "")))
        lines.append(f"{labels.get(k, k):<28} {m:.3f}±{s:.3f}        {p_str:>14}")
    lines.append("=" * 70)
    return "\n".join(lines)


def _print_table2(all_extras: List[Dict]) -> str:
    order = ["agent_frequency", "agent_noise", "agent_fatformer", "agent_spatial",
             "cobra", "daac_gradient_boosting"]
    labels = {
        "agent_frequency": "Frequency", "agent_noise": "Noise",
        "agent_fatformer": "FatFormer", "agent_spatial": "Spatial",
        "cobra": "COBRA", "daac_gradient_boosting": "DAAC-GBM",
    }
    # avg per-class over seeds
    avg: Dict[str, Dict[str, float]] = {}
    for k in order:
        vals_per_class: Dict[str, List[float]] = {c: [] for c in CLASSES}
        for ex in all_extras:
            pc = ex["perclass"].get(k, {})
            for c in CLASSES:
                vals_per_class[c].append(pc.get(c, 0.0))
        avg[k] = {c: float(np.mean(vals_per_class[c])) for c in CLASSES}

    lines = [
        "\n" + "=" * 70,
        "Table 2: Per-class F1 (10 seeds 평균)",
        "=" * 70,
        f"{'Method':<20} {'Authentic':>12} {'Manipulated':>13} {'AI-Gen':>10}",
        "-" * 70,
    ]
    for k in order:
        if k not in avg:
            continue
        a = avg[k].get("authentic", 0)
        m = avg[k].get("manipulated", 0)
        g = avg[k].get("ai_generated", 0)
        lines.append(f"{labels.get(k, k):<20} {a:>12.3f} {m:>13.3f} {g:>10.3f}")
    lines.append("=" * 70)
    return "\n".join(lines)


def main():
    if len(sys.argv) < 2:
        print("Usage: .venv-qwen/bin/python experiments/run_agent_eval.py <config.yaml>")
        sys.exit(1)

    config_path = sys.argv[1]
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    jsonl_path = Path(cfg["precollected_jsonl"])
    seeds: List[int] = cfg["seeds"]
    trust_scores: Dict[str, float] = cfg.get("trust_scores", {})
    split_cfg: Dict = cfg.get("split", {})

    print(f"Loading JSONL: {jsonl_path}")
    collector = AgentOutputCollector()
    samples, _records = AgentOutputCollector.load_jsonl(jsonl_path)
    print(f"  → {len(samples)} samples loaded")

    all_results: List[Dict[str, float]] = []
    all_extras: List[Dict[str, Any]] = []

    for i, seed in enumerate(seeds):
        print(f"  seed {seed} ({i+1}/{len(seeds)}) ...", end=" ", flush=True)
        t0 = __import__("time").time()
        res, extras = _run_seed(samples, seed, trust_scores, split_cfg)
        dt = __import__("time").time() - t0
        print(f"{dt:.1f}s | daac={res['daac_gradient_boosting']:.3f} cobra={res['cobra']:.3f} "
              f"fatformer={res['agent_fatformer']:.3f}")
        all_results.append(res)
        all_extras.append(extras)

    agg = _aggregate(all_results)
    pvalues = _wilcoxon(all_results, ref="daac_gradient_boosting")

    t1 = _print_table1(agg, pvalues)
    t2 = _print_table2(all_extras)
    print(t1)
    print(t2)

    # Save JSON
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = cfg.get("label", Path(config_path).stem)
    out_dir = Path("experiments/results/agent_eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"agent_eval_{label}_{ts}.json"

    output = {
        "timestamp": datetime.now().isoformat(),
        "label": label,
        "config_path": config_path,
        "jsonl_path": str(jsonl_path),
        "n_samples": len(samples),
        "seeds": seeds,
        "results_per_seed": all_results,
        "aggregated": agg,
        "pvalues_vs_daac_gbm": pvalues,
        "extras_per_seed": all_extras,
        "tables": {"table1_methods": t1, "table2_perclass": t2},
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
