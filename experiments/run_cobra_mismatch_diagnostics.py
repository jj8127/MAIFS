#!/usr/bin/env python3
"""
COBRA 성능 저하 원인 진단:
1) trust profile 영향 (static / metrics-derived-train / oracle-test-upper / shuffled)
2) 알고리즘 선택 영향 (rot / drwa / avga / auto)
3) 하이퍼파라미터 민감도 (trust_threshold / epsilon / temperature)

목적:
- "방법 한계" vs "설정 불일치"를 분리해서 확인할 수 있는 증거 생성.

기본 입력은 paper_final 설정(precollected jsonl + split seeds)을 재사용한다.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import yaml
from scipy.stats import wilcoxon
from sklearn.metrics import balanced_accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.trust import derive_trust_from_metrics, resolve_trust
from src.agents.base_agent import AgentResponse, AgentRole
from src.consensus.cobra import COBRAConsensus
from src.meta.collector import AgentOutputCollector
from src.meta.simulator import AGENT_NAMES, SimulatedOutput, TRUE_LABELS
from src.tools.base_tool import Verdict


ROLE_MAP = {
    "frequency": AgentRole.FREQUENCY,
    "noise": AgentRole.NOISE,
    "fatformer": AgentRole.FATFORMER,
    "spatial": AgentRole.SPATIAL,
}


def parse_int_list(raw: str, fallback: Sequence[int]) -> List[int]:
    text = str(raw).strip()
    if not text:
        return [int(x) for x in fallback]
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_float_list(raw: str, fallback: Sequence[float]) -> List[float]:
    text = str(raw).strip()
    if not text:
        return [float(x) for x in fallback]
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_str_list(raw: str, fallback: Sequence[str]) -> List[str]:
    text = str(raw).strip()
    if not text:
        return [str(x) for x in fallback]
    return [str(x).strip() for x in text.split(",") if str(x).strip()]


def verdict_to_label_index(verdict: str) -> int:
    v = str(verdict).lower()
    if v in TRUE_LABELS:
        return TRUE_LABELS.index(v)
    return 0


def y_true_from_samples(samples: Sequence[SimulatedOutput]) -> np.ndarray:
    return np.array([verdict_to_label_index(s.true_label) for s in samples], dtype=np.int64)


def compute_agent_metrics(samples: Sequence[SimulatedOutput]) -> Dict[str, Dict[str, float]]:
    y_true = y_true_from_samples(samples)
    out: Dict[str, Dict[str, float]] = {}
    for agent in AGENT_NAMES:
        y_pred = np.array(
            [verdict_to_label_index(s.agent_verdicts.get(agent, "uncertain")) for s in samples],
            dtype=np.int64,
        )
        out[agent] = {
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        }
    return out


def to_agent_response(agent: str, sample: SimulatedOutput) -> AgentResponse:
    verdict_str = str(sample.agent_verdicts.get(agent, "uncertain")).lower()
    confidence = float(sample.agent_confidences.get(agent, 0.5))
    try:
        verdict = Verdict(verdict_str)
    except ValueError:
        verdict = Verdict.UNCERTAIN

    ai_score = confidence if verdict == Verdict.AI_GENERATED else (1.0 - confidence)
    evidence = {
        "ai_generation_score": max(0.0, min(1.0, ai_score)),
        "source": "cobra_mismatch_diagnostics",
    }

    return AgentResponse(
        agent_name=agent,
        role=ROLE_MAP[agent],
        verdict=verdict,
        confidence=max(0.0, min(1.0, confidence)),
        reasoning="",
        evidence=evidence,
        arguments=[],
    )


def predict_cobra(
    samples: Sequence[SimulatedOutput],
    trust_scores: Dict[str, float],
    algorithm: str,
    trust_threshold: float = 0.7,
    epsilon: float = 0.2,
    temperature: float = 1.0,
) -> np.ndarray:
    engine = COBRAConsensus(
        default_algorithm="drwa",
        trust_threshold=float(trust_threshold),
        epsilon=float(epsilon),
        temperature=float(temperature),
    )
    preds: List[int] = []
    selected_algorithm: Optional[str] = None if algorithm == "auto" else algorithm
    for s in samples:
        responses = {agent: to_agent_response(agent, s) for agent in AGENT_NAMES}
        result = engine.aggregate(responses=responses, trust_scores=trust_scores, algorithm=selected_algorithm)
        preds.append(verdict_to_label_index(result.final_verdict.value))
    return np.array(preds, dtype=np.int64)


def summarize_runs(runs: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(runs, dtype=float)
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    return {"mean": float(np.mean(arr)), "std": std}


def paired_wilcoxon(a: Sequence[float], b: Sequence[float]) -> float:
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    if a_arr.size != b_arr.size or a_arr.size == 0:
        return float("nan")
    if np.allclose(a_arr, b_arr):
        return 1.0
    try:
        return float(wilcoxon(a_arr, b_arr, zero_method="wilcox", alternative="two-sided").pvalue)
    except Exception:
        return float("nan")


def sign_counts(a: Sequence[float], b: Sequence[float]) -> Tuple[int, int, int]:
    pos = 0
    neg = 0
    zero = 0
    for x, y in zip(a, b):
        d = float(x) - float(y)
        if d > 0:
            pos += 1
        elif d < 0:
            neg += 1
        else:
            zero += 1
    return pos, neg, zero


@dataclass
class EvalRow:
    name: str
    macro_f1_runs: List[float]
    bal_acc_runs: List[float]
    delta_vs_baseline_runs: List[float]
    wilcoxon_p_vs_baseline: float
    pos_vs_baseline: int
    neg_vs_baseline: int
    zero_vs_baseline: int

    def to_dict(self) -> Dict[str, object]:
        macro = summarize_runs(self.macro_f1_runs)
        bal = summarize_runs(self.bal_acc_runs)
        delta = summarize_runs(self.delta_vs_baseline_runs)
        return {
            "name": self.name,
            "macro_f1_mean": macro["mean"],
            "macro_f1_std": macro["std"],
            "balanced_accuracy_mean": bal["mean"],
            "balanced_accuracy_std": bal["std"],
            "delta_vs_baseline_mean": delta["mean"],
            "delta_vs_baseline_std": delta["std"],
            "wilcoxon_p_vs_baseline": self.wilcoxon_p_vs_baseline,
            "positive_vs_baseline": self.pos_vs_baseline,
            "negative_vs_baseline": self.neg_vs_baseline,
            "zero_vs_baseline": self.zero_vs_baseline,
            "macro_f1_runs": self.macro_f1_runs,
            "balanced_accuracy_runs": self.bal_acc_runs,
            "delta_vs_baseline_runs": self.delta_vs_baseline_runs,
        }


def build_split(
    samples: Sequence[SimulatedOutput],
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> Tuple[List[SimulatedOutput], List[SimulatedOutput], List[SimulatedOutput]]:
    collector = AgentOutputCollector.__new__(AgentOutputCollector)
    tr_idx, va_idx, te_idx = collector.stratified_split(
        samples,
        train_ratio=float(train_ratio),
        val_ratio=float(val_ratio),
        test_ratio=float(test_ratio),
        seed=int(seed),
        return_indices=True,
    )
    return (
        [samples[i] for i in tr_idx],
        [samples[i] for i in va_idx],
        [samples[i] for i in te_idx],
    )


def eval_setting_over_seeds(
    *,
    samples: Sequence[SimulatedOutput],
    seeds: Sequence[int],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    algorithm: str,
    trust_mode: str,
    static_trust: Dict[str, float],
    n_shuffles: int,
    trust_threshold: float = 0.7,
    epsilon: float = 0.2,
    temperature: float = 1.0,
) -> Tuple[List[float], List[float]]:
    macro_runs: List[float] = []
    bal_runs: List[float] = []

    for seed in seeds:
        train_data, _val_data, test_data = build_split(
            samples=samples,
            seed=int(seed),
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )
        y_test = y_true_from_samples(test_data)

        if trust_mode == "static":
            trust = dict(static_trust)
            y_pred = predict_cobra(
                test_data,
                trust_scores=trust,
                algorithm=algorithm,
                trust_threshold=trust_threshold,
                epsilon=epsilon,
                temperature=temperature,
            )
            macro_runs.append(float(f1_score(y_test, y_pred, average="macro", zero_division=0)))
            bal_runs.append(float(balanced_accuracy_score(y_test, y_pred)))
            continue

        if trust_mode == "metrics_derived_train":
            metrics = compute_agent_metrics(train_data)
            trust = derive_trust_from_metrics(metrics)
            y_pred = predict_cobra(
                test_data,
                trust_scores=trust,
                algorithm=algorithm,
                trust_threshold=trust_threshold,
                epsilon=epsilon,
                temperature=temperature,
            )
            macro_runs.append(float(f1_score(y_test, y_pred, average="macro", zero_division=0)))
            bal_runs.append(float(balanced_accuracy_score(y_test, y_pred)))
            continue

        if trust_mode == "oracle_test_upper":
            # 진단용 상한선: 테스트셋 메트릭으로 trust를 추정 (leakage 허용)
            metrics = compute_agent_metrics(test_data)
            trust = derive_trust_from_metrics(metrics)
            y_pred = predict_cobra(
                test_data,
                trust_scores=trust,
                algorithm=algorithm,
                trust_threshold=trust_threshold,
                epsilon=epsilon,
                temperature=temperature,
            )
            macro_runs.append(float(f1_score(y_test, y_pred, average="macro", zero_division=0)))
            bal_runs.append(float(balanced_accuracy_score(y_test, y_pred)))
            continue

        if trust_mode == "shuffled_static":
            rng = np.random.default_rng(int(seed) + 20260305)
            vals = np.array([static_trust[a] for a in AGENT_NAMES], dtype=float)
            macro_shuffles: List[float] = []
            bal_shuffles: List[float] = []
            for _ in range(max(1, int(n_shuffles))):
                p = rng.permutation(len(AGENT_NAMES))
                shuffled = {AGENT_NAMES[i]: float(vals[p[i]]) for i in range(len(AGENT_NAMES))}
                y_pred = predict_cobra(
                    test_data,
                    trust_scores=shuffled,
                    algorithm=algorithm,
                    trust_threshold=trust_threshold,
                    epsilon=epsilon,
                    temperature=temperature,
                )
                macro_shuffles.append(float(f1_score(y_test, y_pred, average="macro", zero_division=0)))
                bal_shuffles.append(float(balanced_accuracy_score(y_test, y_pred)))
            macro_runs.append(float(np.mean(macro_shuffles)))
            bal_runs.append(float(np.mean(bal_shuffles)))
            continue

        raise ValueError(f"unsupported trust_mode: {trust_mode}")

    return macro_runs, bal_runs


def make_row(name: str, macro_runs: List[float], bal_runs: List[float], baseline_runs: List[float]) -> EvalRow:
    deltas = [float(a - b) for a, b in zip(macro_runs, baseline_runs)]
    pval = paired_wilcoxon(macro_runs, baseline_runs)
    pos, neg, zero = sign_counts(macro_runs, baseline_runs)
    return EvalRow(
        name=name,
        macro_f1_runs=macro_runs,
        bal_acc_runs=bal_runs,
        delta_vs_baseline_runs=deltas,
        wilcoxon_p_vs_baseline=pval,
        pos_vs_baseline=pos,
        neg_vs_baseline=neg,
        zero_vs_baseline=zero,
    )


def fmt(v: float) -> str:
    return f"{v:.4f}"


def fmt_p(v: float) -> str:
    if np.isnan(v):
        return "nan"
    if v < 1e-4:
        return f"{v:.2e}"
    return f"{v:.4f}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run COBRA mismatch diagnostics")
    p.add_argument(
        "config",
        nargs="?",
        default="experiments/configs/paper_final.yaml",
        help="YAML config path (default: paper_final.yaml)",
    )
    p.add_argument("--jsonl", type=str, default="", help="Optional precollected jsonl override")
    p.add_argument("--seeds", type=str, default="", help="Comma-separated split seeds override")
    p.add_argument("--algorithms", type=str, default="rot,drwa,avga,auto")
    p.add_argument("--trust-modes", type=str, default="static,metrics_derived_train,oracle_test_upper,shuffled_static")
    p.add_argument("--n-shuffles", type=int, default=20, help="Per-seed shuffle count for shuffled_static")
    p.add_argument("--rot-thresholds", type=str, default="0.5,0.6,0.7,0.8,0.9")
    p.add_argument("--drwa-epsilons", type=str, default="0.0,0.1,0.2,0.4,0.8")
    p.add_argument("--avga-temperatures", type=str, default="0.5,1.0,1.5,2.0,3.0")
    p.add_argument(
        "--output-json",
        type=str,
        default="",
        help="Optional output JSON path (default: experiments/results/cobra_diagnostics/cobra_mismatch_diagnostics_{ts}.json)",
    )
    p.add_argument(
        "--output-md",
        type=str,
        default="",
        help="Optional output MD path (default: experiments/results/cobra_diagnostics/cobra_mismatch_diagnostics_{ts}.md)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    jsonl_path = str(args.jsonl).strip() or str(cfg.get("precollected_jsonl", ""))
    if not jsonl_path:
        raise ValueError("jsonl path is empty. set in config or pass --jsonl")

    seeds = parse_int_list(str(args.seeds), cfg.get("seeds", list(range(300, 310))))
    split_cfg = cfg.get("split", {})
    train_ratio = float(split_cfg.get("train_ratio", 0.6))
    val_ratio = float(split_cfg.get("val_ratio", 0.2))
    test_ratio = float(split_cfg.get("test_ratio", 0.2))

    algorithms = parse_str_list(args.algorithms, ["rot", "drwa", "avga", "auto"])
    trust_modes = parse_str_list(args.trust_modes, ["static", "metrics_derived_train", "oracle_test_upper", "shuffled_static"])
    rot_thresholds = parse_float_list(args.rot_thresholds, [0.5, 0.6, 0.7, 0.8, 0.9])
    drwa_eps = parse_float_list(args.drwa_epsilons, [0.0, 0.1, 0.2, 0.4, 0.8])
    avga_temps = parse_float_list(args.avga_temperatures, [0.5, 1.0, 1.5, 2.0, 3.0])

    print(f"[load] jsonl={jsonl_path}")
    samples, _ = AgentOutputCollector.load_jsonl(Path(jsonl_path))
    print(f"[load] n_samples={len(samples)} seeds={seeds}")

    static_trust = resolve_trust(cfg.get("trust_scores"))
    print(f"[trust] static={static_trust}")

    # Baseline: DRWA + static trust + default params
    baseline_macro_runs, baseline_bal_runs = eval_setting_over_seeds(
        samples=samples,
        seeds=seeds,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        algorithm="drwa",
        trust_mode="static",
        static_trust=static_trust,
        n_shuffles=max(1, int(args.n_shuffles)),
        trust_threshold=0.7,
        epsilon=0.2,
        temperature=1.0,
    )
    baseline_row = make_row(
        name="baseline_drwa_static_default",
        macro_runs=baseline_macro_runs,
        bal_runs=baseline_bal_runs,
        baseline_runs=baseline_macro_runs,
    )

    print(f"[baseline] macro_f1={summarize_runs(baseline_macro_runs)}")

    # A) trust mode x algorithm grid
    grid_rows: List[EvalRow] = []
    for trust_mode in trust_modes:
        for algorithm in algorithms:
            name = f"{algorithm}__{trust_mode}"
            macro_runs, bal_runs = eval_setting_over_seeds(
                samples=samples,
                seeds=seeds,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                algorithm=algorithm,
                trust_mode=trust_mode,
                static_trust=static_trust,
                n_shuffles=max(1, int(args.n_shuffles)),
                trust_threshold=0.7,
                epsilon=0.2,
                temperature=1.0,
            )
            row = make_row(
                name=name,
                macro_runs=macro_runs,
                bal_runs=bal_runs,
                baseline_runs=baseline_macro_runs,
            )
            grid_rows.append(row)
            stats = row.to_dict()
            print(
                f"[grid] {name} f1={stats['macro_f1_mean']:.4f}±{stats['macro_f1_std']:.4f} "
                f"delta={stats['delta_vs_baseline_mean']:+.4f} p={fmt_p(float(stats['wilcoxon_p_vs_baseline']))}"
            )

    # B) parameter sensitivity (static trust only)
    sens_rows: Dict[str, List[EvalRow]] = {
        "rot_trust_threshold": [],
        "drwa_epsilon": [],
        "avga_temperature": [],
    }

    for t in rot_thresholds:
        name = f"rot_threshold={t:.3f}"
        macro_runs, bal_runs = eval_setting_over_seeds(
            samples=samples,
            seeds=seeds,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            algorithm="rot",
            trust_mode="static",
            static_trust=static_trust,
            n_shuffles=max(1, int(args.n_shuffles)),
            trust_threshold=float(t),
            epsilon=0.2,
            temperature=1.0,
        )
        sens_rows["rot_trust_threshold"].append(
            make_row(name=name, macro_runs=macro_runs, bal_runs=bal_runs, baseline_runs=baseline_macro_runs)
        )

    for eps in drwa_eps:
        name = f"drwa_epsilon={eps:.3f}"
        macro_runs, bal_runs = eval_setting_over_seeds(
            samples=samples,
            seeds=seeds,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            algorithm="drwa",
            trust_mode="static",
            static_trust=static_trust,
            n_shuffles=max(1, int(args.n_shuffles)),
            trust_threshold=0.7,
            epsilon=float(eps),
            temperature=1.0,
        )
        sens_rows["drwa_epsilon"].append(
            make_row(name=name, macro_runs=macro_runs, bal_runs=bal_runs, baseline_runs=baseline_macro_runs)
        )

    for temp in avga_temps:
        name = f"avga_temperature={temp:.3f}"
        macro_runs, bal_runs = eval_setting_over_seeds(
            samples=samples,
            seeds=seeds,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            algorithm="avga",
            trust_mode="static",
            static_trust=static_trust,
            n_shuffles=max(1, int(args.n_shuffles)),
            trust_threshold=0.7,
            epsilon=0.2,
            temperature=float(temp),
        )
        sens_rows["avga_temperature"].append(
            make_row(name=name, macro_runs=macro_runs, bal_runs=bal_runs, baseline_runs=baseline_macro_runs)
        )

    # Serialize
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_dir = Path("experiments/results/cobra_diagnostics")
    default_dir.mkdir(parents=True, exist_ok=True)

    out_json = Path(args.output_json) if str(args.output_json).strip() else default_dir / f"cobra_mismatch_diagnostics_{ts}.json"
    out_md = Path(args.output_md) if str(args.output_md).strip() else default_dir / f"cobra_mismatch_diagnostics_{ts}.md"

    payload = {
        "timestamp": datetime.now().isoformat(),
        "config_path": args.config,
        "jsonl_path": jsonl_path,
        "n_samples": int(len(samples)),
        "seeds": [int(s) for s in seeds],
        "split": {
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
        },
        "static_trust": static_trust,
        "baseline": baseline_row.to_dict(),
        "grid": [r.to_dict() for r in grid_rows],
        "sensitivity": {k: [r.to_dict() for r in rows] for k, rows in sens_rows.items()},
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Markdown
    baseline_stats = baseline_row.to_dict()
    lines: List[str] = []
    lines.append("# COBRA Mismatch Diagnostics")
    lines.append("")
    lines.append(f"- jsonl: `{jsonl_path}`")
    lines.append(f"- seeds: `{seeds}`")
    lines.append(f"- baseline: `DRWA + static trust + default params` = **{fmt(float(baseline_stats['macro_f1_mean']))} ± {fmt(float(baseline_stats['macro_f1_std']))}**")
    lines.append("")
    lines.append("## A. Trust/Algorithm Grid")
    lines.append("")
    lines.append("| Setting | Macro-F1 | Delta vs baseline | Wilcoxon p | sign(+/-/0) |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in sorted(grid_rows, key=lambda x: x.name):
        d = row.to_dict()
        lines.append(
            f"| {row.name} | {fmt(float(d['macro_f1_mean']))} ± {fmt(float(d['macro_f1_std']))} | "
            f"{fmt(float(d['delta_vs_baseline_mean']))} ± {fmt(float(d['delta_vs_baseline_std']))} | "
            f"{fmt_p(float(d['wilcoxon_p_vs_baseline']))} | "
            f"{int(d['positive_vs_baseline'])}/{int(d['negative_vs_baseline'])}/{int(d['zero_vs_baseline'])} |"
        )

    lines.append("")
    lines.append("## B. Sensitivity (static trust)")
    lines.append("")
    for key, title in [
        ("rot_trust_threshold", "RoT trust_threshold"),
        ("drwa_epsilon", "DRWA epsilon"),
        ("avga_temperature", "AVGA temperature"),
    ]:
        lines.append(f"### {title}")
        lines.append("")
        lines.append("| Param | Macro-F1 | Delta vs baseline | Wilcoxon p | sign(+/-/0) |")
        lines.append("|---|---:|---:|---:|---:|")
        for row in sens_rows[key]:
            d = row.to_dict()
            lines.append(
                f"| {row.name} | {fmt(float(d['macro_f1_mean']))} ± {fmt(float(d['macro_f1_std']))} | "
                f"{fmt(float(d['delta_vs_baseline_mean']))} ± {fmt(float(d['delta_vs_baseline_std']))} | "
                f"{fmt_p(float(d['wilcoxon_p_vs_baseline']))} | "
                f"{int(d['positive_vs_baseline'])}/{int(d['negative_vs_baseline'])}/{int(d['zero_vs_baseline'])} |"
            )
        lines.append("")

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[saved] json: {out_json}")
    print(f"[saved] md:   {out_md}")


if __name__ == "__main__":
    main()
