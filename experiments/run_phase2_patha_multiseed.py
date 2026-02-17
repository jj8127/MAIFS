"""
Path A 멀티시드 실행기.

예시:
    .venv-qwen/bin/python experiments/run_phase2_patha_multiseed.py \
      experiments/configs/phase2_patha_scale120.yaml \
      --seeds 42,43,44,45,46
"""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from datetime import datetime
from math import ceil, comb, erfc, sqrt
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Make repository root importable when this file is executed directly.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.evaluate_phase2_gate import evaluate_gate
from experiments.run_phase2_patha import load_config, run_phase2_patha


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Phase2 Path A across multiple seeds")
    p.add_argument(
        "config",
        nargs="?",
        default="experiments/configs/phase2_patha.yaml",
        help="YAML config path",
    )
    p.add_argument(
        "--seeds",
        type=str,
        default="42,43,44,45,46",
        help="Comma-separated seed list",
    )
    p.add_argument(
        "--summary-out",
        type=str,
        default="",
        help="Optional explicit summary json path",
    )
    return p.parse_args()


def _parse_seed_list(raw: str) -> List[int]:
    seeds: List[int] = []
    for x in raw.split(","):
        t = x.strip()
        if not t:
            continue
        seeds.append(int(t))
    if not seeds:
        raise ValueError("empty seed list")
    return seeds


def _exact_sign_test_two_sided(pos_count: int, neg_count: int) -> float:
    """
    시드 단위 개선 방향(+, -)에 대한 exact sign test 양측 p-value.

    zero diff는 정보가 없어 제외한다.
    """
    n = int(pos_count + neg_count)
    if n <= 0:
        return 1.0
    k = int(min(pos_count, neg_count))
    cdf_tail = sum(comb(n, i) for i in range(k + 1)) / float(2**n)
    return float(min(1.0, 2.0 * cdf_tail))


def _load_json(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _normalize_gate_profile(cfg: Dict[str, Any]) -> Dict[str, Any]:
    raw_alpha = cfg.get("downside_cvar_alpha", 0.1)
    return {
        "min_runs": int(cfg.get("min_runs", 10)),
        "min_f1_diff_mean": float(cfg.get("min_f1_diff_mean", 0.01)),
        "min_significant_count": int(cfg.get("min_significant_count", 3)),
        "min_improvement_over_baseline": float(cfg.get("min_improvement_over_baseline", 0.0)),
        "min_positive_seed_count": int(cfg.get("min_positive_seed_count", 0)),
        "max_sign_test_pvalue": cfg.get("max_sign_test_pvalue", None),
        "require_pooled_mcnemar_significant": bool(cfg.get("require_pooled_mcnemar_significant", False)),
        "max_pooled_mcnemar_pvalue": cfg.get("max_pooled_mcnemar_pvalue", None),
        "max_negative_rate": cfg.get("max_negative_rate", None),
        "max_downside_mean": cfg.get("max_downside_mean", None),
        "max_cvar_downside": cfg.get("max_cvar_downside", None),
        "max_worst_case_loss": cfg.get("max_worst_case_loss", None),
        "downside_cvar_alpha": float(0.1 if raw_alpha is None else raw_alpha),
    }


def _profile_presets_from_protocol(protocol: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    presets = protocol.get("gate_profiles")
    if isinstance(presets, dict) and presets:
        out: Dict[str, Dict[str, Any]] = {}
        for name, cfg in presets.items():
            if not isinstance(cfg, dict):
                continue
            out[str(name)] = _normalize_gate_profile(cfg)
        if out:
            return out

    gate = protocol.get("gate")
    if isinstance(gate, dict) and gate:
        return {"strict": _normalize_gate_profile(gate)}

    return {}


def _evaluate_active_gate(summary: Dict[str, Any], summary_path: Path, config: Dict[str, Any]) -> Optional[Path]:
    protocol = config.get("protocol", {})
    active = protocol.get("active_gate_profile")
    if not isinstance(active, str) or not active.strip():
        print("[MultiSeed] skip active gate evaluation: protocol.active_gate_profile is not set")
        return None
    active_profile = active.strip()

    presets = _profile_presets_from_protocol(protocol)
    if active_profile not in presets:
        raise ValueError(
            f"active_gate_profile '{active_profile}' is not available in protocol.gate_profiles/gate: "
            f"{sorted(presets)}"
        )

    baseline_summary = protocol.get("baseline_summary")
    baseline = None
    baseline_summary_path: Optional[str] = None
    if isinstance(baseline_summary, str) and baseline_summary.strip():
        baseline_summary_path = baseline_summary.strip()
        baseline_path = Path(baseline_summary_path)
        if not baseline_path.exists():
            raise FileNotFoundError(f"baseline summary not found: {baseline_summary_path}")
        baseline = _load_json(baseline_summary_path)

    report = evaluate_gate(candidate=summary, baseline=baseline, **presets[active_profile])
    payload = {
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "summary_path": str(summary_path),
        "baseline_summary": baseline_summary_path,
        "active_gate_profile": active_profile,
        "profile_config": presets[active_profile],
        "report": report,
    }

    gate_out = summary_path.with_name(f"{summary_path.stem}_gate_{active_profile}.json")
    gate_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[MultiSeed] active gate profile={active_profile} pass={report['gate_pass']}")
    print("[MultiSeed] active gate report file:", gate_out)
    return gate_out


def _pooled_mcnemar_from_runs(runs: List[Dict]) -> Optional[Tuple[int, int, float, float]]:
    """
    run별 McNemar discordant count(b,c)를 합산해 pooled McNemar를 계산한다.

    Returns:
        (b_total, c_total, statistic, pvalue) or None
    """
    b_total = 0
    c_total = 0
    used = 0

    for r in runs:
        b = r.get("mcnemar_b")
        c = r.get("mcnemar_c")
        if b is None or c is None:
            continue
        b_total += int(b)
        c_total += int(c)
        used += 1

    if used == 0:
        return None
    if b_total + c_total == 0:
        return int(b_total), int(c_total), 0.0, 1.0

    statistic = (abs(b_total - c_total) - 1) ** 2 / (b_total + c_total)
    # chi-square survival(df=1): sf(x)=erfc(sqrt(x/2))
    pvalue = float(erfc(sqrt(statistic / 2.0)))
    return int(b_total), int(c_total), float(statistic), pvalue


def _downside_stats(f1_diffs: np.ndarray, cvar_alpha: float = 0.1) -> Dict[str, float]:
    alpha = float(cvar_alpha)
    if alpha <= 0.0 or alpha > 1.0:
        raise ValueError(f"cvar_alpha must satisfy 0 < alpha <= 1, got {alpha}")

    n = int(f1_diffs.size)
    if n <= 0:
        return {
            "negative_rate": 0.0,
            "downside_mean": 0.0,
            "cvar_downside": 0.0,
            "cvar_f1_diff": 0.0,
            "worst_case_loss": 0.0,
            "downside_cvar_alpha": alpha,
        }

    losses = np.maximum(0.0, -f1_diffs)
    worst_case = float(np.max(losses))
    k = min(n, max(1, int(ceil(alpha * n))))
    tail = np.sort(f1_diffs)[:k]
    tail_losses = np.maximum(0.0, -tail)

    return {
        "negative_rate": float(np.mean(f1_diffs < 0.0)),
        "downside_mean": float(np.mean(losses)),
        "cvar_downside": float(np.mean(tail_losses)),
        "cvar_f1_diff": float(np.mean(tail)),
        "worst_case_loss": worst_case,
        "downside_cvar_alpha": alpha,
    }


def main() -> None:
    args = parse_args()
    base_cfg = load_config(args.config)
    seeds = _parse_seed_list(args.seeds)

    # split/router seed는 루프 변수 seed와 독립적으로 고정한다.
    # collector seed만 루프에서 바꿔서 데이터 샘플링 변동만 관측한다.
    split_cfg = base_cfg.get("split", {})
    collector_cfg = base_cfg.get("collector", {})
    router_cfg = base_cfg.get("router", {}).get("model", {})

    base_split_seed = int(split_cfg.get("seed", collector_cfg.get("seed", 0)))
    split_seed_source = "base_cfg.split.seed" if "seed" in split_cfg else "base_cfg.collector.seed_fallback"

    base_router_seed = int(router_cfg.get("random_state", 42))
    router_seed_source = "base_cfg.router.model.random_state" if "random_state" in router_cfg else "default_42"

    runs: List[Dict] = []
    for seed in seeds:
        cfg = deepcopy(base_cfg)
        cfg.setdefault("collector", {})["seed"] = seed
        cfg.setdefault("split", {})["seed"] = base_split_seed
        cfg.setdefault("router", {}).setdefault("model", {})["random_state"] = base_router_seed

        print("\n" + "=" * 30)
        print(f"[MultiSeed] start seed={seed}")
        result = run_phase2_patha(cfg)

        p1_best = result["phase2_vs_phase1_best"]["phase1_best"]
        p2_best = result["phase2_vs_phase1_best"]["phase2_best"]
        p1_f1 = float(result["phase1_meta"][p1_best]["macro_f1"])
        p2_f1 = float(result["phase2_meta"][p2_best]["macro_f1"])

        best_cmp = result["phase2_vs_phase1_best"]
        mb = best_cmp.get("mcnemar_b")
        mc = best_cmp.get("mcnemar_c")
        mn = best_cmp.get("n_test_samples")

        run = {
            "seed": seed,
            "seed_meta": {
                "collector_seed": seed,
                "split_seed": base_split_seed,
                "router_random_state": base_router_seed,
                "split_seed_source": split_seed_source,
                "router_seed_source": router_seed_source,
            },
            "phase1_best": p1_best,
            "phase2_best": p2_best,
            "phase1_best_f1": p1_f1,
            "phase2_best_f1": p2_f1,
            "f1_diff": float(best_cmp["f1_diff"]),
            "mcnemar_statistic": float(best_cmp.get("mcnemar_statistic", 0.0)),
            "mcnemar_pvalue": float(best_cmp["mcnemar_pvalue"]),
            "mcnemar_b": int(mb) if mb is not None else None,
            "mcnemar_c": int(mc) if mc is not None else None,
            "n_test_samples": int(mn) if mn is not None else None,
            "significant": bool(best_cmp["significant"]),
            "elapsed_seconds": float(result["elapsed_seconds"]),
            "artifacts": result.get("artifacts", {}),
            "result_path": result.get("result_path", ""),
        }
        runs.append(run)
        print(
            f"[MultiSeed] done seed={seed} "
            f"p1={p1_f1:.4f} p2={p2_f1:.4f} diff={run['f1_diff']:+.4f}"
        )

    f1_diffs = np.array([r["f1_diff"] for r in runs], dtype=float)
    p1_vals = np.array([r["phase1_best_f1"] for r in runs], dtype=float)
    p2_vals = np.array([r["phase2_best_f1"] for r in runs], dtype=float)
    pos_count = int(np.sum(f1_diffs > 0.0))
    neg_count = int(np.sum(f1_diffs < 0.0))
    zero_count = int(np.sum(f1_diffs == 0.0))
    sign_pvalue = _exact_sign_test_two_sided(pos_count, neg_count)
    pooled = _pooled_mcnemar_from_runs(runs)
    downside = _downside_stats(f1_diffs, cvar_alpha=0.1)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "config": args.config,
        "seeds": seeds,
        "n_runs": len(runs),
        "runs": runs,
        "aggregate": {
            "phase1_best_f1_mean": float(np.mean(p1_vals)),
            "phase1_best_f1_std": float(np.std(p1_vals, ddof=0)),
            "phase2_best_f1_mean": float(np.mean(p2_vals)),
            "phase2_best_f1_std": float(np.std(p2_vals, ddof=0)),
            "f1_diff_mean": float(np.mean(f1_diffs)),
            "f1_diff_std": float(np.std(f1_diffs, ddof=0)),
            "f1_diff_min": float(np.min(f1_diffs)),
            "f1_diff_max": float(np.max(f1_diffs)),
            "significant_count": int(sum(1 for r in runs if r["significant"])),
            "positive_count": pos_count,
            "negative_count": neg_count,
            "zero_count": zero_count,
            "sign_test_pvalue": sign_pvalue,
            **downside,
        },
    }

    if pooled is not None:
        b_total, c_total, stat, pval = pooled
        summary["aggregate"].update(
            {
                "pooled_mcnemar_b": b_total,
                "pooled_mcnemar_c": c_total,
                "pooled_mcnemar_statistic": stat,
                "pooled_mcnemar_pvalue": pval,
                "pooled_mcnemar_significant": bool(pval < 0.05),
            }
        )

    if args.summary_out:
        out_path = Path(args.summary_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        save_dir = Path(base_cfg.get("output", {}).get("save_dir", "experiments/results/phase2_patha"))
        save_dir.mkdir(parents=True, exist_ok=True)
        out_path = save_dir / f"phase2_patha_multiseed_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n[MultiSeed] summary file:", out_path)
    print("[MultiSeed] aggregate:", summary["aggregate"])
    _evaluate_active_gate(summary=summary, summary_path=out_path, config=base_cfg)


if __name__ == "__main__":
    main()
