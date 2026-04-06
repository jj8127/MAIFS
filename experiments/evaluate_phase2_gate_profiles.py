"""
Path A Phase2 게이트 프로파일 일괄 평가기.

예시:
    .venv-qwen/bin/python experiments/evaluate_phase2_gate_profiles.py \
      experiments/results/phase2_patha_scale120_feat_enhanced36_ridge/summary_10seeds_42_51.json \
      --baseline-summary experiments/results/phase2_patha_scale120/phase2_patha_multiseed_summary_scale120_10seeds_42_51_20260216.json \
      --profiles strict,sign_driven \
      --out experiments/results/phase2_patha_scale120_feat_enhanced36_ridge/gate_profiles_10seeds.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.evaluate_phase2_gate import evaluate_gate


PROFILE_PRESETS: Dict[str, Dict[str, Any]] = {
    "strict": {
        "min_runs": 10,
        "min_f1_diff_mean": 0.01,
        "min_significant_count": 3,
        "min_improvement_over_baseline": 0.0,
        "min_positive_seed_count": 0,
        "max_sign_test_pvalue": None,
        "require_pooled_mcnemar_significant": False,
        "max_pooled_mcnemar_pvalue": None,
    },
    "sign_driven": {
        "min_runs": 10,
        "min_f1_diff_mean": 0.01,
        "min_significant_count": 0,
        "min_improvement_over_baseline": 0.0,
        "min_positive_seed_count": 7,
        "max_sign_test_pvalue": 0.05,
        "require_pooled_mcnemar_significant": False,
        "max_pooled_mcnemar_pvalue": None,
    },
    "pooled_relaxed": {
        "min_runs": 10,
        "min_f1_diff_mean": 0.01,
        "min_significant_count": 0,
        "min_improvement_over_baseline": 0.0,
        "min_positive_seed_count": 0,
        "max_sign_test_pvalue": None,
        "require_pooled_mcnemar_significant": False,
        "max_pooled_mcnemar_pvalue": 0.10,
    },
    "loss_averse": {
        "min_runs": 10,
        "min_f1_diff_mean": 0.0,
        "min_significant_count": 0,
        "min_improvement_over_baseline": -0.005,
        "min_positive_seed_count": 0,
        "max_sign_test_pvalue": None,
        "require_pooled_mcnemar_significant": False,
        "max_pooled_mcnemar_pvalue": 0.8,
        "max_negative_rate": 0.40,
        "max_downside_mean": 0.008,
        "max_cvar_downside": 0.020,
        "max_worst_case_loss": 0.040,
        "downside_cvar_alpha": 0.1,
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate multiple gate profiles from one candidate summary")
    p.add_argument("candidate_summary", type=str, help="Candidate multiseed summary json path")
    p.add_argument(
        "--config",
        type=str,
        default="",
        help="Optional experiment config yaml path. If set, load protocol baseline/profile presets from config.",
    )
    p.add_argument("--baseline-summary", type=str, default="", help="Optional baseline summary path")
    p.add_argument(
        "--profiles",
        type=str,
        default="auto",
        help="Comma-separated profile names. Use 'auto' to use protocol.active_gate_profile if present.",
    )
    p.add_argument("--out", type=str, default="", help="Optional output report json path")
    return p.parse_args()


def _load_json(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_yaml(path: str) -> Dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _parse_profiles(raw: str) -> List[str]:
    names = [x.strip() for x in raw.split(",") if x.strip()]
    if not names:
        raise ValueError("No profiles specified")
    unknown = [x for x in names if x not in PROFILE_PRESETS]
    if unknown:
        raise ValueError(f"Unknown profiles: {unknown}; available={sorted(PROFILE_PRESETS)}")
    return names


def _resolve_profile_names(
    raw_profiles: str,
    profile_presets: Dict[str, Dict[str, Any]],
    active_profile: Optional[str] = None,
) -> List[str]:
    token = raw_profiles.strip()
    if token.lower() == "auto":
        if active_profile:
            if active_profile not in profile_presets:
                raise ValueError(
                    f"active_gate_profile '{active_profile}' is not defined in available profiles: "
                    f"{sorted(profile_presets)}"
                )
            return [active_profile]
        return list(profile_presets.keys())

    names = [x.strip() for x in token.split(",") if x.strip()]
    if not names:
        raise ValueError("No profiles specified")
    unknown = [x for x in names if x not in profile_presets]
    if unknown:
        raise ValueError(f"Unknown profiles: {unknown}; available={sorted(profile_presets)}")
    return names


def _profiles_from_config(config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    protocol.gate_profiles 우선, 없으면 protocol.gate 단일 항목을 strict로 변환.
    """
    protocol = config.get("protocol", {})
    presets = protocol.get("gate_profiles")
    def _alpha(v: Any) -> float:
        return float(0.1 if v is None else v)

    if isinstance(presets, dict) and presets:
        out: Dict[str, Dict[str, Any]] = {}
        for name, cfg in presets.items():
            if not isinstance(cfg, dict):
                continue
            out[str(name)] = {
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
                "downside_cvar_alpha": _alpha(cfg.get("downside_cvar_alpha", 0.1)),
            }
        if out:
            return out

    gate = protocol.get("gate", {})
    if isinstance(gate, dict) and gate:
        return {
            "strict": {
                "min_runs": int(gate.get("min_runs", 10)),
                "min_f1_diff_mean": float(gate.get("min_f1_diff_mean", 0.01)),
                "min_significant_count": int(gate.get("min_significant_count", 3)),
                "min_improvement_over_baseline": float(gate.get("min_improvement_over_baseline", 0.0)),
                "min_positive_seed_count": int(gate.get("min_positive_seed_count", 0)),
                "max_sign_test_pvalue": gate.get("max_sign_test_pvalue", None),
                "require_pooled_mcnemar_significant": bool(gate.get("require_pooled_mcnemar_significant", False)),
                "max_pooled_mcnemar_pvalue": gate.get("max_pooled_mcnemar_pvalue", None),
                "max_negative_rate": gate.get("max_negative_rate", None),
                "max_downside_mean": gate.get("max_downside_mean", None),
                "max_cvar_downside": gate.get("max_cvar_downside", None),
                "max_worst_case_loss": gate.get("max_worst_case_loss", None),
                "downside_cvar_alpha": _alpha(gate.get("downside_cvar_alpha", 0.1)),
            }
        }

    return {}


def main() -> None:
    args = parse_args()
    candidate = _load_json(args.candidate_summary)
    profile_presets = dict(PROFILE_PRESETS)
    baseline_summary_path: Optional[str] = args.baseline_summary or None
    active_profile: Optional[str] = None

    config = None
    if args.config:
        config = _load_yaml(args.config)
        ap = config.get("protocol", {}).get("active_gate_profile")
        if isinstance(ap, str) and ap.strip():
            active_profile = ap.strip()
        from_cfg = _profiles_from_config(config)
        if from_cfg:
            profile_presets = from_cfg
        if not baseline_summary_path:
            b = config.get("protocol", {}).get("baseline_summary")
            if isinstance(b, str) and b.strip():
                baseline_summary_path = b.strip()

    baseline = _load_json(baseline_summary_path) if baseline_summary_path else None
    profile_names = _resolve_profile_names(
        raw_profiles=args.profiles,
        profile_presets=profile_presets,
        active_profile=active_profile,
    )

    results: Dict[str, Dict[str, Any]] = {}
    for name in profile_names:
        cfg = profile_presets[name]
        report = evaluate_gate(candidate=candidate, baseline=baseline, **cfg)
        results[name] = report

    payload = {
        "timestamp": datetime.now().isoformat(),
        "candidate_summary": args.candidate_summary,
        "baseline_summary": baseline_summary_path,
        "config": args.config or None,
        "active_gate_profile": active_profile,
        "profiles": profile_names,
        "profile_presets": profile_presets,
        "results": results,
    }

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print("[GateProfiles] report file:", out_path)

    for name in profile_names:
        report = results[name]
        print(f"[GateProfiles] {name}: pass={report['gate_pass']}")
        sign_stats = report.get("candidate", {}).get("sign_stats", {})
        if sign_stats:
            print(
                "  sign stats:",
                f"+={sign_stats.get('positive_count')}",
                f"-={sign_stats.get('negative_count')}",
                f"0={sign_stats.get('zero_count')}",
                f"p={sign_stats.get('sign_test_pvalue')}",
            )


if __name__ == "__main__":
    main()
