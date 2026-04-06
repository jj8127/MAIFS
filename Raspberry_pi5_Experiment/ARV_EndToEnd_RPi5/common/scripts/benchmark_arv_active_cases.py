#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from arv_stage2_runtime import ARVStage2Runtime
from benchmark_arv_e2e_latency import (
    BENCHMARK_PROTOCOLS,
    append_log,
    collect_edgetpu_system_checks,
    date_stamp,
    detect_device_model,
    default_protocol_value,
    make_stage1_runner,
    summarize,
)
from rpi5_infer_cpu import load_image as load_image_cpu


SCRIPT_DIR = Path(__file__).resolve().parent
COMMON_DIR = SCRIPT_DIR.parent
RESULTS_DIR = COMMON_DIR / "results"
LOGS_DIR = COMMON_DIR / "logs"
ACTIVE_ACTIONS = {"keep_change", "revert_to_base"}


def now_local_iso() -> str:
    return datetime.now().astimezone().isoformat()


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark only the real ARV-active keep/revert path using a discovery manifest."
    )
    parser.add_argument("--mode", required=True, choices=["cpu", "coral", "pcie-hat"])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--case-id", action="append", default=[], help="특정 case_id만 반복 측정")
    parser.add_argument("--action", default="all", choices=["all", "keep_change", "revert_to_base"])
    parser.add_argument("--protocol", default="paper_v2", choices=sorted(BENCHMARK_PROTOCOLS))
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--runs", type=int, default=None)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--delegate-path", default="libedgetpu.so.1")
    parser.add_argument("--w-spec", type=float, default=None)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--allow-inactive", action="store_true", help="active path가 아니어도 실패하지 않고 기록만 남김")
    # run_arv_active_workflow.sh all 모드에서 discovery용 옵션이 그대로 전달될 수 있으므로,
    # benchmark 단계에서는 아래 옵션들을 받아만 두고 사용하지 않는다.
    parser.add_argument("--sub-type", default="", help=argparse.SUPPRESS)
    parser.add_argument("--infer-sub-type-from-path", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--arv-model-key", default="all", help=argparse.SUPPRESS)
    parser.add_argument("--max-images", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-log", type=Path, default=None)
    return parser


def load_cases(manifest_path: Path, action: str, case_ids: list[str], max_cases: int | None) -> list[dict[str, Any]]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases = payload.get("cases", [])
    if action != "all":
        cases = [row for row in cases if row.get("action") == action]
    if case_ids:
        want = set(case_ids)
        cases = [row for row in cases if row.get("case_id") in want]
    if max_cases is not None and max_cases >= 0:
        cases = cases[:max_cases]
    return cases


def benchmark_case(
    stage1_runner,
    runtime: ARVStage2Runtime,
    case: dict[str, Any],
    w_spec: float,
    warmup: int,
    runs: int,
    allow_inactive: bool,
) -> dict[str, Any]:
    image_path = Path(case["image_path"])
    if not image_path.exists():
        raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {image_path}")

    x = load_image_cpu(image_path)
    model_key = str(case["model_key"])
    sub_type = str(case.get("sub_type", ""))

    stage1_vals: list[float] = []
    feature_vals: list[float] = []
    predict_vals: list[float] = []
    stage2_vals: list[float] = []
    e2e_vals: list[float] = []
    action_counts: Counter[str] = Counter()
    active_run_count = 0
    measured = 0
    last_decision: dict[str, Any] = {}

    total_iters = max(warmup, 0) + max(runs, 0)
    for idx in range(total_iters):
        stage1 = stage1_runner.predict_array(x, w_spec=w_spec)
        base_scores = stage1["mnv2_scores"]
        aux_scores = stage1["specm_scores"]
        base_conf = max(base_scores.values())
        aux_conf = max(aux_scores.values())
        decision = runtime.decide(
            model_key=model_key,
            base_scores3=base_scores,
            aux_scores2=aux_scores,
            base_conf=base_conf,
            aux_conf=aux_conf,
            sub_type=sub_type,
            force_stage2=False,
        )
        if idx < warmup:
            continue

        measured += 1
        action_counts[decision.action] += 1
        if decision.action in ACTIVE_ACTIONS:
            active_run_count += 1
            stage1_total = float(stage1["latency"]["total_ms"])
            stage1_vals.append(stage1_total)
            feature_vals.append(float(decision.feature_ms))
            predict_vals.append(float(decision.predict_ms))
            stage2_vals.append(float(decision.total_ms))
            e2e_vals.append(stage1_total + float(decision.total_ms))

        last_decision = {
            "action": decision.action,
            "base_bin_label": decision.base_bin_label,
            "stage1_bin_label": decision.stage1_bin_label,
            "final_label": decision.final_label,
            "override_present": decision.override_present,
            "keep_prob": None if decision.keep_prob is None else round(float(decision.keep_prob), 6),
            "tau": decision.tau,
            "feature_len": decision.feature_len,
        }

    if active_run_count == 0 and not allow_inactive:
        raise RuntimeError(
            f"case_id={case.get('case_id')}는 측정 중 실제 ARV-active path가 나오지 않았습니다. "
            "먼저 discovery 결과를 다시 확인하거나 --allow-inactive로 비엄격 실행을 사용하세요."
        )

    return {
        "case_id": case["case_id"],
        "image_path": str(image_path),
        "image_name": image_path.name,
        "sub_type": sub_type,
        "model_key": model_key,
        "expected_action": case.get("action"),
        "warmup_runs_discarded": int(warmup),
        "measured_runs": int(measured),
        "active_run_count": int(active_run_count),
        "active_run_rate": round(active_run_count / max(measured, 1), 4),
        "observed_action_counts": dict(action_counts),
        "active_real_path_summary_ms": None
        if not e2e_vals
        else {
            "stage1_total": summarize(stage1_vals),
            "stage2_feature": summarize(feature_vals),
            "stage2_predict": summarize(predict_vals),
            "stage2_total": summarize(stage2_vals),
            "e2e_total": summarize(e2e_vals),
        },
        "last_decision": last_decision,
    }


def aggregate_case_results(case_results: list[dict[str, Any]]) -> dict[str, Any]:
    active_cases = [row for row in case_results if row["active_real_path_summary_ms"] is not None]
    if not active_cases:
        return {
            "cases_benchmarked": [row["case_id"] for row in case_results],
            "active_case_count": 0,
            "avg_active_e2e_ms": None,
            "avg_active_stage2_ms": None,
        }

    e2e_vals = [float(row["active_real_path_summary_ms"]["e2e_total"]["avg"]) for row in active_cases]
    stage2_vals = [float(row["active_real_path_summary_ms"]["stage2_total"]["avg"]) for row in active_cases]
    return {
        "cases_benchmarked": [row["case_id"] for row in case_results],
        "active_case_count": len(active_cases),
        "avg_active_e2e_ms": round(statistics.mean(e2e_vals), 3),
        "avg_active_stage2_ms": round(statistics.mean(stage2_vals), 3),
    }


def main() -> None:
    args = build_parser().parse_args()
    ensure_dirs()

    if args.threads is None:
        args.threads = default_protocol_value(args.protocol, "threads")
    warmup = args.warmup if args.warmup is not None else default_protocol_value(args.protocol, "warmup")
    runs = args.runs if args.runs is not None else default_protocol_value(args.protocol, "runs")

    if not args.manifest.exists():
        raise SystemExit(f"manifest를 찾을 수 없습니다: {args.manifest}")

    stamp = date_stamp()
    mode_tag = (
        "rpi5_cpu_arv_active_latency"
        if args.mode == "cpu"
        else ("rpi5_coral_arv_active_latency" if args.mode == "coral" else "rpi5_pcie_hat_arv_active_latency")
    )
    output_json = args.output_json or (RESULTS_DIR / f"{stamp}_{mode_tag}.json")
    output_log = args.output_log or (LOGS_DIR / f"{stamp}_{mode_tag}.log")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_log.parent.mkdir(parents=True, exist_ok=True)

    append_log(output_log, f"[date_local] {now_local_iso()}")
    append_log(output_log, f"[mode] {args.mode}")
    append_log(output_log, f"[manifest] {args.manifest}")
    append_log(output_log, f"[protocol] {args.protocol}")

    selected_cases = load_cases(args.manifest, args.action, args.case_id, args.max_cases)
    if not selected_cases:
        raise SystemExit("선택된 active case가 없습니다. discovery manifest를 먼저 확인하세요.")

    append_log(output_log, f"[selected_cases] {len(selected_cases)}")

    stage1_runner, w_spec = make_stage1_runner(args)
    runtime = ARVStage2Runtime()

    case_results = []
    for case in selected_cases:
        row = benchmark_case(
            stage1_runner=stage1_runner,
            runtime=runtime,
            case=case,
            w_spec=w_spec,
            warmup=warmup,
            runs=runs,
            allow_inactive=bool(args.allow_inactive),
        )
        case_results.append(row)
        append_log(
            output_log,
            f"[{row['case_id']}] active_rate={row['active_run_rate']} "
            f"active_e2e_avg={None if row['active_real_path_summary_ms'] is None else row['active_real_path_summary_ms']['e2e_total']['avg']}",
        )

    payload: dict[str, Any] = {
        "experiment": "arv_active_e2e_latency",
        "date_local": now_local_iso(),
        "device_model": detect_device_model(),
        "mode": args.mode,
        "protocol": args.protocol,
        "manifest": str(args.manifest),
        "w_spec": float(w_spec),
        "warmup_runs_discarded": int(warmup),
        "measured_runs": int(runs),
        "case_results": case_results,
        "aggregate": aggregate_case_results(case_results),
    }
    if args.mode in {"coral", "pcie-hat"}:
        payload["system_checks"] = collect_edgetpu_system_checks()

    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    append_log(output_log, f"[saved_json] {output_json}")


if __name__ == "__main__":
    main()
