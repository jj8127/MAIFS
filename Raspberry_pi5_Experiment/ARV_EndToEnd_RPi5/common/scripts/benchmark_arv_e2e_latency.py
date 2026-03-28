#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any

from arv_stage2_runtime import ARVStage2Runtime
from rpi5_infer_cpu import ShieldCPU, load_image as load_image_cpu
from rpi5_infer_coral import (
    DEFAULT_W_SPEC_CORAL_FT,
    ShieldCoral,
)


SCRIPT_DIR = Path(__file__).resolve().parent
COMMON_DIR = SCRIPT_DIR.parent
BUNDLE_ROOT = COMMON_DIR.parent
RPI_ROOT = SCRIPT_DIR.parents[2]
SHARED_MODEL_DIR = RPI_ROOT / "common" / "models"
ASSET_IMAGE = BUNDLE_ROOT / "assets" / "benchmark_input.png"
RESULTS_DIR = COMMON_DIR / "results"
LOGS_DIR = COMMON_DIR / "logs"

CPU_MNV2 = SHARED_MODEL_DIR / "onnx_quant" / "mnv2_int8_dynamic.onnx"
CPU_SPECM = SHARED_MODEL_DIR / "onnx_quant" / "specm_v4_int8_static.onnx"
CORAL_MNV2 = SHARED_MODEL_DIR / "tflite_edgetpu_sweep" / "mnv2_coral_qsweep_qtpc_cal064_ioint8_edgetpu.tflite"
CORAL_SPECM = SHARED_MODEL_DIR / "tflite_edgetpu" / "specm_v4_coral_ft_int8_full_edgetpu.tflite"

BENCHMARK_PROTOCOLS = {
    "paper_v2": {"warmup": 0, "runs": 10, "threads": 4},
    "extended": {"warmup": 5, "runs": 30, "threads": 4},
}


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


def date_stamp() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d")


def now_local_iso() -> str:
    return datetime.now().astimezone().isoformat()


def detect_device_model() -> str:
    dt_path = Path("/proc/device-tree/model")
    if dt_path.exists():
        try:
            return dt_path.read_text(encoding="utf-8", errors="ignore").replace("\x00", "").strip()
        except OSError:
            pass
    return platform.uname().machine


def append_log(log_path: Path, message: str) -> None:
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(message.rstrip() + "\n")


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "avg": round(statistics.mean(values), 3),
        "std": round(statistics.pstdev(values), 3),
        "min": round(min(values), 3),
        "max": round(max(values), 3),
    }


def default_protocol_value(protocol: str, key: str) -> int:
    return int(BENCHMARK_PROTOCOLS[protocol][key])


def collect_edgetpu_system_checks() -> dict[str, Any]:
    import subprocess

    def cmd_lines(command: str) -> list[str]:
        proc = subprocess.run(["bash", "-lc", command], capture_output=True, text=True)
        txt = (proc.stdout or "").strip()
        if not txt:
            return []
        return [line.strip() for line in txt.splitlines() if line.strip()]

    return {
        "lsusb_google_or_coral": cmd_lines("lsusb 2>/dev/null | grep -i 'Google\\|Coral' || true"),
        "lspci_apex": cmd_lines("lspci 2>/dev/null | grep -i apex || true"),
        "dev_apex": cmd_lines("ls /dev/apex* 2>/dev/null || true"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ARV end-to-end Raspberry Pi 5 latency benchmark")
    parser.add_argument("--mode", required=True, choices=["cpu", "coral", "pcie-hat"])
    parser.add_argument("--image", type=Path, default=ASSET_IMAGE)
    parser.add_argument("--protocol", default="paper_v2", choices=sorted(BENCHMARK_PROTOCOLS))
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--runs", type=int, default=None)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--delegate-path", default="libedgetpu.so.1")
    parser.add_argument("--w-spec", type=float, default=None)
    parser.add_argument("--sub-type", default="")
    parser.add_argument("--arv-model-key", default="all", help="base|dsC|opensdi|aigenproxy|all")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-log", type=Path, default=None)
    return parser


def make_stage1_runner(args: argparse.Namespace):
    if args.mode == "cpu":
        threads = args.threads if args.threads is not None else default_protocol_value(args.protocol, "threads")
        return ShieldCPU(CPU_MNV2, CPU_SPECM, threads=threads), 1.0
    w_spec = args.w_spec if args.w_spec is not None else DEFAULT_W_SPEC_CORAL_FT
    return ShieldCoral(CORAL_MNV2, CORAL_SPECM, delegate_path=args.delegate_path), w_spec


def benchmark_one_model(
    stage1_runner,
    x,
    w_spec: float,
    runtime: ARVStage2Runtime,
    model_key: str,
    sub_type: str,
    warmup: int,
    runs: int,
) -> dict[str, Any]:
    real_stage1_total_vals: list[float] = []
    real_stage2_total_vals: list[float] = []
    real_e2e_total_vals: list[float] = []
    real_feature_vals: list[float] = []
    real_predict_vals: list[float] = []
    forced_stage2_total_vals: list[float] = []
    forced_e2e_total_vals: list[float] = []
    forced_feature_vals: list[float] = []
    forced_predict_vals: list[float] = []
    mnv2_vals: list[float] = []
    specm_vals: list[float] = []
    stage2_exec_flags: list[int] = []
    last_real: dict[str, Any] = {}
    last_forced: dict[str, Any] = {}

    total_iters = max(warmup, 0) + max(runs, 0)
    if total_iters <= 0:
        raise ValueError("runs must be >= 1")

    for idx in range(total_iters):
        stage1 = stage1_runner.predict_array(x, w_spec=w_spec)
        base_scores = stage1["mnv2_scores"]
        aux_scores = stage1["specm_scores"]
        base_conf = max(base_scores.values())
        aux_conf = max(aux_scores.values())
        stage1_total = float(stage1["latency"]["total_ms"])
        mnv2_ms = float(stage1["latency"]["mnv2_ms"])
        specm_ms = float(stage1["latency"]["specm_ms"])

        real_decision = runtime.decide(
            model_key=model_key,
            base_scores3=base_scores,
            aux_scores2=aux_scores,
            base_conf=base_conf,
            aux_conf=aux_conf,
            sub_type=sub_type,
            force_stage2=False,
        )
        forced_decision = runtime.decide(
            model_key=model_key,
            base_scores3=base_scores,
            aux_scores2=aux_scores,
            base_conf=base_conf,
            aux_conf=aux_conf,
            sub_type=sub_type,
            force_stage2=True,
        )

        if idx < warmup:
            continue

        real_stage1_total_vals.append(stage1_total)
        mnv2_vals.append(mnv2_ms)
        specm_vals.append(specm_ms)
        real_stage2_total_vals.append(real_decision.total_ms)
        real_feature_vals.append(real_decision.feature_ms)
        real_predict_vals.append(real_decision.predict_ms)
        real_e2e_total_vals.append(stage1_total + real_decision.total_ms)
        forced_stage2_total_vals.append(forced_decision.total_ms)
        forced_feature_vals.append(forced_decision.feature_ms)
        forced_predict_vals.append(forced_decision.predict_ms)
        forced_e2e_total_vals.append(stage1_total + forced_decision.total_ms)
        stage2_exec_flags.append(1 if real_decision.override_present else 0)

        last_real = {
            "base_bin_label": real_decision.base_bin_label,
            "stage1_bin_label": real_decision.stage1_bin_label,
            "final_label": real_decision.final_label,
            "action": real_decision.action,
            "override_present": real_decision.override_present,
            "keep_prob": None if real_decision.keep_prob is None else round(real_decision.keep_prob, 6),
            "tau": real_decision.tau,
            "feature_len": real_decision.feature_len,
        }
        last_forced = {
            "base_bin_label": forced_decision.base_bin_label,
            "stage1_bin_label": forced_decision.stage1_bin_label,
            "final_label": forced_decision.final_label,
            "action": forced_decision.action,
            "override_present": forced_decision.override_present,
            "keep_prob": None if forced_decision.keep_prob is None else round(forced_decision.keep_prob, 6),
            "tau": forced_decision.tau,
            "feature_len": forced_decision.feature_len,
        }

    return {
        "model_key": model_key,
        "tau": runtime.model_meta(model_key)["tau"],
        "sub_type": sub_type,
        "warmup_runs_discarded": int(warmup),
        "measured_runs": int(runs),
        "stage1_summary_ms": {
            "mnv2": summarize(mnv2_vals),
            "specm": summarize(specm_vals),
            "total": summarize(real_stage1_total_vals),
        },
        "real_path_summary_ms": {
            "stage2_feature": summarize(real_feature_vals),
            "stage2_predict": summarize(real_predict_vals),
            "stage2_total": summarize(real_stage2_total_vals),
            "e2e_total": summarize(real_e2e_total_vals),
        },
        "forced_stage2_summary_ms": {
            "stage2_feature": summarize(forced_feature_vals),
            "stage2_predict": summarize(forced_predict_vals),
            "stage2_total": summarize(forced_stage2_total_vals),
            "e2e_total": summarize(forced_e2e_total_vals),
        },
        "real_path_override_rate": round(sum(stage2_exec_flags) / max(len(stage2_exec_flags), 1), 4),
        "last_real_path": last_real,
        "last_forced_stage2": last_forced,
    }


def aggregate_model_results(model_results: list[dict[str, Any]]) -> dict[str, Any]:
    def avg_of(section: str, metric: str) -> float:
        vals = [float(row[section][metric]["avg"]) for row in model_results]
        return round(statistics.mean(vals), 3)

    return {
        "models_benchmarked": [row["model_key"] for row in model_results],
        "avg_real_path_e2e_ms": avg_of("real_path_summary_ms", "e2e_total"),
        "avg_forced_stage2_e2e_ms": avg_of("forced_stage2_summary_ms", "e2e_total"),
        "avg_real_stage2_ms": avg_of("real_path_summary_ms", "stage2_total"),
        "avg_forced_stage2_ms": avg_of("forced_stage2_summary_ms", "stage2_total"),
    }


def main() -> None:
    args = build_parser().parse_args()
    ensure_dirs()

    image_path = args.image if str(args.image) else ASSET_IMAGE
    if not image_path.exists():
        raise SystemExit(f"입력 이미지를 찾을 수 없습니다: {image_path}")

    warmup = args.warmup if args.warmup is not None else default_protocol_value(args.protocol, "warmup")
    runs = args.runs if args.runs is not None else default_protocol_value(args.protocol, "runs")
    if args.threads is None:
        args.threads = default_protocol_value(args.protocol, "threads")

    stamp = date_stamp()
    mode_tag = "rpi5_cpu_arv_e2e_latency" if args.mode == "cpu" else (
        "rpi5_coral_arv_e2e_latency" if args.mode == "coral" else "rpi5_pcie_hat_arv_e2e_latency"
    )
    output_json = args.output_json or (RESULTS_DIR / f"{stamp}_{mode_tag}.json")
    output_log = args.output_log or (LOGS_DIR / f"{stamp}_{mode_tag}.log")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_log.parent.mkdir(parents=True, exist_ok=True)

    append_log(output_log, f"[date_local] {now_local_iso()}")
    append_log(output_log, f"[mode] {args.mode}")
    append_log(output_log, f"[protocol] {args.protocol}")
    append_log(output_log, f"[image] {image_path}")
    append_log(output_log, f"[sub_type] {args.sub_type}")
    append_log(output_log, f"[arv_model_key] {args.arv_model_key}")

    stage1_runner, w_spec = make_stage1_runner(args)
    runtime = ARVStage2Runtime()
    model_keys = runtime.model_keys if args.arv_model_key == "all" else [args.arv_model_key]

    x = load_image_cpu(image_path)

    model_results = []
    for model_key in model_keys:
        result = benchmark_one_model(
            stage1_runner=stage1_runner,
            x=x,
            w_spec=w_spec,
            runtime=runtime,
            model_key=model_key,
            sub_type=args.sub_type,
            warmup=warmup,
            runs=runs,
        )
        model_results.append(result)
        append_log(
            output_log,
            f"[{model_key}] real_e2e_avg={result['real_path_summary_ms']['e2e_total']['avg']} ms "
            f"forced_e2e_avg={result['forced_stage2_summary_ms']['e2e_total']['avg']} ms",
        )

    payload = {
        "experiment": "rpi5_arv_e2e_latency",
        "date_local": now_local_iso(),
        "device_model": detect_device_model(),
        "mode": args.mode,
        "protocol": args.protocol,
        "image": str(image_path),
        "sub_type": args.sub_type,
        "w_spec": float(w_spec),
        "warmup_runs_discarded": int(warmup),
        "measured_runs": int(runs),
        "models": model_results,
        "aggregate": aggregate_model_results(model_results),
    }
    if args.mode in {"coral", "pcie-hat"}:
        payload["system_checks"] = collect_edgetpu_system_checks()

    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    append_log(output_log, f"[saved_json] {output_json}")


if __name__ == "__main__":
    main()
