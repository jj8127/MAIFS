#!/usr/bin/env python3
"""
One-click Raspberry Pi 5 latency benchmark runner.

- CPU: rpi5_infer_cpu.py
- Coral USB / PCIe HAT: rpi5_infer_coral.py

The default benchmark policy follows the current paper-draft protocol:
single-image warm-latency measurement with threads=4 and 10 measured runs.
If a backend cannot be measured, the script writes a structured
"unmeasured" JSON instead of failing silently.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
COMMON_DIR = SCRIPT_DIR.parent
RPI_ROOT = SCRIPT_DIR.parents[2]
SHARED_COMMON_DIR = RPI_ROOT / "common"

RESULTS_DIR = COMMON_DIR / "results"
LOGS_DIR = COMMON_DIR / "logs"

CPU_SCRIPT = SCRIPT_DIR / "rpi5_infer_cpu.py"
CORAL_SCRIPT = SCRIPT_DIR / "rpi5_infer_coral.py"

TEMP_ONNX_DIR = SHARED_COMMON_DIR / "models" / "onnx_quant"
TEMP_EDGETPU_DIR = SHARED_COMMON_DIR / "models" / "tflite_edgetpu"
TEMP_EDGETPU_SWEEP = SHARED_COMMON_DIR / "models" / "tflite_edgetpu_sweep"

CPU_MNV2_CANDIDATES = [
    TEMP_ONNX_DIR / "mnv2_int8_dynamic.onnx",
]
CPU_SPECM_CANDIDATES = [
    TEMP_ONNX_DIR / "specm_v4_int8_static.onnx",
]

CORAL_MNV2_CANDIDATES = [
    TEMP_EDGETPU_SWEEP / "mnv2_coral_qsweep_qtpc_cal064_ioint8_edgetpu.tflite",
]
CORAL_SPECM_CANDIDATES = [
    TEMP_EDGETPU_DIR / "specm_v4_coral_ft_int8_full_edgetpu.tflite",
]

DEFAULT_DELEGATE = "libedgetpu.so.1"

BENCHMARK_PROTOCOLS: dict[str, dict[str, int]] = {
    "paper_v2": {
        "warmup": 0,
        "runs": 10,
        "threads": 4,
    },
    "extended": {
        "warmup": 5,
        "runs": 30,
        "threads": 4,
    },
}


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


def resolve_first(candidates: list[Path]) -> Path:
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def detect_device_model() -> str:
    dt_path = Path("/proc/device-tree/model")
    if dt_path.exists():
        try:
            return dt_path.read_text(encoding="utf-8", errors="ignore").replace("\x00", "").strip()
        except OSError:
            pass
    return platform.uname().machine


def now_local_iso() -> str:
    return datetime.now().astimezone().isoformat()


def date_stamp() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d")


def append_log(log_path: Path, message: str) -> None:
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(message.rstrip() + "\n")


def run_capture(cmd: list[str], log_path: Path) -> subprocess.CompletedProcess[str]:
    append_log(log_path, f"$ {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    append_log(log_path, f"[returncode] {proc.returncode}")
    if proc.stdout:
        append_log(log_path, "[stdout]")
        append_log(log_path, proc.stdout)
    if proc.stderr:
        append_log(log_path, "[stderr]")
        append_log(log_path, proc.stderr)
    append_log(log_path, "-" * 80)
    return proc


def shell_lines(command: str) -> list[str]:
    proc = subprocess.run(
        ["bash", "-lc", command],
        capture_output=True,
        text=True,
    )
    text = (proc.stdout or "").strip()
    if not text:
        return []
    return [line.strip() for line in text.splitlines() if line.strip()]


def python_check(runner_python: str, code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [runner_python, "-c", code],
        capture_output=True,
        text=True,
    )


def python_available(runner_python: str, module_name: str) -> bool:
    proc = python_check(
        runner_python,
        (
            "import importlib.util; "
            f"print('1' if importlib.util.find_spec('{module_name}') is not None else '0')"
        ),
    )
    return proc.returncode == 0 and proc.stdout.strip() == "1"


def python_version(runner_python: str) -> str:
    proc = python_check(runner_python, "import sys; print(sys.version.split()[0])")
    if proc.returncode == 0:
        return proc.stdout.strip()
    return "unknown"


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "avg": round(statistics.mean(values), 3),
        "std": round(statistics.pstdev(values), 3),
        "min": round(min(values), 3),
        "max": round(max(values), 3),
    }


def collect_coral_checks(runner_python: str) -> dict[str, Any]:
    lib_lines = shell_lines("ldconfig -p 2>/dev/null | grep libedgetpu.so.1 || true")
    usb_lines = shell_lines("lsusb 2>/dev/null | grep -i 'Google\\|Coral' || true")
    pycoral_proc = python_check(
        runner_python,
        "from pycoral.utils import edgetpu; print(edgetpu.list_edge_tpus())",
    )
    return {
        "libedgetpu_present": bool(lib_lines),
        "pycoral_installed_in_runner_env": python_available(runner_python, "pycoral"),
        "tflite_runtime_installed_in_runner_env": python_available(runner_python, "tflite_runtime"),
        "lsusb_available_result": "\n".join(usb_lines) if usb_lines else "",
        "edge_tpu_list": pycoral_proc.stdout.strip() if pycoral_proc.returncode == 0 else "",
        "edge_tpu_error": pycoral_proc.stderr.strip() if pycoral_proc.returncode != 0 else "",
    }


def collect_pcie_checks(runner_python: str) -> dict[str, Any]:
    checks = collect_coral_checks(runner_python)
    checks.update(
        {
            "config_txt_pcie_lines": shell_lines(
                "grep -E 'pciex1|pcie' /boot/firmware/config.txt 2>/dev/null || true"
            ),
            "lspci_apex": shell_lines("lspci 2>/dev/null | grep -i apex || true"),
            "dev_apex": shell_lines("ls /dev/apex* 2>/dev/null || true"),
        }
    )
    return checks


def build_command(
    *,
    mode: str,
    runner_python: str,
    image: Path,
    threads: int,
    delegate_path: str,
    w_spec: float | None,
    mnv2: Path,
    specm: Path,
    warmup: int,
    runs: int,
) -> list[str]:
    if mode == "cpu":
        cmd = [
            runner_python,
            str(CPU_SCRIPT),
            str(image),
            "--json",
            "--benchmark-warmup", str(warmup),
            "--benchmark-runs",   str(runs),
            "--threads",          str(threads),
            "--mnv2",             str(mnv2),
            "--specm",            str(specm),
        ]
        if w_spec is not None:
            cmd += ["--w-spec", str(w_spec)]
        return cmd

    cmd = [
        runner_python,
        str(CORAL_SCRIPT),
        str(image),
        "--json",
        "--benchmark-warmup", str(warmup),
        "--benchmark-runs",   str(runs),
        "--delegate-path",    delegate_path,
        "--mnv2",             str(mnv2),
        "--specm",            str(specm),
    ]
    if w_spec is not None:
        cmd += ["--w-spec", str(w_spec)]
    return cmd


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def experiment_name(mode: str) -> str:
    if mode == "cpu":
        return "rpi5_cpu_latency"
    if mode == "coral":
        return "rpi5_coral_latency"
    return "rpi5_pcie_hat_latency"


def default_w_spec(mode: str) -> float:
    if mode == "cpu":
        return 1.0
    return 0.2


def protocol_defaults(protocol: str) -> dict[str, int]:
    try:
        return BENCHMARK_PROTOCOLS[protocol]
    except KeyError as exc:
        valid = ", ".join(sorted(BENCHMARK_PROTOCOLS))
        raise ValueError(f"unsupported protocol={protocol!r}; expected one of: {valid}") from exc


def resolved_int_arg(value: int | None, protocol: str, key: str) -> int:
    if value is not None:
        return int(value)
    return int(protocol_defaults(protocol)[key])


def run_benchmark(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    ensure_dirs()

    exp_name = experiment_name(args.mode)
    stamp = date_stamp()
    out_path = RESULTS_DIR / f"{stamp}_{exp_name}.json"
    log_path = LOGS_DIR / f"{stamp}_{exp_name}.log"

    if args.output_json:
        out_path = Path(args.output_json)
    if args.output_log:
        log_path = Path(args.output_log)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    model_mnv2 = args.mnv2
    model_specm = args.specm
    if model_mnv2 is None or model_specm is None:
        if args.mode == "cpu":
            model_mnv2 = model_mnv2 or resolve_first(CPU_MNV2_CANDIDATES)
            model_specm = model_specm or resolve_first(CPU_SPECM_CANDIDATES)
        else:
            model_mnv2 = model_mnv2 or resolve_first(CORAL_MNV2_CANDIDATES)
            model_specm = model_specm or resolve_first(CORAL_SPECM_CANDIDATES)

    w_spec = args.w_spec if args.w_spec is not None else default_w_spec(args.mode)

    append_log(log_path, f"[benchmark] {exp_name}")
    append_log(log_path, f"[date_local] {now_local_iso()}")
    append_log(log_path, f"[runner_python] {args.runner_python}")
    append_log(log_path, f"[mode] {args.mode}")
    append_log(log_path, f"[image] {args.image}")
    warmup = resolved_int_arg(args.warmup, args.protocol, "warmup")
    runs = resolved_int_arg(args.runs, args.protocol, "runs")
    threads = resolved_int_arg(args.threads, args.protocol, "threads")

    append_log(log_path, f"[protocol] {args.protocol}")
    append_log(log_path, f"[warmup] {warmup}")
    append_log(log_path, f"[runs] {runs}")
    append_log(log_path, "=" * 80)

    model_paths = {"mnv2": str(model_mnv2), "specm": str(model_specm)}

    if args.mode == "pcie-hat":
        checks = collect_pcie_checks(args.runner_python)
        if not checks["lspci_apex"] and not checks["dev_apex"]:
            payload = {
                "experiment": exp_name,
                "date_local": now_local_iso(),
                "status": "unmeasured",
                "reason": "PCIe Coral 장치가 탐지되지 않아 측정 불가",
                "checks": checks,
                "model_paths": model_paths,
            }
            write_json(out_path, payload)
            return payload, out_path

    cmd = build_command(
        mode=args.mode if args.mode != "pcie-hat" else "coral",
        runner_python=args.runner_python,
        image=args.image,
        threads=threads,
        delegate_path=args.delegate_path,
        w_spec=w_spec,
        mnv2=model_mnv2,
        specm=model_specm,
        warmup=warmup,
        runs=runs,
    )

    proc = run_capture(cmd, log_path)
    if proc.returncode != 0:
        checks = collect_coral_checks(args.runner_python) if args.mode != "cpu" else {}
        if args.mode == "pcie-hat":
            checks.update(collect_pcie_checks(args.runner_python))
        payload = {
            "experiment": exp_name,
            "date_local": now_local_iso(),
            "status": "unmeasured",
            "reason": (
                "Edge TPU delegate 로드 실패로 1단계 추론 시작 불가"
                if args.mode != "cpu"
                else "CPU 추론 스크립트 실행 실패"
            ),
            "checks": checks | {"script_error": (proc.stderr or proc.stdout).strip()},
            "model_paths": model_paths,
            "protocol": args.protocol,
        }
        if args.mode == "cpu":
            payload["threads"] = threads
        write_json(out_path, payload)
        return payload, out_path

    try:
        result = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        payload = {
            "experiment": exp_name,
            "date_local": now_local_iso(),
            "status": "unmeasured",
            "reason": "추론 스크립트 JSON 파싱 실패",
            "checks": {"script_stdout": proc.stdout.strip(), "script_stderr": proc.stderr.strip()},
            "model_paths": model_paths,
            "protocol": args.protocol,
        }
        write_json(out_path, payload)
        raise RuntimeError(f"JSON parse failed for {exp_name}") from exc

    summary_ms = result.get("summary_ms", {})
    mnv2_summary = summary_ms.get("mnv2", {})
    specm_summary = summary_ms.get("specm", {})
    total_summary = summary_ms.get("total", {})
    all_runs = result.get("all_runs", [])

    payload = {
        "experiment": exp_name,
        "date_local": now_local_iso(),
        "device_model": detect_device_model(),
        "os": platform.platform(),
        "python_version": python_version(args.runner_python),
        "runner_python": args.runner_python,
        "protocol": args.protocol,
        "backend": result.get("backend", ""),
        "image": str(args.image),
        "warmup_runs_discarded": result.get("warmup_runs_discarded", warmup),
        "measured_runs": result.get("measured_runs", runs),
        "model_paths": model_paths,
        "load_ms": result.get("load", {}),
        "summary_ms": {
            "mnv2_avg": mnv2_summary.get("avg"),
            "mnv2_std": mnv2_summary.get("std"),
            "specm_avg": specm_summary.get("avg"),
            "specm_std": specm_summary.get("std"),
            "total_avg": total_summary.get("avg"),
            "total_std": total_summary.get("std"),
            "total_min": total_summary.get("min"),
            "total_max": total_summary.get("max"),
        },
        "all_runs": all_runs,
    }

    if args.mode == "cpu":
        payload["threads"] = threads
    else:
        payload["delegate_path"] = args.delegate_path
        payload["w_spec"] = w_spec
        payload["connection"] = "pcie_hat" if args.mode == "pcie-hat" else "coral_usb_or_default"

    write_json(out_path, payload)
    return payload, out_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RPi5 one-click latency benchmark")
    parser.add_argument("--mode", required=True, choices=["cpu", "coral", "pcie-hat"])
    parser.add_argument("--image", type=Path, required=True, help="테스트 이미지 경로")
    parser.add_argument(
        "--protocol",
        default="paper_v2",
        choices=sorted(BENCHMARK_PROTOCOLS),
        help="벤치마크 규약 선택 (기본: 현재 논문 초안 기준 paper_v2)",
    )
    parser.add_argument(
        "--runner-python",
        default=sys.executable,
        help="실제 추론 스크립트를 실행할 Python 경로",
    )
    parser.add_argument("--warmup", type=int, default=None, help="버릴 워밍업 횟수 override")
    parser.add_argument("--runs", type=int, default=None, help="실측 횟수 override")
    parser.add_argument("--threads", type=int, default=None, help="CPU ONNX 스레드 수 override")
    parser.add_argument("--delegate-path", default=DEFAULT_DELEGATE, help="Edge TPU delegate .so 경로")
    parser.add_argument("--w-spec", type=float, default=None, help="SpecM 융합 가중치")
    parser.add_argument("--mnv2", type=Path, default=None, help="MNV2 모델 경로 override")
    parser.add_argument("--specm", type=Path, default=None, help="SpecM 모델 경로 override")
    parser.add_argument("--output-json", default=None, help="출력 JSON 경로 override")
    parser.add_argument("--output-log", default=None, help="출력 로그 경로 override")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.image.exists():
        raise SystemExit(f"이미지를 찾을 수 없습니다: {args.image}")
    payload, out_path = run_benchmark(args)
    print(json.dumps({"status": payload.get("status", "measured"), "output_json": str(out_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
