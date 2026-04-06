#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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
)
from rpi5_infer_cpu import load_image as load_image_cpu


SCRIPT_DIR = Path(__file__).resolve().parent
COMMON_DIR = SCRIPT_DIR.parent
RESULTS_DIR = COMMON_DIR / "results"
LOGS_DIR = COMMON_DIR / "logs"
ACTIVE_ACTIONS = {"keep_change", "revert_to_base"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def now_local_iso() -> str:
    return datetime.now().astimezone().isoformat()


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


def infer_sub_type(path: Path) -> str:
    token = str(path).lower()
    if "casia_tp" in token or "/tp/" in token:
        return "casia_tp"
    if "casia_au" in token or "/au/" in token:
        return "casia_au"
    if "biggan" in token:
        return "biggan"
    if "imd2020" in token or "inpaint" in token:
        return "imd2020_inpainting"
    if "opensdi" in token:
        if "partial" in token:
            return "opensdi_partial_fake"
        if "entire" in token or "whole" in token:
            return "opensdi_entire_fake"
        if "real" in token:
            return "opensdi_real"
    if "aigenproxy" in token or "aigen_proxy" in token:
        if "real" in token:
            return "aigen_proxy_real"
        if "manip" in token:
            return "aigen_proxy_manipulated"
        if "ai" in token or "generated" in token:
            return "aigen_proxy_ai_generated"
    return ""


def iter_image_paths(root: Path, limit: int | None = None) -> list[Path]:
    paths = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    paths.sort()
    if limit is not None and limit >= 0:
        return paths[:limit]
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Find real Raspberry Pi inputs where ARV stage-2 actually performs keep/revert."
    )
    parser.add_argument("--mode", required=True, choices=["cpu", "coral", "pcie-hat"])
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--protocol", default="paper_v2", choices=sorted(BENCHMARK_PROTOCOLS))
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--delegate-path", default="libedgetpu.so.1")
    parser.add_argument("--w-spec", type=float, default=None)
    parser.add_argument("--sub-type", default="")
    parser.add_argument("--infer-sub-type-from-path", action="store_true")
    parser.add_argument("--arv-model-key", default="all", help="base|dsC|opensdi|aigenproxy|all")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-log", type=Path, default=None)
    return parser


def choose_sub_type(path: Path, args: argparse.Namespace) -> str:
    if args.sub_type:
        return str(args.sub_type)
    if args.infer_sub_type_from_path:
        return infer_sub_type(path)
    return ""


def main() -> None:
    args = build_parser().parse_args()
    ensure_dirs()

    if not args.image_dir.exists():
        raise SystemExit(f"이미지 디렉토리를 찾을 수 없습니다: {args.image_dir}")

    if args.threads is None:
        args.threads = default_protocol_value(args.protocol, "threads")

    stamp = date_stamp()
    mode_tag = (
        "rpi5_cpu_arv_active_discovery"
        if args.mode == "cpu"
        else ("rpi5_coral_arv_active_discovery" if args.mode == "coral" else "rpi5_pcie_hat_arv_active_discovery")
    )
    output_json = args.output_json or (RESULTS_DIR / f"{stamp}_{mode_tag}.json")
    output_log = args.output_log or (LOGS_DIR / f"{stamp}_{mode_tag}.log")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_log.parent.mkdir(parents=True, exist_ok=True)

    append_log(output_log, f"[date_local] {now_local_iso()}")
    append_log(output_log, f"[mode] {args.mode}")
    append_log(output_log, f"[image_dir] {args.image_dir}")
    append_log(output_log, f"[protocol] {args.protocol}")
    append_log(output_log, f"[arv_model_key] {args.arv_model_key}")

    image_paths = iter_image_paths(args.image_dir, args.max_images)
    append_log(output_log, f"[images_found] {len(image_paths)}")

    stage1_runner, w_spec = make_stage1_runner(args)
    runtime = ARVStage2Runtime()
    model_keys = runtime.model_keys if args.arv_model_key == "all" else [args.arv_model_key]

    discovered_cases: list[dict[str, Any]] = []
    action_counts: Counter[str] = Counter()
    model_counts: Counter[str] = Counter()

    for idx, image_path in enumerate(image_paths, start=1):
        x = load_image_cpu(image_path)
        stage1 = stage1_runner.predict_array(x, w_spec=w_spec)
        base_scores = stage1["mnv2_scores"]
        aux_scores = stage1["specm_scores"]
        base_conf = max(base_scores.values())
        aux_conf = max(aux_scores.values())
        sub_type = choose_sub_type(image_path, args)

        for model_key in model_keys:
            decision = runtime.decide(
                model_key=model_key,
                base_scores3=base_scores,
                aux_scores2=aux_scores,
                base_conf=base_conf,
                aux_conf=aux_conf,
                sub_type=sub_type,
                force_stage2=False,
            )
            if decision.action not in ACTIVE_ACTIONS:
                continue

            case_id = f"{args.mode}_{model_key}_{decision.action}_{len(discovered_cases)+1:04d}"
            row = {
                "case_id": case_id,
                "image_path": str(image_path),
                "image_name": image_path.name,
                "sub_type": sub_type,
                "model_key": model_key,
                "action": decision.action,
                "base_bin_label": decision.base_bin_label,
                "stage1_bin_label": decision.stage1_bin_label,
                "final_label": decision.final_label,
                "override_present": decision.override_present,
                "keep_prob": None if decision.keep_prob is None else round(float(decision.keep_prob), 6),
                "tau": decision.tau,
                "stage1_latency_ms": float(stage1["latency"]["total_ms"]),
                "base_conf": float(base_conf),
                "aux_conf": float(aux_conf),
                "base_scores3": base_scores,
                "aux_scores2": aux_scores,
            }
            discovered_cases.append(row)
            action_counts[decision.action] += 1
            model_counts[model_key] += 1
            append_log(
                output_log,
                f"[active_case] {case_id} image={image_path.name} model={model_key} action={decision.action} sub_type={sub_type or '-'}",
            )

        if idx % 100 == 0:
            append_log(output_log, f"[progress] scanned={idx}/{len(image_paths)} active_cases={len(discovered_cases)}")

    payload: dict[str, Any] = {
        "experiment": "arv_active_case_discovery",
        "date_local": now_local_iso(),
        "device_model": detect_device_model(),
        "mode": args.mode,
        "protocol": args.protocol,
        "image_dir": str(args.image_dir),
        "sub_type": str(args.sub_type),
        "infer_sub_type_from_path": bool(args.infer_sub_type_from_path),
        "w_spec": float(w_spec),
        "model_keys": model_keys,
        "summary": {
            "images_scanned": len(image_paths),
            "active_case_count": len(discovered_cases),
            "action_counts": dict(action_counts),
            "model_key_counts": dict(model_counts),
        },
        "cases": discovered_cases,
    }
    if args.mode in {"coral", "pcie-hat"}:
        payload["system_checks"] = collect_edgetpu_system_checks()

    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    append_log(output_log, f"[saved_json] {output_json}")


if __name__ == "__main__":
    main()
