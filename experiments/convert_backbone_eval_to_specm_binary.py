#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parents[1]
BE_DIR = ROOT / "experiments" / "results" / "backbone_eval"
OUT_DIR = ROOT / "experiments" / "results" / "specm_eval"
DATASETS = ["base", "dsC", "opensdi", "aigenproxy"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert 3-class backbone_eval JSONL into 2-class SpecM-style JSONL."
    )
    parser.add_argument(
        "--source-prefix",
        type=str,
        default="mobileclip_s2_finetuned",
        help="backbone_eval prefix to convert",
    )
    parser.add_argument(
        "--source-ts",
        type=str,
        default="20260319_061834",
        help="timestamp of the backbone_eval JSONL set",
    )
    parser.add_argument(
        "--target-model-key",
        type=str,
        default="clipft4bin",
        help="specm model key prefix for output files",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def convert_record(rec: Dict) -> Dict:
    scores = rec.get("scores")
    if not isinstance(scores, dict):
        raise KeyError(f"Missing scores dict in record: {rec.get('image_path')}")

    p_auth = float(scores.get("authentic", 0.0))
    p_manip = float(scores.get("manipulated", 0.0))
    p_aigen = float(scores.get("ai_generated", 0.0))
    denom = p_auth + p_manip

    if denom > 0.0:
        auth_bin = p_auth / denom
        manip_bin = p_manip / denom
    else:
        auth_bin = 0.5
        manip_bin = 0.5

    pred_label = "authentic" if auth_bin >= manip_bin else "manipulated"
    return {
        "image_path": rec["image_path"],
        "true_label": rec["true_label"],
        "pred_label": pred_label,
        "authentic_score": auth_bin,
        "manip_score": manip_bin,
        "confidence": max(auth_bin, manip_bin),
        "source_pred_label": rec.get("pred_label", rec.get("verdict")),
        "source_confidence": float(rec.get("confidence", 0.0)),
        "source_aigen_score": p_aigen,
    }


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    summary = {
        "timestamp": datetime.now().isoformat(),
        "source_prefix": args.source_prefix,
        "source_ts": args.source_ts,
        "target_model_key": args.target_model_key,
        "datasets": {},
    }

    for ds_name in DATASETS:
        src_path = BE_DIR / f"{args.source_prefix}_{ds_name}_{args.source_ts}.jsonl"
        if not src_path.exists():
            raise FileNotFoundError(src_path)

        recs = load_jsonl(src_path)
        converted = [convert_record(rec) for rec in recs]
        out_path = OUT_DIR / f"specm_{args.target_model_key}_{ds_name}_{out_ts}.jsonl"
        with open(out_path, "w") as f:
            for row in converted:
                f.write(json.dumps(row) + "\n")

        summary["datasets"][ds_name] = {
            "source_jsonl": str(src_path),
            "output_jsonl": str(out_path),
            "n_records": len(converted),
        }
        print(f"[{ds_name:10s}] saved {out_path}")

    summary_path = OUT_DIR / f"specm_{args.target_model_key}_summary_{out_ts}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nsummary: {summary_path}")


if __name__ == "__main__":
    main()
