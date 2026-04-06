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
        description="Convert a 3-class backbone_eval JSONL into a SpecM-style 2-class auxiliary JSONL."
    )
    parser.add_argument(
        "--source-prefix",
        type=str,
        default="mobileclip_s2_finetuned",
        help="backbone_eval prefix to adapt",
    )
    parser.add_argument(
        "--source-ts",
        type=str,
        default="20260319_061834",
        help="shared timestamp of the source backbone_eval JSONLs",
    )
    parser.add_argument(
        "--aux-key",
        type=str,
        default="clipft_aux",
        help="output auxiliary model key used by the ARV scripts",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DATASETS,
        help="datasets to adapt",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def project_binary(row: Dict) -> Dict:
    scores = row["scores"]
    p_auth = float(scores["authentic"])
    p_manip = float(scores["manipulated"])
    total = p_auth + p_manip
    if total > 0.0:
        p_auth_bin = p_auth / total
        p_manip_bin = p_manip / total
    else:
        p_auth_bin = 0.5
        p_manip_bin = 0.5

    pred_label = "authentic" if p_auth_bin >= p_manip_bin else "manipulated"
    confidence = max(p_auth_bin, p_manip_bin)

    return {
        "image_path": row["image_path"],
        "true_label": row["true_label"],
        "pred_label": pred_label,
        "authentic_score": p_auth_bin,
        "manip_score": p_manip_bin,
        "confidence": confidence,
        "source_prefix": row.get("source_prefix", ""),
    }


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary = {
        "timestamp": out_ts,
        "source_prefix": args.source_prefix,
        "source_ts": args.source_ts,
        "aux_key": args.aux_key,
        "datasets": {},
    }

    for ds in args.datasets:
        src = BE_DIR / f"{args.source_prefix}_{ds}_{args.source_ts}.jsonl"
        if not src.exists():
            raise FileNotFoundError(f"missing source JSONL: {src}")
        rows = load_jsonl(src)
        adapted = []
        for row in rows:
            new_row = project_binary(row)
            new_row["source_prefix"] = args.source_prefix
            adapted.append(new_row)

        out_path = OUT_DIR / f"specm_{args.aux_key}_{ds}_{out_ts}.jsonl"
        with open(out_path, "w") as f:
            for row in adapted:
                f.write(json.dumps(row) + "\n")

        summary["datasets"][ds] = {
            "source": str(src),
            "output": str(out_path),
            "rows": len(adapted),
        }
        print(f"[{ds}] saved {out_path.name} ({len(adapted)} rows)")

    summary_path = OUT_DIR / f"specm_{args.aux_key}_summary_{out_ts}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nsummary: {summary_path}")


if __name__ == "__main__":
    main()
