#!/usr/bin/env python3
"""
Download Qwen model weights from Hugging Face.

Example:
    python scripts/download_qwen_model.py \
      --model-id Qwen/Qwen2.5-32B-Instruct \
      --local-dir ~/models/qwen2.5-32b-instruct
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Qwen model snapshot")
    parser.add_argument(
        "--model-id",
        default="Qwen/Qwen2.5-32B-Instruct",
        help="Hugging Face model id",
    )
    parser.add_argument(
        "--local-dir",
        default="",
        help="Local target directory (default: Hugging Face cache)",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Model revision/tag/branch",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or "",
        help="HF token (optional if public model)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    local_dir = Path(args.local_dir).expanduser() if args.local_dir else None
    if local_dir:
        local_dir.mkdir(parents=True, exist_ok=True)

    print(f"[download] model_id={args.model_id}")
    if local_dir:
        print(f"[download] local_dir={local_dir}")
    else:
        print("[download] local_dir=(HF default cache)")

    path = snapshot_download(
        repo_id=args.model_id,
        repo_type="model",
        revision=args.revision,
        token=args.token or None,
        local_dir=str(local_dir) if local_dir else None,
        local_dir_use_symlinks=False,
    )

    print(f"[done] model snapshot path: {path}")


if __name__ == "__main__":
    main()
