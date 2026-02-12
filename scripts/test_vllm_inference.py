#!/usr/bin/env python3
"""
Simple vLLM OpenAI-compatible inference smoke test.

Usage:
    python scripts/test_vllm_inference.py --base-url http://localhost:8000
    python scripts/test_vllm_inference.py --base-url http://localhost:8000 --model Qwen/Qwen2.5-32B-Instruct
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request


def _http_json(url: str, payload: dict | None = None, timeout: int = 60) -> dict:
    if payload is None:
        req = urllib.request.Request(url=url, method="GET")
    else:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url=url,
            data=data,
            method="POST",
            headers={"Content-Type": "application/json"},
        )

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="vLLM chat completion smoke test")
    parser.add_argument("--base-url", default="http://localhost:8000", help="vLLM server url")
    parser.add_argument("--model", default="", help="model id to use (default: first loaded model)")
    parser.add_argument(
        "--prompt",
        default="Summarize in one sentence what image forensics means.",
        help="test prompt",
    )
    parser.add_argument("--max-tokens", type=int, default=128, help="max output tokens")
    parser.add_argument("--temperature", type=float, default=0.2, help="sampling temperature")
    parser.add_argument("--timeout", type=int, default=120, help="request timeout seconds")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base = args.base_url.rstrip("/")

    try:
        model_resp = _http_json(f"{base}/v1/models", timeout=args.timeout)
    except urllib.error.URLError as exc:
        print(f"[error] cannot connect to vLLM: {exc}", file=sys.stderr)
        return 1

    models = model_resp.get("data", [])
    if not models:
        print("[error] no loaded models returned by /v1/models", file=sys.stderr)
        return 1

    model_id = args.model or models[0].get("id", "")
    if not model_id:
        print("[error] invalid model list response", file=sys.stderr)
        return 1

    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": args.prompt},
        ],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
    }

    try:
        result = _http_json(f"{base}/v1/chat/completions", payload=payload, timeout=args.timeout)
    except urllib.error.URLError as exc:
        print(f"[error] inference request failed: {exc}", file=sys.stderr)
        return 1

    choices = result.get("choices", [])
    if not choices:
        print("[error] no choices in completion response", file=sys.stderr)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 1

    text = choices[0].get("message", {}).get("content", "")
    print(f"[ok] model={model_id}")
    print("--- response ---")
    print(text.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
