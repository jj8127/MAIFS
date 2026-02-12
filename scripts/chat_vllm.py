#!/usr/bin/env python3
"""
Interactive chat client for a vLLM OpenAI-compatible server.

Usage:
  source .venv-qwen/bin/activate
  python scripts/chat_vllm.py --base-url http://localhost:8000
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request


def _http_json(url: str, payload: dict | None = None, timeout: int = 180) -> dict:
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
    parser = argparse.ArgumentParser(description="Interactive vLLM chat")
    parser.add_argument("--base-url", default="http://localhost:8000", help="vLLM server URL")
    parser.add_argument("--model", default="", help="Model id (default: first model from /v1/models)")
    parser.add_argument("--system", default="You are a helpful assistant.", help="System prompt")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max output tokens")
    parser.add_argument("--timeout", type=int, default=180, help="Request timeout (sec)")
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
        print("[error] no models available from /v1/models", file=sys.stderr)
        return 1

    model = args.model or models[0].get("id", "")
    if not model:
        print("[error] failed to resolve model id", file=sys.stderr)
        return 1

    print(f"[connected] model={model}")
    print("Type '/exit' to quit, '/reset' to clear history.")

    messages: list[dict[str, str]] = [{"role": "system", "content": args.system}]

    while True:
        try:
            user_input = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[bye]")
            return 0

        if not user_input:
            continue
        if user_input in {"/exit", "/quit"}:
            print("[bye]")
            return 0
        if user_input == "/reset":
            messages = [{"role": "system", "content": args.system}]
            print("[history reset]")
            continue

        messages.append({"role": "user", "content": user_input})
        payload = {
            "model": model,
            "messages": messages,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
        }

        try:
            resp = _http_json(f"{base}/v1/chat/completions", payload=payload, timeout=args.timeout)
        except urllib.error.URLError as exc:
            print(f"[error] request failed: {exc}")
            continue

        choices = resp.get("choices", [])
        if not choices:
            print("[error] empty response")
            continue

        text = choices[0].get("message", {}).get("content", "").strip()
        print(f"assistant> {text}")
        messages.append({"role": "assistant", "content": text})


if __name__ == "__main__":
    raise SystemExit(main())
