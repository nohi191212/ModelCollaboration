#!/usr/bin/env python3
"""Cache closed-form Qwen3.5 image answers through an OpenAI-compatible API."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import urllib.request
from pathlib import Path


def data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="JSONL with image_path and prompt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default=os.getenv("QWEN_BASE_URL", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--model", default=os.getenv("QWEN_MODEL", "Qwen3.5-9B"))
    parser.add_argument("--max-tokens", type=int, default=32)
    args = parser.parse_args()
    endpoint = args.base_url.rstrip("/") + "/chat/completions"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.input.open(encoding="utf-8") as src, args.output.open("w", encoding="utf-8") as dst:
        for line_no, line in enumerate(src, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            image_path = Path(row["image_path"])
            body = {
                "model": args.model,
                "temperature": 0,
                "max_tokens": args.max_tokens,
                "messages": [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": data_url(image_path)}},
                    {"type": "text", "text": row["prompt"]},
                ]}],
                "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
            }
            request = urllib.request.Request(
                endpoint,
                data=json.dumps(body).encode("utf-8"),
                headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
            )
            with urllib.request.urlopen(request, timeout=300) as response:
                payload = json.load(response)
            out = dict(row)
            out.update({"generalist_model": args.model, "generalist_response": payload})
            dst.write(json.dumps(out, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
