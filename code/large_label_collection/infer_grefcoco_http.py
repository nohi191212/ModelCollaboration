#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def parse_answer(answer: str, width: int, height: int, max_objects: int) -> list[dict]:
    payload = json.loads(answer)
    if not isinstance(payload, dict) or set(payload) != {"objects"}:
        raise ValueError("top-level JSON must contain only objects")
    if not isinstance(payload["objects"], list) or len(payload["objects"]) > max_objects:
        raise ValueError(f"objects must be a list with at most {max_objects} entries")
    parsed = []
    for index, item in enumerate(payload["objects"]):
        if not isinstance(item, dict) or set(item) != {"bbox_2d", "score"}:
            raise ValueError(f"object {index} must contain only bbox_2d and score")
        box = item["bbox_2d"]
        if not isinstance(box, list) or len(box) != 4:
            raise ValueError(f"object {index} bbox_2d must contain four coordinates")
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in box):
            raise ValueError(f"object {index} coordinates must be numbers")
        x1, y1, x2, y2 = map(float, box)
        if not (0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000):
            raise ValueError(f"object {index} coordinates are outside ordered [0,1000] xyxy")
        score = item["score"]
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not 0 <= float(score) <= 1:
            raise ValueError(f"object {index} score is not a number in [0,1]")
        parsed.append({
            "bbox_2d_normalized": [x1, y1, x2, y2],
            "bbox_xyxy_absolute": [
                x1 * width / 1000,
                y1 * height / 1000,
                x2 * width / 1000,
                y2 * height / 1000,
            ],
            "score": float(score),
        })
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--prompt-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--max-objects", type=int, default=18)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line]
    prompt_template = args.prompt_file.read_text(encoding="utf-8").strip()
    if not rows or prompt_template.count("{expression}") != 1:
        raise ValueError("input must be non-empty and prompt must contain {expression} exactly once")
    schema = {
        "type": "object",
        "properties": {
            "objects": {
                "type": "array",
                "maxItems": args.max_objects,
                "items": {
                    "type": "object",
                    "properties": {
                        "bbox_2d": {
                            "type": "array",
                            "items": {"type": "number", "minimum": 0, "maximum": 1000},
                            "minItems": 4,
                            "maxItems": 4,
                        },
                        "score": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                    "required": ["bbox_2d", "score"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["objects"],
        "additionalProperties": False,
    }
    endpoint = args.base_url.rstrip("/") + "/v1/chat/completions"

    def run_one(row: dict) -> dict:
        required = {"sample_id", "image_path", "width", "height", "expression"}
        missing = required - row.keys()
        if missing:
            raise ValueError(f"missing fields {sorted(missing)} in row: {row}")
        image_path = Path(row["image_path"])
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        prompt = prompt_template.replace("{expression}", row["expression"])
        payload = {
            "model": args.model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_path.as_uri()}},
                    {"type": "text", "text": prompt},
                ],
            }],
            "temperature": 0,
            "max_tokens": args.max_new_tokens,
            "structured_outputs": {"json": schema},
        }
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        started = time.perf_counter()
        raw_answer = ""
        finish_reason = None
        request_error = None
        try:
            with urllib.request.urlopen(request, timeout=900) as response:
                response_body = response.read().decode("utf-8")
            response_payload = json.loads(response_body)
            choice = response_payload["choices"][0]
            raw_answer = choice["message"]["content"].strip()
            finish_reason = choice.get("finish_reason")
        except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, KeyError, IndexError, TypeError, ValueError) as error:
            request_error = str(error)
            if isinstance(error, urllib.error.HTTPError):
                raw_answer = error.read().decode("utf-8", errors="replace")

        model_output = {
            "model": args.model,
            "runtime": "vLLM-0.18.0-OpenAI-compatible-server",
            "prompt_id": args.prompt_id,
            "raw_answer": raw_answer,
            "latency_seconds": time.perf_counter() - started,
            "finish_reason": finish_reason,
            "protocol": "one image, one model call, zero/one/multiple targets in one prompt",
            "model_requests_for_image": 1,
        }
        try:
            if request_error is not None:
                raise ValueError(f"HTTP request failed: {request_error}")
            model_output["objects"] = parse_answer(raw_answer, int(row["width"]), int(row["height"]), args.max_objects)
            model_output["parse_status"] = "valid"
        except (json.JSONDecodeError, ValueError) as error:
            model_output["objects"] = []
            model_output["parse_status"] = "invalid_output_counted_wrong"
            model_output["parse_error"] = str(error)
        output = dict(row)
        output["generalist_output"] = model_output
        return output

    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = 0
    with args.output.open("x", encoding="utf-8") as destination:
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            for batch_start in range(0, len(rows), args.concurrency):
                batch = rows[batch_start:batch_start + args.concurrency]
                outputs = list(executor.map(run_one, batch))
                for output in outputs:
                    destination.write(json.dumps(output, ensure_ascii=False) + "\n")
                destination.flush()
                completed += len(outputs)
                valid = sum(output["generalist_output"]["parse_status"] == "valid" for output in outputs)
                print(json.dumps({"prompt_id": args.prompt_id, "completed": completed, "total": len(rows), "batch_valid": valid}), flush=True)


if __name__ == "__main__":
    main()

