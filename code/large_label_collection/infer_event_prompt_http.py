#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


EVENT_MAX_BOXES = {
    "rule_1_ppe_violation": 13,
    "rule_2_fall_protection_violation": 6,
    "rule_3_unprotected_edge_violation": 3,
    "rule_4_excavator_proximity_violation": 2,
}


def parse_answer(answer: str) -> dict:
    payload = json.loads(answer)
    if not isinstance(payload, dict) or set(payload) != {"events"}:
        raise ValueError("top-level JSON must contain only events")
    events = payload["events"]
    if not isinstance(events, dict) or set(events) != set(EVENT_MAX_BOXES):
        raise ValueError("events must contain exactly the four fixed safety rules")

    parsed = {}
    for event_name, max_boxes in EVENT_MAX_BOXES.items():
        event = events[event_name]
        if not isinstance(event, dict) or set(event) != {"present", "reason", "boxes"}:
            raise ValueError(f"{event_name} must contain only present, reason, and boxes")
        if not isinstance(event["present"], bool):
            raise ValueError(f"{event_name}.present must be boolean")
        if not isinstance(event["boxes"], list) or len(event["boxes"]) > max_boxes:
            raise ValueError(f"{event_name}.boxes exceeds the training-derived limit {max_boxes}")
        boxes = []
        for index, box in enumerate(event["boxes"]):
            if not isinstance(box, list) or len(box) != 4:
                raise ValueError(f"{event_name}.boxes[{index}] must contain four coordinates")
            if any(isinstance(value, bool) or not isinstance(value, int) for value in box):
                raise ValueError(f"{event_name}.boxes[{index}] coordinates must be integers")
            x1, y1, x2, y2 = map(float, box)
            if not (0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000):
                raise ValueError(f"{event_name}.boxes[{index}] is outside ordered [0,1000] xyxy")
            boxes.append([x1 / 1000, y1 / 1000, x2 / 1000, y2 / 1000])
        if event["present"]:
            if not isinstance(event["reason"], str) or not event["reason"].strip() or not boxes:
                raise ValueError(f"{event_name} present=true requires a non-empty reason and at least one box")
            reason = event["reason"].strip()
        else:
            if event["reason"] is not None or boxes:
                raise ValueError(f"{event_name} present=false requires reason=null and boxes=[]")
            reason = None
        parsed[event_name] = {"present": event["present"], "reason": reason, "boxes": boxes}
    return parsed


def output_schema() -> dict:
    box_schema = {
        "type": "array",
        "items": {"type": "integer", "minimum": 0, "maximum": 1000},
        "minItems": 4,
        "maxItems": 4,
    }
    event_properties = {}
    for event_name, max_boxes in EVENT_MAX_BOXES.items():
        event_properties[event_name] = {
            "type": "object",
            "properties": {
                "present": {"type": "boolean"},
                "reason": {"type": ["string", "null"]},
                "boxes": {"type": "array", "items": box_schema, "maxItems": max_boxes},
            },
            "required": ["present", "reason", "boxes"],
            "additionalProperties": False,
        }
    return {
        "type": "object",
        "properties": {
            "events": {
                "type": "object",
                "properties": event_properties,
                "required": list(EVENT_MAX_BOXES),
                "additionalProperties": False,
            }
        },
        "required": ["events"],
        "additionalProperties": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--prompt-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.concurrency < 1:
        raise ValueError("concurrency must be positive")
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line]
    prompt_text = args.prompt_file.read_text(encoding="utf-8").strip()
    if not rows or not prompt_text:
        raise ValueError("input and prompt must be non-empty")

    schema = output_schema()
    endpoint = args.base_url.rstrip("/") + "/v1/chat/completions"

    def run_one(row: dict) -> dict:
        if set(row) != {"sample_id", "split", "image_path"}:
            raise ValueError(f"invalid input fields: {row}")
        image_path = Path(row["image_path"])
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        payload = {
            "model": args.model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_path.as_uri()}},
                    {"type": "text", "text": prompt_text},
                ],
            }],
            "temperature": 0,
            "max_tokens": args.max_new_tokens,
            "structured_outputs": {"json": schema},
        }
        encoded = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(endpoint, data=encoded, headers={"Content-Type": "application/json"})
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
        latency = time.perf_counter() - started

        model_output = {
            "model": args.model,
            "runtime": "vLLM-0.18.0-OpenAI-compatible-server",
            "prompt_id": args.prompt_id,
            "raw_answer": raw_answer,
            "latency_seconds": latency,
            "finish_reason": finish_reason,
            "protocol": "one image, one model call, all four safety rules in one prompt",
            "model_requests_for_image": 1,
        }
        try:
            if request_error is not None:
                raise ValueError(f"HTTP request failed: {request_error}")
            model_output["events"] = parse_answer(raw_answer)
            model_output["parse_status"] = "valid"
        except (json.JSONDecodeError, ValueError) as error:
            model_output["events"] = None
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

