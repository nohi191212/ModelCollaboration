#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoTokenizer
from transformers.dynamic_module_utils import get_class_from_dynamic_module


def parse_answer(answer: str, width: int, height: int, max_objects: int) -> list[dict]:
    payload = json.loads(answer)
    if not isinstance(payload, dict) or set(payload) != {"objects"}:
        raise ValueError("top-level JSON must contain only objects")
    if not isinstance(payload["objects"], list):
        raise ValueError("objects must be a list")
    if len(payload["objects"]) > max_objects:
        raise ValueError(f"objects exceeds the training-set maximum of {max_objects}")
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
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ValueError(f"object {index} score must be numeric")
        score = float(score)
        if not 0 <= score <= 1:
            raise ValueError(f"object {index} score is outside [0,1]")
        parsed.append({
            "bbox_2d_normalized": [x1, y1, x2, y2],
            "bbox_xyxy_absolute": [
                x1 * width / 1000,
                y1 * height / 1000,
                x2 * width / 1000,
                y2 * height / 1000,
            ],
            "score": score,
        })
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--max-new-tokens", type=int, required=True)
    parser.add_argument("--max-objects", type=int, required=True)
    parser.add_argument("--attn-implementation", required=True)
    args = parser.parse_args()

    all_rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line]
    end_index = len(all_rows) if args.end_index is None else args.end_index
    if not (0 <= args.start_index < end_index <= len(all_rows)):
        raise ValueError(f"invalid half-open range [{args.start_index}, {end_index}) for {len(all_rows)} rows")
    if args.batch_size < 1 or args.max_new_tokens < 1 or args.max_objects < 1:
        raise ValueError("batch size, maximum output tokens, and maximum objects must be positive")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    rows = all_rows[args.start_index:end_index]

    model_class = get_class_from_dynamic_module("modeling_minicpmv.MiniCPMV", str(args.model))
    model_class.all_tied_weights_keys = {}
    model = model_class.from_pretrained(
        args.model,
        trust_remote_code=True,
        attn_implementation=args.attn_implementation,
        torch_dtype=torch.bfloat16,
    ).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    prompt_prefix = (
        "Locate every visible object that matches the referring expression. The correct answer may contain zero, one, or multiple objects. "
        f"Return no more than {args.max_objects} objects, because {args.max_objects} is the largest target count observed in the training split. "
        "Coordinates must be xyxy values normalized to numbers from 0 to 1000. Give each box a confidence score from 0 to 1. "
        "Reply with exactly one JSON object and no Markdown or explanation. Use this exact schema: "
        "{\"objects\":[{\"bbox_2d\":[x1,y1,x2,y2],\"score\":0.87}]}. "
        "If no matching object is visible, reply exactly {\"objects\":[]}. Referring expression: "
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as destination:
        for batch_start in range(0, len(rows), args.batch_size):
            batch_rows = rows[batch_start:batch_start + args.batch_size]
            images = []
            messages = []
            for row in batch_rows:
                required = {"sample_id", "image_path", "width", "height", "expression"}
                missing = required - row.keys()
                if missing:
                    raise ValueError(f"missing fields {sorted(missing)} in row: {row}")
                image = Image.open(row["image_path"]).convert("RGB")
                images.append(image)
                messages.append([{
                    "role": "user",
                    "content": [image, prompt_prefix + row["expression"]],
                }])

            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.inference_mode():
                answers = model.chat(
                    image=None,
                    msgs=messages,
                    tokenizer=tokenizer,
                    enable_thinking=False,
                    sampling=False,
                    stream=False,
                    max_new_tokens=args.max_new_tokens,
                    max_slice_nums=9,
                    use_image_id=True,
                    num_beams=1,
                    repetition_penalty=1.0,
                )
            torch.cuda.synchronize()
            batch_latency = time.perf_counter() - started
            for image in images:
                image.close()
            if not isinstance(answers, list) or len(answers) != len(batch_rows):
                raise ValueError(f"model returned {type(answers).__name__} with unexpected batch length")

            for row, answer in zip(batch_rows, answers):
                answer = answer.strip()
                output = dict(row)
                model_output = {
                    "model": "MiniCPM-V-4_5",
                    "raw_answer": answer,
                    "latency_seconds": batch_latency / len(batch_rows),
                    "batch_latency_seconds": batch_latency,
                    "batch_size": len(batch_rows),
                    "protocol": "one_image_one_model_request",
                    "model_requests_for_image": 1,
                    "max_objects_from_training_split": args.max_objects,
                }
                try:
                    model_output["objects"] = parse_answer(
                        answer,
                        int(row["width"]),
                        int(row["height"]),
                        args.max_objects,
                    )
                    model_output["parse_status"] = "valid"
                except (json.JSONDecodeError, ValueError) as error:
                    model_output["objects"] = []
                    model_output["parse_status"] = "invalid_output_counted_wrong"
                    model_output["failure_kind"] = "unparseable_or_illegal_model_output"
                    model_output["parse_error"] = str(error)
                output["generalist_output"] = model_output
                destination.write(json.dumps(output, ensure_ascii=False) + "\n")
            destination.flush()
            completed = args.start_index + batch_start + len(batch_rows)
            print(json.dumps({
                "completed_global_exclusive": completed,
                "end_index_exclusive": end_index,
                "batch_size": len(batch_rows),
                "batch_latency_seconds": batch_latency,
            }), flush=True)


if __name__ == "__main__":
    main()
