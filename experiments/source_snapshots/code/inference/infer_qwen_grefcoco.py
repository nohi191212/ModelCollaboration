#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from PIL import Image
from transformers import AutoProcessor
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams


def parse_answer(answer: str, width: int, height: int) -> list[dict]:
    payload = json.loads(answer)
    if not isinstance(payload, dict) or set(payload) != {"objects"}:
        raise ValueError("top-level JSON must contain only objects")
    if not isinstance(payload["objects"], list):
        raise ValueError("objects must be a list")
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
    parser.add_argument("--min-image-pixels", type=int, required=True)
    parser.add_argument("--max-image-pixels", type=int, required=True)
    parser.add_argument("--resize-factor", type=int, required=True)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise ValueError("input is empty")
    processor = AutoProcessor.from_pretrained(args.model)
    model = LLM(
        model=str(args.model),
        trust_remote_code=True,
        tensor_parallel_size=1,
        max_model_len=4096,
        limit_mm_per_prompt={"image": 1},
        mm_processor_kwargs={
            "size": {
                "shortest_edge": args.min_image_pixels,
                "longest_edge": args.max_image_pixels,
            }
        },
    )
    output_schema = {
        "type": "object",
        "properties": {
            "objects": {
                "type": "array",
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
    sampling = SamplingParams(
        temperature=0,
        max_tokens=512,
        structured_outputs=StructuredOutputsParams(
            json=output_schema,
            disable_additional_properties=True,
        ),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as destination:
        for index, row in enumerate(rows, 1):
            required = {"sample_id", "image_path", "width", "height", "expression"}
            missing = required - row.keys()
            if missing:
                raise ValueError(f"missing fields {sorted(missing)} in row: {row}")
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image"},
                    {
                        "type": "text",
                        "text": (
                            "Locate every visible object that matches this referring expression. The correct answer may contain zero, one, or multiple objects. "
                            "Coordinates must be xyxy values normalized to integers from 0 to 1000. Give each box a confidence score from 0 to 1. "
                            "Reply with exactly one JSON object and no Markdown or explanation. Use this exact schema: "
                            "{\"objects\":[{\"bbox_2d\":[x1,y1,x2,y2],\"score\":0.87}]}. "
                            "If no matching object is visible, reply exactly {\"objects\":[]}. "
                            f"Referring expression: {row['expression']}"
                        ),
                    },
                ],
            }]
            prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            image = Image.open(row["image_path"]).convert("RGB")
            original_size = image.size
            resized_height, resized_width = smart_resize(
                image.height,
                image.width,
                factor=args.resize_factor,
                min_pixels=args.min_image_pixels,
                max_pixels=args.max_image_pixels,
            )
            if image.size != (resized_width, resized_height):
                resized = image.resize((resized_width, resized_height), Image.Resampling.BICUBIC)
                image.close()
                image = resized
            started = time.perf_counter()
            generated = model.generate({"prompt": prompt, "multi_modal_data": {"image": image}}, sampling, use_tqdm=False)
            latency = time.perf_counter() - started
            image.close()
            answer = generated[0].outputs[0].text.strip()
            output = dict(row)
            model_output = {
                "model": "Qwen3.8-27B-FP8",
                "raw_answer": answer,
                "latency_seconds": latency,
                "image_preprocessing": {
                    "method": "Qwen smart_resize followed by bicubic resize",
                    "min_pixels": args.min_image_pixels,
                    "max_pixels": args.max_image_pixels,
                    "resize_factor": args.resize_factor,
                    "original_size": list(original_size),
                    "input_size": [resized_width, resized_height],
                },
            }
            try:
                model_output["objects"] = parse_answer(answer, int(row["width"]), int(row["height"]))
                model_output["parse_status"] = "valid"
            except (json.JSONDecodeError, ValueError) as error:
                model_output["parse_status"] = "invalid"
                model_output["parse_error"] = str(error)
                output["generalist_output"] = model_output
                destination.write(json.dumps(output, ensure_ascii=False) + "\n")
                destination.flush()
                raise ValueError(f"sample {row['sample_id']} returned invalid grounding output: {answer!r}") from error
            output["generalist_output"] = model_output
            destination.write(json.dumps(output, ensure_ascii=False) + "\n")
            destination.flush()
            print(json.dumps({"completed": index, "total": len(rows), "objects": len(model_output["objects"])}), flush=True)


if __name__ == "__main__":
    main()
