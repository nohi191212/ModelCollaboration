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


ATTRIBUTES = {
    "illumination": {"night", "normal lighting", "overexposed", "underexposed"},
    "camera_distance": {"long distance", "mid distance", "short distance"},
    "view": {"elevation view", "plan view"},
    "quality_of_info": {"poor info", "rich info"},
}
RAW_TOP_LEVEL_KEYS = {
    "image_caption",
    "illumination",
    "camera_distance",
    "view",
    "quality_of_info",
    "rules",
    "excavator",
    "rebar",
    "worker_with_white_hard_hat",
}


def parse_boxes(boxes: object, field_name: str) -> list[list[float]]:
    if not isinstance(boxes, list):
        raise ValueError(f"{field_name} must be a list")
    parsed = []
    for index, box in enumerate(boxes):
        if not isinstance(box, list) or len(box) != 4:
            raise ValueError(f"{field_name}[{index}] must contain four coordinates")
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in box):
            raise ValueError(f"{field_name}[{index}] coordinates must be numbers")
        x1, y1, x2, y2 = map(float, box)
        if not (0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000):
            raise ValueError(f"{field_name}[{index}] is outside ordered [0,1000] xyxy")
        parsed.append([x1 / 1000, y1 / 1000, x2 / 1000, y2 / 1000])
    return parsed


def parse_answer(answer: str) -> dict:
    payload = json.loads(answer)
    if not isinstance(payload, dict) or set(payload) != RAW_TOP_LEVEL_KEYS:
        raise ValueError("top-level JSON fields do not match the fixed schema")
    if not isinstance(payload["image_caption"], str) or not payload["image_caption"].strip():
        raise ValueError("image_caption must be a non-empty string")
    parsed = {"image_caption": payload["image_caption"].strip()}
    for field_name, choices in ATTRIBUTES.items():
        if payload[field_name] not in choices:
            raise ValueError(f"{field_name} is outside the fixed training vocabulary")
        parsed[field_name] = payload[field_name]
    rules = payload["rules"]
    expected_rule_keys = {f"rule_{rule_id}" for rule_id in range(1, 5)}
    if not isinstance(rules, dict) or set(rules) != expected_rule_keys:
        raise ValueError("rules must contain exactly rule_1, rule_2, rule_3, and rule_4")
    parsed_violations = []
    for rule_id in range(1, 5):
        rule_key = f"rule_{rule_id}"
        rule = rules[rule_key]
        if not isinstance(rule, dict) or set(rule) != {"violation", "reason", "bounding_boxes"}:
            raise ValueError(f"{rule_key} fields do not match the fixed schema")
        if not isinstance(rule["violation"], bool):
            raise ValueError(f"{rule_key}.violation must be boolean")
        boxes = parse_boxes(rule["bounding_boxes"], f"{rule_key}.bounding_boxes")
        if rule["violation"]:
            if not isinstance(rule["reason"], str) or not rule["reason"].strip():
                raise ValueError(f"{rule_key}.reason must be non-empty when violation is true")
            if not boxes:
                raise ValueError(f"{rule_key} must include at least one box when violation is true")
            parsed_violations.append({"rule_id": rule_id, "reason": rule["reason"].strip(), "bounding_boxes": boxes})
        elif rule["reason"] is not None or boxes:
            raise ValueError(f"{rule_key} must use null reason and empty boxes when violation is false")
    parsed["rule_violations"] = parsed_violations
    for object_name in ["excavator", "rebar", "worker_with_white_hard_hat"]:
        parsed[object_name] = parse_boxes(payload[object_name], object_name)
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
        max_model_len=8192,
        limit_mm_per_prompt={"image": 1},
        mm_processor_kwargs={"size": {"shortest_edge": args.min_image_pixels, "longest_edge": args.max_image_pixels}},
    )
    box_schema = {
        "type": "array",
        "items": {"type": "number", "minimum": 0, "maximum": 1000},
        "minItems": 4,
        "maxItems": 4,
    }
    boxes_schema = {"type": "array", "items": box_schema}
    output_schema = {
        "type": "object",
        "properties": {
            "image_caption": {"type": "string", "minLength": 1},
            "illumination": {"type": "string", "enum": sorted(ATTRIBUTES["illumination"])},
            "camera_distance": {"type": "string", "enum": sorted(ATTRIBUTES["camera_distance"])},
            "view": {"type": "string", "enum": sorted(ATTRIBUTES["view"])},
            "quality_of_info": {"type": "string", "enum": sorted(ATTRIBUTES["quality_of_info"])},
            "rules": {
                "type": "object",
                "properties": {
                    f"rule_{rule_id}": {
                    "type": "object",
                    "properties": {
                            "violation": {"type": "boolean"},
                            "reason": {"type": ["string", "null"]},
                            "bounding_boxes": {"type": "array", "items": box_schema},
                        },
                        "required": ["violation", "reason", "bounding_boxes"],
                        "additionalProperties": False,
                    }
                    for rule_id in range(1, 5)
                    },
                "required": [f"rule_{rule_id}" for rule_id in range(1, 5)],
                "additionalProperties": False,
            },
            "excavator": boxes_schema,
            "rebar": boxes_schema,
            "worker_with_white_hard_hat": boxes_schema,
        },
        "required": sorted(RAW_TOP_LEVEL_KEYS),
        "additionalProperties": False,
    }
    sampling = SamplingParams(
        temperature=0,
        max_tokens=1536,
        structured_outputs=StructuredOutputsParams(json=output_schema, disable_additional_properties=True),
    )

    prompt_text = (
        "Inspect this construction-site image and complete every field in the required JSON. "
        "Write a detailed factual caption. Classify illumination, camera distance, view, and information quality using only the allowed strings in the schema. "
        "For each of the four fixed rule slots, report whether that rule is violated, with a short reason and one or more image-relative xyxy boxes using integer coordinates from 0 to 1000. "
        "Rule 1: people on foot must use basic personal protective equipment, including hard hats, proper clothing and footwear, night visibility equipment, and task-specific eye or face protection. "
        "Rule 2: use a safety harness when working at least three meters high without edge protection. "
        "Rule 3: use guardrails, fences, or warnings at unprotected edges or steep excavations at least three meters deep where people can stand. "
        "Rule 4: no worker may be in an operating excavator's blind spot or operating radius. "
        "Match the training annotation convention exactly when drawing rule boxes. "
        "For Rule 1, draw one tight box around the visible extent of each violating worker, not around the missing helmet, vest, shoes, or protective item. "
        "For Rule 2, draw one tight box around the visible extent of each unprotected worker at height, not around the scaffold, roof, ladder, or edge. "
        "For Rule 3, box each continuous unprotected excavation edge or hazardous edge region itself, including the visible edge and the immediately adjacent drop or excavation area needed to identify that hazard; do not box workers, and do not box the whole excavation or whole image when a tighter hazard-region box is possible. Use separate boxes for disconnected unsafe edge regions. "
        "For Rule 4, draw one tight box around the visible extent of each worker inside an excavator blind spot or operating radius; do not box the excavator and do not draw one union box around the worker and machine. "
        "If the same worker violates more than one rule, repeat that worker's box independently in every applicable rule slot. Use one box per violating worker or continuous Rule 3 hazard region, and include only visible extents rather than guessing hidden body parts. "
        "The rules object must contain rule_1, rule_2, rule_3, and rule_4 exactly once. For a visible violation use violation true, a non-empty reason, and at least one box. Otherwise use violation false, reason null, and an empty box list. "
        "Also return every visible excavator, exposed rebar group, and worker wearing a white hard hat as normalized xyxy boxes. Use an empty list when an object type is absent. "
        "Return exactly one JSON object with no Markdown or extra explanation."
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as destination:
        for index, row in enumerate(rows, 1):
            required = {"sample_id", "image_path", "width", "height"}
            missing = required - row.keys()
            if missing:
                raise ValueError(f"missing input fields {sorted(missing)} for {row}")
            messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]}]
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
                model_output["parsed"] = parse_answer(answer)
                model_output["parse_status"] = "valid"
            except (json.JSONDecodeError, ValueError) as error:
                model_output["parse_status"] = "invalid"
                model_output["parse_error"] = str(error)
                model_output["parsed"] = None
                output["generalist_output"] = model_output
                destination.write(json.dumps(output, ensure_ascii=False) + "\n")
                destination.flush()
                print(json.dumps({"completed": index, "total": len(rows), "status": "invalid"}), flush=True)
                continue
            output["generalist_output"] = model_output
            destination.write(json.dumps(output, ensure_ascii=False) + "\n")
            destination.flush()
            print(json.dumps({"completed": index, "total": len(rows)}), flush=True)


if __name__ == "__main__":
    main()
