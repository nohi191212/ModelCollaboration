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
    args = parser.parse_args()

    model_class = get_class_from_dynamic_module("modeling_minicpmv.MiniCPMV", str(args.model))
    model_class.all_tied_weights_keys = {}
    model = model_class.from_pretrained(
        args.model,
        trust_remote_code=True,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
    ).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise ValueError("input is empty")

    prompt_text = (
        "Inspect this construction-site image and return exactly one JSON object with no Markdown or extra text. "
        "The top-level fields must be image_caption, illumination, camera_distance, view, quality_of_info, rules, excavator, rebar, and worker_with_white_hard_hat. "
        "Use only these attribute strings: illumination is night, normal lighting, overexposed, or underexposed; camera_distance is long distance, mid distance, or short distance; view is elevation view or plan view; quality_of_info is poor info or rich info. "
        "The rules object must contain rule_1, rule_2, rule_3, and rule_4 exactly once. Every rule value must contain violation, reason, and bounding_boxes. For a visible violation use true, a short non-empty reason, and at least one image-relative xyxy box using integer coordinates from 0 to 1000. Otherwise use false, null, and an empty list. "
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
        "Also return every visible excavator, exposed rebar group, and worker wearing a white hard hat as image-relative xyxy boxes using integer coordinates from 0 to 1000. Use an empty list when an object type is absent."
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as destination:
        for index, row in enumerate(rows, start=1):
            required = {"sample_id", "image_path", "width", "height"}
            missing = required - row.keys()
            if missing:
                raise ValueError(f"missing input fields {sorted(missing)} for {row}")
            image = Image.open(row["image_path"]).convert("RGB")
            messages = [{"role": "user", "content": [image, prompt_text]}]
            torch.cuda.synchronize()
            started = time.perf_counter()
            answer = model.chat(
                msgs=messages,
                tokenizer=tokenizer,
                enable_thinking=False,
                sampling=False,
                stream=False,
                max_new_tokens=1536,
            )
            torch.cuda.synchronize()
            latency = time.perf_counter() - started
            image.close()
            output = dict(row)
            model_output = {
                "model": "MiniCPM-V-4_5",
                "raw_answer": answer,
                "latency_seconds": latency,
            }
            try:
                model_output["parsed"] = parse_answer(answer.strip())
                model_output["parse_status"] = "valid"
            except (json.JSONDecodeError, ValueError) as error:
                model_output["parsed"] = None
                model_output["parse_status"] = "invalid"
                model_output["parse_error"] = str(error)
            output["generalist_output"] = model_output
            destination.write(json.dumps(output, ensure_ascii=False) + "\n")
            destination.flush()
            print(json.dumps({
                "model": "MiniCPM-V-4_5",
                "completed": index,
                "total": len(rows),
                "status": model_output["parse_status"],
            }), flush=True)


if __name__ == "__main__":
    main()
