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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--classes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--min-image-pixels", type=int, required=True)
    parser.add_argument("--max-image-pixels", type=int, required=True)
    parser.add_argument("--resize-factor", type=int, required=True)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line]
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("input is empty")
    classes = json.loads(args.classes.read_text(encoding="utf-8"))
    if [row["category_id"] for row in classes] != list(range(1, 201)):
        raise ValueError("CUB class map is not exactly 1 through 200")
    class_by_id = {row["category_id"]: row for row in classes}
    class_lines = "\n".join(f"{row['category_id']}: {row['prompt_name']}" for row in classes)

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
    sampling = SamplingParams(temperature=0, max_tokens=8)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as destination:
        for index, row in enumerate(rows, 1):
            required = {"sample_id", "image_path", "width", "height"}
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
                            "Identify the bird species in this image using the fixed CUB-200-2011 class list below. "
                            "Reply with exactly one class number from 1 to 200 and no other text.\n"
                            f"{class_lines}"
                        ),
                    },
                ],
            }]
            prompt = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
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
            generated = model.generate(
                {"prompt": prompt, "multi_modal_data": {"image": image}},
                sampling,
                use_tqdm=False,
            )
            latency = time.perf_counter() - started
            image.close()

            answer = generated[0].outputs[0].text.strip()
            valid = answer.isdigit() and answer == str(int(answer)) and int(answer) in class_by_id
            output = dict(row)
            model_output = {
                "model": "Qwen3.8-27B-FP8",
                "raw_answer": answer,
                "parse_status": "valid" if valid else "invalid",
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
            if valid:
                model_output["predicted_category_id"] = int(answer)
                model_output["predicted_category_name"] = class_by_id[int(answer)]["official_name"]
            output["generalist_output"] = model_output
            destination.write(json.dumps(output, ensure_ascii=False) + "\n")
            destination.flush()
            if not valid:
                raise ValueError(f"sample {row['sample_id']} returned invalid class number: {answer!r}")
            print(json.dumps({"completed": index, "total": len(rows)}), flush=True)


if __name__ == "__main__":
    main()

