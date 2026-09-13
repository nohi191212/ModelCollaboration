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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--min-image-pixels", type=int, required=True)
    parser.add_argument("--max-image-pixels", type=int, required=True)
    parser.add_argument("--resize-factor", type=int, required=True)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if args.start_index < 0 or args.start_index > len(rows):
        raise ValueError(f"invalid start index: {args.start_index}")
    rows = rows[args.start_index :]
    if args.limit is not None:
        rows = rows[:args.limit]
    if not rows:
        raise ValueError("input is empty")

    processor = AutoProcessor.from_pretrained(args.model)
    model = LLM(
        model=str(args.model),
        trust_remote_code=True,
        tensor_parallel_size=1,
        max_model_len=4096,
        limit_mm_per_prompt={"image": 2},
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
            required = {"sample_id", "image1", "image2", "sentence"}
            missing = required - row.keys()
            if missing:
                raise ValueError(f"missing fields {sorted(missing)} in row: {row}")

            messages = [{
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "The first image is the left image."},
                    {"type": "image"},
                    {
                        "type": "text",
                        "text": (
                            "The second image is the right image. Decide whether the sentence "
                            "is true about this pair of images. Reply with exactly True or False "
                            f"and no other text. Sentence: {row['sentence']}"
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
            left_image = Image.open(row["image1"]).convert("RGB")
            right_image = Image.open(row["image2"]).convert("RGB")
            left_original_size = left_image.size
            right_original_size = right_image.size
            left_height, left_width = smart_resize(
                left_image.height,
                left_image.width,
                factor=args.resize_factor,
                min_pixels=args.min_image_pixels,
                max_pixels=args.max_image_pixels,
            )
            right_height, right_width = smart_resize(
                right_image.height,
                right_image.width,
                factor=args.resize_factor,
                min_pixels=args.min_image_pixels,
                max_pixels=args.max_image_pixels,
            )
            if left_image.size != (left_width, left_height):
                resized = left_image.resize((left_width, left_height), Image.Resampling.BICUBIC)
                left_image.close()
                left_image = resized
            if right_image.size != (right_width, right_height):
                resized = right_image.resize((right_width, right_height), Image.Resampling.BICUBIC)
                right_image.close()
                right_image = resized
            started = time.perf_counter()
            generated = model.generate(
                {
                    "prompt": prompt,
                    "multi_modal_data": {"image": [left_image, right_image]},
                },
                sampling,
                use_tqdm=False,
            )
            latency = time.perf_counter() - started
            left_image.close()
            right_image.close()

            answer = generated[0].outputs[0].text.strip()
            output = dict(row)
            output["generalist_output"] = {
                "model": "Qwen3.8-27B-FP8",
                "raw_answer": answer,
                "latency_seconds": latency,
                "parse_status": "valid" if answer in {"True", "False"} else "invalid",
                "image_preprocessing": {
                    "method": "Qwen smart_resize followed by bicubic resize",
                    "min_pixels": args.min_image_pixels,
                    "max_pixels": args.max_image_pixels,
                    "resize_factor": args.resize_factor,
                    "image1_original_size": list(left_original_size),
                    "image1_input_size": [left_width, left_height],
                    "image2_original_size": list(right_original_size),
                    "image2_input_size": [right_width, right_height],
                },
            }
            destination.write(json.dumps(output, ensure_ascii=False) + "\n")
            destination.flush()
            if answer not in {"True", "False"}:
                raise ValueError(
                    f"sample {row['sample_id']} returned non-strict answer: {answer!r}"
                )
            print(json.dumps({"completed": index, "total": len(rows)}), flush=True)


if __name__ == "__main__":
    main()
