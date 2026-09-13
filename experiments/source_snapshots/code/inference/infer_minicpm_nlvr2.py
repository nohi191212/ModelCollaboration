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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--attn-implementation", required=True)
    args = parser.parse_args()

    all_rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line]
    end_index = len(all_rows) if args.end_index is None else args.end_index
    if not (0 <= args.start_index < end_index <= len(all_rows)):
        raise ValueError(f"invalid half-open range [{args.start_index}, {end_index}) for {len(all_rows)} rows")
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
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

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as destination:
        for batch_start in range(0, len(rows), args.batch_size):
            batch_rows = rows[batch_start:batch_start + args.batch_size]
            images = []
            messages = []
            for row in batch_rows:
                required = {"sample_id", "image1", "image2", "sentence"}
                missing = required - row.keys()
                if missing:
                    raise ValueError(f"missing fields {sorted(missing)} in row: {row}")
                left_image = Image.open(row["image1"]).convert("RGB")
                right_image = Image.open(row["image2"]).convert("RGB")
                images.extend([left_image, right_image])
                messages.append([{
                    "role": "user",
                    "content": [
                        left_image,
                        "The first image is the left image.",
                        right_image,
                        (
                            "The second image is the right image. Decide whether the sentence is true about this pair of images. "
                            "Reply with exactly True or False and no other text. Sentence: " + row["sentence"]
                        ),
                    ],
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
                    max_new_tokens=8,
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
                valid = answer in {"True", "False"}
                output = dict(row)
                output["generalist_output"] = {
                    "model": "MiniCPM-V-4_5",
                    "raw_answer": answer,
                    "prediction": answer if valid else None,
                    "latency_seconds": batch_latency / len(batch_rows),
                    "batch_latency_seconds": batch_latency,
                    "batch_size": len(batch_rows),
                    "protocol": "one_sample_one_model_request_with_two_images",
                    "model_requests_for_sample": 1,
                    "parse_status": "valid" if valid else "invalid_output_counted_wrong",
                }
                if not valid:
                    output["generalist_output"]["failure_kind"] = "unparseable_or_illegal_model_output"
                    output["generalist_output"]["parse_error"] = "answer must be exactly True or False"
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
