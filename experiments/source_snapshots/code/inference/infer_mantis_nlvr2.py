#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image

from mantis.models.mfuyu import MFuyuForCausalLM, MFuyuProcessor, chat_mfuyu


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=Path, required=True)
parser.add_argument("--input", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--start-index", type=int, default=0)
parser.add_argument("--end-index", type=int)
parser.add_argument("--max-new-tokens", type=int, required=True)
args = parser.parse_args()

rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line]
end_index = len(rows) if args.end_index is None else args.end_index
if not (0 <= args.start_index < end_index <= len(rows)):
    raise ValueError(f"invalid half-open range [{args.start_index}, {end_index}) for {len(rows)} rows")
if args.max_new_tokens < 1:
    raise ValueError("maximum output tokens must be positive")
if args.output.exists():
    raise FileExistsError(f"refusing to overwrite {args.output}")
rows = rows[args.start_index:end_index]

processor = MFuyuProcessor.from_pretrained(args.model)
processor.image_processor.size = {"height": 480, "width": 660}
model = MFuyuForCausalLM.from_pretrained(
    args.model,
    torch_dtype=torch.bfloat16,
)
model.cuda()
model.eval()

prompt_prefix = (
    "<image><image>\nDecide whether the following statement is true for the two images together. "
    "Reply with exactly True or False and no explanation. Statement: "
)
args.output.parent.mkdir(parents=True, exist_ok=True)
with args.output.open("x", encoding="utf-8") as destination:
    for local_index, row in enumerate(rows):
        required = {"sample_id", "image1", "image2", "sentence", "ground_truth"}
        missing = required - row.keys()
        if missing:
            raise ValueError(f"missing fields {sorted(missing)} in row: {row}")
        images = [Image.open(row["image1"]).convert("RGB"), Image.open(row["image2"]).convert("RGB")]
        started = time.perf_counter()
        with torch.inference_mode():
            raw_answer, _ = chat_mfuyu(
                prompt_prefix + row["sentence"],
                images,
                model,
                processor,
                max_new_tokens=args.max_new_tokens,
                num_beams=1,
                do_sample=False,
                pad_token_id=processor.tokenizer.eos_token_id,
            )
        latency = time.perf_counter() - started
        for image in images:
            image.close()

        answer = raw_answer.strip()
        output = dict(row)
        expert_output = {
            "model": "Mantis-VL/mfuyu_llava_nlvr2_8192_480p",
            "raw_answer": answer,
            "latency_seconds": latency,
            "protocol": "one_sample_one_model_request_with_two_images",
            "model_requests_for_sample": 1,
        }
        if answer in {"True", "False"}:
            expert_output["prediction"] = answer
            expert_output["parse_status"] = "valid"
        else:
            expert_output["prediction"] = None
            expert_output["parse_status"] = "invalid_output_counted_wrong"
            expert_output["failure_kind"] = "unparseable_or_illegal_model_output"
            expert_output["parse_error"] = "answer must be exactly True or False"
        output["expert_output"] = expert_output
        destination.write(json.dumps(output, ensure_ascii=False) + "\n")
        destination.flush()
        print(json.dumps({
            "completed_global_exclusive": args.start_index + local_index + 1,
            "end_index_exclusive": end_index,
            "latency_seconds": latency,
            "parse_status": expert_output["parse_status"],
        }), flush=True)
