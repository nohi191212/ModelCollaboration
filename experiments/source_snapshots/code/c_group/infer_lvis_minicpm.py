from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoTokenizer
from transformers.dynamic_module_utils import get_class_from_dynamic_module


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
args.output.parent.mkdir(parents=True, exist_ok=True)

with args.output.open("w", encoding="utf-8") as handle:
    for index, row in enumerate(rows, start=1):
        image = Image.open(row["image"]).convert("RGB")
        messages = [{"role": "user", "content": [image, row["question"]]}]
        torch.cuda.synchronize()
        started = time.perf_counter()
        answer = model.chat(
            msgs=messages,
            tokenizer=tokenizer,
            enable_thinking=False,
            sampling=False,
            stream=False,
            max_new_tokens=8,
        )
        torch.cuda.synchronize()
        latency = time.perf_counter() - started
        normalized = answer.strip().lower()
        if normalized.startswith("yes"):
            prediction = True
        elif normalized.startswith("no"):
            prediction = False
        else:
            raise ValueError(f"sample {row['sample_id']} returned non yes/no answer: {answer!r}")
        output = dict(row)
        output["model_output"] = {
            "model": "MiniCPM-V-4_5",
            "answer": answer,
            "pred_label": prediction,
            "latency_seconds": latency,
        }
        handle.write(json.dumps(output, ensure_ascii=False) + "\n")
        image.close()
        if index % 100 == 0 or index == len(rows):
            handle.flush()
            print(json.dumps({"model": "MiniCPM-V-4_5", "completed": index, "total": len(rows)}), flush=True)

print(json.dumps({"model": "MiniCPM-V-4_5", "records": len(rows), "output": str(args.output)}))
