from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from PIL import Image
from transformers import AutoProcessor
from vllm import LLM, SamplingParams


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    processor = AutoProcessor.from_pretrained(args.model)
    model = LLM(
        model=str(args.model),
        trust_remote_code=True,
        tensor_parallel_size=1,
        max_model_len=4096,
        limit_mm_per_prompt={"image": 1},
    )
    sampling = SamplingParams(temperature=0, max_tokens=8)
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line]
    if args.limit is not None:
        rows = rows[: args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows, start=1):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": row["question"]},
                    ],
                }
            ]
            prompt = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            image = Image.open(row["image"]).convert("RGB")
            started = time.perf_counter()
            generated = model.generate(
                {
                    "prompt": prompt,
                    "multi_modal_data": {"image": image},
                },
                sampling,
                use_tqdm=False,
            )
            latency = time.perf_counter() - started
            image.close()
            answer = generated[0].outputs[0].text.strip()
            normalized = answer.lower()
            if normalized.startswith("yes"):
                prediction = True
            elif normalized.startswith("no"):
                prediction = False
            else:
                raise ValueError(f"sample {row['sample_id']} returned non yes/no answer: {answer!r}")
            output = dict(row)
            output["model_output"] = {
                "model": "Qwen3.8-27B-FP8",
                "answer": answer,
                "pred_label": prediction,
                "latency_seconds": latency,
            }
            handle.write(json.dumps(output, ensure_ascii=False) + "\n")
            if index % 100 == 0 or index == len(rows):
                handle.flush()
                print(json.dumps({"model": "Qwen3.8-27B-FP8", "completed": index, "total": len(rows)}), flush=True)

    print(json.dumps({"model": "Qwen3.8-27B-FP8", "records": len(rows), "output": str(args.output)}))


if __name__ == "__main__":
    main()
