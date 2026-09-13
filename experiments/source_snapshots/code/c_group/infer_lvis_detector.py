from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=Path, required=True)
parser.add_argument("--model-name", required=True)
parser.add_argument("--input", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--threshold", type=float, required=True)
args = parser.parse_args()

device = torch.device("cuda")
processor = AutoProcessor.from_pretrained(args.model, use_fast=False)
model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model).to(device).eval()
rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line]
args.output.parent.mkdir(parents=True, exist_ok=True)

with args.output.open("w", encoding="utf-8") as handle:
    for index, row in enumerate(rows, start=1):
        image = Image.open(row["image"]).convert("RGB")
        text_labels = [[row["detector_text"]]]
        inputs = processor(images=image, text=text_labels, return_tensors="pt").to(device)
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            outputs = model(**inputs, output_hidden_states=True)
        torch.cuda.synchronize()
        latency = time.perf_counter() - started

        if model.config.model_type == "mm-grounding-dino":
            result = processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=0.0,
                text_threshold=0.0,
                target_sizes=[image.size[::-1]],
                text_labels=text_labels,
            )[0]
            state_tensor = outputs.last_hidden_state
        else:
            result = processor.post_process_grounded_object_detection(
                outputs=outputs,
                target_sizes=[image.size[::-1]],
                threshold=0.0,
                text_labels=text_labels,
            )[0]
            state_tensor = outputs.class_embeds

        top_index = int(result["scores"].argmax().item())
        top_score = float(result["scores"][top_index].item())
        top_box = [round(float(value), 3) for value in result["boxes"][top_index].tolist()]
        state = state_tensor[0]
        while state.ndim > 1:
            state = state.mean(dim=0)

        output = dict(row)
        output["model_output"] = {
            "model": args.model_name,
            "answer": "yes" if top_score >= args.threshold else "no",
            "pred_label": top_score >= args.threshold,
            "confidence": top_score,
            "threshold": args.threshold,
            "detections_above_threshold": int((result["scores"] >= args.threshold).sum().item()),
            "top_box_xyxy": top_box,
            "latency_seconds": latency,
            "internal_state": {
                "source": "last_decoder_hidden_mean" if model.config.model_type == "mm-grounding-dino" else "class_embeds_mean",
                "shape": list(state_tensor.shape),
                "vector": state.float().cpu().tolist(),
            },
        }
        handle.write(json.dumps(output, ensure_ascii=False) + "\n")
        image.close()
        if index % 100 == 0 or index == len(rows):
            handle.flush()
            print(json.dumps({"model": args.model_name, "completed": index, "total": len(rows)}), flush=True)

print(json.dumps({"model": args.model_name, "records": len(rows), "output": str(args.output)}))
