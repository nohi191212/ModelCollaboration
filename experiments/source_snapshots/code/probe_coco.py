# -*- coding: utf-8 -*-
"""Probe T2: can the Omni model do object detection on COCO images?

We ask for objects with bounding boxes in JSON and measure:
  1) parse rate   - fraction of images where a valid box list is returned
  2) box quality  - coarse: mean objects per image, whether coordinates are sane
This is a capability probe only (test2017 has no public GT); formal mAP
evaluation will use COCO val2017 in the main experiments.
Usage: python probe_coco.py --model-path ... --images-dir probe/coco --num-samples 60
"""
import argparse
import glob
import json
import os
import re

from omni_client import OmniClient


def extract_boxes(text):
    """Parse a loose JSON-ish / bracket box list from the model output."""
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return None
    body = m.group(0)
    try:
        data = json.loads(body)
        if isinstance(data, list):
            return data
    except Exception:
        pass
    # fallback: find repeated tuples of 4 numbers
    coords = re.findall(r"\[(\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*)\]", body)
    if coords:
        return [{"bbox": [float(x) for x in c]} for c in coords]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--num-samples", type=int, default=60)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="outputs/probe_coco.json")
    args = ap.parse_args()

    imgs = sorted(glob.glob(os.path.join(args.images_dir, "*.jpg")) +
                  glob.glob(os.path.join(args.images_dir, "*.png")))[: args.num_samples]
    omni = OmniClient(args.model_path, device=args.device)
    question = ("List all objects visible in this image with bounding boxes. "
                'Output strictly as JSON: [{"object": "name", "bbox": [x1,y1,x2,y2]}, ...]. '
                "Use absolute pixel coordinates.")

    rows, n_parse, n_obj = [], 0, 0
    for i, p in enumerate(imgs):
        ans = omni.ask_image(p, question)
        boxes = extract_boxes(ans)
        ok = boxes is not None
        n_parse += int(ok)
        n_obj += len(boxes) if boxes else 0
        rows.append({"image": os.path.basename(p), "parsed": ok,
                     "n_boxes": len(boxes) if boxes else 0, "raw": ans[:400]})
        print(f"[{i+1}/{len(imgs)}] {os.path.basename(p)[:30]} parsed={ok} "
              f"n_boxes={len(boxes) if boxes else 0}", flush=True)

    parse_rate = n_parse / len(imgs)
    avg_obj = n_obj / max(n_parse, 1)
    result = {"task": "coco_detection", "n": len(imgs), "parse_rate": parse_rate,
              "avg_objects_per_parsed": avg_obj, "rows": rows}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n[probe] parse_rate={parse_rate:.3f} avg_objects={avg_obj:.1f} -> {args.out}")


if __name__ == "__main__":
    main()
