# -*- coding: utf-8 -*-
"""Probe T1: UCF-101 video action recognition with the Omni model.

Measures whether the omni large model can classify fine-grained actions
(raw probe: does it beat chance / is it usable as a router target?).
Usage:
  python probe_ucf101.py --videos-dir <dir> --classes "BaseballPitch,Throw,YoYo" \
      --num-samples 30 --out outputs/probe_ucf101.json
"""
import argparse
import glob
import json
import os

from omni_client import OmniClient


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--videos-dir", required=True)
    ap.add_argument("--classes", default="auto",
                    help="comma-separated class names, or 'auto' to scan dirs")
    ap.add_argument("--num-samples", type=int, default=30)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="outputs/probe_ucf101.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.classes.strip().lower() == "auto":
        classes = sorted(d for d in os.listdir(args.videos_dir)
                         if os.path.isdir(os.path.join(args.videos_dir, d)))
        print(f"[probe] auto-detected {len(classes)} classes")
    else:
        classes = [c.strip() for c in args.classes.split(",")]
    video_exts = ("*.mp4", "*.avi", "*.mkv", "*.webm")

    videos = []
    for cls in classes:
        cls_dir = os.path.join(args.videos_dir, cls)
        if not os.path.isdir(cls_dir):
            print(f"[warn] class dir missing: {cls_dir}")
            continue
        files = []
        for ext in video_exts:
            files += glob.glob(os.path.join(cls_dir, "**", ext), recursive=True)
        files = sorted(files)
        if args.seed >= 0:
            import random
            rng = random.Random(args.seed)
            rng.shuffle(files)
        videos += [(f, cls) for f in files[: args.num_samples]]
    print(f"[probe] {len(videos)} videos over {len(classes)} classes")

    omni = OmniClient(args.model_path, device=args.device)
    question = (
        "What action is the person performing in this video? "
        f"Answer with exactly one of: {', '.join(classes)}."
    )

    rows, correct = [], 0
    for i, (vpath, gt) in enumerate(videos):
        ans = omni.ask_video(vpath, question)
        pred = omni.parse_choice(ans, classes)
        ok = pred == gt
        correct += int(ok)
        rows.append({"video": os.path.basename(vpath), "gt": gt,
                     "raw": ans, "pred": pred, "correct": ok})
        print(f"[{i+1}/{len(videos)}] {os.path.basename(vpath)[:40]} gt={gt} "
              f"pred={pred} ok={ok} raw={ans[:60]!r}", flush=True)

    acc = correct / len(rows) if rows else 0.0
    result = {"task": "ucf101", "classes": classes, "n": len(rows),
              "accuracy": acc, "rows": rows}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n[probe] accuracy = {acc:.3f} ({correct}/{len(rows)}) -> {args.out}")


if __name__ == "__main__":
    main()
