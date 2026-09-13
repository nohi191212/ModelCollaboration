# -*- coding: utf-8 -*-
"""Generic image probe: classification or binary judgment with the Omni model.

Covers T5 (FaceForensics++ real/fake), T6 (EuroSAT multi-class) and the
image-level part of T2 (COCO). Input: either a folder per class, or a CSV
with columns path,label.
Usage:
  python probe_image.py --folder <root> --classes "A,B,C" --mode classify ...
  python probe_image.py --csv data.csv --mode binary --yes-label real ...
"""
import argparse
import csv
import glob
import json
import os

from omni_client import OmniClient


def collect_from_folder(root, classes, num_samples, seed):
    exts = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp")
    samples = []
    for cls in classes:
        d = os.path.join(root, cls)
        if not os.path.isdir(d):
            print(f"[warn] dir missing: {d}")
            continue
        files = []
        for e in exts:
            files += glob.glob(os.path.join(d, "**", e), recursive=True)
        files = sorted(files)
        if seed >= 0:
            import random
            rng = random.Random(seed)
            rng.shuffle(files)
        samples += [(f, cls) for f in files[: num_samples]]
    return samples


def collect_from_csv(csv_path, num_samples, seed):
    samples = []
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            samples.append((row["path"], row["label"]))
    if seed >= 0:
        import random
        rng = random.Random(seed)
        rng.shuffle(samples)
    return samples[: num_samples]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--folder")
    ap.add_argument("--csv")
    ap.add_argument("--classes", help="comma-separated classes (classify mode)")
    ap.add_argument("--mode", choices=["classify", "binary"], default="classify")
    ap.add_argument("--yes-label", default="real", help="binary: label meaning YES")
    ap.add_argument("--no-label", default="fake", help="binary: label meaning NO")
    ap.add_argument("--num-samples", type=int, default=50)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="outputs/probe_image.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.csv:
        samples = collect_from_csv(args.csv, args.num_samples, args.seed)
        classes = sorted({s[1] for s in samples})
    else:
        classes = [c.strip() for c in args.classes.split(",")]
        samples = collect_from_folder(args.folder, classes, args.num_samples, args.seed)

    omni = OmniClient(args.model_path, device=args.device)
    if args.mode == "classify":
        question = (f"What is shown in this image? Answer with exactly one of: "
                    f"{', '.join(classes)}.")
        ask = lambda p: omni.ask_image(p, question)
        parse = lambda ans: omni.parse_choice(ans, classes)
    else:
        question = (f"Is this image {args.yes_label} or {args.no_label}? "
                    f"Answer yes or no.")
        ask = lambda p: omni.ask_image(p, question)
        parse = lambda ans: omni.parse_yes_no(ans)

    rows, correct, n_parse = [], 0, 0
    for i, (path, gt) in enumerate(samples):
        ans = ask(path)
        if args.mode == "binary":
            pred = parse(ans)
            ok = (pred is not None) and (
                (pred == 1 and gt == args.yes_label) or (pred == 0 and gt == args.no_label)
            )
        else:
            pred = parse(ans)
            ok = pred == gt
        if pred is not None:
            n_parse += 1
        correct += int(ok)
        rows.append({"path": os.path.basename(path), "gt": gt,
                     "raw": ans, "pred": pred, "correct": ok})
        print(f"[{i+1}/{len(samples)}] {os.path.basename(path)[:40]} gt={gt} "
              f"pred={pred} ok={ok} raw={ans[:50]!r}", flush=True)

    acc = correct / len(samples) if samples else 0.0
    parse_rate = n_parse / len(samples) if samples else 0.0
    result = {"task": args.mode, "n": len(samples), "accuracy": acc,
              "parse_rate": parse_rate, "rows": rows}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n[probe] accuracy={acc:.3f} ({correct}/{len(samples)}) "
          f"parse_rate={parse_rate:.3f} -> {args.out}")


if __name__ == "__main__":
    main()
