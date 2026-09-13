# -*- coding: utf-8 -*-
"""Generic audio probe with the Omni model.

Modes:
  classify  -> ESC-50 style: one audio clip, choose among classes.
  verify    -> VoxCeleb1 style: two clips, is it the same speaker? (binary)
Usage:
  python probe_audio.py --mode classify --folder <root> --classes "A,B" ...
  python probe_audio.py --mode verify --pairs pairs.csv ...
"""
import argparse
import csv
import glob
import json
import os

from omni_client import OmniClient


def collect_folder(root, classes, num_samples, seed):
    exts = ("*.wav", "*.mp3", "*.flac", "*.ogg", "*.m4a")
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


def collect_pairs(csv_path, num_samples, seed):
    pairs = []
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pairs.append((row["audio1"], row["audio2"], row["label"]))
    if seed >= 0:
        import random
        rng = random.Random(seed)
        rng.shuffle(pairs)
    return pairs[: num_samples]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--mode", choices=["classify", "verify"], required=True)
    ap.add_argument("--folder")
    ap.add_argument("--classes", help="comma-separated (classify)")
    ap.add_argument("--pairs", help="csv with audio1,audio2,label (verify)")
    ap.add_argument("--prompt", help="custom question (verify mode)")
    ap.add_argument("--num-samples", type=int, default=50)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="outputs/probe_audio.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    omni = OmniClient(args.model_path, device=args.device)
    rows, correct, n_parse = [], 0, 0

    if args.mode == "classify":
        classes = [c.strip() for c in args.classes.split(",")]
        samples = collect_folder(args.folder, classes, args.num_samples, args.seed)
        question = (f"What sound is this? Answer with exactly one of: "
                    f"{', '.join(classes)}.")
        for i, (path, gt) in enumerate(samples):
            ans = omni.ask_audio(path, question)
            pred = omni.parse_choice(ans, classes)
            ok = pred == gt
            if pred is not None:
                n_parse += 1
            correct += int(ok)
            rows.append({"path": os.path.basename(path), "gt": gt,
                         "raw": ans, "pred": pred, "correct": ok})
            print(f"[{i+1}/{len(samples)}] {os.path.basename(path)[:36]} gt={gt} "
                  f"pred={pred} ok={ok}", flush=True)
        n = len(samples)
    else:
        pairs = collect_pairs(args.pairs, args.num_samples, args.seed)
        question = (args.prompt or
                    "Do the two audio clips contain the same speaker? "
                    "Answer yes or no.")
        for i, (a1, a2, gt) in enumerate(pairs):
            ans = omni.ask_audio_pair(a1, a2, question)
            pred = omni.parse_yes_no(ans)
            expect = 1 if gt.strip().lower() in ("1", "yes", "same", "true") else 0
            ok = pred == expect
            if pred is not None:
                n_parse += 1
            correct += int(ok)
            rows.append({"a1": os.path.basename(a1), "a2": os.path.basename(a2),
                         "gt": gt, "raw": ans, "pred": pred, "correct": ok})
            print(f"[{i+1}/{len(pairs)}] {os.path.basename(a1)[:22]} | "
                  f"{os.path.basename(a2)[:22]} gt={gt} pred={pred} ok={ok}", flush=True)
        n = len(pairs)

    acc = correct / n if n else 0.0
    parse_rate = n_parse / n if n else 0.0
    result = {"task": args.mode, "n": n, "accuracy": acc,
              "parse_rate": parse_rate, "rows": rows}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n[probe] mode={args.mode} accuracy={acc:.3f} ({correct}/{n}) "
          f"parse_rate={parse_rate:.3f} -> {args.out}")


if __name__ == "__main__":
    main()
