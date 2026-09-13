# -*- coding: utf-8 -*-
"""Copy a subset of all 101 UCF-101 classes into probe/ucf101_full.

Each class keeps N videos (random). Usage: python prepare_ucf101_full.py --n 4
"""
import argparse
import glob
import os
import random
import shutil

ROOT = "/mnt/HithinkOmniSSD/user_workspace/caisihang/research/ICASSP_ModelCollaboration"
SRC = os.path.join(ROOT, "data", "ucf101", "UCF-101")
DST = os.path.join(ROOT, "probe", "ucf101_full")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    classes = sorted(d for d in os.listdir(SRC) if os.path.isdir(os.path.join(SRC, d)))
    print(f"[ucf101_full] {len(classes)} classes, {args.n} videos each")
    rng = random.Random(args.seed)
    total = 0
    for cls in classes:
        files = sorted(glob.glob(os.path.join(SRC, cls, "*")))
        rng.shuffle(files)
        dst = os.path.join(DST, cls)
        os.makedirs(dst, exist_ok=True)
        for f in files[: args.n]:
            if not os.path.exists(os.path.join(dst, os.path.basename(f))):
                shutil.copy(f, os.path.join(dst, os.path.basename(f)))
        total += min(len(files), args.n)
    print(f"[ucf101_full] copied {total} videos -> {DST}")


if __name__ == "__main__":
    main()
