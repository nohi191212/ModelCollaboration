# -*- coding: utf-8 -*-
"""Prepare probe subsets for the six tasks.

  ucf101 : pick K easy/hard classes from the full UCF-101
  eurosat: pick K classes, convert .tif -> .jpg (SigLIP2 cannot read tif)
  voxceleb: build same-speaker / different-speaker test pairs -> pairs.csv
  coco   : already downloaded (test2017 subset)
  esc50  : parquet -> wav (requires pyarrow; run after collab env is ready)
Usage: python prepare_probe_data.py
"""
import csv
import glob
import itertools
import os
import random
import shutil
import subprocess

ROOT = "/mnt/HithinkOmniSSD/user_workspace/caisihang/research/ICASSP_ModelCollaboration"
DATA = os.path.join(ROOT, "data")
PROBE = os.path.join(ROOT, "probe")

UCF_CLASSES = ["BaseballPitch", "ThrowDiscus", "YoYo", "BenchPress"]
EURO_CLASSES = ["Forest", "Highway", "Industrial", "AnnualCrop", "River"]
N_PER_CLASS = 30
SEED = 0


def main():
    rng = random.Random(SEED)
    os.makedirs(PROBE, exist_ok=True)

    # ---- UCF-101 ----
    ucf_root = os.path.join(DATA, "ucf101", "UCF-101")
    if os.path.isdir(ucf_root):
        for cls in UCF_CLASSES:
            src = os.path.join(ucf_root, cls)
            dst = os.path.join(PROBE, "ucf101", cls)
            if not os.path.isdir(src):
                print(f"[ucf101] class missing: {cls}")
                continue
            files = sorted(glob.glob(os.path.join(src, "*")))
            rng.shuffle(files)
            os.makedirs(dst, exist_ok=True)
            for f in files[:N_PER_CLASS]:
                shutil.copy(f, os.path.join(dst, os.path.basename(f)))
            print(f"[ucf101] {cls}: copied {min(len(files), N_PER_CLASS)}")

    # ---- EuroSAT (already JPEG despite .tif name in some mirrors; torchgeo ships .jpg) ----
    euro_root = os.path.join(DATA, "eurosat", "2750")
    if os.path.isdir(euro_root):
        for cls in EURO_CLASSES:
            src = os.path.join(euro_root, cls)
            dst = os.path.join(PROBE, "eurosat", cls)
            if not os.path.isdir(src):
                print(f"[eurosat] class missing: {cls}")
                continue
            files = sorted(glob.glob(os.path.join(src, "*.jpg")) +
                           glob.glob(os.path.join(src, "*.tif")))
            rng.shuffle(files)
            os.makedirs(dst, exist_ok=True)
            n = 0
            for f in files[:N_PER_CLASS]:
                out = os.path.join(dst, os.path.basename(f))
                if not os.path.exists(out):
                    shutil.copy(f, out)
                n += 1
            print(f"[eurosat] {cls}: copied {n}")

    # ---- VoxCeleb pairs ----
    vox_dir = os.path.join(DATA, "voxceleb", "test")
    if os.path.isdir(vox_dir):
        by_speaker = {}
        for w in glob.glob(os.path.join(vox_dir, "*.wav")):
            spk = os.path.basename(w).split("-")[0]
            by_speaker.setdefault(spk, []).append(w)
        speakers = list(by_speaker)
        rows, seen = [], set()
        # same-speaker pairs
        for spk in speakers:
            utts = by_speaker[spk]
            for a, b in itertools.combinations(utts, 2):
                if len(rows) >= 60:
                    break
                key = tuple(sorted((a, b)))
                if key in seen:
                    continue
                seen.add(key)
                rows.append((a, b, "same"))
            if len(rows) >= 60:
                break
        # different-speaker pairs
        for a, b in itertools.combinations(speakers, 2):
            if len(rows) >= 120:
                break
            u1 = rng.choice(by_speaker[a])
            u2 = rng.choice(by_speaker[b])
            key = tuple(sorted((u1, u2)))
            if key in seen:
                continue
            seen.add(key)
            rows.append((u1, u2, "diff"))
        with open(os.path.join(PROBE, "voxceleb_pairs.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["audio1", "audio2", "label"])
            for r in rows:
                w.writerow(r)
        n_same = sum(1 for r in rows if r[2] == "same")
        print(f"[voxceleb] pairs.csv: {len(rows)} pairs ({n_same} same / {len(rows)-n_same} diff)")

    print("[prepare] DONE")


if __name__ == "__main__":
    main()
