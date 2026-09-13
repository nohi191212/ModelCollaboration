# -*- coding: utf-8 -*-
"""Prepare deepfake probe subset: real (wiki) vs fake (inpainting), matched by filename.

wiki.zip must be downloaded and unzipped first. Usage: python prepare_deepfake.py
"""
import glob
import os
import random
import shutil
import zipfile

ROOT = "/mnt/HithinkOmniSSD/user_workspace/caisihang/research/ICASSP_ModelCollaboration"
DATA = os.path.join(ROOT, "data")
PROBE = os.path.join(ROOT, "probe", "deepfake")
N = 40
SEED = 0


def main():
    fake_root = os.path.join(DATA, "deepfake_inpainting", "inpainting")
    wiki_zip = os.path.join(DATA, "wiki.zip")
    wiki_root = os.path.join(DATA, "wiki")

    if not os.path.isdir(wiki_root) and os.path.exists(wiki_zip):
        print("[deepfake] unzipping wiki.zip ...")
        with zipfile.ZipFile(wiki_zip) as z:
            z.extractall(DATA)
    if not os.path.isdir(fake_root):
        print("[deepfake] fake root missing:", fake_root)
        return

    fake_files = sorted(glob.glob(os.path.join(fake_root, "*", "*.jpg")) +
                        glob.glob(os.path.join(fake_root, "*", "*.png")) +
                        glob.glob(os.path.join(fake_root, "*.jpg")))
    rng = random.Random(0)
    rng.shuffle(fake_files)
    fake_files = fake_files[:N]

    # real images: sample from the whole wiki tree (no strict pairing needed for probe)
    wiki_files = [p for p in glob.glob(os.path.join(wiki_root, "**", "*"), recursive=True)
                  if os.path.isfile(p)]
    rng.shuffle(wiki_files)
    real_files = wiki_files[:N]

    os.makedirs(os.path.join(PROBE, "real"), exist_ok=True)
    os.makedirs(os.path.join(PROBE, "fake"), exist_ok=True)
    n_real = n_fake = 0
    for f in fake_files:
        dst_f = os.path.join(PROBE, "fake", f"fake_{n_fake}.jpg")
        if not os.path.exists(dst_f):
            shutil.copy(f, dst_f)
        n_fake += 1
    for f in real_files:
        dst_r = os.path.join(PROBE, "real", f"real_{n_real}.jpg")
        if not os.path.exists(dst_r):
            shutil.copy(f, dst_r)
        n_real += 1

    print(f"[deepfake] fake={n_fake} real={n_real} -> {PROBE}")


if __name__ == "__main__":
    main()
