# -*- coding: utf-8 -*-
"""Smoke test: load Qwen3-Omni and run one image / audio / video query.

Usage: python test_omni.py --model-path <path>
"""
import argparse
import glob
import os

from omni_client import OmniClient

ROOT = "/mnt/HithinkOmniSSD/user_workspace/caisihang/research/ICASSP_ModelCollaboration"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    args = ap.parse_args()

    omni = OmniClient(args.model_path)

    img = glob.glob(os.path.join(ROOT, "probe", "eurosat", "Forest", "*.jpg"))[0]
    print("\n[img]", os.path.basename(img))
    print("->", omni.ask_image(img, "What land cover is shown in this satellite image? Answer in one word."))

    aud = glob.glob(os.path.join(ROOT, "probe", "voxceleb", "*.wav")) if os.path.isdir(os.path.join(ROOT, "probe", "voxceleb")) else \
        glob.glob(os.path.join(ROOT, "data", "voxceleb", "test", "*.wav"))
    a1, a2 = aud[:2]
    print("\n[aud-pair]", os.path.basename(a1), "|", os.path.basename(a2))
    print("->", omni.ask_audio_pair(a1, a2, "Do the two clips contain the same speaker? Answer yes or no."))

    vid = glob.glob(os.path.join(ROOT, "probe", "ucf101", "YoYo", "*.avi"))[0]
    print("\n[vid]", os.path.basename(vid))
    print("->", omni.ask_video(vid, "What action is the person performing? Answer in one word."))

    print("\n[smoke] ALL OK")


if __name__ == "__main__":
    main()
