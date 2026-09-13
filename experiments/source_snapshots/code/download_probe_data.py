# -*- coding: utf-8 -*-
"""Download probe-stage datasets from hf-mirror.com.

  ESC-50    -> ashraq/esc50 parquet (full, 773MB)
  EuroSAT   -> torchgeo/eurosat EuroSAT.zip (RGB 10-class, 94MB)
  COCO      -> merve/coco instances_val2017.json + first N val images
  VoxCeleb  -> s3prl/mini_voxceleb1 test wav files (small probe set)
Usage: python download_probe_data.py [--n-coco 60] [--workers 8]
"""
import argparse
import concurrent.futures as cf
import json
import os
import sys
import urllib.request

HF = "https://hf-mirror.com"
DATA = os.path.join(
    "/mnt/HithinkOmniSSD/user_workspace/caisihang/research/ICASSP_ModelCollaboration",
    "data",
)
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}


def _open(url):
    req = urllib.request.Request(url, headers=UA)
    return urllib.request.urlopen(req, timeout=120)


def dl(url, path):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    with _open(url) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    os.replace(tmp, path)
    print(f"  ok {os.path.relpath(path, DATA)} ({os.path.getsize(path)//1024//1024}MB)")


def hf_tree(repo):
    url = f"{HF}/api/datasets/{repo}/tree/main?recursive=true"
    with _open(url) as r:
        return json.load(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-coco", type=int, default=60)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    jobs = []

    # 1) ESC-50 parquet
    for f in [
        "data/train-00000-of-00002-2f1ab7b824ec751f.parquet",
        "data/train-00001-of-00002-27425e5c1846b494.parquet",
        "dataset_infos.json",
    ]:
        jobs.append((f"{HF}/datasets/ashraq/esc50/resolve/main/{f}",
                     os.path.join(DATA, "esc50", os.path.basename(f))))

    # 2) EuroSAT RGB
    jobs.append((f"{HF}/datasets/torchgeo/eurosat/resolve/main/EuroSAT.zip",
                 os.path.join(DATA, "eurosat", "EuroSAT.zip")))

    # 3) COCO annotations + first N val images
    jobs.append((f"{HF}/datasets/merve/coco/resolve/main/annotations/instances_val2017.json",
                 os.path.join(DATA, "coco", "annotations", "instances_val2017.json")))
    tree = hf_tree("merve/coco")
    val_imgs = sorted(x["path"] for x in tree
                      if x["type"] == "file" and x["path"].startswith("val2017/")
                      and x["path"].endswith(".jpg"))[: args.n_coco]
    for p in val_imgs:
        jobs.append((f"{HF}/datasets/merve/coco/resolve/main/{p}",
                     os.path.join(DATA, "coco", p)))

    # 4) VoxCeleb mini test wavs
    tree = hf_tree("s3prl/mini_voxceleb1")
    vox = [x["path"] for x in tree
           if x["type"] == "file" and x["path"].startswith("test/")
           and x["path"].endswith(".wav")]
    for p in vox:
        jobs.append((f"{HF}/datasets/s3prl/mini_voxceleb1/resolve/main/{p}",
                     os.path.join(DATA, "voxceleb", p)))

    print(f"[download] {len(jobs)} files, {args.workers} workers")
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(dl, u, p): p for u, p in jobs}
        done = 0
        for fut in cf.as_completed(futs):
            fut.result()
            done += 1
            if done % 25 == 0:
                print(f"[download] {done}/{len(jobs)}", flush=True)
    print("[download] ALL DONE")


if __name__ == "__main__":
    main()
