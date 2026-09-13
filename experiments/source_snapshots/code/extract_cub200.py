# -*- coding: utf-8 -*-
"""Extract CUB-200 images from anjunhu parquet into per-class folders.

text column looks like 'a photo of a Black Footed Albatross'.
Usage: python extract_cub200.py --n-per-class 30
"""
import argparse
import io
import os
import random

import pyarrow.parquet as pq
from PIL import Image

DATA = "/mnt/HithinkOmniSSD/user_workspace/caisihang/research/ICASSP_ModelCollaboration/data/cub200"
OUT = "/mnt/HithinkOmniSSD/user_workspace/caisihang/research/ICASSP_ModelCollaboration/probe/cub200"


def cls_from_text(t):
    # 'a photo of a Black Footed Albatross' -> 'Black_Footed_Albatross'
    words = t.strip().split()
    # skip leading 'a photo of a/an'
    i = 0
    for w in words:
        if w.lower() in ("a", "an", "of", "photo", "the"):
            i += 1
        else:
            break
    name = "_".join(words[i:]) if i < len(words) else "unknown"
    return name.strip("_")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-class", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tables = []
    for f in ("train.parquet", "test.parquet"):
        p = os.path.join(DATA, f)
        if os.path.exists(p):
            tables.append(pq.read_table(p))
    if not tables:
        print("[cub200] no parquet found")
        return
    table = tables[0]
    for t in tables[1:]:
        import pyarrow as pa
        table = pa.concat_tables([table, t])

    texts = table.column("text").to_pylist()
    imgs = table.column("image").to_pylist()

    by_class = {}
    for t, im in zip(texts, imgs):
        cls = cls_from_text(t)
        by_class.setdefault(cls, []).append(im)

    rng = random.Random(args.seed)
    total = 0
    for cls, items in sorted(by_class.items()):
        rng.shuffle(items)
        dst = os.path.join(OUT, cls)
        os.makedirs(dst, exist_ok=True)
        n = 0
        for im in items[: args.n_per_class]:
            data = im.get("bytes") if isinstance(im, dict) else im
            try:
                img = Image.open(io.BytesIO(data)).convert("RGB")
                img.save(os.path.join(dst, f"{cls}_{n}.jpg"), quality=90)
                n += 1
            except Exception as e:
                print(f"[cub200] skip {cls}: {e}")
        total += n
    print(f"[cub200] {len(by_class)} classes, {total} images -> {OUT}")


if __name__ == "__main__":
    main()
