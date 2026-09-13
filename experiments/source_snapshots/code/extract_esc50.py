# -*- coding: utf-8 -*-
"""Extract ESC-50 audio from ashraq/esc50 parquet into per-class wav folders.

Usage: python extract_esc50.py --classes "Dog,Chainsaw,Siren,Rain,Clock_tick"
"""
import argparse
import os

import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import soundfile as sf

DATA = "/mnt/HithinkOmniSSD/user_workspace/caisihang/research/ICASSP_ModelCollaboration/data/esc50"
OUT = "/mnt/HithinkOmniSSD/user_workspace/caisihang/research/ICASSP_ModelCollaboration/probe/esc50"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", required=True)
    ap.add_argument("--n-per-class", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    classes = [c.strip() for c in args.classes.split(",")]

    tables = []
    for f in sorted(os.listdir(DATA)):
        if f.endswith(".parquet"):
            tables.append(pq.read_table(os.path.join(DATA, f)))
    table = tables[0].combine_chunks()
    for t in tables[1:]:
        table = pa.concat_tables([table, t])
    cols = table.column_names
    print("[esc50] columns:", cols, "rows:", table.num_rows)

    idx = {c: i for i, c in enumerate(cols)}
    audio_col = table.column("audio").to_pylist()
    cat_col = table.column("category").to_pylist() if "category" in idx else None
    lab_col = table.column("label").to_pylist() if "label" in idx else None

    import random
    rng = random.Random(args.seed)
    os.makedirs(OUT, exist_ok=True)
    for cls in classes:
        dst = os.path.join(OUT, cls)
        os.makedirs(dst, exist_ok=True)
        picked = 0
        order = list(range(table.num_rows))
        rng.shuffle(order)
        for i in order:
            name = (cat_col[i] if cat_col is not None else lab_col[i]) if cat_col or lab_col else None
            if name is None:
                continue
            if name != cls:
                continue
            arr = audio_col[i]
            if isinstance(arr, dict):  # parquet struct {'bytes','path'}
                arr = arr.get("bytes")
            data, sr = sf.read(__import__("io").BytesIO(arr), dtype="float32")
            if sr != 16000:
                data = __import__("librosa").resample(data, orig_sr=sr, target_sr=16000)
            sf.write(os.path.join(dst, f"{cls}_{picked}.wav"), data, 16000)
            picked += 1
            if picked >= args.n_per_class:
                break
        print(f"[esc50] {cls}: {picked} wavs")


if __name__ == "__main__":
    main()
