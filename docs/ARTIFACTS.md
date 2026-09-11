# Trained weights and compact experiment data

Download the companion release from [nohi191212/ModelCollaboration on Hugging Face](https://huggingface.co/nohi191212/ModelCollaboration).

- [Trained weights (317 MB)](https://huggingface.co/nohi191212/ModelCollaboration/resolve/main/weights.tar)
- [Compact per-example experiment data](https://huggingface.co/nohi191212/ModelCollaboration/resolve/main/experiment_data.tar.gz)
- [Paper table and figure data](https://huggingface.co/nohi191212/ModelCollaboration/tree/main/paper_results)

```bash
hf download nohi191212/ModelCollaboration --local-dir artifacts
tar -xf artifacts/weights.tar -C artifacts
tar -xzf artifacts/experiment_data.tar.gz -C artifacts
```

It includes the two trained ConstructionSite detectors, the 16 selected main-table routers with normalization statistics, and the final 8M distilled encoders. A compact data archive contains validation/test scores, task outcomes, sample IDs, frozen thresholds and curves. Paper table/figure source data and cost measurements accompany it.

After downloading and extracting the companion archives, run:

```bash
python scripts/replay_paper_results.py --artifacts /path/to/artifacts --output replayed_main.csv
```

This recomputes the saved learned curves and operating points from per-example outcomes, without running models. Full raw-image reproduction additionally requires original datasets, upstream models and feature extraction; multi-GB feature caches are excluded from the compact release. See `DOWNLOADS.md`, `DATA.md` and `REPRODUCE.md`.
