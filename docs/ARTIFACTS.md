# Trained weights and compact experiment data

A companion artifact release is being prepared locally. Its public Hugging Face URL will be added after upload succeeds.

It includes the two trained ConstructionSite detectors, the 16 selected main-table routers with normalization statistics, and the final 8M distilled encoders. A compact data archive contains validation/test scores, task outcomes, sample IDs, frozen thresholds and curves. Paper table/figure source data and cost measurements accompany it.

After downloading and extracting the companion archives, run:

```bash
python scripts/replay_paper_results.py --artifacts /path/to/artifacts --output replayed_main.csv
```

This recomputes the saved learned curves and operating points from per-example outcomes, without running models. Full raw-image reproduction additionally requires original datasets, upstream models and feature extraction; multi-GB feature caches are excluded from the compact release. See `DOWNLOADS.md`, `DATA.md` and `REPRODUCE.md`.
