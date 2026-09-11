# Reproducing the experiments

1. Prepare the external data tree in `DATA.md`, using the recorded source splits and feature order.
2. Distill the encoders with `code/siglip_distillation/prepare_strict_20260910.py`, `cache_teacher.py` and `train_distillation_strict.py`. Check `--help` for cache/output paths and GPU options. The exact 8M recorded configuration is `configs/distilled_8M_record.json`. Distillation uses only permitted training images/text, 300 epochs, batch 2048, AdamW at 0.0001 and cosine decay. Never reuse an invalid earlier checkpoint.
3. Train each file in `configs/main/` with `train_shared_baseline.py`. Keep its run ID as the output directory name. This reads train/validation only. Standardization uses training statistics.
4. To reproduce the sequential search, use `prepare_shared_baseline_study.py --root "$ICASSP_DATA_ROOT" --output /path/to/search`, then `run_shared_baseline_search.py` with `--root`, `--work` and `--trainer` as shown by its help. The saved sequence and final selection are in `results/DSE_STATE.json` and `results/selected_config.json`. Search preparation also reads each specialist's layer inventory, as described in `STATE_FEATURES.md`.
5. Put the final runs, `DSE_STATE.json` and `selected_config.json` under one work directory, then run:

```bash
python code/routing/evaluate_shared_baseline.py --root "$ICASSP_DATA_ROOT" --work /path/to/final_work
```

This is a full evaluation command, not part of repository validation. It chooses thresholds using validation records and then evaluates test predictions. It writes new results and refuses to overwrite an existing final result. The release version removes only the obsolete comparison against earlier runs, leaving score calculation and selection unchanged.

Input-removal experiments are separate: add `--allow-input-ablation` when training a configuration in `configs/input_ablations/`. Normalization and model input dimensions follow the remaining branches. These configurations have **not been run**. Text removal is defined only for gRefCOCO and NLVR2. Keep their outputs separate from main-table runs.

The confidence baseline can be reproduced separately with `code/routing/evaluate_confidence.py --root "$ICASSP_DATA_ROOT" --expert instancevg --endpoint Qwen3.8 --output outputs/confidence_instancevg_qwen.json`. It additionally reads `confidence_columns` from `outputs/router_exploration_25pct_20260910/datasets/<expert>/completion.json`, selects on validation and evaluates the same threshold on test.

Use `python scripts/summarize_results.py` to check the saved paper counts without model execution, and `python scripts/plot_hidden_depth.py` to redraw the depth curve.

All reproduction needs external datasets and cached features. The source bundle alone does not include the large data/weight artifacts. CPU validation performed during packaging checks imports, all configured model input combinations and eight cached validation examples against an existing checkpoint; it does not claim to reproduce full training.
