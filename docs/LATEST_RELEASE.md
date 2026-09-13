# Latest release: 20260913_lr

The release contains the latest paper results and the full experiment record. Download artifacts from `nohi191212/ModelCollaboration` on Hugging Face under `releases/20260913_lr/`.

## Weights and records

`full_budget_20260913_weights.tar.gz` contains exactly **80 router checkpoints**: 16 model pairs × five seeds for the main table. This release publishes no other router weights.

The study-specific `_records.tar.gz` archives retain configurations, training histories, validation/test outputs, curves and selection records for the experiments. Extract them into the same artifact directory; paths under `outputs/` are preserved. Earlier fixed-answer CSS/OvA adaptations are separate from the corrected K-class implementations.

Previously released backbone and specialist weights remain in the root-level `weights.tar`. External models and datasets are linked in `MODELS_AND_DATA.md`.

## Data

Eight `training_inputs_<expert>.tar.gz` files contain the train/validation/test arrays and example records for the latest 8M frozen-backbone setup. They contain image/text features, the selected specialist hidden state, confidence, output features, validity masks and labels. `input_manifest.json` records dimensions, sample counts and hidden-layer names. Set `ICASSP_DATA_ROOT` to the directory containing `training_inputs/` and this manifest.

`paired_outcomes.tar.gz` provides the paired correctness and task-metric sufficient statistics used for result replay. `paper_replay_runs.tar.gz` contains the 400 training runs used by the latest Tables 1 and 3, with raw logits and configurations. These two files are sufficient for `scripts/replay_latest_results.py`.

`additional_analysis_records.tar.gz` contains scoring explorations and cost records. `experiment_sources.tar.gz` preserves original experiment source snapshots, also included in the GitHub repository.

The latest `paper/data/` directory stores table and figure summaries. Main weights were selected over the full 0–100% validation curve. Table 3 retains its earlier training selection setting, as specified in the supplement and run configurations.

## Training examples

Install the dependencies in `requirements.txt` and a PyTorch build for your GPU. Commands below are run from the repository root. They start training only when invoked.

For normalized LR, download/extract a specialist's input archive and `input_manifest.json`, then run:

```bash
export ICASSP_DATA_ROOT="$PWD/artifacts/releases/20260913_lr"
python code/experiments/full_budget_train.py --source configs/full_budget/full_cub_MiniCPM_s2026 --output outputs/my_lr_run
```

For post-hoc deferral, extract its records archive, choose a saved `config.json`, and run:

```bash
python code/experiments/train_ltd.py --root "$PWD" --config "$ICASSP_DATA_ROOT/outputs/ltd_fixed_backbone_20260913/runs/cub_MiniCPM_posthoc_c0_s2026/config.json" --output outputs/my_posthoc_run --device cuda
```

Use `train_three.py` for a configuration from `three_state_20260913`, and `train_native.py` for a configuration from `ltd_kclass_native_20260913`. Configurations are supplied in the records and weight archives. The original objective experiment sources are in `experiments/source_snapshots/`; the five-seed objective trainer is also in `code/objectives/`.

To evaluate a newly trained post-hoc/three-state/native run, use its corresponding `evaluate_ltd.py`, `evaluate_three.py` or `evaluate_native.py` with `--help` for arguments. Input paths are resolved through `ICASSP_DATA_ROOT`.

Earlier architecture studies use other feature sizes or hidden layers. Their original configurations and source snapshots are supplied; regenerating those inputs from images uses the original feature-preparation scripts and the upstream datasets. The cached inputs are the latest common 8M setup.

## Verification

`scripts/replay_latest_results.py` recomputes the latest Tables 1 and 3 using the saved per-example outputs. `verification_latest.json` records 80 main runs, 320 Table 3 runs and 19,392 checked curve points. `weight_reload_verification.json` checks all 80 main weights against saved logits using the released cache loader (128 validation inputs per run).

Original code is MIT licensed. Derived data retain the licenses of their source datasets; consult `MODELS_AND_DATA.md` before redistribution.
