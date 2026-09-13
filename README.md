# Learned Routing for Specialist–VLM Collaboration

Code, trained weights and experiment data for **When Should a Vision Specialist Defer? Budgeted Collaboration with Vision–Language Models**.

[Paper](paper/main.pdf) · [Supplement](paper/appendix.pdf) · [Weights and data](https://huggingface.co/nohi191212/ModelCollaboration/tree/main/releases/20260913_lr) · [Reproduction instructions](docs/REPRODUCE.md)

## What is included

- Latest normalized Learned Routing results: 16 model pairs, five seeds (2026–2030), and validation selection across 0–100% calls.
- Training and evaluation code for the main router, three-state and gain objectives, post-hoc deferral, and classification-based CSS/OvA experiments.
- The 80 main-table router weights (16 model pairs × five seeds).
- Per-example model outputs, paired outcomes, training histories, configurations and saved curves.
- Cached input features and labels for all eight specialists, including train, validation and test splits, to retrain the latest frozen-backbone routers.
- Paper sources, plotting code, task prompts and independent result-replay scripts.

## Recompute the paper results

Install Python 3.12, PyTorch and the Python dependencies, then download the two replay archives:

```bash
pip install -r requirements.txt
hf download nohi191212/ModelCollaboration releases/20260913_lr/paper_replay_runs.tar.gz releases/20260913_lr/paired_outcomes.tar.gz --local-dir artifacts
tar -xzf artifacts/releases/20260913_lr/paper_replay_runs.tar.gz -C artifacts/releases/20260913_lr
tar -xzf artifacts/releases/20260913_lr/paired_outcomes.tar.gz -C artifacts/releases/20260913_lr
python scripts/replay_latest_results.py --artifacts artifacts/releases/20260913_lr --output verification_latest.json
```

The script reconstructs Table 1 and Table 3 from per-example outputs. It checks 80 main runs, 320 training runs behind Table 3, and 19,392 main/confidence curve points. No GPU or model inference is needed for this replay.

## Retrain a router

Download the relevant `training_inputs_<specialist>.tar.gz` and `input_manifest.json` from the same release folder and extract the archive there. For example, after extracting `training_inputs_cub.tar.gz`:

```bash
export ICASSP_DATA_ROOT="$PWD/artifacts/releases/20260913_lr"
python code/experiments/full_budget_train.py --source configs/full_budget/full_cub_MiniCPM_s2026 --output outputs/retrained_cub
```

The input cache contains the frozen encoder features and specialist information, so this trains only the routing head. Commands for the other objectives and the full artifact map are in [REPRODUCE.md](docs/REPRODUCE.md). Original-image experiments use the [upstream models and datasets](docs/MODELS_AND_DATA.md).

## Method and experiments

LR estimates paired model outcomes and ranks inputs using

`(p_rescue - p_harm) / (p_rescue + p_harm + 2*p_both_wrong + 1e-8)^alpha`.

The main experiment selects the epoch and `alpha` from the full validation curve, then selects a threshold over 0–100% calls. Table 3 uses the earlier training selection rule described in the supplement. The release preserves each experiment's actual settings.

The corrected CSS/OvA classification experiments have K class outputs plus a deferral output. Earlier fixed-answer adaptations are retained separately in the experiment archive. The main paper compares post-hoc deferral under fixed specialist answers.

## Repository layout

- `code/`: current training/evaluation entry points and input preparation.
- `configs/full_budget/`: all 80 main-run configurations.
- `experiments/source_snapshots/`: original source files from the experiment studies.
- `paper/`: latest manuscript, tables, figures and numerical summaries.
- `scripts/replay_latest_results.py`: independent paper-result replay.
- `docs/`: artifact mapping, model/data sources and reproduction commands.

Older root-level Hugging Face archives remain available for earlier paper versions; the latest release is `releases/20260913_lr/`.

## License

Our original code is released under the MIT license. Upstream model and dataset licenses remain unchanged. Original images are obtained from their dataset providers; this release supplies the experiment features, labels and outputs. ConstructionSite-derived data retain the source dataset's CC BY-NC 4.0 terms.
