# When Should a Vision Specialist Defer?

Code and supplementary material for specialist–VLM collaboration across CUB-200-2011, gRefCOCO, NLVR2 and ConstructionSite.

## Included

- `paper/main.pdf` and `paper/supplementary.pdf`.
- The original four-state router, strict SigLIP2 distillation, validation-only search and test evaluation.
- Sixteen final configurations, exact task prompts, saved comparison results and separate **unrun** input-removal configurations.
- The confidence baseline and task metrics used by the saved experiment.

## Installation

Use Python 3.12 and install `requirements.txt` in your own environment. Install a PyTorch build matching your CUDA version. VLM serving and specialist frameworks are separate dependencies: use their upstream environments rather than mixing every model in this environment.

## Data and weights

Set `ICASSP_DATA_ROOT` to the external experiment-data root described in `docs/DATA.md`. Large datasets, specialist weights, VLM weights and feature arrays are not copied into Git. See `docs/REPRODUCE.md` for the required files and commands. This repository preserves the recorded data-directory layout to reuse prepared features without changing the model or metric.

```bash
export ICASSP_DATA_ROOT=/path/to/experiment_data
python code/routing/train_shared_baseline.py --root "$ICASSP_DATA_ROOT" --config configs/main/shared_12_instancevg_Qwen3.8.json --output outputs/example --device cuda
```

This command starts training. It is provided for reproduction, not run as part of repository preparation.

## Method

Each branch is projected to 128 dimensions and normalized. A 128-unit GELU head predicts both-correct, rescue, harm and both-wrong outcomes. The final ranking score is rescue logit minus harm logit. Checkpoints maximize validation performance over 0–25% calls. A separate validation search chooses the deployment threshold, transferred unchanged to test. All main fits use seed 42.

## Experiment scope

The main result uses the final shared settings. Earlier input removals use different references and are identified as such in the supplement. Final-setting input-ablation configurations are supplied but have no results unless explicitly run. CUB reused-specialist exposure and earlier test inspection are described in the supplement. Cost curves are estimates assembled from measured components.

## Rights and publication

No GitHub remote is configured and nothing has been published. Dataset examples in the supplement retain their source-dataset attribution. Third-party repositories and model weights are not vendored. A code license has not been selected by the authors; choose one before publishing if you intend to grant reuse rights.
