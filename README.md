# When Should a Vision Specialist Defer?

Code and supplementary material for specialist–VLM collaboration across CUB-200-2011, gRefCOCO, NLVR2 and ConstructionSite.

## Included

- `paper/main.pdf` and `paper/supplementary.pdf`.
- The original four-state router, strict SigLIP2 distillation, validation-only search and test evaluation.
- Sixteen final configurations, exact task prompts, saved comparison results and separate **unrun** input-removal configurations.
- The confidence baseline and task metrics used by the saved experiment.

## Installation

Use Python 3.12 and install `requirements.txt` in your own environment. Install a PyTorch build matching your CUDA version. VLM serving and specialist frameworks are separate dependencies: use their upstream environments rather than mixing every model in this environment.

## Model and dataset downloads

### Trained weights and paper results

Our [Hugging Face release](https://huggingface.co/nohi191212/ModelCollaboration) provides:

| Resource | Download | Contents |
|---|---|---|
| Trained weights (317 MB) | [weights.tar](https://huggingface.co/nohi191212/ModelCollaboration/resolve/main/weights.tar) | 16 main-table routers, normalization parameters, final 8M distilled encoders, and trained YOLO26x/RT-DETR-X specialists |
| Per-example experiment data | [experiment_data.tar.gz](https://huggingface.co/nohi191212/ModelCollaboration/resolve/main/experiment_data.tar.gz) | Validation/test scores, outcomes, sample IDs, thresholds and learned curves |
| Table and figure data | [paper_results/](https://huggingface.co/nohi191212/ModelCollaboration/tree/main/paper_results) | Confidence/learned comparisons, ablations, outcome analysis and cost measurements |

Download and check the reported results without running the models:

```bash
pip install huggingface_hub numpy
hf download nohi191212/ModelCollaboration --local-dir artifacts
tar -xf artifacts/weights.tar -C artifacts
tar -xzf artifacts/experiment_data.tar.gz -C artifacts
python scripts/replay_paper_results.py --artifacts artifacts --output replayed_main.csv
```

This recomputes the learned validation/test curves and validation-selected operating points from saved per-example outcomes. Running from original images additionally requires the datasets, upstream models and feature preparation below. Full feature caches are excluded from the compact release.

### Vision-language models and router teacher

| Model | Hugging Face download | Role |
|---|---|---|
| MiniCPM-V-4.5 | [openbmb/MiniCPM-V-4_5](https://huggingface.co/openbmb/MiniCPM-V-4_5) | VLM in the main comparisons |
| Qwen3.8-27B-FP8 | [Qwen/Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) | FP8 VLM in the main comparisons |
| SigLIP2 Base, patch 16, resolution 224 | [google/siglip2-base-patch16-224](https://huggingface.co/google/siglip2-base-patch16-224) | Teacher for the compact image/text encoders |

```bash
hf download openbmb/MiniCPM-V-4_5 --local-dir models/MiniCPM-V-4_5
hf download Qwen/Qwen3.8-27B-FP8 --local-dir models/Qwen3.8-27B-FP8
hf download google/siglip2-base-patch16-224 --local-dir models/siglip2-base-patch16-224
```

Follow each model card for serving dependencies. Preserve the Qwen FP8 variant and the task prompts in [`prompts/`](prompts/) to match the experiments.

### Specialist models

| Specialist / task | Download or implementation | Checkpoint note |
|---|---|---|
| InstanceVG / gRefCOCO | [Hugging Face weights](https://huggingface.co/Dmmm997/InstanceVG/tree/main/gres/InstanceVG-grefcoco), [implementation](https://github.com/Dmmm1997/InstanceVG) | Use the gRefCOCO checkpoint |
| ViLT / NLVR2 | [dandelin/vilt-b32-finetuned-nlvr2](https://huggingface.co/dandelin/vilt-b32-finetuned-nlvr2) | Task-finetuned upstream weights |
| BEiT-3 / NLVR2 | [Raghavan/beit3_base_patch16_224_nlvr2](https://huggingface.co/Raghavan/beit3_base_patch16_224_nlvr2), [implementation](https://github.com/microsoft/unilm/tree/master/beit3) | Task-finetuned upstream weights; preserve the recorded input processing |
| Grounding DINO / gRefCOCO | [Implementation and checkpoints](https://github.com/IDEA-Research/GroundingDINO) | Match the experiment's architecture and checkpoint |
| ResNet-50 / CUB-200-2011 | [Torchvision ResNet-50](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html) | Initialization only; the CUB-trained checkpoint is also required |
| GLSim / CUB-200-2011 | Paper checkpoint: `cub_vit_b16_16_2.pth` | Exact public download URL pending |
| YOLO26x / ConstructionSite | [Our trained weights](https://huggingface.co/nohi191212/ModelCollaboration/resolve/main/weights.tar) | `specialists/yolo26x/best.pt` inside the archive |
| RT-DETR-X / ConstructionSite | [Our trained weights](https://huggingface.co/nohi191212/ModelCollaboration/resolve/main/weights.tar) | `specialists/rtdetr_x_fp32/best.pt`; recorded inference uses FP32 |

### Datasets

| Dataset | Official download / access instructions | Required contents |
|---|---|---|
| CUB-200-2011 | [Caltech dataset record](https://data.caltech.edu/records/65de6-vp158) | Images, classes, bounding boxes and original split annotations |
| gRefCOCO | [Official annotations](https://github.com/henghuiding/gRefCOCO), [COCO images](https://cocodataset.org/#download) | Expressions, annotations and corresponding COCO images; follow upstream image-version instructions |
| NLVR2 | [Official dataset page](https://lil.nlp.cornell.edu/nlvr/), [download instructions](https://github.com/lil-lab/nlvr/tree/master/nlvr2) | Paired images, statements and labels |
| ConstructionSite (ConstructionSite10k) | [LouisChen15/ConstructionSite](https://huggingface.co/datasets/LouisChen15/ConstructionSite) | Images and annotations; accept the dataset's access conditions |

For ConstructionSite, after access is granted:

```bash
hf auth login
hf download LouisChen15/ConstructionSite --repo-type dataset --local-dir data/constructionsite10k
```

Dataset and model licenses remain those of their authors. ConstructionSite is distributed under CC BY-NC 4.0. Original dataset splits are not automatically the router train/validation/test splits: follow the split definitions, exclusions and evaluation rules in [`paper/supplementary.pdf`](paper/supplementary.pdf).

### Prepare inputs and train

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

The source repository is [nohi191212/ModelCollaboration](https://github.com/nohi191212/ModelCollaboration). Dataset examples in the supplement retain their source-dataset attribution. Third-party repositories and model weights are not vendored. The authors have not yet selected a code license.
