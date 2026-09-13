# Model and dataset downloads

For the current paper, use [LATEST_RELEASE.md](LATEST_RELEASE.md). The latest release contains 80 main-table router weights and eight specialist input caches. The notes below describe the earlier release.

## Vision-language models and router teacher

| Resource | Download | Role |
|---|---|---|
| MiniCPM-V-4.5 | [openbmb/MiniCPM-V-4_5](https://huggingface.co/openbmb/MiniCPM-V-4_5) | VLM used in the main comparisons |
| Qwen3.8-27B-FP8 | [Qwen/Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) | FP8 VLM used in the main comparisons; use this variant rather than substituting the unquantized model |
| SigLIP2 Base, patch 16, resolution 224 | [google/siglip2-base-patch16-224](https://huggingface.co/google/siglip2-base-patch16-224) | Teacher for the compact image/text encoders |

Download with the Hugging Face CLI (install `huggingface_hub` first):

```bash
hf download openbmb/MiniCPM-V-4_5 --local-dir models/MiniCPM-V-4_5
hf download Qwen/Qwen3.8-27B-FP8 --local-dir models/Qwen3.8-27B-FP8
hf download google/siglip2-base-patch16-224 --local-dir models/siglip2-base-patch16-224
```

Follow each model card for serving dependencies. Task prompts are provided in [`prompts/`](../prompts/); do not replace them with a generic captioning prompt.

## Specialist resources

| Specialist / task | Upstream resource | Reproduction note |
|---|---|---|
| InstanceVG / gRefCOCO | [Hugging Face weights](https://huggingface.co/Dmmm997/InstanceVG/tree/main/gres/InstanceVG-grefcoco), [implementation](https://github.com/Dmmm1997/InstanceVG) | Use the gRefCOCO checkpoint, not a RefCOCO checkpoint |
| ViLT / NLVR2 | [dandelin/vilt-b32-finetuned-nlvr2](https://huggingface.co/dandelin/vilt-b32-finetuned-nlvr2) | Task-finetuned upstream weights |
| BEiT-3 / NLVR2 | [Raghavan/beit3_base_patch16_224_nlvr2](https://huggingface.co/Raghavan/beit3_base_patch16_224_nlvr2), [BEiT-3 implementation](https://github.com/microsoft/unilm/tree/master/beit3) | Task-finetuned upstream weights; preserve the experiment's input processing |
| Grounding DINO / gRefCOCO | [implementation and checkpoint instructions](https://github.com/IDEA-Research/GroundingDINO) | Match the architecture and weights specified by the experiment; arbitrary Grounding DINO variants are not interchangeable |
| ResNet-50 / CUB-200-2011 | [Torchvision ResNet-50](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html) | Initialization only; the CUB-trained specialist checkpoint is also required |
| GLSim / CUB-200-2011 | Paper checkpoint: `cub_vit_b16_16_2.pth` | An exact public download URL has not yet been added |
| YOLO26x and RT-DETR-X / ConstructionSite | [Ultralytics model instructions](https://docs.ultralytics.com/models/) | Upstream detection weights are initialization resources; the task-trained checkpoints are required for the reported results |

**Trained weights:** the [Hugging Face companion release](https://huggingface.co/nohi191212/ModelCollaboration) contains the 16 main-table routers, normalization parameters, the final 8M distilled encoders, and the trained YOLO26x/RT-DETR-X specialists. It also provides compact per-example results for checking the reported curves without model inference. See [`ARTIFACTS.md`](ARTIFACTS.md) for download and replay commands. Other specialist checkpoints and full feature caches remain external; full model execution requires the prepared inputs described in [`DATA.md`](DATA.md).

## Datasets

| Dataset | Official download / access instructions | Required contents |
|---|---|---|
| CUB-200-2011 | [Caltech dataset record](https://data.caltech.edu/records/65de6-vp158) | Images, classes, bounding boxes and original split annotations |
| gRefCOCO | [Official repository and annotation downloads](https://github.com/henghuiding/gRefCOCO), [COCO images](https://cocodataset.org/#download) | gRefCOCO expressions/annotations and the corresponding COCO images; follow the upstream image-version instructions |
| NLVR2 | [Official dataset page](https://lil.nlp.cornell.edu/nlvr/), [download instructions and annotations](https://github.com/lil-lab/nlvr/tree/master/nlvr2) | Paired images, statements and labels; follow the image-access instructions |
| ConstructionSite (ConstructionSite10k) | [LouisChen15/ConstructionSite](https://huggingface.co/datasets/LouisChen15/ConstructionSite) | Images and annotations; request access and accept the dataset terms before downloading |

For ConstructionSite, after access is granted:

```bash
hf auth login
hf download LouisChen15/ConstructionSite --repo-type dataset --local-dir data/constructionsite10k
```

Keep datasets outside the Git checkout when preparing large caches. Set `ICASSP_DATA_ROOT` to the prepared experiment-data root, not directly to an upstream dataset download. Source dataset splits are not automatically the router train/validation/test splits. Follow the split definitions, exclusions, label mapping and evaluation rules in [`paper/supplementary.pdf`](../paper/supplementary.pdf), then the layout and commands in [`DATA.md`](DATA.md) and [`REPRODUCE.md`](REPRODUCE.md). Downloading the source datasets alone does not generate model predictions or hidden-state caches.

Dataset and model licenses remain those of their respective authors. In particular, ConstructionSite requires accepting its access conditions and is distributed under CC BY-NC 4.0.
