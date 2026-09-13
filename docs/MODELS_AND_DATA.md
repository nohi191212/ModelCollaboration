# Upstream models and datasets

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


Upstream models and datasets retain their original licenses.
