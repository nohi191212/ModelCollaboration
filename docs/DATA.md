# External data layout

Start with the [official dataset and model download links](DOWNLOADS.md), including access instructions and checkpoint availability.

`ICASSP_DATA_ROOT` is separate from this source repository. It contains:

```
outputs/router_full_20260910/
  data/<expert>/<train|val|test>/records.jsonl
  data/<expert>/<split>/index.json
  data/<expert>/<split>/confidence.npy
  data/<expert>/<split>/output.npy
  features/<student>/<task>/<split>/{image,text,valid}.npy
  features/<student>/<task>/<split>/sample_ids.json
outputs/mini_siglip_strict_20260910_300ep/<1M|2M|4M|8M>/
  config.json, student_tokenizer.json, epoch_300.pt
data/siglip_distillation_strict_20260910/
```

Each `index.json` entry has `shard` and `position`; each shard contains `states.npz` with the configured hidden-layer key. When moving data, update these shard paths to their new locations. Feature sample IDs must match the records exactly. Image features have two ordered 768-dimensional slots with availability flags. CUB and ConstructionSite use one slot; NLVR2 uses both. Text is 768-dimensional. Hidden, confidence and output-summary widths depend on the specialist.

Acquire CUB-200-2011, gRefCOCO/COCO, NLVR2 and ConstructionSite from their original sources and observe their terms. The prepared records combine task inputs, ground truth and model predictions; they are not a replacement for the source dataset licenses. See the supplement for split sizes, exclusions, matching rules and invalid-output handling.

Exact prompts are in `prompts/`. CUB class IDs and candidate names are in `prompts/classes.json`. The collection scripts expect the recorded full-label collection plan and manifests under `outputs/large_model_labels_all_20260909/`; endpoint settings belong in that local plan, not in the public source tree.
