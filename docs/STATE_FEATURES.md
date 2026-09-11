# Specialist hidden features

`code/state_extraction/state_pooling.py` and `added_state_pooling.py` contain the original pooling and layer-selection logic. They attach to a loaded upstream specialist, select recorded relative depths and immediately pool hidden tensors. Use the specialist's own image preparation and normal forward pass; the collectors do not substitute a new model.

Depth groups used in the shared search:

| Specialist | Group |
|---|---|
| ResNet-50 | residual_blocks |
| GLSim | global_local_encoder |
| GroundingDINO | multimodal_encoder |
| InstanceVG | joint_encoder |
| BEiT3 | multimodal_encoder |
| ViLT | ordered_image_text |
| YOLO26x | feature_blocks |
| RT-DETR-X | query_decoder |

The selected position is the ceiling of relative depth times group length. The 30/60/90% comparison starts from the 4M student with width 512 and learning rate .001. The full layer inventory is required by the search preparation script under `outputs/router_training_cache_complete_20260908/<expert>/layer_inventory.json` in the external data root.

Upstream specialist model implementations and their environments are external dependencies. This repository includes the project's pooling code, not copies of all third-party training frameworks.
