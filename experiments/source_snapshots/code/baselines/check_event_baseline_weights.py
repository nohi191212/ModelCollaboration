from pathlib import Path

import torch
import ultralytics
from ultralytics import RTDETR, YOLO


paths = {
    "yolo26x": (YOLO, Path("/mnt/HithinkOmniSSD/user_workspace/caisihang/research/ICASSP_ModelCollaboration/models/construction/yolo26x.pt")),
    "rtdetr_x": (RTDETR, Path("/mnt/HithinkOmniSSD/user_workspace/caisihang/research/ICASSP_ModelCollaboration/models/construction/rtdetr-x.pt")),
}
print("ultralytics", ultralytics.__version__)
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "device_count", torch.cuda.device_count())
for name, (model_class, path) in paths.items():
    if not path.is_file():
        raise FileNotFoundError(path)
    model = model_class(path)
    print(name, "weights_ok", "parameters", sum(parameter.numel() for parameter in model.model.parameters()), "names_count", len(model.names), "first_names", list(model.names.items())[:3])
