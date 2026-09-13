from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("--project", type=Path, required=True)
parser.add_argument("--experiment", type=Path, required=True)
parser.add_argument("--groups", type=int, default=12)
parser.add_argument("--seed", type=int, default=20260826)
args = parser.parse_args()

annotation_path = args.project / "data/lvis/annotations/lvis_v1_val.json"
annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
categories = {row["id"]: row for row in annotation["categories"]}
present_by_image: dict[int, set[int]] = defaultdict(set)
for row in annotation["annotations"]:
    present_by_image[row["image_id"]].add(row["category_id"])

candidates = []
for image in annotation["images"]:
    image_id = image["id"]
    positive_ids = sorted(present_by_image[image_id])
    negative_ids = sorted(
        category_id
        for category_id in image["neg_category_ids"]
        if category_id not in present_by_image[image_id]
    )
    if not positive_ids or not negative_ids:
        continue
    url = image["coco_url"]
    split = "train2017" if "/train2017/" in url else "val2017"
    image_path = args.project / "data/lvis/images" / split / url.rsplit("/", 1)[-1]
    if image_path.is_file():
        candidates.append((image_id, image_path, positive_ids, negative_ids))

rng = random.Random(args.seed)
rng.shuffle(candidates)
selected = candidates[: args.groups]
if len(selected) != args.groups:
    raise RuntimeError(f"need {args.groups} image groups, found {len(selected)}")

input_dir = args.experiment / "inputs"
audit_dir = args.experiment / "audits"
input_dir.mkdir(parents=True, exist_ok=True)
audit_dir.mkdir(parents=True, exist_ok=True)
requests_path = input_dir / "requests.jsonl"
labels_path = input_dir / "ground_truth.jsonl"

requests = []
labels = []
for image_id, image_path, positive_ids, negative_ids in selected:
    for category_id, present, source in [
        (rng.choice(positive_ids), True, "lvis_v1_val.annotations"),
        (rng.choice(negative_ids), False, "lvis_v1_val.images.neg_category_ids"),
    ]:
        category = categories[category_id]
        display_name = category["name"].replace("_", " ")
        sample_id = f"lvis-{image_id}-{category_id}"
        requests.append(
            {
                "sample_id": sample_id,
                "task": "lvis_presence",
                "group_id": str(image_id),
                "group_size": 2,
                "image": str(image_path),
                "category_id": category_id,
                "category_name": category["name"],
                "detector_text": f"a {display_name}",
                "question": f"Is there a {display_name} in this image? Answer only yes or no.",
                "query_type": "presence",
            }
        )
        labels.append(
            {
                "sample_id": sample_id,
                "ground_truth": {
                    "present": present,
                    "source": source,
                    "category_id": category_id,
                },
            }
        )

with requests_path.open("w", encoding="utf-8") as handle:
    for row in requests:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
with labels_path.open("w", encoding="utf-8") as handle:
    for row in labels:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")

(audit_dir / "C1-C3_数据准备.md").write_text(
    "\n".join(
        [
            "# C1–C3 数据准备审计",
            "",
            "## 做了什么",
            "",
            f"- 从 `{annotation_path}` 读取 LVIS 官方标注。",
            f"- 先选出 {len(selected)} 张图片，再在每张图片内构造 1 条正查询和 1 条官方明确负查询，共 {len(requests)} 条。",
            "- 模型请求与真值分开保存；模型请求文件不包含目标是否存在的答案。",
            "",
            "## 负例依据",
            "",
            "- 负例只来自 LVIS 图片记录里的 `neg_category_ids`，没有把“未标注类别”当成负例。",
            "",
            "## 输入与输出",
            "",
            f"- 输入：`{annotation_path}`",
            f"- 模型请求：`{requests_path}`",
            f"- 真值：`{labels_path}`",
            f"- 分组单位：图片，共 {len(selected)} 组。",
            f"- 随机种子：{args.seed}",
            "",
        ]
    ),
    encoding="utf-8",
)

print(json.dumps({"groups": len(selected), "queries": len(requests), "requests": str(requests_path), "ground_truth": str(labels_path)}))
