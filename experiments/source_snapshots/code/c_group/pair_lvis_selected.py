from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("--experiment", type=Path, required=True)
args = parser.parse_args()

requests_path = args.experiment / "inputs/requests.jsonl"
labels_path = args.experiment / "inputs/ground_truth.jsonl"
output_dir = args.experiment / "model_outputs"
paired_dir = args.experiment / "paired"
paired_dir.mkdir(parents=True, exist_ok=True)

requests = [json.loads(line) for line in requests_path.read_text(encoding="utf-8").splitlines() if line]
labels = {
    row["sample_id"]: row["ground_truth"]
    for row in (json.loads(line) for line in labels_path.read_text(encoding="utf-8").splitlines() if line)
}
model_files = {
    "mm_grounding_dino_tiny": output_dir / "mm_grounding_dino_tiny.jsonl",
    "owlv2_base": output_dir / "owlv2_base.jsonl",
    "MiniCPM-V-4_5": output_dir / "minicpm_v45.jsonl",
    "Qwen3.8-27B-FP8": output_dir / "qwen38_27b_fp8.jsonl",
}
model_outputs = {
    name: {
        row["sample_id"]: row["model_output"]
        for row in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)
    }
    for name, path in model_files.items()
}
pairs = [
    ("owlv2_base", "Qwen3.8-27B-FP8"),
    ("mm_grounding_dino_tiny", "MiniCPM-V-4_5"),
]

paired_path = paired_dir / "paired_outputs.jsonl"
summary = {}
with paired_path.open("w", encoding="utf-8") as handle:
    for small_name, large_name in pairs:
        pair_name = f"{small_name}__to__{large_name}"
        outcomes = Counter()
        small_correct_count = 0
        large_correct_count = 0
        for request in requests:
            sample_id = request["sample_id"]
            ground_truth = labels[sample_id]
            small = model_outputs[small_name][sample_id]
            large = model_outputs[large_name][sample_id]
            small_correct = small["pred_label"] == ground_truth["present"]
            large_correct = large["pred_label"] == ground_truth["present"]
            if small_correct and large_correct:
                outcome = "both_correct"
            elif small_correct:
                outcome = "harm"
            elif large_correct:
                outcome = "help"
            else:
                outcome = "both_wrong"
            outcomes[outcome] += 1
            small_correct_count += int(small_correct)
            large_correct_count += int(large_correct)
            handle.write(
                json.dumps(
                    {
                        "sample_id": f"{sample_id}__{pair_name}",
                        "base_sample_id": sample_id,
                        "task": request["task"],
                        "group_id": request["group_id"],
                        "group_size": request["group_size"],
                        "model_pair": pair_name,
                        "input": {
                            "image": request["image"],
                            "category_id": request["category_id"],
                            "category_name": request["category_name"],
                            "question": request["question"],
                        },
                        "expert": small,
                        "adjudicator": large,
                        "ground_truth": ground_truth,
                        "expert_correct": small_correct,
                        "generalist_correct": large_correct,
                        "outcome": outcome,
                        "signed_gain": int(large_correct) - int(small_correct),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        summary[pair_name] = {
            "records": len(requests),
            "expert_accuracy": small_correct_count / len(requests),
            "generalist_accuracy": large_correct_count / len(requests),
            "outcomes": {
                name: outcomes[name]
                for name in ["help", "harm", "both_correct", "both_wrong"]
            },
        }

summary_path = paired_dir / "summary.json"
summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
result_lines = []
for pair_name, stats in summary.items():
    outcomes = stats["outcomes"]
    result_lines.append(
        f"- `{pair_name}`：小模型正确率 {stats['expert_accuracy']:.6f}，"
        f"大模型正确率 {stats['generalist_accuracy']:.6f}；"
        f"help={outcomes['help']}，harm={outcomes['harm']}，"
        f"都对={outcomes['both_correct']}，都错={outcomes['both_wrong']}。"
    )
(args.experiment / "audits/全量配对与结果.md").write_text(
    "\n".join(
        [
            "# LVIS 全量配对与结果审计",
            "",
            "## 做了什么",
            "",
            "- 按 `sample_id` 对齐四个模型和官方真值。",
            "- 只生成用户指定的两个搭档：OWLv2→Qwen3.8-VL、GroundingDINO→MiniCPM。",
            "- 逐条计算 help、harm、都对、都错和带符号收益。",
            "",
            "## 输入与输出",
            "",
            f"- 基础查询：{len(requests)} 条。",
            f"- 成对记录：{len(requests) * len(pairs)} 条。",
            f"- 成对输出：`{paired_path}`",
            f"- 汇总：`{summary_path}`",
            "",
            "## 原始结果",
            "",
            *result_lines,
            "",
        ]
    ),
    encoding="utf-8",
)

print(
    json.dumps(
        {
            "paired_records": len(requests) * len(pairs),
            "output": str(paired_path),
            "summary": str(summary_path),
        }
    )
)
