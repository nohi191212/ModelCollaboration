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
model_outputs = {}
for name, path in model_files.items():
    model_outputs[name] = {
        row["sample_id"]: row["model_output"]
        for row in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)
    }

small_models = ["mm_grounding_dino_tiny", "owlv2_base"]
large_models = ["MiniCPM-V-4_5", "Qwen3.8-27B-FP8"]
paired_path = paired_dir / "paired_outputs.jsonl"
summary = {}

with paired_path.open("w", encoding="utf-8") as handle:
    for small_name in small_models:
        for large_name in large_models:
            pair_name = f"{small_name}__to__{large_name}"
            outcomes = Counter()
            expert_correct_count = 0
            generalist_correct_count = 0
            for request in requests:
                sample_id = request["sample_id"]
                ground_truth = labels[sample_id]
                expert = model_outputs[small_name][sample_id]
                generalist = model_outputs[large_name][sample_id]
                expert_correct = expert["pred_label"] == ground_truth["present"]
                generalist_correct = generalist["pred_label"] == ground_truth["present"]
                if expert_correct and generalist_correct:
                    outcome = "both_correct"
                elif expert_correct:
                    outcome = "harm"
                elif generalist_correct:
                    outcome = "help"
                else:
                    outcome = "both_wrong"
                outcomes[outcome] += 1
                expert_correct_count += int(expert_correct)
                generalist_correct_count += int(generalist_correct)
                record = {
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
                    "expert": expert,
                    "adjudicator": generalist,
                    "ground_truth": ground_truth,
                    "expert_correct": expert_correct,
                    "generalist_correct": generalist_correct,
                    "outcome": outcome,
                    "signed_gain": int(generalist_correct) - int(expert_correct),
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            summary[pair_name] = {
                "records": len(requests),
                "expert_accuracy": expert_correct_count / len(requests),
                "generalist_accuracy": generalist_correct_count / len(requests),
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
        f"- `{pair_name}`：小模型正确率 {stats['expert_accuracy']:.4f}，"
        f"大模型正确率 {stats['generalist_accuracy']:.4f}；"
        f"help={outcomes['help']}，harm={outcomes['harm']}，"
        f"都对={outcomes['both_correct']}，都错={outcomes['both_wrong']}。"
    )
(args.experiment / "audits/C6-C7_配对与结果.md").write_text(
    "\n".join(
        [
            "# C6–C7 配对与结果审计",
            "",
            "## 做了什么",
            "",
            "- 按 `sample_id` 对齐两个小模型、两个大模型和官方真值。",
            "- 对每条基础查询构造 2×2 共四个模型对。",
            "- 为每个模型对计算 help、harm、都对、都错和带符号收益。",
            "",
            "## 输入与输出",
            "",
            f"- 基础查询：{len(requests)} 条。",
            f"- 成对记录：{len(requests) * len(small_models) * len(large_models)} 条。",
            f"- 成对输出：`{paired_path}`",
            f"- 汇总：`{summary_path}`",
            "",
            "## 原始结果",
            "",
            *result_lines,
            "",
            "以上只是 24 条基础查询的链路试跑结果，不作为论文最终结论。",
            "",
        ]
    ),
    encoding="utf-8",
)

print(json.dumps({"paired_records": len(requests) * len(small_models) * len(large_models), "output": str(paired_path), "summary": str(summary_path)}))
