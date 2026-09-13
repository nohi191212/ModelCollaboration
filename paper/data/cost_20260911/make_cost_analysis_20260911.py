"""Build the A800 cost--performance supplement for two representative pairs.

The script reads the frozen test curves and the measured A800 component records.
It does not select a test-set operating point: the selected points are copied from
the validation-locked records, and all curve values retain their nominal budget.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PAIR_FILES = {
    "grefcoco|instancevg|MiniCPM": {
        "label": "gRefCOCO / InstanceVG + MiniCPM",
        "specialist": "instancevg_specialist.json",
        "router": "grefcoco_router.json",
        "large_latency": "minicpm_grefcoco.json",
        "large_flops": "minicpm_grefcoco_flops.json",
    },
    "construction|yolo26x|Qwen3.8": {
        "label": "Construction / YOLO26x + Qwen3.8",
        "specialist": "yolo26x_specialist.json",
        "router": "construction_router.json",
        "large_latency": "qwen_construction.json",
        "large_flops": "qwen_construction_flops.json",
    },
}

METHOD_LABELS = {
    "both": "两项同时改",
    "original": "原方法",
    "confidence": "置信度路由",
}
PLOT_LABELS = {"both": "Two changes", "original": "Original", "confidence": "Confidence"}
PLOT_METHODS = ("both", "original", "confidence")
PLOT_COLORS = {"both": "#2166ac", "original": "#d6604d", "confidence": "#252525"}
PLOT_STYLES = {"both": "-", "original": "--", "confidence": ":"}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def mean_std(values):
    values = np.asarray(values, dtype=float)
    return float(values.mean()), float(values.std(ddof=0))


def router_flops(architecture: dict) -> tuple[int, dict[str, int]]:
    d = int(architecture["representation_dim"])
    branch_dims = architecture["input_dimensions"]
    branch_costs = {}
    for branch in architecture["branches"]:
        input_key = "image_flat_plus_valid" if branch == "image" else branch
        branch_costs[branch] = 2 * int(branch_dims[input_key]) * d
    head_input = len(architecture["branches"]) * d
    branch_costs["head_input_to_hidden"] = 2 * head_input * int(architecture["head_hidden_dim"])
    branch_costs["head_hidden_to_classes"] = 2 * int(architecture["head_hidden_dim"]) * int(architecture["class_count"])
    return sum(branch_costs.values()), branch_costs


def build_cost_components(root: Path, architectures: list[dict]) -> dict[str, dict]:
    components = {}
    for architecture in architectures:
        key = architecture["pair_key"]
        files = PAIR_FILES[key]
        specialist = read_json(root / files["specialist"])
        router = read_json(root / files["router"])
        large_latency = read_json(root / files["large_latency"])
        large_flops = read_json(root / files["large_flops"])
        router_flop_count, router_breakdown = router_flops(architecture)
        completion_tokens = [
            output["completion_tokens"]
            for batch in large_latency["batches"]
            for output in batch["outputs"]
        ]
        prefill = float(large_flops["mean_prefill_gflops_counted"])
        decode = float(large_flops["mean_decode_one_token_gflops_counted"])
        mean_tokens = float(large_latency["mean_completion_tokens"])
        large_estimated = prefill + decode * mean_tokens
        components[key] = {
            "pair_key": key,
            "task": architecture["task"],
            "expert": architecture["expert"],
            "endpoint": architecture["endpoint"],
            "label": PAIR_FILES[key]["label"],
            "specialist_gflops_per_input": float(specialist["mean_counted_gflops_per_sample"]),
            "specialist_forward_latency_ms": float(specialist["mean_forward_latency_ms_per_sample"]),
            "specialist_end_to_end_latency_ms": float(specialist["mean_end_to_end_latency_ms_per_sample"]),
            "specialist_sample_count": int(specialist["sample_count"]),
            "router_gflops_per_input": router_flop_count / 1e9,
            "router_flop_count": router_flop_count,
            "router_flop_breakdown": router_breakdown,
            "router_measured_gflops_per_input": float(router["gflops_counted_per_sample"]),
            "router_latency_ms": float(router["mean_latency_ms_per_sample"]),
            "router_sample_count": int(router["sample_count"]),
            "large_prefill_gflops": prefill,
            "large_decode_one_token_gflops": decode,
            "large_mean_completion_tokens": mean_tokens,
            "large_completion_token_min": int(min(completion_tokens)),
            "large_completion_token_max": int(max(completion_tokens)),
            "large_estimated_full_generation_gflops": large_estimated,
            "large_batch_wall_latency_ms_per_input": float(large_latency["mean_batch_wall_ms_per_sample"]),
            "large_measured_sample_count": int(large_latency["measured_count"]),
            "large_flops_probe_count": int(large_flops["sample_count"]),
            "flops_formula": "specialist + router + actual_call_fraction * (prefill + mean_completion_tokens * one_token_decode)",
            "latency_formula": "specialist end-to-end + router + actual_call_fraction * large-model batch-wall latency",
            "specialist_flops_scope": specialist["flops_scope"],
            "specialist_latency_scope": specialist["latency_scope"],
            "large_flops_scope": large_flops["scope"],
            "large_latency_scope": large_latency["latency_scope"],
        }
    return components


def load_relevant_groups(results: dict, pair_key: str) -> dict[str, list[dict]]:
    task, expert, endpoint = pair_key.split("|")
    groups = {
        method: []
        for method in PLOT_METHODS
    }
    for group in results["groups"]:
        if (group["task"], group["expert"], group["endpoint"]) != (task, expert, endpoint):
            continue
        if group["method"] in groups:
            groups[group["method"]].append(group)
    for method, entries in groups.items():
        if not entries:
            raise ValueError(f"missing {method} group for {pair_key}")
    return groups


def curve_rows(groups: dict[str, list[dict]], pair_key: str, components: dict) -> list[dict]:
    rows = []
    for method, entries in groups.items():
        per_budget = {}
        for group in entries:
            curve = group["deployments"]["endpoint"]["test_curve"]
            for point in curve:
                budget = round(float(point["budget"]), 10)
                per_budget.setdefault(budget, []).append(point)
        for budget in sorted(per_budget):
            points = per_budget[budget]
            metrics = [float(point["metric"]) for point in points]
            calls = [float(point["actual_fraction"]) for point in points]
            metric_mean, metric_std = mean_std(metrics)
            call_mean, call_std = mean_std(calls)
            component = components[pair_key]
            rows.append({
                "pair_key": pair_key,
                "pair_label": component["label"],
                "task": component["task"],
                "method": method,
                "method_label": METHOD_LABELS[method],
                "nominal_budget": budget,
                "seed_count": len(points),
                "test_metric": metric_mean,
                "test_metric_std": metric_std,
                "actual_test_call_fraction": call_mean,
                "actual_test_call_fraction_std": call_std,
                "flops_gflops_per_input": component["specialist_gflops_per_input"] + component["router_gflops_per_input"] + call_mean * component["large_estimated_full_generation_gflops"],
                "latency_ms_per_input": component["specialist_end_to_end_latency_ms"] + component["router_latency_ms"] + call_mean * component["large_batch_wall_latency_ms_per_input"],
            })
    return rows


def selected_rows(groups: dict[str, list[dict]], pair_key: str, components: dict) -> list[dict]:
    rows = []
    component = components[pair_key]
    for method, entries in groups.items():
        points = [entry["deployments"]["endpoint"]["test_point"] for entry in entries]
        metrics = [float(point["metric"]) for point in points]
        calls = [float(point["actual_fraction"]) for point in points]
        budgets = [float(point["budget"]) for point in points]
        metric_mean, metric_std = mean_std(metrics)
        call_mean, call_std = mean_std(calls)
        rows.append({
            "pair_key": pair_key,
            "pair_label": component["label"],
            "task": component["task"],
            "method": method,
            "method_label": METHOD_LABELS[method],
            "seed_count": len(points),
            "selected_nominal_budget_mean": float(np.mean(budgets)),
            "selected_nominal_budgets": budgets,
            "test_metric": metric_mean,
            "test_metric_std": metric_std,
            "actual_test_call_fraction": call_mean,
            "actual_test_call_fraction_std": call_std,
            "flops_gflops_per_input": component["specialist_gflops_per_input"] + component["router_gflops_per_input"] + call_mean * component["large_estimated_full_generation_gflops"],
            "latency_ms_per_input": component["specialist_end_to_end_latency_ms"] + component["router_latency_ms"] + call_mean * component["large_batch_wall_latency_ms_per_input"],
        })
    return rows


def fixed_budget_rows(curves: list[dict], budgets=(0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.99, 1.0)) -> list[dict]:
    wanted = {round(float(budget), 10) for budget in budgets}
    return [row for row in curves if round(float(row["nominal_budget"]), 10) in wanted]


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_plot(curves: list[dict], selected: list[dict], out_path: Path, cost_field: str, x_label: str, log_x: bool) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), constrained_layout=True)
    pair_keys = list(PAIR_FILES)
    for index, (ax, pair_key) in enumerate(zip(axes, pair_keys)):
        pair_rows = [row for row in curves if row["pair_key"] == pair_key]
        all_y = []
        for method in PLOT_METHODS:
            method_rows = [row for row in pair_rows if row["method"] == method]
            x = np.asarray([row[cost_field] for row in method_rows], dtype=float)
            y = 100 * np.asarray([row["test_metric"] for row in method_rows], dtype=float)
            y_std = 100 * np.asarray([row["test_metric_std"] for row in method_rows], dtype=float)
            all_y.extend((y - y_std).tolist())
            all_y.extend((y + y_std).tolist())
            label = PLOT_LABELS[method]
            ax.plot(x, y, PLOT_STYLES[method], color=PLOT_COLORS[method], linewidth=1.25, label=label)
            if method in ("both", "original") and np.any(y_std > 0):
                ax.fill_between(x, y - y_std, y + y_std, color=PLOT_COLORS[method], alpha=0.10, linewidth=0)
            selected_row = next(row for row in selected if row["pair_key"] == pair_key and row["method"] == method)
            ax.scatter(
                [selected_row[cost_field]],
                [100 * selected_row["test_metric"]],
                color=PLOT_COLORS[method],
                marker="*",
                s=42,
                edgecolor="white",
                linewidth=0.45,
                zorder=5,
            )
        y_min, y_max = min(all_y), max(all_y)
        y_pad = max(0.05 * (y_max - y_min), 0.2)
        ax.set_ylim(y_min - y_pad, y_max + 2.0 * y_pad)
        task_label = "gRefCOCO" if PAIR_FILES[pair_key]["label"].startswith("gRefCOCO") else "Construction"
        ax.text(0.02, 0.98, f"({'ab'[index]}) {task_label}", transform=ax.transAxes, ha="left", va="top", fontsize=8.5)
        ax.set_xlabel(x_label)
        ax.set_ylabel("Test performance (%)")
        ax.grid(True, color="#d9d9d9", linewidth=0.45, alpha=0.65)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if log_x:
            ax.set_xscale("log")
        ax.tick_params(labelsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.045), ncol=3, frameon=False, fontsize=8)
    fig.savefig(out_path, format="pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(out_path.with_suffix(".png"), format="png", dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def fmt(value: float, digits=3) -> str:
    return f"{value:.{digits}f}"


def pct(value: float, digits=2) -> str:
    return f"{100 * value:.{digits}f}%"


def report_markdown(components: dict, selected: list[dict], fixed: list[dict], root: Path, summary: dict) -> str:
    selected_by_key = {(row["pair_key"], row["method"]): row for row in selected}
    lines = [
        "# A800成本—性能补充报告（2026-09-11）",
        "",
        "## 1. 做了什么",
        "",
        "- 在A800上补测了正文中两个代表性模型对：gRefCOCO的InstanceVG + MiniCPM，以及Construction的YOLO26x + Qwen3.8。",
        "- 分别实测了小模型端、路由器端和大模型端的延迟与可计数运算量；大模型端延迟采用批处理均摊后的真实墙钟时间。",
        "- 从已经锁定的测试曲线中提取代表性调用预算，按测试集实际调用率换算每个输入的成本，并生成完整曲线。没有用测试集重新挑选模型、阈值或预算。",
        "",
        "## 2. 为什么做",
        "",
        "正文原有成本图只是代理估算，缺少InstanceVG、完整解码和真实延迟。此次补测用于回答一个更直接的问题：两项同时改相对置信度路由的性能提升，是否值得它增加的计算和时间成本。",
        "",
        "## 3. 原始实测成本",
        "",
        "| 模型对 | 小模型端到端延迟（毫秒/样本） | 小模型可计数运算量（GFLOPs/样本） | 路由器延迟（毫秒/样本） | 路由器稠密运算量（GFLOPs/样本） | 大模型调用延迟（毫秒/样本） | 大模型完整生成估算（GFLOPs/调用） |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, component in components.items():
        lines.append(
            f"| {component['label']} | {fmt(component['specialist_end_to_end_latency_ms'])} | {fmt(component['specialist_gflops_per_input'])} | {fmt(component['router_latency_ms'], 6)} | {fmt(component['router_gflops_per_input'], 6)} | {fmt(component['large_batch_wall_latency_ms_per_input'])} | {fmt(component['large_estimated_full_generation_gflops'])} |"
        )
    lines.extend([
        "",
        "说明：GFLOPs按乘加计2次运算，只覆盖计数器能识别的稠密乘法、卷积和注意力等算子；路由器的计数器返回0，因此按已加载检查点的线性层结构精确计算了稠密乘加量。大模型完整生成GFLOPs为“预填充GFLOPs + 平均生成长度 × 单Token解码GFLOPs”的估算，不把未计数算子冒充成完整硬件计数。",
        "",
        "## 4. 验证集锁定点在测试集上的结果",
        "",
        "下表的均值和标准差只在三个随机种子之间统计。成本使用对应的平均实际调用率；星号表示验证集选出的运行点。",
        "",
        "| 模型对 | 方法 | 测试性能 | 实际调用率 | 估算GFLOPs/输入 | 端到端延迟（毫秒/输入） |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for key in components:
        for method in ("both", "original", "confidence"):
            row = selected_by_key[(key, method)]
            metric = f"{100 * row['test_metric']:.2f}±{100 * row['test_metric_std']:.2f}%"
            calls = f"{100 * row['actual_test_call_fraction']:.2f}±{100 * row['actual_test_call_fraction_std']:.2f}%"
            lines.append(f"| {row['pair_label']} | {row['method_label']}* | {metric} | {calls} | {fmt(row['flops_gflops_per_input'])} | {fmt(row['latency_ms_per_input'])} |")
    lines.extend([
        "",
        "### 两项同时改相对置信度路由",
        "",
        "| 模型对 | 性能变化 | 调用率变化 | 估算GFLOPs变化 | 实际延迟变化 |",
        "|---|---:|---:|---:|---:|",
    ])
    for key in components:
        learned = selected_by_key[(key, "both")]
        confidence = selected_by_key[(key, "confidence")]
        metric_delta = 100 * (learned["test_metric"] - confidence["test_metric"])
        call_delta = 100 * (learned["actual_test_call_fraction"] - confidence["actual_test_call_fraction"])
        flop_delta = 100 * (learned["flops_gflops_per_input"] / confidence["flops_gflops_per_input"] - 1)
        latency_delta = 100 * (learned["latency_ms_per_input"] / confidence["latency_ms_per_input"] - 1)
        lines.append(f"| {learned['pair_label']} | {metric_delta:+.2f} 个百分点 | {call_delta:+.2f} 个百分点 | {flop_delta:+.1f}% | {latency_delta:+.1f}% |")
        summary[key]["both_vs_confidence"] = {
            "metric_delta_percentage_points": metric_delta,
            "call_delta_percentage_points": call_delta,
            "flops_relative_percent": flop_delta,
            "latency_relative_percent": latency_delta,
        }
    lines.extend([
        "",
        "直白地说：两项同时改在这两个代表性模型对上都提高了测试性能，同时减少了实际调用，因此计算量和延迟也一起下降。原因不是路由器本身更快，而是它让大模型少被调用；路由器自身成本相对很小。",
        "",
        "## 5. 固定名义调用预算的代表点",
        "",
        "名义预算是验证阶段设定的阈值预算；阈值原样转到测试集后，实际调用率会有偏差。完整的99个阈值点和端点均在CSV中。",
        "",
        "| 模型对 | 名义预算 | 两项同时改：性能 / 实际调用 | 置信度：性能 / 实际调用 | 两项同时改：GFLOPs/输入 | 两项同时改：毫秒/输入 |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for key in components:
        rows = {(row["nominal_budget"], row["method"]): row for row in fixed if row["pair_key"] == key}
        for budget in (0.25, 0.50):
            learned = rows[(budget, "both")]
            confidence = rows[(budget, "confidence")]
            lines.append(
                f"| {components[key]['label']} | {100 * budget:.0f}% | {100 * learned['test_metric']:.2f}% / {100 * learned['actual_test_call_fraction']:.2f}% | {100 * confidence['test_metric']:.2f}% / {100 * confidence['actual_test_call_fraction']:.2f}% | {fmt(learned['flops_gflops_per_input'])} | {fmt(learned['latency_ms_per_input'])} |"
            )
    lines.extend([
        "",
        "## 6. 结果怎么理解",
        "",
        "1. gRefCOCO的InstanceVG + MiniCPM是这两组中相对置信度提升更明显的一组；Construction的YOLO26x + Qwen3.8也有稳定的正向变化。",
        "2. 这里的成本是按真实测试调用率换算的平均每输入成本，不是把名义预算当成实际调用率。",
        "3. 大模型延迟是A800、批量4下的均摊值，模型加载和预热没有计入；实际系统的并发度、批量大小和请求排队会改变墙钟延迟。",
        "4. InstanceVG使用了远端已有的MMCV等价算子兼容路径，因为原生detrex扩展不可用；这不改变本次模型输出链路，但应在复现实验时保留该环境说明。",
        "5. 本补充不修改正文，也不覆盖旧实验结果；它是针对两个代表性模型对的成本实测和成本—性能换算。",
        "",
        "## 7. 文件在哪里",
        "",
        f"- 原始A800测量记录：`{root.as_posix()}`目录下的8个JSON文件。",
        f"- 完整成本—性能曲线数据：`{(root / 'full_cost_performance_curves.csv').as_posix()}`。",
        f"- 固定预算代表点：`{(root / 'fixed_budget_cost_performance.csv').as_posix()}`。",
        f"- FLOPs—性能图：`{(root / 'flops_performance.pdf').as_posix()}`；实际延迟—性能图：`{(root / 'latency_performance.pdf').as_posix()}`。",
        f"- 可复现脚本：`{(root / 'make_cost_analysis_20260911.py').as_posix()}`；插入论文的LaTeX片段：`{(root / 'latex_includes.tex').as_posix()}`。",
    ])
    return "\n".join(lines) + "\n"


def latex_includes() -> str:
    return r"""% A800 cost--performance supplement (generated 2026-09-11)
\begin{figure*}[t]
    \centering
    \includegraphics[width=0.98\textwidth]{dse_results/method_improvement_20260911/cost_analysis_20260911/flops_performance.pdf}
    \caption{A800 counted-FLOPs--performance curves for the two representative model pairs. Each point uses a validation-calibrated nominal budget transferred unchanged to test; the horizontal coordinate uses the resulting actual test call rate. The star marks the validation-selected operating point. Large-model full-generation FLOPs are estimated from measured prefill FLOPs, one-token decode FLOPs and the measured mean completion length; unsupported operations are not included.}
    \label{fig:a800-flops-performance}
\end{figure*}

\begin{figure*}[t]
    \centering
    \includegraphics[width=0.98\textwidth]{dse_results/method_improvement_20260911/cost_analysis_20260911/latency_performance.pdf}
    \caption{A800 end-to-end latency--performance curves. Specialist and router latency are always paid; the large-model term is weighted by the actual test call rate and uses the measured batch-wall latency per input. Model loading and warm-up are excluded.}
    \label{fig:a800-latency-performance}
\end{figure*}
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    root = args.output_dir.resolve()
    experiment_dir = args.experiment_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)

    architectures = read_json(root / "router_architecture.json")
    architecture_keys = {item["pair_key"] for item in architectures}
    if architecture_keys != set(PAIR_FILES):
        raise ValueError("router architecture keys do not match selected pairs")
    results = read_json(experiment_dir / "test_results.json")
    components = build_cost_components(root, architectures)
    all_curves = []
    all_selected = []
    for pair_key in PAIR_FILES:
        groups = load_relevant_groups(results, pair_key)
        all_curves.extend(curve_rows(groups, pair_key, components))
        all_selected.extend(selected_rows(groups, pair_key, components))
    all_fixed = fixed_budget_rows(all_curves)

    write_csv(root / "full_cost_performance_curves.csv", all_curves)
    write_csv(root / "fixed_budget_cost_performance.csv", all_fixed)
    make_plot(all_curves, all_selected, root / "flops_performance.pdf", "flops_gflops_per_input", "Estimated GFLOPs / input", True)
    make_plot(all_curves, all_selected, root / "latency_performance.pdf", "latency_ms_per_input", "End-to-end latency (ms / input)", False)
    (root / "latex_includes.tex").write_text(latex_includes(), encoding="utf-8")

    summary = {
        "status": "complete",
        "protocol": {
            "test_results": str((experiment_dir / "test_results.json").resolve()),
            "curves_use": "validation-calibrated thresholds transferred unchanged to test",
            "cost_uses": "actual test call fraction",
            "flop_counting": "multiply-add counted as 2 FLOPs; unsupported operations excluded",
            "large_generation_flops": "mean prefill + mean completion tokens * one-token decode probe",
            "seed_aggregation": "mean and population standard deviation across three seeds for learned/original methods",
        },
        "components": components,
        "selected_points": all_selected,
        "fixed_budget_points": all_fixed,
    }
    for key in components:
        summary.setdefault("comparisons", {})
        learned = next(row for row in all_selected if row["pair_key"] == key and row["method"] == "both")
        confidence = next(row for row in all_selected if row["pair_key"] == key and row["method"] == "confidence")
        summary["comparisons"][key] = {
            "both_vs_confidence_metric_delta_percentage_points": 100 * (learned["test_metric"] - confidence["test_metric"]),
            "both_vs_confidence_call_delta_percentage_points": 100 * (learned["actual_test_call_fraction"] - confidence["actual_test_call_fraction"]),
            "both_vs_confidence_flops_relative_percent": 100 * (learned["flops_gflops_per_input"] / confidence["flops_gflops_per_input"] - 1),
            "both_vs_confidence_latency_relative_percent": 100 * (learned["latency_ms_per_input"] / confidence["latency_ms_per_input"] - 1),
        }
    write_json(root / "cost_summary.json", summary)
    (root / "成本性能补充报告.md").write_text(report_markdown(components, all_selected, all_fixed, root, summary["comparisons"]), encoding="utf-8")
    print(json.dumps({"status": "complete", "output_dir": str(root), "curve_rows": len(all_curves), "fixed_rows": len(all_fixed)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
