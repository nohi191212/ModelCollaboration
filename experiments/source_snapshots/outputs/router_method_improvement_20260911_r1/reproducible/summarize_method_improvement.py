"""Build raw comparison tables, curves, figures, and a Chinese report."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


METHOD_ORDER = [
    'small', 'large', 'confidence', 'original', 'score_gain',
    'checkpoint_perf', 'both', 'score_only_gain',
    'fusion_alpha_0', 'fusion_alpha_0.25', 'fusion_alpha_0.5',
    'fusion_alpha_0.75', 'fusion_alpha_1',
]
TASK_ORDER = ['cub', 'grefcoco', 'nlvr2', 'construction']
EPS = 1e-12


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--previous-results', type=Path, default=None)
    return parser.parse_args()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def compact_method(result):
    method = result['method']
    if method == 'fusion':
        return f"fusion_alpha_{result['alpha']:g}"
    return method


def rows_from_results(results, rule):
    rows = []
    for result in results['groups']:
        method = compact_method(result)
        deployment = result['deployments'][rule]
        point = deployment['test_point']
        validation = deployment['validation_point']
        rows.append({
            'method': method,
            'method_raw': result['method'],
            'method_id': result['method_id'],
            'task': result['task'],
            'expert': result['expert'],
            'endpoint': result['endpoint'],
            'seed': result.get('seed'),
            'pair_key': f"{result['task']}|{result['expert']}|{result['endpoint']}",
            'trajectory_id': result.get('trajectory_id'),
            'budget': validation['budget'],
            'threshold': validation['threshold'],
            'operator': validation['operator'],
            'validation_metric': validation['metric'],
            'validation_actual_fraction': validation['actual_fraction'],
            'test_metric': point['metric'],
            'test_actual_fraction': point['actual_fraction'],
            'rescue': point.get('rescue'),
            'harm': point.get('harm'),
            'test_curve': deployment.get('test_curve', []),
            'validation_curve': deployment.get('validation_point'),
        })
    return rows


def unique_rows(rows):
    result = {}
    for row in rows:
        seed_key = row['seed'] if row['method'] not in ('small', 'large', 'confidence') else None
        result[(row['method'], row['task'], row['expert'], row['endpoint'], seed_key)] = row
    return list(result.values())


def aggregate(rows, keys):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    output = []
    for group_key, members in sorted(grouped.items(), key=lambda item: tuple(str(x) for x in item[0])):
        metrics = np.asarray([member['test_metric'] for member in members], dtype=float)
        calls = np.asarray([member['test_actual_fraction'] for member in members], dtype=float)
        validation_metrics = np.asarray([member['validation_metric'] for member in members], dtype=float)
        output.append({
            **{key: value for key, value in zip(keys, group_key)},
            'n': len(members),
            'test_metric_mean': float(metrics.mean()),
            'test_metric_std': float(metrics.std(ddof=0)),
            'test_call_mean': float(calls.mean()),
            'test_call_std': float(calls.std(ddof=0)),
            'validation_metric_mean': float(validation_metrics.mean()),
            'validation_metric_std': float(validation_metrics.std(ddof=0)),
        })
    return output


def comparison(rows, baseline_method, candidate_method):
    base = {}
    candidate = {}
    for row in rows:
        if row['method'] == baseline_method and row['seed'] is not None:
            base[(row['task'], row['expert'], row['endpoint'], row['seed'])] = row
        if row['method'] == candidate_method and row['seed'] is not None:
            candidate[(row['task'], row['expert'], row['endpoint'], row['seed'])] = row
    pairs = []
    for key, row in sorted(candidate.items()):
        if key not in base:
            raise ValueError(('missing baseline row', baseline_method, candidate_method, key))
        reference = base[key]
        pairs.append({
            'task': key[0], 'expert': key[1], 'endpoint': key[2], 'seed': key[3],
            'pair_key': '|'.join(str(x) for x in key[:3]),
            'baseline_metric': reference['test_metric'],
            'candidate_metric': row['test_metric'],
            'performance_delta': row['test_metric'] - reference['test_metric'],
            'baseline_call': reference['test_actual_fraction'],
            'candidate_call': row['test_actual_fraction'],
            'call_delta': row['test_actual_fraction'] - reference['test_actual_fraction'],
        })
    counts = {
        'total': len(pairs),
        'performance_wins': sum(item['performance_delta'] > EPS for item in pairs),
        'performance_ties': sum(abs(item['performance_delta']) <= EPS for item in pairs),
        'performance_losses': sum(item['performance_delta'] < -EPS for item in pairs),
        'call_less': sum(item['call_delta'] < -EPS for item in pairs),
        'call_ties': sum(abs(item['call_delta']) <= EPS for item in pairs),
        'call_more': sum(item['call_delta'] > EPS for item in pairs),
        'both_performance_win_and_call_less': sum(item['performance_delta'] > EPS and item['call_delta'] < -EPS for item in pairs),
        'both_performance_tie_and_call_less': sum(abs(item['performance_delta']) <= EPS and item['call_delta'] < -EPS for item in pairs),
    }
    return {'baseline': baseline_method, 'candidate': candidate_method, 'counts': counts, 'pairs': pairs}


def previous_reference(path):
    if path is None:
        return {}
    source = load(path)
    reference = {}
    for group in source['groups']:
        key = tuple(group['key'])
        learned = group['methods']['learned']['test_point']
        confidence = group['methods']['confidence']['test_point']
        reference[key] = {
            'metric': learned['metric'],
            'call': learned['actual_fraction'],
            'confidence_metric': confidence['metric'],
            'confidence_call': confidence['actual_fraction'],
        }
    return reference


def write_csv(path, rows):
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def make_curves(output, rows, rule):
    curve_rows = []
    for row in rows:
        if not row['test_curve']:
            continue
        for point in row['test_curve']:
            curve_rows.append({
                'method': row['method'],
                'task': row['task'],
                'expert': row['expert'],
                'endpoint': row['endpoint'],
                'seed': row['seed'],
                'pair_key': row['pair_key'],
                'rule': rule,
                'nominal_budget': point['budget'],
                'actual_test_call_fraction': point['actual_fraction'],
                'test_metric': point['metric'],
                'threshold': point.get('threshold'),
                'operator': point.get('operator'),
            })
    write_csv(output / 'curves' / f'fixed_threshold_test_curves_{rule}.csv', curve_rows)
    return curve_rows


def make_batch_curves(output, results):
    rows = []
    for result in results['groups']:
        if result.get('batch_sorting_diagnostic') is None:
            continue
        for point in result['batch_sorting_diagnostic']:
            rows.append({
                'method': compact_method(result),
                'task': result['task'],
                'expert': result['expert'],
                'endpoint': result['endpoint'],
                'seed': result.get('seed'),
                'pair_key': f"{result['task']}|{result['expert']}|{result['endpoint']}",
                'nominal_budget': point['budget'],
                'actual_test_call_fraction': point['actual_fraction'],
                'test_metric': point['metric'],
            })
    write_csv(output / 'curves' / 'batch_sorting_test_curves.csv', rows)
    return rows


def plot_curves(output, curve_rows, batch_rows):
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    methods = ['confidence', 'original', 'score_gain', 'checkpoint_perf', 'both', 'score_only_gain']
    colors = {
        'confidence': '#6c757d', 'original': '#1f77b4', 'score_gain': '#2ca02c',
        'checkpoint_perf': '#ff7f0e', 'both': '#d62728', 'score_only_gain': '#9467bd',
    }
    for rule in ('historical', 'endpoint'):
        figure, axes = plt.subplots(4, 4, figsize=(16, 14), sharex=False, sharey=False)
        pairs = sorted({(row['task'], row['pair_key']) for row in curve_rows if row['rule'] == rule})
        for axis, (task, pair_key) in zip(axes.flat, pairs[:16]):
            for method in methods:
                subset = [row for row in curve_rows if row['rule'] == rule and row['pair_key'] == pair_key and row['method'] == method and row['seed'] is not None]
                if not subset:
                    continue
                budgets = sorted({row['nominal_budget'] for row in subset})
                x, y = [], []
                for budget in budgets:
                    values = [row for row in subset if row['nominal_budget'] == budget]
                    x.append(float(np.mean([row['actual_test_call_fraction'] for row in values])))
                    y.append(float(np.mean([row['test_metric'] for row in values])))
                axis.plot(x, y, label=method, color=colors.get(method), linewidth=1.2)
            axis.set_title(pair_key.split('|', 1)[-1], fontsize=8)
            axis.set_xlabel('测试实际调用率', fontsize=7)
            axis.set_ylabel('测试任务性能', fontsize=7)
            axis.grid(alpha=.25)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        if handles:
            figure.legend(handles, labels, loc='lower center', ncol=3, fontsize=8)
        figure.suptitle(f'固定阈值迁移后的测试调用率—性能曲线（{rule}）')
        figure.tight_layout(rect=(0, .04, 1, .96))
        figure.savefig(output / f'actual_test_call_vs_performance_{rule}.png', dpi=180)
        plt.close(figure)
    figure, axes = plt.subplots(4, 4, figsize=(16, 14))
    pairs = sorted({row['pair_key'] for row in batch_rows})
    for axis, pair_key in zip(axes.flat, pairs[:16]):
        for method in methods:
            subset = [row for row in batch_rows if row['pair_key'] == pair_key and row['method'] == method and row['seed'] is not None]
            if not subset:
                continue
            budgets = sorted({row['nominal_budget'] for row in subset})
            x = [float(np.mean([row['actual_test_call_fraction'] for row in subset if row['nominal_budget'] == budget])) for budget in budgets]
            axis.plot(budgets, x, label=method, color=colors.get(method), linewidth=1.2)
        axis.set_title(pair_key.split('|', 1)[-1], fontsize=8)
        axis.set_xlabel('预设批量调用率', fontsize=7)
        axis.set_ylabel('测试实际调用率', fontsize=7)
        axis.grid(alpha=.25)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        figure.legend(handles, labels, loc='lower center', ncol=3, fontsize=8)
    figure.suptitle('验证预算迁移到测试后的实际调用率曲线')
    figure.tight_layout(rect=(0, .04, 1, .96))
    figure.savefig(output / 'validation_budget_to_test_call_rate.png', dpi=180)
    plt.close(figure)


def fmt(value, digits=4):
    if value is None:
        return '—'
    return f'{value:.{digits}f}'


def report_text(summary, comparisons, previous_deltas, failure_rows, subgroup_summary):
    lines = []
    lines.append('# 方法改进全量实验结果报告（2026-09-11）')
    lines.append('')
    lines.append('## 1. 做了什么')
    lines.append('')
    lines.append('- 使用四个既有任务、八个专家模型、两个大模型端点，覆盖16组模型对；训练集、验证集、测试集严格分开。')
    lines.append('- 固定8M蒸馏编码器、隐藏层相对深度90%、表示宽度128、拼接结构、四状态交叉熵、自然采样、学习率0.003、批量256、最多50轮、早停耐心8轮。')
    lines.append('- 42、43、44三个随机种子。每个模型对只训练一条完全相同的路由器轨迹，再按四个预先规定的检查点标准独立选择，形成192个逻辑训练条件。')
    lines.append('- 四个训练条件：原方法、只改分数、只改检查点选择、两项同时改；另做同一原方法检查点只换分数的对照。')
    lines.append('- 阈值只在验证集选择，分别报告历史1%—99%口径和补齐0%/100%端点口径；测试只做锁定后的最终评估。')
    lines.append('- 在“两项同时改”检查点上测试五个固定融合系数，并按四任务等权、三种子共同验证规则各锁定一个统一系数。')
    lines.append('')
    lines.append('## 2. 为什么做')
    lines.append('')
    lines.append('当前方法在16组中有11组性能更高、15组调用更少、10组同时满足。此次改动分别检验：性能损益排序是否应使用概率差、检查点是否应直接看任务性能、两者叠加是否稳定，以及融合置信度后能否扩大覆盖面。')
    lines.append('')
    lines.append('## 3. 结果如何')
    lines.append('')
    lines.append('以下数值均用未四舍五入结果统计；表中性能为测试集最终混合预测性能，调用为测试集实际调用率。')
    lines.append('')
    lines.append('### 3.1 三个种子的总体结果')
    lines.append('')
    lines.append('| 口径 | 方法 | 测试性能均值 | 测试性能标准差 | 实际调用率均值 | 实际调用率标准差 | 组数 |')
    lines.append('|---|---|---:|---:|---:|---:|---:|')
    for row in summary['method_overall']:
        lines.append(f"| {row['rule']} | {row['method']} | {fmt(row['test_metric_mean'])} | {fmt(row['test_metric_std'])} | {fmt(row['test_call_mean'])} | {fmt(row['test_call_std'])} | {row['n']} |")
    lines.append('')
    lines.append('### 3.2 每个随机种子：16组均值')
    lines.append('')
    lines.append('| 口径 | 方法 | 种子 | 性能均值 | 性能标准差 | 调用率均值 | 调用率标准差 |')
    lines.append('|---|---|---:|---:|---:|---:|---:|')
    for row in summary['method_seed']:
        lines.append(f"| {row['rule']} | {row['method']} | {row['seed']} | {fmt(row['test_metric_mean'])} | {fmt(row['test_metric_std'])} | {fmt(row['test_call_mean'])} | {fmt(row['test_call_std'])} |")
    lines.append('')
    lines.append('### 3.3 相对原方法的改善、持平和退步')
    lines.append('')
    lines.append('| 口径 | 方法 | 性能胜出 | 性能持平 | 性能退步 | 调用更少 | 二者同时满足 |')
    lines.append('|---|---|---:|---:|---:|---:|---:|')
    for item in comparisons:
        c = item['counts']
        lines.append(f"| {item['rule']} | {item['candidate']} | {c['performance_wins']}/{c['total']} | {c['performance_ties']}/{c['total']} | {c['performance_losses']}/{c['total']} | {c['call_less']}/{c['total']} | {c['both_performance_win_and_call_less']}/{c['total']} |")
    lines.append('')
    lines.append('### 3.4 五个融合系数')
    lines.append('')
    for rule, item in summary['fusion_alpha_selection'].items():
        lines.append(f"- {rule}口径按验证集锁定的统一系数：`alpha={item['selected_alpha']:g}`。选择依据是四任务等权验证性能；同分时先看验证调用更少，再选更小系数。")
    lines.append('')
    lines.append('| 口径 | 系数 | 验证综合性能 | 验证综合调用率 |')
    lines.append('|---|---:|---:|---:|')
    for rule, item in summary['fusion_alpha_selection'].items():
        for candidate in item['candidates'].values():
            lines.append(f"| {rule} | {candidate['alpha']:g} | {fmt(candidate['equal_task_mean_validation_metric'])} | {fmt(candidate['equal_task_mean_validation_actual_fraction'])} |")
    lines.append('')
    lines.append('### 3.5 42号种子与上一版历史结果')
    lines.append('')
    if previous_deltas:
        lines.append('| 模型对 | 新原方法性能变化 | 新原方法调用变化 |')
        lines.append('|---|---:|---:|')
        for row in previous_deltas:
            lines.append(f"| {row['pair_key']} | {row['metric_delta']:+.6f} | {row['call_delta']:+.6f} |")
    else:
        lines.append('未提供上一版结果文件，因此这里只保留本轮42号种子数据。')
    lines.append('')
    lines.append('### 3.6 原先五组失败的专门检查')
    lines.append('')
    lines.append('| 模型对 | 本轮原方法 | 只改分数 | 只改检查点 | 两项同时改 |')
    lines.append('|---|---:|---:|---:|---:|')
    for row in failure_rows:
        lines.append(f"| {row['label']} | {fmt(row['original'])} | {fmt(row['score_gain'])} | {fmt(row['checkpoint_perf'])} | {fmt(row['both'])} |")
    lines.append('')
    lines.append('GroundingDINO 的 testA/testB、单目标/多目标/无目标分组结果保存在单独的分组文件中；分组只做描述，没有用测试分组重新调阈值。')
    lines.append('')
    lines.append('## 4. 结论')
    lines.append('')
    lines.append('1. “改善”只按测试性能严格高于原方法计数，“持平”单独计数；没有把持平算成胜出。')
    lines.append('2. 统一融合系数的推荐与否，以验证集预先锁定结果为准；测试集只用于报告，不反向挑选模型、预算或系数。')
    lines.append('3. 小幅差异要结合三个种子的波动看，不能仅凭一个模型对的微小变化宣称显著。')
    lines.append('4. 本实验沿用既有划分和已经看过的测试结果，属于既有基准上的方法开发验证，不是全新的盲测。')
    lines.append('')
    lines.append('## 5. 文件在哪里')
    lines.append('')
    lines.append('- 完整远端结果目录：`outputs/router_method_improvement_20260911/`。')
    lines.append('- 完整训练轨迹、192个条件的配置/检查点/训练日志、验证与测试四维输出、路由分数、样本编号和选择掩码均在该目录。')
    lines.append('- 原始汇总表：`test_results.json`；验证锁定记录：`validation_manifest.json`、`fusion_alpha_selection.json`。')
    lines.append('- 曲线数据在 `curves/`，图在 `figures/`，GroundingDINO 描述性分组在 `grefcoco_subgroups.json`。')
    lines.append('- 可复现脚本与固定协议在本报告同目录的脚本文件和 `method_improvement_spec.json`。')
    return '\n'.join(lines) + '\n'


def main():
    args = parse_args()
    output = args.output
    results = load(output / 'test_results.json')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'curves').mkdir(exist_ok=True)
    (output / 'figures').mkdir(exist_ok=True)
    summary = {
        'method_overall': [],
        'method_seed': [],
        'method_task': [],
        'fusion_alpha_selection': results['fusion_alpha_selection'],
    }
    all_rows = []
    rows_by_rule = {}
    for rule in ('historical', 'endpoint'):
        current = rows_from_results(results, rule)
        current = unique_rows(current)
        rows_by_rule[rule] = current
        all_rows.extend([{**row, 'rule': rule} for row in current])
        for aggregate_row in aggregate(current, ['method']):
            aggregate_row['rule'] = rule
            summary['method_overall'].append(aggregate_row)
        for aggregate_row in aggregate([row for row in current if row['seed'] is not None], ['method', 'seed']):
            aggregate_row['rule'] = rule
            summary['method_seed'].append(aggregate_row)
        for aggregate_row in aggregate([row for row in current if row['seed'] is not None], ['method', 'task']):
            aggregate_row['rule'] = rule
            summary['method_task'].append(aggregate_row)
    comparisons = []
    for rule, current in rows_by_rule.items():
        for method in METHOD_ORDER:
            if method in ('small', 'large', 'original', 'fusion_alpha_0'):
                continue
            if not any(row['method'] == method and row['seed'] is not None for row in current):
                continue
            item = comparison(current, 'original', method)
            item['rule'] = rule
            comparisons.append(item)
        confidence = [row for row in current if row['method'] == 'confidence']
        if confidence:
            base = {(row['task'], row['expert'], row['endpoint']): row for row in current if row['method'] == 'original' and row['seed'] is not None}
            pairs = []
            for row in confidence:
                for seed in (42, 43, 44):
                    reference = base[(row['task'], row['expert'], row['endpoint'])]
                    pairs.append({
                        'task': row['task'], 'expert': row['expert'], 'endpoint': row['endpoint'], 'seed': seed,
                        'pair_key': row['pair_key'],
                        'baseline_metric': reference['test_metric'], 'candidate_metric': row['test_metric'],
                        'performance_delta': row['test_metric'] - reference['test_metric'],
                        'baseline_call': reference['test_actual_fraction'], 'candidate_call': row['test_actual_fraction'],
                        'call_delta': row['test_actual_fraction'] - reference['test_actual_fraction'],
                    })
            comparisons.append({
                'rule': rule, 'baseline': 'original', 'candidate': 'confidence',
                'counts': {
                    'total': len(pairs),
                    'performance_wins': sum(item['performance_delta'] > EPS for item in pairs),
                    'performance_ties': sum(abs(item['performance_delta']) <= EPS for item in pairs),
                    'performance_losses': sum(item['performance_delta'] < -EPS for item in pairs),
                    'call_less': sum(item['call_delta'] < -EPS for item in pairs),
                    'call_ties': sum(abs(item['call_delta']) <= EPS for item in pairs),
                    'call_more': sum(item['call_delta'] > EPS for item in pairs),
                    'both_performance_win_and_call_less': sum(item['performance_delta'] > EPS and item['call_delta'] < -EPS for item in pairs),
                    'both_performance_tie_and_call_less': sum(abs(item['performance_delta']) <= EPS and item['call_delta'] < -EPS for item in pairs),
                },
                'pairs': pairs,
            })
    reference = previous_reference(args.previous_results)
    previous_deltas = []
    if reference:
        new42 = {(row['task'], row['expert'], row['endpoint']): row for row in rows_by_rule['historical'] if row['method'] == 'original' and row['seed'] == 42}
        for key, row in sorted(new42.items()):
            old = reference.get(key)
            if old is not None:
                previous_deltas.append({
                    'pair_key': '|'.join(key),
                    'metric_delta': row['test_metric'] - old['metric'],
                    'call_delta': row['test_actual_fraction'] - old['call'],
                })
    labels = {
        ('cub', 'cub', 'Qwen3.8'): 'CUB ResNet/Qwen',
        ('grefcoco', 'groundingdino', 'MiniCPM'): 'gRefCOCO GroundingDINO/MiniCPM',
        ('grefcoco', 'groundingdino', 'Qwen3.8'): 'gRefCOCO GroundingDINO/Qwen',
        ('nlvr2', 'nlvr2', 'MiniCPM'): 'NLVR2 BEiT3/MiniCPM',
        ('construction', 'rtdetr_x_fp32', 'Qwen3.8'): '工地 RT-DETR-X/Qwen',
    }
    failure_rows = []
    current = rows_by_rule['historical']
    for key, label in labels.items():
        row = {'label': label}
        for method in ('original', 'score_gain', 'checkpoint_perf', 'both'):
            values = [item['test_metric'] for item in current if item['method'] == method and item['task'] == key[0] and item['expert'] == key[1] and item['endpoint'] == key[2]]
            row[method] = float(np.mean(values)) if values else None
        failure_rows.append(row)
    subgroup_summary = load(output / 'grefcoco_subgroups.json') if (output / 'grefcoco_subgroups.json').exists() else []
    curve_rows = []
    for rule in ('historical', 'endpoint'):
        curve_rows.extend(make_curves(output, rows_by_rule[rule], rule))
    batch_rows = make_batch_curves(output, results)
    plot_curves(output, curve_rows, batch_rows)
    summary['comparisons'] = comparisons
    summary['previous_deltas'] = previous_deltas
    summary['failure_rows'] = failure_rows
    summary['grefcoco_subgroups'] = subgroup_summary
    summary['raw_rows'] = all_rows
    write_json = lambda path, value: Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    write_json(output / 'summary.json', summary)
    write_json(output / 'comparisons.json', comparisons)
    (output / '方法改进结果报告.md').write_text(report_text(summary, comparisons, previous_deltas, failure_rows, subgroup_summary), encoding='utf-8')
    print('REPORT_COMPLETE', output / '方法改进结果报告.md', flush=True)


if __name__ == '__main__':
    main()
