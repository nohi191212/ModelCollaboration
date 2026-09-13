from pathlib import Path
import csv
import json
import fitz

root = Path(__file__).resolve().parent
summary = json.loads((root/'cost_summary.json').read_text())
groups = json.loads((root/'main_table_results.json').read_text())['groups']
rows = list(csv.DictReader((root/'full_cost_performance_curves.csv').open()))
assert len(rows) == 404
lines = ['# 主表方法：成本—性能曲线', '', '## 1 做了什么', '',
    '- 使用主表原检查点、种子42及原验证集阈值；未重新训练，也未改用后续改进方法。',
    '- 在 A800 的0、1号卡测量两个模型对；每个大模型预热2张、正式测量64张，每次1张。专家和路由部分各测64张。',
    '- 重算原路由头，并核对全测试集的选定阈值调用决定与主表完全一致。性能数值沿用主表固定的预测记录。',
    '- 只画学习路由与置信度路由；共404个曲线点。另提供单栏四面板紧凑版。', '',
    '## 2 为什么做', '', '让成本曲线与主实验比较同一个方法，避免混入后续改进结果；补计蒸馏编码器及路由头开销。', '',
    '## 3 做完有什么效果', '', '| 数据集 | 方法 | 测试性能（%） | 调用率（%） | 估算GFLOPs/输入 | 估算毫秒/输入 |',
    '|---|---|---:|---:|---:|---:|']
comparisons = []
for group in groups:
    key = '|'.join(group['key'])
    chosen = {p['method']:p for p in summary['selected_points'] if p['pair']==key}
    for method,label in [('learned','学习路由'),('confidence','置信度路由')]:
        p = chosen[method]
        original = group['methods'][method]['test_point']
        assert p['test_metric'] == original['metric']
        assert p['actual_test_call_fraction'] == original['actual_fraction']
        lines.append(f"| {group['key'][0]} | {label} | {100*p['test_metric']:.2f} | {100*p['actual_test_call_fraction']:.2f} | {p['estimated_gflops_per_input']:,.1f} | {p['estimated_time_ms_per_input']:,.1f} |")
    a,b = chosen['learned'],chosen['confidence']
    comparisons.append(f"{group['key'][0]}：相对置信度路由，性能提高 {100*(a['test_metric']-b['test_metric']):.2f} 个百分点，估算 FLOPs 减少 {100*(1-a['estimated_gflops_per_input']/b['estimated_gflops_per_input']):.1f}%，估算耗时减少 {100*(1-a['estimated_time_ms_per_input']/b['estimated_time_ms_per_input']):.1f}%。")
for comparison in comparisons:
    lines += ['', comparison]
lines += ['']
lines += ['### 数值口径', '',
    '- 每个曲线点使用验证集确定的阈值，横轴按该阈值在测试集上的实际调用率计算；圆点对应主表选定点，没有按测试集重新选点。',
    '- 置信度路由成本 = 专家成本 + 实际调用率 × 大模型平均成本。学习路由再加编码器和路由头成本。',
    '- 耗时为分项实测后相加的串行估算，并非各策略整套系统的直接端到端计时。加载模型和预热不计入。',
    '- FLOPs 按乘加计2次统计支持的算子；大模型用8个输入估计预填充成本（含视觉编码），再加平均输出长度乘单词元解码成本。专家统计4个输入，编码器与路由头统计8个输入。',
    '- 各策略共用大模型平均单次成本，尚未测量所选样本的长度差异对成本的影响；专家隐藏特征来自缓存，其额外提取开销未单独测量。置信度阈值判断开销未单独计时。',
    '- 本轮单张测量与上一轮批量测量口径不同，绝对耗时不宜直接比较。', '',
    '- YOLO 使用权重记录的依赖版本8.3.222，安装在任务临时目录；结束时0、1号卡显存均为0，未操作2、3号卡服务。', '',
    '## 4 对应文件在哪', '',
    '- `cost_compact.pdf`：推荐插入正文的单栏四面板图，宽3.44英寸、高2.05英寸。',
    '- `flops_performance.pdf`、`latency_performance.pdf`：两张独立双面板图。',
    '- `full_cost_performance_curves.csv`：全部404个点；`selected_points.csv`：主表4个运行点。',
    '- `components/`：8份实测记录，含原始回答、错误记录及路由核验。',
    '- `build_figures.py`：绘图及成本计算；`reproducible/`：远端测量脚本；`latex_includes.tex`：插入代码及图注。',
    '- 远端结果：`/mnt/HithinkOmniSSD/user_workspace/caisihang/research/ICASSP_ModelCollaboration/outputs/main_table_cost_20260912`。', '',
    '### 检查记录', '', '已核对404个曲线点、4个主表运行点、两组全测试集调用决定及单位。PDF文字为嵌入字体，另渲染检查排版。']
(root/'成本性能报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
for stem in ['cost_compact','flops_performance','latency_performance']:
    doc = fitz.open(root/(stem+'.pdf'))
    assert len(doc)==1
    assert all(font[1] != 'n/a' for font in doc[0].get_fonts())
    doc[0].get_pixmap(matrix=fitz.Matrix(4,4)).save(root/(stem+'_verified.png'))
print('\n'.join(lines))
