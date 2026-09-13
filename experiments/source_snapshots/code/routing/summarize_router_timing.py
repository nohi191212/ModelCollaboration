"""Keep router-only timing and calibrated online validation results distinct."""
import json
from pathlib import Path
w=Path('/mnt/HithinkOmniSSD/user_workspace/caisihang/research/ICASSP_ModelCollaboration/outputs/router_exploration_25pct_20260910')
assert (w/'router_timing/master.exit').read_text().strip()=='0'
calibration=json.loads((w/'online_validation_calibration/summary.json').read_text())
assert calibration['status']=='passed'
rows=[]
for i in range(4):
    report=json.loads((w/'router_timing'/f'worker{i}.json').read_text())
    assert report['status']=='complete'
    rows+=report['rows']
assert len(rows)==16
names={'cub':'ResNet','glsim':'GLSim','yolo26x':'YOLO26x','rtdetr_x_fp32':'RT-DETR-X','groundingdino':'GroundingDINO','instancevg':'InstanceVG','nlvr2':'NLVR2专用模型','vilt':'ViLT'}
text='# 路由器耗时与现场阈值核对\n\n## 1、做了什么\n\n四张A800每卡一个计时进程，测16个冻结候选。每种设置预热一遍、再测三遍取中位数；单条请求测128条均匀抽取的验证样本，批量16测完整验证集。没有训练。\n\n## 2、计时包括什么\n\n计入预处理后像素读取、需要的图文编码、路由判断、输出复制回CPU。隐藏状态、置信度和回答特征已在显存；模型加载、图像解码缩放、分词准备、大小模型推理、网络均不计入。没有图文输入的候选只测判断头。**这不是整套系统的综合推理速度，也不是测试集耗时。**\n\n## 3、测量结果\n\n| 专用模型 | 大模型 | 实测路径 | 加载参数量（百万） | 单条毫秒 | 批量16毫秒/样本 |\n|---|---|---|---:|---:|---:|\n'
for r in sorted(rows,key=lambda r:(r['expert'],r['endpoint'])):
    m={x['batch_size']:x for x in r['measurements'] if x['mode']=='pixel_encoder_and_head'}
    path='图文编码＋判断头' if r['student_loaded_parameters'] else '仅判断头'
    text+=f"| {names[r['expert']]} | {r['endpoint']} | {path} | {(r['router_parameters']+r['student_loaded_parameters'])/1e6:.3f} | {m[1]['median_ms_per_row']:.3f} | {m[16]['median_ms_per_row']:.3f} |\n"
text+='\n参数量为当前实现实际加载的编码器和判断头参数，不是大小模型总参数量；编码器中未使用的分支尚未裁剪。各进程共享CPU和存储，不能解释为完全隔离条件下的速度上限。\n\n### 缓存阈值不能直接冒充现场阈值\n\n现场批量编码与原缓存批量不同，6个带图文候选中3个出现调用决定变化。保留旧结果，另用第一次现场验证分数重新确定阈值，再在第二次独立前向中原样应用，没有添加容差或调整权重。\n\n| 专用模型 | 大模型 | 现场阈值实际调用 | 现场验证指标×100 | 第二次决定变化总数（各预算） |\n|---|---|---:|---:|---:|\n'
for r in calibration['rows']:
    p=r['calibrated_budget25']
    text+=f"| {names[r['expert']]} | {r['endpoint']} | {p['actual_fraction']*100:.2f}% | {p['metric']*100:.3f} | {sum(c['decision_changes'] for c in r['checks'])} |\n"
text+='\n这验证的是固定批量16、相同设备、相同样本顺序下的重复性，不是新测试成绩。未来数据、批量、设备和顺序变化仍可能改变分数和调用比例；固定阈值不保证未来数据的硬预算。此表与旧缓存图表分别保留，不静默替换原结果。\n\n## 4、文件在哪里\n\n`router_timing/worker*.json` 保存每次计时；`online_validation_calibration/summary.json` 和逐候选阈值文件保存校准及重复检查。真实整套系统成本仍未测完。\n'
(w/'router_timing_summary.md').write_text(text)
print(json.dumps({'timed_candidates':len(rows),'calibrated_candidates':len(calibration['rows']),'system_latency_complete':False}))
