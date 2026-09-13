"""Validation budget curves for all 16 completed light-stage pairings."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
parser=argparse.ArgumentParser();parser.add_argument('--font',required=True);args=parser.parse_args()
font_manager.fontManager.addfont(args.font)
plt.rcParams.update({'font.family':font_manager.FontProperties(fname=args.font).get_name(),'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'axes.unicode_minus':False})
w=Path('/mnt/HithinkOmniSSD/user_workspace/caisihang/research/ICASSP_ModelCollaboration/outputs/router_exploration_25pct_20260910')
a=json.loads((w/'interaction_light_queue_analysis.json').read_text())
assert a['all_queue_fits_complete'] and not a['warnings']
best={}
for row in a['groups']:
    assert row['expected_seeds']==[42]
    key=(row['config']['expert'],row['endpoint'])
    if key not in best or row['mean']>best[key]['mean']:best[key]=row
names={'cub':'ResNet / CUB','glsim':'GLSim / CUB','groundingdino':'GroundingDINO / gRefCOCO','instancevg':'InstanceVG / gRefCOCO','nlvr2':'BEiT3 / NLVR2','vilt':'ViLT / NLVR2','yolo26x':'YOLO26x / 工地','rtdetr_x_fp32':'RT-DETR-X / 工地'}
dest=w/'report_figures';dest.mkdir(exist_ok=True);metadata=[];latex=[]
for (expert,endpoint),row in sorted(best.items()):
    run=row['runs'][0]['id'];metrics=json.loads((w/'runs'/run/(endpoint+'_metrics.json')).read_text())
    ref=json.loads((w/'references'/(expert+'_'+endpoint+'.json')).read_text())
    x=[p['budget']*100 for p in metrics['curve'][:6]]
    curves={'本轮候选':[p['metric']*100 for p in metrics['curve'][:6]],'置信规则':[p['metric']*100 for p in ref['confidence_curve']['curve'][:6]],'随机调用均值':[p['metric_mean']*100 for p in ref['random_curve_summary'][:6]],'不调用大模型':[ref['small_metric']*100]*6}
    fig,ax=plt.subplots(figsize=(5,3.5),layout='constrained')
    for (label,y),color,marker,style in zip(curves.items(),['#0072B2','#D55E00','#009E73','#555555'],['o','s','^',None],['-','--','-.',':']):
        ax.plot(x,y,label=label,color=color,marker=marker,markersize=4,linestyle=style,linewidth=1.5)
    ax.set_xlabel('大模型调用预算（%）');ax.set_ylabel('事件宏平均F1（%）' if row['config']['task']=='construction' else '严格正确率（%）')
    ax.set_xticks(x);ax.legend(frameon=False,fontsize=8,loc='best')
    name=expert+'_'+endpoint+'_budget'
    fig.savefig(dest/(name+'.pdf'),bbox_inches='tight');fig.savefig(dest/(name+'.png'),dpi=300,bbox_inches='tight');plt.close(fig)
    caption=f"{names[expert]} → {endpoint}。第三批已完成轻量配置中按种子42的0至25%验证曲线面积选择候选；不是独立测试结果或最终方法。随机调用为既有8种子均值，无统计显著性结论；排序预算曲线不是同分情况下等价的固定阈值曲线。"
    metadata.append({'name':name,'caption':caption,'expert':expert,'endpoint':endpoint,'run_id':run,'budgets':x,'curves':curves})
    latex.append('\\begin{figure}[t]\n\\centering\n\\includegraphics[width=0.48\\textwidth]{report_figures/'+name+'.pdf}\n\\caption{'+caption+'}\n\\end{figure}\n')
assert len(metadata)==16
(dest/'figure_data.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2))
(dest/'latex_includes.tex').write_text('\n'.join(latex))
print(json.dumps({'figures':16,'formats':['pdf','png'],'validation_only':True,'gpu_launched':False}))
