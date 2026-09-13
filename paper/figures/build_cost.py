from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator, LogLocator, NullFormatter
root=Path(__file__).resolve().parents[1]
groups=json.loads((root/'data/results.json').read_text())['groups']
local=root/'data/cost_components.json'
if not local.exists(): local.write_text((root.parents[1]/'dse_results/main_table_cost_qwen_20260912/cost_summary.json').read_text())
components=json.loads(local.read_text())['components']
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman'],'font.size':7,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,'pdf.fonttype':42,'mathtext.fontset':'stix'})
fig,axes=plt.subplots(2,2,figsize=(3.44,2.05))
fig.subplots_adjust(left=.15,right=.985,top=.83,bottom=.15,hspace=.54,wspace=.42)
records=[];selected=[]
for col,(key,c) in enumerate(components.items()):
 g=next(g for g in groups if '|'.join(g['key'])==key)
 for method,color,label in [('learned','#176f91','Learned'),('confidence','#c78324','conf.')]:
  v=g['methods'][method]; overhead=int(method=='learned')
  for row,unit in enumerate(['gflops','ms']):
   ax=axes[row,col];curve=v['test_curve'];calls=np.array([p['actual_fraction'] for p in curve]);ys=np.array([100*p['metric'] for p in curve]);xs=c['specialist_'+unit]+overhead*c['router_'+unit]+calls*c['large_'+unit]
   if unit=='ms':xs=xs/1000
   ax.plot(xs,ys,color=color,lw=1.0,ls='-' if overhead else '--',label=label)
   if overhead:
    sd=np.array([100*p.get('metric_std',0) for p in curve]);ax.fill_between(xs,ys-sd,ys+sd,color=color,alpha=.13,lw=0)
   p=v['test_point'];xp=c['specialist_'+unit]+overhead*c['router_'+unit]+p['actual_fraction']*c['large_'+unit]
   if unit=='ms':xp/=1000
   ax.scatter(xp,100*p['metric'],s=14,color=color,edgecolor='white',lw=.35,zorder=5,clip_on=False)
   ax.yaxis.set_major_locator(MaxNLocator(3));ax.tick_params(labelsize=6,pad=1.5);ax.grid(axis='y',color='#dce1e4',lw=.4)
   if row==0:
    ax.set_xscale('log');ax.set_xlabel('GFLOPs / input',fontsize=7,labelpad=1);ax.xaxis.set_major_locator(LogLocator(base=10,numticks=4));ax.xaxis.set_minor_formatter(NullFormatter())
   else:ax.set_xlabel('Est. seconds / input',fontsize=7,labelpad=1);ax.xaxis.set_major_locator(MaxNLocator(3))
   if col==0:ax.set_ylabel('Test metric (%)',fontsize=7,labelpad=2)
  for ispoint,points in [(False,v['test_curve']),(True,[v['test_point']])]:
   for p in points:
    r=dict(pair=key,method=method,metric=p['metric'],calls=p['actual_fraction'],gflops=c['specialist_gflops']+overhead*c['router_gflops']+p['actual_fraction']*c['large_gflops'],ms=c['specialist_ms']+overhead*c['router_ms']+p['actual_fraction']*c['large_ms'])
    (selected if ispoint else records).append(r)
 axes[0,col].set_title(['InstanceVG / M','YOLO26x / Q'][col],fontsize=7.5,pad=3)
handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.56,1.015),ncol=2,frameon=False,fontsize=7,handlelength=1.8)
fig.savefig(root/'figures/cost_compact.pdf');fig.savefig(root/'figures/cost_compact.png',dpi=300)
(root/'data/cost_five_seeds.json').write_text(json.dumps(dict(components=components,selected=selected,curves=records,protocol='Five-seed mean call rates times existing measured mean component costs; not integrated latency measurements.'),indent=2))
for key in components:
 l,c=[p for p in selected if p['pair']==key]
 print(key,'gain',100*(l['metric']-c['metric']),'FLOPs saving',100*(1-l['gflops']/c['gflops']),'time saving',100*(1-l['ms']/c['ms']))
