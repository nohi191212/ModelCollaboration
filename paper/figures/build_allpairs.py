from pathlib import Path
import json,fitz,shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1]
groups=json.loads((root/'data/results.json').read_text())['groups']
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman'],'font.size':8,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
for endpoint,suffix in [('MiniCPM','m'),('Qwen3.8','q')]:
 subset=[g for g in groups if g['key'][2]==endpoint]
 fig,axs=plt.subplots(4,4,figsize=(10,8))
 for i,g in enumerate(subset):
  row=2*(i//4);col=i%4
  for method,color in [('learned','#176f91'),('confidence','#c78324')]:
   v=g['methods'][method];curve=v['test_curve'];xs=[p['budget']*100 for p in curve]
   for r,field in [(row,'metric'),(row+1,'actual_fraction')]:
    ax=axs[r,col];ax.plot(xs,[100*p[field] for p in curve],color=color,lw=1.1,ls='-' if method=='learned' else '--',label='LR' if method=='learned' else 'conf.')
    if method=='confidence':b=v['validation_point']['budget']
    else:b=sum(p['budget'] for p in v['test_point']['per_seed'])/5
    ax.scatter(b,100*v['test_point'][field],color=color,s=12);ax.grid(axis='y',alpha=.2);ax.set_xlim(0,100)
  axs[row,col].set_title(g['key'][0]+' / '+g['expert_display'],fontsize=9)
  axs[row+1,col].plot([0,100],[0,100],color='gray',ls=':',lw=.6)
  axs[row+1,col].set_xlabel('Validation budget (%)')
  if col==0:axs[row,col].set_ylabel('Test metric (%)');axs[row+1,col].set_ylabel('Test calls (%)')
 handles,labels=axs[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=2,frameon=False)
 fig.tight_layout(rect=(0,0,1,.96));fig.savefig(root/f'figures/budget_all_{suffix}.pdf');plt.close(fig)
shutil.copyfile(root/'figures/gains_compact.pdf',root/'figures/collaboration_gains.pdf')
print('Updated method and all-pair curves.')
