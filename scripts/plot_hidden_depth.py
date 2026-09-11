"""Rebuild the three-point hidden-depth comparison from recorded validation scores."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parents[1]
rows=json.loads((root/'results/DSE_STATE.json').read_text())['candidates'][:3]
rows=sorted(rows,key=lambda r:r['settings']['hidden_percent'])
plt.rcParams.update({'font.family':'serif','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(1,5,figsize=(10,2.5))
for ax,key,title in zip(axs,['cub','grefcoco','nlvr2','construction','mean'],['CUB-200-2011','gRefCOCO','NLVR2','ConstructionSite','Task-balanced mean']):
    values=[100*(r['validation_mean'] if key=='mean' else r['task_means'][key]) for r in rows]
    ax.plot([30,60,90],values,'o-',color='#0072B2');ax.set_title(title,fontsize=9);ax.set_xlabel('Hidden depth (%)');ax.set_xticks([30,60,90]);ax.ticklabel_format(axis='y',useOffset=False);ax.grid(alpha=.2)
axs[0].set_ylabel('Validation score (%)');fig.tight_layout()
dest=root/'outputs';dest.mkdir(exist_ok=True);fig.savefig(dest/'hidden_depth.pdf',bbox_inches='tight')
