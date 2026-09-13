"""Main-table curves plus fresh measured component costs; no threshold re-selection."""
from pathlib import Path
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator,LogLocator,NullLocator

root=Path(__file__).resolve().parent
groups=json.loads((root/'main_table_results.json').read_text(encoding='utf-8'))['groups']
components={}
curves=[]
selected=[]
for g,tag in zip(groups,['qwen_gref','qwen']):
    expert=g['key'][1]
    spec=json.loads((root/f'components/{expert}_specialist.json').read_text())
    router=json.loads((root/f'components/{expert}_router.json').read_text())
    large=json.loads((root/f'components/{tag}_latency.json').read_text())
    flops=json.loads((root/f'components/{tag}_flops.json').read_text())
    assert spec['completed'] and router['status']==large['status']==flops['status']=='complete'
    assert router['run']==g['run'] and router['selected_decisions_match_main']
    assert abs(router['actual_call_fraction']-g['methods']['learned']['test_point']['actual_fraction'])<1e-12
    key='|'.join(g['key'])
    c={'specialist_gflops':spec['mean_counted_gflops_per_sample'],
       'router_gflops':router['mean_gflops_per_input'],
       'large_gflops':flops['mean_prefill_gflops_counted']+large['mean_completion_tokens']*flops['mean_decode_one_token_gflops_counted'],
       'specialist_ms':spec['mean_end_to_end_latency_ms_per_sample'],
       'router_ms':router['mean_latency_ms_per_input'],
       'large_ms':large['mean_batch_wall_ms_per_sample'],
       'large_samples':large['measured_count'], 'flops_probes':flops['sample_count'],
       'specialist_samples':spec['sample_count'],'router_samples':router['sample_count'],'batch_size':large['batch_size']}
    components[key]=c
    for method in ['learned','confidence']:
        for kind,points in [('curve',g['methods'][method]['test_curve']),('selected',[g['methods'][method]['test_point']])]:
            for p in points:
                rate=p['actual_fraction']
                row={'pair':key,'method':method,'nominal_validation_budget':p['budget'],'actual_test_call_fraction':rate,
                     'test_metric':p['metric'],'estimated_gflops_per_input':c['specialist_gflops']+(c['router_gflops'] if method=='learned' else 0)+rate*c['large_gflops'],
                     'estimated_time_ms_per_input':c['specialist_ms']+(c['router_ms'] if method=='learned' else 0)+rate*c['large_ms']}
                (curves if kind=='curve' else selected).append(row)
for name,rows in [('full_cost_performance_curves',curves),('selected_points',selected),('fixed_budget_cost_performance',[p for p in curves if p['nominal_validation_budget'] in [0,.05,.1,.25,.5,.75,1]])]:
    with (root/(name+'.csv')).open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
summary={'components':components,'selected_points':selected,'protocol':
    {'method':'original main-table logit contrast; original checkpoints and validation thresholds; seed 42',
     'large_timing':'64 measured examples; batch size one; same full task prompt; two separate warmup examples',
     'cost':'serial component estimates, not a timed integrated service; includes active distilled encoders and routing head only for learned',
     'flops':'multiply-add=2; supported operators; mean prefill (including visual embedding) plus average generated length times one-token decode probe',
     'scope_limits':'mean call cost extrapolated across policies; no policy-specific input/output-length costs; specialist-feature extraction overhead is not separately measured; confidence threshold overhead not separately timed'}}
(root/'cost_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')

plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman'],'font.size':7,
    'axes.labelsize':7,'xtick.labelsize':6.5,'ytick.labelsize':6.5,'legend.fontsize':7,
    'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,
    'axes.linewidth':.55,'mathtext.fontset':'stix'})
styles=[('learned','learned','#176f91','-'),('confidence','conf.','#c78324','--')]
keys=list(components)

def panel(ax,key,field):
    for method,label,color,ls in styles:
        pts=[p for p in curves if p['pair']==key and p['method']==method]
        assert len(pts)==101
        x=np.array([p[field] for p in pts]);y=np.array([p['test_metric']*100 for p in pts])
        assert np.all(np.diff(x)>=-1e-7)
        ax.plot(x,y,color=color,ls=ls,lw=1.05,label=label)
        p=next(p for p in selected if p['pair']==key and p['method']==method)
        ax.scatter(p[field],100*p['test_metric'],color=color,s=13,edgecolor='white',linewidth=.4,zorder=5)
    ax.grid(axis='y',color='#dce1e4',lw=.4)
    ax.tick_params(length=2.3,pad=1.5)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
    if 'gflops' in field:
        ax.set_xscale('log');ax.xaxis.set_major_locator(LogLocator(base=10,numticks=4));ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlabel('Est. GFLOPs/input',labelpad=1)
    else:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3));ax.set_xlabel('Est. time (ms/input)',labelpad=1)
    ax.set_ylabel('Acc. (%)' if key.startswith('grefcoco') else 'Macro-F1 (%)',labelpad=1)

for field,stem in [('estimated_gflops_per_input','flops_performance'),('estimated_time_ms_per_input','latency_performance')]:
    fig,axes=plt.subplots(1,2,figsize=(3.44,1.50))
    fig.subplots_adjust(left=.115,right=.995,bottom=.31,top=.83,wspace=.43)
    for ax,key,title in zip(axes,keys,['gRefCOCO / Q','Construction / Q']):
        panel(ax,key,field)
        ax.text(.5,1.10,title,transform=ax.transAxes,ha='center',fontsize=7)
    h,l=axes[0].get_legend_handles_labels()
    fig.legend(h,l,loc='lower center',bbox_to_anchor=(.53,-.025),ncol=2,frameon=False,handlelength=2.2,columnspacing=1.8)
    fig.savefig(root/(stem+'.pdf'));fig.savefig(root/(stem+'.png'),dpi=400)
    plt.close(fig)

fig,axes=plt.subplots(2,2,figsize=(3.44,2.05))
fig.subplots_adjust(left=.115,right=.995,bottom=.20,top=.89,wspace=.43,hspace=.64)
for col,key in enumerate(keys):
    for row,field in enumerate(['estimated_gflops_per_input','estimated_time_ms_per_input']):
        panel(axes[row,col],key,field)
    axes[0,col].text(.5,1.14,'gRefCOCO / Q' if col==0 else 'Construction / Q',transform=axes[0,col].transAxes,ha='center',fontsize=7)
h,l=axes[0,0].get_legend_handles_labels()
fig.legend(h,l,loc='lower center',bbox_to_anchor=(.53,-.005),ncol=2,frameon=False,handlelength=2.2,columnspacing=1.8)
fig.savefig(root/'cost_compact.pdf');fig.savefig(root/'cost_compact.png',dpi=400)
plt.close(fig)
print(json.dumps(summary,indent=2))
