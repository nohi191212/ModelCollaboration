"""Plot archived results without rerunning any model."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import MaxNLocator
root=Path(__file__).resolve().parents[1]
groups=json.loads((root/"data/results.json").read_text(encoding="utf-8"))["groups"]
scaling=json.loads((root/"data/scaling_selected.json").read_text(encoding="utf-8"))
names={"cub":"CUB-200-2011","grefcoco":"gRefCOCO","nlvr2":"NLVR2","construction":"ConstructionSite"}
plt.rcParams.update({"font.family":"serif","font.serif":["Times New Roman"],"font.size":8,
 "axes.titlesize":8.5,"axes.labelsize":8,"xtick.labelsize":7,"ytick.labelsize":7,"legend.fontsize":7.5,
 "pdf.fonttype":42,"ps.fonttype":42,"axes.spines.top":False,"axes.spines.right":False,"axes.linewidth":.6,"mathtext.fontset":"stix"})
colors=["#a7c8c0","#2c8c69","#d77960","#d5d9df"]
curve_colors={"learned":"#176f91","confidence":"#c78324","small":"#536174","large":"#8a537b"}
fig,axes=plt.subplots(2,4,figsize=(7.16,2.75),gridspec_kw={"height_ratios":[.66,1.15]})
fig.subplots_adjust(left=.064,right=.978,bottom=.19,top=.85,wspace=.30,hspace=.67)
for col,(task,expert,endpoint) in enumerate(zip(names,["glsim","instancevg","vilt","yolo26x"],["MiniCPM","MiniCPM","Qwen3.8","Qwen3.8"])):
    subset=[g for g in groups if g["key"][0]==task]
    chosen=next(g for g in subset if g["key"][1:]==[expert,endpoint])
    top,ax=axes[:,col]
    xs=np.array([0,1,3,4])
    floor=np.zeros(4)
    for state,color in zip(["both_correct","rescuable","harmful","both_wrong"],colors):
        values=np.array([100*g["opportunity"]["test"][state]/g["opportunity"]["test"]["total"] for g in subset])
        top.bar(xs,values,bottom=floor,width=.72,color=color,linewidth=0,zorder=3)
        floor+=values
    assert np.allclose(floor,100)
    selected_bar=next(i for i,g in enumerate(subset) if g["key"]==chosen["key"])
    top.add_patch(Rectangle((xs[selected_bar]-.36,0),.72,100,fill=False,edgecolor="#263d46",lw=.8,zorder=5,clip_on=False))
    top.set(ylim=(0,101),xlim=(-.65,4.65),yticks=[0,50,100],xticks=xs)
    top.set_xticklabels(["M","Q","M","Q"])
    top.tick_params(axis="x",length=0,pad=1)
    for x,label in ((.5,subset[0]["expert_display"]),(3.5,subset[2]["expert_display"])):
        top.text(x,-.31,label.replace("YOLO26x", "YOLO26X"),ha="center",va="top",transform=top.get_xaxis_transform(),fontsize=6.7)
    top.set_title(names[task],fontweight="bold",pad=3)
    top.grid(axis="y",color="#dce1e4",lw=.4,zorder=0)
    if col==0:
        top.set_ylabel("Outcome (%)",labelpad=2)
    else:
        top.tick_params(labelleft=False)
    for method,label in (("learned","LR"),("confidence","conf.")):
        saved=chosen["methods"][method]
        curve=saved["test_curve"]
        calls=[100*p["actual_fraction"] for p in curve]
        assert np.all(np.diff(calls)>=-1e-10)
        ax.plot(calls,[100*p["metric"] for p in curve],lw=1.15,color=curve_colors[method],label=label)
        p=saved["test_point"]
        ax.scatter(100*p["actual_fraction"],100*p["metric"],s=16,color=curve_colors[method],edgecolor="white",linewidth=.4,zorder=5,clip_on=False)
    for model,label,ls in (("small","Specialist","--"),("large","VLM",":")):
        ax.axhline(100*chosen[model]["test"],ls=ls,lw=.9,color=curve_colors[model],label=label)
    ax.set(xlim=(0,100),xticks=[0,50,100])
    ax.set_title(chosen["expert_display"].replace("YOLO26x", "YOLO26X")+(" / M" if endpoint=="MiniCPM" else " / Q"),fontsize=8,pad=3)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.grid(axis="y",color="#dce1e4",lw=.4)
    if col==0:
        ax.set_ylabel("Accuracy (%)",labelpad=2)
    elif col==3:
        ax.set_ylabel("Macro-F1 (%)",labelpad=1)
fig.legend([Patch(facecolor=c) for c in colors],["Both correct","Rescue","Harm","Both wrong"],loc="upper center",bbox_to_anchor=(.51,1.01),ncol=4,frameon=False,handlelength=1.3,columnspacing=1.7)
handles,labels=axes[1,0].get_legend_handles_labels()
fig.legend(handles,labels,loc="lower center",bbox_to_anchor=(.52,-.01),ncol=4,frameon=False,handlelength=2,columnspacing=1.8)
fig.text(.52,.083,"Actual test calls (%)",ha="center",fontsize=8)
fig.savefig(root/"figures/gains_compact.pdf")
fig.savefig(root/"figures/gains_compact.png",dpi=300)
plt.close(fig)

fig,axes=plt.subplots(2,4,figsize=(7.16,4.3),sharex=True)
fig.subplots_adjust(left=.075,right=.985,bottom=.12,top=.90,hspace=.42,wspace=.38)
for row,endpoint in enumerate(("MiniCPM","Qwen3.8")):
    for col,task in enumerate(names):
        ax=axes[row,col]
        selected=[r for r in scaling if r["key"][0]==task and r["key"][2]==endpoint]
        for entry,color,marker in zip(selected,["#176f91","#c78324"],["o","s"]):
            name=next(g["expert_display"] for g in groups if g["key"]==entry["key"])
            ax.plot([25,50,100],[100*r["rank25_metric"] for r in entry["rows"]],marker=marker,lw=1.2,ms=4,color=color,label=name)
        ax.set_title(names[task]+(" / M" if row==0 else " / Q"),fontweight="bold")
        ax.set_xticks([25,50,100])
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.grid(axis="y",lw=.45,color="#dce1e4")
        ax.legend(loc="best",frameon=False,fontsize=6.9,handlelength=1)
        if col==0:
            ax.set_ylabel("Test metric (%)")
fig.suptitle("Earlier configurations: exactly 25% test calls",fontsize=11,fontweight="bold")
fig.text(.52,.035,"Router training data (%)",ha="center",fontsize=9)
fig.savefig(root/"figures/scaling_ranked.pdf")
fig.savefig(root/"figures/scaling_ranked.png",dpi=240)
plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(7.16,3.0),sharey=True)
fig.subplots_adjust(left=.16,right=.98,top=.84,bottom=.19,wspace=.32)
for ax,endpoint in zip(axes,("MiniCPM","Qwen3.8")):
    records=json.loads((root/f"data/joint_{endpoint}_comparison_20260910.json").read_text(encoding="utf-8"))
    subset=[g for g in groups if g["key"][2]==endpoint]
    vals=[100*next(j["delta_multi_minus_single"] for j in records if j["expert"]==g["key"][1]) for g in subset]
    ax.barh(np.arange(8),vals,color=["#2c8c69" if v>=0 else "#d77960" for v in vals],height=.64)
    ax.axvline(0,color="#8c949b",lw=.65)
    ax.set_yticks(np.arange(8),[g["expert_display"] for g in subset])
    ax.set_title(endpoint.replace("MiniCPM","MiniCPM-V-4.5"),fontweight="bold")
    ax.grid(axis="x",color="#dce1e4",lw=.4)
    ax.set_axisbelow(True)
    ax.set_xlim(-1.9,4.2)
axes[0].invert_yaxis()
fig.text(.55,.045,"Joint minus single-task test metric (percentage points)",ha="center",fontsize=9)
fig.savefig(root/"figures/joint_delta.pdf")
fig.savefig(root/"figures/joint_delta.png",dpi=240)
plt.close(fig)
print("Generated compact gains, matched-call data scaling, and joint-training figures.")
