"""Rebuild tables from saved results; no model execution."""
from pathlib import Path
import json

root = Path(__file__).resolve().parents[1]
groups = json.loads((root/"data/results.json").read_text(encoding="utf-8"))["groups"]
diagnostics = json.loads((root/"data/budget_diagnostics.json").read_text(encoding="utf-8"))
earlier = json.loads((root/"data/earlier_results.json").read_text(encoding="utf-8"))["rows"]
task_names = {"cub":"CUB-200-2011","grefcoco":"gRefCOCO","nlvr2":"NLVR2","construction":"ConstructionSite"}
joint = []
for endpoint in ("MiniCPM","Qwen3.8"):
    joint.extend(json.loads((root/f"data/joint_{endpoint}_comparison_20260910.json").read_text(encoding="utf-8")))
rows = {name:[] for name in ("main_test","main_full","thresholds","outcomes_full","scaling_full","joint_full")}
scaling, audit = [], []
counts = dict(better_metric=0,fewer_calls=0,both=0,above_endpoints=0,better_ranked_area=0)
for index,g in enumerate(groups):
    task,expert,endpoint = g["key"]
    prefix = f"{task_names[task] if index%4==0 else ''} & {g['expert_display']} & {'M' if endpoint=='MiniCPM' else 'Q'}"
    if index and index%4==0:
        for table in rows.values():
            table.append(r"\midrule")
    blocks, raw = [], {}
    for split in ("validation","test"):
        conf = g["methods"]["confidence"][f"{split}_point"]
        learned = g["methods"]["learned"][f"{split}_point"]
        change = 100*(learned["actual_fraction"]-conf["actual_fraction"])
        macro = "callup" if change>1e-10 else "calldown" if change < -1e-10 else "samechg"
        delta = rf"\{macro}{{{abs(change):.2f}}}"
        metric_change = 100*(learned["metric"]-conf["metric"])
        metric_macro = "upchg" if metric_change>1e-10 else "downchg" if metric_change < -1e-10 else "samechg"
        metric_delta = rf"\{metric_macro}{{{abs(metric_change):.2f}}}"
        cv,lv = f"{100*conf['metric']:.2f}",f"{100*learned['metric']:.2f}"
        if conf["metric"]>=learned["metric"]:
            cv = rf"\textbf{{{cv}}}"
        if learned["metric"]>=conf["metric"]:
            lv = rf"\textbf{{{lv}}}"
        if split=="test" and learned["metric"]>max(g["small"][split],g["large"][split])+1e-12:
            lv += r"$^\dagger$"
        blocks.append(f"{100*g['small'][split]:.2f} & {100*g['large'][split]:.2f} & {cv} & {100*conf['actual_fraction']:.2f} & {lv}{metric_delta} & {100*learned['actual_fraction']:.2f}{delta}")
        raw[split] = dict(spec=100*g["small"][split],vlm=100*g["large"][split],conf=100*conf["metric"],call_conf=100*conf["actual_fraction"],learned=100*learned["metric"],call_learned=100*learned["actual_fraction"],change_pp=change,metric_change_pp=metric_change)
        for method in ("confidence","learned"):
            p=g["methods"][method]
            if method=="learned":
                for a,b in zip(p[f"{split}_point"]["per_seed"],p["validation_point"]["per_seed"]):
                    assert (a['seed'],a['threshold'],a['operator'])==(b['seed'],b['threshold'],b['operator'])
            else:
                assert p[f"{split}_point"]["threshold"]==p["validation_point"]["threshold"]
                assert p[f"{split}_point"]["operator"]==p["validation_point"]["operator"]
    rows["main_test"].append(prefix+" & "+blocks[1]+r" \\")
    rows["main_full"].append(prefix+" & "+" & ".join(blocks)+r" \\")
    selected=g["methods"]["learned"]["test_point"]
    reference=g["methods"]["confidence"]["test_point"]
    better=selected["metric"]>reference["metric"]+1e-12
    fewer=selected["actual_fraction"]<reference["actual_fraction"]-1e-12
    counts["better_metric"]+=better
    counts["fewer_calls"]+=fewer
    counts["both"]+=better and fewer
    counts["above_endpoints"]+=selected["metric"]>max(g["small"]["test"],g["large"]["test"])+1e-12
    d=next(d for d in diagnostics if d["key"]==g["key"])
    counts["better_ranked_area"]+=d["areas"]["learned"]["test_ranked_A25_diagnostic"]>d["areas"]["confidence"]["test_ranked_A25_diagnostic"]
    cutoffs=[]
    for method in ("confidence","learned"):
        p=g["methods"][method]["validation_point"]
        if method=="learned":
            cutoffs.extend([f"{sum(100*x['budget'] for x in p['per_seed'])/5:.1f}","Per seed"])
            continue
        op=r"\geq" if p["operator"]=="ge" else ">"
        cutoff=p['operator'].capitalize() if p['threshold'] is None else "$"+f"{op}{p['threshold']:.6g}"+"$"
        cutoffs.extend([f"{100*p['budget']:.0f}",cutoff])
    rows["thresholds"].append(prefix+" & "+" & ".join(cutoffs)+r" \\")
    opp=g["opportunity"]["test"]
    keys=("both_correct","rescuable","harmful","both_wrong")
    assert sum(opp[k] for k in keys)==opp["total"]
    gain=100*(selected["metric"]-g["small"]["test"])
    if task!="construction":
        assert abs(gain-100*(selected["rescued"]-selected["harmed"])/g["test_rows"])<1e-9
    values=[opp[k] for k in keys]+[selected[k] for k in ("rescued","harmed","missed_rescue","avoided_harm")]
    rows["outcomes_full"].append(prefix+" & "+" & ".join(f"{v:,}" if isinstance(v,int) else f"{v:,.1f}" for v in values)+f" & {gain:+.3f}"+r" \\")
    scale=[]
    for fraction,tag in ((.25,"baseline_25pct"),(.5,"baseline_50pct"),(1.,"baseline_100pct")):
        matches=[r for r in earlier if (r["task"],r["expert"],r["endpoint"],r["tag"])==(task,expert,endpoint,tag)]
        assert len(matches)==1,(g["key"],tag,len(matches))
        assert matches[0]["fraction"]==fraction
        scale.append(matches[0])
    for field in ("student","branches","fusion","target","strategy","representation_dim","hidden_layer"):
        assert all(r[field]==scale[0][field] for r in scale),(g["key"],field)
    rows["scaling_full"].append(prefix+" & "+" & ".join(f"{100*r['fixed25_metric']:.2f} & {100*r['fixed25_actual_fraction']:.2f} & {100*r['rank25_metric']:.2f}" for r in scale)+r" \\")
    scaling.append(dict(key=g["key"],rows=scale))
    j=next(j for j in joint if (j["expert"],j["endpoint"])==(expert,endpoint))
    rows["joint_full"].append(prefix+f" & {100*j['single_task_test_fixed25']:.3f} & {100*j['multi_task_test_fixed25']:.3f} & {100*j['delta_multi_minus_single']:+.3f}"+r" \\")
    audit.append(dict(key=g["key"],**raw))
assert counts==json.loads((root/'data/five_seed_counts.json').read_text()),counts

head=r"\textcolor{testink}{spec.} & \textcolor{testink}{vlm.} & \textcolor{confink}{conf.} & \textcolor{confink}{call$_{\rm conf}$} & \textcolor{learnink}{learned$\uparrow$} & \textcolor{learnink}{call$_{\rm learned}\downarrow$}"
cols=r">{\columncolor{basefill}}r>{\columncolor{basefill}}r>{\columncolor{conffill}}r>{\columncolor{conffill}}r>{\columncolor{learnfill}}r>{\columncolor{learnfill}}r"
specs=[
("main_test","table*","t","tab:main","lll"+cols, "Dataset & Specialist & VLM & "+head,
 r"Test results at validation-selected thresholds. Scores and actual call rates are percentages; parentheses give learned-minus-conf. differences in percentage points. \textcolor{upgreen}{Green} marks improvement and \textcolor{downred}{red} degradation: higher scores and fewer calls are better. Bold compares routed scores; $\dagger$ exceeds both models. M: MiniCPM-V-4.5; Q: Qwen3.8-27B-FP8. ",9,6.8),
("main_full","table","H","tab:full","lll"+cols+cols,
 r"\multirow{2}{*}{Task} & \multirow{2}{*}{Specialist} & \multirow{2}{*}{VLM} & \multicolumn{6}{c}{\cellcolor{valfill}\textcolor{valink}{Validation}} & \multicolumn{6}{c}{\cellcolor{testfill}\textcolor{testink}{Test}} \\ \cmidrule(lr){4-9}\cmidrule(l){10-15} & & & "+head+" & "+head,
 r"Complete four-way comparison. Scores and actual call rates are percentages. Thresholds selected on the 1--99\% validation grid transfer unchanged to test. Parentheses give learned-minus-confidence differences in percentage points before rounding. Green marks improvement and red degradation: higher scores and fewer calls are better. Bold marks the higher routed score; $\dagger$ exceeds both test endpoints. M: MiniCPM-V-4.5; Q: Qwen3.8-27B-FP8.",8,1.8),
("thresholds","table","H","tab:thresholds","lllrrrr",
 r"Task & Specialist & VLM & Conf. budget & Conf. cutoff & Learned budget & Learned cutoff",
 "Selected nominal validation budgets and score cutoffs. The operator is part of each stored threshold. Displayed cutoffs are rounded; the archived JSON retains full precision. Budget is a percentage, not the numerical threshold.",9,4),
("outcomes_full","table","H","tab:outcomes_full","lllrrrrrrrrr",
 r"Task & Specialist & VLM & BC & R & H & BW & Sel. R & Sel. H & Miss R & Avoid H & Gain (pp)",
 "Test outcome counts at the learned operating point. BC/BW: both correct/wrong; R/H: rescue/harm. Selected rescues, selected harms, missed rescues and avoided harms are counted separately. Construction counts 12,016 events, with routing over 3,004 images. Metric gains are recomputed; macro-F1 is not an event-count ratio.",8.5,3),
("scaling_full","table","H","tab:scaling_full","lllrrrrrrrrr",
 r"Task & Specialist & VLM & \multicolumn{3}{c}{25\% training} & \multicolumn{3}{c}{50\% training} & \multicolumn{3}{c}{100\% training}\\ \cmidrule(lr){4-6}\cmidrule(lr){7-9}\cmidrule(l){10-12} & & & Metric & Calls & Rank & Metric & Calls & Rank & Metric & Calls & Rank",
 r"Earlier training-data comparisons. Each pair retains its exploratory reference configuration; only router-training fraction changes. Within each fraction: test metric, actual test calls, and test metric at exactly 25\% ranked calls. The first two use a validation-selected 25\% threshold. These are not final shared-recipe ablations. All entries are percentages.",8.5,3),
("joint_full","table","H","tab:joint_full","lllrrr",
 r"Task & Specialist & VLM & Single & Joint & Difference (pp)",
 r"Earlier single-task versus joint-training comparison. Both use 4M students, image/text inputs and specialist-error supervision. Metrics use validation-selected 25\% thresholds; actual call rates are not present in this comparison file. Differences are joint minus single in percentage points. All tasks were seen during training.",9,5)]
for name,env,place,label,spec,header,caption,size,pad in specs:
    if name in ['main_test','main_full']:
        caption='Learned results are means over five router seeds (2026--2030); the supplement reports standard deviations. '+caption
        if name == 'main_test':
            caption=r'Test results at validation-selected thresholds; learned values average five seeds. Scores/calls are percentages; parentheses show changes from conf. in points (green: better; red: worse). Bold compares routers; $\dagger$ exceeds both endpoints using unrounded values. M: MiniCPM-V-4.5; Q: Qwen3.8-27B-FP8. Validation results and standard deviations are in the supplement.'
    if name=='thresholds':
        caption='Confidence cutoffs and mean learned validation budgets across five seeds. Each learned seed uses its own validation cutoff unchanged on test; exact per-seed cutoffs are provided in the accompanying data/per\\_seed.csv.'
    if name=='outcomes_full':
        caption='Selected outcome counts and gains are five-seed means; available outcome counts are fixed. '+caption
    content=f"\\begin{{{env}}}[{place}]\n\\centering\n\\caption{{{caption}}}\n\\label{{{label}}}\n"
    content+=f"\\fontsize{{{size}}}{{{size*1.2}}}\\selectfont\n\\setlength{{\\tabcolsep}}{{{pad}pt}}\n\\renewcommand{{\\arraystretch}}{{1.12}}\n"
    content+="\\begin{tabular}{@{}"+spec+"@{}}\n\\toprule\n"+header+r" \\"+"\n\\midrule\n"
    content+="\n".join(rows[name])+"\n\\bottomrule\n\\end{tabular}\n"+f"\\end{{{env}}}\n"
    if name == "main_test":
        content = content.replace(r"\renewcommand{\arraystretch}{1.12}", r"\renewcommand{\arraystretch}{1.0}")
        content = content.replace(r"\begin{table*}[t]", r"\begin{minipage}{\textwidth}")
        content = content.replace(r"\caption{", r"\captionof{table}{")
        content = content.replace(r"\end{table*}", r"\end{minipage}")
        content = content.replace(
            r">{\columncolor{learnfill}}r@{}}",
            r">{\columncolor{learnfill}[\tabcolsep][0pt]}r@{}}",
        )
    (root/f"tables/{name}.tex").write_text(content,encoding="utf-8")
(root/"data/table_audit.json").write_text(json.dumps(dict(counts=counts,rows=audit),indent=2),encoding="utf-8")
(root/"data/scaling_selected.json").write_text(json.dumps(scaling,indent=2),encoding="utf-8")
print(json.dumps(counts))
print("Rebuilt all 16-pair tables from saved results.")
