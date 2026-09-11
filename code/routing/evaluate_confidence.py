"""Select a confidence threshold on validation and transfer it unchanged to test."""
import argparse,json
from pathlib import Path
import numpy as np
import train_router_formal as formal
from evaluate_formal_dense_curves import label_arrays,score_metric,decision

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--expert',required=True)
    ap.add_argument('--endpoint',choices=['MiniCPM','Qwen3.8'],required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    meta=json.loads((args.root/'outputs/router_exploration_25pct_20260910/datasets'/args.expert/'completion.json').read_text())
    columns=meta['confidence_columns'];column='max_probability' if 'max_probability' in columns else 'max_score'
    source=args.root/'outputs/router_full_20260910/data'/args.expert
    formal.BUDGETS=[i/100 for i in range(101)]
    curves={};points=None;selected=None
    for split in ['val','test']:
        records=formal.read_rows(source/split/'records.jsonl')
        confidence=np.load(source/split/'confidence.npy')
        scores=np.where(confidence[:,columns.index(column+'_valid')].astype(bool),-confidence[:,columns.index(column)],1.)
        if split=='val':points=formal.validation_thresholds(records,args.endpoint,scores)
        arrays=label_arrays(records,args.endpoint)
        curves[split]=[dict(p,actual_fraction=float(decision(scores,p).mean()),metric=score_metric(arrays,decision(scores,p))) for p in points]
        if split=='val':selected=max(range(1,100),key=lambda i:(curves[split][i]['metric'],-curves[split][i]['actual_fraction'],-i))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps({'expert':args.expert,'endpoint':args.endpoint,'confidence_column':column,'selected_index':selected,'validation_point':curves['val'][selected],'test_point':curves['test'][selected],'curves':curves},indent=2))

if __name__=='__main__':main()
