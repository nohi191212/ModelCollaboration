"""Output-only control: explicit prediction fields, never targets/correctness."""
import argparse,json
from pathlib import Path
import numpy as np

ROOT=Path('/mnt/HithinkOmniSSD/user_workspace/caisihang/research/ICASSP_ModelCollaboration')
EVENTS=['rule_1_ppe_violation','rule_2_fall_protection_violation','rule_3_unprotected_edge_violation','rule_4_excavator_proximity_violation']

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--expert',nargs='+');args=ap.parse_args()
    work=ROOT/'outputs/router_exploration_25pct_20260910';summaries=[]
    all_experts=['cub','glsim','nlvr2','vilt','groundingdino','instancevg','yolo26x','rtdetr_x_fp32']
    for expert in args.expert or all_experts:
        assert expert in all_experts
        dest=work/'datasets'/expert;meta=json.loads((dest/'completion.json').read_text());task=meta['task']
        if task=='cub':columns=[f'predicted_class_{i}' for i in range(1,201)]+['valid_prediction']
        elif task=='nlvr2':columns=['predicted_true','valid_prediction']
        elif task=='grefcoco':columns=['predicted_box_count','valid_prediction']
        else:columns=[e+'_predicted_present' for e in EVENTS]+[e+'_valid_prediction' for e in EVENTS]
        for split in ['train','val']:
            output=dest/split/'output.npy'
            if output.exists():raise FileExistsError(output)
            records=[json.loads(x) for x in (dest/split/'records.jsonl').read_text().splitlines()]
            values=np.zeros((len(records),len(columns)),dtype=np.float32)
            for i,row in enumerate(records):
                prediction=row['small_label']
                if task=='cub':
                    valid=prediction['valid_output'];category=prediction['prediction']
                    if valid:
                        assert isinstance(category,int) and 1<=category<=200
                        values[i,category-1]=1
                    values[i,-1]=float(valid)
                elif task=='nlvr2':
                    answer=prediction['prediction'];valid=prediction['valid_output']
                    assert not valid or answer in ['True','False']
                    values[i]=[float(answer=='True'),float(valid)]
                elif task=='grefcoco':
                    count=prediction['predicted_boxes'];assert isinstance(count,int) and count>=0
                    values[i]=[count,float(prediction['valid_output'])]
                else:
                    for j,event in enumerate(EVENTS):
                        answer=prediction['events'][event]['prediction']
                        values[i,j]=float(answer is True);values[i,j+4]=float(answer is not None)
            assert np.isfinite(values).all()
            np.save(output,values)
            summaries.append({'expert':expert,'split':split,'rows':len(records),'dimensions':len(columns)})
        (dest/'output_features.json').write_text(json.dumps({'columns':columns,'source_fields':'small_label prediction / predicted_boxes / valid_output / events.prediction only',
                 'target_correctness_used_as_input':False,'normalization':'No standardization of one-hot/binary output indicators; same projection/LayerNorm architecture.'},indent=2))
    ready=[e for e in all_experts if (work/'datasets'/e/'output_features.json').exists() and all((work/'datasets'/e/s/'output.npy').exists() for s in ['train','val'])]
    (work/'output_features_preparation.json').write_text(json.dumps({'status':'complete' if len(ready)==len(all_experts) else 'partial','ready_experts':ready,'splits_prepared_this_call':summaries,'test_rows_used':0},indent=2))
    print(json.dumps(summaries,indent=2))
