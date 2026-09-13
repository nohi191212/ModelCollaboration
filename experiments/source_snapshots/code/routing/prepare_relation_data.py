"""Image/text row pairs from strict gRefCOCO train only; no new teacher inference."""
import json
from pathlib import Path
import numpy as np

ROOT=Path('/mnt/HithinkOmniSSD/user_workspace/caisihang/research/ICASSP_ModelCollaboration')

if __name__=='__main__':
    cache=ROOT/'data/siglip_distillation_strict_20260910'
    audit=json.loads((cache/'split_audit.json').read_text())
    assert audit['status']=='split_checks_passed' and audit['heldout_image_identity_overlap']==audit['heldout_normalized_text_overlap']==0
    images=json.loads((cache/'images.json').read_text());texts=json.loads((cache/'texts.json').read_text())
    image_index={r['path']:i for i,r in enumerate(images) if r['source']=='grefcoco'}
    text_index={r['text']:i for i,r in enumerate(texts) if 'grefcoco' in r['sources']}
    refs=json.loads((ROOT/'data/grefcoco/official_annotations/grefs(unc).json').read_text())
    expressions={(r['ref_id'],s['sent_id']):(r,s) for r in refs for s in r['sentences']}
    train=[json.loads(x) for x in (ROOT/'data/router_splits_strict_20260910/grefcoco/train.jsonl').read_text().splitlines()]
    dest=ROOT/'outputs/router_exploration_25pct_20260910/distillation_controls/relation_data_official'
    dest.mkdir(parents=True,exist_ok=False)
    selected=[];excluded=[]
    for r in train:
        assert r['image_path'] in image_index,('Missing strict training image',r['sample_id'])
        ref,sentence=expressions[(r['ref_id'],r['sentence_id'])]
        assert ref['split']=='train' and ref['image_id']==r['image_id']
        assert r['expression'].strip() in [sentence['sent'].strip(),sentence['raw'].strip()]
        if sentence['raw'] not in text_index:
            excluded.append({'sample_id':r['sample_id'],'reason':'Official raw expression absent from strict eligible teacher-text cache; held-out exclusions remain enforced.'})
            continue
        selected.append({'sample_id':r['sample_id'],'image_index':image_index[r['image_path']],'text_index':text_index[sentence['raw']],
                         'text_form':'official raw expression linked by ref_id and sentence_id',
                         'image_id':r['image_id'],'target_type':r['ground_truth']['target_type']})
    assert selected
    pairs=np.array([[r['image_index'],r['text_index']] for r in selected],dtype=np.int64)
    assert pairs[:,0].max()<len(images) and pairs[:,1].max()<len(texts)
    np.save(dest/'pairs.npy',pairs)
    (dest/'records.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in selected))
    (dest/'excluded.json').write_text(json.dumps(excluded,indent=2))
    report={'status':'complete','source_rows':len(train),'paired_rows':len(selected),'excluded_text_rows':len(excluded),
            'unique_images':len(set(pairs[:,0].tolist())),'unique_texts':len(set(pairs[:,1].tolist())),
            'target_counts':{kind:sum(r['target_type']==kind for r in selected) for kind in sorted({r['target_type'] for r in selected})},
            'source':'Strict full gRefCOCO train for distillation, not 25% router supervision.',
            'relation_supervision':'Teacher similarity matrix; never assume every source pair is a positive image/text match.',
            'validation_rows_used':0,'test_rows_used':0,'teacher_cache':str(cache)}
    (dest/'completion.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
