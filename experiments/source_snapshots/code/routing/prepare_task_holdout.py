"""Task-held-out distillation indices and fresh vocabularies; no GPU training."""
import json
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer,models,trainers,pre_tokenizers,decoders

ROOT=Path('/mnt/HithinkOmniSSD/user_workspace/caisihang/research/ICASSP_ModelCollaboration')
if __name__=='__main__':
    source=ROOT/'data/siglip_distillation_strict_20260910'
    images=json.loads((source/'images.json').read_text());texts=json.loads((source/'texts.json').read_text())
    audit=json.loads((source/'split_audit.json').read_text())
    assert audit['heldout_image_identity_overlap']==audit['heldout_normalized_text_overlap']==0
    work=ROOT/'outputs/router_exploration_25pct_20260910'
    dest=work/'task_holdout';dest.mkdir(exist_ok=False)
    pairs=np.load(work/'distillation_controls/relation_data_official/pairs.npy')
    reports=[]
    for task,tag in [('cub','cub200'),('construction','construction'),('grefcoco','grefcoco'),('nlvr2','nlvr2')]:
        out=dest/task;out.mkdir()
        blocked_paths={str(Path(x['path']).resolve()) for x in images if x['source']==tag}
        image_indices=[i for i,x in enumerate(images) if x['source']!=tag and str(Path(x['path']).resolve()) not in blocked_paths]
        blocked_text={' '.join(x['text'].split()).casefold() for x in texts if tag in x['sources']}
        text_indices=[i for i,x in enumerate(texts) if tag not in x['sources'] and ' '.join(x['text'].split()).casefold() not in blocked_text]
        kept_images=[images[i] for i in image_indices];kept_texts=[texts[i] for i in text_indices]
        assert kept_images and kept_texts
        assert not {x['source'] for x in kept_images}&{tag}
        assert not {str(Path(x['path']).resolve()) for x in kept_images}&blocked_paths
        assert not {' '.join(x['text'].split()).casefold() for x in kept_texts}&blocked_text
        tokenizer=Tokenizer(models.BPE(unk_token='[UNK]'))
        tokenizer.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False);tokenizer.decoder=decoders.ByteLevel()
        tokenizer.train_from_iterator([x['text'] for x in kept_texts],trainers.BpeTrainer(vocab_size=4096,special_tokens=['[PAD]','[UNK]'],initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
        tokenizer.enable_truncation(max_length=64);tokenizer.enable_padding(length=64,pad_id=0,pad_token='[PAD]')
        tokenizer.save(str(out/'student_tokenizer.json'))
        tokens=np.asarray([x.ids for x in tokenizer.encode_batch([x['text'] for x in kept_texts])],dtype=np.int32)
        assert tokens.shape==(len(kept_texts),64) and np.all((tokens>=0)&(tokens<4096))
        np.save(out/'text_ids.npy',tokens)
        np.save(out/'image_source_indices.npy',np.asarray(image_indices,dtype=np.int64))
        np.save(out/'text_source_indices.npy',np.asarray(text_indices,dtype=np.int64))
        image_map={old:new for new,old in enumerate(image_indices)};text_map={old:new for new,old in enumerate(text_indices)}
        selected=[(image_map[int(i)],text_map[int(t)]) for i,t in pairs if int(i) in image_map and int(t) in text_map]
        local_pairs=np.asarray(selected,dtype=np.int64).reshape(-1,2)
        if task=='grefcoco':assert len(local_pairs)==0
        np.save(out/'relation_pairs.npy',local_pairs)
        (out/'images.json').write_text(json.dumps(kept_images,ensure_ascii=False))
        (out/'texts.json').write_text(json.dumps(kept_texts,ensure_ascii=False))
        report={'status':'inputs_prepared_not_trained','heldout_task':task,'source_cache':str(source),'images':len(kept_images),'texts':len(kept_texts),
                'relation_pairs':len(local_pairs),'vocab_size':tokenizer.get_vocab_size(),'heldout_source_images':0,'heldout_normalized_text_overlap':0,
                'index_semantics':'image/text source indices refer to strict cache; tokens and relation pairs use local retained indices',
                'scope':'Source task and shared resolved paths/texts excluded. No arbitrary cross-dataset perceptual duplicate search or external teacher pretraining claim.',
                'student_training_complete':False,'router_training_complete':False}
        (out/'preparation.json').write_text(json.dumps(report,indent=2));reports.append(report)
    (dest/'preparation.json').write_text(json.dumps(reports,indent=2))
    print(json.dumps(reports,indent=2))
