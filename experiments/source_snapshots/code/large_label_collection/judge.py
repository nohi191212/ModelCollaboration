"""生成路由监督；非法模型回答强制记错，运输失败不生成模型对错标签。"""
import numpy as np

EVENTS = ['rule_1_ppe_violation','rule_2_fall_protection_violation',
          'rule_3_unprotected_edge_violation','rule_4_excavator_proximity_violation']


def judge(task, row, output):
    valid = output['parse_status'] == 'valid'
    result = dict(sample_id=row['sample_id'], split=row['split'], valid_output=valid,
                  correct=False, error_reason=output.get('parse_error'))
    gt = row['ground_truth']
    if task == 'cub':
        result.update(correct=valid and output['predicted_category_id']==gt['category_id'],
                      target=gt['category_id'], prediction=output['predicted_category_id'],
                      correctness_rule='category_exact_match')
    elif task == 'nlvr2':
        result.update(correct=valid and output['prediction']==gt, target=gt,
                      prediction=output['prediction'], correctness_rule='True_False_exact_match')
    elif task == 'grefcoco':
        targets = gt['target_boxes']
        boxes = [o['bbox_xyxy_absolute'] for o in output['objects'] if o['score']>=0.7] if valid else []
        assert gt['target_type'] in ['no_target','single_target','multi_target']
        assert bool(targets) == (gt['target_type']!='no_target')
        tp = 0
        if valid and boxes and targets:
            a = np.asarray(boxes,dtype=np.float64)
            b = np.asarray([o['bbox_xyxy'] for o in targets],dtype=np.float64)
            sizes = np.maximum(np.minimum(a[:,None,2:],b[None,:,2:])-np.maximum(a[:,None,:2],b[None,:,:2]),0)
            intersection = sizes[:,:,0]*sizes[:,:,1]
            union = ((a[:,2]-a[:,0])*(a[:,3]-a[:,1]))[:,None] + ((b[:,2]-b[:,0])*(b[:,3]-b[:,1]))[None,:]-intersection
            enclosure = np.maximum(a[:,None,2:],b[None,:,2:])-np.minimum(a[:,None,:2],b[None,:,:2])
            area = enclosure[:,:,0]*enclosure[:,:,1]
            giou = intersection/union-(area-union)/area
            for _ in range(min(len(a),len(b))):
                index = np.argmax(giou)
                i,j = np.unravel_index(index,giou.shape)
                if giou[i,j]<0.5:
                    break
                tp += 1
                giou[i,:]=0
                giou[:,j]=0
        f1 = (2*tp/(len(boxes)+len(targets)) if targets else float(not boxes)) if valid else 0.0
        result.update(correct=bool(valid and f1>=1.0),instance_f1=f1,target_type=gt['target_type'],
                      predicted_boxes=len(boxes),target_boxes=len(targets),matched_boxes=tp,
                      correctness_rule='score_0.7_greedy_GIoU_0.5_sample_F1_1; invalid_always_wrong')
    else:
        assert task=='construction' and set(gt['events'])==set(EVENTS)
        labels={}
        for event in EVENTS:
            reference=gt['events'][event]
            candidate=output['events'][event] if valid else None
            predicted=candidate['present'] if valid else None
            mask_iou=None
            if reference['boxes']:
                masks=[]
                for boxes in [reference['boxes'],candidate['boxes'] if valid else []]:
                    mask=np.zeros((100,100),dtype=bool)
                    for x1,y1,x2,y2 in boxes:
                        mask[int(y1*100):int(y2*100)+1,int(x1*100):int(x2*100)+1]=True
                    masks.append(mask)
                mask_iou=float(np.logical_and(*masks).sum()/np.logical_or(*masks).sum())
            labels[event]=dict(target=reference['present'],prediction=predicted,
                               correct=bool(valid and predicted==reference['present']),
                               positive_target_mask_iou=mask_iou)
        result.update(correct=bool(valid and all(x['correct'] for x in labels.values())),
            events=labels,correct_event_count=sum(x['correct'] for x in labels.values()),
            correctness_rule='all_four_event_presence_exact_match; invalid_always_wrong',
            native_metric_note='macro_presence_F1 must be recomputed over a dataset; correct is not F1')
    return result
