"""Original task metrics and help/harm curves, strictly from held-out validation labels."""
import numpy as np

EVENTS=['rule_1_ppe_violation','rule_2_fall_protection_violation','rule_3_unprotected_edge_violation','rule_4_excavator_proximity_violation']
BUDGETS=[0,.05,.10,.15,.20,.25,.50,.75,1.]

def correctness(records,endpoint):
    small=[r['small_label'] for r in records];large=[r['large_labels'][endpoint] for r in records]
    if 'events' in small[0]:
        s=np.array([[r['events'][e]['correct'] for e in EVENTS] for r in small],dtype=float)
        l=np.array([[r['events'][e]['correct'] for e in EVENTS] for r in large],dtype=float)
    else:
        s=np.array([[r['correct']] for r in small],dtype=float);l=np.array([[r['correct']] for r in large],dtype=float)
    return s,l

def metric(records,endpoint,upgrade):
    chosen=[r['large_labels'][endpoint] if use else r['small_label'] for r,use in zip(records,upgrade)]
    if 'events' not in chosen[0]: return float(np.mean([r['correct'] for r in chosen]))
    values=[]
    for event in EVENTS:
        target=np.array([r['events'][event]['target'] for r in chosen],dtype=bool)
        # Invalid outputs remain None. They predict no positive event, not a fabricated answer.
        positive=np.array([r['events'][event]['prediction'] is True for r in chosen])
        tp=np.sum(target&positive);fp=np.sum(~target&positive);fn=np.sum(target&~positive)
        values.append(float(2*tp/(2*tp+fp+fn)) if 2*tp+fp+fn else 0.)
    return float(np.mean(values))

def evaluate(records,endpoint,scores):
    scores=np.asarray(scores,dtype=float)
    assert scores.shape==(len(records),) and np.isfinite(scores).all()
    order=np.argsort(-scores,kind='stable');s,l=correctness(records,endpoint)
    rescue=(s==0)&(l==1);harm=(s==1)&(l==0)
    points=[]
    for budget in BUDGETS:
        count=int(np.floor(len(records)*budget));upgrade=np.zeros(len(records),dtype=bool);upgrade[order[:count]]=True
        selected=upgrade[:,None]
        threshold=None if count in [0,len(records)] else float(scores[order[count-1]])
        points.append({'budget':budget,'actual_fraction':float(upgrade.mean()),'metric':metric(records,endpoint,upgrade),
                       'rescue':int((rescue&selected).sum()),'harm':int((harm&selected).sum()),'missed_rescue':int((rescue&~selected).sum()),
                       'wasted_events':int(((s==l)&selected).sum()),'threshold_boundary':threshold,'threshold_tie_count':0 if threshold is None else int((scores==threshold).sum())})
    area=float(sum((left['metric']+right['metric'])*.5*(right['budget']-left['budget']) for left,right in zip(points[:5],points[1:6]))/.25)
    return {'curve':points,'selection_score':area,'score_std':float(scores.std()),'score_min':float(scores.min()),'score_max':float(scores.max()),'event_counts':s.shape[1],
            'threshold_note':'budget curve uses stable ranks; boundary is not an exactly equivalent fixed threshold when scores tie',
            'invalid_note':'invalid responses retained as errors for correctness; presence-F1 treats them as no predicted positive event; no fabricated labels'}

def fixed_threshold_curve(records,endpoint,scores):
    """Validation-selected thresholds; tied scores never exceed the requested budget.

    This is descriptive validation performance, not independent test performance.
    Persist both threshold and comparison operator for subsequent deployment.
    """
    scores=np.asarray(scores,dtype=float)
    assert scores.shape==(len(records),) and np.isfinite(scores).all()
    order=np.argsort(-scores,kind='stable');s,l=correctness(records,endpoint)
    rescue=(s==0)&(l==1);harm=(s==1)&(l==0);points=[]
    for budget in BUDGETS:
        count=int(np.floor(len(records)*budget));threshold=None
        if count==0:
            operator='never';upgrade=np.zeros(len(records),dtype=bool)
        elif count==len(records):
            operator='always';upgrade=np.ones(len(records),dtype=bool)
        else:
            threshold=float(scores[order[count-1]])
            operator='ge' if np.sum(scores>=threshold)<=count else 'gt'
            upgrade=scores>=threshold if operator=='ge' else scores>threshold
        assert int(upgrade.sum())<=count
        selected=upgrade[:,None]
        points.append({'budget':budget,'threshold':threshold,'operator':operator,
                       'actual_fraction':float(upgrade.mean()),'metric':metric(records,endpoint,upgrade),
                       'rescue':int((rescue&selected).sum()),'harm':int((harm&selected).sum()),
                       'missed_rescue':int((rescue&~selected).sum()),
                       'threshold_tie_count':0 if threshold is None else int((scores==threshold).sum())})
    return {'curve':points,'selection_split':'validation',
            'note':'Threshold and operator selected on validation; reuse unchanged on future evaluation. Future call fraction may differ. This is not test performance.'}
