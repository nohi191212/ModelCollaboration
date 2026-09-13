"""Calibrate on equally weighted source experts, apply unchanged to targets."""
import numpy as np
from routing.evaluate_router import BUDGETS,correctness,metric

def source_thresholds(expert_scores):
    assert expert_scores
    values=[];weights=[]
    for scores in expert_scores.values():
        scores=np.asarray(scores,dtype=float);assert scores.ndim==1 and len(scores)>0 and np.isfinite(scores).all()
        values.append(scores);weights.append(np.full(len(scores),1/(len(expert_scores)*len(scores))))
    values=np.concatenate(values);weights=np.concatenate(weights)
    unique,inverse=np.unique(values,return_inverse=True)
    cumulative=np.cumsum(np.bincount(inverse,weights=weights)[::-1]);descending=unique[::-1]
    rules=[]
    for budget in BUDGETS:
        if budget==0:rule={'budget':budget,'operator':'never','threshold':None}
        elif budget==1:rule={'budget':budget,'operator':'always','threshold':None}
        else:
            eligible=np.flatnonzero(cumulative<=budget)
            rule={'budget':budget,'operator':'ge' if len(eligible) else 'gt','threshold':float(descending[eligible[-1] if len(eligible) else 0])}
        rates={name:float(apply_rule(np.asarray(scores),rule).mean()) for name,scores in expert_scores.items()}
        rule.update(source_expert_fractions=rates,source_equal_expert_fraction=sum(rates.values())/len(rates))
        assert rule['source_equal_expert_fraction']<=budget+1e-12
        rules.append(rule)
    return {'calibration':'source validation scores only, equal weight per expert','rules':rules,'target_data_used':False}

def apply_rule(scores,rule):
    operator=rule['operator']
    if operator=='never':return np.zeros(len(scores),dtype=bool)
    if operator=='always':return np.ones(len(scores),dtype=bool)
    if operator=='ge':return scores>=rule['threshold']
    assert operator=='gt';return scores>rule['threshold']

def target_threshold_results(records,endpoint,scores,calibration):
    scores=np.asarray(scores);assert len(scores)==len(records) and np.isfinite(scores).all()
    s,l=correctness(records,endpoint);points=[]
    for rule in calibration['rules']:
        use=apply_rule(scores,rule);selected=use[:,None]
        points.append({'source_budget':rule['budget'],'threshold':rule['threshold'],'operator':rule['operator'],'target_actual_fraction':float(use.mean()),'metric':metric(records,endpoint,use),'rescue':int(((s==0)&(l==1)&selected).sum()),'harm':int(((s==1)&(l==0)&selected).sum())})
    return {'curve':points,'note':'Source-calibrated thresholds unchanged. Target call fraction may exceed or fall below source budget; do not relabel as exact-budget performance.'}
