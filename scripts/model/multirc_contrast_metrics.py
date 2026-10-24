"""Grouped contrast uncertainty without pair-matrix materialization."""
import numpy as np
from model.multirc_context import check

def pair_sums(logits,ids,p):
    rr=p['rows'];y=p['labels'];correct=np.zeros(len(rr),bool);correct[ids]=logits.argmax(1)==y[ids]
    keys=sorted({r['group'] for r in rr if r['partition']=='evaluation' and r['available'] and r['quality_clean']})
    addresses={key:i for i,key in enumerate(keys)};out=np.zeros((len(keys),4),np.int64)
    for col,collection in ((0,p['questions']),(2,p['contrasts'])):
        for ii in collection:
            r=rr[ii[0]]
            if r['partition']!='evaluation' or not r['available'] or not r['quality_clean']:continue
            positive=y[ii]==1;np_=int(positive.sum());nn=len(ii)-np_
            out[addresses[r['group']],col:col+2]+=[np_*nn,int(correct[ii][positive].sum())*int(correct[ii][~positive].sum())]
    check(out[:,0].sum()==8326 and out[:,2].sum()==457,'CONTRAST_METRIC_DENOMINATORS')
    return out

def compare(a,b):
    check(np.array_equal(a[:,[0,2]],b[:,[0,2]]),'CONTRAST_COMPONENTS')
    def rate(x):return np.stack((x[...,1]/x[...,0],x[...,3]/x[...,2]),-1)
    intervals=[]
    for seed in range(5):
        draws=np.random.default_rng(seed).integers(len(a),size=(999,len(a)))
        aa=a[draws].sum(1);bb=b[draws].sum(1);check((aa[:,[0,2]]>0).all(),'CONTRAST_EMPTY_DRAW')
        intervals.append(np.quantile(rate(aa)-rate(bb),[.025,.975],axis=0))
    intervals=np.asarray(intervals)
    return dict(metrics=['within_question_pair_accuracy','same_answer_cross_question_pair_accuracy'],
                difference=(rate(a.sum(0))-rate(b.sum(0))).tolist(),seed_intervals=intervals.tolist(),
                descriptive_envelope=[intervals[:,0].min(0).tolist(),intervals[:,1].max(0).tolist()],
                components=len(a),draws_per_seed=999,estimator_seeds=5,sealed_confirmation=False)

def advancement(comparisons):
    tests={}
    for seed in (25,26,27):
        prefix=f'seed-{seed}/joint-rank/'
        for arm,metric in (('joint-ce',1),('no-context-rank',1),('no-question-rank',1),('no-answer-rank',0)):
            key=prefix+arm;tests[key]=comparisons[key]['descriptive_envelope'][0][metric]>0
    return dict(all_predeclared_research_checks_pass=all(tests.values()),checks=tests,world_truth_assessed=False,
                epistemic_understanding_proven=False,scientific_admission=False,episodic_promotion_authorized=False,sealed_confirmation=False)
