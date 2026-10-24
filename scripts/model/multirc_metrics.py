"""Source-label diagnostics, complete denominators, and component resampling."""
from collections import defaultdict
import numpy as np
from model.multirc_context import check

def probabilities(logits):
    x=logits.astype(np.float64);x-=x.max(1,keepdims=True);e=np.exp(x);return e/e.sum(1,keepdims=True)

def plan(ctx):
    rr=ctx['rows'];questions=defaultdict(list)
    for i,r in enumerate(rr):questions[r['case']].append(i)
    lookup={(r['case'],r['option']):i for i,r in enumerate(rr)}
    contrasts=[np.array([lookup[m['case'],m['option']] for m in g['members']],np.int64) for g in ctx['contrasts']]
    return dict(rows=rr,labels=ctx['labels'],questions=[np.array(v,np.int64) for v in questions.values()],contrasts=contrasts)

def evaluate(logits,ids,p):
    rr=p['rows'];y=p['labels'];n=len(rr);check(logits.shape==(len(ids),2) and np.isfinite(logits).all(),'METRIC_LOGITS')
    prob=np.zeros((n,2),np.float64);prob[ids]=probabilities(logits);available=np.zeros(n,bool);available[ids]=True
    pred=np.full(n,-1,np.int64);pred[ids]=prob[ids].argmax(1);correct=(pred==y)&available
    nll=np.zeros(n);brier=np.zeros(n);nll[ids]=-np.log(np.clip(prob[ids,y[ids]],1e-300,1));brier[ids]=(prob[ids,1]-y[ids])**2
    qrows=[]
    for q in p['questions']:
        tp=int(((pred[q]==1)&(y[q]==1)).sum());den=int((pred[q]==1).sum()+(y[q]==1).sum());complete=bool(available[q].all())
        pos=int((y[q]==1).sum());neg=len(q)-pos
        qrows.append(dict(ids=q,first=int(q[0]),exact=int(correct[q].all()),f1=(2*tp/den if den else 1.) if complete else 0.,
                          pairs=pos*neg,both=int(correct[q][y[q]==1].sum())*int(correct[q][y[q]==0].sum())))
    result={}
    for split in ('train','calibration','evaluation'):
        result[split]={}
        for cohort in ('original','eligible','clean'):
            chosen=np.array([r['partition']==split and (cohort=='original' or r['available']) and
                             (cohort!='clean' or r['quality_clean']) for r in rr]);sel=np.flatnonzero(chosen);observed=sel[available[sel]]
            qq=[q for q in qrows if chosen[q['first']]];den=sum(q['pairs'] for q in qq)
            result[split][cohort]=dict(options=len(sel),questions=len(qq),components=len({rr[i]['group'] for i in sel}),
                scored_options=len(observed),unavailable_options=int((~available[sel]).sum()),correct_options=int(correct[sel].sum()),
                accuracy=float(correct[sel].mean()),nll_scored=float(nll[observed].mean()),brier_scored=float(brier[observed].mean()),
                exact_questions=sum(q['exact'] for q in qq),question_exact=float(np.mean([q['exact'] for q in qq])),
                question_macro_f1=float(np.mean([q['f1'] for q in qq])),opposite_label_pairs=den,
                pairs_both_correct=sum(q['both'] for q in qq),pair_accuracy=sum(q['both'] for q in qq)/den if den else None)
        cc=[c for c in p['contrasts'] if rr[c[0]]['partition']==split];den=sum(int((y[c]==1).sum())*int((y[c]==0).sum()) for c in cc)
        both=sum(int(correct[c][y[c]==1].sum())*int(correct[c][y[c]==0].sum()) for c in cc)
        result[split]['same_answer_contrasts']=dict(groups=len(cc),pairs=den,both_correct=both,pair_accuracy=both/den)
    # Pre-aggregate once; resampling never rescans source rows.
    groups=sorted({r['group'] for r in rr if r['partition']=='evaluation' and r['available'] and r['quality_clean']})
    group_index={g:i for i,g in enumerate(groups)};sums=np.zeros((len(groups),8),np.float64)
    for q in qrows:
        r=rr[q['first']]
        if r['partition']=='evaluation' and r['available'] and r['quality_clean']:
            j=group_index[r['group']];ii=q['ids'];sums[j]+=np.array([len(ii),int(correct[ii].sum()),nll[ii].sum(),brier[ii].sum(),1,q['exact'],q['f1'],0])
    return result,dict(groups=groups,sums=sums),dict(probabilities=prob,prediction=pred,available=available)

def comparisons(left,right):
    check(left['groups']==right['groups'],'BOOTSTRAP_GROUPS');a=left['sums'];b=right['sums'];check(np.array_equal(a[:,[0,4]],b[:,[0,4]]),'BOOTSTRAP_DENOMINATORS')
    def metrics(x):return np.stack([x[...,1]/x[...,0],x[...,2]/x[...,0],x[...,3]/x[...,0],x[...,5]/x[...,4],x[...,6]/x[...,4]],-1)
    estimates=[]
    for seed in range(5):
        draws=np.random.default_rng(seed).integers(len(a),size=(999,len(a)))
        estimates.append(np.quantile(metrics(a[draws].sum(1))-metrics(b[draws].sum(1)),[.025,.975],axis=0))
    intervals=np.stack(estimates)
    return dict(metrics=['accuracy','nll','brier','question_exact','question_macro_f1'],difference=(metrics(a.sum(0))-metrics(b.sum(0))).tolist(),
                seed_intervals=intervals.tolist(),descriptive_envelope=np.stack((intervals[:,0].min(0),intervals[:,1].max(0))).tolist(),
                groups=len(a),draws_per_seed=999,estimator_seeds=5,sealed_confirmation=False)

def intervention_change(base,changed):
    p=probabilities(base);q=probabilities(changed)
    return dict(rows=len(p),argmax_changed=int((p.argmax(1)!=q.argmax(1)).sum()),mean_absolute_probability_change=float(np.abs(p-q).mean()),
                mean_kl=float(np.sum(p*(np.log(np.clip(p,1e-300,1))-np.log(np.clip(q,1e-300,1))),axis=1).mean()),
                semantic_labels_for_swaps=None,world_truth_assessed=False)
