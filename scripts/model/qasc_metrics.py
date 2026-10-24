"""Saved probability diagnostics and fixed group-clustered paired comparisons."""
import numpy as np
from model.qasc_context import check

def subsets(rows):
    result={s:np.array([i for i,r in enumerate(rows) if r['partition']==s],np.int64) for s in ('train','calibration','evaluation')}
    for s in ('calibration','evaluation'):
        for seen in (0,1,2):result[f'{s}/facts-seen-{seen}']=np.array([i for i in result[s] if rows[i]['seen_training_facts']==seen],np.int64)
    return result

def measure(prob,rows,labels):
    p=np.asarray(prob,np.float64);check(p.shape==(len(rows),8) and np.isfinite(p).all() and (p>0).all() and np.allclose(p.sum(1),1,atol=2e-6,rtol=0),'PROBABILITIES')
    out={}
    for name,ids in subsets(rows).items():
        if not len(ids):out[name]=dict(cases=0);continue
        pp=p[ids];yy=labels[ids];pred=pp.argmax(1);confidence=pp.max(1);diagnostics={}
        for t in (.5,.75,.9,.95,.99):
            accepted=confidence>=t;n=int(accepted.sum());wrong=int(np.sum(accepted&(pred!=yy)))
            diagnostics[str(t)]=dict(accepted=n,wrong=wrong,coverage=n/len(ids),error_rate=wrong/n if n else None,production_assertions=0)
        out[name]=dict(cases=len(ids),groups=len({rows[i]['group'] for i in ids}),accuracy=float((pred==yy).mean()),
                       nll_nats=float(-np.log(pp[np.arange(len(ids)),yy]).mean()),brier=float(((pp-np.eye(8)[yy])**2).sum(1).mean()),confidence=diagnostics)
    return out

def comparisons(outputs,rows,labels,seed):
    rng=np.random.default_rng(seed);plans={};samples={};summary={}
    for subset,ids in subsets(rows).items():
        if not subset.startswith('evaluation') or not len(ids):continue
        groups=sorted({rows[i]['group'] for i in ids});index={g:j for j,g in enumerate(groups)}
        membership=np.array([index[rows[i]['group']] for i in ids]);counts=np.bincount(membership,minlength=len(groups))
        draws=rng.integers(0,len(groups),size=(500,len(groups)));weights=np.stack([np.bincount(r,minlength=len(groups)) for r in draws]);plans[subset]=draws
        base=(outputs['both-facts']['probabilities'][ids].argmax(1)==labels[ids]).astype(float)
        alternatives={name:out['probabilities'] for name,out in outputs.items() if name!='both-facts'}
        alternatives.update({f'intervention-{n}':p for n,p in outputs['both-facts'].items() if n!='probabilities'})
        for name,p in alternatives.items():
            delta=base-(p[ids].argmax(1)==labels[ids]);sums=np.bincount(membership,weights=delta,minlength=len(groups))
            values=(weights@sums)/(weights@counts);key=subset+'|'+name;samples[key]=values
            summary[key]=dict(cases=len(ids),groups=len(groups),paired_accuracy_difference=float(delta.mean()),
                              percentile_95=np.quantile(values,[.025,.975]).tolist(),draws=500)
    return plans,samples,summary
