"""Per-case sufficient statistics and descriptive paired intervals, typed by lens."""
from collections import defaultdict
import hashlib
import json

import numpy as np

from audit.legal_breadth_comparison_source import SUBSETS
from text.constellation_corpus import need

DRAWS = 2000


def arrays(row):
    truth = row['target'].numpy(); n = len(truth); indices = np.arange(n); values = {}
    for level in range(3):
        for metric in ('accuracy','prefix_accuracy'):
            scores = {}
            for name,prediction in row['predictions'].items():
                if name=='root_transition' and level>0: continue
                equal = prediction.numpy()==truth
                scores[name] = equal[:,level] if metric=='accuracy' else equal[:,:level+1].all(1)
            values[f'{metric}/{level}'] = (True,False,scores)
        p = row['probabilities'][level].numpy(); wrong = row['permuted_probabilities'][level].numpy()
        for v in (p,wrong):
            need(v.dtype==np.float64 and np.isfinite(v).all() and bool((v>=0).all())
                 and np.allclose(v.sum(1),1,rtol=0,atol=1e-12), 'BREADTH_COMPARE_PROBABILITY')
        selected = p[indices,truth[:,level]]
        shuffled = wrong[indices,truth[row['permutation'].numpy(),level]]
        need(bool((selected>0).all()) and bool((shuffled>0).all()), 'BREADTH_COMPARE_ZERO_PROBABILITY')
        ce = -np.log(selected); brier = (p*p).sum(1)-2*selected+1; permuted = -np.log(shuffled)
        for key,value in [('conditional_ce',ce),('conditional_brier',brier),('permuted_target_ce',permuted)]:
            need(np.allclose(value,row[key][:,level].numpy(),rtol=0,atol=1e-12), 'BREADTH_COMPARE_SCORE_RECOUNT')
        values[f'conditional_ce/{level}'] = (False,False,dict(model=ce))
        values[f'conditional_brier/{level}'] = (False,False,dict(model=brier))
        values[f'target_permutation_gap/{level}'] = (True,True,dict(model=permuted-ce))
    for name,r in row['native'].items():
        d = r['ordered_distances'].numpy()
        need(d.shape==(n,3) and np.isfinite(d).all() and bool((d>=0).all()), 'BREADTH_COMPARE_NATIVE_SCORE')
        values['ordered_distance/'+name] = (False,False,dict(model=d[:,0],copy_current=d[:,1],train_mean=d[:,2]))
        values['count_accuracy/'+name] = (True,False,dict(model=(r['predicted_count']==r['target_count']).numpy()))
    return values


def document(a,b,selected):
    left,right = arrays(a),arrays(b); result=[]
    for subset in SUBSETS:
        for metric,(higher,contrast,values) in left.items():
            mask = a['masks'][subset].numpy().copy()
            if contrast: mask &= a['masks']['permuted_target_distinct'].numpy()
            n = int(mask.sum())
            sums = {arm:{name:float(np.asarray(v,np.float64)[mask].sum()) for name,v in x.items()}
                    for arm,x in [('breadth',values),('repeated',right[metric][2])]}
            result.append(dict(document=a['document'],case=selected['case_component'],side=a['side'],
                source=a['source_collection'],subset=subset,metric=metric,higher_is_better=higher,n=n,sums=sums))
    return result


def draw_key(ids):
    return hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()


def draw_indices(ids):
    key = draw_key(ids)
    return np.random.default_rng(int(key[:16],16)).integers(len(ids),size=(DRAWS,len(ids)),dtype=np.int64)


def aggregate(records, output):
    groups = defaultdict(list); draws={}; result=[]
    for r in records:
        for source in (r['source'],'ALL_COLLECTIONS'):
            groups[(r['side'],source,r['subset'],r['metric'])].append(r)
    for (side,source,subset,metric),all_rows in sorted(groups.items()):
        rows = sorted((r for r in all_rows if r['n']),key=lambda r:r['document'])
        n = sum(r['n'] for r in rows); base=dict(side=side,source=source,subset=subset,metric=metric,
            documents=len(rows),excluded_empty_documents=len(all_rows)-len(rows),rows=n)
        if not n: result.append(dict(base,status='unmeasured_empty_support')); continue
        need(len({r['case'] for r in rows})==len(rows),'BREADTH_COMPARE_AGGREGATE_CASE')
        direction = 1 if rows[0]['higher_is_better'] else -1
        means = {arm:{name:sum(r['sums'][arm][name] for r in rows)/n for name in rows[0]['sums'][arm]}
                 for arm in ('breadth','repeated')}
        deltas=np.asarray([direction*(r['sums']['breadth']['model']-r['sums']['repeated']['model'])/r['n'] for r in rows])
        ids=[r['case'] for r in rows];key=draw_key(ids)
        if key not in draws:
            draws[key]=draw_indices(ids);np.save(output/(key+'.npy'),draws[key],allow_pickle=False)
        distribution=deltas[draws[key]].mean(1)
        baseline_comparison={}
        for arm in means:
            baseline_comparison[arm]={name:dict(
                micro_gain=direction*(means[arm]['model']-means[arm][name]),
                documents_better=sum(direction*(r['sums'][arm]['model']-r['sums'][arm][name])>0 for r in rows),
                documents_worse=sum(direction*(r['sums'][arm]['model']-r['sums'][arm][name])<0 for r in rows),
                documents_tied=sum(r['sums'][arm]['model']==r['sums'][arm][name] for r in rows))
                for name in means[arm] if name!='model'}
        result.append(dict(base,status='measured',higher_is_better=direction==1,means=means,
            breadth_gain_micro=direction*(means['breadth']['model']-means['repeated']['model']),
            breadth_gain_document_mean=float(deltas.mean()),
            breadth_document_interval=np.quantile(distribution,[.025,.975]).tolist() if len(rows)>1 else None,
            breadth_documents_better=int((deltas>0).sum()),breadth_documents_worse=int((deltas<0).sum()),
            breadth_documents_tied=int((deltas==0).sum()),draws=key,baseline_comparison=baseline_comparison))
    return result
