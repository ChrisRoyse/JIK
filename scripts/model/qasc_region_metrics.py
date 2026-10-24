"""Per-sense prediction scores; no geometric score is an anchor-correctness claim."""
import numpy as np

def summarize(cosines,correct,rows):
    result={}
    for split in ('train','calibration','evaluation','familiar-facts-unseen-pair'):
        mask=np.array([r['partition']==split if split!='familiar-facts-unseen-pair'
                       else r['partition']=='evaluation' and r['seen_training_facts']==2 for r in rows])
        ids=np.flatnonzero(mask)
        result[split]=dict(cases=len(ids),groups=len({rows[i]['group'] for i in ids}),
                           centered_cosine_by_sense=cosines[ids].mean(0,dtype=np.float64).tolist() if len(ids) else None,
                           region_accuracy_by_sense=correct[ids].mean(0,dtype=np.float64).tolist() if correct is not None and len(ids) else None)
    return result

def scores(values,truth,target_index,logits=None,codes=None):
    cosine=np.stack([np.sum(v.astype(np.float64)*y[target_index].astype(np.float64),axis=1) for v,y in zip(values,truth)],1)
    correct=np.stack([l.argmax(1)==codes[target_index,k] for k,l in enumerate(logits)],1) if logits is not None else None
    return cosine,correct
