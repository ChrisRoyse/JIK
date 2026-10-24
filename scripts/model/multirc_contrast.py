"""Train-only matched groups, fixed paired sampling and label-scoped ranking loss."""
import numpy as np
import torch
from torch.nn import functional as F
from model.multirc_binding import CONFIG as BASE_CONFIG
from model.multirc_context import check

BASELINE='32bb300d927ca047a42e00afd5223bf8439debad47db2e6b13070221c47c0f12'
CONFIG={**BASE_CONFIG,'ordinary_options':16,'pairs_per_family':12,'pair_weights':[.5,.5],'initial_states_parent':BASELINE}
ARMS=('joint-ce','joint-rank','no-context-rank','no-question-rank','no-answer-rank')

def scope(arm):return 'joint' if arm in ('joint-ce','joint-rank','shuffled-rank') else arm.removesuffix('-rank')

def groups(ctx):
    rr=ctx['rows'];y=ctx['labels'];by_question={};lookup={(r['case'],r['option']):i for i,r in enumerate(rr)}
    train=set(ctx['train'].tolist())
    for i in ctx['train']:by_question.setdefault(rr[i]['case'],[]).append(int(i))
    def group(key,ids):
        p=sorted(i for i in ids if y[i]==1);n=sorted(i for i in ids if y[i]==0)
        check(set(ids)<=train and len({rr[i]['group'] for i in ids})==1,'PAIR_GROUP_SCOPE')
        return dict(key=key,positive=p,negative=n,possible_pairs=len(p)*len(n))
    within=[group(k,v) for k,v in sorted(by_question.items()) if len(set(y[v].tolist()))==2]
    across=[]
    for g in sorted(ctx['contrasts'],key=lambda g:(g['document'],g['option_normalized'])):
        if g['members'][0]['partition']=='train':
            ids=[lookup[m['case'],m['option']] for m in g['members']]
            across.append(group([g['document'],g['option_normalized']],ids))
    check([len(within),len(across)]==[3309,621] and [sum(g['possible_pairs'] for g in gs) for gs in (within,across)]==[25234,1747],'PAIR_CENSUS')
    return dict(within_question=within,across_question=across)

def schedule(train,grouped,seed):
    rng=np.random.default_rng(seed);families=[grouped[k] for k in ('within_question','across_question')]
    weights=[np.array([g['possible_pairs'] for g in gs],np.float64) for gs in families]
    weights=[w/w.sum() for w in weights];steps=[];positions=[];ordinary=[];selected=[]
    for _ in range(CONFIG['steps']):
        ids=rng.choice(train,16,replace=False).tolist();chosen=[]
        for gs,w in zip(families,weights):
            which=rng.choice(len(gs),12,replace=True,p=w);chosen.extend(which.tolist())
            for j in which:ids.extend((int(rng.choice(gs[j]['positive'])),int(rng.choice(gs[j]['negative']))))
        order=rng.permutation(64);inverse=np.argsort(order)
        steps.append(np.asarray(ids,np.int64)[order]);positions.append(inverse[np.arange(16,64).reshape(24,2)])
        ordinary.append(inverse[:16]);selected.append(chosen)
    return dict(rows=np.asarray(steps),pairs=np.asarray(positions),ordinary=np.asarray(ordinary),groups=np.asarray(selected))

def objective(logits,labels,pairs,ranked=True):
    check(logits.shape==(64,2) and labels.shape==(64,) and pairs.shape==(24,2),'PAIR_LOSS_SHAPE')
    check(bool(((pairs>=0)&(pairs<64)).all()),'PAIR_LOSS_ADDRESS')
    score=logits[:,1]-logits[:,0];sign=(labels[pairs[:,0]]-labels[pairs[:,1]]).to(logits.dtype)
    terms=[F.cross_entropy(logits,labels)];counts=[]
    for part in (slice(0,12),slice(12,24)):
        pp=pairs[part];ss=sign[part];valid=(ss!=0).to(logits.dtype);counts.append(valid.sum())
        values=F.softplus(-ss*(score[pp[:,0]]-score[pp[:,1]]))
        terms.append((valid*values).sum()/valid.sum().clamp_min(1))
    terms=torch.stack(terms);loss=terms[0]+(.5*terms[1]+.5*terms[2] if ranked else 0)
    return loss,terms,torch.stack(counts)

@torch.no_grad()
def objective_controls(logits,labels,pairs,ranked):
    base,terms,counts=objective(logits,labels,pairs,ranked)
    flipped,tt,cc=objective(logits,labels,pairs.flip(1),ranked)
    check(torch.equal(base,flipped) and torch.equal(terms,tt) and torch.equal(counts,cc),'PAIR_ORDER_SYMMETRY')
    bad=pairs.clone();bad[0,0]=-1
    try:objective(logits,labels,bad,ranked)
    except ValueError as exc:check('PAIR_LOSS_ADDRESS' in str(exc),'PAIR_ADDRESS_ERROR')
    else:raise ValueError('MNLM_MULTIRC_BINDING_NEGATIVE_PAIR_ADDRESS_ACCEPTED')
    equal=torch.zeros_like(labels);_,empty,nn=objective(logits,equal,pairs,ranked)
    check(torch.equal(empty[1:],torch.zeros_like(empty[1:])) and torch.equal(nn,torch.zeros_like(nn)),'EQUAL_LABEL_RANKING')
    return dict(pair_endpoint_swap_bitwise_invariant=True,invalid_pair_address_refused=True,equal_labels_add_zero_ranking=True,
                objective_uses_fit_labels_only=True,world_truth_assessed=False)
