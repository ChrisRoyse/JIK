"""TRAIN-only ending diagnostics over frozen, typed hidden states; never serving."""
import math

import numpy as np
import torch

from model.native_population_training import fingerprint
from text.constellation_corpus import need

RIDGE=0.001
TOL=1e-9


def seal(row):
    return fingerprint({k:v for k,v in row.items() if k!='pin'})


def check(rows,fit=False):
    need(rows and len({r['document'] for r in rows})==len(rows),'END_PROBE_DOCUMENTS')
    roster=tuple(rows[0]['hidden']);width=rows[0]['hidden'][roster[0]].shape[1]
    for r in rows:
        need(seal(r)==r['pin'],'END_PROBE_INPUT_PIN')
        need(not fit or (r['side']=='train' and r['history']=='reference'),'END_PROBE_NONTRAIN')
        need(tuple(r['hidden'])==roster and r['end'].dtype==torch.bool and len(r['end'])>=2
             and r['end'].sum().item()==1 and bool(r['end'][-1]),'END_PROBE_BOUNDARY')
        need(all(x.shape==(len(r['end']),width) and x.dtype==torch.float32 and bool(x.isfinite().all())
                 and not x.requires_grad for x in r['hidden'].values()),'END_PROBE_HIDDEN')
        need(r['stop'].shape==(len(r['end']),2) and bool(r['stop'].isfinite().all()),'END_PROBE_STOP')
    return roster,width


def arrays(rows,shuffled=False):
    roster,width=check(rows,True);targets=[];weights=[];positions={}
    for r in rows:
        n=len(r['end']);index=n-1
        if shuffled:
            # Exactly one positive is moved to a nonterminal position, independently per document.
            index=int(np.random.default_rng(20261024+r['document']).integers(n-1))
        y=np.zeros(n,dtype=np.float64);y[index]=1.;targets.append(y)
        weights.append(np.full(n,1./(len(rows)*n)));positions[r['document']]=index
    return ({n:np.concatenate([r['hidden'][n].numpy().astype(np.float64) for r in rows]) for n in roster},
            np.concatenate(targets),np.concatenate(weights),positions)


def logits(theta,x):
    n=len(x);width=next(iter(x.values())).shape[1]
    return sum(v@theta[i*width:(i+1)*width]/n for i,v in enumerate(x.values()))+theta[-1]


def statistics(theta,x,y,weight):
    z=logits(theta,x);p=np.exp(-np.logaddexp(0.,-z));residual=weight*(p-y);q=weight*p*(1-p)
    lanes=len(x);width=next(iter(x.values())).shape[1];d=lanes*width+1
    grad=np.zeros(d);hessian=np.zeros((d,d));parts=list(x.values())
    for i,a in enumerate(parts):
        sl=slice(i*width,(i+1)*width)
        grad[sl]=a.T@residual/lanes+RIDGE*theta[sl]
        hessian[sl,-1]=a.T@q/lanes;hessian[-1,sl]=hessian[sl,-1]
        for j,b in enumerate(parts):hessian[sl,j*width:(j+1)*width]=a.T@(q[:,None]*b)/(lanes*lanes)
    hessian[:-1,:-1]+=np.eye(d-1)*RIDGE;hessian[-1,-1]=q.sum();grad[-1]=residual.sum()
    objective=float(weight@(np.logaddexp(0.,z)-y*z)+RIDGE*(theta[:-1]@theta[:-1])/2)
    need(np.isfinite(objective) and np.isfinite(grad).all() and np.isfinite(hessian).all(),'END_PROBE_NUMERIC')
    return objective,grad,hessian


def fit(rows,shuffled=False):
    x,y,weight,positions=arrays(rows,shuffled);d=sum(v.shape[1] for v in x.values())+1
    theta=np.zeros(d);prior=float(weight@y);theta[-1]=math.log(prior/(1-prior));trace=[]
    while True:
        obj,g,h=statistics(theta,x,y,weight)
        entry=dict(theta=torch.from_numpy(theta.copy()),objective=obj,gradient=torch.from_numpy(g.copy()),
                   curvature=fingerprint(torch.from_numpy(h)),step=0.)
        if np.max(np.abs(g))<=TOL:trace.append(entry);break
        direction=-np.linalg.solve(h,g);slope=float(g@direction);need(np.isfinite(direction).all() and slope<0,'END_PROBE_DIRECTION')
        step=1.
        while statistics(theta+step*direction,x,y,weight)[0]>obj+1e-4*step*slope:
            step*=.5;need(step>=np.finfo(np.float64).eps,'END_PROBE_LINE_SEARCH')
        entry['step']=step;trace.append(entry);theta=theta+step*direction
    return dict(schema='ending-linear-diagnostic-v1',source={r['document']:r['pin'] for r in rows},
        roster=tuple(x),width=next(iter(x.values())).shape[1],ridge=RIDGE,tolerance=TOL,shuffled=shuffled,
        label_positions=positions,theta=torch.from_numpy(theta.copy()),trace=trace)


def baselines(train):
    roster,_=check(train,True);constant=sum(1./len(r['end']) for r in train)/len(train)
    centroids={}
    for n in roster:
        xs=[r['hidden'][n].double().numpy() for r in train]
        centroids[n]=dict(end=np.stack([x[-1] for x in xs]).mean(0).tolist(),
                          continuation=np.concatenate([x[:-1] for x in xs]).mean(0).tolist())
    return dict(constant=constant,lengths=[len(r['end'])-1 for r in train],centroids=centroids)


def score(row,base,linear,shuffled):
    x={n:v.double().numpy() for n,v in row['hidden'].items()};length=len(row['end'])
    c=base['constant'];hazards=[]
    for index in range(length):
        risk=sum(a>=index for a in base['lengths']);events=sum(a==index for a in base['lengths'])
        hazards.append(math.log((events+1)/(risk-events+1)))
    cosine=[]
    for n,v in x.items():
        z=base['centroids'][n]
        def cos(t):
            t=np.asarray(t,dtype=np.float64)
            denom=np.linalg.norm(v,axis=1)*np.linalg.norm(t)
            need(bool(np.all(denom>0)),'END_PROBE_CENTROID_ZERO');return v@t/denom
        cosine.append(cos(z['end'])-cos(z['continuation']))
    return dict(current=(row['stop'][:,1].double()-row['stop'][:,0].double()).numpy(),
        constant=np.full(length,math.log(c/(1-c))),length=np.array(hazards),centroid=np.mean(cosine,axis=0),
        linear=logits(linear['theta'].numpy(),x),linear_shuffled=logits(shuffled['theta'].numpy(),x))


def metrics(rows,scores):
    need(len(rows)==len(scores),'END_PROBE_SCORE_COUNT');result={}
    for method in scores[0]:
        briers=[];losses=[];positive=[];negative=[];correct=early=censored=tp=fp=tn=fn=0;documents=[]
        for r,s in zip(rows,scores):
            z=np.asarray(s[method],dtype=np.float64);y=r['end'].numpy();need(z.shape==y.shape and np.isfinite(z).all(),'END_PROBE_SCORE')
            p=np.exp(-np.logaddexp(0.,-z));decision=z>0 # torch argmax ties select CONTINUE.
            briers.append(float(np.mean((p-y)**2)));losses.append(float(np.mean(np.logaddexp(0.,z)-y*z)))
            positive.extend(z[y]);negative.extend(z[~y]);indices=np.flatnonzero(decision);first=None if not len(indices) else int(indices[0])
            correct+=first==len(y)-1;early+=first is not None and first<len(y)-1;censored+=first is None
            tp+=int(np.sum(decision&y));fp+=int(np.sum(decision&~y));tn+=int(np.sum(~decision&~y));fn+=int(np.sum(~decision&y))
            documents.append(dict(document=r['document'],first_end=first,expected_end=len(y)-1,brier=briers[-1],log_loss=losses[-1]))
        auc=sum(float(a>b)+.5*float(a==b) for a in positive for b in negative)/(len(positive)*len(negative))
        result[method]=dict(documents=documents,brier=float(np.mean(briers)),log_loss=float(np.mean(losses)),auroc=auc,
            correct_end=correct,early_end=early,censored=censored,tp=tp,fp=fp,tn=tn,fn=fn,
            precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn))
    return result
