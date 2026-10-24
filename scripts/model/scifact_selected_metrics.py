"""Descriptive source-anchor and intervention diagnostics, never admission."""
import numpy as np
from model.scifact_readout import metrics
from audit.description_baseline import bounds
from audit.scifact_comparisons import contrasts


def measure(prob,selection,ctx,truth):
    rows=ctx["rows"];labels=np.array([{"NOT_ENOUGH_EVIDENCE":0,"CONTRADICT":1,"SUPPORT":2}[t["label"]] for t in ctx["targets"]])
    pred=prob.argmax(1);evidence={}
    for split in ("train","calibration","evaluation"):
        ids=[i for i,r in enumerate(rows) if r["partition"]==split];annotated=[i for i in ids if truth["paths"][i]["alternatives"]]
        exact=[];complete=[];multi=[];multi_complete=[];multi_joint=[]
        for i in annotated:
            chosen=set(np.flatnonzero(selection[i]));gold=[set(p[:-1]) for p in truth["paths"][i]["alternatives"]]
            ex=any(g==chosen for g in gold);co=any(g<=chosen for g in gold)
            if ex:exact.append(i)
            if co:complete.append(i)
            if min(map(len,gold))>1:
                multi.append(i)
                if co:multi_complete.append(i)
                if ex and pred[i]==labels[i]:multi_joint.append(i)
        evidence[split]=dict(cases=len(ids),annotated_cases=len(annotated),exact_sets=len(exact),complete_sets=len(complete),
            label_and_exact=sum(int(pred[i]==labels[i]) for i in exact),label_and_complete=sum(int(pred[i]==labels[i]) for i in complete),
            selected_sentences=int(selection[ids].sum()),missing_evidence_refusals=int((~selection[ids].any(1)).sum()),
            positive_label_without_evidence=int(((pred[ids]>0)&~selection[ids].any(1)).sum()),
            mandatory_multi_rows=multi,mandatory_multi_complete=len(multi_complete),mandatory_multi_label_and_exact=len(multi_joint))
    return dict(relation=metrics(prob,rows,labels),evidence=evidence)


def interventions(outputs,rows):
    p=outputs["probabilities"].astype(float);out={}
    for split in ("train","calibration","evaluation"):
        ix=np.array([i for i,r in enumerate(rows) if r["partition"]==split]);out[split]={}
        for name,q in outputs.items():
            if name=="probabilities":continue
            delta=np.abs(p[ix]-q[ix]);out[split][name]=dict(cases=len(ix),changed_labels=int((p[ix].argmax(1)!=q[ix].argmax(1)).sum()),
                mean_total_variation=float(delta.sum(1).mean()/2),max_probability_change=float(delta.max()),
                truth_assigned=False,production_assertions=0)
    return out


def comparisons(ctx,outputs):
    rows=ctx["rows"];labels=np.array([{"NOT_ENOUGH_EVIDENCE":0,"CONTRADICT":1,"SUPPORT":2}[t["label"]] for t in ctx["targets"]])
    ids=np.array([i for i,r in enumerate(rows) if r["partition"]=="evaluation"]);groups=sorted({rows[i]["group"] for i in ids})
    lookup={g:i for i,g in enumerate(groups)};member=np.array([lookup[rows[i]["group"]] for i in ids]);counts=np.bincount(member)
    draw=np.stack([np.random.default_rng(s).integers(len(groups),size=(999,len(groups))) for s in (7,17,27,37,47)])
    findings={};samples={};y=labels[ids]
    for seed in (25,26,27):
        p=outputs[f"{seed}/selected"]["probabilities"][ids].astype(float)
        for name in ("all-sentences","claim-only","evidence-only"):
            q=outputs[f"{seed}/{name}"]["probabilities"][ids].astype(float)
            delta=np.stack([(p.argmax(1)==y).astype(float)-(q.argmax(1)==y),np.log(p[np.arange(len(y)),y])-np.log(q[np.arange(len(y)),y])],1)
            sums=np.zeros((len(groups),2));np.add.at(sums,member,delta);sample=sums[draw].sum(2)/counts[draw].sum(2)[...,None]
            key=f"{seed}-{name}";samples[key]=sample;findings[key]=bounds(sample,delta.mean(0))
    return dict(rows=ids,membership=member,draw=draw),samples,dict(groups=groups,findings=findings,
        metrics=["selected_minus_control_accuracy","control_minus_selected_nll"],scope="consumed development; descriptive, not CMI or confirmatory"),contrasts(rows,labels,{k:v["probabilities"] for k,v in outputs.items()})
