"""Annotation-agreement diagnostics; an unannotated sentence is not labelled false."""
import numpy as np
from scipy.special import logsumexp
from model.scifact_readout import metrics,check


def evidence_metrics(output,ctx,truth):
    rows=ctx["rows"];labels=np.array([("NOT_ENOUGH_EVIDENCE","CONTRADICT","SUPPORT").index(t["label"]) for t in ctx["targets"]]);stop=truth["stop"]
    agreement=[]
    for i,row in enumerate(rows):
        valid=truth["valid"][row["document_index"]];path=output["paths"][i];end=int(np.flatnonzero(path==stop)[0]);picked=path[:end].tolist()
        check(picked==sorted(set(picked)) and all(valid[j] for j in picked) and np.all(path[end:]==stop),"DECODE_PATH")
        gold=[p[:-1] for p in truth["paths"][i]["alternatives"]];selected=set(picked);exact=any(selected==set(g) for g in gold);complete=any(set(g)<=selected for g in gold)
        f1=max((2*len(selected&set(g))/(len(selected)+len(g)) for g in gold),default=0.)
        logs=[]
        for j,reference in enumerate(truth["paths"][i]["alternatives"]):
            value=0.;previous=-1
            for t,choice in enumerate(reference):
                raw=output["teacher_logits"][i,j,t].astype(float);legal=np.r_[valid&(np.arange(stop)>previous),True]
                value+=raw[choice]-logsumexp(raw[legal]);previous=choice
            logs.append(value)
        accepted_log_probability=float(logsumexp(logs)) if logs else None
        check(accepted_log_probability is None or accepted_log_probability<=1e-10,"ANNOTATED_PROBABILITY")
        agreement.append(dict(case=row["id"],selected=picked,annotated=bool(gold),exact=exact,complete=complete,best_annotated_f1=f1,
            label_correct=int(output["probabilities"][i].argmax())==int(labels[i]),annotated_nll=-accepted_log_probability if logs else None))
    result={}
    for split in ("train","calibration","evaluation"):
        values=[a for a,r in zip(agreement,rows) if r["partition"]==split];annotated=[a for a in values if a["annotated"]]
        result[split]=dict(cases=len(values),annotated_cases=len(annotated),empty_selections=sum(not a["selected"] for a in values),
            mean_selected_sentences=float(np.mean([len(a["selected"]) for a in values])),
            exact_annotated_sets=sum(a["exact"] for a in annotated),complete_annotated_sets=sum(a["complete"] for a in annotated),
            label_and_exact=sum(a["label_correct"] and a["exact"] for a in annotated),label_and_complete=sum(a["label_correct"] and a["complete"] for a in annotated),
            mean_best_annotated_f1=float(np.mean([a["best_annotated_f1"] for a in annotated])),
            annotated_nll_nats=float(np.mean([a["annotated_nll"] for a in annotated])),
            positive_label_without_selected_evidence=sum(int(output["probabilities"][i].argmax())!=0 and not agreement[i]["selected"] for i,r in enumerate(rows) if r["partition"]==split))
    return dict(relation=metrics(output["probabilities"],rows,labels),evidence=result),agreement
