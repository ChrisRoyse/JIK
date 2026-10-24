"""Independent source-grammar, complete-path likelihood and evidence-agreement audit."""
import math
import numpy as np
from audit.scifact_result_readback import score


def check(ok,code):
    if not ok:raise ValueError("MNLM_RATIONALE_SCORE_"+code)


def log_probability(raw,choice,previous,sentences,stop):
    legal=list(range(previous+1,sentences))+[stop]
    check(choice in legal,"PATH_GRAMMAR");peak=max(float(raw[k]) for k in legal)
    return float(raw[choice])-peak-math.log(math.fsum(math.exp(float(raw[k])-peak) for k in legal))


def path_nll(raw,alternatives,sentences,stop):
    values=[]
    for j,path in enumerate(alternatives):
        previous=-1;lp=0.
        for t,choice in enumerate(path):lp+=log_probability(raw[j,t],choice,previous,sentences,stop);previous=choice
        values.append(lp)
    if not values:return None
    maximum=max(values);value=-maximum-math.log(math.fsum(math.exp(x-maximum) for x in values))
    check(value>=-1e-10 and math.isfinite(value),"PATH_PROBABILITY");return value


def audit_outputs(output,ctx,truth):
    rows=ctx["rows"];stop=truth["stop"];n=len(rows)
    shapes=dict(probabilities=(n,3),paths=(n,stop+1),path_steps=(n,stop+1),greedy_logits=(n,stop+1,stop+1),teacher_logits=(n,6,5,stop+1))
    check(set(output)==set(shapes),"OUTPUT_KEYS")
    for k,s in shapes.items():
        check(output[k].shape==s and np.isfinite(output[k]).all(),"OUTPUT_SHAPE")
        check(output[k].dtype==({"paths":np.int64,"path_steps":np.bool_}.get(k,np.float32)),"OUTPUT_DTYPE")
    labels=np.array([{"NOT_ENOUGH_EVIDENCE":0,"CONTRADICT":1,"SUPPORT":2}[r["label"]] for r in ctx["targets"]]);agreements=[]
    for i,row in enumerate(rows):
        sentences=ctx["documents"][row["document_index"]]["sentences"];previous=-1;selected=[];ended=False
        for t in range(stop+1):
            choice=int(output["paths"][i,t]);active=bool(output["path_steps"][i,t]);raw=output["greedy_logits"][i,t]
            if ended:check(not active and choice==stop and not raw.any(),"AFTER_STOP");continue
            check(active,"MISSING_STEP");legal=list(range(previous+1,sentences))+[stop]
            predicted=max(legal,key=lambda k:float(raw[k]));check(predicted==choice,"GREEDY_CHOICE")
            if choice==stop:ended=True
            else:check(previous<choice<sentences,"DECODE_GRAMMAR");selected.append(choice);previous=choice
        check(ended,"NO_STOP");alternatives=truth["paths"][i]["alternatives"];gold=[set(p[:-1]) for p in alternatives];chosen=set(selected)
        loss=path_nll(output["teacher_logits"][i],alternatives,sentences,stop)
        agreements.append(dict(case=row["id"],selected=selected,annotated=bool(gold),exact=any(chosen==s for s in gold),
            complete=any(all(sid in chosen for sid in s) for s in gold),best_annotated_f1=max((2*len(chosen&s)/(len(chosen)+len(s)) for s in gold),default=0.),
            label_correct=int(np.argmax(output["probabilities"][i]))==int(labels[i]),annotated_nll=loss))
    evidence={}
    for split in ("train","calibration","evaluation"):
        ids=[i for i,r in enumerate(rows) if r["partition"]==split];annotated=[i for i in ids if agreements[i]["annotated"]]
        evidence[split]=dict(cases=len(ids),annotated_cases=len(annotated),empty_selections=sum(len(agreements[i]["selected"])==0 for i in ids),
            mean_selected_sentences=sum(len(agreements[i]["selected"]) for i in ids)/len(ids),
            exact_annotated_sets=sum(agreements[i]["exact"] for i in annotated),complete_annotated_sets=sum(agreements[i]["complete"] for i in annotated),
            label_and_exact=sum(agreements[i]["label_correct"] and agreements[i]["exact"] for i in annotated),
            label_and_complete=sum(agreements[i]["label_correct"] and agreements[i]["complete"] for i in annotated),
            mean_best_annotated_f1=sum(agreements[i]["best_annotated_f1"] for i in annotated)/len(annotated),
            annotated_nll_nats=sum(agreements[i]["annotated_nll"] for i in annotated)/len(annotated),
            positive_label_without_selected_evidence=sum(int(output["probabilities"][i].argmax())>0 and not agreements[i]["selected"] for i in ids))
    return dict(relation=score(output["probabilities"],rows,labels),evidence=evidence),agreements


def objective(probe,weight,ctx,stop):
    labels=probe["labels"];classification=probe["classification_logits"].astype(float);loss=[];aux=[]
    for j,i in enumerate(probe["case_rows"]):
        raw=classification[j];peak=float(raw.max());loss.append(peak+math.log(math.fsum(math.exp(float(v)-peak) for v in raw))-raw[labels[j]])
        paths=[probe["choices"][j,k][probe["steps"][j,k]].tolist() for k in np.flatnonzero(probe["present"][j])]
        sentences=ctx["documents"][ctx["rows"][i]["document_index"]]["sentences"]
        value=path_nll(probe["teacher_logits"][j],paths,sentences,stop)
        if value is not None:aux.append(value)
    relation=float(np.mean(loss));auxiliary=float(np.mean(aux)) if weight and aux else 0.
    return dict(relation_loss=relation,auxiliary_loss=auxiliary,total_loss=relation+weight*auxiliary)
