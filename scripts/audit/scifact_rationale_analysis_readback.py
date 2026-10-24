"""Independent finite-source positional, conjunction, contrast and bootstrap arithmetic."""
from fractions import Fraction
import numpy as np
from audit.description_readback import percentile


def derive(ctx,truth,inspected):
    rows=ctx["rows"];label=[{"NOT_ENOUGH_EVIDENCE":0,"CONTRADICT":1,"SUPPORT":2}[r["label"]] for r in ctx["targets"]]
    train=[i for i,r in enumerate(rows) if r["partition"]=="train" and truth["paths"][i]["alternatives"]];h=[Fraction(0) for _ in range(10)]
    for i in train:
        size=ctx["documents"][rows[i]["document_index"]]["sentences"];paths=truth["paths"][i]["alternatives"]
        for p in paths:
            for sid in p[:-1]:h[(sid*9)//max(size-1,1)]+=Fraction(1,len(paths)*(len(p)-1))
    histogram=[float(x) for x in h];predicted={k:[] for k in ("first","last","train-relative-position")}
    for r in rows:
        count=ctx["documents"][r["document_index"]]["sentences"];bins=[(j*9)//max(count-1,1) for j in range(count)]
        predicted["first"].append(0);predicted["last"].append(count-1);predicted["train-relative-position"].append(sorted(range(count),key=lambda j:(-histogram[bins[j]],j))[0])
    scores={};multi={};contrasts={}
    for split in ("train","calibration","evaluation"):
        ids=[i for i,r in enumerate(rows) if r["partition"]==split];annotated=[i for i in ids if truth["paths"][i]["alternatives"]];subset={}
        for name,choices in predicted.items():
            exact=complete=0;f1=[]
            for i in annotated:
                sets=[set(p[:-1]) for p in truth["paths"][i]["alternatives"]];s={choices[i]}
                exact+=int(any(g==s for g in sets));complete+=int(any(all(j in s for j in g) for g in sets));f1.append(max(2*len(s&g)/(len(g)+1) for g in sets))
            subset[name]=dict(annotated_cases=len(annotated),exact_annotated_sets=exact,complete_annotated_sets=complete,mean_best_annotated_f1=sum(f1)/len(f1))
        scores[split]=subset;conjunction=[i for i in annotated if min(len(p) for p in truth["paths"][i]["alternatives"])>2];multi_arms={}
        for key,a in inspected["agreements"].items():
            multi_arms[key]=dict(exact=sum(int(a[i]["exact"]) for i in conjunction),complete=sum(int(a[i]["complete"]) for i in conjunction),
                label_and_exact=sum(int(a[i]["label_correct"] and a[i]["exact"]) for i in conjunction))
        multi[split]=dict(case_rows=conjunction,cases=len(conjunction),arms=multi_arms)
        documents={}
        for i in ids:documents.setdefault(rows[i]["document"],[]).append(i)
        documents={d:ix for d,ix in documents.items() if 1 in [label[i] for i in ix] and 2 in [label[i] for i in ix]}
        pairs=[(i,j) for ix in documents.values() for i in ix for j in ix if label[i]==2 and label[j]==1];arms={}
        for key,o in inspected["outputs"].items():
            p=o["probabilities"];a=inspected["agreements"][key];both=sum(int(p[i].argmax()==2 and p[j].argmax()==1) for i,j in pairs)
            arms[key]=dict(both_correct=both,predicted_labels_differ=sum(int(p[i].argmax()!=p[j].argmax()) for i,j in pairs),
                both_probabilities_ordered=sum(int(p[i,2]>p[j,2] and p[j,1]>p[i,1]) for i,j in pairs),both_correct_rate=both/len(pairs) if pairs else None,
                both_labels_and_exact_sets=sum(int(a[i]["label_correct"] and a[j]["label_correct"] and a[i]["exact"] and a[j]["exact"]) for i,j in pairs))
        contrasts[split]=dict(documents=len(documents),pairs=[list(v) for v in pairs],arms=arms)
    position=dict(training_case_rows=train,training_position_histogram=histogram,selected=predicted,metrics=scores,baseline_is_sense=False)
    ids=np.array([i for i,r in enumerate(rows) if r["partition"]=="evaluation"]);groups=sorted({rows[i]["group"] for i in ids});member=np.array([groups.index(rows[i]["group"]) for i in ids])
    draw=np.stack([np.random.default_rng(seed).integers(len(groups),size=(999,len(groups))) for seed in (7,17,27,37,47)]);samples={};findings={}
    for seed in (25,26,27):
        head=str(seed)+"/rationale";base=str(seed)+"/relation-only";delta=[];mask=[]
        for i in ids:
            a=inspected["agreements"][head][i];b=inspected["agreements"][base][i];p=inspected["outputs"][head]["probabilities"][i];q=inspected["outputs"][base]["probabilities"][i]
            active=a["annotated"];mask.append([1,1]+[int(active)]*6)
            delta.append([int(p.argmax()==label[i])-int(q.argmax()==label[i]),np.log(float(p[label[i]]))-np.log(float(q[label[i]])),
                int(a["exact"])-int(b["exact"]),int(a["complete"])-int(b["complete"]),
                int(a["label_correct"] and a["exact"])-int(b["label_correct"] and b["exact"]),
                int(a["label_correct"] and a["complete"])-int(b["label_correct"] and b["complete"]),
                (b["annotated_nll"]-a["annotated_nll"]) if active else 0.,a["best_annotated_f1"]-b["best_annotated_f1"]])
        mask=np.array(mask);delta=np.array(delta)*mask;totals=np.stack([delta[member==g].sum(0) for g in range(len(groups))]);counts=np.stack([mask[member==g].sum(0) for g in range(len(groups))])
        sample=totals[draw].sum(2)/counts[draw].sum(2);samples[str(seed)]=sample
        per={str(s):dict(low=percentile(sample[j],.025).tolist(),high=percentile(sample[j],.975).tolist()) for j,s in enumerate((7,17,27,37,47))}
        findings[str(seed)]=dict(point=(delta.sum(0)/mask.sum(0)).tolist(),per_seed=per,low=np.min([x["low"] for x in per.values()],axis=0).tolist(),high=np.max([x["high"] for x in per.values()],axis=0).tolist())
    metrics=["rationale_minus_control_accuracy","control_minus_rationale_nll","rationale_minus_control_exact_set","rationale_minus_control_complete_set",
        "rationale_minus_control_label_and_exact","rationale_minus_control_label_and_complete","control_minus_rationale_annotated_nll","rationale_minus_control_annotated_f1"]
    comparison=dict(groups=groups,metrics=metrics,findings=findings,denominators=mask.sum(0).tolist(),scope="consumed development population; descriptive component bootstrap, not CMI or confirmatory risk")
    return dict(position=position,multi=multi,contrasts=contrasts,plan=dict(rows=ids,membership=member,draw=draw),samples=samples,comparisons=comparison)
