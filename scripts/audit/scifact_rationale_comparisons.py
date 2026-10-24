"""Fixed development comparisons, position-only controls, and conjunction subsets."""
import numpy as np
from audit.description_baseline import bounds
from audit.scifact_comparisons import contrasts

METRICS=["rationale_minus_control_accuracy","control_minus_rationale_nll","rationale_minus_control_exact_set",
         "rationale_minus_control_complete_set","rationale_minus_control_label_and_exact","rationale_minus_control_label_and_complete",
         "control_minus_rationale_annotated_nll","rationale_minus_control_annotated_f1"]


def row_values(output,agreement,labels):
    probability=output["probabilities"].astype(float);annotated=np.array([r["annotated"] for r in agreement])
    values=np.array([[int(p.argmax()==labels[i]),-np.log(p[labels[i]]),int(a["exact"]),int(a["complete"]),
        int(a["label_correct"] and a["exact"]),int(a["label_correct"] and a["complete"]),a["annotated_nll"] or 0.,a["best_annotated_f1"]]
        for i,(p,a) in enumerate(zip(probability,agreement))])
    valid=np.ones(values.shape,bool);valid[:,2:]=annotated[:,None]
    return values,valid


def paired(ctx,inspected):
    rows=ctx["rows"];labels=np.array([{"NOT_ENOUGH_EVIDENCE":0,"CONTRADICT":1,"SUPPORT":2}[r["label"]] for r in ctx["targets"]])
    ids=np.array([i for i,r in enumerate(rows) if r["partition"]=="evaluation"]);groups=sorted({rows[i]["group"] for i in ids});membership=np.array([groups.index(rows[i]["group"]) for i in ids])
    draw=np.array([np.random.default_rng(seed).integers(0,len(groups),size=(999,len(groups))) for seed in (7,17,27,37,47)],np.int32)
    weights=np.array([np.bincount(d,minlength=len(groups)) for d in draw.reshape(-1,len(groups))]);samples={};findings={}
    for seed in (25,26,27):
        key=str(seed)+"/rationale";base=str(seed)+"/relation-only"
        a,valid=row_values(inspected["outputs"][key],inspected["agreements"][key],labels)
        b,_=row_values(inspected["outputs"][base],inspected["agreements"][base],labels)
        delta=(a-b)*np.array([1,-1,1,1,1,1,-1,1]);delta=delta[ids]*valid[ids]
        sums=np.zeros((len(groups),8));counts=np.zeros_like(sums);np.add.at(sums,membership,delta);np.add.at(counts,membership,valid[ids])
        denominator=weights@counts;assert (denominator>0).all();sample=((weights@sums)/denominator).reshape(5,999,8)
        samples[str(seed)]=sample;findings[str(seed)]=bounds(sample,delta.sum(0)/valid[ids].sum(0))
    report=dict(groups=groups,metrics=METRICS,findings=findings,denominators=valid[ids].sum(0).tolist(),
        scope="consumed development population; descriptive component bootstrap, not CMI or confirmatory risk")
    return dict(rows=ids,membership=membership,draw=draw),samples,report


def diagnostics(ctx,truth,inspected):
    rows=ctx["rows"];hist=np.zeros(10);train=[]
    for i,row in enumerate(rows):
        paths=truth["paths"][i]["alternatives"]
        if row["partition"]!="train" or not paths:continue
        train.append(i);n=ctx["documents"][row["document_index"]]["sentences"]
        for path in paths:
            source=path[:-1]
            for index in source:hist[(index*9)//max(1,n-1)]+=1/(len(paths)*len(source))
    selected={k:[] for k in ("first","last","train-relative-position")}
    for row in rows:
        n=ctx["documents"][row["document_index"]]["sentences"]
        selected["first"].append(0);selected["last"].append(n-1)
        selected["train-relative-position"].append(max(range(n),key=lambda j:hist[(j*9)//max(1,n-1)]))
    control_scores={};multi={}
    for split in ("train","calibration","evaluation"):
        ids=[i for i,r in enumerate(rows) if r["partition"]==split and truth["paths"][i]["alternatives"]];values={}
        for name,picks in selected.items():
            exact=complete=0;f1=[]
            for i in ids:
                sets=[set(p[:-1]) for p in truth["paths"][i]["alternatives"]];chosen={picks[i]}
                exact+=any(chosen==s for s in sets);complete+=any(s<=chosen for s in sets)
                f1.append(max(2*len(chosen&s)/(1+len(s)) for s in sets))
            values[name]=dict(annotated_cases=len(ids),exact_annotated_sets=exact,complete_annotated_sets=complete,mean_best_annotated_f1=float(np.mean(f1)))
        control_scores[split]=values;conjunction=[i for i in ids if all(len(p)>2 for p in truth["paths"][i]["alternatives"])]
        multi[split]=dict(case_rows=conjunction,cases=len(conjunction),arms={k:dict(exact=sum(a[i]["exact"] for i in conjunction),
            complete=sum(a[i]["complete"] for i in conjunction),label_and_exact=sum(a[i]["label_correct"] and a[i]["exact"] for i in conjunction)) for k,a in inspected["agreements"].items()})
    labels=np.array([{"NOT_ENOUGH_EVIDENCE":0,"CONTRADICT":1,"SUPPORT":2}[r["label"]] for r in ctx["targets"]])
    pairs=contrasts(rows,labels,{k:v["probabilities"] for k,v in inspected["outputs"].items()})
    for part in pairs.values():
        for name,arm in part["arms"].items():
            a=inspected["agreements"][name];arm["both_labels_and_exact_sets"]=sum(a[i]["label_correct"] and a[i]["exact"] and a[j]["label_correct"] and a[j]["exact"] for i,j in part["pairs"])
    return dict(training_case_rows=train,training_position_histogram=hist.tolist(),selected=selected,metrics=control_scores,
        baseline_is_sense=False),multi,pairs
