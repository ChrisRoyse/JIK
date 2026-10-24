"""Independent source-address, objective, metrics and result readback."""
import json
import math
from pathlib import Path
import numpy as np
import torch
from audit.scifact_result_readback import score,equal,array_sha
from audit.scifact_rationale_analysis_readback import percentile
from text.evaluation_multiscale import file_sha


def check(ok,code):
    if not ok:raise ValueError("MNLM_SELECTED_READ_"+code)


def expected_selection(source):
    ctx=source["ctx"];rows=ctx["rows"];selected={n:np.zeros((len(rows),42),bool) for n in ("all","none","25","26","27")}
    for i,r in enumerate(rows):
        count=ctx["documents"][r["document_index"]]["sentences"];selected["all"][i,:count]=True
        for seed,p in source["selector_paths"].items():
            choices=p[i].tolist();end=choices.index(42);values=choices[:end]
            check(values==sorted(set(values)) and all(0<=j<count for j in values),"SELECTOR_GRAMMAR")
            selected[seed][i,values]=True
    return selected


def read_plans(root,source):
    ctx=source["ctx"];rows=ctx["rows"];selected=expected_selection(source)
    with np.load(root/"selections.npz",allow_pickle=False) as z:
        check(set(z.files)==set(selected),"SELECTION_KEYS")
        for n,v in selected.items():check(z[n].dtype==bool and np.array_equal(z[n],v),"SELECTION_"+n)
    donor=np.load(root/"donors.npy",allow_pickle=False);expected=np.zeros(len(rows),np.int64)
    for split in ("train","calibration","evaluation"):
        gs=sorted({r["group"] for r in rows if r["partition"]==split});first={}
        for i,r in enumerate(rows):
            if r["partition"]==split:first.setdefault(r["group"],i)
        following={g:first[gs[(j+1)%len(gs)]] for j,g in enumerate(gs)}
        for i,r in enumerate(rows):
            if r["partition"]==split:expected[i]=following[r["group"]]
    check(donor.dtype==np.int64 and np.array_equal(donor,expected),"DONORS");counts={};signatures={}
    for name,selection in selected.items():
        counts[name]=[];signatures[name]=[]
        for k,ptr in enumerate(ctx["pointers"]):
            bags=[]
            for i,row in enumerate(rows):
                d=ctx["documents"][row["document_index"]];bag=[]
                for view in d["views"]:
                    if view["level"]!=1 or not selection[i,view["span"][0]]:continue
                    begin,end=ptr[view["unit"]:view["unit"]+2]
                    for index in range(begin,end):bag.append((index,(view["span"][0]/d["sentences"],1/d["sentences"],(index-begin)/(end-begin))))
                bags.append(bag)
            width=max(1,max(map(len,bags)));ix=np.zeros((len(rows),width),np.int64);mask=np.zeros(ix.shape,bool);geo=np.zeros((*ix.shape,3),np.float32)
            for i,bag in enumerate(bags):
                for j,(index,pos) in enumerate(bag):ix[i,j]=index;mask[i,j]=True;geo[i,j]=pos
            values=dict(indices=ix,mask=mask,geometry=geo)
            with np.load(root/name/f"{k}.npz",allow_pickle=False) as z:
                check(set(z.files)==set(values),"PLAN_KEYS")
                for n,v in values.items():check(z[n].dtype==v.dtype and np.array_equal(z[n],v),"PLAN_"+n.upper())
            counts[name].append(int(mask.sum()));signatures[name].append({n:array_sha(v) for n,v in values.items()})
    names={"selections.npz","donors.npy"}|{f"{n}/{k}.npz" for n in selected for k in range(14)}
    check({p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}==names,"PLAN_FILES")
    return dict(cases=len(rows),windows=counts,selected_sentences={n:int(v.sum()) for n,v in selected.items()},
        signatures=signatures,donors_sha256=array_sha(donor),source_native_windows_exact=True,selector_hidden_states_supplied=False,
        evidence_only_is_claim_conditioned_routing=True,scientific_admission=False,assertion_verdict="REFUSE_UNADMITTED_PANEL")


def anchor_score(p,chosen,source):
    ctx=source["ctx"];rows=ctx["rows"];targets=ctx["targets"];labels=np.array([{"NOT_ENOUGH_EVIDENCE":0,"CONTRADICT":1,"SUPPORT":2}[t["label"]] for t in targets])
    pred=p.argmax(1);out={}
    for split in ("train","calibration","evaluation"):
        ids=[i for i,r in enumerate(rows) if r["partition"]==split];counts=dict(exact_sets=0,complete_sets=0,label_and_exact=0,label_and_complete=0,
            mandatory_multi_complete=0,mandatory_multi_label_and_exact=0);multi=[];annotated=0
        for i in ids:
            gold=[set(r["sentences"]) for r in targets[i]["rationales"]]
            if not gold:continue
            annotated+=1;sel=set(np.flatnonzero(chosen[i]));ex=any(sel==g for g in gold);co=any(g.issubset(sel) for g in gold);correct=bool(pred[i]==labels[i])
            for n,v in dict(exact_sets=ex,complete_sets=co,label_and_exact=ex and correct,label_and_complete=co and correct).items():counts[n]+=int(v)
            if all(len(g)>1 for g in gold):
                multi.append(i);counts["mandatory_multi_complete"]+=int(co);counts["mandatory_multi_label_and_exact"]+=int(ex and correct)
        out[split]=dict(cases=len(ids),annotated_cases=annotated,**counts,selected_sentences=sum(int(chosen[i].sum()) for i in ids),
            missing_evidence_refusals=sum(int(not chosen[i].any()) for i in ids),
            positive_label_without_evidence=sum(int(pred[i]>0 and not chosen[i].any()) for i in ids),mandatory_multi_rows=multi)
    return dict(relation=score(p,rows,labels),evidence=out)


def intervention_score(outputs,rows):
    out={};p=outputs["probabilities"]
    for split in ("train","calibration","evaluation"):
        ids=[i for i,r in enumerate(rows) if r["partition"]==split];out[split]={}
        for name,q in outputs.items():
            if name=="probabilities":continue
            delta=[[abs(float(p[i,k])-float(q[i,k])) for k in range(3)] for i in ids]
            out[split][name]=dict(cases=len(ids),changed_labels=sum(int(p[i].argmax()!=q[i].argmax()) for i in ids),
                mean_total_variation=sum(sum(v) for v in delta)/(2*len(ids)),max_probability_change=max(max(v) for v in delta),
                truth_assigned=False,production_assertions=0)
    return out


def comparison_readback(root,outputs,ctx):
    rows=ctx["rows"];y=np.array([{"NOT_ENOUGH_EVIDENCE":0,"CONTRADICT":1,"SUPPORT":2}[t["label"]] for t in ctx["targets"]])
    ids=np.array([i for i,r in enumerate(rows) if r["partition"]=="evaluation"]);groups=sorted({rows[i]["group"] for i in ids});member=np.array([groups.index(rows[i]["group"]) for i in ids])
    seeds=(7,17,27,37,47);draw=np.array([np.random.default_rng(s).integers(len(groups),size=(999,len(groups))) for s in seeds]);findings={}
    with np.load(root/"bootstrap-plan.npz",allow_pickle=False) as z:
        check(set(z.files)=={"rows","membership","draw"} and all(np.array_equal(z[n],v) for n,v in dict(rows=ids,membership=member,draw=draw).items()),"BOOTSTRAP_PLAN")
    with np.load(root/"bootstrap-samples.npz",allow_pickle=False) as z:
        check(set(z.files)=={f"{s}-{a}" for s in (25,26,27) for a in ("all-sentences","claim-only","evidence-only")},"BOOTSTRAP_KEYS")
        for s in (25,26,27):
            for a in ("all-sentences","claim-only","evidence-only"):
                p=outputs[f"{s}/selected"]["probabilities"];q=outputs[f"{s}/{a}"]["probabilities"];key=f"{s}-{a}"
                delta=np.array([[int(p[i].argmax()==y[i])-int(q[i].argmax()==y[i]),math.log(float(p[i,y[i]]))-math.log(float(q[i,y[i]]))] for i in ids])
                counts=np.array([sum(member==g) for g in range(len(groups))]);total=np.array([delta[member==g].sum(0) for g in range(len(groups))])
                sample=total[draw].sum(2)/counts[draw].sum(2)[...,None];check(np.allclose(z[key],sample,rtol=0,atol=1e-12),"BOOTSTRAP_SAMPLE")
                per={str(seed):dict(low=percentile(sample[j],.025).tolist(),high=percentile(sample[j],.975).tolist()) for j,seed in enumerate(seeds)}
                findings[key]=dict(point=delta.mean(0).tolist(),per_seed=per,low=np.min([v["low"] for v in per.values()],0).tolist(),high=np.max([v["high"] for v in per.values()],0).tolist())
    comparison=dict(groups=groups,findings=findings,metrics=["selected_minus_control_accuracy","control_minus_selected_nll"],scope="consumed development; descriptive, not CMI or confirmatory")
    check(equal(json.loads((root/"comparisons.json").read_bytes()),comparison),"COMPARISONS")
    contrasts={}
    for split in ("train","calibration","evaluation"):
        docs={}
        for i,r in enumerate(rows):
            if r["partition"]==split:docs.setdefault(r["document"],[]).append(i)
        docs={d:v for d,v in docs.items() if any(y[i]==1 for i in v) and any(y[i]==2 for i in v)}
        pairs=[(i,j) for v in docs.values() for i in v for j in v if y[i]==2 and y[j]==1];arms={}
        for key,o in outputs.items():
            p=o["probabilities"];both=sum(int(p[i].argmax()==2 and p[j].argmax()==1) for i,j in pairs)
            arms[key]=dict(both_correct=both,predicted_labels_differ=sum(int(p[i].argmax()!=p[j].argmax()) for i,j in pairs),
                both_probabilities_ordered=sum(int(p[i,2]>p[j,2] and p[j,1]>p[i,1]) for i,j in pairs),both_correct_rate=both/len(pairs) if pairs else None)
        contrasts[split]=dict(documents=len(docs),pairs=[list(v) for v in pairs],arms=arms)
    check(equal(json.loads((root/"contrasts.json").read_bytes()),contrasts),"CONTRASTS")
    return comparison,contrasts


def read_result(root,pin,source):
    check(file_sha(root/"report.json")==pin,"RESULT_PIN");r=json.loads((root/"report.json").read_bytes());ctx=source["ctx"];rows=ctx["rows"]
    from model.scifact_selected_source import PINS
    check(r["schema"]=="scifact-selected-results-v1" and r["parents"]==PINS and r["complete"] is True and r["errors"]==[],"STATUS")
    check(r["assertion_verdict"]=="REFUSE_UNADMITTED_PANEL" and all(r[k] is False for k in ("served","scientific_admission","sealed_evaluation")),"CLAIM")
    check({p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}==set(r["files"])|{"report.json"},"FILES")
    for n,v in r["files"].items():check(file_sha(root/n)==v["sha256"] and (root/n).stat().st_size==v["bytes"],"FILE_HASH")
    load=lambda n:json.loads((root/n).read_bytes());check(load("before.json")==load("after.json"),"SOURCE_DRIFT")
    check(load("parents.json")==PINS,"PARENTS");coldplan=read_plans(root/"inputs",source);check(load("plan-readback.json")==coldplan,"PLAN_READBACK")
    runtime=load("runtime.json");check(runtime["platform"]=="linux" and bool(runtime["cuda"]),"RUNTIME")
    for n,v in runtime["code"].items():check(file_sha(Path(__file__).parents[1]/n)==v,"CODE_DRIFT")
    config=load("config.json");check(config==dict(width=64,heads=4,classes=3,batch=32,steps=200,learning_rate=.001,weight_decay=.01,seeds=[25,26,27],precision="fp32",
        confidence_thresholds=[.5,.75,.9,.95,.99],arms=["all-sentences","selected","claim-only","evidence-only","shuffled"],
        absence="learned-null-not-a-reading",selector="frozen-prior-greedy-source-indices"),"CONFIG")
    selected=expected_selection(source);labels=np.array([{"NOT_ENOUGH_EVIDENCE":0,"CONTRADICT":1,"SUPPORT":2}[t["label"]] for t in ctx["targets"]],np.int64)
    check(np.array_equal(np.load(root/"labels.npy"),labels),"LABELS");train=np.array([i for i,r in enumerate(rows) if r["partition"]=="train"])
    outputs={};arms={};expected_arms={f"{s}/{n}" for s in (25,26,27) for n in config["arms"] if n!="shuffled" or s==25};check(set(r["arms"])==expected_arms,"ARMS")
    for s in (25,26,27):
        rng=np.random.default_rng(s);steps=np.array([rng.choice(train,32,replace=False) for _ in range(200)]);seedroot=root/f"seed-{s}"
        check(np.array_equal(np.load(seedroot/"training-rows.npy"),steps),"SCHEDULE")
        initial=torch.load(seedroot/"initial.pt",map_location="cpu",weights_only=True)
        original=torch.load(source["baseline"]/f"seed-{s}/initial.pt",map_location="cpu",weights_only=True)
        check(set(initial)==set(original)|{"absence"} and all(torch.equal(v,initial[k]) for k,v in original.items()),"BASE_INITIALIZATION")
        check(initial["absence"].shape==(14,1,64) and not initial["absence"].any(),"INITIAL_ABSENCE")
        for n in config["arms"]:
            key=f"{s}/{n}"
            if key not in expected_arms:continue
            folder=seedroot/n;ar=json.loads((folder/"report.json").read_bytes());plan="all" if n=="all-sentences" else "none" if n=="claim-only" else str(s)
            check(file_sha(folder/"report.json")==r["arms"][key] and ar["seed"]==s and ar["arm"]==n and ar["plan"]==plan and ar["scope"]==("evidence" if n=="evidence-only" else "joint"),"ARM_BINDING")
            check(ar["model_sha256"]==file_sha(folder/"model.pt") and ar["initial_sha256"]==file_sha(seedroot/"initial.pt"),"CHECKPOINT_PIN")
            target=labels[train].copy()
            if n=="shuffled":target=np.random.default_rng(s).permutation(target)
            check(np.array_equal(np.load(folder/"training-labels.npy"),target),"TRAINING_TARGET")
            with np.load(folder/"objective.npz",allow_pickle=False) as z:
                check(set(z.files)=={"rows","labels","logits"} and np.array_equal(z["rows"],steps[0]) and np.array_equal(z["labels"],target[np.searchsorted(train,steps[0])]),"OBJECTIVE_TARGET")
                losses=[]
                for v,y in zip(z["logits"].astype(float),z["labels"]):peak=v.max();losses.append(peak+math.log(math.fsum(math.exp(t-peak) for t in v))-v[y])
                loss=sum(losses)/len(losses)
            probe=json.loads((folder/"objective.json").read_bytes());check(abs(probe["loss"]-loss)<1e-5,"OBJECTIVE_LOSS")
            state=torch.load(folder/"model.pt",map_location="cpu",weights_only=True)
            check(set(state)==set(initial)==set(probe["gradients"]) and any(not torch.equal(v,initial[k]) for k,v in state.items()),"LEARNING")
            check(all(v.dtype==torch.float32 and v.shape==initial[k].shape and torch.isfinite(v).all() for k,v in state.items()) and ar["parameters"]==sum(v.numel() for v in state.values()),"CHECKPOINT_SHAPE")
            lossarray=np.load(folder/"losses.npy");check(lossarray.shape==(200,) and np.isfinite(lossarray).all() and (lossarray>=0).all() and abs(lossarray[0]-loss)<1e-5,"LOSS_TRACE")
            timing=json.loads((folder/"training-time.json").read_bytes());check(timing["updates"]==200 and timing["case_updates"]==6400 and timing["seconds"]>0 and timing["peak_reserved_bytes"]>=timing["peak_allocated_bytes"]>0,"TRAINING_TIME")
            with np.load(folder/"outputs.npz",allow_pickle=False) as z:o={k:z[k].copy() for k in z.files}
            check(set(o)==({"probabilities","removed","query_swapped","evidence_swapped"} if n=="selected" else {"probabilities"}),"OUTPUT_KEYS")
            for k,v in o.items():
                check(v.dtype==np.float32 and v.shape==(len(rows),3) and ar["output_sha256"][k]==array_sha(v),"OUTPUT")
                score(v,rows,labels)
            metric=anchor_score(o["probabilities"],selected[plan],source);check(equal(json.loads((folder/"metrics.json").read_bytes()),metric),"METRICS")
            effects=intervention_score(o,rows);check(equal(json.loads((folder/"interventions.json").read_bytes()),effects),"INTERVENTIONS")
            control=json.loads((folder/"controls.json").read_bytes())
            if n=="selected":check(control==dict(gold_rejected=True,excluded_native_mutations=14,selected_native_mutations=14,excluded_packed_and_output_exact=True,selected_output_changed=True,restored_output_exact=True),"CONTROLS")
            else:check(control is None,"CONTROL_SCOPE")
            outputs[key]=o;arms[key]=dict(metrics=metric,interventions=effects,training=timing,parameters=ar["parameters"],model_sha256=ar["model_sha256"])
    comparison,contrasts=comparison_readback(root,outputs,ctx)
    check(r["phase_coverage"]>=.9 and abs(sum(r["phases_seconds"].values())/r["wall_seconds"]-r["phase_coverage"])<1e-12,"PHASES")
    return dict(cold=dict(report_sha256=pin,parents=PINS,arms=arms,plans=coldplan,comparisons=comparison,contrasts=contrasts,
                         scientific_admission=False,assertion_verdict="REFUSE_UNADMITTED_PANEL"),outputs=outputs,runtime=runtime)


def read_replay(root,pin,result):
    check(file_sha(root/"report.json")==pin,"REPLAY_PIN");r=json.loads((root/"report.json").read_bytes())
    check(r["schema"]=="scifact-selected-replay-v1" and r["result_sha256"]==result["cold"]["report_sha256"],"REPLAY_BINDING")
    check(r["complete"] is True and r["inference_mode"] is True and r["parameters_require_grad_preserved"] is True
        and r["served"] is False and r["scientific_admission"] is False and r["assertion_verdict"]=="REFUSE_UNADMITTED_PANEL","REPLAY_CLAIM")
    names={"before.json","after.json","cold-readback.json","gpu-before.txt","runtime.json"}
    for key in result["outputs"]:names|={key.replace('/','-')+'/'+n for n in ("report.json","outputs.npz")}
    check(set(r["files"])==names and {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}==names|{"report.json"},"REPLAY_FILES")
    for n,v in r["files"].items():check(file_sha(root/n)==v["sha256"] and (root/n).stat().st_size==v["bytes"],"REPLAY_FILE_HASH")
    load=lambda n:json.loads((root/n).read_bytes());check(load("before.json")==load("after.json"),"REPLAY_SOURCE_DRIFT")
    check(equal(load("cold-readback.json"),result["cold"]),"REPLAY_COLD")
    check(load("runtime.json")=={k:v for k,v in result["runtime"].items() if k!="code"},"REPLAY_RUNTIME")
    check(set(r["arms"])==set(result["outputs"]),"REPLAY_ARMS");arrays=0
    for key,o in result["outputs"].items():
        folder=key.replace('/','-');arm=load(folder+"/report.json")
        check(file_sha(root/folder/"report.json")==r["arms"][key] and arm["model_sha256"]==result["cold"]["arms"][key]["model_sha256"],"REPLAY_CHECKPOINT")
        check(arm["cases"]==1258 and arm["output_arrays"]==len(o) and arm["all_outputs_exact"] is True,"REPLAY_COUNT")
        with np.load(root/folder/"outputs.npz",allow_pickle=False) as z:
            check(set(z.files)==set(o) and all(np.array_equal(z[n],v) and arm["output_sha256"][n]==array_sha(v) for n,v in o.items()),"REPLAY_OUTPUTS")
        arrays+=len(o)
    check(r["phase_coverage"]>=.9 and abs(sum(r["phases_seconds"].values())/r["wall_seconds"]-r["phase_coverage"])<1e-12,"REPLAY_PHASES")
    return dict(report_sha256=pin,result_sha256=r["result_sha256"],checkpoints=len(result["outputs"]),output_arrays=arrays,
        replayed_case_vectors=arrays*1258,all_exact=True,scientific_admission=False,assertion_verdict="REFUSE_UNADMITTED_PANEL")
