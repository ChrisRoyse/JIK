"""Cold source/checkpoint/prediction audit; independently reconstruct all metrics."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit.scifact_population_readback import reference,read as population_read
from text.evaluation_multiscale import file_sha

ARMS={"joint-all":(14,"joint"),"joint-public":(6,"joint"),"claim-only":(14,"claim"),
      "evidence-only":(14,"evidence"),"sentences-only":(14,"sentences"),"permuted":(14,"joint")}
CONFIG=dict(width=64,heads=4,classes=3,batch=32,steps=200,learning_rate=.001,weight_decay=.01,
            seeds=[25,26,27],precision="fp32",confidence_thresholds=[.5,.75,.9,.95,.99])


def check(ok,code):
    if not ok:raise ValueError("MNLM_RELATION_RESULT_"+code)


def array_sha(x):
    x=np.ascontiguousarray(x);h=hashlib.sha256(str(x.dtype).encode()+str(x.shape).encode());h.update(memoryview(x).cast('B'));return h.hexdigest()


def score(prob,rows,labels):
    p=np.asarray(prob,np.float64);check(p.shape==(len(rows),3) and np.isfinite(p).all() and (p>0).all()
        and np.allclose(p.sum(1),1,rtol=0,atol=1e-6),"PROBABILITY");out={}
    for part in ("train","calibration","evaluation"):
        ids=[i for i,r in enumerate(rows) if r["partition"]==part];matrix=np.zeros((3,3),np.int64)
        prediction=p[ids].argmax(1);gold=labels[ids];conf=p[ids].max(1)
        for y,yhat in zip(gold,prediction):matrix[y,yhat]+=1
        f1=[]
        for k in range(3):
            den=int(matrix[k].sum()+matrix[:,k].sum());f1.append(2*int(matrix[k,k])/den if den else 0.)
        curves={}
        for threshold in (.5,.75,.9,.95,.99):
            accepted=np.flatnonzero(conf>=threshold);wrong=int(np.sum(prediction[accepted]!=gold[accepted]));n=len(accepted)
            curves[str(threshold)]=dict(accepted=n,wrong=wrong,coverage=n/len(ids),error_rate=wrong/n if n else None,production_assertions=0)
        out[part]=dict(rows=len(ids),components=len({rows[i]["group"] for i in ids}),confusion=matrix.tolist(),
            accuracy=float(np.trace(matrix)/len(ids)),macro_f1=float(np.mean(f1)),
            nll_nats=float(np.mean([-np.log(p[i,labels[i]]) for i in ids])),
            brier=float(np.mean([sum((p[i,k]-int(labels[i]==k))**2 for k in range(3)) for i in ids])),confidence_diagnostics=curves)
    return out


def equal(a,b):
    if isinstance(a,dict):return isinstance(b,dict) and set(a)==set(b) and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,list):return isinstance(b,list) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    if isinstance(a,float):return isinstance(b,(int,float)) and abs(a-b)<=1e-12
    return type(a)==type(b) and a==b


def read(root,pin,population,population_pin,inventory,context=None):
    ctx=reference(inventory) if context is None else context;pop=population_read(population,population_pin,inventory,ctx)
    check(file_sha(root/"report.json")==pin,"PIN");rep=json.loads((root/"report.json").read_bytes())
    check(rep["schema"]=="scifact-readout-results-v1" and rep["complete"] is True and rep["errors"]==[],"COMPLETE")
    check(rep["population_sha256"]==population_pin and rep["native_readings_unchanged"] is True,"PARENTS")
    check(rep["assertion_verdict"]=="REFUSE_UNADMITTED_PANEL" and rep["scientific_admission"] is False and rep["served"] is False,"CLAIM")
    check(rep["phase_coverage"]>=.9 and abs(sum(rep["phases_seconds"].values())/rep["wall_seconds"]-rep["phase_coverage"])<1e-12,"PHASES")
    expected={"config.json","input-pins.json","population-readback.json","labels.npy","runtime.json","prior.npy","prior-metrics.json","gpu-before.txt"}
    names=[(s,n) for s in (25,26,27) for n in ARMS if n!="permuted" or s==25]
    for seed in (25,26,27):expected|={f"seed-{seed}/initial.pt",f"seed-{seed}/training-rows.npy"}
    for seed,name in names:
        expected|={f"seed-{seed}/{name}/{n}" for n in ("target-source-rows.npy","model.pt","probabilities.npy","losses.npy","replay-probabilities.npy","metrics.json","report.json")}
    check(set(rep["files"])==expected and {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}==expected|{"report.json"},"FILES")
    for name,v in rep["files"].items():check(file_sha(root/name)==v["sha256"] and (root/name).stat().st_size==v["bytes"],"FILE_HASH")
    check(rep["code_sha256"]==file_sha(Path(__file__).parents[1]/"model/train_scifact_readout.py"),"TRAINING_CODE")
    load=lambda name:json.loads((root/name).read_bytes())
    check(load("config.json")==CONFIG and load("population-readback.json")==pop,"CONFIG_POPULATION")
    runtime=load("runtime.json");check(runtime["platform"]=="linux" and runtime["cuda"] and runtime["model_code_sha256"]==file_sha(Path(__file__).parents[1]/"model/scifact_readout.py"),"RUNTIME")
    rows=ctx["rows"];labels=np.array([("NOT_ENOUGH_EVIDENCE","CONTRADICT","SUPPORT").index(t["label"]) for t in ctx["targets"]],np.int64)
    check(np.array_equal(np.load(root/"labels.npy",allow_pickle=False),labels),"TARGETS")
    pins=[dict(name=l["name"],native_sha256=array_sha(x),maps={k:array_sha(v) for k,v in m.items()}) for l,x,m in zip(ctx["lanes"],ctx["native"],ctx["maps"])]
    check(load("input-pins.json")==pins,"NATIVE_PINS")
    train=np.array([i for i,r in enumerate(rows) if r["partition"]=="train"],np.int64)
    prior=np.tile(np.bincount(labels[train],minlength=3)/len(train),(len(rows),1))
    check(np.array_equal(np.load(root/"prior.npy",allow_pickle=False),prior),"PRIOR")
    prior_score=score(prior,rows,labels);check(equal(prior_score,load("prior-metrics.json")),"METRICS")
    check(set(rep["arms"])=={f"{s}/{n}" for s,n in names},"ARMS");summaries={}
    for seed in (25,26,27):
        folder=root/f"seed-{seed}";initial=torch.load(folder/"initial.pt",map_location="cpu",weights_only=True)
        check(all(v.dtype==torch.float32 and torch.isfinite(v).all() for v in initial.values()),"INITIAL")
        rng=np.random.default_rng(seed);schedule=np.array([rng.choice(train,32,replace=False) for _ in range(200)])
        check(np.array_equal(np.load(folder/"training-rows.npy",allow_pickle=False),schedule),"SCHEDULE")
        for name in ARMS:
            if (seed,name) not in names:continue
            arm=folder/name;key=f"{seed}/{name}";ar=json.loads((arm/"report.json").read_bytes());active,scope=ARMS[name]
            check(file_sha(arm/"report.json")==rep["arms"][key]["report_sha256"],"ARM_PIN")
            check((ar["seed"],ar["arm"],ar["active_lanes"],ar["scope"],ar["updates"])==(seed,name,active,scope,200),"ARM_CONFIG")
            check(ar["initial_sha256"]==file_sha(folder/"initial.pt") and ar["model_sha256"]==file_sha(arm/"model.pt"),"CHECKPOINT_PIN")
            final=torch.load(arm/"model.pt",map_location="cpu",weights_only=True)
            check(set(final)==set(initial) and all(v.shape==initial[k].shape and v.dtype==initial[k].dtype and torch.isfinite(v).all() for k,v in final.items()),"CHECKPOINT")
            changed=[k for k in initial if not torch.equal(initial[k],final[k])]
            check(bool(changed) and changed==ar["changed_parameters"] and ar["parameter_count"]==sum(v.numel() for v in initial.values()),"LEARNING")
            if active==6:check(not any(k.startswith(p+str(i)+".") for k in changed for p in ("projections.","attention.","relations.") for i in range(6,14)),"INACTIVE_DRIFT")
            mapping=np.arange(len(rows),dtype=np.int64)
            if name=="permuted":mapping[train]=np.random.default_rng(seed).permutation(train)
            check(np.array_equal(np.load(arm/"target-source-rows.npy",allow_pickle=False),mapping),"TARGET_MAPPING")
            loss=np.load(arm/"losses.npy",allow_pickle=False);check(loss.shape==(200,) and np.isfinite(loss).all() and (loss>=0).all(),"LOSS")
            prob=np.load(arm/"probabilities.npy",allow_pickle=False);check(prob.dtype==np.float32,"PROBABILITY_DTYPE")
            check(ar["checkpoint_replay_exact"] is True and np.array_equal(prob,np.load(arm/"replay-probabilities.npy",allow_pickle=False)),"REPLAY")
            control={"labels_rejected":True,"query_ignored" if scope=="evidence" else "query_changes_logits":True,
                     "evidence_ignored" if scope=="claim" else "evidence_changes_logits":True}
            if scope=="sentences":control["non_sentence_views_ignored"]=True
            check(ar["input_controls"]==control,"CONTROLS")
            scores=score(prob,rows,labels);check(equal(scores,ar["metrics"]) and equal(scores,json.loads((arm/"metrics.json").read_bytes())),"METRICS")
            summaries[key]=dict(metrics=scores,parameters=ar["parameter_count"],model_sha256=ar["model_sha256"],predictions_sha256=array_sha(prob))
    return dict(result_sha256=pin,population_sha256=population_pin,cases=len(rows),arms=summaries,prior=prior_score,
                complete=True,assertion_verdict="REFUSE_UNADMITTED_PANEL",scientific_admission=False)


def read_replayed(root,pin,population,population_pin,inventory,replay,replay_pin,context=None):
    cold=read(root,pin,population,population_pin,inventory,context)
    check(file_sha(replay/"report.json")==replay_pin,"FRESH_PIN");rep=json.loads((replay/"report.json").read_bytes())
    check(rep["schema"]=="scifact-fresh-replay-v1" and rep["result_sha256"]==pin,"FRESH_RESULT_PIN")
    check(rep["exact"] is True and rep["cases"]==20128 and rep["source_checkpoint_native_unchanged"] is True
        and rep["checkpoint_parameter_flags_preserved"] is True and rep["inference_mode"] is True and rep["scientific_admission"] is False,"FRESH_CLAIM")
    check(set(rep["arms"])==set(cold["arms"]),"FRESH_ARMS")
    names={"before.json","after.json","cold-readback.json","gpu-before.txt"}|{key.replace('/','-')+".npy" for key in cold["arms"]}
    check(set(rep["files"])==names and {p.relative_to(replay).as_posix() for p in replay.rglob('*') if p.is_file()}==names|{"report.json"},"FRESH_FILES")
    for name,v in rep["files"].items():check(file_sha(replay/name)==v["sha256"] and (replay/name).stat().st_size==v["bytes"],"FRESH_FILE_HASH")
    check(equal(json.loads((replay/"cold-readback.json").read_bytes()),cold),"FRESH_COLD_READBACK")
    check(json.loads((replay/"before.json").read_bytes())==json.loads((replay/"after.json").read_bytes()),"FRESH_DRIFT")
    for key,meta in rep["arms"].items():
        prob=np.load(replay/(key.replace('/','-')+".npy"),allow_pickle=False)
        check(meta["exact"] is True and meta["cases"]==1258 and meta["sha256"]==array_sha(prob)==cold["arms"][key]["predictions_sha256"],"FRESH_PROBABILITIES")
    return {**cold,"fresh_replay_sha256":replay_pin,"fresh_replayed_predictions":20128}
