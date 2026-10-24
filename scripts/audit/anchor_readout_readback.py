"""Recount source support from persisted probabilities; never certify world truth.

No producer model/metric imports. CPU only reads tensors; GPU replay is recorded
by the real training path and is separately available for same-runtime replay.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit.anchor_population_readback import read as population_read,arrays,lines
from text.evaluation_multiscale import file_sha

CONFIG=dict(width=64,heads=4,questions=3,classes=6,batch=32,steps=200,
            learning_rate=0.001,weight_decay=0.01,seeds=[25,26,27],precision="fp32")


def check(ok,code):
    if not ok:raise ValueError("MNLM_ANCHOR_RESULT_READ_"+code)


def array_sha(value):
    digest=hashlib.sha256(str(value.dtype).encode()+str(value.shape).encode())
    digest.update(value.tobytes(order="C"));return digest.hexdigest()


def metrics(prob,rows):
    totals={str(q):{} for q in range(3)}
    for q in range(3):
        for split in ("train","calibration","evaluation"):
            records=[(i,r) for i,r in enumerate(rows) if r["question"]==q and r["split"]==split]
            supported=[sum(float(prob[i,j]) for j in r["positive_classes"]) for i,r in records]
            correct=sum(int(np.argmax(prob[i])) in r["positive_classes"] for i,r in records)
            totals[str(q)][split]=dict(rows=len(records),correct=correct,accuracy=correct/len(records),
                mean_supported_probability=float(np.mean(supported)),nll_nats=float(np.mean([-np.log(v) for v in supported])))
    return totals


def same_metrics(actual,expected):
    check(actual.keys()==expected.keys(),"METRICS")
    for q,values in expected.items():
        check(actual[q].keys()==values.keys(),"METRICS")
        for split,m in values.items():
            check(actual[q][split].keys()==m.keys(),"METRICS")
            for key,v in m.items():check(abs(actual[q][split][key]-v)<1e-12,"METRICS")


def probability_check(prob,rows):
    check(prob.shape==(len(rows),6) and np.isfinite(prob).all() and (prob>0).all()
          and np.allclose(prob.sum(1),1,atol=2e-6,rtol=0),"PROBABILITIES")


def verdict(prob,row):
    check(len(prob)==6 and np.isfinite(prob).all() and np.all(np.asarray(prob)>=0),"VERDICT_INPUT")
    chosen=int(np.argmax(prob))
    return dict(case=row["case"],source_value=row["classes"][chosen],source_supported=chosen in row["positive_classes"],
                asserted=False,reason="MNLM_ANCHOR_READOUT_UNADMITTED_PANEL",current_world_truth_certified=False)


def read(population,root,pin,audit_inputs=True):
    check(file_sha(root/"report.json")==pin,"REPORT_PIN");report=json.loads((root/"report.json").read_bytes())
    check(report["schema"]=="anchor-readout-diagnostic-v1" and report["config"]==CONFIG,"CONTRACT")
    check(all(report[k] is False for k in ("scientific_admission","production_assertions","shannon_cmi_claim"))
          and all(report[k] is True for k in ("source_truth_only","metadata_not_input","all_native_windows"))
          and report["checkpoint_selection"]=="last-fixed-update","CLAIM")
    actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    check(actual==set(report["files"])|{"report.json"},"FILE_SET")
    for name,f in report["files"].items():
        check((root/name).stat().st_size==f["bytes"] and file_sha(root/name)==f["sha256"],"FILE_PIN")
    population_read(population,report["population_sha256"])
    rep=json.loads((population/"report.json").read_bytes());rows=lines(population/"rows.jsonl")
    check(report["rows"]==len(rows)==1512 and report["lanes"]==rep["lanes"],"POPULATION")
    if audit_inputs:
        pins=json.loads((root/"input-pins.json").read_bytes());check(len(pins)==10,"INPUT_COUNT")
        for lane,pinned in zip(rep["lanes"],pins):
            x,ptr=arrays(population,lane);maximum=lane["max_document_windows"]
            memory=np.zeros((1512,maximum,lane["dimension"]),np.float32);mask=np.zeros((1512,maximum),bool)
            for i,r in enumerate(rows):
                # Reconstruct with an explicit native-row index, unlike the producer's slice copy.
                ids=[j for c in r["chunks"] for j in range(int(ptr[c]),int(ptr[c+1]))]
                memory[i,:len(ids)]=x[ids];mask[i,:len(ids)]=True
            check(pinned==dict(name=lane["name"],memory_sha256=array_sha(memory),mask_sha256=array_sha(mask),
                               shape=list(memory.shape),valid_windows=int(mask.sum())),"INPUT_BINDING")
    train=np.array([i for i,r in enumerate(rows) if r["split"]=="train"],np.int64)
    baseline=np.load(root/"prior.npy",allow_pickle=False);probability_check(baseline,rows)
    for q in range(3):
        counts=np.zeros(6)
        for i in train:
            r=rows[i]
            if r["question"]==q:
                for j in r["positive_classes"]:counts[j]+=1/len(r["positive_classes"])
        counts/=counts.sum()
        check(np.array_equal(baseline[[i for i,r in enumerate(rows) if r["question"]==q]],np.tile(counts,(504,1))),"PRIOR")
    base_metrics=metrics(baseline,rows);same_metrics(json.loads((root/"prior-metrics.json").read_bytes()),base_metrics)
    measured={};refusals=Counter();pairs={}
    for seed in (25,26,27):
        seedroot=root/("seed-"+str(seed));initial=torch.load(seedroot/"initial.pt",map_location="cpu",weights_only=True)
        steps=np.load(seedroot/"training-rows.npy",allow_pickle=False);rng=np.random.default_rng(seed)
        expected=np.array([rng.choice(train,32,replace=False) for _ in range(200)])
        check(np.array_equal(steps,expected),"TRAINING_SPLIT")
        names=["public-six","all-ten"]+(["all-ten-permuted"] if seed==25 else [])
        for name in names:
            folder=seedroot/name;arm=json.loads((folder/"report.json").read_bytes());active=6 if name=="public-six" else 10
            check(arm==report["arms"][str(seed)+"/"+name] and arm["seed"]==seed and arm["arm"]==name
                  and arm["active_lanes"]==active and arm["updates"]==200 and arm["permuted_training_targets"]==(name=="all-ten-permuted"),"ARM")
            mapping=np.load(folder/"target-source-rows.npy",allow_pickle=False);expected=np.arange(len(rows));rng=np.random.default_rng(seed)
            if name=="all-ten-permuted":
                for q in range(3):
                    ids=np.array([i for i in train if rows[i]["question"]==q]);expected[ids]=rng.permutation(ids)
            check(np.array_equal(mapping,expected),"TARGET_MAPPING")
            weights=torch.load(folder/"model.pt",map_location="cpu",weights_only=True)
            check(weights.keys()==initial.keys() and all(weights[k].shape==initial[k].shape and torch.isfinite(weights[k]).all() for k in weights),"WEIGHTS")
            check(arm["initial_sha256"]==file_sha(seedroot/"initial.pt") and arm["model_sha256"]==file_sha(folder/"model.pt"),"CHECKPOINT_PIN")
            count=sum(w.numel() for w in weights.values());check(count==613074==arm["parameter_count"]==report["parameter_count"],"PARAMETERS")
            changed=[k for k in initial if not torch.equal(weights[k],initial[k])]
            check(bool(changed) and changed==arm["changed_parameters"],"WEIGHT_CHANGES")
            if active==6:
                check(not any(any(k.startswith(p+str(i)+".") for p in ("projections.","window_attention.") for i in range(6,10)) for k in changed),"INACTIVE_BRANCH")
            losses=np.load(folder/"losses.npy",allow_pickle=False);check(losses.shape==(200,) and np.isfinite(losses).all() and (losses>=0).all(),"LOSSES")
            prob=np.load(folder/"probabilities.npy",allow_pickle=False);probability_check(prob,rows)
            replay=np.load(folder/"replay-probabilities.npy",allow_pickle=False)
            check(np.array_equal(prob,replay) and arm["checkpoint_replay_exact"] and arm["metadata_not_input"] and arm["senses_mutation_changes_logits"],"REPLAY")
            values=metrics(prob,rows);same_metrics(values,arm["metrics"]);same_metrics(values,json.loads((folder/"metrics.json").read_bytes()))
            measured[str(seed)+"/"+name]=values
            for i,r in enumerate(rows):refusals[verdict(prob[i],r)["reason"]]+=1
        pairs[str(seed)]={q:{s:{"accuracy_delta":measured[str(seed)+"/all-ten"][q][s]["accuracy"]-measured[str(seed)+"/public-six"][q][s]["accuracy"],
                               "nll_reduction_nats":measured[str(seed)+"/public-six"][q][s]["nll_nats"]-measured[str(seed)+"/all-ten"][q][s]["nll_nats"]}
                            for s in ("train","calibration","evaluation")} for q in ("0","1","2")}
    check(set(report["arms"])==set(measured),"ARMS")
    return dict(author_checked=True,report_sha256=pin,population_sha256=report["population_sha256"],rows=len(rows),
                arms=measured,prior=base_metrics,paired_differences=pairs,production_refusals=dict(refusals),
                scientific_admission=False,production_assertions=False,shannon_cmi_claim=False,source_truth_only=True)
