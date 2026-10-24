"""Fixed source-description controls and paired component-bootstrap diagnostics."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit.description_baseline import fit,donors,bounds
from audit.scifact_population_readback import reference
from audit.scifact_result_readback import read_replayed,check
from model.scifact_readout import metrics
from senses.trajectory_population import publish
from text.evaluation_multiscale import file_sha

CONFIG=dict(kinds=["word","char3"],scopes=["claim","evidence","pair"],k=5,prior_weight=.1,
            seeds=[7,17,27,37,47],resamples=999,source_only=True)


def texts_for(inventory,rows):
    inv=json.loads(inventory.read_bytes());root=Path(inv["publication"]["path"])
    read=lambda n:[json.loads(line) for line in (root/n).read_bytes().splitlines()]
    claims={r["id"]:r["text"] for r in read("claims.jsonl")}
    docs={r["id"]:r["title"]+'\n'+'\n'.join(r["views"]["source_sentences"]) for r in read("documents.jsonl")}
    claim=[claims[r["claim"]] for r in rows];evidence=[docs[r["document"]] for r in rows]
    return dict(claim=claim,evidence=evidence,pair=[c+'\n'+e for c,e in zip(claim,evidence)])


def contrasts(rows,labels,probabilities):
    output={}
    for split in ("train","calibration","evaluation"):
        groups={}
        for i,r in enumerate(rows):
            if r["partition"]==split:groups.setdefault(r["document"],[]).append(i)
        selected={d:ids for d,ids in groups.items() if {1,2}<=set(labels[ids])}
        pairs=[(i,j) for ids in selected.values() for i in ids if labels[i]==2 for j in ids if labels[j]==1]
        arms={}
        for name,p in probabilities.items():
            both=sum(int(p[i].argmax()==2 and p[j].argmax()==1) for i,j in pairs)
            different=sum(int(p[i].argmax()!=p[j].argmax()) for i,j in pairs)
            ordered=sum(int(p[i,2]>p[j,2] and p[j,1]>p[i,1]) for i,j in pairs)
            arms[name]=dict(both_correct=both,predicted_labels_differ=different,both_probabilities_ordered=ordered,
                both_correct_rate=both/len(pairs) if pairs else None)
        output[split]=dict(documents=len(selected),pairs=[list(v) for v in pairs],arms=arms)
    return output


def compare(root,rows,labels,probabilities):
    ids=np.array([i for i,r in enumerate(rows) if r["partition"]=="evaluation"]);groups=sorted({rows[i]["group"] for i in ids})
    membership=np.array([groups.index(rows[i]["group"]) for i in ids]);counts=np.bincount(membership)
    draw=np.array([np.random.default_rng(s).integers(0,len(groups),size=(999,len(groups))) for s in CONFIG["seeds"]],np.int32)
    np.savez(root/"bootstrap-plan.npz",rows=ids,membership=membership,counts=counts,draw=draw)
    native=[n for n in probabilities if '/' in n];base=[n for n in probabilities if '/' not in n]
    pairs=[(n,b) for n in native for b in base]+[(str(s)+"/joint-all",str(s)+"/"+n) for s in (25,26,27)
        for n in ("joint-public","claim-only","evidence-only","sentences-only")]
    samples={};findings={};denom=counts[draw].sum(2)
    for name,control in pairs:
        p=probabilities[name].astype(np.float64);b=probabilities[control].astype(np.float64)
        delta=np.stack([(p[ids].argmax(1)==labels[ids]).astype(float)-(b[ids].argmax(1)==labels[ids]),
            np.log(p[ids,labels[ids]])-np.log(b[ids,labels[ids]])],axis=1)
        sums=np.zeros((len(groups),2));np.add.at(sums,membership,delta);value=sums[draw].sum(2)/denom[:,:,None]
        key=name+'|'+control;samples[key]=value;findings[key]=bounds(value,delta.mean(0))
    np.savez(root/"bootstrap-samples.npz",**samples)
    report=dict(groups=groups,metrics=["head_minus_control_accuracy","control_minus_head_nll_nats"],findings=findings,
        scope="fixed evaluation overlap-component descriptive bootstrap; no IID, population-risk or Shannon CMI certificate")
    publish(root/"paired-comparisons.json",report)


def main():
    p=argparse.ArgumentParser()
    for n in ("population","inventory","result","replay","output"):p.add_argument("--"+n,type=Path,required=True)
    for n in ("population-sha256","result-sha256","replay-sha256"):p.add_argument("--"+n,required=True)
    a=p.parse_args();torch.set_num_threads(1);check(not a.output.exists(),"OUTPUT_EXISTS");start=time.perf_counter();phases={}
    ctx=reference(a.inventory);cold=read_replayed(a.result,a.result_sha256,a.population,a.population_sha256,a.inventory,a.replay,a.replay_sha256,ctx)
    a.output.mkdir(parents=True);publish(a.output/"config.json",CONFIG);rows=ctx["rows"];texts=texts_for(a.inventory,rows)
    labels=np.array([("NOT_ENOUGH_EVIDENCE","CONTRADICT","SUPPORT").index(t["label"]) for t in ctx["targets"]],np.int64)
    ids=np.arange(len(rows));train=np.array([i for i,r in enumerate(rows) if r["partition"]=="train"]);groups=[r["group"] for r in rows]
    prior=np.bincount(labels[train],minlength=3)/len(train);target=np.eye(3)[labels];probabilities={"prior":np.tile(prior,(len(rows),1))}
    publish(a.output/"native-readback.json",cold);bindings=[dict(case=r["id"],group=r["group"],partition=r["partition"],
        text_sha256={k:__import__('hashlib').sha256(v[i].encode()).hexdigest() for k,v in texts.items()}) for i,r in enumerate(rows)]
    publish(a.output/"input-rows.json",bindings)
    phases["source_and_complete_native_result_readback"]=time.perf_counter()-start;tick=time.perf_counter()
    for kind in CONFIG["kinds"]:
        for scope in CONFIG["scopes"]:
            name=kind+'-'+scope;folder=a.output/name;folder.mkdir();features=fit(kind,texts[scope],train,folder,ids)
            chosen,scores=donors(features,ids,train,groups,kind);prob=np.empty((len(rows),3),np.float64)
            for i,ix in enumerate(chosen):
                selected=ix[ix>=0];check(all(rows[j]["partition"]=="train" and groups[j]!=groups[i] for j in selected),"DONOR_SCOPE")
                prob[i]=.9*target[selected].mean(0)+.1*prior if len(selected) else prior
            np.save(folder/"donors.npy",chosen);np.save(folder/"similarities.npy",scores);np.save(folder/"probabilities.npy",prob)
            publish(folder/"metrics.json",metrics(prob,rows,labels));probabilities[name]=prob
            phases[name+"_features_donors_predictions"]=time.perf_counter()-tick;tick=time.perf_counter()
    for key in cold["arms"]:
        seed,name=key.split('/');probabilities[key]=np.load(a.result/("seed-"+seed)/name/"probabilities.npy",allow_pickle=False)
    compare(a.output,rows,labels,probabilities);publish(a.output/"same-document-contrasts.json",contrasts(rows,labels,probabilities))
    phases["component_bootstrap_and_same_document_contrasts"]=time.perf_counter()-tick;tick=time.perf_counter()
    check(all(file_sha(Path(path))==pin for path,pin in ctx["sources"].items()),"SOURCE_DRIFT")
    check(file_sha(a.result/"report.json")==a.result_sha256 and file_sha(a.replay/"report.json")==a.replay_sha256,"RESULT_DRIFT")
    files={p.relative_to(a.output).as_posix():dict(bytes=p.stat().st_size,sha256=file_sha(p)) for p in a.output.rglob('*') if p.is_file()}
    phases["source_drift_and_output_pins"]=time.perf_counter()-tick;wall=time.perf_counter()-start
    report=dict(schema="scifact-comparisons-v1",population_sha256=a.population_sha256,result_sha256=a.result_sha256,replay_sha256=a.replay_sha256,
        files=files,scientific_admission=False,baseline_is_sense=False,shannon_cmi_claim=False,served=False,
        assertion_verdict="REFUSE_UNADMITTED_PANEL",phases_seconds=phases,wall_seconds=wall,phase_coverage=sum(phases.values())/wall,
        code_sha256=file_sha(Path(__file__)))
    publish(a.output/"report.json",report);print(json.dumps(dict(report_sha256=file_sha(a.output/"report.json"),wall_seconds=wall)),flush=True)


if __name__=="__main__":main()
