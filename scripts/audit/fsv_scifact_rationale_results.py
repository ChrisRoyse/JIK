"""Source-supervision, actual optimizer/checkpoint and final consumer fault boundaries."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit.scifact_rationale_result_readback import context,inspect_training,read_completed
from audit.scifact_rationale_analysis_readback import derive
from senses.contract import canonical
from senses.trajectory_population import publish
from text.evaluation_multiscale import file_sha


def main():
    p=argparse.ArgumentParser()
    for n in ("population","inventory","rationale","baseline","capture","completed","output"):p.add_argument("--"+n,type=Path,required=True)
    for n in ("population-sha256","rationale-sha256","baseline-sha256","capture-sha256","completed-sha256"):p.add_argument("--"+n,required=True)
    a=p.parse_args();torch.set_num_threads(1);start=time.perf_counter();phases={};a.output.mkdir(parents=True,exist_ok=False)
    ctx=context(a.population,a.population_sha256,a.inventory,a.rationale,a.rationale_sha256,a.baseline,a.baseline_sha256)
    inspected=inspect_training(a.capture,a.capture_sha256,ctx);analysis=derive(ctx["native"],ctx["truth"],inspected)
    cold=read_completed(a.completed,a.completed_sha256,inspected,analysis,a.population_sha256);publish(a.output/"cold-readback.json",cold)
    before=dict(ctx["native"]["sources"])
    for root in (a.population,a.rationale,a.baseline,a.capture,a.completed):before.update({str(p):file_sha(p) for p in root.rglob('*') if p.is_file()})
    publish(a.output/"before.json",before);checks=[dict(name="B1/B2/B3 full source paths, optimizer objectives, checkpoints, fresh replay, positions, conjunctions and bootstrap",passed=True)]
    phases["independent_full_capture_and_completed_readback"]=time.perf_counter()-start;tick=time.perf_counter()
    shadow=a.output/"completed-shadow";capture=a.output/"capture-shadow";shutil.copytree(a.completed,shadow);shutil.copytree(a.capture,capture);controls=a.output/"controls";controls.mkdir()
    def fault(name,path,fn,code,kind="json",training=False,mirror=False):
        base=capture if training else shadow;file=base/path;restore={path:file.read_bytes(),"report.json":(base/"report.json").read_bytes()}
        if kind=="npz":
            with np.load(file,allow_pickle=False) as z:value={k:z[k].copy() for k in z.files}
        elif kind=="npy":value=np.load(file,allow_pickle=False).copy()
        elif kind=="pt":value=torch.load(file,map_location="cpu",weights_only=True)
        else:value=json.loads(file.read_bytes())
        changed=fn(value)
        if changed is not None:value=changed
        if kind=="npz":np.savez_compressed(file,**value)
        elif kind=="npy":np.save(file,value)
        elif kind=="pt":torch.save(value,file)
        else:file.write_bytes(canonical(value))
        touched=[path]
        if mirror:
            name2=str(Path(path).with_name("replay.npz")).replace('\\','/');restore[name2]=(base/name2).read_bytes();(base/name2).write_bytes(file.read_bytes());touched.append(name2)
        if path!="report.json":
            rep=json.loads(restore["report.json"])
            for item in touched:rep["files"][item]=dict(bytes=(base/item).stat().st_size,sha256=file_sha(base/item))
            if not training and path.endswith('/report.json'):rep["arms"][path.split('/')[0].replace('-', '/',1)]=file_sha(file)
            (base/"report.json").write_bytes(canonical(rep))
        publish(controls/(name+".json"),dict(path=path,mutant_sha256=file_sha(file),training_capture=training))
        try:
            if training:
                altered=inspect_training(capture,file_sha(capture/"report.json"),ctx);read_completed(shadow,a.completed_sha256,altered,analysis,a.population_sha256)
            else:read_completed(shadow,file_sha(shadow/"report.json"),inspected,analysis,a.population_sha256)
            observed="accepted"
        except ValueError as exc:observed=str(exc)
        checks.append(dict(name=name,expected=code,observed=observed,passed=code in observed))
        for n,raw in restore.items():(base/n).write_bytes(raw)
    fault("B1_changed_label","labels.npy",lambda x:x.__setitem__(0,(x[0]+1)%3),"LABELS_OR_TRAIN","npy",True)
    fault("B1_changed_schedule","seed-25/training-rows.npy",lambda x:x.__setitem__((0,0),1258),"SCHEDULE","npy",True)
    train=[i for i,r in enumerate(ctx["native"]["rows"]) if r["partition"]=="train"]
    multi,alternative=next((j,k) for j,i in enumerate(train) for k,p in enumerate(ctx["truth"]["paths"][i]["alternatives"]) if len(p)>2)
    fault("B1_partial_AND_target","seed-25/rationale/training-targets.npz",lambda x:x["steps"].__setitem__((multi,alternative,0),False),"TARGETS","npz",True)
    fault("B2_wrong_objective","seed-25/rationale/objective-probe.json",lambda x:x.update(auxiliary_loss=999.),"OBJECTIVE",training=True)
    fault("B2_wrong_gradient","seed-25/relation-only/objective-probe.json",lambda x:x["gradients"].update({"pointer.start":1.}),"POINTER_GRADIENT",training=True)
    fault("B2_changed_loss","seed-25/rationale/losses.npy",lambda x:x.__setitem__((0,2),999.),"LOSSES","npy",True)
    def weight(x):next(iter(x.values())).view(-1)[0]+=1
    fault("B2_changed_checkpoint","seed-25/relation-only/model.pt",weight,"BASELINE_REPRODUCTION","pt",True)
    row=next(i for i,p in enumerate(inspected["outputs"]["25/rationale"]["paths"]) if p[0]!=42)
    fault("B2_inconsistent_decode","seed-25/rationale/outputs.npz",lambda x:x["paths"].__setitem__((row,0),42),"GREEDY_CHOICE","npz",True,True)
    fault("B3_wrong_capture_binding","report.json",lambda x:x.update(capture_sha256="0"*64),"CAPTURE_BINDING")
    fault("B3_wrong_source_parent","report.json",lambda x:x.update(parent_population_sha256="0"*64),"PARENTS")
    fault("B3_wrong_runtime","runtime.json",lambda x:x.update(cuda="different"),"RUNTIME")
    fault("B3_wrong_model_binding","25-rationale/report.json",lambda x:x.update(model_sha256="0"*64),"MODEL_BINDING")
    fault("B3_changed_fresh_logits","25-rationale/replay.npz",lambda x:x["teacher_logits"].__setitem__((0,0,0,0),999.),"FRESH_REPLAY","npz")
    fault("B3_wrong_metrics","25-rationale/metrics.json",lambda x:x["evidence"]["evaluation"].update(exact_annotated_sets=96),"METRICS")
    fault("B3_wrong_evidence_set","25-rationale/agreements.json",lambda x:x[0].update(selected=[41]),"METRICS")
    fault("B3_training_position_leak","position-controls.json",lambda x:x.update(training_case_rows=[]),"POSITION")
    fault("B3_false_conjunction_success","multi-sentence-only.json",lambda x:x["evaluation"]["arms"]["25/rationale"].update(exact=999),"MULTI")
    fault("B3_wrong_contrast","same-document-contrasts.json",lambda x:x["evaluation"].update(documents=999),"CONTRASTS")
    fault("B3_wrong_bootstrap_cluster","bootstrap-plan.npz",lambda x:x["membership"].__setitem__(0,999),"BOOTSTRAP_PLAN","npz")
    fault("B3_wrong_bootstrap_sample","bootstrap-samples.npz",lambda x:x["25"].__setitem__((0,0,0),999.),"BOOTSTRAP_SAMPLES","npz")
    fault("B3_wrong_interval","paired-comparisons.json",lambda x:x["findings"]["25"]["low"].__setitem__(0,999.),"COMPARISONS")
    for flag in ("scientific_admission","served","sealed_evaluation"):
        fault("B3_false_"+flag,"report.json",lambda x,k=flag:x.update({k:True}),"CLAIM")
    fault("B3_false_accept","report.json",lambda x:x.update(assertion_verdict="ACCEPT"),"CLAIM")
    phases["persisted_source_training_and_consumer_faults"]=time.perf_counter()-tick;tick=time.perf_counter()
    restored=inspect_training(capture,a.capture_sha256,ctx);final=read_completed(shadow,a.completed_sha256,restored,analysis,a.population_sha256)
    after={p:file_sha(Path(p)) for p in before};publish(a.output/"after.json",after)
    checks.append(dict(name="B1/B2/B3 all originals unchanged and exact restored captures/publication",passed=before==after and final==cold))
    phases["complete_restoration_and_source_drift"]=time.perf_counter()-tick;wall=time.perf_counter()-start
    result=dict(passed=all(c["passed"] for c in checks),checks=checks,acceptance="author-checked",scientific_admission=False,phases_seconds=phases,wall_seconds=wall,phase_coverage=sum(phases.values())/wall)
    publish(a.output/"fsv.json",result);print(json.dumps(dict(passed=result["passed"],checks=len(checks),failed=[c for c in checks if not c["passed"]],wall_seconds=wall)))
    return 0 if result["passed"] else 2


if __name__=="__main__":raise SystemExit(main())
