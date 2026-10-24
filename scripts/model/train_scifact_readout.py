"""Sixteen predeclared relation fits; final checkpoints and complete predictions."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit.scifact_population_readback import read,reference
from model.scifact_readout import CONFIG,ARMS,NativeStore,RelationReadout,array_sha,check,schedule,metrics,probabilities
from model.renderer_conditioning import gpu_lease
from lens.lenscommon import run_cli
from senses.trajectory_population import publish
from text.evaluation_multiscale import file_sha


def controls(model,store,active,scope):
    data=store.batch(range(32));checks={}
    with torch.inference_mode():
        original=model(data,active,scope)
        query=[{**m,"query":-m["query"]} for m in data]
        evidence=[{**m,"evidence":-m["evidence"]} for m in data]
        q=model(query,active,scope);e=model(evidence,active,scope)
        checks["query_ignored" if scope=="evidence" else "query_changes_logits"]=bool(torch.equal(original,q)) if scope=="evidence" else not torch.equal(original,q)
        checks["evidence_ignored" if scope=="claim" else "evidence_changes_logits"]=bool(torch.equal(original,e)) if scope=="claim" else not torch.equal(original,e)
        if scope=="sentences":
            hidden=[{**m,"evidence":torch.where((m["levels"]!=1).unsqueeze(-1),-m["evidence"],m["evidence"])} for m in data]
            checks["non_sentence_views_ignored"]=torch.equal(original,model(hidden,active,scope))
        try:model([{**m,"label":0} for m in data],active,scope);checks["labels_rejected"]=False
        except ValueError as exc:checks["labels_rejected"]="INPUT_FIELDS" in str(exc)
    check(all(checks.values()),"INPUT_DEPENDENCE");return checks


def main():
    p=argparse.ArgumentParser()
    for n in ("population","inventory","output"):p.add_argument("--"+n,type=Path,required=True)
    p.add_argument("--report-sha256",required=True);p.add_argument("--reviewed-desktop-residents",type=Path)
    a=p.parse_args();check(not a.output.exists(),"OUTPUT_EXISTS");start=time.perf_counter();phases={}
    context=reference(a.inventory);cold=read(a.population,a.report_sha256,a.inventory,context);rows=context["rows"]
    labels=np.array([("NOT_ENOUGH_EVIDENCE","CONTRADICT","SUPPORT").index(t["label"]) for t in context["targets"]],np.int64)
    train=np.array([i for i,r in enumerate(rows) if r["partition"]=="train"],np.int64)
    check(len(train)==860 and len(rows)==1258 and np.all(np.bincount(labels[train],minlength=3)>0),"POPULATION")
    pins=[dict(name=l["name"],native_sha256=array_sha(x),maps={k:array_sha(v) for k,v in m.items()}) for l,x,m in zip(context["lanes"],context["native"],context["maps"])]
    phases["source_population_and_native_readback"]=time.perf_counter()-start;tick=time.perf_counter()
    residents=gpu_lease(a.reviewed_desktop_residents,"scifact-relations-sixteen-fixed-fits")
    torch.set_num_threads(min(32,os.cpu_count() or 1));a.output.mkdir(parents=True)
    (a.output/"gpu-before.txt").write_text(residents,encoding="utf-8");publish(a.output/"config.json",CONFIG);publish(a.output/"input-pins.json",pins)
    publish(a.output/"population-readback.json",cold);np.save(a.output/"labels.npy",labels)
    store=NativeStore(context,"cuda");target=torch.from_numpy(labels).cuda();dims=[l["dimension"] for l in context["lanes"]]
    check(all(not x.requires_grad for x in store.native),"UNFROZEN_READINGS")
    runtime=dict(torch=torch.__version__,cuda=torch.version.cuda,platform=sys.platform,device=torch.cuda.get_device_name(0),
                 model_code_sha256=file_sha(Path(sys.modules[RelationReadout.__module__].__file__)))
    publish(a.output/"runtime.json",runtime)
    prior=np.repeat((np.bincount(labels[train],minlength=3)/len(train))[None,:],len(rows),axis=0)
    np.save(a.output/"prior.npy",prior);publish(a.output/"prior-metrics.json",metrics(prior,rows,labels))
    phases["lease_upload_and_training_prior"]=time.perf_counter()-tick;tick=time.perf_counter();results={};errors=[]
    for seed in CONFIG["seeds"]:
        torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);initial=RelationReadout(dims)
        state={k:v.detach().cpu().clone() for k,v in initial.state_dict().items()};parameters=sum(p.numel() for p in initial.parameters());del initial
        seedroot=a.output/("seed-"+str(seed));seedroot.mkdir();torch.save(state,seedroot/"initial.pt")
        steps=schedule(rows,seed);np.save(seedroot/"training-rows.npy",steps)
        for name,(active,scope) in ARMS.items():
            if name=="permuted" and seed!=25:continue
            root=seedroot/name;root.mkdir();at=time.perf_counter();model=opt=cold_model=None
            try:
                mapping=np.arange(len(rows),dtype=np.int64)
                if name=="permuted":mapping[train]=np.random.default_rng(seed).permutation(train)
                np.save(root/"target-source-rows.npy",mapping);fit_target=target[torch.from_numpy(mapping).cuda()]
                model=RelationReadout(dims).cuda();model.load_state_dict(state)
                opt=torch.optim.AdamW(model.parameters(),lr=CONFIG["learning_rate"],weight_decay=CONFIG["weight_decay"])
                torch.cuda.reset_peak_memory_stats();losses=[]
                for ids in steps:
                    index=torch.from_numpy(ids).cuda();opt.zero_grad(set_to_none=True)
                    logits=model(store.batch(index),active,scope);loss=torch.nn.functional.cross_entropy(logits,fit_target[index])
                    check(bool(torch.isfinite(loss)),"NONFINITE_LOSS");loss.backward();opt.step();losses.append(float(loss.detach()))
                model.eval();prob=probabilities(model,store,rows,active,scope)
                check(np.isfinite(prob).all() and np.all(prob>0) and np.allclose(prob.sum(1),1,atol=1e-6),"PROBABILITIES")
                learned={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                changed=[k for k in state if not torch.equal(state[k],learned[k])];check(bool(changed),"NO_LEARNING")
                if active==6:
                    check(all(torch.equal(state[k],learned[k]) for k in state if any(k.startswith(prefix+str(i)+".") for prefix in ("projections.","attention.","relations.") for i in range(6,14))),"INACTIVE_BRANCH_DRIFT")
                torch.save(learned,root/"model.pt");np.save(root/"probabilities.npy",prob);np.save(root/"losses.npy",np.asarray(losses))
                cold_model=RelationReadout(dims).cuda().eval();cold_model.load_state_dict(torch.load(root/"model.pt",map_location="cpu",weights_only=True))
                replay=probabilities(cold_model,store,rows,active,scope);np.save(root/"replay-probabilities.npy",replay)
                check(np.array_equal(prob,replay),"CHECKPOINT_REPLAY");checks=controls(cold_model,store,active,scope)
                torch.cuda.synchronize();score=metrics(prob,rows,labels);publish(root/"metrics.json",score)
                rep=dict(seed=seed,arm=name,active_lanes=active,scope=scope,updates=200,parameter_count=parameters,
                    changed_parameters=changed,initial_sha256=file_sha(seedroot/"initial.pt"),model_sha256=file_sha(root/"model.pt"),
                    checkpoint_replay_exact=True,input_controls=checks,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(),wall_seconds=time.perf_counter()-at,metrics=score)
                publish(root/"report.json",rep);results[str(seed)+"/"+name]=dict(report_sha256=file_sha(root/"report.json"))
                print(json.dumps(dict(seed=seed,arm=name,parameters=parameters,evaluation=score["evaluation"],wall_seconds=rep["wall_seconds"])),flush=True)
            except (RuntimeError,ValueError) as exc:
                failure=dict(seed=seed,arm=name,error=str(exc));errors.append(failure);publish(root/"failure.json",failure);print(json.dumps(failure),flush=True)
            finally:
                del model,opt,cold_model;torch.cuda.empty_cache()
    phases["sixteen_fits_full_predictions_and_checkpoint_replays"]=time.perf_counter()-tick;tick=time.perf_counter()
    gpu_hashes=[array_sha(x.cpu().numpy()) for x in store.native]
    check(gpu_hashes==[p["native_sha256"] for p in pins],"NATIVE_MEMORY_DRIFT")
    check(all(file_sha(Path(path))==pin for path,pin in context["sources"].items()),"PARENT_DRIFT")
    check(file_sha(a.population/"report.json")==a.report_sha256,"POPULATION_DRIFT")
    files={p.relative_to(a.output).as_posix():dict(bytes=p.stat().st_size,sha256=file_sha(p)) for p in a.output.rglob('*') if p.is_file()}
    phases["native_parent_drift_and_final_publication"]=time.perf_counter()-tick;wall=time.perf_counter()-start
    report=dict(schema="scifact-readout-results-v1",population_sha256=a.report_sha256,files=files,arms=results,errors=errors,complete=not errors,
        native_readings_unchanged=True,assertion_verdict="REFUSE_UNADMITTED_PANEL",scientific_admission=False,served=False,
        phases_seconds=phases,wall_seconds=wall,phase_coverage=sum(phases.values())/wall,code_sha256=file_sha(Path(__file__)))
    publish(a.output/"report.json",report)
    print(json.dumps(dict(complete=report["complete"],arms=len(results),wall_seconds=wall,phase_coverage=report["phase_coverage"])),flush=True)
    return 0 if not errors else 2


if __name__=="__main__":sys.exit(run_cli(main))
