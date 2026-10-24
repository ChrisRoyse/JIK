"""Seven fixed fits of complete-rationale auxiliary supervision on frozen readings."""
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
from audit.scifact_population_readback import reference
from audit.scifact_rationale_readback import read,expected
from audit.scifact_result_readback import read as old_read
from model.scifact_readout import schedule,array_sha,check
from model.scifact_rationale import CONFIG,RationaleStore,RationaleReadout,evaluate
from model.scifact_rationale_metrics import evidence_metrics
from model.renderer_conditioning import gpu_lease
from senses.trajectory_population import publish
from text.evaluation_multiscale import file_sha
from lens.lenscommon import run_cli


def training_targets(truth,ctx,train,shift):
    target={k:v[train].copy() for k,v in truth["targets"].items()}
    if shift:
        for j,i in enumerate(train):
            count=int(truth["valid"][ctx["rows"][i]["document_index"]].sum())
            for k in np.flatnonzero(target["present"][j]):
                length=int(target["steps"][j,k].sum())-1
                target["choices"][j,k,:length]=np.sort((target["choices"][j,k,:length]+1)%count)
    return target


@torch.inference_mode()
def controls(model,store):
    memories,valid=store.batch(range(32));before,hidden=model(memories,valid)
    q,hq=model([{**m,"query":-m["query"]} for m in memories],valid)
    e,he=model([{**m,"evidence":-m["evidence"]} for m in memories],valid)
    result=dict(query_changes_relation=not torch.equal(before,q),evidence_changes_relation=not torch.equal(before,e),
        query_changes_pointer_memory=not torch.equal(hidden,hq),evidence_changes_pointer_memory=not torch.equal(hidden,he))
    try:model([{**m,"rationale":0} for m in memories],valid);result["gold_input_rejected"]=False
    except ValueError as exc:result["gold_input_rejected"]="INPUT_FIELDS" in str(exc)
    modified=[{**m,"membership":m["membership"].clone()} for m in memories];modified[0]["membership"][0,0]=False
    try:model(modified,valid);result["missing_sentence_rejected"]=False
    except ValueError as exc:result["missing_sentence_rejected"]="MEMBERSHIP_COVERAGE" in str(exc)
    check(all(result.values()),"RATIONALE_CONTROLS");return result


def main():
    p=argparse.ArgumentParser()
    for n in ("population","inventory","rationale","baseline","output"):p.add_argument("--"+n,type=Path,required=True)
    for n in ("population-sha256","rationale-sha256","baseline-sha256"):p.add_argument("--"+n,required=True)
    a=p.parse_args();start=time.perf_counter();phases={};check(not a.output.exists(),"OUTPUT_EXISTS")
    torch.set_num_threads(min(32,os.cpu_count() or 1));ctx=reference(a.inventory);truth=expected(ctx)
    cold=read(a.rationale,a.rationale_sha256,a.population,a.population_sha256,a.inventory,ctx,truth)
    old=old_read(a.baseline,a.baseline_sha256,a.population,a.population_sha256,a.inventory,ctx)
    rows=ctx["rows"];train=np.array([i for i,r in enumerate(rows) if r["partition"]=="train"],np.int64)
    labels=np.array([("NOT_ENOUGH_EVIDENCE","CONTRADICT","SUPPORT").index(t["label"]) for t in ctx["targets"]],np.int64)
    before=dict(ctx["sources"])
    for root in (a.population,a.rationale,a.baseline):before.update({str(p):file_sha(p) for p in root.rglob('*') if p.is_file()})
    phases["complete_source_native_and_baseline_readback"]=time.perf_counter()-start;tick=time.perf_counter()
    residents=gpu_lease(None,"scifact-complete-rationale-seven-fixed-fits");a.output.mkdir(parents=True)
    (a.output/"gpu-before.txt").write_text(residents,encoding="utf-8");publish(a.output/"before.json",before)
    publish(a.output/"config.json",CONFIG);publish(a.output/"population-readback.json",cold);publish(a.output/"baseline-readback.json",old)
    np.save(a.output/"training-cases.npy",train);np.save(a.output/"labels.npy",labels)
    store=RationaleStore(ctx,truth,"cuda");dims=[l["dimension"] for l in ctx["lanes"]];class_targets=torch.from_numpy(labels[train]).cuda()
    reverse=np.full(len(rows),-1,np.int64);reverse[train]=np.arange(len(train));mapping=torch.from_numpy(reverse).cuda()
    runtime=dict(torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(0),platform=sys.platform,
        code={name:file_sha(Path(__file__).with_name(name)) for name in ("scifact_rationale.py","scifact_rationale_metrics.py","scifact_readout.py","train_scifact_rationale.py")})
    publish(a.output/"runtime.json",runtime);phases["lease_native_upload_and_fixed_supervision"]=time.perf_counter()-tick;tick=time.perf_counter();arms={};errors=[]
    for seed in CONFIG["seeds"]:
        torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);initial=RationaleReadout(dims);state={k:v.detach().clone() for k,v in initial.state_dict().items()}
        original=torch.load(a.baseline/f"seed-{seed}/initial.pt",map_location="cpu",weights_only=True)
        check(all(torch.equal(v,state[k]) for k,v in original.items()),"BASELINE_INITIALIZATION")
        parameters=sum(p.numel() for p in initial.parameters());del initial
        seedroot=a.output/f"seed-{seed}";seedroot.mkdir();torch.save(state,seedroot/"initial.pt");steps=schedule(rows,seed);np.save(seedroot/"training-rows.npy",steps)
        for name,weight in CONFIG["arms"].items():
            if name=="shifted-rationale" and seed!=25:continue
            root=seedroot/name;root.mkdir();at=time.perf_counter();model=opt=reloaded=None
            try:
                target=training_targets(truth,ctx,train,name=="shifted-rationale");np.savez(root/"training-targets.npz",**target)
                target={k:torch.from_numpy(v).cuda() for k,v in target.items()};model=RationaleReadout(dims).cuda();model.load_state_dict(state)
                opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01);torch.cuda.reset_peak_memory_stats();losses=[]
                for step,ids in enumerate(steps):
                    ix=torch.from_numpy(ids).cuda();local=mapping[ix];check(bool((local>=0).all()),"NONTRAIN_TARGET")
                    opt.zero_grad(set_to_none=True);memories,valid=store.batch(ix);logits,hidden=model(memories,valid)
                    relation=torch.nn.functional.cross_entropy(logits,class_targets[local]);chosen={k:v[local] for k,v in target.items()}
                    aux=model.pointer.loss(hidden,valid,chosen) if weight else torch.zeros((),device="cuda")
                    loss=relation+weight*aux;check(bool(torch.isfinite(loss)),"NONFINITE_LOSS");loss.backward()
                    if step==0:
                        with torch.no_grad():path_log,raw=model.pointer.paths(hidden,valid,chosen["choices"],chosen["steps"])
                        np.savez_compressed(root/"objective-probe.npz",case_rows=ids,classification_logits=logits.detach().cpu().numpy(),
                            labels=class_targets[local].cpu().numpy(),teacher_logits=raw.cpu().numpy(),**{k:v.cpu().numpy() for k,v in chosen.items()})
                        gradients={k:float(p.grad.norm()) if p.grad is not None else None for k,p in model.named_parameters()}
                        check(any(v is not None and v>0 for k,v in gradients.items() if k.startswith("pointer.")) if weight else all(v is None for k,v in gradients.items() if k.startswith("pointer.")),"AUX_GRADIENT")
                        publish(root/"objective-probe.json",dict(relation_loss=float(relation.detach()),auxiliary_loss=float(aux.detach()),total_loss=float(loss.detach()),gradients=gradients))
                    opt.step();losses.append([float(relation.detach()),float(aux.detach()),float(loss.detach())])
                model.eval();outputs=evaluate(model,store,rows,truth);np.savez_compressed(root/"outputs.npz",**outputs);np.save(root/"losses.npy",np.array(losses))
                learned={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};torch.save(learned,root/"model.pt")
                baseline_exact=None
                if weight==0:
                    prior=torch.load(a.baseline/f"seed-{seed}/joint-all/model.pt",map_location="cpu",weights_only=True)
                    baseline_exact=all(torch.equal(v,learned[k]) for k,v in prior.items()) and np.array_equal(outputs["probabilities"],np.load(a.baseline/f"seed-{seed}/joint-all/probabilities.npy"))
                    check(baseline_exact,"BASELINE_REPRODUCTION")
                reloaded=RationaleReadout(dims).cuda().eval();reloaded.load_state_dict(torch.load(root/"model.pt",map_location="cpu",weights_only=True))
                replay=evaluate(reloaded,store,rows,truth);np.savez_compressed(root/"replay.npz",**replay);check(all(np.array_equal(outputs[k],v) for k,v in replay.items()),"CHECKPOINT_REPLAY")
                scope=controls(reloaded,store);score,agreements=evidence_metrics(outputs,ctx,truth);publish(root/"metrics.json",score);publish(root/"agreements.json",agreements)
                torch.cuda.synchronize();rep=dict(seed=seed,arm=name,weight=weight,parameters=parameters,updates=200,baseline_exact=baseline_exact,
                    checkpoint_replay_exact=True,input_controls=scope,model_sha256=file_sha(root/"model.pt"),initial_sha256=file_sha(seedroot/"initial.pt"),
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),wall_seconds=time.perf_counter()-at,metrics=score)
                publish(root/"report.json",rep);arms[f"{seed}/{name}"]=file_sha(root/"report.json")
                print(json.dumps(dict(seed=seed,arm=name,seconds=rep["wall_seconds"],relation=score["relation"]["evaluation"]["accuracy"],evidence=score["evidence"]["evaluation"])),flush=True)
            except (ValueError,RuntimeError) as exc:
                error=dict(seed=seed,arm=name,error=str(exc));errors.append(error);publish(root/"failure.json",error);print(json.dumps(error),flush=True)
            finally:del model,opt,reloaded;torch.cuda.empty_cache()
    phases["seven_fits_actual_objectives_predictions_and_replays"]=time.perf_counter()-tick;tick=time.perf_counter()
    check(all(array_sha(x.cpu().numpy())==array_sha(y) for x,y in zip(store.native,ctx["native"])),"NATIVE_DRIFT")
    after={p:file_sha(Path(p)) for p in before};check(before==after,"PARENT_DRIFT");publish(a.output/"after.json",after)
    files={p.relative_to(a.output).as_posix():dict(bytes=p.stat().st_size,sha256=file_sha(p)) for p in a.output.rglob('*') if p.is_file()}
    phases["parent_native_and_output_pins"]=time.perf_counter()-tick;wall=time.perf_counter()-start
    publish(a.output/"report.json",dict(schema="scifact-rationale-results-v1",files=files,arms=arms,errors=errors,complete=not errors,
        parent_population_sha256=a.population_sha256,rationale_sha256=a.rationale_sha256,baseline_sha256=a.baseline_sha256,
        assertion_verdict="REFUSE_UNADMITTED_PANEL",scientific_admission=False,served=False,sealed_evaluation=False,
        phases_seconds=phases,wall_seconds=wall,phase_coverage=sum(phases.values())/wall))
    print(json.dumps(dict(complete=not errors,arms=len(arms),wall_seconds=wall)),flush=True);return 0 if not errors else 2


if __name__=="__main__":sys.exit(run_cli(main))
