"""Thirteen fixed matched fits with no document path around selected observations."""
import json
import os
from pathlib import Path
import sys
import time
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from model.scifact_selected import CONFIG,ARMS,check,SelectedStore,SelectedReadout,arm_plan,evaluate,controls
from model.scifact_selected_source import arguments,load,PINS
from model.scifact_selected_metrics import measure,interventions,comparisons
from model.scifact_readout import schedule,array_sha
from model.renderer_conditioning import gpu_lease
from audit.scifact_selected_readback import read_plans
from senses.trajectory_population import publish
from text.evaluation_multiscale import file_sha
from lens.lenscommon import run_cli


def save_plans(root,source):
    root.mkdir();np.savez(root/"selections.npz",**source["selection"]);np.save(root/"donors.npy",source["donors"])
    for name,lanes in source["plans"].items():
        folder=root/name;folder.mkdir()
        for k,m in enumerate(lanes):np.savez_compressed(folder/f"{k}.npz",**m)


def all_outputs(model,store,rows,plan,name,donor):
    scope="evidence" if name=="evidence-only" else "joint"
    out=dict(probabilities=evaluate(model,store,rows,plan,scope))
    if name=="selected":
        out.update(removed=evaluate(model,store,rows,"none"),query_swapped=evaluate(model,store,rows,plan,donor=donor,mode="query"),
                   evidence_swapped=evaluate(model,store,rows,plan,donor=donor,mode="evidence"))
    return out


def main():
    a=arguments();check(not a.output.exists(),"OUTPUT_EXISTS");start=time.perf_counter();torch.set_num_threads(32);phases={}
    source=load(a);ctx=source["ctx"];truth=source["truth"];rows=ctx["rows"];a.output.mkdir(parents=True)
    publish(a.output/"before.json",source["before"]);publish(a.output/"config.json",CONFIG);publish(a.output/"parents.json",PINS)
    save_plans(a.output/"inputs",source);publish(a.output/"plan-readback.json",read_plans(a.output/"inputs",source))
    phases["complete_source_selector_and_gather_readback"]=time.perf_counter()-start;tick=time.perf_counter()
    residents=gpu_lease(None,"scifact-selected-source-thirteen-matched-fits");(a.output/"gpu-before.txt").write_text(residents,encoding="utf-8")
    store=SelectedStore(ctx,source["plans"],"cuda");dims=[l["dimension"] for l in ctx["lanes"]]
    labels=np.array([{"NOT_ENOUGH_EVIDENCE":0,"CONTRADICT":1,"SUPPORT":2}[t["label"]] for t in ctx["targets"]],np.int64)
    train=np.array([i for i,r in enumerate(rows) if r["partition"]=="train"],np.int64);np.save(a.output/"labels.npy",labels)
    code={str(p.relative_to(Path(__file__).parents[1])):file_sha(p) for p in [Path(__file__),Path(__file__).with_name("scifact_selected.py"),
        Path(__file__).with_name("scifact_selected_source.py"),Path(__file__).with_name("scifact_selected_metrics.py"),Path(__file__).with_name("scifact_readout.py")]}
    publish(a.output/"runtime.json",dict(torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(0),platform=sys.platform,code=code))
    phases["lease_native_upload_and_runtime"]=time.perf_counter()-tick;tick=time.perf_counter();results={};outputs={};errors=[]
    for seed in (25,26,27):
        torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);initial=SelectedReadout(dims);state={k:v.detach().clone() for k,v in initial.state_dict().items()}
        original=torch.load(a.baseline/f"seed-{seed}/initial.pt",map_location="cpu",weights_only=True)
        check(all(torch.equal(v,state[k]) for k,v in original.items()),"BASELINE_INITIAL")
        params=sum(p.numel() for p in initial.parameters());del initial;seedroot=a.output/f"seed-{seed}";seedroot.mkdir()
        torch.save(state,seedroot/"initial.pt");steps=schedule(rows,seed);np.save(seedroot/"training-rows.npy",steps)
        for s,name in ARMS:
            if s!=seed:continue
            root=seedroot/name;root.mkdir();at=time.perf_counter();plan=arm_plan(seed,name);scope="evidence" if name=="evidence-only" else "joint"
            model=opt=None
            try:
                targets=labels.copy()
                if name=="shuffled":targets[train]=np.random.default_rng(seed).permutation(labels[train])
                np.save(root/"training-labels.npy",targets[train]);target=torch.from_numpy(targets).cuda()
                model=SelectedReadout(dims).cuda();model.load_state_dict(state);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
                torch.cuda.reset_peak_memory_stats();losses=[];fit=time.perf_counter()
                for step,ids in enumerate(steps):
                    opt.zero_grad(set_to_none=True);logits=model(store.batch(ids,plan),scope)
                    loss=torch.nn.functional.cross_entropy(logits,target[ids]);check(bool(torch.isfinite(loss)),"NONFINITE_LOSS");loss.backward()
                    if step==0:
                        np.savez(root/"objective.npz",rows=ids,logits=logits.detach().cpu().numpy(),labels=targets[ids])
                        publish(root/"objective.json",dict(loss=float(loss.detach()),gradients={k:float(p.grad.norm()) if p.grad is not None else None for k,p in model.named_parameters()}))
                    opt.step();losses.append(float(loss.detach()))
                torch.cuda.synchronize();publish(root/"training-time.json",dict(seconds=time.perf_counter()-fit,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(),updates=200,case_updates=6400))
                np.save(root/"losses.npy",np.array(losses));torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},root/"model.pt")
                model.eval();o=all_outputs(model,store,rows,plan,name,source["donors"]);np.savez(root/"outputs.npz",**o)
                c=controls(model,store,ctx,plan) if name=="selected" else None;publish(root/"controls.json",c)
                score=measure(o["probabilities"],source["selection"][plan],ctx,truth);publish(root/"metrics.json",score)
                publish(root/"interventions.json",interventions(o,rows));outputs[f"{seed}/{name}"]=o
                rep=dict(seed=seed,arm=name,plan=plan,scope=scope,parameters=params,model_sha256=file_sha(root/"model.pt"),
                    initial_sha256=file_sha(seedroot/"initial.pt"),output_sha256={k:array_sha(v) for k,v in o.items()},wall_seconds=time.perf_counter()-at)
                publish(root/"report.json",rep);results[f"{seed}/{name}"]=file_sha(root/"report.json")
                print(json.dumps(dict(seed=seed,arm=name,accuracy=score["relation"]["evaluation"]["accuracy"],seconds=rep["wall_seconds"])),flush=True)
            except (ValueError,RuntimeError) as exc:
                error=dict(seed=seed,arm=name,error=str(exc));publish(root/"failure.json",error);errors.append(error);print(json.dumps(error),flush=True)
            finally:del model,opt;torch.cuda.empty_cache()
    phases["thirteen_fixed_fits_and_interventions"]=time.perf_counter()-tick;tick=time.perf_counter()
    if not errors:
        plan,samples,comp,contrast=comparisons(ctx,outputs);np.savez(a.output/"bootstrap-plan.npz",**plan);np.savez(a.output/"bootstrap-samples.npz",**samples)
        publish(a.output/"comparisons.json",comp);publish(a.output/"contrasts.json",contrast)
    phases["component_bootstrap_and_same_document_contrasts"]=time.perf_counter()-tick;tick=time.perf_counter()
    check(all(array_sha(x.cpu().numpy())==array_sha(y) for x,y in zip(store.native,ctx["native"])),"NATIVE_DRIFT")
    after={p:file_sha(Path(p)) for p in source["before"]};check(after==source["before"],"SOURCE_DRIFT");publish(a.output/"after.json",after)
    files={p.relative_to(a.output).as_posix():dict(sha256=file_sha(p),bytes=p.stat().st_size) for p in a.output.rglob('*') if p.is_file()}
    phases["unchanged_parents_and_result_pins"]=time.perf_counter()-tick;wall=time.perf_counter()-start
    publish(a.output/"report.json",dict(schema="scifact-selected-results-v1",parents=PINS,files=files,arms=results,errors=errors,complete=not errors,
        scientific_admission=False,served=False,sealed_evaluation=False,assertion_verdict="REFUSE_UNADMITTED_PANEL",
        phase_coverage=sum(phases.values())/wall,phases_seconds=phases,wall_seconds=wall))
    print(json.dumps(dict(complete=not errors,report_sha256=file_sha(a.output/"report.json"),wall_seconds=wall)),flush=True)
    return 2 if errors else 0


if __name__=="__main__":sys.exit(run_cli(main))
