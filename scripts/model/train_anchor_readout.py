"""Fixed-budget six-vs-ten typed readout; GPU-only fitting and cold checkpoint replay."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit.anchor_population_readback import read,arrays
from model.anchor_readout import (CONFIG,AnchorReadout,check,prepare,bind_native,array_sha,batch,
                                 schedule,target_permutation,set_loss,score,prior)
from model.renderer_conditioning import gpu_lease
from lens.lenscommon import run_cli
from senses.trajectory_population import publish
from text.evaluation_multiscale import file_sha


@torch.inference_mode()
def probabilities(model,memories,masks,questions,active):
    result=[]
    for start in range(0,len(questions),CONFIG["batch"]):
        args=batch(memories,masks,questions,slice(start,start+CONFIG["batch"]))
        result.append(model(*args,active).softmax(1).cpu().numpy())
    return np.vstack(result)


def main():
    p=argparse.ArgumentParser()
    for name in ("population","output"):p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--report-sha256",required=True);p.add_argument("--reviewed-desktop-residents",type=Path)
    a=p.parse_args();check(not a.output.exists(),"OUTPUT_EXISTS");started=time.perf_counter();phases={}
    read(a.population,a.report_sha256);rep=json.loads((a.population/"report.json").read_bytes())
    check(all(s["ready"] for s in rep["support"].values()),"UNDERPOWERED")
    rows,memories,masks,questions,targets,train,input_pins=prepare(a.population)
    # Mutating answers/identifiers while retaining actual chunks cannot alter model inputs.
    altered=copy.deepcopy(rows[:32])
    for r in altered:r.update(entity="not-an-input",values=["wrong"],positive_classes=[5],split="other")
    for k,lane in enumerate(rep["lanes"]):
        x,ptr=arrays(a.population,lane);m,mask=bind_native(x,ptr,altered,lane["max_document_windows"])
        check(np.array_equal(m,memories[k][:32].numpy()) and np.array_equal(mask,masks[k][:32].numpy()),"METADATA_LEAKAGE")
    phases["population_read_native_binding_and_metadata_control"]=time.perf_counter()-started;tick=time.perf_counter()
    residents=gpu_lease(a.reviewed_desktop_residents,"source-anchor-readout-200-step-seven-arms")
    a.output.mkdir(parents=True);(a.output/"gpu-before.txt").write_text(residents,encoding="utf-8")
    publish(a.output/"input-pins.json",input_pins);publish(a.output/"config.json",CONFIG)
    torch.set_num_threads(min(32,os.cpu_count() or 1))
    memories=[m.cuda() for m in memories];masks=[m.cuda() for m in masks];questions=questions.cuda();targets=targets.cuda()
    dims=[l["dimension"] for l in rep["lanes"]];results={};parameter_count=None
    phases["lease_and_native_memory_upload"]=time.perf_counter()-tick;tick=time.perf_counter()
    base=prior(rows,train);np.save(a.output/"prior.npy",base);publish(a.output/"prior-metrics.json",score(base,rows))
    for seed in CONFIG["seeds"]:
        torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        initial=AnchorReadout(dims);state={k:v.detach().cpu().clone() for k,v in initial.state_dict().items()}
        parameter_count=sum(p.numel() for p in initial.parameters());del initial
        seedroot=a.output/("seed-"+str(seed));seedroot.mkdir();torch.save(state,seedroot/"initial.pt")
        steps=schedule(train,seed);np.save(seedroot/"training-rows.npy",steps)
        arms=[("public-six",6,False),("all-ten",10,False)]
        if seed==25:arms.append(("all-ten-permuted",10,True))
        for name,active,permuted in arms:
            arm_start=time.perf_counter();root=seedroot/name;root.mkdir();model=AnchorReadout(dims).cuda();model.load_state_dict(state)
            opt=torch.optim.AdamW(model.parameters(),lr=CONFIG["learning_rate"],weight_decay=CONFIG["weight_decay"])
            mapping=target_permutation(rows,train,seed) if permuted else np.arange(len(rows),dtype=np.int64)
            np.save(root/"target-source-rows.npy",mapping);losses=[];fit_targets=targets[torch.tensor(mapping,device="cuda")]
            for ids in steps:
                index=torch.tensor(ids,device="cuda");opt.zero_grad(set_to_none=True)
                logits=model(*batch(memories,masks,questions,index),active);loss=set_loss(logits,fit_targets[index])
                check(bool(torch.isfinite(loss)),"NONFINITE_LOSS");loss.backward();opt.step();losses.append(float(loss.detach()))
            model.eval();probs=probabilities(model,memories,masks,questions,active)
            check(np.isfinite(probs).all() and np.all(probs>0),"NONFINITE_PROBABILITY")
            learned={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            changed=[k for k in state if not torch.equal(state[k],learned[k])];check(bool(changed),"NO_LEARNING")
            if active==6:
                check(all(torch.equal(state[k],learned[k]) for k in state if
                          any(k.startswith(prefix+str(i)+".") for prefix in ("projections.","window_attention.") for i in range(6,10))),"INACTIVE_BRANCH_DRIFT")
            torch.save(learned,root/"model.pt");np.save(root/"probabilities.npy",probs);np.save(root/"losses.npy",np.asarray(losses))
            # Reopen the final persisted weights and reproduce the actual batched probabilities.
            cold=AnchorReadout(dims).cuda().eval();cold.load_state_dict(torch.load(root/"model.pt",map_location="cpu",weights_only=True))
            replay=probabilities(cold,memories,masks,questions,active);np.save(root/"replay-probabilities.npy",replay)
            check(np.array_equal(replay,probs),"CHECKPOINT_REPLAY")
            # Required sensory-input control; no labels enter either forward call.
            args=batch(memories,masks,questions,slice(0,32))
            with torch.inference_mode():
                original=cold(*args,active);mutated=cold([-x for x in args[0]],args[1],args[2],active)
            check(not torch.equal(original,mutated),"SENSES_IGNORED")
            torch.cuda.synchronize()
            metrics=score(probs,rows);publish(root/"metrics.json",metrics)
            outcome=dict(seed=seed,arm=name,active_lanes=active,permuted_training_targets=permuted,updates=200,
                         parameter_count=parameter_count,changed_parameters=changed,initial_sha256=file_sha(seedroot/"initial.pt"),
                         model_sha256=file_sha(root/"model.pt"),checkpoint_replay_exact=True,metadata_not_input=True,
                         senses_mutation_changes_logits=True,wall_seconds=time.perf_counter()-arm_start,metrics=metrics)
            publish(root/"report.json",outcome);results[str(seed)+"/"+name]=outcome
            print(json.dumps(dict(seed=seed,arm=name,updates=200,wall_seconds=outcome["wall_seconds"],
                                 evaluation_accuracy={q:m["evaluation"]["accuracy"] for q,m in metrics.items()})),flush=True)
            del model,cold,opt,learned
    phases["seven_fixed_fits_predictions_checkpoint_replays"]=time.perf_counter()-tick;tick=time.perf_counter()
    read(a.population,a.report_sha256)
    del memories,masks,questions,targets
    files={f.relative_to(a.output).as_posix():dict(bytes=f.stat().st_size,sha256=file_sha(f)) for f in a.output.rglob('*') if f.is_file()}
    phases["frozen_source_recheck_and_publication"]=time.perf_counter()-tick;wall=time.perf_counter()-started
    publish(a.output/"report.json",dict(schema="anchor-readout-diagnostic-v1",population_sha256=a.report_sha256,
        config=CONFIG,rows=len(rows),lanes=rep["lanes"],files=files,arms=results,parameter_count=parameter_count,
        metadata_not_input=True,all_native_windows=True,checkpoint_selection="last-fixed-update",
        scientific_admission=False,production_assertions=False,shannon_cmi_claim=False,source_truth_only=True,
        runtime=dict(torch=str(torch.__version__),cuda=torch.version.cuda,platform=sys.platform,device=torch.cuda.get_device_name(0)),
        code_sha256={n:file_sha(Path(__file__).parent/n) for n in ("anchor_readout.py","train_anchor_readout.py")},
        phases_seconds=phases,wall_seconds=wall,phase_coverage=sum(phases.values())/wall))
    print(json.dumps(dict(report_sha256=file_sha(a.output/"report.json"),wall_seconds=wall,parameter_count=parameter_count)))


if __name__=="__main__":sys.exit(run_cli(main))
