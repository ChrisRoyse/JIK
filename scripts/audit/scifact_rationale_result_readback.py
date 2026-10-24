"""Audit retained training captures independently before completing publication."""
import json
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit.scifact_population_readback import reference
from audit.scifact_rationale_readback import expected,read as population_read
from audit.scifact_result_readback import read as baseline_read,array_sha,equal
from audit.scifact_rationale_scores import audit_outputs,objective
from text.evaluation_multiscale import file_sha

ERROR="MNLM_UCR_VALUE: Object of type int64 is not JSON serializable"
ARMS=[(s,n) for s in (25,26,27) for n in ("relation-only","rationale","shifted-rationale") if n!="shifted-rationale" or s==25]
CONFIG=dict(width=64,heads=4,batch=32,steps=200,learning_rate=.001,weight_decay=.01,seeds=[25,26,27],
            arms={"relation-only":0.,"rationale":1.,"shifted-rationale":1.},stop=42,precision="fp32")


def check(ok,code):
    if not ok:raise ValueError("MNLM_RATIONALE_RESULT_"+code)


def context(population,population_pin,inventory,rationale,rationale_pin,baseline,baseline_pin):
    ctx=reference(inventory);truth=expected(ctx);pop=population_read(rationale,rationale_pin,population,population_pin,inventory,ctx,truth)
    old=baseline_read(baseline,baseline_pin,population,population_pin,inventory,ctx)
    return dict(native=ctx,truth=truth,population=pop,baseline=old,baseline_root=baseline)


def training_target(ctx,truth,train,shift):
    value={k:v[train].copy() for k,v in truth["targets"].items()}
    if shift:
        for j,i in enumerate(train):
            size=ctx["documents"][ctx["rows"][i]["document_index"]]["sentences"]
            for alt in range(value["present"].shape[1]):
                if value["present"][j,alt]:
                    count=int(value["steps"][j,alt].sum())-1;source=value["choices"][j,alt,:count].tolist()
                    value["choices"][j,alt,:count]=sorted((v+1)%size for v in source)
    return value


def inspect_training(root,pin,context):
    check(file_sha(root/"report.json")==pin,"CAPTURE_PIN");rep=json.loads((root/"report.json").read_bytes());ctx=context["native"];truth=context["truth"]
    check(rep["schema"]=="scifact-rationale-results-v1" and rep["complete"] is False and rep["arms"]=={},"CAPTURE_STATUS")
    check(rep["errors"]==[dict(seed=s,arm=n,error=ERROR) for s,n in ARMS],"FAILURE_SCOPE")
    check(rep["parent_population_sha256"]==context["population"]["parent_population_sha256"] and rep["rationale_sha256"]==context["population"]["report_sha256"]
        and rep["baseline_sha256"]==context["baseline"]["result_sha256"],"PARENTS")
    check(rep["assertion_verdict"]=="REFUSE_UNADMITTED_PANEL" and all(rep[k] is False for k in ("scientific_admission","served","sealed_evaluation")),"CLAIM")
    names={"gpu-before.txt","before.json","after.json","config.json","population-readback.json","baseline-readback.json","training-cases.npy","labels.npy","runtime.json"}
    for s in (25,26,27):names|={f"seed-{s}/initial.pt",f"seed-{s}/training-rows.npy"}
    for s,n in ARMS:names|={f"seed-{s}/{n}/"+f for f in ("failure.json","losses.npy","model.pt","objective-probe.json","objective-probe.npz","outputs.npz","replay.npz","training-targets.npz")}
    check(set(rep["files"])==names and {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}==names|{"report.json"},"FILES")
    for n,v in rep["files"].items():check(file_sha(root/n)==v["sha256"] and (root/n).stat().st_size==v["bytes"],"FILE_HASH")
    load=lambda n:json.loads((root/n).read_bytes())
    check(load("before.json")==load("after.json"),"SOURCE_DRIFT")
    check(load("config.json")==CONFIG and load("population-readback.json")==context["population"] and equal(load("baseline-readback.json"),context["baseline"]),"CONFIG_POPULATION")
    runtime=load("runtime.json");check(runtime["platform"]=="linux" and bool(runtime["cuda"]),"RUNTIME")
    code=Path(__file__).parents[1]/"model"
    for n in ("scifact_rationale.py","scifact_readout.py","train_scifact_rationale.py"):check(runtime["code"][n]==file_sha(code/n),"MODEL_CODE")
    check(runtime["code"]["scifact_rationale_metrics.py"]=="9d9b8bd2dfb0e1c81dd539fa46292a1ae2534709ecb1318a804166809eb65ffb","CAPTURE_METRICS_CODE")
    train=np.array([i for i,r in enumerate(ctx["rows"]) if r["partition"]=="train"],np.int64)
    labels=np.array([{"NOT_ENOUGH_EVIDENCE":0,"CONTRADICT":1,"SUPPORT":2}[t["label"]] for t in ctx["targets"]],np.int64)
    check(np.array_equal(np.load(root/"training-cases.npy"),train) and np.array_equal(np.load(root/"labels.npy"),labels),"LABELS_OR_TRAIN")
    outputs={};agreements={};findings={}
    for seed in (25,26,27):
        initial=torch.load(root/f"seed-{seed}/initial.pt",map_location="cpu",weights_only=True)
        old_initial=torch.load(context["baseline_root"]/f"seed-{seed}/initial.pt",map_location="cpu",weights_only=True)
        check(all(torch.equal(initial[k],v) for k,v in old_initial.items()),"BASELINE_INITIAL")
        rng=np.random.default_rng(seed);plan=np.array([rng.choice(train,32,replace=False) for _ in range(200)])
        check(np.array_equal(np.load(root/f"seed-{seed}/training-rows.npy"),plan),"SCHEDULE")
        for _,name in [v for v in ARMS if v[0]==seed]:
            key=f"{seed}/{name}";folder=root/f"seed-{seed}"/name;weight=0 if name=="relation-only" else 1
            check(json.loads((folder/"failure.json").read_bytes())==dict(seed=seed,arm=name,error=ERROR),"FAILURE")
            target=training_target(ctx,truth,train,name=="shifted-rationale")
            with np.load(folder/"training-targets.npz",allow_pickle=False) as z:
                check(set(z.files)==set(target),"TARGET_KEYS")
                for k,v in target.items():check(z[k].dtype==v.dtype and np.array_equal(z[k],v),"TARGETS")
            learned=torch.load(folder/"model.pt",map_location="cpu",weights_only=True)
            check(set(initial)==set(learned) and all(v.dtype==torch.float32 and v.shape==initial[k].shape and torch.isfinite(v).all() for k,v in learned.items()),"CHECKPOINT")
            check(any(not torch.equal(v,initial[k]) for k,v in learned.items()),"NO_LEARNING")
            with np.load(folder/"outputs.npz",allow_pickle=False) as z:output={k:z[k].copy() for k in z.files}
            with np.load(folder/"replay.npz",allow_pickle=False) as z:
                check(set(z.files)==set(output) and all(np.array_equal(z[k],v) for k,v in output.items()),"SAME_PROCESS_REPLAY")
            baseline_exact=None
            if not weight:
                old=torch.load(context["baseline_root"]/f"seed-{seed}/joint-all/model.pt",map_location="cpu",weights_only=True)
                baseline_exact=all(torch.equal(v,learned[k]) for k,v in old.items()) and np.array_equal(output["probabilities"],np.load(context["baseline_root"]/f"seed-{seed}/joint-all/probabilities.npy"))
                check(baseline_exact and all(torch.equal(v,initial[k]) for k,v in learned.items() if k.startswith("pointer.")),"BASELINE_REPRODUCTION")
            losses=np.load(folder/"losses.npy",allow_pickle=False);check(losses.shape==(200,3) and np.isfinite(losses).all() and (losses>=0).all()
                and np.allclose(losses[:,0]+weight*losses[:,1],losses[:,2],rtol=0,atol=1e-5),"LOSSES")
            with np.load(folder/"objective-probe.npz",allow_pickle=False) as z:probe={k:z[k].copy() for k in z.files}
            check(set(probe)=={"case_rows","classification_logits","labels","teacher_logits","choices","steps","present"},"PROBE_KEYS")
            check(np.array_equal(probe["case_rows"],plan[0]) and np.array_equal(probe["labels"],labels[plan[0]]),"PROBE_SCOPE")
            positions=np.searchsorted(train,plan[0])
            for k,v in target.items():check(np.array_equal(probe[k],v[positions]),"PROBE_TARGETS")
            actual=json.loads((folder/"objective-probe.json").read_bytes());calculated=objective(probe,weight,ctx,truth["stop"])
            check(all(abs(actual[k]-v)<=1e-5 for k,v in calculated.items()),"OBJECTIVE")
            grad=actual["gradients"];check(set(grad)==set(learned),"GRADIENTS")
            check(any(v is not None and v>0 for k,v in grad.items() if k.startswith("pointer.")) if weight else all(v is None for k,v in grad.items() if k.startswith("pointer.")),"POINTER_GRADIENT")
            scores,agree=audit_outputs(output,ctx,truth);outputs[key]=output;agreements[key]=agree
            findings[key]=dict(metrics=scores,model_sha256=file_sha(folder/"model.pt"),initial_sha256=file_sha(root/f"seed-{seed}/initial.pt"),
                parameters=sum(v.numel() for v in learned.values()),baseline_exact=baseline_exact,objective_readback=calculated,
                output_sha256={k:array_sha(v) for k,v in output.items()})
    check(rep["phase_coverage"]>=.9 and abs(sum(rep["phases_seconds"].values())/rep["wall_seconds"]-rep["phase_coverage"])<1e-12,"PHASES")
    cold=dict(capture_sha256=pin,rationale_sha256=rep["rationale_sha256"],baseline_sha256=rep["baseline_sha256"],arms=findings,
        training_capture_complete=False,serialization_failure_retained=True,complete_checkpoint_payloads=True,scientific_admission=False,assertion_verdict="REFUSE_UNADMITTED_PANEL")
    return dict(cold=cold,outputs=outputs,agreements=agreements,runtime=runtime)


def read_completed(root,pin,inspected,analysis,parent_pin):
    check(file_sha(root/"report.json")==pin,"COMPLETED_PIN");rep=json.loads((root/"report.json").read_bytes());cold=inspected["cold"]
    check(rep["schema"]=="scifact-rationale-completed-v1" and rep["capture_sha256"]==cold["capture_sha256"],"CAPTURE_BINDING")
    check(rep["parent_population_sha256"]==parent_pin and rep["rationale_sha256"]==cold["rationale_sha256"] and rep["baseline_sha256"]==cold["baseline_sha256"],"PARENTS")
    check(rep["complete"] is True and rep["all_output_arrays_replayed"] is True and rep["cases_replayed"]==8806
        and rep["failed_capture_retained"] is True and rep["training_repeated"] is False,"REPLAY_CLAIM")
    check(rep["assertion_verdict"]=="REFUSE_UNADMITTED_PANEL" and all(rep[k] is False for k in ("scientific_admission","served","sealed_evaluation")),"CLAIM")
    check(rep["per_arm_training_time_and_vram"]=="unavailable:failed-before-publication","TRAINING_MEASUREMENT_GAP")
    names={"before.json","after.json","capture-readback.json","gpu-before.txt","runtime.json","bootstrap-plan.npz","bootstrap-samples.npz",
        "paired-comparisons.json","position-controls.json","multi-sentence-only.json","same-document-contrasts.json"}
    for key in cold["arms"]:names|={key.replace('/','-')+'/'+n for n in ("replay.npz","metrics.json","agreements.json","report.json")}
    check(set(rep["files"])==names and {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}==names|{"report.json"},"FILES")
    for name,v in rep["files"].items():check(file_sha(root/name)==v["sha256"] and (root/name).stat().st_size==v["bytes"],"FILE_HASH")
    load=lambda n:json.loads((root/n).read_bytes())
    check(load("before.json")==load("after.json"),"SOURCE_DRIFT");check(equal(load("capture-readback.json"),cold),"CAPTURE_READBACK")
    runtime=load("runtime.json");code=Path(__file__).parents[1]/"model"
    check(all(runtime[k]==inspected["runtime"][k] for k in ("torch","cuda","device","platform")),"RUNTIME")
    check(runtime["checkpoint_parameter_flags_preserved"] is True and runtime["inference_mode"] is True and runtime["platform"]=="linux"
        and runtime["model_code_sha256"]==file_sha(code/"scifact_rationale.py") and runtime["metrics_code_sha256"]==file_sha(code/"scifact_rationale_metrics.py"),"RUNTIME")
    check(set(rep["arms"])==set(cold["arms"]),"ARMS")
    for key,expected_arm in cold["arms"].items():
        folder=key.replace('/','-');ar=load(folder+"/report.json")
        check(file_sha(root/folder/"report.json")==rep["arms"][key] and ar["model_sha256"]==expected_arm["model_sha256"],"MODEL_BINDING")
        check(ar["cases"]==1258 and ar["all_output_arrays_exact"] is True and ar["output_sha256"]==expected_arm["output_sha256"],"OUTPUT_BINDING")
        check(ar["input_controls"]==dict(query_changes_relation=True,evidence_changes_relation=True,query_changes_pointer_memory=True,
            evidence_changes_pointer_memory=True,gold_input_rejected=True,missing_sentence_rejected=True),"INPUT_CONTROLS")
        with np.load(root/folder/"replay.npz",allow_pickle=False) as z:
            check(set(z.files)==set(inspected["outputs"][key]) and all(np.array_equal(z[k],v) for k,v in inspected["outputs"][key].items()),"FRESH_REPLAY")
        check(equal(load(folder+"/metrics.json"),expected_arm["metrics"]) and equal(load(folder+"/agreements.json"),inspected["agreements"][key]),"METRICS")
    for file,key in (("bootstrap-plan.npz","plan"),("bootstrap-samples.npz","samples")):
        with np.load(root/file,allow_pickle=False) as z:
            check(set(z.files)==set(analysis[key]),"BOOTSTRAP_KEYS")
            for name,value in analysis[key].items():check(z[name].shape==value.shape and np.isfinite(z[name]).all() and np.allclose(z[name],value,rtol=0,atol=0 if key=="plan" else 1e-12),"BOOTSTRAP_"+key.upper())
    for file,key in (("paired-comparisons.json","comparisons"),("position-controls.json","position"),("multi-sentence-only.json","multi"),("same-document-contrasts.json","contrasts")):
        check(equal(load(file),analysis[key]),key.upper())
    check(rep["phase_coverage"]>=.9 and abs(sum(rep["phases_seconds"].values())/rep["wall_seconds"]-rep["phase_coverage"])<1e-12,"PHASES")
    return dict(report_sha256=pin,capture_sha256=cold["capture_sha256"],arms=cold["arms"],comparisons=analysis["comparisons"],position=analysis["position"],
        multi=analysis["multi"],contrasts=analysis["contrasts"],full_fresh_replay_cases=8806,scientific_admission=False,assertion_verdict="REFUSE_UNADMITTED_PANEL")
