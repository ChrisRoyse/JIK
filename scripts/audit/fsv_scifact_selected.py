"""Three persisted-state boundaries, semantic mutations and restored-shadow readback."""
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from model.scifact_selected_source import arguments,load
from audit.scifact_selected_readback import read_plans,read_result,read_replay,check
from audit.scifact_result_readback import equal
from senses.trajectory_population import publish
from text.evaluation_multiscale import file_sha
from lens.lenscommon import run_cli


def main():
    a=arguments(("result","replay"));check(not a.output.exists(),"FSV_OUTPUT_EXISTS");start=time.perf_counter();torch.set_num_threads(1);phases={}
    source=load(a);rpin=file_sha(a.result/"report.json");ppin=file_sha(a.replay/"report.json");result=read_result(a.result,rpin,source)
    replay=read_replay(a.replay,ppin,result);a.output.mkdir(parents=True);publish(a.output/"cold-readback.json",result["cold"]);publish(a.output/"replay-readback.json",replay)
    before={**source["before"],**{str(p):file_sha(p) for root in (a.result,a.replay) for p in root.rglob('*') if p.is_file()}}
    publish(a.output/"before.json",before);phases["complete_sources_and_three_boundary_readbacks"]=time.perf_counter()-start;tick=time.perf_counter()
    shadow=a.output/"result-shadow";pshadow=a.output/"replay-shadow";shutil.copytree(a.result,shadow);shutil.copytree(a.replay,pshadow)
    checks=[dict(boundary=1,name="complete-source-selected-window-readback",passed=True),dict(boundary=2,name="thirteen-learned-checkpoints-and-objectives",passed=True),
            dict(boundary=3,name="fresh-replay-and-independent-anchor-metrics",passed=True)]

    def replace_shadow(path,value):
        check(any(path.resolve().is_relative_to(r.resolve()) for r in (shadow,pshadow)),"MUTATION_OUTSIDE_SHADOW")
        path.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n',encoding="utf-8")

    def refresh(root):
        path=root/"report.json";rep=json.loads(path.read_bytes())
        for n in rep["files"]:rep["files"][n]=dict(sha256=file_sha(root/n),bytes=(root/n).stat().st_size)
        for key in rep["arms"]:
            s,n=key.split('/');folder=("seed-"+s+'/'+n) if root==shadow else s+'-'+n;rep["arms"][key]=file_sha(root/folder/"report.json")
        replace_shadow(path,rep)

    def negative(boundary,name,root,relative,change,read,expected):
        path=root/relative;raw=path.read_bytes();report=(root/"report.json").read_bytes();prior=file_sha(path);error=None
        try:
            change(path);refresh(root)
            try:read()
            except (ValueError,KeyError) as exc:error=str(exc)
            check(error is not None and expected in error,"NEGATIVE_"+name)
            changed=file_sha(path)
        finally:path.write_bytes(raw);(root/"report.json").write_bytes(report)
        check(file_sha(path)==prior,"SHADOW_RESTORE");checks.append(dict(boundary=boundary,name=name,passed=True,before_sha256=prior,action_sha256=changed,after_sha256=file_sha(path),refusal=error))

    def npz_change(key,fn):
        def change(path):
            with np.load(path,allow_pickle=False) as z:values={k:z[k].copy() for k in z.files}
            fn(values[key]);np.savez_compressed(path,**values)
        return change

    def json_change(fn):
        def change(path):v=json.loads(path.read_bytes());fn(v);replace_shadow(path,v)
        return change

    plans=lambda:read_plans(shadow/"inputs",source)
    rr=lambda:read_result(shadow,file_sha(shadow/"report.json"),source)
    rp=lambda:read_replay(pshadow,file_sha(pshadow/"report.json"),result)
    negative(1,"selected-sentence-changed",shadow,"inputs/selections.npz",npz_change("25",lambda v:v.__setitem__((0,0),not v[0,0])),plans,"SELECTION_25")
    negative(1,"parent-window-injected",shadow,"inputs/all/0.npz",npz_change("indices",lambda v:v.__setitem__((0,0),source["ctx"]["maps"][0]["indices"][source["ctx"]["rows"][0]["document_index"],0])),plans,"PLAN_INDICES")
    negative(1,"native-window-dropped",shadow,"inputs/all/0.npz",npz_change("mask",lambda v:v.__setitem__((0,0),False)),plans,"PLAN_MASK")
    negative(1,"source-geometry-changed",shadow,"inputs/all/0.npz",npz_change("geometry",lambda v:v.__setitem__((0,0,0),v[0,0,0]+.1)),plans,"PLAN_GEOMETRY")
    def wrong_donor(path):v=np.load(path);v[0]=0;np.save(path,v)
    negative(1,"same-component-donor",shadow,"inputs/donors.npy",wrong_donor,plans,"DONORS")
    phases["source_address_boundary_negatives_and_restores"]=time.perf_counter()-tick;tick=time.perf_counter()
    def wrong_steps(path):v=np.load(path);v[0,0]=next(i for i,r in enumerate(source["ctx"]["rows"]) if r["partition"]=="evaluation");np.save(path,v)
    negative(2,"evaluation-row-in-training",shadow,"seed-25/training-rows.npy",wrong_steps,rr,"SCHEDULE")
    def wrong_label(path):v=np.load(path);v[0]=(v[0]+1)%3;np.save(path,v)
    negative(2,"training-label-changed",shadow,"seed-25/selected/training-labels.npy",wrong_label,rr,"TRAINING_TARGET")
    negative(2,"objective-value-changed",shadow,"seed-25/selected/objective.json",json_change(lambda v:v.__setitem__("loss",v["loss"]+.1)),rr,"OBJECTIVE_LOSS")
    negative(2,"checkpoint-binding-changed",shadow,"seed-25/selected/report.json",json_change(lambda v:v.__setitem__("model_sha256","0"*64)),rr,"CHECKPOINT_PIN")
    negative(2,"wrong-plan-route",shadow,"seed-25/selected/report.json",json_change(lambda v:v.__setitem__("plan","all")),rr,"ARM_BINDING")
    negative(2,"output-probability-changed",shadow,"seed-25/selected/outputs.npz",npz_change("probabilities",lambda v:v.__setitem__(0,v[0,::-1])),rr,"OUTPUT")
    phases["optimization_binding_boundary_negatives_and_restores"]=time.perf_counter()-tick;tick=time.perf_counter()
    negative(3,"accuracy-inflated",shadow,"seed-25/selected/metrics.json",json_change(lambda v:v["relation"]["evaluation"].__setitem__("accuracy",1.)),rr,"METRICS")
    negative(3,"missing-evidence-hidden",shadow,"seed-25/selected/metrics.json",json_change(lambda v:v["evidence"]["evaluation"].__setitem__("missing_evidence_refusals",-1)),rr,"METRICS")
    negative(3,"intervention-labelled-truth",shadow,"seed-25/selected/interventions.json",json_change(lambda v:v["evaluation"]["removed"].__setitem__("truth_assigned",True)),rr,"INTERVENTIONS")
    negative(3,"bootstrap-membership-changed",shadow,"bootstrap-plan.npz",npz_change("membership",lambda v:v.__setitem__(0,-1)),rr,"BOOTSTRAP_PLAN")
    negative(3,"replay-probability-changed",pshadow,"25-selected/outputs.npz",npz_change("probabilities",lambda v:v.__setitem__(0,v[0,::-1])),rp,"REPLAY_OUTPUTS")
    # Mutate root claims without refreshing unrelated payloads.
    raw=(shadow/"report.json").read_bytes();r=json.loads(raw);r["scientific_admission"]=True;replace_shadow(shadow/"report.json",r)
    try:
        refused=False
        try:rr()
        except ValueError as exc:refused="CLAIM" in str(exc)
        check(refused,"FALSE_ADMISSION");checks.append(dict(boundary=3,name="false-admission",passed=True))
    finally:(shadow/"report.json").write_bytes(raw)
    phases["anchor_refusal_replay_boundary_negatives_and_restores"]=time.perf_counter()-tick;tick=time.perf_counter()
    check(equal(read_result(shadow,rpin,source)["cold"],result["cold"]) and read_replay(pshadow,ppin,result)==replay,"RESTORED_READBACK")
    after={p:file_sha(Path(p)) for p in before};check(after==before,"ORIGINALS_CHANGED");publish(a.output/"after.json",after)
    phases["restored_shadows_and_unchanged_originals"]=time.perf_counter()-tick;wall=time.perf_counter()-start
    publish(a.output/"fsv.json",dict(passed=True,checks=checks,result_sha256=rpin,replay_sha256=ppin,phases_seconds=phases,wall_seconds=wall,
        phase_coverage=sum(phases.values())/wall,scientific_admission=False,assertion_verdict="REFUSE_UNADMITTED_PANEL"))
    print(json.dumps(dict(passed=True,checks=len(checks),seconds=wall)),flush=True)


if __name__=="__main__":sys.exit(run_cli(main))
