"""Three-boundary persisted relation-result fault audit; source originals immutable."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit.scifact_population_readback import reference
from audit.scifact_result_readback import read_replayed
from senses.contract import canonical
from senses.trajectory_population import publish
from text.evaluation_multiscale import file_sha


def main():
    p=argparse.ArgumentParser()
    for n in ("population","inventory","result","replay","output"):p.add_argument("--"+n,type=Path,required=True)
    for n in ("population-sha256","result-sha256","replay-sha256"):p.add_argument("--"+n,required=True)
    a=p.parse_args();torch.set_num_threads(1);start=time.perf_counter();phases={};a.output.mkdir(parents=True,exist_ok=False)
    context=reference(a.inventory);before={str(p):file_sha(p) for root in (a.result,a.replay,a.population) for p in root.rglob('*') if p.is_file()}
    before.update(context["sources"]);publish(a.output/"before.json",before)
    def read(root,pin,replay,replay_pin):return read_replayed(root,pin,a.population,a.population_sha256,a.inventory,replay,replay_pin,context)
    cold=read(a.result,a.result_sha256,a.replay,a.replay_sha256);publish(a.output/"cold-readback.json",cold)
    checks=[dict(name="B1/B2/B3 full native bindings, targets, schedules, checkpoints, fresh replay and independent metrics",passed=True)]
    phases["source_and_complete_result_readback"]=time.perf_counter()-start;tick=time.perf_counter()
    shadow=a.output/"owned-shadow";shutil.copytree(a.result,shadow);replay_shadow=a.output/"replay-shadow";shutil.copytree(a.replay,replay_shadow)
    controls=a.output/"controls";controls.mkdir()
    def fault(name,path,fn,code,kind="json",fresh=False):
        base=replay_shadow if fresh else shadow;file=base/path;restore={path:file.read_bytes(),"report.json":(base/"report.json").read_bytes()}
        if kind=="npy":value=np.load(file,allow_pickle=False).copy()
        elif kind=="pt":value=torch.load(file,map_location="cpu",weights_only=True)
        else:value=json.loads(file.read_bytes())
        changed=fn(value)
        if changed is not None:value=changed
        if kind=="npy":np.save(file,value)
        elif kind=="pt":torch.save(value,file)
        else:file.write_bytes(canonical(value))
        if path!="report.json":
            rep=json.loads(restore["report.json"]);rep["files"][path]=dict(bytes=file.stat().st_size,sha256=file_sha(file))
            if path.endswith('/report.json'):
                seed,arm,_=path.split('/');rep["arms"][seed.removeprefix('seed-')+'/'+arm]["report_sha256"]=file_sha(file)
            (base/"report.json").write_bytes(canonical(rep))
        publish(controls/(name+".json"),dict(path=path,mutant_sha256=file_sha(file),fresh=fresh))
        try:read(shadow,file_sha(shadow/"report.json"),replay_shadow,file_sha(replay_shadow/"report.json"));observed="accepted"
        except ValueError as exc:observed=str(exc)
        checks.append(dict(name=name,expected=code,observed=observed,passed=code in observed))
        for n,raw in restore.items():(base/n).write_bytes(raw)
    fault("B1_wrong_parent","report.json",lambda r:r.update(population_sha256="0"*64),"PARENTS")
    fault("B1_wrong_native_pin","input-pins.json",lambda r:r[0].update(native_sha256="0"*64),"NATIVE_PINS")
    fault("B1_wrong_gold","labels.npy",lambda r:r.__setitem__(0,(r[0]+1)%3),"TARGETS","npy")
    fault("B1_wrong_train_rows","seed-25/training-rows.npy",lambda r:r.__setitem__((0,0),1258),"SCHEDULE","npy")
    fault("B1_wrong_permutation","seed-25/permuted/target-source-rows.npy",lambda r:r.__setitem__(0,1258),"TARGET_MAPPING","npy")
    fault("B2_wrong_checkpoint_pin","seed-25/joint-all/report.json",lambda r:r.update(model_sha256="0"*64),"CHECKPOINT_PIN")
    def weight(r):next(iter(r.values())).view(-1)[0]+=1
    fault("B2_changed_checkpoint","seed-25/joint-all/model.pt",weight,"CHECKPOINT_PIN","pt")
    fault("B2_changed_predictions","seed-25/joint-all/probabilities.npy",lambda r:r.__setitem__((0,0),r[0,0]+.01),"REPLAY","npy")
    fault("B2_false_scope_control","seed-25/claim-only/report.json",lambda r:r["input_controls"].update(evidence_ignored=False),"CONTROLS")
    fault("B3_wrong_metrics","seed-25/joint-all/metrics.json",lambda r:r["evaluation"].update(accuracy=1.),"METRICS")
    fault("B3_wrong_prior","prior.npy",lambda r:r.__setitem__((0,0),.99),"PRIOR","npy")
    for flag in ("served","scientific_admission"):
        fault("B3_false_"+flag,"report.json",lambda r,k=flag:r.update({k:True}),"CLAIM")
    fault("B3_false_accept","report.json",lambda r:r.update(assertion_verdict="ACCEPT"),"CLAIM")
    fault("B3_wrong_fresh_parent","report.json",lambda r:r.update(result_sha256="0"*64),"FRESH_RESULT_PIN",fresh=True)
    fault("B3_changed_fresh_predictions","25-joint-all.npy",lambda r:r.__setitem__((0,0),r[0,0]+.01),"FRESH_PROBABILITIES","npy",True)
    fault("B3_false_fresh_admission","report.json",lambda r:r.update(scientific_admission=True),"FRESH_CLAIM",fresh=True)
    phases["persisted_fault_controls"]=time.perf_counter()-tick;tick=time.perf_counter()
    after={n:file_sha(Path(n)) for n in before};publish(a.output/"after.json",after)
    restored=read(shadow,a.result_sha256,replay_shadow,a.replay_sha256)
    checks.append(dict(name="B1/B2/B3 unchanged originals and exact restored publications",passed=before==after and restored==cold))
    phases["source_drift_and_restoration"]=time.perf_counter()-tick;wall=time.perf_counter()-start
    result=dict(passed=all(c["passed"] for c in checks),checks=checks,acceptance="author-checked",scientific_admission=False,
        phases_seconds=phases,wall_seconds=wall,phase_coverage=sum(phases.values())/wall)
    publish(a.output/"fsv.json",result);print(json.dumps(dict(passed=result["passed"],checks=len(checks),failed=[c for c in checks if not c["passed"]],wall_seconds=wall)))
    return 0 if result["passed"] else 2


if __name__=="__main__":raise SystemExit(main())
