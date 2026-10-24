"""Three-boundary FSV over actual fixed-budget sensory readout artifacts."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit.anchor_readout_readback import read,verdict
from audit.anchor_population_readback import lines
from text.evaluation_multiscale import file_sha
from senses.contract import canonical
from senses.trajectory_population import publish


def files(root):return {f.relative_to(root).as_posix():file_sha(f) for f in root.rglob('*') if f.is_file()}


def main():
    p=argparse.ArgumentParser()
    for n in ("population","result","output"):p.add_argument("--"+n,type=Path,required=True)
    p.add_argument("--report-sha256",required=True);a=p.parse_args();started=time.perf_counter();phases={}
    a.output.mkdir(parents=True,exist_ok=False);before=files(a.result);publish(a.output/"before.json",before)
    cold=read(a.population,a.result,a.report_sha256);publish(a.output/"cold-readback.json",cold)
    shadow=a.output/"owned-shadow";shutil.copytree(a.result,shadow);controls=a.output/"controls";controls.mkdir()
    report=json.loads((shadow/"report.json").read_bytes());checks=[dict(name="B1/B2/B3 native input checkpoint source outcomes",passed=True)]
    phases["full_cold_read_and_owned_copy"]=time.perf_counter()-started;tick=time.perf_counter()

    def fault(name,relative,mutation,code,kind="json",audit_inputs=False):
        path=shadow/relative;original=path.read_bytes();r=copy.deepcopy(report)
        value=np.load(path,allow_pickle=False) if kind=="npy" else json.loads(original)
        mutation(value)
        if kind=="npy":np.save(path,value)
        else:path.write_bytes(canonical(value))
        if relative!="report.json":
            r["files"][relative]=dict(bytes=path.stat().st_size,sha256=file_sha(path));(shadow/"report.json").write_bytes(canonical(r))
        publish(controls/(name+".json"),dict(relative=relative,mutated_sha256=file_sha(path),expected=code))
        try:read(a.population,shadow,file_sha(shadow/"report.json"),audit_inputs=audit_inputs);observed="accepted"
        except ValueError as exc:observed=str(exc)
        finally:path.write_bytes(original);(shadow/"report.json").write_bytes((a.result/"report.json").read_bytes())
        checks.append(dict(name=name,expected=code,observed=observed,passed=observed==code))

    prefix="MNLM_ANCHOR_RESULT_READ_";base="seed-25/all-ten/"
    fault("B1_wrong_population","report.json",lambda r:r.update(population_sha256="0"*64),"MNLM_ANCHOR_POP_READ_REPORT_PIN")
    fault("B1_native_input_rebinding","input-pins.json",lambda r:r[0].update(memory_sha256="0"*64),prefix+"INPUT_BINDING",audit_inputs=True)
    eval_row=next(i for i,r in enumerate(lines(a.population/"rows.jsonl")) if r["split"]=="evaluation")
    fault("B2_evaluation_row_in_training","seed-25/training-rows.npy",lambda r:r.__setitem__((0,0),eval_row),prefix+"TRAINING_SPLIT","npy")
    fault("B2_wrong_target_mapping",base+"target-source-rows.npy",lambda r:r.__setitem__(0,(int(r[0])+1)%len(r)),prefix+"TARGET_MAPPING","npy")
    fault("B2_false_model_pin","report.json",lambda r:r["arms"]["25/all-ten"].update(model_sha256="0"*64),prefix+"ARM")
    fault("B2_nonfinite_loss",base+"losses.npy",lambda r:r.__setitem__(0,float('nan')),prefix+"LOSSES","npy")
    fault("B3_nonfinite_probability",base+"probabilities.npy",lambda r:r.__setitem__((0,0),float('nan')),prefix+"PROBABILITIES","npy")
    fault("B3_replayed_different_probability",base+"replay-probabilities.npy",lambda r:r.__setitem__((0,0),float(r[0,0])+0.01),prefix+"REPLAY","npy")
    fault("B3_forged_accuracy",base+"metrics.json",lambda r:r["0"]["evaluation"].update(accuracy=42),prefix+"METRICS")
    fault("B3_false_prior","prior.npy",lambda r:r.__setitem__((0,slice(None)),np.roll(r[0],1)),prefix+"PRIOR","npy")
    for flag in ("scientific_admission","production_assertions","shannon_cmi_claim"):
        fault("B3_false_"+flag,"report.json",lambda r,k=flag:r.update({k:True}),prefix+"CLAIM")
    rows=lines(a.population/"rows.jsonl");verdicts=[]
    for arm in report["arms"]:
        seed,name=arm.split('/');prob=np.load(a.result/("seed-"+seed)/name/"probabilities.npy",allow_pickle=False)
        for i,row in enumerate(rows):verdicts.append(dict(arm=arm,**verdict(prob[i],row)))
    (a.output/"source-verdicts.jsonl").write_bytes(b''.join(canonical(r)+b'\n' for r in verdicts))
    checks.append(dict(name="B3 all source predictions refuse production assertion",passed=len(verdicts)==10584 and all(r["asserted"] is False for r in verdicts)))
    phases["persisted_negative_controls_and_consumer_verdicts"]=time.perf_counter()-tick;tick=time.perf_counter()
    after=files(a.result);publish(a.output/"after.json",after)
    checks.append(dict(name="B1 original and restored result unchanged",passed=before==after==files(shadow) and read(a.population,shadow,a.report_sha256)==cold))
    phases["final_cold_read_and_drift"]=time.perf_counter()-tick;wall=time.perf_counter()-started
    result=dict(author_checked=True,passed=all(r["passed"] for r in checks),checks=checks,result_sha256=a.report_sha256,
        population_sha256=cold["population_sha256"],scientific_admission=False,phases_seconds=phases,wall_seconds=wall,phase_coverage=sum(phases.values())/wall)
    publish(a.output/"fsv.json",result)
    print(json.dumps(dict(passed=result["passed"],checks=len(checks),failed=[r for r in checks if not r["passed"]],wall_seconds=wall)))
    return 0 if result["passed"] else 2


if __name__=="__main__":sys.exit(main())
