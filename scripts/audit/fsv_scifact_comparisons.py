"""Source, training-only donors, and paired comparison FSV with owned fault shadows."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit.scifact_comparison_readback import reference,read
from senses.contract import canonical
from senses.trajectory_population import publish
from text.evaluation_multiscale import file_sha


def main():
    p=argparse.ArgumentParser()
    for n in ("population","inventory","result","replay","comparisons","output"):p.add_argument("--"+n,type=Path,required=True)
    p.add_argument("--report-sha256",required=True);a=p.parse_args();torch.set_num_threads(1)
    start=time.perf_counter();phases={};a.output.mkdir(parents=True,exist_ok=False)
    ctx=reference(a.comparisons,a.population,a.inventory,a.result,a.replay)
    before={str(p):file_sha(p) for root in (a.comparisons,a.result,a.replay,a.population) for p in root.rglob('*') if p.is_file()};before.update(ctx["sources"])
    publish(a.output/"before.json",before);cold=read(a.comparisons,a.report_sha256,ctx);publish(a.output/"cold-readback.json",cold)
    checks=[dict(name="B1/B2/B3 independent source text, train-only features, every donor, prediction, metric and clustered comparison",passed=True)]
    phases["independent_source_feature_donor_bootstrap_readback"]=time.perf_counter()-start;tick=time.perf_counter()
    shadow=a.output/"owned-shadow";shutil.copytree(a.comparisons,shadow);controls=a.output/"controls";controls.mkdir()
    def fault(name,path,fn,code,kind="json"):
        file=shadow/path;restore={path:file.read_bytes(),"report.json":(shadow/"report.json").read_bytes()}
        if kind=="npy":value=np.load(file,allow_pickle=False).copy()
        elif kind=="npz":
            with np.load(file,allow_pickle=False) as z:value={k:z[k].copy() for k in z.files}
        else:value=json.loads(file.read_bytes())
        changed=fn(value)
        if changed is not None:value=changed
        if kind=="npy":np.save(file,value)
        elif kind=="npz":np.savez(file,**value)
        else:file.write_bytes(canonical(value))
        if path!="report.json":
            rep=json.loads(restore["report.json"]);rep["files"][path]=dict(bytes=file.stat().st_size,sha256=file_sha(file));(shadow/"report.json").write_bytes(canonical(rep))
        publish(controls/(name+".json"),dict(path=path,mutant_sha256=file_sha(file)))
        try:read(shadow,file_sha(shadow/"report.json"),ctx);observed="accepted"
        except ValueError as exc:observed=str(exc)
        checks.append(dict(name=name,expected=code,observed=observed,passed=code in observed))
        for n,raw in restore.items():(shadow/n).write_bytes(raw)
    fault("B1_wrong_parent","report.json",lambda r:r.update(result_sha256="0"*64),"PARENTS")
    fault("B1_wrong_source_text","input-rows.json",lambda r:r[0]["text_sha256"].update(claim="0"*64),"SOURCE_BINDINGS")
    fault("B1_wrong_component","input-rows.json",lambda r:r[0].update(group="wrong"),"SOURCE_BINDINGS")
    fault("B2_wrong_training_rows","word-claim/state.json",lambda r:r.update(training_rows=[]),"TRAINING_FEATURE_STATE")
    fault("B2_wrong_idf","char3-evidence/state.json",lambda r:r["idf"].__setitem__(0,99.),"TRAINING_FEATURE_STATE")
    fault("B2_wrong_feature","word-evidence/features.npz",lambda r:r["data"].__setitem__(0,r["data"][0]+.01),"FEATURES","npz")
    fault("B2_same_component_donor","word-claim/donors.npy",lambda r:r.__setitem__((0,0),0),"DONORS","npy")
    fault("B2_wrong_similarity","char3-pair/similarities.npy",lambda r:r.__setitem__((0,0),99.),"SIMILARITIES","npy")
    fault("B3_wrong_prediction","word-pair/probabilities.npy",lambda r:r.__setitem__((0,0),.999),"PROBABILITIES","npy")
    fault("B3_wrong_metric","char3-claim/metrics.json",lambda r:r["evaluation"].update(accuracy=1.),"METRICS")
    fault("B3_wrong_cluster","bootstrap-plan.npz",lambda r:r["membership"].__setitem__(0,99),"BOOTSTRAP_PLAN","npz")
    fault("B3_wrong_draw","bootstrap-plan.npz",lambda r:r["draw"].__setitem__((0,0,0),99),"BOOTSTRAP_PLAN","npz")
    def sample(r):next(iter(r.values()))[0,0,0]+=1
    fault("B3_wrong_sample","bootstrap-samples.npz",sample,"BOOTSTRAP_SAMPLES","npz")
    fault("B3_wrong_interval","paired-comparisons.json",lambda r:next(iter(r["findings"].values()))["low"].__setitem__(0,99.),"INTERVALS")
    fault("B3_wrong_contrast","same-document-contrasts.json",lambda r:r["evaluation"].update(documents=999),"CONTRASTS")
    for flag in ("scientific_admission","baseline_is_sense","shannon_cmi_claim","served"):
        fault("B3_false_"+flag,"report.json",lambda r,k=flag:r.update({k:True}),"CLAIM")
    phases["persisted_fault_controls"]=time.perf_counter()-tick;tick=time.perf_counter()
    after={n:file_sha(Path(n)) for n in before};publish(a.output/"after.json",after);restored=read(shadow,a.report_sha256,ctx)
    checks.append(dict(name="B1/B2/B3 immutable source/result bytes and exact shadow restoration",passed=before==after and restored==cold))
    phases["source_drift_and_restoration"]=time.perf_counter()-tick;wall=time.perf_counter()-start
    result=dict(passed=all(c["passed"] for c in checks),checks=checks,acceptance="author-checked",scientific_admission=False,
        phases_seconds=phases,wall_seconds=wall,phase_coverage=sum(phases.values())/wall)
    publish(a.output/"fsv.json",result);print(json.dumps(dict(passed=result["passed"],checks=len(checks),failed=[c for c in checks if not c["passed"]],wall_seconds=wall)))
    return 0 if result["passed"] else 2


if __name__=="__main__":raise SystemExit(main())
