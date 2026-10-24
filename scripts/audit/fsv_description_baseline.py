"""Physical three-boundary controls over the actual description diagnostic output."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit.description_readback import read
from audit.anchor_population_readback import lines
from senses.contract import canonical
from senses.trajectory_population import publish
from text.evaluation_multiscale import file_sha


def inventory(root):return {p.relative_to(root).as_posix():file_sha(p) for p in root.rglob('*') if p.is_file()}


def main():
    p=argparse.ArgumentParser()
    for n in ("population","result","diagnostic","output"):p.add_argument("--"+n,type=Path,required=True)
    p.add_argument("--report-sha256",required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter();phases={};before=inventory(a.diagnostic);publish(a.output/"before.json",before)
    cold=read(a.population,a.result,a.diagnostic,a.report_sha256);publish(a.output/"cold-readback.json",cold)
    shadow=a.output/"owned-shadow";shutil.copytree(a.diagnostic,shadow);controls=a.output/"controls";controls.mkdir()
    report=json.loads((shadow/"report.json").read_bytes());checks=[dict(name="B1/B2/B3 independent complete source features donors metrics and intervals",passed=True)]
    phases["full_readback_and_owned_shadow"]=time.perf_counter()-start;tick=time.perf_counter()

    def fault(name,relative,mutate,code,kind="json",features=False):
        path=shadow/relative;original=path.read_bytes();r=copy.deepcopy(report)
        value=np.load(path,allow_pickle=False) if kind=="npy" else json.loads(original)
        mutate(value)
        if kind=="npy":np.save(path,value)
        else:path.write_bytes(canonical(value))
        if relative!="report.json":
            r["files"][relative]=dict(bytes=path.stat().st_size,sha256=file_sha(path));(shadow/"report.json").write_bytes(canonical(r))
        publish(controls/(name+".json"),dict(relative=relative,mutated_sha256=file_sha(path),expected=code))
        try:read(a.population,a.result,shadow,file_sha(shadow/"report.json"),audit_features=features);observed="accepted"
        except ValueError as exc:observed=str(exc)
        finally:path.write_bytes(original);(shadow/"report.json").write_bytes((a.diagnostic/"report.json").read_bytes())
        checks.append(dict(name=name,expected=code,observed=observed,passed=observed==code))

    pre="MNLM_DESCRIPTION_READ_";rows=lines(a.population/"rows.jsonl")
    train=next(i for i,r in enumerate(rows) if r["split"]=="train" and r["question"]==0)
    evaluation=next(i for i,r in enumerate(rows) if r["split"]=="evaluation" and r["question"]==0)
    fault("B1_source_substitution","input-rows.json",lambda r:r[0].update(text_sha256="0"*64),pre+"INPUT_ROWS")
    fault("B1_wrong_parent","report.json",lambda r:r.update(population_sha256="0"*64),pre+"POPULATION_PIN")
    fault("B2_evaluation_fit_contamination","word/0/state.json",lambda r:r["training_rows"].__setitem__(0,evaluation),pre+"TRAINING_ROWS",features=True)
    fault("B2_label_derived_feature_state","word/0/state.json",lambda r:r.update(anchor_label_feature=True),pre+"FEATURE_STATE",features=True)
    fault("B2_changed_idf","word/0/state.json",lambda r:r["idf"].__setitem__(0,r["idf"][0]+1),pre+"IDF",features=True)
    fault("B2_changed_scalar_normalization","structure/0/state.json",lambda r:r["mean"].__setitem__(0,r["mean"][0]+1),pre+"NORMALIZATION",features=True)
    fault("B2_heldout_donor","word/donors.npy",lambda r:r.__setitem__((train,0),evaluation),pre+"DONOR_SCOPE","npy")
    fault("B2_same_group_self_donor","word/donors.npy",lambda r:r.__setitem__((train,0),train),pre+"DONOR_SCOPE","npy")
    fault("B3_false_probability","word/probabilities.npy",lambda r:r.__setitem__((0,0),float('nan')),pre+"PROBABILITIES","npy")
    fault("B3_false_metrics","word/metrics.json",lambda r:r["0"]["evaluation"].update(accuracy=42),"MNLM_ANCHOR_RESULT_READ_METRICS")
    fault("B3_false_interval","paired-comparisons.json",lambda r:next(iter(r["findings"].values()))["low"].__setitem__(0,42),pre+"COMPARISONS")
    for flag in ("scientific_admission","production_assertions","shannon_cmi_claim","baseline_is_sense"):
        fault("B3_false_"+flag,"report.json",lambda r,k=flag:r.update({k:True}),pre+"CLAIM")
    verdicts=[]
    for kind in ("word","char3","structure"):
        prob=np.load(a.diagnostic/kind/"probabilities.npy",allow_pickle=False);donors=np.load(a.diagnostic/kind/"donors.npy",allow_pickle=False)
        for i,r in enumerate(rows):
            verdicts.append(dict(baseline=kind,case=r["case"],source_supported=int(prob[i].argmax()) in r["positive_classes"],asserted=False,
                reason="MNLM_DESCRIPTION_DIAGNOSTIC_ONLY",retrieval_status="NO_POSITIVE_DESCRIPTION_MATCH" if donors[i,0]<0 else "TRAINING_DESCRIPTION_NEIGHBORS"))
    (a.output/"source-verdicts.jsonl").write_bytes(b''.join(canonical(v)+b'\n' for v in verdicts))
    checks.append(dict(name="B3 every baseline prediction refuses production assertion",passed=len(verdicts)==4536 and all(v["asserted"] is False for v in verdicts)))
    phases["persisted_negative_controls_and_refusals"]=time.perf_counter()-tick;tick=time.perf_counter()
    after=inventory(a.diagnostic);publish(a.output/"after.json",after)
    checks.append(dict(name="B1/B3 originals and restored shadow unchanged",passed=before==after==inventory(shadow) and read(a.population,a.result,shadow,a.report_sha256)==cold))
    phases["source_drift_and_final_full_readback"]=time.perf_counter()-tick;wall=time.perf_counter()-start
    r=dict(author_checked=True,passed=all(c["passed"] for c in checks),checks=checks,report_sha256=a.report_sha256,
        scientific_admission=False,phases_seconds=phases,wall_seconds=wall,phase_coverage=sum(phases.values())/wall)
    publish(a.output/"fsv.json",r);print(json.dumps(dict(passed=r["passed"],checks=len(checks),failed=[c for c in checks if not c["passed"]],wall_seconds=wall)))
    return 0 if r["passed"] else 2


if __name__=="__main__":sys.exit(main())
