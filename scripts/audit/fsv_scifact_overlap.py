"""Persisted faults across source views, overlap closure and case partitions."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit.scifact_overlap_readback import read
from audit.scifact_readback import jsonl
from senses.contract import canonical
from senses.trajectory_population import publish
from text.evaluation_multiscale import file_sha


def inventory(root):return {p.relative_to(root).as_posix():file_sha(p) for p in root.rglob('*') if p.is_file()}


def main():
    p=argparse.ArgumentParser()
    for n in ("source","publication","inputs","overlap","output"):p.add_argument("--"+n,type=Path,required=True)
    for n in ("source","publication","inputs","overlap"):p.add_argument("--"+n+"-sha256",required=True)
    a=p.parse_args();started=time.perf_counter();a.output.mkdir(parents=True,exist_ok=False);phases={}
    parents=[a.source,a.publication,a.inputs];pins=[a.source_sha256,a.publication_sha256,a.inputs_sha256]
    before={n:inventory(getattr(a,n)) for n in ("source","publication","inputs","overlap")};publish(a.output/"before.json",before)
    def audit(root,pin):return read(*parents,pins,root,pin)
    cold=audit(a.overlap,a.overlap_sha256);publish(a.output/"cold-readback.json",cold)
    checks=[dict(name="B1/B2/B3 all source views, complete lexical candidates and case partitions independently reconstructed",passed=True)]
    phases["cold_all_boundaries"]=time.perf_counter()-started;tick=time.perf_counter()
    shadow=a.output/"owned-shadow";shutil.copytree(a.overlap,shadow);controls=a.output/"controls";controls.mkdir()
    originals={n:(shadow/n).read_bytes() for n in before["overlap"]}
    def fault(name,file,change,code,kind="jsonl"):
        path=shadow/file
        value=np.load(path,allow_pickle=False) if kind=="npy" else json.loads(path.read_bytes()) if kind=="json" else jsonl(path)
        revised=change(value)
        if revised is not None:value=revised
        if kind=="npy":np.save(path,value)
        else:path.write_bytes(canonical(value) if kind=="json" else b''.join(canonical(r)+b'\n' for r in value))
        if file.startswith("overlap/") and file!="overlap/report.json":
            rep=json.loads((shadow/"overlap/report.json").read_bytes());local=file.removeprefix("overlap/")
            rep["files"][local]=dict(bytes=path.stat().st_size,sha256=file_sha(path));(shadow/"overlap/report.json").write_bytes(canonical(rep))
        if file!="report.json":
            rep=json.loads(originals["report.json"])
            for n in rep["files"]:rep["files"][n]=dict(bytes=(shadow/n).stat().st_size,sha256=file_sha(shadow/n))
            (shadow/"report.json").write_bytes(canonical(rep))
        publish(controls/(name+".json"),dict(file=file,mutant_sha256=file_sha(path)))
        try:audit(shadow,file_sha(shadow/"report.json"));observed="accepted"
        except (ValueError,RuntimeError) as exc:observed=str(exc)
        checks.append(dict(name=name,expected=code,observed=observed,passed=code in observed))
        for n,raw in originals.items():(shadow/n).write_bytes(raw)
    fault("B1_wrong_parent","report.json",lambda r:r.update(parent_inputs_sha256="0"*64),"MNLM_SCIFACT_OVERLAP_READBACK_PARENT","json")
    fault("B1_omitted_view","capture/scifact.jsonl",lambda r:r[:-1],"MNLM_SCIFACT_OVERLAP_READBACK_CAPTURE_POPULATION")
    fault("B1_changed_source_group","capture/scifact.jsonl",lambda r:r[0].update(group_keys=["wrong"]),"MNLM_SCIFACT_OVERLAP_READBACK_CAPTURE_SOURCE")
    fault("B1_changed_text","capture/scifact.jsonl",lambda r:r[0]["fields"][0].update(text="changed"),"MNLM_SCIFACT_OVERLAP_READBACK_CAPTURE_SOURCE")
    fault("B2_changed_threshold","overlap/report.json",lambda r:r.update(threshold=dict(numerator=4,denominator=5)),"MNLM_OVERLAP_READBACK_CONTRACT","json")
    fault("B2_omitted_candidate","overlap/candidates.npy",lambda r:r[:-1],"MNLM_OVERLAP_READBACK_CANDIDATES","npy")
    fault("B2_omitted_link","overlap/links.jsonl",lambda r:r[:-1],"MNLM_OVERLAP_READBACK_LINKS")
    fault("B2_omitted_group_member","overlap/groups.jsonl",lambda r:r[0].update(members=r[0]["members"][:-1]),"MNLM_OVERLAP_READBACK_GROUPS")
    fault("B2_reassigned_source_component","source-groups.json",lambda r:r.update({next(iter(r)):"wrong"}),"MNLM_SCIFACT_OVERLAP_READBACK_GROUP_MAP","json")
    fault("B3_missing_case","cases.jsonl",lambda r:r[:-1],"MNLM_SCIFACT_OVERLAP_READBACK_CASE_POPULATION")
    fault("B3_case_crosses_component","cases.jsonl",lambda r:r[0].update(group="wrong"),"MNLM_SCIFACT_OVERLAP_READBACK_CASE_BINDING")
    fault("B3_case_crosses_partition","cases.jsonl",lambda r:r[0].update(partition="evaluation" if r[0]["partition"]!="evaluation" else "train"),"MNLM_SCIFACT_OVERLAP_READBACK_CASE_BINDING")
    fault("B3_erased_parent_partition","cases.jsonl",lambda r:r[0].update(parent_partition="wrong"),"MNLM_SCIFACT_OVERLAP_READBACK_CASE_BINDING")
    for flag in ("labels_used_for_grouping","sealed_evaluation","scientific_admission","source_time_composition_separated","encoder_exposure_certified","served"):
        fault("B3_false_"+flag,"report.json",lambda r,k=flag:r.update({k:True}),"MNLM_SCIFACT_OVERLAP_READBACK_CLAIM","json")
    fault("B3_lower_native_floor","report.json",lambda r:r.update(native_text_floor=4),"MNLM_SCIFACT_OVERLAP_READBACK_CLAIM","json")
    phases["persisted_fault_readbacks"]=time.perf_counter()-tick;tick=time.perf_counter()
    after={n:inventory(getattr(a,n)) for n in before};publish(a.output/"after.json",after)
    restored=audit(shadow,a.overlap_sha256)
    checks.append(dict(name="B1/B2/B3 original sources unchanged and owned shadow exactly restored",passed=before==after and inventory(shadow)==before["overlap"] and cold==restored))
    phases["final_source_and_restoration"]=time.perf_counter()-tick;wall=time.perf_counter()-started
    result=dict(passed=all(c["passed"] for c in checks),checks=checks,acceptance="author-checked",scientific_admission=False,
                phases_seconds=phases,wall_seconds=wall,phase_coverage=sum(phases.values())/wall)
    publish(a.output/"fsv.json",result)
    print(json.dumps(dict(passed=result["passed"],checks=len(checks),failed=[c for c in checks if not c["passed"]],wall_seconds=wall,phase_coverage=result["phase_coverage"])))
    return 0 if result["passed"] else 2


if __name__=="__main__":raise SystemExit(main())
