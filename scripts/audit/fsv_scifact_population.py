"""Complete persisted population readback plus real native batch gather controls."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit.scifact_population_readback import read,reference
from audit.scifact_readback import jsonl
from model.scifact_readout import NativeStore,array_sha
from senses.contract import canonical
from senses.trajectory_population import publish
from text.evaluation_multiscale import file_sha


def main():
    p=argparse.ArgumentParser()
    for n in ("population","inventory","output"):p.add_argument("--"+n,type=Path,required=True)
    p.add_argument("--report-sha256",required=True);a=p.parse_args();start=time.perf_counter();phases={};a.output.mkdir(parents=True,exist_ok=False)
    context=reference(a.inventory);before={str(p):file_sha(p) for p in a.population.rglob('*') if p.is_file()};before.update(context["sources"])
    publish(a.output/"before.json",before);cold=read(a.population,a.report_sha256,a.inventory,context);publish(a.output/"cold-readback.json",cold)
    checks=[dict(name="B1/B2 full source cases, targets, native shapes and every gather-map cell independently reconstructed",passed=True)]
    phases["independent_source_and_native_readback"]=time.perf_counter()-start;tick=time.perf_counter()
    shadow=a.output/"owned-shadow";shutil.copytree(a.population,shadow);controls=a.output/"controls";controls.mkdir()
    original={p.relative_to(shadow).as_posix():p.read_bytes() for p in shadow.rglob('*') if p.is_file()}
    def fault(name,path,fn,code,kind="jsonl"):
        file=shadow/path
        if kind=="npz":
            with np.load(file,allow_pickle=False) as z:value={k:z[k].copy() for k in z.files}
        else:value=json.loads(file.read_bytes()) if kind=="json" else jsonl(file)
        changed=fn(value)
        if changed is not None:value=changed
        if kind=="npz":np.savez(file,**value)
        else:file.write_bytes(canonical(value) if kind=="json" else b''.join(canonical(v)+b'\n' for v in value))
        if path!="report.json":
            rep=json.loads(original["report.json"]);rep["files"][path]=dict(bytes=file.stat().st_size,sha256=file_sha(file));(shadow/"report.json").write_bytes(canonical(rep))
        publish(controls/(name+".json"),dict(path=path,mutant_sha256=file_sha(file)))
        try:read(shadow,file_sha(shadow/"report.json"),a.inventory,context);observed="accepted"
        except ValueError as exc:observed=str(exc)
        checks.append(dict(name=name,expected=code,observed=observed,passed=code in observed))
        for n,raw in original.items():(shadow/n).write_bytes(raw)
    fault("B1_wrong_source","report.json",lambda r:r["parents"].update(inputs="0"*64),"PARENTS","json")
    for field,value in (("query_unit",11405),("document_index",666),("partition","wrong"),("group","wrong")):
        fault("B1_wrong_"+field,"rows.jsonl",lambda r,k=field,v=value:r[0].update({k:v}),"ROWS")
    fault("B1_changed_target","targets.jsonl",lambda r:r[0].update(label="SUPPORT"),"TARGETS")
    fault("B1_removed_source_view","documents.jsonl",lambda r:r[0].update(views=r[0]["views"][:-1]),"DOCUMENTS")
    path="maps/"+context["lanes"][0]["name"]+".npz"
    for key in ("query","indices","mask","levels","geometry"):
        def alter(m,k=key):
            arr=m[k];arr.flat[0]=not arr.flat[0] if k=="mask" else arr.flat[0]+1
        fault("B2_wrong_"+key,path,alter,"MAP_"+key.upper(),"npz")
    fault("B2_injected_label",path,lambda r:r.update(label=np.zeros(1258)),"MAP_KEYS","npz")
    fault("B2_swapped_lens","report.json",lambda r:r["lanes"].reverse(),"COUNTS","json")
    for flag in ("scientific_admission","served","sealed_evaluation"):
        fault("B3_false_"+flag,"report.json",lambda r,k=flag:r.update({k:True}),"CLAIM","json")
    fault("B3_false_admitted_count","report.json",lambda r:r.update(admitted_senses=14),"CLAIM","json")
    fault("B3_lower_floor","report.json",lambda r:r.update(native_text_floor=4),"CLAIM","json")
    phases["persisted_fault_controls"]=time.perf_counter()-tick;tick=time.perf_counter()
    import torch
    torch.set_num_threads(1);store=NativeStore(context,"cpu")
    largest=int(np.argmax(context["maps"][-1]["mask"].sum(1)));selected=sorted({0,629,1257,next(i for i,r in enumerate(context["rows"]) if r["document_index"]==largest)})
    actual=store.batch(selected);binding=[]
    for k,(memory,m,x) in enumerate(zip(actual,context["maps"],context["native"])):
        docs=np.array([context["rows"][i]["document_index"] for i in selected]);query=x[m["query"][selected]];evidence=x[m["indices"][docs]]
        equal=np.array_equal(memory["query"].numpy(),query) and np.array_equal(memory["evidence"].numpy(),evidence)
        equal=equal and all(np.array_equal(memory[key].numpy(),m[key][docs]) for key in ("mask","levels","geometry"))
        checks.append(dict(name="B3 actual typed gather "+context["lanes"][k]["name"],passed=equal))
        binding.append(dict(name=context["lanes"][k]["name"],query_sha256=array_sha(query),evidence_sha256=array_sha(evidence)))
    publish(a.output/"consumer-readback.json",dict(case_rows=selected,bindings=binding,encoder_forward=False,model_forward=False))
    phases["actual_consumer_gather"]=time.perf_counter()-tick;tick=time.perf_counter()
    after={n:file_sha(Path(n)) for n in before};publish(a.output/"after.json",after)
    final=read(shadow,a.report_sha256,a.inventory,context)
    checks.append(dict(name="B1/B2/B3 unchanged source/native bytes and exact shadow restoration",passed=before==after and cold==final and all((shadow/n).read_bytes()==raw for n,raw in original.items())))
    phases["final_drift_and_restoration"]=time.perf_counter()-tick;wall=time.perf_counter()-start
    result=dict(passed=all(c["passed"] for c in checks),checks=checks,acceptance="author-checked",scientific_admission=False,
                phases_seconds=phases,wall_seconds=wall,phase_coverage=sum(phases.values())/wall)
    publish(a.output/"fsv.json",result);print(json.dumps(dict(passed=result["passed"],checks=len(checks),failed=[c for c in checks if not c["passed"]],wall_seconds=wall,phase_coverage=result["phase_coverage"])))
    return 0 if result["passed"] else 2


if __name__=="__main__":raise SystemExit(main())
