"""Source-exact sentence membership and canonical OR-of-AND supervision paths."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit.scifact_population_readback import reference,read
from model.scifact_readout import check
from senses.contract import canonical
from senses.trajectory_population import publish
from text.evaluation_multiscale import file_sha


def project(population,pin,inventory,output):
    check(not output.exists(),"OUTPUT_EXISTS");start=time.perf_counter();phases={};ctx=reference(inventory);read(population,pin,inventory,ctx)
    output.mkdir(parents=True);(output/"membership").mkdir();docs=ctx["documents"];width=max(d["sentences"] for d in docs)
    valid=np.arange(width)[None,:]<np.array([d["sentences"] for d in docs])[:,None];np.save(output/"valid-sentences.npy",valid)
    records=[]
    for d in docs:
        views=[v for v in d["views"] if v["level"]==1];check(len(views)==d["sentences"] and [v["span"] for v in views]==[[i,i+1] for i in range(d["sentences"])],"SOURCE_SENTENCES")
        records.append(dict(document=d["document"],sentences=views))
    (output/"sentences.jsonl").write_bytes(b''.join(canonical(r)+b'\n' for r in records));counts={}
    for lane,ptr,m in zip(ctx["lanes"],ctx["pointers"],ctx["maps"]):
        member=np.zeros((len(docs),width,m["indices"].shape[1]),bool)
        for i,d in enumerate(docs):
            column=0
            for v in d["views"]:
                lo,hi=ptr[v["unit"]:v["unit"]+2];end=column+int(hi-lo)
                check(np.array_equal(m["indices"][i,column:end],np.arange(lo,hi)) and m["mask"][i,column:end].all(),"WINDOW_ORDER")
                if v["level"]==1:member[i,v["span"][0],column:end]=True
                column=end
            check(column==int(m["mask"][i].sum()),"WINDOW_COVERAGE")
        check(np.array_equal(member.any(1),m["mask"]&(m["levels"]==1)),"SENTENCE_COVERAGE")
        np.savez_compressed(output/"membership"/(lane["name"]+".npz"),membership=member);counts[lane["name"]]=int(member.sum())
    phases["parent_readback_and_source_window_membership"]=time.perf_counter()-start;tick=time.perf_counter();paths=[]
    for row,target in zip(ctx["rows"],ctx["targets"]):
        check(row["id"]==target["case"],"TARGET_ORDER");alternatives=[]
        for rationale in target["rationales"]:
            s=sorted(rationale["sentences"]);check(s and len(set(s))==len(s) and 0<=s[0]<=s[-1]<docs[row["document_index"]]["sentences"],"RATIONALE")
            check(rationale["label"]==target["label"] and s+[width] not in alternatives,"RATIONALE_LABEL_OR_DUPLICATE");alternatives.append(s+[width])
        paths.append(dict(case=row["id"],partition=row["partition"],alternatives=alternatives))
    (output/"paths.jsonl").write_bytes(b''.join(canonical(r)+b'\n' for r in paths))
    alternatives=max(len(r["alternatives"]) for r in paths);length=max(len(p) for r in paths for p in r["alternatives"])
    choices=np.full((len(paths),alternatives,length),width,np.int64);steps=np.zeros(choices.shape,bool);present=np.zeros(choices.shape[:2],bool)
    for i,r in enumerate(paths):
        for j,p in enumerate(r["alternatives"]):choices[i,j,:len(p)]=p;steps[i,j,:len(p)]=True;present[i,j]=True
    np.savez(output/"targets.npz",choices=choices,steps=steps,present=present)
    phases["complete_alternative_paths"]=time.perf_counter()-tick;tick=time.perf_counter()
    check(all(file_sha(Path(p))==h for p,h in ctx["sources"].items()),"SOURCE_DRIFT")
    files={p.relative_to(output).as_posix():dict(bytes=p.stat().st_size,sha256=file_sha(p)) for p in output.rglob('*') if p.is_file()}
    phases["immutable_parents_and_output_pins"]=time.perf_counter()-tick;wall=time.perf_counter()-start
    report=dict(schema="scifact-rationale-population-v1",parent_population_sha256=pin,files=files,stop=width,cases=len(paths),documents=len(docs),
        max_alternatives=alternatives,max_decisions=length,sentence_windows=counts,semantics="OR-distinct-complete-paths/AND-every-sentence-plus-STOP",
        zero_rationale_has_no_auxiliary_target=True,scientific_admission=False,served=False,sealed_evaluation=False,assertion_verdict="REFUSE_UNADMITTED_PANEL",
        phases_seconds=phases,wall_seconds=wall,phase_coverage=sum(phases.values())/wall)
    publish(output/"report.json",report);print(json.dumps(dict(report_sha256=file_sha(output/"report.json"),wall_seconds=wall)),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for n in ("population","inventory","output"):p.add_argument("--"+n,type=Path,required=True)
    p.add_argument("--report-sha256",required=True);a=p.parse_args();project(a.population,a.report_sha256,a.inventory,a.output)
