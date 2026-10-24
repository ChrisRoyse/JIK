"""Independent exact source sentence/window and annotated-path reconstruction."""
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit.scifact_population_readback import reference,read as parent_read
from audit.scifact_readback import jsonl
from text.evaluation_multiscale import file_sha


def check(ok,code):
    if not ok:raise ValueError("MNLM_RATIONALE_READ_"+code)


def expected(ctx):
    docs=ctx["documents"];stop=max(d["sentences"] for d in docs);valid=np.zeros((len(docs),stop),bool);records=[];members=[]
    for i,d in enumerate(docs):
        valid[i,:d["sentences"]]=True;sentences=[]
        for index in range(d["sentences"]):
            choices=[v for v in d["views"] if v["level"]==1 and v["span"]==[index,index+1]]
            check(len(choices)==1,"SOURCE_VIEW");sentences.append(choices[0])
        records.append(dict(document=d["document"],sentences=sentences))
    for ptr,m in zip(ctx["pointers"],ctx["maps"]):
        member=np.zeros((len(docs),stop,m["indices"].shape[1]),bool)
        for i,d in enumerate(docs):
            sequence=[(v,n) for v in d["views"] for n in range(int(ptr[v["unit"]]),int(ptr[v["unit"]+1]))]
            check(len(sequence)==int(m["mask"][i].sum()),"SOURCE_WINDOW_COUNT")
            for column,(view,native) in enumerate(sequence):
                check(m["mask"][i,column] and m["indices"][i,column]==native,"NATIVE_WINDOW")
                if view["level"]==1:member[i,view["span"][0],column]=True
        members.append(member)
    paths=[]
    for row,t in zip(ctx["rows"],ctx["targets"]):
        p=[sorted(r["sentences"])+[stop] for r in t["rationales"]]
        check(len({tuple(v) for v in p})==len(p) and all(v[-1]==stop and len(v)==len(set(v)) for v in p),"PATH_SOURCE")
        paths.append(dict(case=row["id"],partition=row["partition"],alternatives=p))
    maximum=max(len(r["alternatives"]) for r in paths);length=max(len(p) for r in paths for p in r["alternatives"])
    choices=np.full((len(paths),maximum,length),stop,np.int64);steps=np.zeros_like(choices,bool);present=np.zeros(choices.shape[:2],bool)
    for i,r in enumerate(paths):
        for j,p in enumerate(r["alternatives"]):
            for t,v in enumerate(p):choices[i,j,t]=v;steps[i,j,t]=True
            present[i,j]=True
    return dict(valid=valid,records=records,members=members,paths=paths,targets=dict(choices=choices,steps=steps,present=present),stop=stop)


def read(root,pin,population,parent_pin,inventory,ctx=None,truth=None):
    ctx=reference(inventory) if ctx is None else ctx;parent_read(population,parent_pin,inventory,ctx);truth=expected(ctx) if truth is None else truth
    check(file_sha(root/"report.json")==pin,"PIN");r=json.loads((root/"report.json").read_bytes())
    check(r["schema"]=="scifact-rationale-population-v1" and r["parent_population_sha256"]==parent_pin,"PARENT")
    check(r["scientific_admission"] is False and r["served"] is False and r["sealed_evaluation"] is False
        and r["assertion_verdict"]=="REFUSE_UNADMITTED_PANEL" and r["zero_rationale_has_no_auxiliary_target"] is True,"CLAIM")
    check(r["semantics"]=="OR-distinct-complete-paths/AND-every-sentence-plus-STOP","SEMANTICS")
    names={"valid-sentences.npy","sentences.jsonl","paths.jsonl","targets.npz"}|{"membership/"+l["name"]+".npz" for l in ctx["lanes"]}
    check(set(r["files"])==names and {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}==names|{"report.json"},"FILES")
    for name,v in r["files"].items():check(file_sha(root/name)==v["sha256"] and (root/name).stat().st_size==v["bytes"],"FILE_HASH")
    valid=np.load(root/"valid-sentences.npy",allow_pickle=False);check(valid.dtype==bool and np.array_equal(valid,truth["valid"]),"VALID")
    check(jsonl(root/"sentences.jsonl")==truth["records"],"SENTENCES");check(jsonl(root/"paths.jsonl")==truth["paths"],"PATHS")
    counts={}
    for lane,m in zip(ctx["lanes"],truth["members"]):
        with np.load(root/"membership"/(lane["name"]+".npz"),allow_pickle=False) as z:
            check(z.files==["membership"] and z["membership"].dtype==bool and np.array_equal(z["membership"],m),"MEMBERSHIP")
        counts[lane["name"]]=int(m.sum())
    with np.load(root/"targets.npz",allow_pickle=False) as z:
        check(set(z.files)==set(truth["targets"]),"TARGET_KEYS")
        for name,v in truth["targets"].items():check(z[name].dtype==v.dtype and np.array_equal(z[name],v),"TARGET_"+name.upper())
    check((r["stop"],r["cases"],r["documents"],r["max_alternatives"],r["max_decisions"])
        ==(truth["stop"],len(ctx["rows"]),len(ctx["documents"]),truth["targets"]["choices"].shape[1],truth["targets"]["choices"].shape[2]) and r["sentence_windows"]==counts,"COUNTS")
    return dict(report_sha256=pin,parent_population_sha256=parent_pin,cases=r["cases"],documents=r["documents"],stop=r["stop"],sentence_windows=counts,
        alternatives=int(truth["targets"]["present"].sum()),annotated_cases=int(truth["targets"]["present"].any(1).sum()),
        complete_source_window_and_alternative_sets_equal=True,assertion_verdict="REFUSE_UNADMITTED_PANEL",scientific_admission=False)
