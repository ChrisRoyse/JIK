"""Portable relation cases and typed gather maps over immutable native readings."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time

import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit.scifact_overlap_readback import read as overlap_read
from audit.scifact_readback import jsonl
from senses.contract import canonical
from senses.trajectory_population import publish
from text.evaluation_multiscale import file_sha

SCHEMA="scifact-readout-population-v1"
LEVELS=("title","source_sentence","sentence_pair","abstract_context")
LABELS=("NOT_ENOUGH_EVIDENCE","CONTRADICT","SUPPORT")


def check(ok,code):
    if not ok:raise ValueError("MNLM_RELATION_"+code)


def pinned_files(root,report):
    for name,info in report["files"].items():
        path=root/name
        check(path.stat().st_size==info["bytes"] and file_sha(path)==info["sha256"],"PARENT_FILE")


def parents(inventory):
    inv=json.loads(inventory.read_bytes());check(inv["schema"]=="scifact-readout-inventory-v1","INVENTORY")
    paths={n:Path(inv[n]["path"]) for n in ("source","publication","inputs","overlap")}
    pins=[inv[n]["sha256"] for n in ("source","publication","inputs")]
    overlap_read(paths["source"],paths["publication"],paths["inputs"],pins,paths["overlap"],inv["overlap"]["sha256"])
    public=Path(inv["public"]["path"]);check(file_sha(public/"report.json")==inv["public"]["sha256"],"PUBLIC_PIN")
    pr=json.loads((public/"report.json").read_bytes())
    check(pr["complete"] and not pr["errors"] and pr["calibration_report_sha256"]==pins[2],"PUBLIC_SOURCE")
    check(len(inv["lanes"])==14 and len({v["name"] for v in inv["lanes"]})==14,"ROSTER")
    check([v["name"] for v in inv["lanes"][:6]]==sorted(pr["lenses"]),"ROSTER_ORDER")
    binding={n:inv[n]["sha256"] for n in ("source","publication","inputs","overlap","public")}
    binding["lanes"]=[{k:v for k,v in lane.items() if k not in ("path","plan")} for lane in inv["lanes"]]
    return inv,paths,binding


def native(inv,lane):
    root=Path(lane["path"]);check(file_sha(root/"report.json")==lane["sha256"],"NATIVE_PIN")
    r=json.loads((root/"report.json").read_bytes());pinned_files(root,r)
    check(r["inputs"]==11406 and r["scientific_admission"] is False and r["served"] is False,"NATIVE_SCOPE")
    if lane["kind"]=="public":
        p=json.loads((Path(inv["public"]["path"])/"report.json").read_bytes())
        check(p["lenses"][lane["name"]]["report_sha256"]==lane["sha256"],"PUBLIC_LANE")
        with np.load(root/"slots.npz",allow_pickle=False) as z:x,ptr=z["vectors"].copy(),z["offsets"].copy()
    else:
        check(lane["kind"]=="commissioned","KIND");plan=Path(lane["plan"])
        check(file_sha(plan/"report.json")==lane["plan_sha256"]==r["plan_report_sha256"],"PLAN_PIN")
        p=json.loads((plan/"report.json").read_bytes());pinned_files(plan,p)
        check(p["calibration_report_sha256"]==inv["inputs"]["sha256"] and p["inputs"]==11406,"PLAN_SOURCE")
        x=np.load(root/"vectors.npy",allow_pickle=False);ptr=np.load(plan/"offsets.npy",allow_pickle=False)
    check(x.dtype==np.float32 and x.ndim==2 and np.isfinite(x).all() and len(x)==r["windows"]
          and np.allclose(np.linalg.norm(x,axis=1),1,rtol=0,atol=3e-6),"NATIVE_VECTORS")
    check(ptr.dtype==np.int64 and ptr.shape==(11407,) and ptr[0]==0 and ptr[-1]==len(x) and np.all(np.diff(ptr)>0),"NATIVE_POINTERS")
    check(r["contract"]["shape"]==f"MultiWindows({x.shape[1]})" and r["contract"]["window_reduction"]=="none:ordered-window-vectors","NATIVE_TYPE")
    return x,ptr,r


def records(paths):
    refs=jsonl(paths["inputs"]/"references.jsonl");by_doc=defaultdict(list)
    queries={r["owner"]:r for r in refs if r["kind"]=="claim"}
    for r in refs:
        if r["kind"]!="claim":by_doc[r["owner"]].append(r)
    docs=[];indices={}
    for d in jsonl(paths["publication"]/"documents.jsonl"):
        indices[d["id"]]=len(docs)
        docs.append(dict(document=d["id"],sentences=len(d["views"]["source_sentences"]),views=[dict(
            ref=r["id"],unit=r["unit"],level=LEVELS.index(r["kind"]),span=r["source_sentence_range"]) for r in by_doc[d["id"]]]))
    rows=[{**c,"query_ref":queries[c["claim"]]["id"],"query_unit":queries[c["claim"]]["unit"],"document_index":indices[c["document"]]}
          for c in jsonl(paths["overlap"]/"cases.jsonl")]
    labels=jsonl(paths["publication"]/"labels.jsonl")
    check([r["id"] for r in rows]==[r["case"] for r in labels] and all(r["label"] in LABELS for r in labels),"TARGET_ORDER")
    return rows,docs,labels


def gather_maps(ptr,rows,docs):
    q=np.asarray([ptr[r["query_unit"]] for r in rows],np.int64)
    check(all(ptr[r["query_unit"]+1]-ptr[r["query_unit"]]==1 for r in rows),"QUERY_OVERFLOW_REQUIRES_NEW_SCHEMA")
    sizes=[sum(int(ptr[v["unit"]+1]-ptr[v["unit"]]) for v in d["views"]) for d in docs]
    indices=np.zeros((len(docs),max(sizes)),np.int64);mask=np.zeros_like(indices,dtype=bool)
    levels=np.zeros_like(indices);geometry=np.zeros((*indices.shape,3),np.float32)
    for i,d in enumerate(docs):
        cursor=0
        for v in d["views"]:
            lo,hi=map(int,ptr[v["unit"]:v["unit"]+2]);n=hi-lo
            span=v["span"] or [0,0]
            for j in range(n):
                indices[i,cursor]=lo+j;mask[i,cursor]=True;levels[i,cursor]=v["level"]
                geometry[i,cursor]=[span[0]/d["sentences"],(span[1]-span[0])/d["sentences"],j/n];cursor+=1
        check(cursor==sizes[i],"WINDOW_CONSERVATION")
    return dict(query=q,indices=indices,mask=mask,levels=levels,geometry=geometry)


def project(inventory,output):
    start=time.perf_counter();output.mkdir(parents=True,exist_ok=False);phases={}
    inv,paths,binding=parents(inventory);rows,docs,labels=records(paths)
    for name,data in (("rows",rows),("documents",docs),("targets",labels)):
        with (output/(name+".jsonl")).open("xb") as f:
            for r in data:f.write(canonical(r)+b"\n")
    phases["source_and_target_binding"]=time.perf_counter()-start;tick=time.perf_counter();lanes=[];(output/"maps").mkdir()
    for lane in inv["lanes"]:
        x,ptr,r=native(inv,lane);maps=gather_maps(ptr,rows,docs);np.savez(output/"maps"/(lane["name"]+".npz"),**maps)
        lanes.append(dict(name=lane["name"],lens_id=r["lens_id"],dimension=x.shape[1],windows=len(x),
            document_windows=int(maps["mask"].sum()),max_document_windows=maps["mask"].shape[1]))
        del x
    phases["native_pins_and_complete_gather_maps"]=time.perf_counter()-tick;tick=time.perf_counter()
    files={p.relative_to(output).as_posix():dict(bytes=p.stat().st_size,sha256=file_sha(p)) for p in output.rglob('*') if p.is_file()}
    phases["final_file_hashes"]=time.perf_counter()-tick;wall=time.perf_counter()-start
    report=dict(schema=SCHEMA,parents=binding,files=files,cases=len(rows),documents=len(docs),lanes=lanes,labels=list(LABELS),levels=list(LEVELS),
        native_text_floor=5,admitted_senses=0,assertion_verdict="REFUSE_UNADMITTED_PANEL",scientific_admission=False,served=False,
        sealed_evaluation=False,phases_seconds=phases,wall_seconds=wall,phase_coverage=sum(phases.values())/wall,code_sha256=file_sha(Path(__file__)))
    publish(output/"report.json",report);return report


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--inventory",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();r=project(a.inventory,a.output);print(json.dumps({k:r[k] for k in ("cases","documents","lanes","wall_seconds","phase_coverage")}))
