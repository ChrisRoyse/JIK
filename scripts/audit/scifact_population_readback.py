"""Independent native/source reconstruction, reusable within one immutable audit."""
import json
from pathlib import Path
import sys

import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit.scifact_overlap_readback import read as overlap_read
from audit.scifact_readback import jsonl
from text.evaluation_multiscale import file_sha


def check(ok,code):
    if not ok:raise ValueError("MNLM_RELATION_READBACK_"+code)


def reference(inventory):
    inv=json.loads(inventory.read_bytes());check(inv["schema"]=="scifact-readout-inventory-v1","INVENTORY")
    roots={k:Path(inv[k]["path"]) for k in ("source","publication","inputs","overlap","public")};pins={k:inv[k]["sha256"] for k in roots}
    overlap_read(roots["source"],roots["publication"],roots["inputs"],[pins[k] for k in ("source","publication","inputs")],roots["overlap"],pins["overlap"])
    sources={}
    def parent(root,pin):
        check(file_sha(root/"report.json")==pin,"PARENT_PIN");sources[str(root/"report.json")]=pin
        r=json.loads((root/"report.json").read_bytes())
        for name,v in r.get("files",{}).items():
            check(file_sha(root/name)==v["sha256"] and (root/name).stat().st_size==v["bytes"],"PARENT_FILE")
            sources[str(root/name)]=v["sha256"]
        return r
    for k in ("source","publication","inputs","overlap"):parent(roots[k],pins[k])
    public=parent(roots["public"],pins["public"])
    check(public["complete"] and not public["errors"] and public["calibration_report_sha256"]==pins["inputs"],"PUBLIC_SOURCE")
    lanes=inv["lanes"];check(len(lanes)==14 and len({l["name"] for l in lanes})==14 and [l["name"] for l in lanes[:6]]==sorted(public["lenses"]),"ROSTER")
    native=[];pointers=[];lane_meta=[]
    for i,l in enumerate(lanes):
        root=Path(l["path"]);r=parent(root,l["sha256"])
        check(r["inputs"]==11406 and r["scientific_admission"] is False and r["served"] is False,"NATIVE_SCOPE")
        if i<6:
            check(l["kind"]=="public" and public["lenses"][l["name"]]["report_sha256"]==l["sha256"],"PUBLIC_LANE")
            with np.load(root/"slots.npz",allow_pickle=False) as z:x,offset=z["vectors"].copy(),z["offsets"].copy()
        else:
            check(l["kind"]=="commissioned","KIND");plan=Path(l["plan"]);p=parent(plan,l["plan_sha256"])
            check(r["plan_report_sha256"]==l["plan_sha256"] and p["calibration_report_sha256"]==pins["inputs"],"PLAN_SOURCE")
            x=np.load(root/"vectors.npy",allow_pickle=False);offset=np.load(plan/"offsets.npy",allow_pickle=False)
        check(x.dtype==np.float32 and x.ndim==2 and np.isfinite(x).all() and len(x)==r["windows"]
              and np.allclose(np.linalg.norm(x,axis=1),1,rtol=0,atol=3e-6),"NATIVE_VECTORS")
        check(offset.dtype==np.int64 and offset.shape==(11407,) and offset[0]==0 and offset[-1]==len(x) and np.all(np.diff(offset)>0),"NATIVE_POINTERS")
        check(r["contract"]["shape"]==f"MultiWindows({x.shape[1]})" and r["contract"]["window_reduction"]=="none:ordered-window-vectors","NATIVE_TYPE")
        native.append(x);pointers.append(offset);lane_meta.append(dict(name=l["name"],lens_id=r["lens_id"],dimension=x.shape[1],windows=len(x)))
    refs=jsonl(roots["inputs"]/"references.jsonl");claims={r["owner"]:r for r in refs if r["kind"]=="claim"}
    groups={}
    for r in refs:
        if r["kind"]!="claim":groups.setdefault(r["owner"],[]).append(r)
    docs=[];lookup={};levels=["title","source_sentence","sentence_pair","abstract_context"]
    for original in jsonl(roots["publication"]/"documents.jsonl"):
        lookup[original["id"]]=len(docs)
        docs.append(dict(document=original["id"],sentences=len(original["views"]["source_sentences"]),views=[dict(
            ref=r["id"],unit=r["unit"],level=levels.index(r["kind"]),span=r["source_sentence_range"]) for r in groups[original["id"]]]))
    rows=[]
    for original in jsonl(roots["overlap"]/"cases.jsonl"):
        q=claims[original["claim"]]
        rows.append({**original,"query_ref":q["id"],"query_unit":q["unit"],"document_index":lookup[original["document"]]})
    targets=jsonl(roots["publication"]/"labels.jsonl");maps=[]
    for k,ptr in enumerate(pointers):
        query=np.array([ptr[r["query_unit"]] for r in rows],np.int64)
        check(all(ptr[r["query_unit"]+1]-ptr[r["query_unit"]]==1 for r in rows),"QUERY_SCOPE")
        bags=[]
        for doc in docs:
            bag=[]
            for view in doc["views"]:
                begin,end=ptr[view["unit"]:view["unit"]+2];span=view["span"] if view["span"] is not None else [0,0]
                bag.extend((int(j),view["level"],(span[0]/doc["sentences"],(span[1]-span[0])/doc["sentences"],(j-begin)/(end-begin))) for j in range(begin,end))
            bags.append(bag)
        width=max(map(len,bags));indices=np.zeros((len(bags),width),np.int64);level=np.zeros_like(indices)
        mask=np.zeros(indices.shape,bool);geo=np.zeros((*indices.shape,3),np.float32)
        for i,bag in enumerate(bags):
            for j,(index,typ,pos) in enumerate(bag):indices[i,j]=index;level[i,j]=typ;mask[i,j]=True;geo[i,j]=pos
        maps.append(dict(query=query,indices=indices,mask=mask,levels=level,geometry=geo))
        lane_meta[k].update(document_windows=sum(map(len,bags)),max_document_windows=width)
    binding={**pins,"lanes":[{k:v for k,v in lane.items() if k not in ("path","plan")} for lane in lanes]}
    return dict(inventory_sha256=file_sha(inventory),parents=binding,rows=rows,documents=docs,targets=targets,maps=maps,native=native,pointers=pointers,lanes=lane_meta,sources=sources)


def read(root,pin,inventory,context=None):
    ctx=reference(inventory) if context is None else context
    check(file_sha(inventory)==ctx["inventory_sha256"],"INVENTORY_DRIFT")
    check(file_sha(root/"report.json")==pin,"PIN");r=json.loads((root/"report.json").read_bytes())
    check(r["schema"]=="scifact-readout-population-v1" and r["parents"]==ctx["parents"],"PARENTS")
    check(r["native_text_floor"]==5 and r["admitted_senses"]==0 and r["assertion_verdict"]=="REFUSE_UNADMITTED_PANEL"
          and all(r[k] is False for k in ("scientific_admission","served","sealed_evaluation")),"CLAIM")
    check(r["levels"]==["title","source_sentence","sentence_pair","abstract_context"]
          and r["labels"]==["NOT_ENOUGH_EVIDENCE","CONTRADICT","SUPPORT"],"SCHEMA")
    names={"rows.jsonl","documents.jsonl","targets.jsonl"}|{"maps/"+l["name"]+".npz" for l in ctx["lanes"]}
    check(set(r["files"])==names and {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}==names|{"report.json"},"FILES")
    for name,v in r["files"].items():check(file_sha(root/name)==v["sha256"] and (root/name).stat().st_size==v["bytes"],"FILE_HASH")
    for name in ("rows","documents","targets"):check(jsonl(root/(name+".jsonl"))==ctx[name],name.upper())
    for meta,expected in zip(ctx["lanes"],ctx["maps"]):
        with np.load(root/"maps"/(meta["name"]+".npz"),allow_pickle=False) as z:
            check(set(z.files)==set(expected),"MAP_KEYS")
            for name,value in expected.items():check(z[name].dtype==value.dtype and np.array_equal(z[name],value),"MAP_"+name.upper())
    check(r["cases"]==len(ctx["rows"]) and r["documents"]==len(ctx["documents"]) and r["lanes"]==ctx["lanes"],"COUNTS")
    return dict(report_sha256=pin,cases=r["cases"],documents=r["documents"],lanes=r["lanes"],all_source_targets_and_native_maps_equal=True,
                assertion_verdict=r["assertion_verdict"],scientific_admission=False)
