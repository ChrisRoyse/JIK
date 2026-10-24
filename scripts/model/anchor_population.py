"""Source-bound A2 readout population; entire overlap components stay together.

Uses all already measured native windows. No feature pooling, token inputs,
new encoder execution, class resampling, or scientific sense admission.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit.anchor_lineage_readback import read as lineage_read
from text.evaluation_overlap import Features, prefix_join, components
from text.evaluation_multiscale import file_sha
from senses.contract import canonical
from senses.trajectory_population import publish

QUESTIONS=("biomedical/P1057","biomedical/P1995","science/P123")
SCHEMA="anchor-readout-population-v1"


def check(ok,code):
    if not ok:raise ValueError("MNLM_ANCHOR_READOUT_"+code)


def lines(path):return [json.loads(l) for l in path.read_bytes().splitlines()]


def split(group):
    n=int(hashlib.sha256(("anchor-readout-v1|25|"+group).encode()).hexdigest()[:8],16)%100
    return "train" if n<70 else "calibration" if n<85 else "evaluation"


def source_rows(lineage,calibration):
    cases=lines(lineage/"cases.jsonl");records={};chunks=[]
    root=json.loads((calibration/"report.json").read_bytes())
    for family in ("biomedical","science"):
        anchors=(calibration/family/"anchors.json").read_bytes()
        check(anchors==(lineage/"parents"/(family+"-anchors.json")).read_bytes(),"ANCHOR_JOIN")
        summary=json.loads((calibration/family/"summary.json").read_bytes())
        check(file_sha(calibration/family/"summary.json")==root["families"][family]["summary_sha256"]
              and file_hash(anchors)==summary["anchors_sha256"],"CALIBRATION_PIN")
        check(file_sha(calibration/family/"chunks/atoms.parquet")==summary["chunks_sha256"],"CHUNK_PIN")
        records[family]=lines(lineage/"parents"/(family+"-rows.jsonl"))
        chunks.extend([{**r,"family":family} for r in pq.read_table(calibration/family/"chunks/atoms.parquet").to_pylist()])
    by_row=defaultdict(list)
    for i,c in enumerate(chunks):by_row[c["family"],c["row"]].append(i)
    docs=[];features=Features()
    for case in cases:
        src=records[case["family"]][case["record_row"]];text=src["text"]
        ix=by_row[case["family"],case["record_row"]];part=[chunks[i] for i in ix]
        check(src["entity"]==case["entity"] and file_hash(text.encode())==case["text_sha256"],"TEXT_JOIN")
        cursor=0
        for c in part:
            check(c["start"]==cursor and text[c["start"]:c["end"]]==c["text"] and c["entity"]==case["entity"],"CHUNK_TILING")
            cursor=c["end"]
        check(cursor==len(text) and ''.join(c["text"] for c in part)==text,"CHUNK_COVERAGE")
        field=dict(name="text",text=text,sha256=case["text_sha256"],characters=len(text),utf8_bytes=len(text.encode()))
        features.add(dict(id=case["case"],domain=case["family"],group_keys=["entity:"+case["entity"]],fields=[field]))
        positive=[i for i,v in enumerate(case["classes"]) if v in case["values"]]
        check(bool(positive) and case["status"] in ("singleton","multiple"),"ANCHOR_AVAILABLE")
        docs.append({**case,"question":QUESTIONS.index(case["family"]+"/"+case["property"]),"positive_classes":positive,"chunks":ix})
    check(len(docs)==1512 and len(chunks)==6552,"FIXED_POPULATION")
    check(sorted(i for d in docs for i in d["chunks"])==list(range(len(chunks))),"CHUNK_OWNERSHIP")
    return docs,chunks,features


def file_hash(raw):return hashlib.sha256(raw).hexdigest()


def run(a):
    start=time.perf_counter();a.output.mkdir(parents=True,exist_ok=False);phases={}
    inv=json.loads(a.inventory.read_bytes())
    check(file_sha(a.lineage/"report.json")==inv["lineage_report_sha256"] and file_sha(a.calibration/"report.json")==inv["calibration_report_sha256"],"PARENT_PIN")
    witness=json.loads(a.original_replay.read_bytes());lineage_read(a.lineage,inv["lineage_report_sha256"],witness)
    docs,chunks,features=source_rows(a.lineage,a.calibration)
    (a.output/"parents").mkdir();shutil.copytree(a.lineage,a.output/"parents/lineage")
    shutil.copytree(a.calibration,a.output/"parents/calibration")
    shutil.copyfile(a.original_replay,a.output/"parents/original-replay.json")
    shutil.copyfile(a.inventory,a.output/"inventory.json")
    phases["source_pins_and_lossless_join"]=time.perf_counter()-start
    tick=time.perf_counter();candidates,near,stats=prefix_join(features.sets);links,groups=components(features.documents,near)
    for group in groups:
        for i in group["members"]:docs[i].update(group=group["id"],split=split(group["id"]))
    overlap=a.output/"overlap";overlap.mkdir();features.save(overlap)
    np.save(overlap/"candidates.npy",np.asarray(candidates,dtype=np.uint32).reshape(-1,2))
    publish(overlap/"links.json",links);publish(overlap/"groups.json",groups)
    (a.output/"rows.jsonl").write_bytes(b''.join(canonical(d)+b'\n' for d in docs))
    phases["complete_overlap_and_group_partition"]=time.perf_counter()-tick
    tick=time.perf_counter();lanes=[];(a.output/"lanes").mkdir()
    public=Path(inv["instruments"][0]["root"]).parent
    check(file_sha(public/"report.json")==inv["public_report_sha256"],"PUBLIC_PARENT")
    pub=json.loads((public/"report.json").read_bytes())
    check(pub["calibration_report_sha256"]==inv["calibration_report_sha256"] and pub["complete"] and not pub["errors"],"PUBLIC_CALIBRATION")
    shutil.copyfile(public/"report.json",a.output/"parents/public-measurement.json")
    for spec in inv["instruments"]:
        root=Path(spec["root"]);check(file_sha(root/"report.json")==spec["report_sha256"],"LENS_REPORT_PIN")
        r=json.loads((root/"report.json").read_bytes());dest=a.output/"lanes"/spec["name"];dest.mkdir()
        check(r["scientific_admission"] is False and r["served"] is False and r["inputs"]==len(chunks),"LENS_SCOPE")
        if "plan" in spec:
            plan=Path(spec["plan"]);check(file_sha(plan/"report.json")==spec["plan_sha256"]==r["plan_report_sha256"],"PLAN_PIN")
            pr=json.loads((plan/"report.json").read_bytes());check(pr["calibration_report_sha256"]==inv["calibration_report_sha256"],"PLAN_POPULATION")
            check(file_sha(root/"vectors.npy")==r["files"]["vectors.npy"]["sha256"] and file_sha(plan/"offsets.npy")==pr["files"]["offsets.npy"]["sha256"],"NATIVE_PIN")
            x=np.load(root/"vectors.npy",allow_pickle=False);ptr=np.load(plan/"offsets.npy",allow_pickle=False)
            for src,name in ((root/"vectors.npy","vectors.npy"),(plan/"offsets.npy","offsets.npy"),(plan/"report.json","plan.json")):
                shutil.copyfile(src,dest/name)
            storage="npy"
        else:
            check(pub["lenses"][spec["name"]]["report_sha256"]==spec["report_sha256"] and pub["lenses"][spec["name"]]["lens_id"]==r["lens_id"],"PUBLIC_LANE")
            check(file_sha(root/"slots.npz")==r["files"]["slots.npz"]["sha256"],"NATIVE_PIN")
            with np.load(root/"slots.npz",allow_pickle=False) as z:x=z["vectors"];ptr=z["offsets"]
            shutil.copyfile(root/"slots.npz",dest/"slots.npz");storage="npz"
        check(x.dtype==np.float32 and list(x.shape)==spec["shape"] and np.isfinite(x).all()
              and ptr.shape==(len(chunks)+1,) and ptr[0]==0 and ptr[-1]==len(x) and np.all(np.diff(ptr)>0),"NATIVE_SHAPE")
        check(np.all(np.linalg.norm(x,axis=1)>0),"ZERO_NATIVE")
        shutil.copyfile(root/"report.json",dest/"measurement.json")
        max_windows=max(sum(int(ptr[c+1]-ptr[c]) for c in d["chunks"]) for d in docs)
        lanes.append(dict(name=spec["name"],lens_id=r["lens_id"],storage=storage,dimension=x.shape[1],windows=len(x),max_document_windows=max_windows))
    phases["native_window_binding_and_copy"]=time.perf_counter()-tick
    tick=time.perf_counter();support={}
    for q in range(3):
        part=[d for d in docs if d["question"]==q];counts={s:sum(d["split"]==s for d in part) for s in ("train","calibration","evaluation")}
        train=[sum(d["split"]=="train" and j in d["positive_classes"] for d in part) for j in range(6)]
        support[QUESTIONS[q]]=dict(counts=counts,training_support=train,ready=min(train)>0 and counts["calibration"]>=50 and counts["evaluation"]>=50)
    files={str(p.relative_to(a.output)).replace('\\','/'):dict(bytes=p.stat().st_size,sha256=file_sha(p)) for p in a.output.rglob('*') if p.is_file()}
    phases["final_hashes_and_support"]=time.perf_counter()-tick;wall=time.perf_counter()-start
    report=dict(schema=SCHEMA,files=files,rows=len(docs),chunks=len(chunks),source_characters=inv["source_characters"],
                lanes=lanes,questions=list(QUESTIONS),support=support,groups=len(groups),largest_group=max(len(g["members"]) for g in groups),
                overlap_stats=stats,near_links=len(near),links=len(links),native_storage_unchanged=True,
                lineage_report_sha256=inv["lineage_report_sha256"],calibration_report_sha256=inv["calibration_report_sha256"],
                scientific_admission=False,sealed_evaluation=False,source_time_composition_separated=False,
                encoder_training_exposure_certified=False,served=False,code_sha256=file_sha(Path(__file__)),
                phases_seconds=phases,wall_seconds=wall,phase_coverage=sum(phases.values())/wall)
    publish(a.output/"report.json",report);print(json.dumps({k:report[k] for k in ("rows","chunks","groups","largest_group","support","near_links","wall_seconds","phase_coverage")}))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for n in ("lineage","calibration","inventory","original-replay","output"):p.add_argument("--"+n,type=Path,required=True)
    run(p.parse_args())
