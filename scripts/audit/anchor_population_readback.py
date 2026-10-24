"""Independent source grouping, complete chunk and native-window readback."""
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit.anchor_lineage_readback import read as lineage_read
from audit.evaluation_overlap_readback import reference_candidates,reference_groups
from text.evaluation_multiscale import file_sha
from senses.contract import canonical


def check(ok,code):
    if not ok:raise ValueError("MNLM_ANCHOR_POP_READ_"+code)


def lines(path):return [json.loads(l) for l in path.read_bytes().splitlines()]


def arrays(root,lane):
    path=root/"lanes"/lane["name"]
    if lane["storage"]=="npz":
        with np.load(path/"slots.npz",allow_pickle=False) as z:return z["vectors"],z["offsets"]
    return np.load(path/"vectors.npy",allow_pickle=False),np.load(path/"offsets.npy",allow_pickle=False)


def read(root,pin):
    check(file_sha(root/"report.json")==pin,"REPORT_PIN");rep=json.loads((root/"report.json").read_bytes())
    check(rep["schema"]=="anchor-readout-population-v1" and rep["native_storage_unchanged"] is True
          and all(rep[k] is False for k in ("scientific_admission","sealed_evaluation","source_time_composition_separated","encoder_training_exposure_certified","served")),"CLAIM")
    actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    check(actual==set(rep["files"])|{"report.json"},"FILE_SET")
    for name,f in rep["files"].items():
        check((root/name).is_file() and (root/name).stat().st_size==f["bytes"] and file_sha(root/name)==f["sha256"],"FILE_PIN")
    lineage=root/"parents/lineage";cal=root/"parents/calibration"
    lineage_read(lineage,rep["lineage_report_sha256"],json.loads((root/"parents/original-replay.json").read_bytes()))
    inv=json.loads((root/"inventory.json").read_bytes());calrep=json.loads((cal/"report.json").read_bytes())
    check(file_sha(cal/"report.json")==rep["calibration_report_sha256"]==inv["calibration_report_sha256"],"CALIBRATION")
    rows=lines(root/"rows.jsonl");cases=lines(lineage/"cases.jsonl");chunks=[];source={}
    for family in ("biomedical","science"):
        check(file_sha(cal/family/"summary.json")==calrep["families"][family]["summary_sha256"],"CAL_SUMMARY")
        summary=json.loads((cal/family/"summary.json").read_bytes())
        check(file_sha(cal/family/"anchors.json")==summary["anchors_sha256"]==file_sha(lineage/"parents"/(family+"-anchors.json")),"ANCHOR_JOIN")
        check(file_sha(cal/family/"chunks/atoms.parquet")==summary["chunks_sha256"],"CHUNK_PIN")
        source[family]=lines(lineage/"parents"/(family+"-rows.jsonl"))
        chunks.extend([{**c,"family":family} for c in pq.read_table(cal/family/"chunks/atoms.parquet").to_pylist()])
    check(len(rows)==len(cases)==rep["rows"]==1512 and len(chunks)==rep["chunks"]==6552,"COUNT")
    questions=["biomedical/P1057","biomedical/P1995","science/P123"];check(rep["questions"]==questions,"QUESTIONS")
    docs=[];sets=[];actual_parts=defaultdict(list);all_used=[];characters=0
    for i,c in enumerate(chunks):actual_parts[c["family"],c["row"]].append(i)
    vocabulary=json.loads((root/"overlap/words.json").read_bytes());word_index={w:i for i,w in enumerate(vocabulary)}
    tuples=np.load(root/"overlap/shingles.npy",allow_pickle=False);index={tuple(map(int,s)):i for i,s in enumerate(tuples)}
    check(len(index)==len(tuples) and len(word_index)==len(vocabulary),"FEATURE_UNIQUENESS")
    ptr=np.load(root/"overlap/indptr.npy");flat=np.load(root/"overlap/members.npy")
    for i,(row,case) in enumerate(zip(rows,cases)):
        check(all(row[k]==v for k,v in case.items()),"CASE")
        text=source[case["family"]][case["record_row"]]["text"]
        characters+=len(text)
        check(hashlib.sha256(text.encode()).hexdigest()==case["text_sha256"],"TEXT")
        chunk_ids=actual_parts[case["family"],case["record_row"]];check(row["chunks"]==chunk_ids,"CHUNK_OWNERSHIP")
        start=0
        for j in chunk_ids:
            c=chunks[j];check(c["entity"]==case["entity"] and c["source_sha256"]==case["text_sha256"]
                             and c["start"]==start and c["text"]==text[start:c["end"]],"CHUNK_TILING");start=c["end"]
        check(start==len(text),"CHUNK_COVERAGE");all_used.extend(chunk_ids)
        check(row["question"]==questions.index(case["family"]+"/"+case["property"])
              and row["positive_classes"]==[j for j,v in enumerate(case["classes"]) if v in case["values"]],"TARGET_SET")
        words=re.findall(r"\w+",text.lower());tokens=[word_index[w] for w in words]
        features={index[tuple(tokens[j:j+5])] for j in range(len(tokens)-4)};sets.append(features)
        check(features==set(map(int,flat[ptr[i]:ptr[i+1]])),"FEATURE_CONTENT")
        docs.append(dict(id=case["case"],domain=case["family"],group_keys=["entity:"+case["entity"]],
                         fields=[dict(name="text",sha256=case["text_sha256"],characters=len(text),words=len(words))],
                         exact_sha256=hashlib.sha256(canonical([text])).hexdigest()))
    check(sorted(all_used)==list(range(6552)),"WINDOW_OWNERSHIP")
    check(characters==rep["source_characters"]==inv["source_characters"],"SOURCE_SIZE")
    check(docs==lines(root/"overlap/documents.jsonl"),"DOCUMENT_FEATURES")
    pairs,stats=reference_candidates(sets)
    check(pairs==[tuple(map(int,p)) for p in np.load(root/"overlap/candidates.npy")],"CANDIDATES")
    near=[]
    for i,j in pairs:
        intersection=len(sets[i]&sets[j]);union=len(sets[i]|sets[j])
        if 5*intersection>=3*union:near.append(dict(left=i,right=j,kind="near_exact_jaccard",intersection=intersection,union=union))
    links,groups=reference_groups(docs,near)
    check(links==json.loads((root/"overlap/links.json").read_bytes()) and groups==json.loads((root/"overlap/groups.json").read_bytes()),"GROUPS")
    for g in groups:
        bucket=int(hashlib.sha256(("anchor-readout-v1|25|"+g["id"]).encode()).hexdigest()[:8],16)%100
        split="train" if bucket<70 else "calibration" if bucket<85 else "evaluation"
        check(all(rows[i]["group"]==g["id"] and rows[i]["split"]==split for i in g["members"]),"SPLIT_LEAKAGE")
    pub=json.loads((root/"parents/public-measurement.json").read_bytes())
    check(file_sha(root/"parents/public-measurement.json")==inv["public_report_sha256"]
          and pub["calibration_report_sha256"]==rep["calibration_report_sha256"],"PUBLIC_BINDING")
    check(len(rep["lanes"])==len(inv["instruments"])==10,"LANES")
    for lane,spec in zip(rep["lanes"],inv["instruments"]):
        path=root/"lanes"/lane["name"];check(lane["name"]==spec["name"] and file_sha(path/"measurement.json")==spec["report_sha256"],"LENS_PIN")
        r=json.loads((path/"measurement.json").read_bytes());check(lane["lens_id"]==r["lens_id"],"LENS_ID")
        check(r["scientific_admission"] is False and r["served"] is False and r["inputs"]==6552,"LENS_SCOPE")
        if lane["storage"]=="npz":
            check(pub["lenses"][lane["name"]]["report_sha256"]==spec["report_sha256"]
                  and file_sha(path/"slots.npz")==r["files"]["slots.npz"]["sha256"],"NATIVE_PIN")
        else:
            check(file_sha(path/"plan.json")==spec["plan_sha256"]==r["plan_report_sha256"],"PLAN_PIN")
            plan=json.loads((path/"plan.json").read_bytes())
            check(plan["calibration_report_sha256"]==rep["calibration_report_sha256"]
                  and file_sha(path/"offsets.npy")==plan["files"]["offsets.npy"]["sha256"]
                  and file_sha(path/"vectors.npy")==r["files"]["vectors.npy"]["sha256"],"NATIVE_PIN")
        x,offsets=arrays(root,lane)
        check(x.dtype==np.float32 and list(x.shape)==spec["shape"] and np.isfinite(x).all() and np.all(np.linalg.norm(x,axis=1)>0),"NATIVE_VALUES")
        check(offsets.shape==(6553,) and offsets[0]==0 and offsets[-1]==len(x) and np.all(np.diff(offsets)>0),"NATIVE_OFFSETS")
        check(lane["dimension"]==x.shape[1] and lane["windows"]==len(x) and lane["max_document_windows"]==max(sum(int(offsets[j+1]-offsets[j]) for j in d["chunks"]) for d in rows),"NATIVE_COUNT")
    support={}
    for q,name in enumerate(questions):
        part=[r for r in rows if r["question"]==q];counts={s:sum(r["split"]==s for r in part) for s in ("train","calibration","evaluation")}
        train=[sum(r["split"]=="train" and j in r["positive_classes"] for r in part) for j in range(6)]
        support[name]=dict(counts=counts,training_support=train,ready=min(train)>0 and counts["calibration"]>=50 and counts["evaluation"]>=50)
    check(rep["support"]==support and rep["groups"]==len(groups) and rep["near_links"]==len(near) and rep["links"]==len(links)
          and rep["largest_group"]==max(len(g["members"]) for g in groups),"SUMMARY")
    return dict(author_checked=True,report_sha256=pin,rows=len(rows),groups=len(groups),near_links=len(near),support=support,
                native_windows=sum(l["windows"] for l in rep["lanes"]),all_windows_preserved=True,
                scientific_admission=False,sealed_evaluation=False,source_time_composition_separated=False)
