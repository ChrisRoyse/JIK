"""Fresh-source finance calibration for the unchanged questions (#12).

Fixed text-hash roles, balanced quotas, exact-text decontamination, and full
source text. No instrument readings or scores participate in selection.
"""
import argparse
from collections import defaultdict
import hashlib
import io
import json
from pathlib import Path
import sys
import time

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from canon.common import iter_canon_docs
from text.chunk_spans import chunk_spans, CONTRACT
from senses.trajectory_population import publish

ROLES = {"design":504,"evaluation":1008}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def file_sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def check(ok, code):
    if not ok:
        raise ValueError("MNLM_FIXED_FINANCE_"+code)


def role(pin):
    return "design" if int(sha(("finance-fixed-v1|"+pin).encode()),16)%3==0 else "evaluation"


def run(a):
    start=time.perf_counter(); phases={}
    check(not a.output.exists(),"OUTPUT_EXISTS")
    manifest=json.loads((a.census/"manifest.json").read_bytes())
    report=json.loads((a.census/"report.json").read_bytes())
    check(file_sha(a.census/"report.json")==manifest["report_sha256"] and report["complete"],"CENSUS_PIN")
    proof=json.loads((a.census/"cold-v1/report.json").read_bytes())
    seal=json.loads((a.census/"cold-v1/manifest.json").read_bytes())
    check(file_sha(a.census/"cold-v1/report.json")==seal["report_sha256"] and proof["complete"] and
          proof["source_report_sha256"]==manifest["report_sha256"],"CENSUS_READBACK")
    contract=json.loads((a.census/"contract.json").read_bytes())
    check(file_sha(a.census/"contract.json")==report["config_sha256"],"CONTRACT")
    questions=contract["families"]["finance"]["questions"]
    path=a.census/"private/eligible.jsonl"
    check(file_sha(path)==manifest["eligible_sha256"]==report["eligible_sha256"],"CENSUS_ROWS")
    rows=[json.loads(line) for line in path.read_bytes().splitlines()]
    historical_ids=set(); historical_text=set()
    for family,spec in contract["families"].items():
        path=a.records/"records"/family/"rows.jsonl"
        check(file_sha(path)==spec["rows_sha256"],"HISTORICAL_PIN")
        for line in path.read_bytes().splitlines():
            r=json.loads(line);historical_ids.add(r["entity"])
            check(sha(r["text"].encode())==r["text_source"]["text_sha256"],"HISTORICAL_TEXT")
            historical_text.add(r["text_source"]["text_sha256"])
    groups=defaultdict(list)
    for r in rows:
        if r["family"]=="finance" and r["entity"] not in historical_ids and r["text_source"]["sha256"] not in historical_text:
            groups[r["text_source"]["sha256"]].append(r)
    selections={}; available={}; chosen={}; conflicts={}
    for prop,classes in questions.items():
        pool=defaultdict(list); conflicts[prop]=0
        for pin,members in groups.items():
            value_sets={tuple(r["values"].get(prop,[])) for r in members}
            if len(value_sets)!=1:
                conflicts[prop]+=1;continue
            values=next(iter(value_sets))
            if len(values)!=1 or values[0] not in classes:
                continue
            representative=min(members,key=lambda r:r["entity"])
            pool[role(pin),values[0]].append(representative)
        selections[prop]=[];available[prop]={}
        for partition,total in ROLES.items():
            check(total%len(classes)==0,"BALANCED_QUOTA")
            quota=total//len(classes)
            available[prop][partition]=[len(pool[partition,c]) for c in classes]
            check(all(len(pool[partition,c])>=quota for c in classes),"INSUFFICIENT_FIXED_CLASS_ROLE")
            for label,value in enumerate(classes):
                ordered=sorted(pool[partition,value],key=lambda r:sha(("finance-fixed-v1|order|"+r["text_source"]["sha256"]).encode()))
                for r in ordered[:quota]:
                    selections[prop].append(dict(entity=r["entity"],label=label,partition=partition))
                    chosen[r["entity"]]=r
    a.output.mkdir(parents=True)
    publish(a.output/"intent.json",dict(census_report_sha256=manifest["report_sha256"],
        census_readback_sha256=seal["report_sha256"],questions=questions,roles=ROLES,available=available,
        duplicate_label_conflicts=conflicts,scientific_admission=False,served=False))
    phases["binding_and_selection_seconds"]=time.perf_counter()-start;tick=time.perf_counter()
    check(file_sha(a.canon/"manifest.jsonl")==report["manifest_sha256"],"ORIGINAL_MANIFEST")
    source=report["source_manifest"]["wikidata5m_text"]; texts={};n=0
    for rec in iter_canon_docs(a.canon,"wikidata5m_text",source["out_blake3"]):
        n+=1;entity=rec["f"].get("entity_id")
        if entity not in chosen:
            continue
        expected=chosen[entity]["text_source"];raw=rec["text"].encode()
        check(entity not in texts and rec["row"]==expected["row"] and sha(raw)==expected["sha256"]
              and len(raw)==expected["utf8_bytes"],"SOURCE_TEXT")
        texts[entity]=rec["text"]
    check(n==source["docs"] and set(texts)==set(chosen),"SOURCE_COMPLETE")
    phases["original_prose_seconds"]=time.perf_counter()-tick;tick=time.perf_counter()
    ordered=sorted(chosen);index={entity:i for i,entity in enumerate(ordered)}
    dest=a.output/"finance"; (dest/"chunks").mkdir(parents=True)
    chunks=[];documents=[]
    for entity in ordered:
        record=chosen[entity];text=texts[entity];spans=chunk_spans(text)
        check(''.join(p["text"] for p in spans)==text,"LOSSLESS_CHUNKS")
        documents.append(dict(entity=entity,row=index[entity],text=text,text_source=record["text_source"],
                              partition=role(record["text_source"]["sha256"])))
        for j,span in enumerate(spans):
            chunks.append(dict(**span,row=index[entity],entity=entity,chunk=j,source_sha256=record["text_source"]["sha256"]))
    anchors={p:dict(classes=questions[p],rows=[index[r["entity"]] for r in selected],
                    labels=[r["label"] for r in selected],partitions=[r["partition"] for r in selected])
             for p,selected in selections.items()}
    publish(dest/"anchors.json",anchors)
    with (dest/"rows.jsonl").open("x",encoding="utf-8") as f:
        for r in documents:f.write(json.dumps(r,sort_keys=True)+"\n")
    pq.write_table(pa.Table.from_pylist(chunks),dest/"chunks/atoms.parquet",compression="zstd")
    # Physical readback of the consumer input, independent of the in-memory list.
    stored=pq.read_table(dest/"chunks/atoms.parquet").to_pylist()
    assembled=defaultdict(list)
    for r in stored:assembled[r["entity"]].append(r["text"])
    check(set(assembled)==set(chosen) and all(''.join(assembled[e])==texts[e] for e in chosen),"PERSISTED_CHUNKS")
    summary=dict(schema="fixed-finance-calibration-v1",family="finance",chunks=len(chunks),entities=len(chosen),
        chunks_sha256=file_sha(dest/"chunks/atoms.parquet"),anchors_sha256=file_sha(dest/"anchors.json"),
        rows_sha256=file_sha(dest/"rows.jsonl"),source_characters=sum(map(len,texts.values())),
        chunker=CONTRACT,available_questions=sorted(questions),calibration_only=True,scientific_admission=False,served=False,
        source_novelty="historical entity and exact-text exclusion; pretraining and near-duplicate independence not certified")
    publish(dest/"summary.json",summary)
    phases["publication_and_readback_seconds"]=time.perf_counter()-tick
    result=dict(schema="fixed-finance-calibration-publication-v1",complete=True,errors=[],calibration_only=True,
        scientific_admission=False,served=False,census_report_sha256=manifest["report_sha256"],
        intent_sha256=file_sha(a.output/"intent.json"),families=dict(finance=dict(summary_sha256=file_sha(dest/"summary.json"),
        entities=len(chosen),chunks=len(chunks),ready=sorted(questions),unavailable=[])),
        phases_seconds=phases,wall_seconds=time.perf_counter()-start,code_sha256=file_sha(Path(__file__)))
    publish(a.output/"report.json",result)
    publish(a.output/"manifest.json",dict(report_sha256=file_sha(a.output/"report.json")))
    print(json.dumps(result),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("census","records","canon","output"):
        p.add_argument("--"+name,type=Path,required=True)
    run(p.parse_args())
