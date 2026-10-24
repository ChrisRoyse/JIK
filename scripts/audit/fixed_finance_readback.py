"""Cold source/role/label/chunk checks for the fixed finance population."""
import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import json
from pathlib import Path
import time

import pyarrow.parquet as pq


def check(ok, code):
    if not ok:
        raise ValueError("MNLM_FINANCE_READBACK_"+code)


def sha(path):
    with path.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()


def roles_and_labels(anchors,documents,oracle,questions):
    for prop,record in anchors.items():
        check(record["classes"]==questions[prop],"CLASSES")
        check(len(record["rows"])==len(record["labels"])==len(record["partitions"])==1512,"ROSTER")
        check(len(set(record["rows"]))==1512,"DUPLICATE_ENTITY")
        support=Counter()
        for index,label,partition in zip(record["rows"],record["labels"],record["partitions"]):
            doc=documents[index];source=oracle[doc["entity"]]
            check(source["values"].get(prop)==[record["classes"][label]],"SOURCE_LABEL")
            check(partition==doc["partition"],"ROLE")
            support[partition,label]+=1
        for partition,total in (("design",504),("evaluation",1008)):
            check(all(support[partition,i]==total//len(record["classes"]) for i in range(len(record["classes"]))),"QUOTA")


def chunks_match(chunks,documents):
    grouped=defaultdict(list)
    for chunk in chunks:grouped[chunk["row"]].append(chunk)
    check(set(grouped)==set(range(len(documents))),"CHUNK_ROSTER")
    for i,doc in enumerate(documents):
        cursor=0
        for j,chunk in enumerate(grouped[i]):
            check(chunk["entity"]==doc["entity"] and chunk["chunk"]==j and chunk["start"]==cursor and
                  chunk["end"]>cursor and doc["text"][cursor:chunk["end"]]==chunk["text"] and
                  chunk["source_sha256"]==doc["text_source"]["sha256"],"CHUNK_SPAN")
            cursor=chunk["end"]
        check(cursor==len(doc["text"]),"CHUNK_TAIL")


def run(a):
    start=time.perf_counter();a.output.mkdir(parents=True,exist_ok=False)
    root=json.loads((a.population/"report.json").read_bytes())
    publication=json.loads((a.population/"manifest.json").read_bytes())
    check(sha(a.population/"report.json")==publication["report_sha256"] and root["complete"] and
          root["scientific_admission"] is False and root["served"] is False,"PUBLICATION")
    intent=json.loads((a.population/"intent.json").read_bytes())
    check(sha(a.population/"intent.json")==root["intent_sha256"],"INTENT")
    census=json.loads((a.census/"report.json").read_bytes())
    check(sha(a.census/"report.json")==root["census_report_sha256"] and
          sha(a.census/"cold-v1/report.json")==intent["census_readback_sha256"],"CENSUS_PIN")
    check(sha(a.census/"private/eligible.jsonl")==census["eligible_sha256"],"ORACLE_PIN")
    oracle={r["entity"]:r for r in map(json.loads,(a.census/"private/eligible.jsonl").read_bytes().splitlines())}
    contract=json.loads((a.census/"contract.json").read_bytes())
    check(sha(a.census/"contract.json")==census["config_sha256"],"CONTRACT_PIN")
    questions=contract["families"]["finance"]["questions"]
    historical_ids=set();historical_text=set()
    for family,spec in contract["families"].items():
        path=a.records/"records"/family/"rows.jsonl";check(sha(path)==spec["rows_sha256"],"HISTORY_PIN")
        for line in path.read_bytes().splitlines():
            row=json.loads(line);historical_ids.add(row["entity"]);historical_text.add(hashlib.sha256(row["text"].encode()).hexdigest())
    dest=a.population/"finance";summary=json.loads((dest/"summary.json").read_bytes())
    check(sha(dest/"summary.json")==root["families"]["finance"]["summary_sha256"],"SUMMARY_PIN")
    for name,key in (("rows.jsonl","rows_sha256"),("anchors.json","anchors_sha256"),("chunks/atoms.parquet","chunks_sha256")):
        check(sha(dest/name)==summary[key],"DATA_PIN")
    documents=[json.loads(line) for line in (dest/"rows.jsonl").read_bytes().splitlines()]
    check(len(documents)==summary["entities"]==len({r["entity"] for r in documents}),"DOCUMENTS")
    text_pins=set()
    for i,doc in enumerate(documents):
        pin=hashlib.sha256(doc["text"].encode()).hexdigest();source=oracle[doc["entity"]]
        check(doc["row"]==i and doc["text_source"]==source["text_source"] and pin==source["text_source"]["sha256"],"SOURCE_TEXT")
        check(doc["entity"] not in historical_ids and pin not in historical_text and pin not in text_pins,"DECONTAMINATION")
        text_pins.add(pin)
        partition="design" if int(hashlib.sha256(("finance-fixed-v1|"+pin).encode()).hexdigest(),16)%3==0 else "evaluation"
        check(doc["partition"]==partition,"HASH_ROLE")
    anchors=json.loads((dest/"anchors.json").read_bytes());check(set(anchors)==set(questions),"QUESTIONS")
    roles_and_labels(anchors,documents,oracle,questions)
    chunks=pq.read_table(dest/"chunks/atoms.parquet").to_pylist();check(len(chunks)==summary["chunks"],"CHUNKS")
    chunks_match(chunks,documents)
    bad=copy.deepcopy(anchors);key=next(iter(bad));bad[key]["labels"][0]=(bad[key]["labels"][0]+1)%len(bad[key]["classes"])
    (a.output/"fault-label.json").write_text(json.dumps(bad),encoding="utf-8")
    pq.write_table(pq.read_table(dest/"chunks/atoms.parquet").slice(0,len(chunks)-1),a.output/"fault-tail.parquet")
    bad_role=copy.deepcopy(anchors);bad_role[key]["partitions"][0]="evaluation"
    (a.output/"fault-role.json").write_text(json.dumps(bad_role),encoding="utf-8")
    negatives={}
    for name,action in {
        "changed_label":lambda:roles_and_labels(json.loads((a.output/"fault-label.json").read_bytes()),documents,oracle,questions),
        "changed_role":lambda:roles_and_labels(json.loads((a.output/"fault-role.json").read_bytes()),documents,oracle,questions),
        "dropped_tail":lambda:chunks_match(pq.read_table(a.output/"fault-tail.parquet").to_pylist(),documents),
    }.items():
        try:action()
        except ValueError as exc:negatives[name]=str(exc)
        else:raise ValueError("MNLM_FINANCE_READBACK_NEGATIVE_ACCEPTED")
    result=dict(complete=True,source_report_sha256=publication["report_sha256"],entities=len(documents),chunks=len(chunks),
        questions=list(questions),role_counts={p:dict(Counter(v["partitions"])) for p,v in anchors.items()},
        negative_controls=negatives,acceptance="author-checked",scientific_admission=False,served=False,
        wall_seconds=time.perf_counter()-start,code_sha256=sha(Path(__file__)))
    (a.output/"report.json").write_text(json.dumps(result,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    (a.output/"manifest.json").write_text(json.dumps(dict(report_sha256=sha(a.output/"report.json")))+"\n",encoding="utf-8")
    print(json.dumps(result),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("population","census","records","output"):p.add_argument("--"+name,type=Path,required=True)
    run(p.parse_args())
