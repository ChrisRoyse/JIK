"""Complete source-field capture for evaluation split construction (#14/#27).

Samples raw document IDs, not sentences or scores. This publication is not yet
a sealed test: exact/near-duplicate, referent, time and composition work follows.
"""
import argparse
from collections import Counter
import hashlib
import heapq
import json
from pathlib import Path
import sys
import time

import pyarrow.parquet as pq

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from text.atomise_pilot import records, doc_id
from senses.trajectory_population import publish

SCHEMA = "evaluation-source-capture-v1"
SECTIONS = tuple("section_"+s for s in ("1","1A","1B","2","3","4","5","6","7","7A",
                 "8","9","9A","9B","10","11","12","13","14","15"))


def require(ok, code):
    if not ok:
        raise ValueError("MNLM_EVAL_CAPTURE_"+code)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def file_sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()


def rank(seed, domain, document):
    return int(sha(canonical([SCHEMA,seed,domain,document])),16)


def fields(domain, row):
    names = SECTIONS if domain == "finance" else ("text",)
    require(all(n in row and isinstance(row[n],str) for n in names), "TEXT_FIELDS")
    if domain == "finance":
        require({n for n in row if n.startswith("section_")} == set(SECTIONS), "SECTION_INVENTORY")
    return [{"name":n,"text":row[n],"characters":len(row[n]),"utf8_bytes":len(row[n].encode()),
             "sha256":sha(row[n].encode())} for n in names]


def clock(domain, row):
    if domain in ("biomedical","science"):
        return {"field":"created","value":row.get("created"),"status":"provider-created:semantics-unadjudicated"}
    if domain == "finance":
        return {"field":"year","value":row.get("year"),"status":"provider-year:semantics-unadjudicated"}
    if domain == "legal":
        return {"field":"created","value":row.get("created"),"status":"ingestion-time:not-decision-time"}
    return {"field":None,"value":None,"status":"absent:document-event-time"}


def capture(src, row, ordinal, document, selection_rank):
    views = fields(src["domain"],row)
    excluded = {v["name"] for v in views}
    identity = {"repo":src["repo"],"revision":src["revision"],"file":src["file"],
                "source_sha256":src["sha256"],"ordinal":ordinal,"document_id":document,
                "record_sha256":sha(canonical(row))}
    groups = ["document:"+sha(canonical([src["repo"],document]))]
    status = "document-only:world-referent-unresolved"
    if src["domain"] == "finance" and isinstance(row.get("cik"),str) and row["cik"]:
        groups.append("sec-cik:"+row["cik"]); status="source-attested:sec-cik"
    return {"id":sha(canonical(identity)),"domain":src["domain"],"source":identity,
            "selection_rank_hex":f"{selection_rank:064x}","fields":views,
            "metadata":{k:v for k,v in row.items() if k not in excluded},"source_licence":src["licence"],
            "group_keys":groups,"entity_resolution":status,"clock":clock(src["domain"],row),
            "page_status":"absent:no-attested-page-offsets","sealed_evaluation":False}


def exposure(history_root, contract):
    seen, reports = set(), {}
    for name, expected in contract["history"].items():
        root = history_root/name; raw=(root/"manifest.json").read_bytes(); report=json.loads(raw)
        require(sha(raw)==expected and report["raw_manifest_sha256"]==contract["raw_manifest_sha256"], "HISTORY_MANIFEST")
        path=root/"atoms.parquet"
        require(file_sha(path)==report["atoms_sha256"], "HISTORY_ATOMS")
        count=0; local=set()
        for batch in pq.ParquetFile(path).iter_batches(batch_size=65536,columns=["domain","doc"]):
            values=batch.to_pydict(); count+=len(batch)
            local.update(zip(values["domain"],values["doc"]))
        require(count==report["atoms"] and set(k for k,_ in local)==set(contract["domains"]), "HISTORY_COUNT")
        seen.update(local)
        reports[name]={"manifest_sha256":sha(raw),"atoms_sha256":report["atoms_sha256"],"atoms":count,"documents":len(local)}
    return seen,reports


def select(src, path, seen, target, seed, error_path):
    start=time.perf_counter()
    require(path.stat().st_size==src["bytes"] and file_sha(path)==src["sha256"], "SOURCE_BYTES")
    hashed=time.perf_counter()
    counts=Counter(); identifiers=set(); heap=[]
    schema_counts=Counter()
    with error_path.open("xb") as errors:
        for ordinal,row in enumerate(records(path)):
            counts["raw_records"]+=1
            try:
                document=doc_id(src["domain"],row,ordinal)
                require(isinstance(document,str) and bool(document), "DOCUMENT_ID")
                schema_counts[sha(canonical(sorted(row)))]+=1
                if document in identifiers:
                    counts["duplicate_document_ids"]+=1;continue
                identifiers.add(document)
                if (src["domain"],document) in seen:
                    counts["historically_exposed_documents"]+=1;continue
                names=SECTIONS if src["domain"]=="finance" else ("text",)
                require(all(n in row and isinstance(row[n],str) for n in names),"TEXT_FIELDS")
                if src["domain"]=="finance":
                    require({n for n in row if n.startswith("section_")}==set(SECTIONS),"SECTION_INVENTORY")
                if not any(row[n].strip() for n in names):
                    counts["empty_documents"]+=1;continue
                counts["eligible_documents"]+=1
                r=rank(seed,src["domain"],document)
                if len(heap)<target or (r,ordinal)<(-heap[0][0],-heap[0][1]):
                    item=(-r,-ordinal,capture(src,row,ordinal,document,r))
                    if len(heap)<target:heapq.heappush(heap,item)
                    else:heapq.heapreplace(heap,item)
            except (ValueError,KeyError,TypeError) as exc:
                counts["errors"]+=1
                errors.write(canonical({"ordinal":ordinal,"error":str(exc)})+b"\n")
    result=[item[2] for item in sorted(heap,key=lambda i:(-i[0],-i[1]))]
    counts["selected_documents"]=len(result)
    counts["selected_fields"]=sum(len(r["fields"]) for r in result)
    counts["selected_characters"]=sum(f["characters"] for r in result for f in r["fields"])
    elapsed=time.perf_counter()-start
    return result,{"counts":dict(counts),"schema_counts":dict(schema_counts),"requested_documents":target,
                  "available":len(result)==target,"source_sha256":src["sha256"],"source_bytes":src["bytes"],
                  "timing":{"source_hash_seconds":hashed-start,"scan_selection_seconds":elapsed-(hashed-start),"wall_seconds":elapsed}}


def run(raw_root, history_root, config, output):
    start=time.perf_counter();require(not output.exists(),"EXISTS")
    config_raw=config.read_bytes();contract=json.loads(config_raw)
    require(contract["schema"]==SCHEMA and contract["sealed_evaluation"] is False,"CONTRACT")
    target=contract["target_documents_per_source"]
    require(type(target) is int and target>0,"TARGET")
    manifest_raw=(raw_root/"manifest.json").read_bytes();manifest=json.loads(manifest_raw)
    require(sha(manifest_raw)==contract["raw_manifest_sha256"],"SOURCE_MANIFEST")
    sources=manifest["sources"]
    require(len(sources)==len(contract["domains"]) and {s["domain"] for s in sources}==set(contract["domains"]),"SOURCE_INVENTORY")
    seen,histories=exposure(history_root,contract)
    loaded=time.perf_counter();output.mkdir(parents=True)
    reports,failures={},[]
    for src in sources:
        domain=src["domain"]
        try:
            path=(raw_root/src["path"]).resolve()
            require(path.is_relative_to(raw_root.resolve()),"SOURCE_PATH")
            selected,summary=select(src,path,seen,target,contract["selection_seed"],output/(domain+".errors.jsonl"))
            dest=output/(domain+".jsonl")
            with dest.open("xb") as f:
                for record in selected:f.write(canonical(record)+b"\n")
            summary.update(file=dest.name,sha256=file_sha(dest),bytes=dest.stat().st_size,
                           errors_sha256=file_sha(output/(domain+".errors.jsonl")))
            reports[domain]=summary
            if summary["counts"].get("errors",0):failures.append({"domain":domain,"error":"MNLM_EVAL_CAPTURE_RECORD_ERRORS"})
            print(json.dumps({"domain":domain,**summary["counts"],"available":summary["available"]}),flush=True)
        except (OSError,ValueError,KeyError,TypeError) as exc:
            failures.append({"domain":domain,"error":str(exc)})
    result={"schema":SCHEMA,"contract_sha256":sha(config_raw),"raw_manifest_sha256":sha(manifest_raw),
            "history":histories,"historically_exposed_document_ids":len(seen),"sources":reports,"errors":failures,
            "complete":not failures and set(reports)==set(contract["domains"]),
            "all_requested_source_counts_available":all(r["available"] for r in reports.values()) and len(reports)==len(sources),
            "sealed_evaluation":False,"scientific_admission":False,"served":False,
            "pending":["exact/near-duplicate closure","cross-document referent resolution","adjudicated event-time clocks",
                       "source/time/composition partition","grounded correctness anchors","final once-only test seal"],
            "code_sha256":sha(Path(__file__).read_bytes()),"reader_code_sha256":sha(Path(sys.modules[records.__module__].__file__).read_bytes()),
            "timing":{"history_and_setup_seconds":loaded-start,"wall_seconds":time.perf_counter()-start}}
    publish(output/"report.json",result)
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ("raw","history","contract","output"):p.add_argument("--"+arg,type=Path,required=True)
    a=p.parse_args();result=run(a.raw,a.history,a.contract,a.output)
    print(json.dumps({k:result[k] for k in ("complete","all_requested_source_counts_available","errors","timing")}))
    if not result["complete"]:raise SystemExit(2)
