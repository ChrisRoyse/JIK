"""Independent source-witness replay and census readback; no producer imports."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from canon.common import iter_canon_docs


def check(ok, code):
    if not ok:
        raise ValueError("MNLM_FIXED_ANCHOR_READBACK_"+code)


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def bound(path, pin):
    check(sha(path) == pin, "PIN")
    return json.loads(path.read_bytes())


def counts(rows, contract):
    result = {}
    for family,spec in contract["families"].items():
        result[family] = {}
        for prop,classes in spec["questions"].items():
            bins = {k:[0]*len(classes) for k in ("source_supported","singleton_value","fresh_singleton")}
            for row in rows:
                if row["family"] != family:
                    continue
                values = {f[1] for f in row["assertions"] if f[0] == prop}
                check(sorted(values) == row["values"].get(prop,[]), "ASSERTION_VALUES")
                for i,value in enumerate(classes):
                    bins["source_supported"][i] += value in values
                    bins["singleton_value"][i] += values == {value}
                    bins["fresh_singleton"][i] += values == {value} and not row["historical"]
            result[family][prop] = dict(classes=classes,**bins,
                balanced_fresh_max=min(bins["fresh_singleton"])*len(classes))
    return result


def run(a):
    start=time.perf_counter(); a.output.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((a.source/"manifest.json").read_bytes())
    report=bound(a.source/"report.json",manifest["report_sha256"])
    intent=bound(a.source/"intent.json",manifest["intent_sha256"])
    contract=bound(a.source/"contract.json",report["config_sha256"])
    check(report["complete"] and report["scientific_admission"] is False and report["serving"] is False,
          "CLAIMS")
    path=a.source/"private/eligible.jsonl"
    check(sha(path)==manifest["eligible_sha256"]==report["eligible_sha256"],"ROWS_PIN")
    rows=[json.loads(line) for line in path.read_bytes().splitlines()]
    check(len(rows)==report["with_prose"] and len({r["entity"] for r in rows})==len(rows),"ROW_COUNT")
    historic=set()
    for family,spec in contract["families"].items():
        path=a.records/"records"/family/"rows.jsonl"
        check(sha(path)==spec["rows_sha256"],"HISTORICAL_PIN")
        historic.update(json.loads(line)["entity"] for line in path.read_bytes().splitlines())
    check(all(r["historical"]==(r["entity"] in historic) for r in rows),"HISTORICAL_EXCLUSION")
    actual=counts(rows,contract)
    check(actual==report["findings"],"COUNTS")
    # Before-score hash selection; inspect actual original rows, not summaries.
    selected=sorted(rows,key=lambda r:hashlib.sha256(r["entity"].encode()).hexdigest())[:32]
    expected={}
    for row in selected:
        for family,kind,shard,index in row["memberships"]:
            check(kind in intent["domain_classes"][family],"MEMBERSHIP_CLASS")
            expected[shard,index]=dict(s=row["entity"],p="P31",o=kind)
        for prop,value,shard,index in row["assertions"]:
            expected[shard,index]=dict(s=row["entity"],p=prop,o=value)
        text=row["text_source"]
        expected[text["shard"],text["row"]]=dict(entity=row["entity"],sha256=text["sha256"],bytes=text["utf8_bytes"])
    check(sha(a.canon/"manifest.jsonl")==report["manifest_sha256"],"ORIGINAL_MANIFEST")
    matched=set(); scanned={}
    for shard,declaration in report["source_manifest"].items():
        n=0
        for record in iter_canon_docs(a.canon,shard,declaration["out_blake3"]):
            n+=1; key=(shard,record["row"])
            if key not in expected:
                continue
            target=expected[key]
            if shard=="wikidata5m_text":
                raw=record["text"].encode()
                check(record["f"]["entity_id"]==target["entity"] and len(raw)==target["bytes"]
                      and hashlib.sha256(raw).hexdigest()==target["sha256"],"ORIGINAL_PROSE")
            else:
                check(all(record["f"][k]==v for k,v in target.items()),"ORIGINAL_ASSERTION")
            matched.add(key)
        check(n==declaration["docs"]==report["source_rows"][shard],"SOURCE_COUNT")
        scanned[shard]=n
    check(set(expected)==matched,"WITNESS_COMPLETE")
    negative={}
    for name,action in {
        "changed_class":lambda:bound(a.source/"fault-class.json",report["config_sha256"]),
        "changed_counts":lambda:check(json.loads((a.source/"fault-counts.json").read_bytes())==actual,"COUNTS"),
    }.items():
        try:
            action()
        except ValueError as exc:
            negative[name]=str(exc)
        else:
            raise ValueError("MNLM_FIXED_ANCHOR_READBACK_NEGATIVE_ACCEPTED")
    changed=copy.deepcopy(rows[0]); changed["historical"]=not changed["historical"]
    (a.output/"fault-membership.json").write_text(json.dumps(changed),encoding="utf-8")
    try:
        record=json.loads((a.output/"fault-membership.json").read_bytes())
        check(record["historical"]==(record["entity"] in historic),"HISTORICAL_EXCLUSION")
    except ValueError as exc:
        negative["changed_historical_membership"]=str(exc)
    else:
        raise ValueError("MNLM_FIXED_ANCHOR_READBACK_NEGATIVE_ACCEPTED")
    result=dict(source_report_sha256=manifest["report_sha256"],eligible_rows=len(rows),source_rows=scanned,
        original_cases_replayed=len(selected),original_records_replayed=len(matched),negative_controls=negative,
        findings=actual,complete=True,acceptance="author-checked",scientific_admission=False,serving=False,
        wall_seconds=time.perf_counter()-start,code_sha256=sha(__file__))
    (a.output/"report.json").write_text(json.dumps(result,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    (a.output/"manifest.json").write_text(json.dumps(dict(report_sha256=sha(a.output/"report.json")))+"\n",encoding="utf-8")
    print(json.dumps(dict(source_rows=scanned,original_records_replayed=len(matched),negative_controls=negative,
                         wall_seconds=result["wall_seconds"])),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("source","records","canon","output"):
        p.add_argument("--"+name,type=Path,required=True)
    run(p.parse_args())
