"""Complete source support for the unchanged A2 questions; no model scoring (#12).

One statement pass and one prose pass. Retain property metadata, never the whole
text corpus. Published private rows are anchor witnesses, not instrument inputs.
Historical membership is excluded from the fresh-source counts, not erased.
"""
import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

import blake3

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from canon.common import iter_canon_docs
from text.anchor_domains import CLASSES, SHARDS


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def check(ok, code):
    if not ok:
        raise ValueError("MNLM_FIXED_ANCHOR_CENSUS_"+code)


def write(path, value):
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")


def independent_counts(rows, config):
    """Read persisted assertions rather than trusting producer counters."""
    out = {}
    for family, spec in config["families"].items():
        out[family] = {}
        for prop, classes in spec["questions"].items():
            selected = [r for r in rows if r["family"] == family and prop in r["values"]]
            support, unique, fresh = [], [], []
            for value in classes:
                support.append(sum(value in r["values"][prop] for r in selected))
                unique.append(sum(r["values"][prop] == [value] for r in selected))
                fresh.append(sum(r["values"][prop] == [value] and not r["historical"] for r in selected))
            out[family][prop] = dict(classes=classes, source_supported=support, singleton_value=unique,
                                      fresh_singleton=fresh, balanced_fresh_max=min(fresh)*len(classes))
    return out


def run(a):
    start = time.perf_counter(); phases = {}
    config_pin, manifest_pin = sha(a.contract), sha(a.canon/"manifest.jsonl")
    config = json.loads(a.contract.read_bytes())
    check(config["schema"] == "a2-calibration-contract-v1", "CONTRACT")
    declarations = [json.loads(line) for line in (a.canon/"manifest.jsonl").read_bytes().splitlines()]
    manifest = {}
    for name in SHARDS+("wikidata5m_text",):
        records = [r for r in declarations if r["shard"] == name]
        check(len(records) == 1, "SOURCE_PUBLICATION")
        record = records[0]
        check(record["source"] == "wikidata5m" and record["contract"] == "canon-v1" and
              (a.canon/(name+".jsonl.zst")).stat().st_size == record["out_bytes"], "SOURCE_SIZE")
        manifest[name] = record
    historical = set()
    for family, spec in config["families"].items():
        path = a.records/"records"/family/"rows.jsonl"
        check(sha(path) == spec["rows_sha256"], "HISTORICAL_PIN")
        historical.update(json.loads(line)["entity"] for line in path.read_bytes().splitlines())
    check(not a.output.exists(), "OUTPUT_EXISTS")
    a.output.mkdir(parents=True)
    write(a.output/"intent.json", dict(config_sha256=config_pin, manifest_sha256=manifest_pin,
        questions={f:s["questions"] for f,s in config["families"].items()}, domain_classes=CLASSES,
        historical_entities=len(historical), scientific_admission=False, serving=False))
    (a.output/"contract.json").write_bytes(a.contract.read_bytes())
    (a.output/"source-manifest.jsonl").write_bytes((a.canon/"manifest.jsonl").read_bytes())
    phases["input_binding_seconds"] = time.perf_counter()-start
    tick = time.perf_counter()
    properties = {p for s in config["families"].values() for p in s["questions"]}
    class_family = {c:f for f,cs in CLASSES.items() for c in cs}
    facts, memberships = defaultdict(list), defaultdict(list)
    scanned = {}
    for shard in SHARDS:
        n = 0
        for rec in iter_canon_docs(a.canon, shard, manifest[shard]["out_blake3"]):
            n += 1; f = rec["f"]
            if f["p"] == "P31" and f["o"] in class_family:
                memberships[f["s"]].append((class_family[f["o"]], f["o"], shard, rec["row"]))
            elif f["p"] in properties:
                facts[f["s"]].append((f["p"], f["o"], shard, rec["row"]))
        check(n == manifest[shard]["docs"], "STATEMENT_COUNT")
        scanned[shard] = n
        print(json.dumps(dict(stage="statements", shard=shard, rows=n)), flush=True)
    family_order = list(CLASSES)
    candidates = {}
    for entity, member in memberships.items():
        family = min((m[0] for m in member), key=family_order.index)
        relevant = [f for f in facts.get(entity, ()) if f[0] in config["families"][family]["questions"]]
        values = defaultdict(set)
        for prop,value,_,_ in relevant:
            values[prop].add(value)
        if not any(set(config["families"][family]["questions"][p]) & v for p,v in values.items()):
            continue
        candidates[entity] = dict(entity=entity, family=family, historical=entity in historical,
            memberships=member, values={p:sorted(v) for p,v in values.items()}, assertions=relevant)
    del facts, memberships
    phases["statement_census_seconds"] = time.perf_counter()-tick; tick = time.perf_counter()
    shard = "wikidata5m_text"; rows = []; seen = set(); n = 0
    for rec in iter_canon_docs(a.canon, shard, manifest[shard]["out_blake3"]):
        n += 1; entity = rec["f"].get("entity_id")
        if entity not in candidates or not rec["text"].strip():
            continue
        check(entity not in seen, "DUPLICATE_PROSE")
        seen.add(entity); raw = rec["text"].encode()
        check(rec["n"] == len(raw) and rec["b3"] == blake3.blake3(raw).hexdigest()[:32], "PROSE_RECORD")
        rows.append({**candidates[entity], "text_source":dict(shard=shard, row=rec["row"],
            sha256=hashlib.sha256(raw).hexdigest(), utf8_bytes=len(raw))})
    check(n == manifest[shard]["docs"], "PROSE_COUNT"); scanned[shard] = n
    phases["prose_binding_seconds"] = time.perf_counter()-tick; tick = time.perf_counter()
    rows.sort(key=lambda r:r["entity"])
    counts = defaultdict(Counter)
    for row in rows:
        for prop,values in row["values"].items():
            classes = config["families"][row["family"]]["questions"][prop]
            for value in set(values)&set(classes):
                key = (row["family"],prop,value)
                counts[key]["source_supported"] += 1
                if len(values) == 1:
                    counts[key]["singleton_value"] += 1
                    counts[key]["fresh_singleton"] += not row["historical"]
    summary = {}
    for family,spec in config["families"].items():
        summary[family] = {}
        for prop,classes in spec["questions"].items():
            record = {k:[counts[family,prop,c][k] for c in classes]
                      for k in ("source_supported","singleton_value","fresh_singleton")}
            summary[family][prop] = dict(classes=classes,**record,
                balanced_fresh_max=min(record["fresh_singleton"])*len(classes))
    (a.output/"private").mkdir()
    path = a.output/"private/eligible.jsonl"
    with path.open("x",encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row,sort_keys=True)+"\n")
    del rows
    persisted = [json.loads(line) for line in path.read_bytes().splitlines()]
    check(independent_counts(persisted,config) == summary, "INDEPENDENT_COUNTS")
    check(sha(a.contract) == config_pin and sha(a.canon/"manifest.jsonl") == manifest_pin, "SOURCE_DRIFT")
    # Physical corruptions cannot be a new accepted corpus/target.
    fault_config = copy.deepcopy(config)
    fam = next(iter(fault_config["families"])); prop = next(iter(fault_config["families"][fam]["questions"]))
    fault_config["families"][fam]["questions"][prop][0] = "Q0"
    write(a.output/"fault-class.json",fault_config)
    check(sha(a.output/"fault-class.json") != config_pin, "NEGATIVE_CLASS_PIN")
    fault_counts = copy.deepcopy(summary)
    fault_counts[fam][prop]["source_supported"][0] += 1
    write(a.output/"fault-counts.json",fault_counts)
    check(independent_counts(persisted,config) != fault_counts, "NEGATIVE_COUNTS")
    phases["publication_and_recount_seconds"] = time.perf_counter()-tick
    result = dict(schema="fixed-anchor-census-v1",complete=True, scientific_admission=False,serving=False,
        acceptance="author-checked",config_sha256=config_pin,manifest_sha256=manifest_pin,
        source_rows=scanned,candidate_entities=len(candidates),with_prose=len(persisted),
        eligible_sha256=sha(path),findings=summary,source_manifest=manifest,
        negative_controls=dict(changed_class_refused=True,changed_counts_refused=True),
        phases_seconds=phases,wall_seconds=time.perf_counter()-start,code_sha256=sha(__file__))
    write(a.output/"report.json",result)
    write(a.output/"manifest.json",dict(report_sha256=sha(a.output/"report.json"),intent_sha256=sha(a.output/"intent.json"),
        eligible_sha256=sha(path)))
    print(json.dumps(dict(with_prose=len(persisted),balanced_fresh_max={f:{p:v["balanced_fresh_max"] for p,v in q.items()}
                     for f,q in summary.items()},wall_seconds=result["wall_seconds"])),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("canon","records","contract","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    run(parser.parse_args())
