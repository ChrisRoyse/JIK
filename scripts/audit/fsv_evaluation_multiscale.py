"""Actual-source minimum proof, complete multiscale run and boundary controls."""
import argparse
import copy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from text.evaluation_multiscale import field_record, run, file_sha, canonical
from text.multiscale_spans import causal_view
from audit.evaluation_multiscale_readback import audit, audit_field, audit_sequence


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("source", "publication", "output"):
        p.add_argument("--"+arg, type=Path, required=True)
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=False)
    results = []

    def put(path, value):
        path.write_bytes(canonical(value)+b"\n")

    def check(name, passed, **evidence):
        results.append(dict(name=name, passed=bool(passed), **evidence))

    def refuses(name, code, fn):
        observed = None
        try:
            fn()
        except ValueError as exc:
            observed = str(exc)
        check(name, observed is not None and code in observed, expected=code, observed=observed)

    before = {f.name: file_sha(f) for f in a.source.iterdir() if f.is_file()}
    put(a.output/"before.json", before)
    manifest = json.loads((a.source/"report.json").read_bytes())
    proof = []; chosen = None; finance = None
    for domain in manifest["sources"]:
        with (a.source/(domain+".jsonl")).open("rb") as f:
            row = json.loads(next(f))
        graphs = [field_record(row, field) for field in row["fields"]]
        path = a.output/(domain+"-minimum.json")
        put(path, dict(source_record=row, records=graphs))
        persisted = json.loads(path.read_bytes())
        count = audit_sequence([persisted["source_record"]], iter(persisted["records"]))
        proof.append(dict(domain=domain, counts=dict(count), source_file_sha256=before[domain+".jsonl"], proof_sha256=file_sha(path)))
        if domain == "finance":
            finance = row, graphs
        for field, record in zip(row["fields"], graphs):
            if record["graph"] and sum(n["level"] == "sentence" for n in record["graph"]["nodes"]) >= 3:
                chosen = row, field, record
    put(a.output/"minimum-proof.json", proof)
    check("B1 six persisted actual source documents conserve every field", len(proof) == 6)
    check("B1 real multi-sentence field exercises parent context", chosen is not None)
    row, field, record = chosen
    bad = copy.deepcopy(record); bad["graph"]["nodes"][0]["start"] += 1
    put(a.output/"shifted-span.json", bad)
    refuses("B2 shifted span refuses", "MNLM_MULTI_READBACK_ID", lambda: audit_field(bad, row, field))
    bad = copy.deepcopy(record); bad["graph"]["edges"].pop()
    put(a.output/"missing-edge.json", bad)
    refuses("B2 missing structural edge refuses", "MNLM_MULTI_READBACK_RELATIONS", lambda: audit_field(bad, row, field))
    bad = copy.deepcopy(record); bad["group_keys"] = ["different-document"]
    put(a.output/"wrong-group.json", bad)
    refuses("B3 wrong parent group refuses", "MNLM_EVAL_MULTI_READBACK_GROUP", lambda: audit_field(bad, row, field))
    bad_field = copy.deepcopy(field); bad_field["text"] = field["text"][:-1]
    put(a.output/"changed-field.json", bad_field)
    refuses("B1 changed captured field refuses", "MNLM_EVAL_MULTI_FIELD_HASH", lambda: field_record(row, bad_field))
    frow, fgraphs = finance
    put(a.output/"missing-finance-field.json", fgraphs[:-1])
    refuses("B2 omitted financial field refuses", "MNLM_EVAL_MULTI_READBACK_MISSING_FIELD",
            lambda: audit_sequence([frow], iter(fgraphs[:-1])))
    put(a.output/"duplicate-finance-field.json", fgraphs+[fgraphs[-1]])
    refuses("B2 duplicated financial field refuses", "MNLM_EVAL_MULTI_READBACK_EXTRA_FIELD",
            lambda: audit_sequence([frow], iter(fgraphs+[fgraphs[-1]])))
    graph = record["graph"]
    cutoff = next(n["end"] for n in graph["nodes"] if n["level"] == "sentence")
    visible = causal_view(graph, cutoff, record["split_group"])
    put(a.output/"causal-view.json", dict(cutoff=cutoff, visible=visible))
    ids = {n["id"] for n in visible["nodes"]}
    check("B3 partially observed parents and future edges hidden", len(ids) < len(graph["nodes"])
          and all(n["end"] <= cutoff for n in visible["nodes"])
          and all(e["from"] in ids and e["to"] in ids for e in visible["edges"]))
    refuses("B3 wrong-group consumer refuses", "MNLM_MULTISCALE_SPLIT_GROUP", lambda: causal_view(graph, cutoff, "other"))
    refuses("B3 invalid cutoff refuses", "MNLM_MULTISCALE_CUTOFF", lambda: causal_view(graph, len(field["text"])+1, record["split_group"]))
    check("B1 actual source has no invented pages", all(not r["graph"] or not any(n["level"] == "page" for n in r["graph"]["nodes"]) for r in fgraphs))
    put(a.output/"minimum-checks.json", results)
    if not all(r["passed"] for r in results):
        raise SystemExit("MNLM_EVAL_MULTI_FSV_MINIMUM_FAILED")
    pin = before["report.json"]
    refuses("B1 wrong source manifest refuses before production", "MNLM_EVAL_MULTI_SOURCE_PIN",
            lambda: run(a.source, "0"*64, a.output/"wrong-source"))
    result = run(a.source, pin, a.publication)
    check("B2 full real publication complete", result["complete"], errors=result["errors"])
    cold = audit(a.publication, a.source)
    put(a.output/"cold-readback.json", cold)
    check("B2 every field span and edge cold-read back", cold["all_spans_and_relations_equal"], totals=cold["totals"])
    refuses("B3 existing generation refuses overwrite", "MNLM_EVAL_MULTI_EXISTS", lambda: run(a.source, pin, a.publication))
    after = {f.name: file_sha(f) for f in a.source.iterdir() if f.is_file()}
    put(a.output/"after.json", after)
    check("B3 captured source bytes unchanged", before == after)
    result = dict(checks=results, passed=sum(r["passed"] for r in results), total=len(results),
                  cold_readback=cold, acceptance="author-checked", scientific_admission=False)
    put(a.output/"fsv.json", result)
    print(json.dumps({k: result[k] for k in ("passed", "total")}))
    if result["passed"] != result["total"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
