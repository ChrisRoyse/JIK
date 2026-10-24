"""Read source scores, checkpoint bytes and selection events independently of the controller."""
import argparse
import json
import math
from pathlib import Path
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.text_learning_readback import checkpoint, journal
from senses.contract import canonical, digest
from senses.trajectory_population import publish
from text.constellation_corpus import file_sha, need


def measures(root, name, groups):
    domain_of = {i: g["domain"] for g in groups if g["split"] == "development" for i in g["indices"]}
    rows = {n: [] for n in ("general", "legal", "science")}
    with sqlite3.connect(f"file:{root/(name+'.db')}?mode=ro", uri=True) as db:
        for raw, in db.execute("SELECT body FROM events ORDER BY seq"):
            body = json.loads(raw)
            if body["kind"] == "source_reveal" and body["left"] in domain_of:
                value = json.loads(db.execute("SELECT data FROM payloads WHERE hash=?", (body["payload"],)).fetchone()[0])
                rows[domain_of[body["left"]]].append(value)
    result = {}
    for domain, items in rows.items():
        result[domain] = dict(count=len(items),
            region_nll_bits=math.fsum(v["region_nll_bits"] for v in items)/len(items),
            instruments={n: {k: math.fsum(v["instruments"][n][k] for v in items)/len(items)
                             for k in ("prediction", "persistence")} for n in items[0]["instruments"]})
    return result


def compare(payload, scores, groups, selected, established):
    name = payload["candidate"]
    need(name in ("cycle-1", "cycle-2"), "SELECTION_READBACK_CANDIDATE")
    parent = "baseline" if name == "cycle-1" else "cycle-1"
    parent_name = "initial-training" if name == "cycle-1" else "cycle-1"
    acquire, retain = (["general", "legal"], []) if name == "cycle-1" else (["science"], ["general", "legal"])
    declared = {g["domain"]: g for g in groups if g["split"] == "development"}
    result = payload["result"]
    need(set(result["checks"]) == set(acquire+retain), "SELECTION_READBACK_DOMAIN_ROSTER")
    acquired, preserved = True, True
    for domain in acquire+retain:
        old, new = scores[parent][domain], scores[name][domain]
        row = result["checks"][domain]
        acquisition = domain in acquire
        need(row["document"] == declared[domain]["context"] and row["count"] == old["count"] == new["count"]
             and row["role"] == ("acquisition" if acquisition else "retention")
             and set(row["instruments"]) == set(old["instruments"]) == set(new["instruments"]), "SELECTION_READBACK_UNIT")
        # Independent summation is a numerical readback check, not a relaxed gate.
        need(abs(row["region_logloss"]["parent"]-old["region_nll_bits"]) < 1e-12
             and abs(row["region_logloss"]["candidate"]-new["region_nll_bits"]) < 1e-12, "SELECTION_READBACK_LOGSCORE")
        a, b = row["region_logloss"]["parent"], row["region_logloss"]["candidate"]
        passed = b < a if acquisition else b <= a
        for lane, values in row["instruments"].items():
            need(abs(values["parent"]-old["instruments"][lane]["prediction"]) < 1e-12
                 and abs(values["candidate"]-new["instruments"][lane]["prediction"]) < 1e-12
                 and abs(values["persistence"]-new["instruments"][lane]["persistence"]) < 1e-12,
                 "SELECTION_READBACK_NATIVE_SCORE")
            ok = values["candidate"] < min(values["parent"], values["persistence"]) if acquisition else values["candidate"] <= values["parent"]
            need(values["passed"] == ok, "SELECTION_READBACK_NATIVE_GATE")
            passed &= ok
        need(row["passed"] == passed, "SELECTION_READBACK_UNIT_GATE")
        if acquisition:
            acquired &= passed
        else:
            preserved &= passed
    reasons = ([] if selected == parent_name else ["UNRETAINED_PARENT"])
    reasons += [] if set(established) == set(retain) else ["ESTABLISHED_DOMAIN_MISMATCH"]
    reasons += [] if acquired else ["ACQUISITION_GATE_FAILED"]
    reasons += [] if preserved else ["RETENTION_GATE_FAILED"]
    need(result["reasons"] == reasons and result["eligible_diagnostic_lineage"] == (not reasons)
         and result["acquisition_passed"] == acquired and result["retention_passed"] == preserved
         and all(result[k] is False for k in ("serving", "meaning_asserted", "conditional_bits_measured")), "SELECTION_READBACK_DECISION")
    expected = selected if reasons else name
    expected_domains = established if reasons else sorted(set(established+acquire))
    need(payload["before"] == selected and payload["selected"] == expected
         and payload["established"] == expected_domains, "SELECTION_READBACK_POINTER")
    return expected, expected_domains


def run(a):
    need(file_sha(a.run/"report.json") == a.pin and file_sha(a.source_run/"report.json") == a.source_pin, "SELECTION_READBACK_PIN")
    report = json.loads((a.run/"report.json").read_bytes())
    source = json.loads((a.source_run/"report.json").read_bytes())
    need(report["source_run"] == a.source_pin, "SELECTION_READBACK_SOURCE")
    base = source["data"]
    while "groups" not in base:
        base = base["base"]
    groups = base["groups"]
    scores = {n: measures(a.source_run, n, groups) for n in source["tapes"]}
    source_journals = {n: journal(a.source_run/(n+".db"), r["retained"], r) for n, r in source["tapes"].items()}
    checkpoints = {n: checkpoint(a.source_run/n, pin)[0] for n, pin in source["checkpoints"].items()}
    selected, established, head, count, decisions = "initial-training", [], "0"*64, 0, []
    with sqlite3.connect(f"file:{a.run/'selection.db'}?mode=ro", uri=True) as db:
        contract = json.loads(db.execute("SELECT contract FROM metadata").fetchone()[0])
        need(digest(contract) == report["retained"]["contract"], "SELECTION_READBACK_CONTRACT")
        for seq, previous, raw, pin in db.execute("SELECT seq,previous,body,hash FROM events ORDER BY seq"):
            body = json.loads(raw)
            need(seq == count and previous == head and raw == canonical(body)
                 and pin == digest(dict(seq=seq, previous=previous, body=body)), "SELECTION_READBACK_CHAIN")
            blob = db.execute("SELECT data FROM payloads WHERE hash=?", (body["payload"],)).fetchone()[0]
            payload = json.loads(blob)
            need(canonical(payload) == blob and digest(payload) == body["payload"], "SELECTION_READBACK_PAYLOAD")
            if count == 0:
                need(body["kind"] == "initialize_diagnostic_learner" and payload["selected"] == selected
                     and payload["established"] == [], "SELECTION_READBACK_INITIAL")
                need(payload["record"]["fingerprint"] == checkpoints[selected]["fingerprint"], "SELECTION_READBACK_INITIAL_BYTES")
            else:
                need(body["kind"] == "review_candidate" and payload["candidate"] == "cycle-"+str(count), "SELECTION_READBACK_ORDER")
                selected, established = compare(payload, scores, groups, selected, established)
                need(payload["selected_fingerprint"] == checkpoints[selected]["fingerprint"]
                     and payload["candidate_record"]["fingerprint"] == checkpoints[payload["candidate"]]["fingerprint"], "SELECTION_READBACK_ACTUAL_MODEL")
                decisions.append(dict(candidate=payload["candidate"], selected=selected, reasons=payload["result"]["reasons"]))
            count, head = count+1, pin
    need(count == 3 and report["retained"] == dict(head=head, contract=digest(contract), selected=selected, established=established), "SELECTION_READBACK_RETAINED")
    chosen = report["selected_predictions"]
    output = journal(a.run/"selected.db", chosen["retained"], chosen)
    need(chosen["outcomes"] == source["tapes"]["baseline"]["outcomes"]
         and chosen["fingerprint"] == checkpoints[selected]["fingerprint"] and selected == "initial-training",
         "SELECTION_READBACK_REJECTED_MODEL_USE")
    publish(a.output, dict(schema="text-selection-byte-readback-v1", run=a.pin, source=a.source_pin,
        source_journals=source_journals, decisions=decisions, retained=report["retained"], output=output,
        successful_acquisition_demonstrated=False, positive_promotion_exercised=False,
        acceptance="author-checked only", serving=False))
    print(json.dumps(dict(report_sha256=file_sha(a.output), selected=selected, decisions=len(decisions))))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for n in ("run", "source-run", "output"):
        p.add_argument("--"+n, type=Path, required=True)
    for n in ("pin", "source-pin"):
        p.add_argument("--"+n, required=True)
    run(p.parse_args())
