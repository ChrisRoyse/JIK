"""Real-input UCR acceptance harness, with two whole-source holdouts and controls.

Evidence is author-checked. This measures adapter transfer and observation retention;
it does not relabel them model learning, certified information or Stage 3 acceptance.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from senses.adapters import atom, prepare
from senses.contract import SenseError, absent, canonical, digest, field, present, seal_record
from senses.instruments import contracts, coverage, measure, roster, slot_digest
from senses.pipeline import file_hash, ingest_file, read_bound, write_records
from senses.store import Store
from senses.views import masked, view


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    source, dest = Path(args.input), Path(args.output)
    dest.mkdir(parents=True, exist_ok=False)
    capture = json.loads((source / "capture.json").read_text())
    from senses.scope import allowed
    capture = [s for s in capture if allowed(s["source"])]
    checks = []

    def check(name, value, **evidence):
        checks.append({"name": name, "pass": bool(value), **evidence})
        if not value:
            raise AssertionError(name)

    def refuses(name, fn, code=None):
        try:
            fn()
        except (SenseError, sqlite3.IntegrityError) as exc:
            got = getattr(exc, "code", str(exc))
            check(name, code is None or code == got, code=got)
            return
        check(name, False)

    sources = {}
    before = roster()["panel_digest"]
    for item in capture:
        name = item["source"]
        rows = [atom(prepare(e)) for e in read_bound(source / (name + ".jsonl"), item["sha256"])]
        check("input_count_" + name, len(rows) == item["n"])
        for r in rows:
            check("oracle_fields_excluded", not ({x["name"] for x in r["channels"]["fields"]["items"]} & {"unit_tests", "anchors", "average_test_score", "tests_execution_status"}))
        path = dest / (name + ".ucr.jsonl")
        manifest = write_records(path, rows, {"capture": item, "scope": "64-record prefix; not whole-corpus coverage"})
        sources[name] = (path, manifest, rows)
    store = Store(dest / "channels.sqlite", create=True)
    # Neither holdout source is used by the base ingestion. Same instruments then
    # consume both; no fit/admission/model is silently performed during transfer.
    holdouts = ["gdelt-events-90d", "opencodeinstruct"]
    stages, snapshots = [], []
    for phase, names in [("base", [n for n in sources if n not in holdouts]), *[("holdout_" + n, [n]) for n in holdouts]]:
        old = {a: body for a, body in store.db.execute("SELECT atom,body FROM records")}
        oldslots = dict(((a, l), h) for a, l, h in store.db.execute("SELECT atom,lens,digest FROM slots"))
        start = time.monotonic()
        for name in names:
            path, manifest, _ = sources[name]
            ingest_file(store, path, manifest, store.head)
        check(phase + "_panel_unchanged", before == roster()["panel_digest"])
        check(phase + "_old_observations_retained", all(canonical(store.get(a)) == b for a, b in old.items()))
        check(phase + "_old_slots_retained", all(slot_digest(store.read_slot(a, l)) == h for (a, l), h in oldslots.items()))
        audit = store.audit()
        (dest / (phase + ".json")).write_bytes(canonical(audit))
        stages.append({"phase": phase, "sources_added": names, "retained_records": len(old), "retained_slots": len(oldslots),
                       "new_lenses": 0, "seconds": time.monotonic() - start, "audit": audit})
        snapshots.append(store.head)
        print(json.dumps({"phase": phase, "records": audit["records"], "seconds": stages[-1]["seconds"]}), flush=True)
    sample = sources["wikidata5m"][2][0]
    check("exact_locator_lookup", store.find(**{k: sample["locator"][k] for k in ("source", "shard", "row", "span")})["atom_id"] == sample["atom_id"])
    head = store.head
    altered = copy.deepcopy(sample)
    altered["channels"]["fields"] = present([field("new", 1, "integer")])
    seal_record(altered)
    refuses("conflicting_source_identity", lambda: store.ingest([altered], {"input_sha256": digest(altered), "count": 1}, head), "MNLM_UCR_IDENTITY_CONFLICT")
    check("conflict_preserves_head", store.head == head)
    refuses("stale_writer", lambda: store.ingest([], {"input_sha256": "negative-control", "count": 0}, snapshots[0]), "MNLM_UCR_STALE_HEAD")
    wrong = copy.deepcopy(sources["wikidata5m"][1])
    wrong["sha256"] = "0" * 64
    refuses("changed_input_rollback", lambda: ingest_file(store, sources["wikidata5m"][0], wrong, head), "MNLM_UCR_INPUT_DIGEST")
    check("failed_batch_preserves_head", store.head == head)
    refuses("record_mutation", lambda: store.db.execute("UPDATE records SET body=? WHERE atom=?", (b"{}", sample["atom_id"])))
    store.db.rollback()
    bad = copy.deepcopy(sample)
    bad["channels"]["fields"] = present([field("value", float("nan"), "number")])
    refuses("nonfinite_refused", lambda: seal_record(bad), "MNLM_UCR_VALUE")
    allslots = {c["name"]: measure(sample, c["channel"]) for c in contracts()}
    check("available_is_not_admitted", coverage(allslots)["d4"] is False)
    refuses("forged_admission", lambda: coverage(allslots, list(allslots)), "MNLM_SHANNON_CONDITIONAL_UNMEASURED")
    # Controls are deliberately constructed, clearly separate from real corpus metrics.
    from senses.fsv_controls import run_controls
    run_controls(dest, sample, check, refuses)
    store.close()
    cold = Store(dest / "channels.sqlite")
    audit = cold.audit()
    cold.close()
    check("cold_reopen_head", audit["head"] == head)
    output = {"acceptance": "author-checked", "metric_kind": "measured adapter transfer and observation retention; not model continual learning",
              "scope": {"sources": len(sources), "records": sum(v[1]["count"] for v in sources.values()), "capture_sha256": file_hash(source / "capture.json")},
              "stages": stages, "checks": checks, "passed": sum(c["pass"] for c in checks), "failed": sum(not c["pass"] for c in checks),
              "panel": roster(), "cold_readback": audit, "remaining_gate": "MNLM_SHANNON_CONDITIONAL_UNMEASURED"}
    output["code_sha256"] = {p.name: file_hash(p) for p in Path(__file__).parent.glob("*.py")}
    (dest / "report.json").write_bytes(canonical(output) + b"\n")
    manifest = {p.name: file_hash(p) for p in sorted(dest.iterdir()) if p.is_file() and not p.name.endswith(("-wal", "-shm"))}
    (dest / "SHA256.json").write_bytes(canonical(manifest) + b"\n")
    print(json.dumps({"report": str(dest / "report.json"), "passed": output["passed"], "failed": output["failed"]}))


if __name__ == "__main__":
    main()
