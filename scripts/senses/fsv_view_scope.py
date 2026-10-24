"""FSV: retired sources remain auditable but cannot enter measurement views.

Historical-store controls are explicitly synthetic and never training data.
One optional named real source store is cold-audited without mutation.
"""
import argparse
import copy
import json
from pathlib import Path
import sqlite3
import sys
import time

sys.path[:0] = [str(Path(__file__).resolve().parents[1]),
               str(Path(__file__).resolve().parents[1] / "lens")]
from senses.contract import CHANNELS, SenseError, absent, canonical, digest, field, present, seal_record
from senses.instruments import measure, slot_digest
from senses.pipeline import file_hash
from senses.scope import POLICY
from senses.store import Store
from senses.views import view
from universal import ChannelReader
from training_scope import POLICY_SHA256


def observation(index, source, day, unit=None, key="scope:shared"):
    text = f"Known-truth observation {index}; not training data."
    r = {"schema": "ucr-v1", "locator": {"source": source, "shard": "known-truth",
         "row": index, "span": [0, len(text)], "record_digest": digest([index]),
         "manifest_digest": digest("synthetic historical source")},
         "unit": unit or f"unit-{index}",
         "referents": [{"key": key, "method": "source_identifier", "confidence": 1,
                        "evidence": "explicit synthetic FSV control"}],
         "channels": {c: absent("control_not_supplied") for c in CHANNELS}}
    r["channels"]["surface"] = present([{"text": text}])
    r["channels"]["fields"] = present([field("observed", f"2026-01-0{day}T00:00:00Z",
                                                "clock", "UTC", "observed")])
    return seal_record(r)


def run(out, real_store=None):
    start = time.perf_counter()
    out.mkdir(parents=True, exist_ok=False)
    checks, phases = [], {}

    def check(name, ok, **details):
        checks.append({"boundary": name, "pass": bool(ok), **details})

    def refuses(name, call):
        before = file_hash(dbpath)
        code = None
        try:
            call()
        except SenseError as exc:
            code = exc.code
        after = file_hash(dbpath)
        check(name, code == "MNLM_SOURCE_EXCLUDED" and before == after,
              code=code, before=before, after=after)

    retired = "cuyahoga-eighth-district"
    records = [observation(0, "scope-control-a", 2), observation(1, "scope-control-b", 1),
               observation(2, "scope-control-b", 1, unit="unit-0"),
               observation(3, "scope-control-b", 1, key="scope:unrelated"),
               observation(4, retired, 1), observation(5, retired, 3),
               observation(6, "scope-control-c", 1)]
    (out / "controls.jsonl").write_bytes(b"".join(canonical(r) + b"\n" for r in records))
    policy = {"schema": "ucr-view-v1", "question": "scope-control", "allow_channels": list(CHANNELS),
              "deny_fields": [], "deny_relations": [], "blocked_units": [], "label_aliases": [],
              "rationale": "historical-source exclusion control; no scientific admission"}
    pp = out / "policies.json"
    pp.write_bytes(canonical({policy["question"]: policy}))
    dbpath = out / "historical-control.sqlite"
    store = Store(dbpath, create=True, records_only=True)

    def ingest(rows):
        return store.ingest(rows, {"count": len(rows), "input_sha256": digest(rows),
                                  "kind": "synthetic-known-truth historical scope control"},
                            store.head)["generation"]

    first = ingest(records[:4])
    # Simulate an actual pre-retirement generation. This in-memory fixture-only
    # policy change does not edit config or weaken any production read path.
    reason = POLICY["excluded_sources"].pop(retired)
    try:
        historical = ingest(records[4:6])
    finally:
        POLICY["excluded_sources"][retired] = reason
    latest = ingest(records[6:])
    store.close()
    phases["create_historical_controls"] = time.perf_counter() - start
    t = time.perf_counter()
    before = file_hash(dbpath)
    store = Store(dbpath)
    cold = store.audit()
    check("historical_raw_audit_remains_readable", cold["records"] == 7 and cold["head"] == latest)
    raw = sqlite3.connect(dbpath.resolve().as_uri() + "?mode=ro", uri=True)
    check("physical_historical_rows", raw.execute("SELECT count(*) FROM records WHERE source=?",
                                                  (retired,)).fetchone()[0] == 2)
    raw.close()
    target = store.get(records[0]["atom_id"], first)
    _, contexts, binding = view(store, target, policy, first)
    expected = {records[1]["atom_id"]}
    for channel in ("history", "co_reference"):
        check(f"old_generation_{channel}_sibling_and_foreign_key_controls",
              {r["locator"]["row"] for r in contexts[channel]} == {1})
        check(f"physical_{channel}_join", {r["atom_id"] for r in store.context(target, channel, first)} == expected)
        scoped = {**policy, "allow_channels": ["surface", "fields", channel]}
        refuses(f"retired_{channel}_cannot_enter_view", lambda p=scoped: view(store, target, p, historical))
    refuses("retired_target_direct_view", lambda: view(store, records[4], policy, historical))
    reader = ChannelReader(dbpath, pp, historical)
    row = {**target["locator"], "ucr_span": target["locator"]["span"],
           "text": target["channels"]["surface"]["items"][0]["text"]}
    refuses("real_channel_reader_rejects_retired_dependency", lambda: reader.read(row, policy))
    reader.close()
    limited = {**policy, "blocked_units": [records[4]["unit"]]}
    visible, ctx, bound = view(store, target, limited, historical)
    for channel in ("history", "co_reference"):
        check(f"blocked_and_future_sources_not_read_{channel}", [r["locator"]["row"] for r in ctx[channel]] == [1])
        slot = measure(visible, channel, ctx[channel])
        repeat = measure(visible, channel, ctx[channel])
        check(f"allowed_{channel}_native_slot_replay", slot["kind"] == "Multi"
              and slot_digest(slot) == slot_digest(repeat)
              and slot["dependencies"] == [r["atom_id"] for r in ctx[channel]])
    view(store, target, {**policy, "allow_channels": ["surface"]}, historical)
    check("excluded_context_channel_not_consumed", True)
    check("view_binds_current_training_policy", binding.get("training_scope_sha256") == POLICY_SHA256
          and bound.get("training_scope_sha256") == POLICY_SHA256)
    check("old_generation_still_replays_after_append", view(store, target, policy, first)[2] == binding)
    store.close()
    check("all_control_reads_preserve_database_bytes", before == file_hash(dbpath), before=before, after=file_hash(dbpath))
    phases["read_boundaries"] = time.perf_counter() - t
    if real_store:
        t = time.perf_counter()
        old = file_hash(real_store)
        real = Store(real_store)
        census = real.audit()
        real.close()
        check("real_approved_store_cold_read_unchanged", file_hash(real_store) == old
              and census["records_only"] and census["slots"] == 0
              and retired not in census["sources"], before=old, after=file_hash(real_store), census=census)
        phases["real_store_readback"] = time.perf_counter() - t
    elapsed = time.perf_counter() - start
    report = {"acceptance": "author-checked", "claim": "source-policy enforcement at measurement views",
              "synthetic_controls_are_training_data": False, "scientific_admission": False,
              "checks": checks, "passed": sum(c["pass"] for c in checks),
              "failed": sum(not c["pass"] for c in checks), "seconds": elapsed,
              "phases": phases, "phase_coverage": sum(phases.values()) / elapsed,
              "code": {p: file_hash(Path(__file__).parent / p) for p in
                       ("views.py", "store.py", "scope.py", "fsv_view_scope.py")}}
    (out / "report.json").write_bytes(canonical(report) + b"\n")
    files = {p.name: {"sha256": file_hash(p), "bytes": p.stat().st_size}
             for p in sorted(out.iterdir()) if p.is_file()}
    (out / "SHA256.json").write_bytes(canonical(files) + b"\n")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--real-store", type=Path)
    args = ap.parse_args()
    result = run(args.output, args.real_store)
    print(json.dumps({k: result[k] for k in ("passed", "failed", "seconds", "phase_coverage")}))
    sys.exit(bool(result["failed"]))
