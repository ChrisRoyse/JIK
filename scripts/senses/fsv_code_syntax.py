"""Code bytes -> typed syntax -> unchanged structure encoder, on retained spans.

Engineering proof only. It does not commission a lens, a grammar for other
languages, an execution trace, resolved call targets or code understanding.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import sqlite3
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from senses.adapters import atom, prepare
from senses.code_syntax import SyntaxSource, MAX_BYTES, MAX_NODES, contract, syntax_fields
from senses.contract import SenseError, canonical
from senses.instruments import measure, roster, slot_digest, tokens
from senses.pipeline import file_hash, ingest_file, read_bound, write_records
from senses.store import Store


def status(items):
    return next(x["value"] for x in items if x["name"] == "syntax.status")


def stream_boundary(records, store_path, head, out, check):
    """Use the production writer/reader with an explicitly engineering-only plan."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lens"))
    import numpy as np
    from universal import specs
    from production.input_plan import code_hashes, environment, POLICY_SHA256
    from production.stream import run, paths
    from production.publication import Publication
    selected = records[:4]  # first four retained code atoms, before examining slots
    rows_dir = out / "rows"
    rows_dir.mkdir()
    policy = {"schema": "ucr-view-v1", "question": "syntax-boundary", "allow_channels": ["structure"],
              "deny_fields": [], "deny_relations": [], "blocked_units": [], "label_aliases": [],
              "rationale": "Native syntax transport only; no label, fit or admission."}
    policy_path = out / "policies.json"
    policy_path.write_bytes(canonical({"syntax-boundary": policy}))
    rows = [{**r["locator"], "atom": r["atom_id"], "i": i,
             "text": r["channels"]["surface"]["items"][0]["text"]} for i, r in enumerate(selected)]
    row_path = rows_dir / "syntax-0.jsonl"
    row_path.write_bytes(b"".join(canonical(r) + b"\n" for r in rows))
    spec = next(s for s in specs() if s["channel"] == "structure")
    name = spec["name"]
    plan = {"schema": "tier-input-plan-v1", "purpose": "engineering_boundary", "storage": "raw-f32",
            "training_scope_sha256": POLICY_SHA256, "generation": head, "policies_sha256": file_hash(policy_path),
            "code_sha256": code_hashes(), "environment": environment(), "batch_rows": 2,
            "slot_budget_bytes": 1 << 20, "metadata_budget_bytes": 1 << 20,
            "blob_budget_bytes": 4 << 20, "work_budget_bytes": 32 << 20,
            "instruments": {name: {"spec": spec, "weights_sha256": spec["weights_sha256"],
                                   "tier": "deterministic", "slot_budget_bytes": 1 << 20}},
            "shards": {"syntax-0": {"n": len(rows), "rows_sha256": file_hash(row_path),
                                     "rows_bytes": row_path.stat().st_size, "family": "syntax-boundary",
                                     "tier": "deterministic", "instruments": [name]}}}
    (out / "stream-plan.json").write_bytes(canonical(plan))
    first = run(plan, store_path, policy_path, rows_dir, out / "stream-job")
    second = run(plan, store_path, policy_path, rows_dir, out / "stream-job")
    check("stream_exact_restart", first["after"] == second["after"] and second["checkpoints_reused"] == 1
          and second["loaded_instruments"] == [])
    slot, _ = paths(out / "stream-job", "syntax-0", name)
    with np.load(slot, allow_pickle=False) as z:
        check("native_stream_preserves_all_syntax_rows", all(np.array_equal(
            z["tokens"][z["offsets"][i]:z["offsets"][i + 1]], measure(r, "structure")["tokens"])
            for i, r in enumerate(selected)))
    pub = Publication(out / "stream-job" / "publication")
    try:
        body, audit = pub.read("syntax-0")
    finally:
        pub.close()
    (out / "stream-readback.json").write_bytes(canonical({"first": first, "restart": second, "body": body, "audit": audit}))
    check("committed_stream_cold_readable", bool(audit))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--sha256", required=True)
    ap.add_argument("--spans", required=True, type=Path)
    ap.add_argument("--spans-sha256", required=True)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    started, checks = time.monotonic(), []

    def check(name, ok, **detail):
        checks.append({"name": name, "pass": bool(ok), **detail})
        (args.out / "checks.json").write_bytes(canonical(checks))
        if not ok:
            raise AssertionError(name)

    def refuses(name, fn, code):
        try:
            fn()
        except SenseError as exc:
            check(name, exc.code == code, code=exc.code)
        else:
            check(name, False)

    check("bound_input", file_hash(args.input) == args.sha256)
    check("bound_retained_spans", file_hash(args.spans) == args.spans_sha256)
    envelopes = list(read_bound(args.input, args.sha256))
    sp = json.loads(args.spans.read_bytes())
    source, shard = sp["scope"]["source"], sp["scope"]["shard"]
    check("retained_scope", source == "opencodeinstruct" and sp["schema"] == "retained-code-span-proof-v1"
          and len(envelopes) == 64 and not sp["parent_manifest"].get("error"))
    env = {e["record"]["row"]: e for e in envelopes}
    check("unique_source_rows", len(env) == len(envelopes)
          and all(e["source"] == source and e["shard"] == shard for e in envelopes))
    panel_before = roster()["panel_digest"]
    structure_id = next(c["lens_id"] for c in roster()["instruments"] if c["channel"] == "structure")
    prepared = {row: prepare(e) for row, e in env.items()}
    records, boundaries = [], []
    for span in sp["spans"]:
        if span["role"] != 6:
            continue
        row, start, end = (span[k] for k in ("row", "start", "end"))
        r = atom(prepared[row], start, end, span["role"])
        fs = r["channels"]["structure"]["items"]
        records.append(r)
        boundaries.append({"atom": r["atom_id"], "row": row, "span": [start, end], "status": status(fs),
                           "nodes": next((x["value"] for x in fs if x["name"] == "syntax.nodes"), 0)})
    check("nonempty_actual_code", bool(records) and len(records) == sum(s["role"] == 6 for s in sp["spans"]))
    check("exact_surface_retention", all(r["channels"]["surface"]["items"][0]["text"] ==
          env[r["locator"]["row"]]["record"]["text"][slice(*r["locator"]["span"])] for r in records))
    counts = Counter(b["status"] for b in boundaries)
    check("parsed_and_missing_grammar_represented", counts["parsed"] > 0 and counts["MNLM_SYNTAX_LANGUAGE_UNSUPPORTED"] > 0)
    check("no_partial_ast_on_failure", all(b["nodes"] == 0 for b in boundaries if b["status"] != "parsed"))
    # Change the external answer sheet. Its values must not alter any syntax slot.
    changed = copy.deepcopy(envelopes[0])
    for key in ("unit_tests", "average_test_score", "tests_execution_status", "llm_judgement"):
        changed["record"]["f"][key] = "ORACLE_CONTROL_SHOULD_NOT_ENTER_SYNTAX"
    before = [r for r in records if r["locator"]["row"] == changed["record"]["row"]]
    after_p = prepare(changed)
    check("oracle_mutation_not_sensed", bool(before) and all(
        slot_digest(measure(r, "structure")) == slot_digest(measure(atom(after_p, *r["locator"]["span"], 6), "structure"))
        for r in before))
    # Renaming the source/adapter changes lineage, never the structural readings.
    renamed = copy.deepcopy(envelopes[0])
    renamed["source"] = renamed["source_manifest"]["source"] = "syntax-adapter-control"
    renamed_p = prepare(renamed, {"contract": "ucr-adapter-v1", "source": renamed["source"], "fields": {}})
    check("source_neutral_encoder", all(slot_digest(measure(r, "structure")) ==
          slot_digest(measure(atom(renamed_p, *r["locator"]["span"], 6), "structure")) for r in before))
    # Negative/contrast controls never join the retained proof population.
    variants = ["def f(x, y):\n    return x + y\n", "def f(x, y):\n    return x - y\n",
                "def f(x, y):\n    return not (x < y)\n", "def f(x, y):\n    return x < y\n"]
    slots = [tokens(syntax_fields(t, "python"), "structure").tobytes() for t in variants]
    check("operator_and_negation_distinct", slots[0] != slots[1] and slots[2] != slots[3])
    original = syntax_fields(variants[0], "python")
    check("syntax_replay_exact", canonical(original) == canonical(syntax_fields(variants[0], "python")))
    check("malformed_python_named", status(syntax_fields("def broken(:", "python")) == "MNLM_SYNTAX_UNPARSED")
    check("language_not_inferred", status(syntax_fields(variants[0], "javascript")) == "MNLM_SYNTAX_LANGUAGE_UNSUPPORTED")
    check("plain_text_not_promoted_to_code", SyntaxSource(variants[0]).for_span(0, len(variants[0])) == [])
    check("byte_budget_no_partial_tree", status(syntax_fields("#" * (MAX_BYTES + 1), "python")) == "MNLM_SYNTAX_BYTE_BUDGET")
    too_many = syntax_fields("x=1\n" * MAX_NODES, "python")
    check("node_budget_no_partial_tree", status(too_many) == "MNLM_SYNTAX_NODE_BUDGET"
          and not any(x["name"] == "syntax.node" for x in too_many))
    local = "```python\nx=1\n```\n```python\ny=2\n```\n"
    altered = local.replace("y=2", "z=3")
    check("sibling_content_not_parsed", canonical(SyntaxSource(local).for_span(10, 14)) ==
          canonical(SyntaxSource(altered).for_span(10, 14)))
    refuses("invalid_span_refuses", lambda: SyntaxSource(local).for_span(0, len(local) + 1), "MNLM_SYNTAX_SPAN")
    unicode = syntax_fields("é = 'é'; z = é + 1\n", "python")
    check("utf8_coordinates", any(x["name"] == "syntax.end" and x["value"] == 9 for x in unicode))
    marker = args.out / "must-not-execute"
    evil = "from pathlib import Path\nPath(" + repr(str(marker)) + ").write_text('executed')\n"
    check("no_code_execution", status(syntax_fields(evil, "python")) == "parsed" and not marker.exists())
    exported = args.out / "code.ucr.jsonl"
    binding = {"kind": "retained-stage2-code-spans", "input_sha256": args.sha256,
               "span_proof_sha256": args.spans_sha256, "scope": sp["scope"], "syntax": contract(),
               "retention_owner": "MNLM issue #15; preserve source and readback evidence"}
    manifest = write_records(exported, records, binding)
    store_path = args.out / "syntax.sqlite"
    with_store = Store(store_path, create=True)
    try:
        ingest_file(with_store, exported, manifest, None)
        head = with_store.head
        audit = with_store.audit()
        check("real_store_readback", audit["records"] == len(records))
        digests = {r["atom_id"]: slot_digest(with_store.read_slot(r["atom_id"], structure_id)) for r in records}
    finally:
        with_store.close()
    cold = Store(store_path)
    try:
        check("cold_generation_unchanged", cold.head == head)
        check("cold_source_and_slot_replay", all(canonical(cold.get(r["atom_id"])) == canonical(r)
              and slot_digest(cold.read_slot(r["atom_id"], structure_id)) == digests[r["atom_id"]]
              == slot_digest(measure(r, "structure")) for r in records))
    finally:
        cold.close()
    check("frozen_encoder_unchanged", roster()["panel_digest"] == panel_before)
    check("no_admission_claim", roster()["admitted"] == [] and roster()["served"] is False)
    stream_boundary(records, store_path, head, args.out, check)
    # These two databases belong only to this completed proof run. Checkpoint
    # them into self-contained archives before sealing; a read-only WAL reader
    # can create/remove -shm files, which are not durable evidence. No live
    # source/publication database or production journaling setting is changed.
    for path in (store_path, args.out / "stream-job" / "publication" / "index.sqlite"):
        db = sqlite3.connect(path.resolve().as_uri() + "?mode=rw", uri=True)
        try:
            checkpoint = list(db.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone())
            mode = db.execute("PRAGMA journal_mode=DELETE").fetchone()[0]
        finally:
            db.close()
        check("self_contained_archive_" + path.name, checkpoint == [0, 0, 0] and mode == "delete"
              and not Path(str(path) + "-wal").exists() and not Path(str(path) + "-shm").exists(), checkpoint=checkpoint)
    # Serialize boundaries and outputs for a separate, producer-independent reader.
    (args.out / "boundaries.json").write_bytes(canonical(boundaries))
    (args.out / "slot-digests.json").write_bytes(canonical(digests))
    (args.out / "store-audit.json").write_bytes(canonical(audit))
    result = {"schema": "code-syntax-fsv-v1", "acceptance": "author-checked only", "checks": checks,
              "input": str(args.input), "input_sha256": args.sha256, "span_proof": str(args.spans),
              "span_proof_sha256": args.spans_sha256, "syntax": contract(), "generation": head,
              "records": len(records), "status_counts": dict(counts), "nodes": sum(b["nodes"] for b in boundaries),
              "encoder_panel_sha256": panel_before, "wall_seconds": time.monotonic() - started,
              "meaning_asserted": False, "stage3_admitted": False, "stage4_production": False}
    (args.out / "report.json").write_bytes(canonical(result))
    inventory = {p.relative_to(args.out).as_posix(): file_hash(p) for p in sorted(args.out.rglob("*")) if p.is_file()}
    (args.out / "inventory.json").write_bytes(canonical(inventory))
    print(json.dumps({"checks": len(checks), "records": len(records), "nodes": result["nodes"],
                      "statuses": dict(counts), "report_sha256": file_hash(args.out / "report.json")}))


if __name__ == "__main__":
    main()
