"""Real NIST editions -> scientific reference policy -> durable correction readback."""
import argparse
import copy
from decimal import Decimal
import json
from pathlib import Path
import re
import shutil
import sqlite3
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from claims.codata_source import CodataSource, SCOPE, check, sha
from claims.episodic_memory import canonical, digest, ZERO
from claims.scientific_memory import ScientificMemory


def write(path, value):
    with path.open("xb") as f:
        f.write(canonical(value))


def independent_source(source):
    # Separate parser: delimiter groups, not the production fixed-width slices.
    rows, results = {}, {}
    for year, raw in ((y, source.raw[y+".txt"]) for y in ("2018", "2022")):
        parsed = {}
        for lineno, line in enumerate(raw.decode("ascii").splitlines()[11:], 12):
            parts = re.split(r"\s{2,}", line.rstrip())
            assert len(parts) in (3, 4)
            name, value, uncertainty = parts[:3]
            unit = parts[3] if len(parts) == 4 else ""
            row = source.rows[year][name]
            assert row["raw_line"] == line and row["line"] == lineno
            assert row["unit"] == unit and row["value"]["display"] == value
            compact = "".join(value.split())
            assert row["value"]["compact"] == compact
            assert row["value"]["display_truncated"] == ("..." in compact)
            assert row["value"]["decimal"] == (None if "..." in compact else str(Decimal(compact)))
            assert row["exact_as_published"] == (uncertainty == "(exact)")
            if uncertainty != "(exact)":
                assert Decimal(row["uncertainty"]["decimal"]) == Decimal("".join(uncertainty.split()))
            parsed[name] = (compact, "".join(uncertainty.split()), unit)
        assert len(parsed) == len(source.rows[year])
        rows[year] = parsed
        results[year] = dict(rows=len(parsed), exact=sum(x[1] == "(exact)" for x in parsed.values()),
                             truncated=sum("..." in x[0] for x in parsed.values()))
    left, right = rows["2018"], rows["2022"]
    delta = dict(added=sorted(right.keys()-left.keys()), removed=sorted(left.keys()-right.keys()),
                 changed=sorted(k for k in left.keys() & right.keys() if left[k] != right[k]),
                 unchanged=sorted(k for k in left.keys() & right.keys() if left[k] == right[k]))
    assert delta == source.changes("2018", "2022")
    return dict(editions=results, changes={k: len(v) for k, v in delta.items()}, delta=delta)


def independent_ledger(path, source, expected):
    # Rebuild projection directly from persisted SQL, without Memory or ledger replay.
    states, episodes, head, edition = {}, {}, ZERO, None
    count, memory_count, memory_head = 0, 0, ZERO
    db = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    try:
        for seq, previous, raw, pin in db.execute("SELECT seq,previous,body,hash FROM events ORDER BY seq"):
            body = json.loads(raw)
            assert raw == canonical(body) and previous == head and seq == count
            count += 1
            assert pin == digest(dict(seq=seq, previous=previous, body=body))
            head = pin
            if body["kind"] == "adopt":
                assert body["prior"] == edition
                edition = body["edition"]
                ids = []
                for q in sorted(source.rows[edition]):
                    ep = source.episode(edition, q)
                    state = states[ep["id"]]
                    assert state["active"] and state["grounded_scope"] == SCOPE
                    ids.append(state["receipt"]["id"])
                assert body["receipt_set"] == digest(ids) and body["quantity_count"] == len(ids)
                continue
            record = body["record"]
            assert record["sequence"] == memory_count and record["previous"] == memory_head
            assert record["sha256"] == digest({k: v for k, v in record.items() if k != "sha256"})
            memory_head = record["sha256"]
            memory_count += 1
            e = record["event"]
            if e["op"] == "intake":
                ep = e["episode"]
                ref = ep["source"]
                row = source.rows[ref["edition"]][ref["quantity"]]
                assert ep["content"] == row["raw_line"] and ref["line"] == row["line"]
                assert ep["id"] == digest({k: v for k, v in ep.items() if k != "id"})
                assert ep["id"] not in episodes and all(x in episodes for x in ep["evidence"])
                episodes[ep["id"]] = ep
                states[ep["id"]] = dict(active=True, grounded_scope=None, review="unassessed", receipt=None)
                continue
            key, op = e["episode"], e["op"]
            state = states[key]
            if op == "check_source":
                assert episodes[key]["claim_kind"] == "source_statement"
                value = dict(checker="codata-publication-v1", episode=key, claim_kind="source_statement",
                             scope=SCOPE, source=episodes[key]["source"], world_validity="not_independently_replicated")
                assert e["result"] == dict(id=digest(value), **value)
                state.update(receipt=e["result"], review="source_checked")
            elif op == "promote":
                assert state["receipt"] and e["receipt"] == state["receipt"]["id"]
                state.update(grounded_scope=SCOPE, review="source_checked")
            else:
                assert op == "invalidate"
                affected = {key}
                while True:
                    more = affected | {k for k, v in episodes.items() if affected & set(v["evidence"])}
                    if more == affected:
                        break
                    affected = more
                assert e["affected"] == sorted(affected)
                for k in affected:
                    states[k].update(grounded_scope=None, review="needs_review", receipt=None)
        assert head == expected["head"] and edition == expected["adopted_edition"]
        assert states == expected["memory"]["states"] and seq+1 == expected["events"]
        assert memory_head == expected["memory"]["head"]
        return dict(events=seq+1, episodes=len(episodes), head=head, projection_exact=True)
    finally:
        db.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--source-pin", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cold", action="store_true")
    p.add_argument("--run-pin")
    args = p.parse_args()
    start = time.perf_counter()
    source = CodataSource(args.source, args.source_pin)
    parsed = independent_source(source)
    if args.cold:
        check(args.run_pin is not None and sha((args.output/"run.json").read_bytes()) == args.run_pin, "RUN_PIN")
        run = json.loads((args.output/"run.json").read_bytes())
        check(run["source"] == args.source_pin, "RUN_SOURCE")
        ledger = ScientificMemory(args.output/"science.sqlite3", source,
                                  expected_head=run["head"], expected_contract=run["contract"])
        expected = json.loads((args.output/"snapshot.json").read_bytes())
        check(ledger.snapshot() == expected, "COLD_SNAPSHOT")
        receipt = independent_ledger(args.output/"science.sqlite3", source, expected)
        queries = {q: ledger.resolve(q) for q in ("electron mass", "Planck constant", "atomic unit of action")}
        check(queries == json.loads((args.output/"queries.json").read_bytes()), "COLD_QUERY")
        ledger.close()
        write(args.output/"cold-readback.json", dict(complete=True, acceptance="author-checked",
              ledger=receipt, query_exact=True, wall_seconds=time.perf_counter()-start))
        print(json.dumps(dict(cold=True, complete=True)))
        return
    args.output.mkdir(parents=True, exist_ok=False)
    phases = {"source_and_independent_parse": time.perf_counter()-start}
    tick = time.perf_counter()
    path = args.output/"science.sqlite3"
    ledger = ScientificMemory(path, source, create=True)
    controls = []

    def refuse(label, code, fn):
        before = digest(ledger.snapshot())
        try:
            fn()
        except ValueError as exc:
            check(str(exc) == code, "WRONG_REFUSAL_"+label)
        else:
            raise AssertionError(label+" did not refuse")
        check(digest(ledger.snapshot()) == before, "REFUSAL_MUTATED")
        controls.append(dict(name=label, error=code, before=before, after=before))

    def promote(ep):
        ledger.apply_memory(dict(op="check_source", episode=ep["id"]))
        receipt = ledger.memory.state[ep["id"]]["receipt"]["id"]
        ledger.apply_memory(dict(op="promote", episode=ep["id"], receipt=receipt))

    refuse("no_edition", "MNLM_CODATA_NO_ADOPTED_EDITION", lambda: ledger.resolve("electron mass"))
    refuse("empty_adoption", "MNLM_CODATA_CURRENT_EVIDENCE_UNAVAILABLE", lambda: ledger.adopt("2018"))
    boundaries = {}
    for year in ("2018", "2022"):
        for q in sorted(source.rows[year]):
            ep = source.episode(year, q)
            ledger.apply_memory(dict(op="intake", episode=ep))
            promote(ep)
        event = ledger.adopt(year)
        boundaries[year] = dict(head=ledger.head, electron=ledger.resolve("electron mass"),
                                planck=ledger.resolve("Planck constant"), adoption=event)
    assert boundaries["2018"]["electron"]["row"]["value"] != boundaries["2022"]["electron"]["row"]["value"]
    assert boundaries["2018"]["planck"]["row"]["value"] == boundaries["2022"]["planck"]["row"]["value"]
    phases["intake_receipts_and_two_adoptions"] = time.perf_counter()-tick
    tick = time.perf_counter()
    ep = source.episode("2022", "electron mass")
    dependent = source.episode("2022", "electron mass", "unclassified", [ep["id"]])
    ledger.apply_memory(dict(op="intake", episode=dependent))
    stale = ScientificMemory(path, source, expected_head=ledger.head, expected_contract=ledger.contract)
    old_receipt = ledger.memory.state[ep["id"]]["receipt"]["id"]
    ledger.apply_memory(dict(op="invalidate", episode=ep["id"], reason="FSV withdrawal: recheck pinned reference"))
    refuse("withdrawn_current_no_old_fallback", "MNLM_CODATA_CURRENT_EVIDENCE_UNAVAILABLE",
           lambda: ledger.resolve("electron mass"))
    refuse("stale_process_cannot_serve", "MNLM_CODATA_STALE_LEDGER", lambda: stale.resolve("electron mass"))
    stale.close()
    refuse("old_receipt_after_withdrawal", "MNLM_MEMORY_NO_MATCHING_CHECKED_RECEIPT",
           lambda: ledger.apply_memory(dict(op="promote", episode=ep["id"], receipt=old_receipt)))
    assert ledger.memory.state[dependent["id"]]["review"] == "needs_review"
    assert ledger.resolve("electron mass", edition="2018")["edition"] == "2018"
    boundaries["withdrawal"] = dict(snapshot=ledger.snapshot(), historical=ledger.resolve("electron mass", edition="2018"))
    promote(ep)
    refuse("causality_not_in_table", "MNLM_CODATA_NO_WORLD_CHECKER",
           lambda: source.assess(source.episode("2022", "electron mass", "causal_explanation")))
    refuse("world_truth_from_authority", "MNLM_CODATA_NO_WORLD_CHECKER",
           lambda: ledger.resolve("electron mass", require_world_truth=True))
    refuse("ellipsis_is_not_exact_decimal", "MNLM_CODATA_TRUNCATED_DISPLAY",
           lambda: ledger.resolve("atomic unit of action", require_complete_decimal=True))
    refuse("downgrade", "MNLM_CODATA_ADOPTION_ORDER", lambda: ledger.adopt("2018"))
    bad = source.episode("2022", "electron mass")
    bad["content"] = bad["content"].replace("kg", "g")
    refuse("changed_unit", "MNLM_CODATA_EPISODE_CONTENT", lambda: ledger.apply_memory(dict(op="intake", episode=bad)))
    bad = source.episode("2022", "electron mass")
    bad["content"] = bad["content"].replace("0.000 000 0028", "0.000 000 0000")
    refuse("false_zero_uncertainty", "MNLM_CODATA_EPISODE_CONTENT",
           lambda: ledger.apply_memory(dict(op="intake", episode=bad)))
    bad = source.episode("2022", "electron mass")
    bad["source"]["edition"] = "2018"
    refuse("mislabelled_edition", "MNLM_CODATA_EPISODE_CONTENT",
           lambda: ledger.apply_memory(dict(op="intake", episode=bad)))
    other = source.episode("2022", "Planck constant")
    refuse("borrowed_receipt", "MNLM_MEMORY_NO_MATCHING_CHECKED_RECEIPT",
           lambda: ledger.apply_memory(dict(op="promote", episode=ep["id"],
                                            receipt=ledger.memory.state[other["id"]]["receipt"]["id"])))
    phases["withdrawal_reassessment_and_semantic_controls"] = time.perf_counter()-tick
    tick = time.perf_counter()
    snapshot = ledger.snapshot()
    queries = {q: ledger.resolve(q) for q in ("electron mass", "Planck constant", "atomic unit of action")}
    write(args.output/"snapshot.json", snapshot)
    write(args.output/"queries.json", queries)
    write(args.output/"boundaries.json", boundaries)
    write(args.output/"source-readback.json", parsed)
    write(args.output/"run.json", dict(head=ledger.head, contract=ledger.contract, source=args.source_pin))
    ledger.close()
    rebuilt = ScientificMemory(path, source, expected_head=snapshot["head"], expected_contract=snapshot["contract"])
    assert rebuilt.snapshot() == snapshot
    independent = independent_ledger(path, source, snapshot)
    ledger = rebuilt
    refuse("foreign_snapshot_pin", "MNLM_CODATA_SOURCE_PIN", lambda: CodataSource(args.source, "0"*64))
    refuse("foreign_head", "MNLM_CODATA_HEAD_PIN",
           lambda: ScientificMemory(path, source, expected_head="0"*64, expected_contract=ledger.contract))
    badsource = args.output/"changed-source"
    shutil.copytree(args.source, badsource)
    f = badsource/"2022.txt"
    f.write_bytes(f.read_bytes().replace(b"9.109 383 7139", b"9.109 383 7138", 1))
    refuse("changed_source_bytes", "MNLM_CODATA_SOURCE_BYTES", lambda: CodataSource(badsource, args.source_pin))
    incomplete = args.output/"incomplete-source"
    incomplete.mkdir()
    refuse("incomplete_source", "MNLM_CODATA_SOURCE_INCOMPLETE", lambda: CodataSource(incomplete, args.source_pin))
    truncated = args.output/"truncated.sqlite3"
    with sqlite3.connect(truncated) as db:
        ledger.db.backup(db)
        db.execute("DELETE FROM events WHERE seq=(SELECT MAX(seq) FROM events)")
        db.commit()
    refuse("ledger_truncation", "MNLM_CODATA_HEAD_PIN",
           lambda: ScientificMemory(truncated, source, expected_head=snapshot["head"], expected_contract=ledger.contract))
    # Fully rehash a forged history so this tests semantic replay, not just checksums.
    altered = args.output/"wrong-publication-date.sqlite3"
    with sqlite3.connect(altered) as db:
        ledger.db.backup(db)
        records = db.execute("SELECT seq,body FROM events ORDER BY seq").fetchall()
        head, changed = ZERO, False
        for seq, raw in records:
            body = json.loads(raw)
            if body["kind"] == "adopt" and not changed:
                body["publication_date"] = "2018-01-01"
                changed = True
            pin = digest(dict(seq=seq, previous=head, body=body))
            db.execute("UPDATE events SET previous=?,body=?,hash=? WHERE seq=?",
                       (head, canonical(body), pin, seq))
            head = pin
        db.commit()
    refuse("rehashed_wrong_publication_date", "MNLM_CODATA_ADOPTION_REPLAY",
           lambda: ScientificMemory(altered, source, expected_head=head, expected_contract=ledger.contract))
    ledger.close()
    phases["persisted_readback_and_corruption_controls"] = time.perf_counter()-tick
    report = dict(complete=True, acceptance="author-checked", measured=dict(
                  source_rows=sum(len(v) for v in source.rows.values()), changes=parsed["changes"],
                  ledger=independent, controls=len(controls)), controls=controls,
                  world_truth_established=False, useful_capability_gain_demonstrated=False, production_admission=False,
                  phase_seconds=phases, wall_seconds=time.perf_counter()-start,
                  files={n: sha((args.output/n).read_bytes()) for n in (
                      "snapshot.json", "queries.json", "boundaries.json", "source-readback.json", "run.json", "science.sqlite3")})
    write(args.output/"report.json", report)
    print(json.dumps(dict(complete=True, measured=report["measured"], wall_seconds=report["wall_seconds"])))


if __name__ == "__main__":
    main()
