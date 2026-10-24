"""Reassess the existing complete curriculum; no training or model execution."""
import argparse
import copy
import json
from pathlib import Path
import sqlite3
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.tool_growth_readback import readback
from audit.tool_retention_readback import inspect
from audit.tool_retention_controls import controls
from claims.codata_source import check, sha
from claims.episodic_memory import canonical, digest
from claims.tool_temporal_source import ToolSource, write
from model.tool_retention import compare, combine
from model.tool_temporal_input import ToolInputs


def main():
    parser = argparse.ArgumentParser()
    for name in ("raw", "population", "native", "parent", "run", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    for name in ("population-pin", "native-pin", "parent-pin", "run-pin"):
        parser.add_argument("--"+name, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    inputs = ToolInputs(ToolSource(args.raw), args.population, args.native,
                        population_pin=args.population_pin, native_pin=args.native_pin)
    phases = dict(source_native=time.perf_counter()-started)
    tick = time.perf_counter()
    existing = readback(inputs, args.run, args.parent, args.parent_pin, args.run_pin)
    check(existing["journal"]["retention_required"] is False, "RETENTION_EXPECTED_LEGACY")
    phases["independent_original_read"] = time.perf_counter()-tick
    tick = time.perf_counter()
    with sqlite3.connect((args.run/"experience.sqlite3").as_uri()+"?mode=ro", uri=True) as db:
        events = [json.loads(r[0]) for r in db.execute("SELECT body FROM events ORDER BY seq")]
        candidates = [v for v in events if v["kind"] == "candidate"]
        payloads = {e["payload"]: json.loads(db.execute(
            "SELECT data FROM payloads WHERE hash=?", (e["payload"],)).fetchone()[0]) for e in candidates}
    check(len(candidates) == 2, "RETENTION_CANDIDATE_ROSTER")
    instruments = {s.name: s.identity for s in inputs.specs}
    failures, results, history = [], [], []

    def refuse(name, code, fn):
        before = digest(payloads)
        try:
            fn()
        except ValueError as exc:
            check(str(exc) == "MNLM_CODATA_"+code, "RETENTION_CONTROL_"+name)
        else:
            raise AssertionError(name+" did not refuse")
        check(digest(payloads) == before, "RETENTION_CONTROL_MUTATION")
        failures.append(dict(name=name, code=code))

    for index, event in enumerate(candidates):
        payload = payloads[event["payload"]]
        previous, comparisons = None, {}
        if index:
            prior = history[-1]
            previous = dict(candidate=prior["candidate"], pin=prior["pin"], payload=prior["payload"],
                            scope="quarantined_learner_checks")
            saved = payloads[prior["payload"]]["candidate"]
            comparisons = compare(saved, payload["candidate"], prior["evaluate"], instruments)
            controls(saved, payload["candidate"], prior["evaluate"], instruments, refuse, "retention")
        result = combine(payload["result"], previous, comparisons)
        checked_body, checked_payload = dict(event, result=result), dict(payload, result=result)
        counts = inspect(checked_body, checked_payload, history, payloads, instruments)
        # This is a separately persisted reassessment, never an edit of the original journal.
        check(len(canonical(checked_body)) <= 65536, "RETENTION_EVENT_BUDGET")
        if index:
            forged = copy.deepcopy(checked_body)
            forged["result"]["retention"]["passed"] = True
            refuse("forged_retention_pass", "RETENTION_READ_COMPARISON",
                   lambda: inspect(forged, checked_payload, history, payloads, instruments))
            missing = copy.deepcopy(checked_body)
            missing["result"]["strict_gain_ceilings"] = {}
            check(result["strict_gain_ceilings"], "RETENTION_EXPECTED_ZERO_BASELINE")
            refuse("removed_zero_ceiling", "RETENTION_READ_CEILING",
                   lambda: inspect(missing, checked_payload, history, payloads, instruments))
        results.append(dict(candidate=event["candidate"], prior_decision=digest(event),
            source_payload=event["payload"], result=result, independent=counts,
            event_bytes_before=len(canonical(event)), event_bytes_after=len(canonical(checked_body))))
        history.append(event)
    phases["reassessment_controls"] = time.perf_counter()-tick
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output/"reassessment.json", results)
    wall = time.perf_counter()-started
    check(sum(phases.values())/wall >= .9, "RETENTION_TIMING")
    write(args.output/"report.json", dict(complete=True, acceptance="author-checked",
        original_run=args.run_pin, original_readback=existing, controls=failures, phases=phases,
        timing_coverage=sum(phases.values())/wall, wall_seconds=wall,
        reassessment_pin=sha((args.output/"reassessment.json").read_bytes()),
        production_gate_changed_only=True, model_execution=False, scientific_admission=False))
    print(json.dumps(dict(complete=True, controls=len(failures),
                         readback=[r["independent"] for r in results], wall_seconds=wall)))


if __name__ == "__main__":
    main()
