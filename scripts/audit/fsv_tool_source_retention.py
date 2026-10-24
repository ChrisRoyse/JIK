"""Retain source-only constellation behavior when execution receipts are optional."""
import argparse
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.fsv_identity_loop import configure
from audit.tool_loop_readback import readback
from claims.codata_source import check, sha
from claims.episodic_memory import canonical, digest
from claims.tool_temporal_source import ToolSource
from lens.lenscommon import run_cli, take_lease
from model.tool_growth import GrowingToolLoop, parameters
from model.tool_kernel import ToolKernel
from model.tool_temporal_input import ToolInputs


def contents(path):
    with sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True) as db:
        check(db.execute("PRAGMA integrity_check").fetchall() == [("ok",)], "RETENTION_SQLITE")
        contract = json.loads(db.execute("SELECT contract FROM metadata").fetchone()[0])
        events = [json.loads(r[0]) for r in db.execute("SELECT body FROM events ORDER BY seq")]
        payloads = {pin: json.loads(raw) for pin, raw in db.execute("SELECT hash,data FROM payloads")}
    return contract, events, payloads


def neural(events, payloads):
    advances = [b for b in events if b["kind"] == "advance"]
    forecasts = [payloads[b["payload"]]["forecast"] for b in advances
                 if b.get("payload") and "forecast" in payloads[b["payload"]]]
    return dict(hidden=[b["hidden"] for b in advances],
                forecasts=[dict(region=f["region"], predictions=f["predictions"]) for f in forecasts])


def main():
    p = argparse.ArgumentParser()
    for name in ("raw", "population", "native", "parent", "kernel", "baseline", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    for name in ("population-pin", "native-pin", "parent-pin"):
        p.add_argument("--"+name, required=True)
    p.add_argument("--cold", action="store_true")
    args = p.parse_args()
    start = time.perf_counter()
    resident = subprocess.check_output(["nvidia-smi"]).decode()
    take_lease("28", "tool-source-retention")
    configure()
    source = ToolSource(args.raw)
    inputs = ToolInputs(source, args.population, args.native,
                        population_pin=args.population_pin, native_pin=args.native_pin)
    raw_parent = (args.parent/"run.json").read_bytes()
    check(sha(raw_parent) == args.parent_pin, "RETENTION_PARENT")
    parent = json.loads(raw_parent)
    routing = ToolKernel(inputs, args.kernel, (args.kernel/"COMMITTED").read_text())
    phases = dict(source_native=time.perf_counter()-start)
    def session(**kw):
        return GrowingToolLoop(args.output/"experience.sqlite3", source, inputs, args.parent/"initial",
            model_pin=parent["initial"], standard=parent["standard"], world=parent["world"],
            contexts={"train:1":"tonioo__sievelib-79"}, device="cuda", routing=routing,
            growth_root=args.output/"growth", **kw)
    tick = time.perf_counter()
    if args.cold:
        saved = json.loads((args.output/"report.json").read_bytes())
        loop = session(expected_head=saved["snapshot"]["head"],
                       expected_contract=saved["snapshot"]["contract"])
        check(loop.snapshot() == saved["snapshot"], "RETENTION_COLD")
        loop.close()
        direct = readback(args.output/"experience.sqlite3", inputs, saved["snapshot"])
        check(direct == saved["readback"], "RETENTION_COLD_READBACK")
        phases["replay_readback"] = time.perf_counter()-tick
        result = dict(complete=True, readback=direct, phases=phases,
                      wall_seconds=time.perf_counter()-start)
        destination = args.output/"cold.json"
    else:
        args.output.mkdir(exist_ok=False)
        (args.output/"gpu-before.txt").write_text(resident)
        loop = session(create=True)
        initial = parameters(loop.model)
        for exchange in range(5):
            for phase in range(5):
                loop.advance("train:1", receipt=source.receipt("tonioo__sievelib-79", 3+2*exchange)
                             if phase == 3 else None)
        snapshot = loop.snapshot()
        check("process_observations" not in snapshot and not snapshot["memory"]["grounded_world_claims"]
              and snapshot["memory"]["grounded_source_statements"], "RETENTION_SCOPE")
        before = digest(snapshot)
        try:
            loop.observe_execution(next(iter(loop.forecasts)), "unconfigured")
        except ValueError as error:
            check("TOOL_NO_EXECUTION_CHECKER" in str(error), "RETENTION_CONTROL")
        else:
            raise AssertionError("RETENTION_CONTROL_ACCEPTED")
        check(digest(loop.snapshot()) == before and parameters(loop.model) == initial, "RETENTION_STATE")
        loop.close()
        phases["source_only_experience"] = time.perf_counter()-tick
        tick = time.perf_counter()
        direct = readback(args.output/"experience.sqlite3", inputs, snapshot)
        contract, events, payloads = contents(args.output/"experience.sqlite3")
        other_contract, other_events, other_payloads = contents(args.baseline/"experience.sqlite3")
        baseline = json.loads((args.baseline/"snapshot.json").read_bytes())
        readback(args.baseline/"experience.sqlite3", inputs, baseline)
        check("process_observations" not in contract and "process_observations" in other_contract,
              "RETENTION_CONTRACT")
        a, b = neural(events, payloads), neural(other_events, other_payloads)
        check(a == b and len(a["hidden"]) == 25 and len(a["forecasts"]) == 2, "RETENTION_NEURAL")
        phases["independent_comparison"] = time.perf_counter()-tick
        result = dict(complete=True, snapshot=snapshot, readback=direct, phases=phases,
            identical_intrinsic_states=len(a["hidden"]), identical_constellation_forecasts=len(a["forecasts"]),
            instruments=len(inputs.specs), parameters_unchanged=initial,
            control=dict(code="TOOL_NO_EXECUTION_CHECKER", unchanged_snapshot=before),
            baseline_snapshot_sha256=sha((args.baseline/"snapshot.json").read_bytes()),
            wall_seconds=time.perf_counter()-start, scientific_admission=False, useful_capability=False)
        destination = args.output/"report.json"
    result["timing_coverage"] = sum(phases.values())/result["wall_seconds"]
    with destination.open("xb") as stream:
        stream.write(canonical(result))
    print(json.dumps(dict(complete=True, cold=args.cold, wall_seconds=result["wall_seconds"])))


if __name__ == "__main__":
    raise SystemExit(run_cli(main))
