"""Publish and physically audit real temporal prefixes; no execution of source commands."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from claims.tool_temporal_source import ToolSource, TemporalPopulation, build, write, CUTOFFS
from claims.codata_source import check, sha
from claims.episodic_memory import digest, canonical
from audit.tool_temporal_readback import readback
from senses.contract import SenseError
from senses.trajectory_population import open_bound


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--population", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--existing-pin")
    args = p.parse_args()
    started = time.perf_counter()
    source = ToolSource(args.raw)
    phases = dict(raw_source=time.perf_counter()-started)
    tick = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=False)
    pin = args.existing_pin or build(source, args.population)
    population = TemporalPopulation(args.population, source, pin)
    phases["publish_and_bind"] = time.perf_counter()-tick
    tick = time.perf_counter()
    controls = []
    def refuse(name, code, fn):
        try:
            fn()
        except (ValueError, SenseError) as e:
            assert getattr(e, "code", str(e)) == code, (name, str(e))
        else:
            raise AssertionError(name+" did not refuse")
        controls.append(dict(name=name, code=code))
    before = digest(source.episodes)
    ready = [e for e in source.episodes if e["status"] == "ready"]
    views = []
    for ep in ready:
        task = ep["task"]
        for cutoff in CUTOFFS:
            rows = population.prefix(task, cutoff, partition=ep["partition"], observed_through=cutoff)
            echo = population.echo(task, cutoff, receipt=source.receipt(task, cutoff+1))
            assert [r["turn"] for r in rows] == list(range(1, cutoff+1))
            assert all(r["task"] == task for r in rows) and echo not in rows and echo["turn"] == cutoff+1
            views.append(dict(task=task, cutoff=cutoff, references=digest(rows), echo=digest(echo)))
    ep = ready[0]
    task = ep["task"]
    refuse("future_as_cutoff", "MNLM_CODATA_TOOL_CUTOFF",
           lambda: population.prefix(task, 7, partition=ep["partition"], observed_through=7))
    refuse("unobserved_prefix", "MNLM_CODATA_TOOL_CUTOFF",
           lambda: population.prefix(task, 10, partition=ep["partition"], observed_through=6))
    refuse("wrong_partition", "MNLM_CODATA_TOOL_SPLIT",
           lambda: population.prefix(task, 6, partition="not-"+ep["partition"], observed_through=6))
    refuse("invented_time", "MNLM_CODATA_TOOL_OCCURRENCE_CLOCK_UNAVAILABLE", population.require_wall_clock)
    refuse("proposal_as_echo", "MNLM_CODATA_TOOL_ECHO_ROLE", lambda: source.receipt(task, 6))
    refuse("self_authored_receipt", "MNLM_CODATA_TOOL_ECHO_RECEIPT",
           lambda: population.echo(task, 6, receipt=dict(id="model-confidence")))
    other = ready[1]["task"]
    refuse("other_task_echo", "MNLM_CODATA_TOOL_ECHO_RECEIPT",
           lambda: population.echo(task, 6, receipt=source.receipt(other, 7)))
    gap = next(e for e in source.episodes if e["status"] != "ready")
    refuse("missing_prefix", "MNLM_CODATA_TOOL_TASK_UNAVAILABLE",
           lambda: population.prefix(gap["task"], 6, partition=gap["partition"], observed_through=6))
    bank = population.initial_bank()
    assert bank and all(r["turn"] in (3, 5) and r["partition"] == "train" for r in bank)
    corrupt = args.output/"corrupt-population"
    shutil.copytree(args.population, corrupt)
    raw = bytearray((corrupt/"references.jsonl").read_bytes())
    raw[0] ^= 1
    (corrupt/"references.jsonl").write_bytes(raw)
    refuse("corrupt_alias", "MNLM_CODATA_TOOL_POPULATION_BYTES",
           lambda: TemporalPopulation(corrupt, source, pin))
    refs = copy.deepcopy(population.refs)
    refs[0]["turn"] = 15
    (corrupt/"references.jsonl").write_bytes(b"".join(canonical(r)+b"\n" for r in refs))
    report = json.loads((corrupt/"report.json").read_bytes())
    report["files"]["references.jsonl"] = dict(bytes=(corrupt/"references.jsonl").stat().st_size,
        sha256=sha((corrupt/"references.jsonl").read_bytes()))
    write_path = corrupt/"report.json"
    write_path.write_bytes(canonical(report))
    forged_pin = sha(write_path.read_bytes())
    (corrupt/"COMMITTED").write_text(forged_pin, encoding="ascii")
    refuse("rehashed_turn_alias", "MNLM_CODATA_TOOL_POPULATION_SOURCE",
           lambda: TemporalPopulation(corrupt, source, forged_pin))
    # Same-sized byte corruption of a real shard, in the owned negative-control copy.
    shard = source.scope["shards"][0]
    target = args.output/"corrupt-source"/shard["path"]
    target.parent.mkdir(parents=True)
    shutil.copyfile(args.raw/shard["path"], target)
    with target.open("r+b") as f:
        f.seek(100)
        byte = f.read(1)
        f.seek(100)
        f.write(bytes([byte[0] ^ 1]))
    refuse("corrupt_raw_shard", "MNLM_UCR_INPUT_DIGEST",
           lambda: open_bound(args.output/"corrupt-source", shard))
    assert digest(source.episodes) == before
    phases["cutoff_and_corruption_controls"] = time.perf_counter()-tick
    tick = time.perf_counter()
    independent = readback(args.raw, args.population)
    phases["independent_raw_readback"] = time.perf_counter()-tick
    wall = time.perf_counter()-started
    check(sum(phases.values())/wall >= .9, "TOOL_PHASE_COVERAGE")
    result = dict(complete=True, acceptance="author-checked", source_pin=source.pin, population_pin=pin,
        independent=independent, controls=controls, views=views, initial_bank_references=len(bank),
        initial_bank_units=len({r["unit"] for r in bank}), source_unchanged=True,
        phases_seconds=phases, wall_seconds=wall)
    write(args.output/"report.json", result)
    print(json.dumps(dict(complete=True, population=pin, independent=independent,
        controls=len(controls), initial_bank_units=result["initial_bank_units"], wall_seconds=wall)))


if __name__ == "__main__":
    main()
