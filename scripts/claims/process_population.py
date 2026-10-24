"""Exact captured output surfaces; empty streams and probe lineage stay explicit."""
import argparse
import json
from pathlib import Path
import sys
import time

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from claims.codata_source import check, sha
from claims.episodic_memory import canonical, digest
from claims.process_observation import ProcessObservations
from claims.tool_temporal_source import ToolSource, TemporalPopulation, write

FILES = ("process/chunks/atoms.parquet", "process/summary.json", "references.jsonl", "probe-source.json")


def expected(observations, source_population):
    rows, refs = [], []
    for key, entry in sorted(observations.entries.items()):
        observed = observations.read(key)
        for stream in ("stdout", "stderr"):
            text = observed["captured"][stream]
            unit = len(rows) if text else None
            pin = sha(text.encode())
            if text:
                rows.append(dict(row=unit, chunk=0, entity=pin, text=text, start=0, end=len(text)))
            refs.append(dict(observation=key, stream=stream, unit=unit, bytes=len(text.encode()),
                             text_sha256=pin, target=entry["target"], source_set=observations.pin))
    # Existing real source, used only for the unchanged measurement probe.
    probe = next((r for r in source_population.atoms if 1800 <= len(r["text"]) <= 3000), None)
    check(probe is not None, "PROCESS_PROBE_MISSING")
    lineage = dict(population=source_population.pin, atom=probe, unit=len(rows),
                   purpose="overflow_and_determinism_only_excluded_from_feedback")
    rows.append(dict(probe, row=len(rows)))
    return rows, refs, lineage


def build(observations, source_population, root):
    start = time.perf_counter()
    rows, refs, probe = expected(observations, source_population)
    check(not root.exists(), "PROCESS_POPULATION_EXISTS")
    (root/"process/chunks").mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows), root/FILES[0])
    write(root/FILES[1], dict(chunks=len(rows), chunks_sha256=sha((root/FILES[0]).read_bytes())))
    with (root/FILES[2]).open("xb") as f:
        for ref in refs:
            f.write(canonical(ref)+b"\n")
    write(root/FILES[3], probe)
    report = dict(schema="process-output-population-v1", observations=observations.pin,
        source_population=source_population.pin, complete=True, errors=[], calibration_only=True,
        scientific_admission=False, scope="captured_output_surfaces_only",
        families=dict(process=dict(ready=True, summary_sha256=sha((root/FILES[1]).read_bytes()))),
        inputs=len(rows), nonempty_streams=sum(r["unit"] is not None for r in refs),
        utf8_bytes=sum(len(r["text"].encode()) for r in rows),
        files={n:dict(bytes=(root/n).stat().st_size, sha256=sha((root/n).read_bytes())) for n in FILES},
        wall_seconds=time.perf_counter()-start)
    write(root/"report.json", report)
    pin = sha((root/"report.json").read_bytes())
    (root/"COMMITTED").write_text(pin, encoding="ascii")
    return pin


class ProcessPopulation:
    def __init__(self, root, observations, source_population, pin):
        self.root, self.pin = Path(root), pin
        raw = (self.root/"report.json").read_bytes()
        report = json.loads(raw)
        check(sha(raw) == pin == (self.root/"COMMITTED").read_text()
              and report["schema"] == "process-output-population-v1"
              and report["observations"] == observations.pin
              and report["source_population"] == source_population.pin
              and report["complete"] and not report["errors"], "PROCESS_POPULATION_PIN")
        check(set(report["files"]) == set(FILES), "PROCESS_POPULATION_FILES")
        for name, expected_pin in report["files"].items():
            value = (self.root/name).read_bytes()
            check(len(value) == expected_pin["bytes"] and sha(value) == expected_pin["sha256"],
                  "PROCESS_POPULATION_BYTES")
        rows, refs, probe = expected(observations, source_population)
        check(pq.read_table(self.root/FILES[0]).to_pylist() == rows
              and [json.loads(r) for r in (self.root/FILES[2]).read_bytes().splitlines()] == refs
              and json.loads((self.root/FILES[3]).read_bytes()) == probe, "PROCESS_POPULATION_SOURCE")
        self.rows, self.refs, self.probe = rows, refs, probe


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for name in ("raw", "source-population", "observations", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--source-population-pin", required=True)
    p.add_argument("--observation-pin", required=True)
    a = p.parse_args()
    s = ToolSource(a.raw)
    population = TemporalPopulation(a.source_population, s, a.source_population_pin)
    observations = ProcessObservations(a.observations, a.observation_pin, s)
    pin = build(observations, population, a.output)
    ProcessPopulation(a.output, observations, population, pin)
    print(json.dumps(dict(complete=True, pin=pin)))
