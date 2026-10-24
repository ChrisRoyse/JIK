"""Exact pre-execution surfaces, with field membership separate from native values."""
import json
from pathlib import Path
import time

import pyarrow as pa
import pyarrow.parquet as pq

from claims.codata_source import check, sha
from claims.episodic_memory import canonical
from claims.tool_temporal_source import write

FILES = ("condition/chunks/atoms.parquet", "condition/summary.json", "fields.json", "probe.json")


def expected(conditions, source_population):
    c = conditions.commitment
    source = source_population.source
    ref = c["program_source"]
    check(c["source"] == source.pin and ref == source.reference(ref["task"], ref["turn"]),
          "CONDITION_POPULATION_SOURCE")
    left, right = c["program_span"]
    program = source.texts[ref["task"], ref["turn"]].encode()[left:right]
    rows, fields, units = [], [], {}
    for key in sorted(conditions.inputs):
        value = conditions.read(key)
        check(value["program"].encode() == program, "CONDITION_POPULATION_PROGRAM")
        surfaces = [("program", value["program"]), *[("wheel_filename", n) for n in sorted(value["wheels"])]]
        for ordinal, (field, text) in enumerate(surfaces):
            pin = sha(text.encode())
            if pin not in units:
                unit = len(rows)
                units[pin] = unit
                rows.append(dict(row=unit, chunk=0, entity=pin, text=text, start=0, end=len(text)))
            fields.append(dict(condition=key, field=field, ordinal=ordinal, unit=units[pin], text_sha256=pin))
    probe = next((r for r in source_population.atoms if 1800 <= len(r["text"]) <= 3000), None)
    check(probe is not None, "CONDITION_POPULATION_PROBE")
    lineage = dict(population=source_population.pin, atom=probe, unit=len(rows),
                   purpose="overflow_determinism_only_excluded_from_condition_input")
    rows.append(dict(probe, row=len(rows)))
    return rows, fields, lineage


def build(conditions, source_population, root):
    started = time.perf_counter()
    root = Path(root)
    rows, fields, probe = expected(conditions, source_population)
    check(not root.exists(), "CONDITION_POPULATION_EXISTS")
    (root/"condition/chunks").mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows), root/FILES[0])
    write(root/FILES[1], dict(chunks=len(rows), chunks_sha256=sha((root/FILES[0]).read_bytes())))
    write(root/FILES[2], fields)
    write(root/FILES[3], probe)
    report = dict(schema="execution-condition-population-v1", commitment=conditions.pin,
        public_inputs=conditions.commitment["inputs"], source_population=source_population.pin,
        complete=True, errors=[], calibration_only=True, scientific_admission=False,
        scope="exact_program_and_available_wheel_filenames", inputs=len(rows), field_instances=len(fields),
        utf8_bytes=sum(len(r["text"].encode()) for r in rows),
        families=dict(condition=dict(ready=True, summary_sha256=sha((root/FILES[1]).read_bytes()))),
        files={n: dict(bytes=(root/n).stat().st_size, sha256=sha((root/n).read_bytes())) for n in FILES},
        wall_seconds=time.perf_counter()-started)
    write(root/"report.json", report)
    pin = sha((root/"report.json").read_bytes())
    (root/"COMMITTED").write_text(pin, encoding="ascii")
    return pin


class ConditionPopulation:
    def __init__(self, root, conditions, source_population, pin):
        self.root, self.pin = Path(root), pin
        raw = (self.root/"report.json").read_bytes()
        report = json.loads(raw)
        check(sha(raw) == pin == (self.root/"COMMITTED").read_text()
              and report["schema"] == "execution-condition-population-v1"
              and report["commitment"] == conditions.pin
              and report["public_inputs"] == conditions.commitment["inputs"]
              and report["source_population"] == source_population.pin
              and report["complete"] and not report["errors"], "CONDITION_POPULATION_PIN")
        check(set(report["files"]) == set(FILES), "CONDITION_POPULATION_FILES")
        for name, pin in report["files"].items():
            raw = (self.root/name).read_bytes()
            check(len(raw) == pin["bytes"] and sha(raw) == pin["sha256"], "CONDITION_POPULATION_BYTES")
        rows, fields, probe = expected(conditions, source_population)
        check(pq.read_table(self.root/FILES[0]).to_pylist() == rows
              and (self.root/FILES[2]).read_bytes() == canonical(fields)
              and (self.root/FILES[3]).read_bytes() == canonical(probe), "CONDITION_POPULATION_CONTENT")
        self.rows, self.fields, self.probe = rows, fields, probe
