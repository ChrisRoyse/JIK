"""Exact source spans for scientific native observations; deduplicate measurements."""
import argparse
import json
from pathlib import Path
import sys
import time

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from claims.codata_source import CodataSource, check, sha
from claims.episodic_memory import canonical, digest


def write(path, value):
    with path.open("xb") as f:
        f.write(canonical(value))


def expected(source):
    refs, texts, units, seen = [], [], [], {}
    offsets = {}
    for year in source.rows:
        parts = source.raw[year+".txt"].decode("ascii").splitlines(keepends=True)
        starts, end = [], 0
        for line in parts:
            starts.append(end)
            end += len(line)
        offsets[year] = starts

    def add(year, kind, quantity, start, end):
        raw = source.raw[year+".txt"]
        text = raw[start:end].decode("ascii")
        pin = sha(text.encode())
        if pin not in seen:
            index = len(units)
            seen[pin] = index
            texts.append(text)
            units.append(dict(row=index, chunk=0, entity=pin, text=text, start=0, end=len(text)))
        index = seen[pin]
        check(texts[index] == text, "SURFACE_HASH_COLLISION")
        refs.append(dict(unit=index, kind=kind, quantity=quantity, edition=year,
                         source_sha256=source.manifest["files"][year+".txt"]["sha256"],
                         source_span=[start, end], text_sha256=pin,
                         partition="development_exposed", world_truth_established=False))
    for year, rows in source.rows.items():
        for name, row in rows.items():
            start = offsets[year][row["line"]-1]
            add(year, "row", name, start, start+len(row["raw_line"].rstrip()))
    # Source-name substrings serve as real query observations. They are not
    # generated questions and never include values, editions or checker results.
    for name, row in source.rows["2022"].items():
        start = offsets["2022"][row["line"]-1]
        add("2022", "name", name, start, start+len(name))
    # Exact contiguous table block; probe only, excluded from world regions.
    start = offsets["2018"][11]
    end = next(end for end in offsets["2018"][12:] if 1800 <= end-start <= 3000)
    add("2018", "probe", None, start, end)
    return units, refs


def build(source, output):
    start = time.perf_counter()
    units, refs = expected(source)
    check(not output.exists(), "POPULATION_EXISTS")
    (output/"science/chunks").mkdir(parents=True)
    parsed = time.perf_counter()
    pq.write_table(pa.Table.from_pylist(units), output/"science/chunks/atoms.parquet")
    with (output/"references.jsonl").open("xb") as f:
        for ref in refs:
            f.write(canonical(ref)+b"\n")
    summary = dict(chunks=len(units), chunks_sha256=sha((output/"science/chunks/atoms.parquet").read_bytes()))
    write(output/"science/summary.json", summary)
    names = ("science/chunks/atoms.parquet", "science/summary.json", "references.jsonl")
    report = dict(schema="scientific-native-population-v1", complete=True, errors=[],
                  calibration_only=True, purpose="exposed_engineering_observations_not_scientific_calibration",
                  families=dict(science=dict(ready=True, summary_sha256=sha((output/"science/summary.json").read_bytes()))),
                  source_pin=source.pin, units=len(units), references=len(refs),
                  duplicate_measurements_avoided=len(refs)-len(units),
                  files={n: dict(sha256=sha((output/n).read_bytes()), bytes=(output/n).stat().st_size) for n in names},
                  scientific_admission=False, meaning_asserted=False, served=False,
                  phase_seconds=dict(source_spans_and_dedup=parsed-start, persist=time.perf_counter()-parsed),
                  wall_seconds=time.perf_counter()-start)
    write(output/"report.json", report)
    pin = sha((output/"report.json").read_bytes())
    (output/"COMMITTED").write_text(pin, encoding="ascii")
    return dict(report_sha256=pin, units=len(units), references=len(refs),
                unique_utf8_bytes=sum(len(u["text"].encode()) for u in units))


def readback(output, source, expected_pin):
    raw = (output/"report.json").read_bytes()
    check(sha(raw) == expected_pin == (output/"COMMITTED").read_text(), "POPULATION_PIN")
    report = json.loads(raw)
    check(report["source_pin"] == source.pin and report["complete"]
          and report["scientific_admission"] is False, "POPULATION_SOURCE")
    for name, meta in report["files"].items():
        check(name in ("science/chunks/atoms.parquet", "science/summary.json", "references.jsonl"), "POPULATION_FILE")
        payload = (output/name).read_bytes()
        check(len(payload) == meta["bytes"] and sha(payload) == meta["sha256"], "POPULATION_BYTES")
    units = pq.read_table(output/"science/chunks/atoms.parquet").to_pylist()
    refs = [json.loads(line) for line in (output/"references.jsonl").read_bytes().splitlines()]
    check((units, refs) == expected(source), "POPULATION_CONTENT")
    check(report["units"] == len(units) and report["references"] == len(refs), "POPULATION_COUNT")
    for ref in refs:
        a, b = ref["source_span"]
        actual = source.raw[ref["edition"]+".txt"][a:b]
        check(actual == units[ref["unit"]]["text"].encode() and sha(actual) == ref["text_sha256"], "SPAN_READBACK")
    return units, refs, report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--source-pin", required=True)
    args = p.parse_args()
    source = CodataSource(args.source, args.source_pin)
    result = build(source, args.output)
    readback(args.output, source, result["report_sha256"])
    print(json.dumps(result))
