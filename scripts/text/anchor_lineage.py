"""Bind the fixed A2 calibration labels to complete sets in original triples.

One source scan, no label rewriting, no inference from unobserved facts.
Snapshot membership is distinct from uniqueness and real-world correctness.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tarfile
import time

import blake3
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from canon.common import iter_canon_docs
from audit.renderer_population_readback import sha, check
from senses.contract import canonical, digest
from senses.trajectory_population import publish

SCHEMA = "a2-anchor-lineage-v1"
SHARDS = ("wikidata5m_transductive_train", "wikidata5m_transductive_valid", "wikidata5m_transductive_test")
QUESTIONS = {"biomedical": ("P1057", "P1995"), "science": ("P123",)}


def load_selection(calibration, records, output):
    """Copy exact bounded parent files so cold readers can replay row selection."""
    files, selected = {}, []
    report = json.loads((calibration / "report.json").read_bytes())
    check(report["complete"] and report["scientific_admission"] is False, "ANCHOR_PARENT")
    contract = json.loads((calibration / "contract.json").read_bytes())
    for family, properties in QUESTIONS.items():
        summary = json.loads((calibration / family / "summary.json").read_bytes())
        check(sha(calibration / family / "summary.json") == report["families"][family]["summary_sha256"], "ANCHOR_CAL_PIN")
        anchors = json.loads((calibration / family / "anchors.json").read_bytes())
        check(sha(calibration / family / "anchors.json") == summary["anchors_sha256"], "ANCHOR_LABEL_PIN")
        record_root = records / "records" / family
        meta = json.loads((record_root / "summary.json").read_bytes())
        check(sha(record_root / "summary.json") == contract["families"][family]["summary_sha256"]
              and sha(record_root / "rows.jsonl") == meta["rows_sha256"] == summary["parent_rows_sha256"], "ANCHOR_RECORD_PIN")
        rows = [json.loads(line) for line in (record_root / "rows.jsonl").read_bytes().splitlines()]
        check(set(anchors) == set(properties) and len(rows) == meta["rows"], "ANCHOR_QUESTIONS")
        for prop in properties:
            anchor = anchors[prop]
            check(len(anchor["rows"]) == len(anchor["labels"]) == 504, "ANCHOR_FIXED_N")
            for row, label in zip(anchor["rows"], anchor["labels"]):
                record = rows[row]; value = anchor["classes"][label]
                check(meta["values"][prop][record["y"][prop]] == value, "ANCHOR_LABEL_BINDING")
                selected.append({"case": digest([SCHEMA, family, prop, record["entity"]]), "family": family,
                    "property": prop, "entity": record["entity"], "record_row": row, "label": label,
                    "legacy_value": value, "classes": anchor["classes"], "text_sha256": record["text_source"]["text_sha256"]})
        for key, path in (("rows", record_root / "rows.jsonl"), ("record-summary", record_root / "summary.json"),
                          ("anchors", calibration / family / "anchors.json"), ("cal-summary", calibration / family / "summary.json")):
            filename = family + "-" + key + (".jsonl" if key == "rows" else ".json")
            shutil.copyfile(path, output / filename)
            files[filename] = {"sha256": sha(path), "bytes": path.stat().st_size}
    for filename in ("report.json", "contract.json"):
        shutil.copyfile(calibration / filename, output / ("cal-" + filename))
        files["cal-" + filename] = {"sha256": sha(calibration / filename), "bytes": (calibration / filename).stat().st_size}
    check(len(selected) == len({c["case"] for c in selected}) == 1512, "ANCHOR_CASES")
    return selected, files


def classify(case, occurrences):
    ordered = sorted(occurrences, key=lambda r: (SHARDS.index(r["shard"]), r["row"]))
    values = sorted({r["object"] for r in ordered})
    first = ordered[0]["object"] if ordered else None
    status = "missing" if not values else "legacy_mismatch" if first != case["legacy_value"] else "singleton" if len(values) == 1 else "multiple"
    return {**case, "values": values, "first_value": first, "status": status,
            "legacy_value_present": case["legacy_value"] in values,
            "alternative_in_class": [v for v in values if v != case["legacy_value"] and v in case["classes"]],
            "source_occurrences": [{k: r[k] for k in ("shard", "row", "line_sha256")} for r in ordered],
            "real_world_uniqueness_certified": False, "currentness_certified": False,
            "statement_qualifiers_available": False, "scientific_admission": False}


def judge(case, candidate_value, unique=False):
    check(isinstance(candidate_value, str) and candidate_value.startswith("Q") and candidate_value[1:].isdigit(), "ANCHOR_VALUE_TYPE")
    values = case["values"]
    if case["status"] in ("missing", "legacy_mismatch"):
        status = "MNLM_A2_ANCHOR_UNAVAILABLE"
    elif unique and len(values) != 1:
        status = "MNLM_A2_AMBIGUOUS_VALUE_SET"
    elif candidate_value not in values:
        status = "MNLM_A2_VALUE_UNOBSERVED"
    else:
        status = "source_unique" if unique else "source_supported"
    return {"case": case["case"], "candidate_value": candidate_value, "status": status,
            "scope": "pinned-source-value-set", "asserted": False, "real_world_correctness_certified": False}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("calibration", "records", "canon", "raw", "output"):
        ap.add_argument("--" + name, required=True, type=Path)
    ap.add_argument("--manifest-sha256", required=True)
    ap.add_argument("--calibration-sha256", required=True)
    a = ap.parse_args(); start = time.perf_counter(); phases = {}
    check(not a.output.exists() and sha(a.canon / "manifest.jsonl") == a.manifest_sha256
          and sha(a.calibration / "report.json") == a.calibration_sha256, "ANCHOR_INPUT_PIN")
    manifest = [json.loads(l) for l in (a.canon / "manifest.jsonl").read_bytes().splitlines()]
    declarations = {s: next(m for m in manifest if m["shard"] == s) for s in SHARDS}
    check(all(m["raw_sha256"] == [declarations[SHARDS[0]]["raw_sha256"][0]] for m in declarations.values()), "ANCHOR_RAW_DECLARATION")
    raw_pin = sha(a.raw)
    check(raw_pin == declarations[SHARDS[0]]["raw_sha256"][0] and a.raw.stat().st_size == declarations[SHARDS[0]]["raw_bytes"], "ANCHOR_RAW_PIN")
    a.output.mkdir(parents=True); (a.output / "parents").mkdir()
    selected, parent_files = load_selection(a.calibration, a.records, a.output / "parents")
    shutil.copyfile(a.canon / "manifest.jsonl", a.output / "canonical-manifest.jsonl")
    wanted = {(c["entity"], c["property"]) for c in selected}
    check(len(wanted) == len(selected), "ANCHOR_DUPLICATE_CASE")
    phases["input_pins_and_population"] = time.perf_counter() - start
    tick = time.perf_counter(); raw_rows, counts = [], {}
    with tarfile.open(a.raw, "r|gz") as tar:
        for member in tar:
            check(member.isfile() and member.name.removesuffix(".txt") in SHARDS, "ANCHOR_RAW_MEMBER")
            shard = member.name.removesuffix(".txt"); check(shard not in counts, "ANCHOR_DUPLICATE_MEMBER")
            count = 0; size = 0
            for row, line in enumerate(tar.extractfile(member)):
                count += 1; size += len(line)
                subject, prop, obj = line.rstrip(b"\r\n").decode("utf-8", "strict").split("\t")
                if (subject, prop) in wanted:
                    raw_rows.append({"shard": shard, "row": row, "subject": subject, "property": prop, "object": obj,
                                     "line_hex": line.hex(), "line_sha256": hashlib.sha256(line).hexdigest()})
            check(count == declarations[shard]["docs"] and size == member.size, "ANCHOR_RAW_COUNTS")
            counts[shard] = {"rows": count, "bytes": size}
    check(set(counts) == set(SHARDS), "ANCHOR_RAW_SHARDS")
    phases["original_triple_stream"] = time.perf_counter() - tick
    tick = time.perf_counter(); canonical_rows = []; canonical_counts = {}
    for shard in SHARDS:
        declaration = declarations[shard]
        check((a.canon / (shard + ".jsonl.zst")).stat().st_size == declaration["out_bytes"], "ANCHOR_CANON_SIZE")
        count = 0
        for rec in iter_canon_docs(a.canon, shard, expected_blake3=declaration["out_blake3"]):
            check(rec["row"] == count, "ANCHOR_CANON_ORDER"); count += 1
            fields = rec["f"]
            if (fields["s"], fields["p"]) not in wanted:
                continue
            payload = "\t".join((fields["s"], fields["p"], fields["o"])).encode()
            check(rec["kind"] == "row" and rec["text"].encode() == payload and rec["n"] == len(payload)
                  and rec["b3"] == blake3.blake3(payload).hexdigest()[:32] and fields["split"] == shard + ".txt", "ANCHOR_CANON_PAYLOAD")
            canonical_rows.append({"shard": shard, "record": rec})
        check(count == declaration["docs"], "ANCHOR_CANON_COUNTS"); canonical_counts[shard] = count
        print(json.dumps({"shard": shard, "rows": count, "selected_occurrences": len(canonical_rows)}), flush=True)
    phases["canonical_triple_stream"] = time.perf_counter() - tick
    tick = time.perf_counter()
    source_map = {(r["shard"], r["row"]): r for r in raw_rows}
    check(len(source_map) == len(raw_rows) == len(canonical_rows), "ANCHOR_OCCURRENCES")
    for r in canonical_rows:
        s = source_map[(r["shard"], r["record"]["row"])]; f = r["record"]["f"]
        check((s["subject"], s["property"], s["object"]) == (f["s"], f["p"], f["o"]), "ANCHOR_RAW_CANON_MATCH")
    by_case = {key: [] for key in wanted}
    for r in raw_rows:
        by_case[(r["subject"], r["property"])].append(r)
    cases = [classify(c, by_case[(c["entity"], c["property"])]) for c in selected]
    summaries = {}
    for family, props in QUESTIONS.items():
        for prop in props:
            part = [c for c in cases if c["family"] == family and c["property"] == prop]
            summaries[family + "/" + prop] = {"cases": len(part), "statuses": dict(Counter(c["status"] for c in part)),
                "legacy_not_present": sum(not c["legacy_value_present"] for c in part),
                "alternative_in_class": sum(bool(c["alternative_in_class"]) for c in part),
                "singleton_by_label": {v: sum(c["status"] == "singleton" and c["legacy_value"] == v for c in part) for v in part[0]["classes"]}}
    phases["lineage_and_classification"] = time.perf_counter() - tick
    tick = time.perf_counter()
    for filename, values in (("selected.jsonl", selected), ("raw-selected.jsonl", raw_rows), ("canonical-selected.jsonl", canonical_rows), ("cases.jsonl", cases)):
        (a.output / filename).write_bytes(b"".join(canonical(v) + b"\n" for v in values))
    check(sha(a.raw) == raw_pin and sha(a.canon / "manifest.jsonl") == a.manifest_sha256, "ANCHOR_SOURCE_DRIFT")
    files = {str(p.relative_to(a.output)).replace("\\", "/"): {"sha256": sha(p), "bytes": p.stat().st_size}
             for p in a.output.rglob("*") if p.is_file()}
    phases["publication_and_final_pins"] = time.perf_counter() - tick
    wall = time.perf_counter() - start
    report = {"schema": SCHEMA, "files": files, "parent_files": parent_files, "manifest_sha256": a.manifest_sha256,
              "calibration_sha256": a.calibration_sha256, "original_archive_sha256": raw_pin, "original_archive_bytes": a.raw.stat().st_size,
              "raw_counts": counts, "canonical_counts": canonical_counts, "cases": len(cases), "retained_occurrences": len(raw_rows),
              "summaries": summaries, "source_code_sha256": sha(Path(__file__)), "phases_seconds": phases,
              "wall_seconds": wall, "phase_coverage": sum(phases.values()) / wall, "scientific_admission": False,
              "labels_rewritten": False, "real_world_correctness_certified": False, "sealed_evaluation": False}
    publish(a.output / "report.json", report)
    print(json.dumps({"report_sha256": sha(a.output / "report.json"), "summaries": summaries, "wall_seconds": wall,
                      "phase_coverage": report["phase_coverage"]}), flush=True)


if __name__ == "__main__":
    main()
