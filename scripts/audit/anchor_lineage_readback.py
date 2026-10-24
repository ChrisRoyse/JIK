"""Independent original-archive recount and A2 label/value-set reconstruction."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import tarfile
import time

import blake3
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.renderer_population_readback import check, sha
from audit.guard_render_readback import lines
from senses.contract import digest
from senses.trajectory_population import publish

SHARDS = ("wikidata5m_transductive_train", "wikidata5m_transductive_valid", "wikidata5m_transductive_test")
QUESTIONS = {"biomedical": ("P1057", "P1995"), "science": ("P123",)}


def parents(root, rep):
    p = root / "parents"
    cal = json.loads((p / "cal-report.json").read_bytes())
    contract = json.loads((p / "cal-contract.json").read_bytes())
    check(sha(p / "cal-report.json") == rep["calibration_sha256"], "AL_CAL_PIN")
    chosen = []
    for family, props in QUESTIONS.items():
        summary = json.loads((p / (family + "-cal-summary.json")).read_bytes())
        check(sha(p / (family + "-cal-summary.json")) == cal["families"][family]["summary_sha256"], "AL_PARENT_PIN")
        check(sha(p / (family + "-anchors.json")) == summary["anchors_sha256"], "AL_PARENT_PIN")
        meta = json.loads((p / (family + "-record-summary.json")).read_bytes())
        check(sha(p / (family + "-record-summary.json")) == contract["families"][family]["summary_sha256"]
              and sha(p / (family + "-rows.jsonl")) == meta["rows_sha256"] == summary["parent_rows_sha256"], "AL_PARENT_PIN")
        rows = lines(p / (family + "-rows.jsonl")); anchors = json.loads((p / (family + "-anchors.json")).read_bytes())
        check(set(anchors) == set(props), "AL_QUESTION_SET")
        for prop in props:
            a = anchors[prop]
            check(len(a["rows"]) == len(a["labels"]) == 504, "AL_SAMPLE_COUNT")
            for i in range(504):
                row, label = a["rows"][i], a["labels"][i]; source = rows[row]
                check(meta["values"][prop][source["y"][prop]] == a["classes"][label], "AL_LABEL_BINDING")
                check(hashlib.sha256(source["text"].encode()).hexdigest() == source["text_source"]["text_sha256"], "AL_TEXT_PIN")
                chosen.append({"case": digest(["a2-anchor-lineage-v1", family, prop, source["entity"]]), "family": family,
                    "property": prop, "entity": source["entity"], "record_row": row, "label": label,
                    "legacy_value": a["classes"][label], "classes": a["classes"], "text_sha256": source["text_source"]["text_sha256"]})
    return chosen


def witness(raw, raw_pin, wanted):
    """Read original bytes once; this immutable evidence is invariant across fault probes."""
    check(sha(raw) == raw_pin, "AL_ORIGINAL_PIN")
    retained, counts = [], {}
    with tarfile.open(raw, "r|gz") as archive:
        for member in archive:
            check(member.isfile() and member.name in {s + ".txt" for s in SHARDS}, "AL_ORIGINAL_MEMBER")
            shard = member.name[:-4]; check(shard not in counts, "AL_ORIGINAL_DUPLICATE_MEMBER")
            n, size = 0, 0
            for line in archive.extractfile(member):
                fields = line.rstrip(b"\r\n").split(b"\t")
                check(len(fields) == 3, "AL_ORIGINAL_TSV")
                fields = tuple(v.decode("utf-8") for v in fields)
                if fields[:2] in wanted:
                    retained.append({"shard": shard, "row": n, "subject": fields[0], "property": fields[1], "object": fields[2],
                                     "line_hex": line.hex(), "line_sha256": hashlib.sha256(line).hexdigest()})
                n += 1; size += len(line)
            check(size == member.size, "AL_ORIGINAL_BYTES")
            counts[shard] = {"rows": n, "bytes": size}
    check(set(counts) == set(SHARDS) and sha(raw) == raw_pin, "AL_ORIGINAL_DRIFT")
    return {"original_archive_sha256": raw_pin, "original_archive_bytes": raw.stat().st_size,
            "counts": counts, "selected": retained, "wanted": sorted([list(p) for p in wanted])}


def read(root, pin, original):
    check(sha(root / "report.json") == pin, "AL_REPORT_PIN")
    rep = json.loads((root / "report.json").read_bytes())
    check(rep["schema"] == "a2-anchor-lineage-v1" and all(rep[k] is False for k in
          ("scientific_admission", "labels_rewritten", "real_world_correctness_certified", "sealed_evaluation")), "AL_CLAIM")
    expected_files = {"canonical-manifest.jsonl", "selected.jsonl", "raw-selected.jsonl", "canonical-selected.jsonl", "cases.jsonl",
                      "parents/cal-report.json", "parents/cal-contract.json"}
    expected_files |= {"parents/" + f + "-" + k for f in QUESTIONS for k in
                      ("rows.jsonl", "record-summary.json", "anchors.json", "cal-summary.json")}
    check(set(rep["files"]) == expected_files, "AL_FILE_SET")
    for name, item in rep["files"].items():
        check(sha(root / name) == item["sha256"] and (root / name).stat().st_size == item["bytes"], "AL_FILE_PIN")
    check(rep["parent_files"] == {name.removeprefix("parents/"): item for name, item in rep["files"].items() if name.startswith("parents/")}, "AL_PARENT_FILES")
    selected = parents(root, rep)
    check(lines(root / "selected.jsonl") == selected and len(selected) == rep["cases"] == 1512, "AL_SELECTION")
    manifest = lines(root / "canonical-manifest.jsonl"); declarations = {s: next(m for m in manifest if m["shard"] == s) for s in SHARDS}
    check(sha(root / "canonical-manifest.jsonl") == rep["manifest_sha256"], "AL_MANIFEST_PIN")
    check(rep["original_archive_sha256"] == original["original_archive_sha256"]
          and rep["original_archive_bytes"] == original["original_archive_bytes"]
          and all(d["raw_sha256"] == [rep["original_archive_sha256"]] and d["raw_bytes"] == rep["original_archive_bytes"]
                  for d in declarations.values()), "AL_SOURCE_BINDING")
    check(original["wanted"] == sorted([[c["entity"], c["property"]] for c in selected]), "AL_SOURCE_CASES")
    raw = lines(root / "raw-selected.jsonl")
    check(raw == original["selected"] and rep["raw_counts"] == original["counts"], "AL_SOURCE_COMPLETENESS")
    check(all(rep["raw_counts"][s]["rows"] == rep["canonical_counts"][s] == declarations[s]["docs"] for s in SHARDS), "AL_SOURCE_COUNTS")
    by_pointer = {(r["shard"], r["row"]): r for r in raw}
    canon = lines(root / "canonical-selected.jsonl")
    check(len(by_pointer) == len(raw) == len(canon) == rep["retained_occurrences"], "AL_CANON_COUNT")
    seen, values, occurrences = set(), {}, {}
    for r in canon:
        c = r["record"]; pointer = (r["shard"], c["row"])
        check(pointer in by_pointer and pointer not in seen, "AL_CANON_POINTER"); seen.add(pointer)
        source = by_pointer[pointer]; content = bytes.fromhex(source["line_hex"])
        check(hashlib.sha256(content).hexdigest() == source["line_sha256"], "AL_RAW_LINE_PIN")
        text = "\t".join((source["subject"], source["property"], source["object"]))
        check(content.rstrip(b"\r\n").decode() == text and c == {"row": source["row"], "kind": "row", "text": text,
              "n": len(text.encode()), "b3": blake3.blake3(text.encode()).hexdigest()[:32],
              "f": {"s": source["subject"], "p": source["property"], "o": source["object"], "split": source["shard"] + ".txt"}}, "AL_CANON_PAYLOAD")
        key = (source["subject"], source["property"])
        values.setdefault(key, set()).add(source["object"]); occurrences.setdefault(key, []).append(source)
    cases = lines(root / "cases.jsonl"); check(len(cases) == len(selected), "AL_CASE_COUNT")
    summaries = {}
    for case, chosen in zip(cases, selected):
        key = (chosen["entity"], chosen["property"]); actual = sorted(values.get(key, set()))
        pointers = sorted(occurrences.get(key, []), key=lambda r: (SHARDS.index(r["shard"]), r["row"]))
        first = pointers[0]["object"] if pointers else None
        status = "missing" if not actual else "legacy_mismatch" if first != chosen["legacy_value"] else "singleton" if len(actual) == 1 else "multiple"
        expected = {**chosen, "values": actual, "first_value": first, "status": status,
                    "legacy_value_present": chosen["legacy_value"] in actual,
                    "alternative_in_class": [v for v in actual if v != chosen["legacy_value"] and v in chosen["classes"]],
                    "source_occurrences": [{k:r[k] for k in ("shard", "row", "line_sha256")} for r in pointers],
                    "real_world_uniqueness_certified": False, "currentness_certified": False,
                    "statement_qualifiers_available": False, "scientific_admission": False}
        check(case == expected, "AL_CASE_VALUES")
        summaries.setdefault(chosen["family"] + "/" + chosen["property"], []).append(case)
    counts = {key: {"cases": len(part), "statuses": dict(Counter(c["status"] for c in part)),
                   "legacy_not_present": sum(not c["legacy_value_present"] for c in part),
                   "alternative_in_class": sum(bool(c["alternative_in_class"]) for c in part),
                   "singleton_by_label": {v: sum(c["status"] == "singleton" and c["legacy_value"] == v for c in part) for v in part[0]["classes"]}}
              for key, part in summaries.items()}
    check(counts == rep["summaries"], "AL_SUMMARY")
    check(.9 <= rep["phase_coverage"] <= 1.001 and abs(sum(rep["phases_seconds"].values()) / rep["wall_seconds"]
          - rep["phase_coverage"]) < 1e-10, "AL_TIMING")
    return {"author_checked": True, "report_sha256": pin, "source_rows_recounted": sum(v["rows"] for v in original["counts"].values()),
            "selected_occurrences": len(raw), "cases": len(cases), "summaries": counts,
            "all_original_value_occurrences_retained": True, "scientific_admission": False, "real_world_correctness_certified": False}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for n in ("root", "raw", "output"):
        ap.add_argument("--" + n, type=Path, required=True)
    ap.add_argument("--report-sha256", required=True)
    a = ap.parse_args(); a.output.mkdir(parents=True, exist_ok=False); started = time.perf_counter()
    rep = json.loads((a.root / "report.json").read_bytes())
    selection = parents(a.root, rep)
    original = witness(a.raw, rep["original_archive_sha256"], {(c["entity"], c["property"]) for c in selection})
    publish(a.output / "original-replay.json", original)
    result = read(a.root, a.report_sha256, original)
    result["wall_seconds"] = time.perf_counter() - started
    publish(a.output / "readback.json", result); print(json.dumps(result))
