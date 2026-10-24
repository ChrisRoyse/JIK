"""Census source defects and publish source-bound multiscale material routing.

PC-07/13/35: six named sources and existing graph streams; no pairwise scan.
Cost O(characters + nodes*log(obstructions+1) + edges); field-bounded memory.
Old atomizer diagnostics replay its policy, never modify its historical bytes.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from canon.segment import ROLE_SENTENCE, segment
from senses.trajectory_population import publish
from text.atomise_pilot import text_of, MIN_CHARS, MAX_CHARS, WORDS, MAX_PER_DOC
from text.evaluation_multiscale import canonical, file_sha
from text.source_material import CONTRACT, check, digest, material, project

SCHEMA = "source-quality-routing-v1"


def legacy(row):
    fields = {f["name"]: f["text"] for f in row["fields"]}
    text = text_of(row["domain"], fields)
    counts = Counter(normalized_input_characters=len(text), source_fields=len(fields))
    selected = []; previous = None
    for a, b, role in segment(text):
        value = text[a:b].strip()
        if role != ROLE_SENTENCE:
            continue
        if len(value) < MIN_CHARS or len(WORDS.findall(value)) < 4:
            counts["short_or_few_word_sentences"] += 1; continue
        if len(value) > MAX_CHARS:
            counts["long_sentences_dropped"] += 1; continue
        counts["eligible_sentences"] += 1
        if len(selected) >= MAX_PER_DOC:
            counts["eligible_sentences_after_head_cap"] += 1; continue
        selected.append(value)
        if previous is not None and text[previous:a].strip():
            counts["nonempty_gap_bridges"] += 1
        previous = b
    counts["retained_sentences"] = len(selected) if len(selected) >= 3 else 0
    counts["retained_characters"] = sum(map(len, selected)) if len(selected) >= 3 else 0
    counts["document_passes_minimum"] = int(len(selected) >= 3)
    if len(selected) < 3:
        counts["nonempty_gap_bridges"] = 0
    if row["domain"] == "finance":
        counts["fields_not_read"] = len(fields)-3
        counts["characters_in_unread_fields"] = sum(len(v) for k, v in fields.items()
                                                   if k not in ("section_1", "section_1A", "section_7"))
    return dict(counts)


def pinned(root, meta):
    path = root/meta["file"]
    check(path.parent.resolve() == root.resolve(), "PATH")
    check(path.stat().st_size == meta["bytes"] and file_sha(path) == meta["sha256"], "PARENT_FILE")
    return path


def load_config(config):
    cfg = json.loads(config.read_bytes()); reports = {}
    for name in ("source", "multiscale"):
        entry = cfg[name]; path = Path(entry["root"])/"report.json"
        check(file_sha(path) == entry["sha256"], "PARENT_PIN")
        reports[name] = json.loads(path.read_bytes())
        check(reports[name]["complete"] and not reports[name]["errors"]
              and reports[name]["sealed_evaluation"] is False, "PARENT_STATE")
    check(reports["source"]["schema"] == "evaluation-source-capture-v1"
          and reports["multiscale"]["schema"] == "evaluation-multiscale-v1", "PARENT_SCHEMA")
    check(reports["multiscale"]["source_report_sha256"] == cfg["source"]["sha256"], "PARENT_SOURCE")
    return cfg, reports


def run(config, output):
    started = time.perf_counter(); check(not output.exists(), "EXISTS")
    cfg, parents = load_config(config); output.mkdir(parents=True)
    phase = Counter(); phase["input_seconds"] = time.perf_counter()-started
    files, summaries, errors, duplicates = {}, {}, [], {}
    with (output/"recovery.jsonl").open("xb") as recovery:
        for domain, meta in parents["source"]["sources"].items():
            t = time.perf_counter()
            src = pinned(Path(cfg["source"]["root"]), meta)
            gm = parents["multiscale"]["files"][domain]
            check(gm["source_sha256"] == meta["sha256"], "GRAPH_PARENT")
            geo = pinned(Path(cfg["multiscale"]["root"]), gm)
            phase["input_seconds"] += time.perf_counter()-t
            counts, old = Counter(), Counter(); dest = output/(domain+".jsonl")
            with src.open("rb") as sf, geo.open("rb") as gf, dest.open("xb") as out:
                graphs = iter(gf)
                for line in sf:
                    t = time.perf_counter(); row = json.loads(line)
                    phase["input_seconds"] += time.perf_counter()-t
                    t = time.perf_counter(); previous = legacy(row); old.update(previous)
                    phase["legacy_seconds"] += time.perf_counter()-t
                    counts["documents"] += 1; doc_status = set()
                    for field in row["fields"]:
                        try:
                            t = time.perf_counter(); base = json.loads(next(graphs))
                            check((base["document_id"], base["field_name"]) == (row["id"], field["name"]), "FIELD_ORDER")
                            phase["input_seconds"] += time.perf_counter()-t
                            t = time.perf_counter(); annotation = material(field)
                            projection = project(field, base["graph"], annotation)
                            result = dict(document_id=row["id"], domain=domain, field_name=field["name"],
                                          source_record_sha256=row["source"]["record_sha256"],
                                          group_keys=row["group_keys"], annotation=annotation, projection=projection)
                            counts.update(fields=1, source_characters=len(field["text"]))
                            counts.update(projection["counts"]); counts[annotation["status"]] += 1
                            counts["image_references"] += len(annotation["image_references"])
                            counts["encoding_damage_runs"] += len(annotation["encoding_damage"])
                            doc_status.add(annotation["status"])
                            if field["text"].strip():
                                key = field["sha256"]
                                if key in duplicates: counts["repeated_exact_fields"] += 1
                                duplicates.setdefault(key, []).append([row["id"], field["name"]])
                            phase["material_seconds"] += time.perf_counter()-t
                            t = time.perf_counter(); out.write(canonical(result)+b"\n")
                            if annotation["obstructions"]:
                                recovery.write(canonical(dict(document_id=row["id"], domain=domain,
                                    field_name=field["name"], source=row["source"], metadata=row["metadata"],
                                    source_licence=row["source_licence"], status=annotation["status"],
                                    native_panel_requirements=annotation["native_panel_requirements"],
                                    text_sha256=field["sha256"], action="retain-source;recover-original-modality-or-audited-extraction"))+b"\n")
                            phase["write_seconds"] += time.perf_counter()-t
                        except (KeyError, TypeError, ValueError, StopIteration) as exc:
                            errors.append(dict(document_id=row["id"], field_name=field["name"], error=str(exc)))
                    for status in doc_status: counts["documents:"+status] += 1
                check(next(graphs, None) is None, "EXTRA_GRAPH")
            check(counts["documents"] == meta["counts"]["selected_documents"], "DOCUMENT_COUNT")
            check(counts["fields"] == meta["counts"]["selected_fields"], "FIELD_COUNT")
            t = time.perf_counter()
            files[domain] = dict(file=dest.name, sha256=file_sha(dest), bytes=dest.stat().st_size)
            phase["write_seconds"] += time.perf_counter()-t
            summaries[domain] = dict(counts=dict(counts), legacy=dict(old))
            print(json.dumps(dict(domain=domain, counts=dict(counts), legacy=dict(old))), flush=True)
    t = time.perf_counter()
    publish(output/"exact-field-groups.json", {k:v for k,v in duplicates.items() if len(v) > 1})
    for name in ("recovery.jsonl", "exact-field-groups.json"):
        files[name] = dict(file=name, sha256=file_sha(output/name), bytes=(output/name).stat().st_size)
    phase["write_seconds"] += time.perf_counter()-t
    wall = time.perf_counter()-started; phase["other_seconds"] = wall-sum(phase.values())
    report = dict(schema=SCHEMA, contract=CONTRACT, parents=cfg, files=files, domains=summaries,
                  errors=errors, complete=not errors, scientific_admission=False, sealed_evaluation=False,
                  training_authorized=False, source_modified=False, quality_assessed=False,
                  scope="inspected-development;material-availability-only", timing=dict(phase, wall_seconds=wall),
                  code_sha256=file_sha(Path(__file__)), material_code_sha256=file_sha(Path(__file__).with_name("source_material.py")))
    publish(output/"report.json", report)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(); r = run(a.config, a.output)
    print(json.dumps(dict(complete=r["complete"], errors=r["errors"], timing=r["timing"])))
    if not r["complete"]: raise SystemExit(2)
