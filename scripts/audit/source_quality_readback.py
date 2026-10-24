"""Independent interval and sequence audit of material routing, not admission."""
import argparse
import bisect
from collections import Counter
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from text.source_material import IMAGE, DAMAGE, CONTRACT, check, digest
from text.source_quality import load_config, pinned
from text.evaluation_multiscale import file_sha
from text.atomise_pilot import atoms_of, text_of
from senses.trajectory_population import publish


def audit_field(row, field, base, result):
    text = field["text"]; annotation = result["annotation"]; projection = result["projection"]
    check(digest(text) == field["sha256"] and len(text) == field["characters"]
          and len(text.encode()) == field["utf8_bytes"], "READBACK_SOURCE")
    check(result["document_id"] == row["id"] == base["document_id"]
          and result["field_name"] == field["name"] == base["field_name"]
          and result["domain"] == row["domain"] and result["group_keys"] == row["group_keys"]
          and result["source_record_sha256"] == row["source"]["record_sha256"], "READBACK_LINEAGE")
    # Syntax constants define the narrow detector. Independently enumerate every
    # match and rederive the policy; producer classification is not called.
    images = [dict(start=m.start(), end=m.end(), sha256=digest(m[0]),
                   code="IMAGE_CONTENT_UNAVAILABLE", scanned_page="scanned-page" in m[0].lower())
              for m in IMAGE.finditer(text)]
    damage = [dict(start=m.start(), end=m.end(), sha256=digest(m[0]), code="ENCODING_DAMAGE")
              for m in DAMAGE.finditer(text)]
    residual_parts = []; cursor = 0
    for image in images:
        residual_parts.append(text[cursor:image["start"]]); cursor = image["end"]
    residual_parts.append(text[cursor:]); residual = "".join(residual_parts).strip().lower()
    image_only = bool(images) and residual in ("", "pmc")
    obstructions = sorted(images+damage, key=lambda r: (r["start"], r["end"], r["code"]))
    if image_only:
        obstructions = [dict(start=0, end=len(text), sha256=field["sha256"], code="IMAGE_ONLY_SOURCE")]
    status = "EMPTY_SOURCE" if not text.strip() else ("IMAGE_ONLY_SOURCE" if image_only else (
        "PARTIAL_TEXT_AVAILABLE" if obstructions else "TEXT_AVAILABLE_UNASSESSED"))
    expected = dict(contract=CONTRACT, field_name=field["name"], text_sha256=field["sha256"],
                    source_characters=len(text), status=status, image_references=images,
                    encoding_damage=damage, obstructions=obstructions,
                    native_panel_requirements={m: 5 for m in (("text", "image") if images else ("text",))},
                    source_modified=False, quality_assessed=False, grounded=False)
    check(annotation == expected, "READBACK_ANNOTATION")
    counts = Counter(); ready, deferred, next_edges = [], [], []
    graph = base["graph"]
    if graph:
        check(graph["source_text_sha256"] == field["sha256"], "READBACK_GRAPH")
        starts = [r["start"] for r in obstructions]; maximum = []; end = 0
        for r in obstructions:
            end = max(end, r["end"]); maximum.append(end)
        levels = {}
        for n in graph["nodes"]:
            a, b = n["start"], n["end"]
            check(0 <= a < b <= len(text) and digest(text[a:b]) == n["content_sha256"], "READBACK_NODE")
            j = bisect.bisect_left(starts, b)-1
            code = "UNAVAILABLE_MATERIAL" if j >= 0 and maximum[j] > a else (
                None if text[a:b].strip() else "EMPTY_TEXT")
            if code:
                deferred.append(dict(id=n["id"], code=code)); counts["deferred:"+n["level"]] += 1
            else:
                ready.append(n["id"]); counts["available:"+n["level"]] += 1
            levels.setdefault(n["level"], []).append(n)
        available = set(ready)
        for nodes in levels.values():
            for left, right in zip(nodes, nodes[1:]):
                check(left["end"] == right["start"], "READBACK_TILING")
                if left["id"] in available and right["id"] in available:
                    next_edges.append(dict(relation="next", **{"from":left["id"], "to":right["id"]}))
        counts["retained_next_edges"] = len(next_edges)
    else:
        check(not text.strip(), "READBACK_MISSING_GRAPH")
    check(projection == dict(ready_node_ids=ready, deferred_nodes=deferred,
                             next_edges=next_edges, counts=dict(counts)), "READBACK_PROJECTION")
    counts.update(fields=1, source_characters=len(text), image_references=len(images), encoding_damage_runs=len(damage))
    counts[status] += 1
    return counts


def audit(config, output):
    started = time.perf_counter(); cfg, parents = load_config(config)
    report = json.loads((output/"report.json").read_bytes())
    check(set(report["parents"]) == set(cfg) and all(report["parents"][n]["sha256"] == cfg[n]["sha256"] for n in cfg)
          and report["complete"] and not report["errors"], "READBACK_PARENT")
    check(all(report[k] is False for k in ("scientific_admission", "sealed_evaluation", "training_authorized",
                                          "source_modified", "quality_assessed")), "READBACK_CLAIM")
    totals = Counter(); duplicates = {}; expected_recovery = []
    for domain, meta in parents["source"]["sources"].items():
        src = pinned(Path(cfg["source"]["root"]), meta)
        geo = pinned(Path(cfg["multiscale"]["root"]), parents["multiscale"]["files"][domain])
        result = pinned(output, report["files"][domain]); counts = Counter(); old = Counter()
        with src.open("rb") as sf, geo.open("rb") as gf, result.open("rb") as rf:
            for line in sf:
                row = json.loads(line); statuses = set(); counts["documents"] += 1
                historical = atoms_of(text_of(domain, {f["name"]:f["text"] for f in row["fields"]}))
                if len(historical) >= 3:
                    old.update(document_passes_minimum=1, retained_sentences=len(historical),
                               retained_characters=sum(map(len, historical)))
                for field in row["fields"]:
                    base, record = json.loads(next(gf)), json.loads(next(rf))
                    counts.update(audit_field(row, field, base, record)); statuses.add(record["annotation"]["status"])
                    if field["text"].strip():
                        key = field["sha256"]
                        if key in duplicates: counts["repeated_exact_fields"] += 1
                        duplicates.setdefault(key, []).append([row["id"], field["name"]])
                    if record["annotation"]["obstructions"]:
                        expected_recovery.append(dict(document_id=row["id"], domain=domain, field_name=field["name"],
                            source=row["source"], metadata=row["metadata"], source_licence=row["source_licence"],
                            status=record["annotation"]["status"], text_sha256=field["sha256"],
                            native_panel_requirements=record["annotation"]["native_panel_requirements"],
                            action="retain-source;recover-original-modality-or-audited-extraction"))
                for status in statuses: counts["documents:"+status] += 1
            check(next(gf, None) is None and next(rf, None) is None, "READBACK_EXTRA_FIELD")
        check(dict(counts) == report["domains"][domain]["counts"], "READBACK_COUNTS")
        for key in ("document_passes_minimum", "retained_sentences", "retained_characters"):
            check(old[key] == report["domains"][domain]["legacy"][key], "READBACK_LEGACY")
        totals.update(counts)
    path = pinned(output, report["files"]["recovery.jsonl"])
    with path.open("rb") as f: recovery = [json.loads(line) for line in f]
    check(recovery == expected_recovery, "READBACK_RECOVERY")
    path = pinned(output, report["files"]["exact-field-groups.json"])
    check(json.loads(path.read_bytes()) == {k:v for k,v in duplicates.items() if len(v)>1}, "READBACK_DUPLICATES")
    return dict(author_checked=True, all_fields_read_back=True, totals=dict(totals),
                recovery_fields=len(recovery), report_sha256=file_sha(output/"report.json"),
                legacy_retention_replayed=True, wall_seconds=time.perf_counter()-started)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "publication", "output"): p.add_argument("--"+name, type=Path, required=True)
    a = p.parse_args(); result = audit(a.config, a.publication); publish(a.output, result)
    print(json.dumps(result))
