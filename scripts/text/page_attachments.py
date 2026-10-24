"""Attach discontinuous physical pages to unchanged multiscale source graphs."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from text.cap_layout import canonical, file_sha, sha
from text.multiscale_spans import identity, tile
from senses.trajectory_population import publish

CONTRACT = "multiscale-physical-page-attachment-v1"
LEVELS = ("sentence", "sentence_pair", "paragraph")


def check(ok, code):
    if not ok:
        raise ValueError("MNLM_PAGE_ATTACH_"+code)


def attach(base, layout, group):
    graph = base["graph"]
    check(layout["layout_accepted"] is True and layout["contract"] == "cap-physical-page-spans-v1", "LAYOUT")
    check(base["document_id"] == layout["document_id"] and base["field_name"] == "text"
          and base["source"]["text_sha256"] == graph["source_text_sha256"] == layout["text_sha256"]
          and base["group_keys"] == layout["group_keys"] and graph["contract"] == "source-multiscale-v3", "LINEAGE")
    check(isinstance(group, str) and len(group) == 64, "GROUP")
    length = base["source_characters"]; check(length == layout["source_characters"], "LENGTH")
    pages = []; fragments = []; lengths = {}; page_by_id = {}
    namespace = identity([CONTRACT, base["source"], layout["artifacts"]])
    for p in layout["pages"]:
        spans = p["source_spans"]
        check(spans and all(type(a) is int and type(b) is int and 0 <= a < b <= length for a, b in spans)
              and all(a[1] < b[0] for a, b in zip(spans, spans[1:])), "PAGE_SPANS")
        node = dict(id=identity([namespace, p]), level="physical_page", page_order=p["page_order"], label=p["label"],
                    source_spans=spans, part_sha256=p["part_sha256"], visible_after=spans[-1][1])
        pages.append(node); page_by_id[node["id"]] = node; lengths[node["id"]] = sum(b-a for a, b in spans)
        fragments.extend((a, b, node["id"]) for a, b in spans)
    fragments.sort()
    check(tile([(a, b) for a, b, _ in fragments], length) and len(page_by_id) == len(pages), "COVERAGE")
    check([p["page_order"] for p in pages] == sorted({p["page_order"] for p in pages}), "PAGE_ORDER")
    edges = []; visits = 0
    for level in LEVELS:
        nodes = [n for n in graph["nodes"] if n["level"] == level]
        check(tile([(n["start"], n["end"]) for n in nodes], length), "BASE_PARTITION")
        joined = defaultdict(list); by_id = {n["id"]: n for n in nodes}; i = j = 0
        while i < len(nodes) and j < len(fragments):
            n = nodes[i]; a, b, page = fragments[j]; visits += 1
            lo, hi = max(n["start"], a), min(n["end"], b)
            if lo < hi:
                joined[(page, n["id"])].append([lo, hi])
            if n["end"] <= b:
                i += 1
            if b <= n["end"]:
                j += 1
        for (page, node), spans in joined.items():
            n = by_id[node]; overlap = sum(b-a for a, b in spans)
            whole_page, whole_node = overlap == lengths[page], overlap == n["end"]-n["start"]
            relation = "coextensive" if whole_page and whole_node else "contains" if whole_page or whole_node else "overlaps"
            left, right = (node, page) if whole_page and not whole_node else (page, node)
            edges.append(dict(relation=relation, **{"from": left, "to": right}, source_spans=spans))
    for a, b in zip(pages, pages[1:]):
        edges.append(dict(relation="next_physical_page", **{"from": a["id"], "to": b["id"]}))
    return dict(contract=CONTRACT, document_id=base["document_id"], field_name="text", source=base["source"],
                source_characters=length, overlap_group=group, group_keys=base["group_keys"],
                base_graph_sha256=sha(canonical(graph)), layout_sha256=sha(canonical(layout)),
                pages=pages, edges=edges, interval_visits=visits, attached=True,
                semantic_edges="pending:structural relations only", measurement_status="pending:frozen-instrument-readings",
                old_slot_reuse_allowed=False, scientific_admission=False, sealed_evaluation=False)


def causal_attachment(record, base_graph, cutoff, group):
    check(record["attached"] is True, "UNAVAILABLE")
    check(record["overlap_group"] == group, "GROUP")
    check(sha(canonical(base_graph)) == record["base_graph_sha256"], "BASE_HASH")
    check(type(cutoff) is int and 0 <= cutoff <= record["source_characters"], "CUTOFF")
    check(all(p["source_spans"] and p["visible_after"] == max(b for _, b in p["source_spans"])
              for p in record["pages"]), "VISIBILITY")
    pages = [p for p in record["pages"] if p["visible_after"] <= cutoff]
    visible = {p["id"] for p in pages} | {n["id"] for n in base_graph["nodes"] if n["end"] <= cutoff}
    return dict(pages=pages, edges=[e for e in record["edges"] if e["from"] in visible and e["to"] in visible])


def inputs(multiscale, layout, overlap, pins):
    roots = dict(multiscale=multiscale, layout=layout, overlap=overlap); reports = {}
    for name, root in roots.items():
        raw = (root/"report.json").read_bytes(); check(sha(raw) == pins[name], "PIN_"+name.upper())
        reports[name] = json.loads(raw)
    m, l, o = (reports[n] for n in ("multiscale", "layout", "overlap"))
    check(m["complete"] is True and not m["errors"] and o["complete"] is True and not o["errors"], "PARENT_INCOMPLETE")
    check(m["source_report_sha256"] == l["source_report_sha256"] == o["source_report_sha256"], "SOURCE_PARENT")
    check(m["geometry_contract"] == "source-multiscale-v3" and l["contract"] == "cap-physical-page-spans-v1"
          and o["schema"] == "evaluation-exact-overlap-v1", "CONTRACT")
    paths = dict(multiscale=multiscale/"legal.jsonl", layout=layout/"layouts.jsonl", overlap=overlap/"groups.jsonl")
    for name, rec in (("multiscale", m["files"]["legal"]), ("layout", l["file"]), ("overlap", o["files"]["groups.jsonl"])):
        path = paths[name]; check(file_sha(path) == rec["sha256"] and path.stat().st_size == rec["bytes"], "INPUT_HASH_"+name.upper())
    groups = {}
    with paths["overlap"].open("rb") as f:
        for line in f:
            row = json.loads(line)
            for document in row["document_ids"]:
                check(document not in groups, "DUPLICATE_GROUP")
                groups[document] = row["id"]
    check(len(groups) == o["counts"]["documents"], "GROUP_COVERAGE")
    return reports, paths, groups


def run(multiscale, layout, overlap, pins, output):
    started = time.perf_counter(); check(not output.exists(), "EXISTS")
    reports, paths, groups = inputs(multiscale, layout, overlap, pins)
    phases = dict(input_seconds=time.perf_counter()-started); output.mkdir(parents=True)
    counts = Counter(); errors = []; dest = output/"attachments.jsonl"; t = time.perf_counter()
    with paths["multiscale"].open("rb") as m, paths["layout"].open("rb") as l, dest.open("xb") as out:
        for line in m:
            base = json.loads(line); ll = next(l, None); check(ll is not None, "MISSING_LAYOUT")
            page = json.loads(ll); check(base["document_id"] == page["document_id"], "ROW_ORDER")
            counts["cases"] += 1
            try:
                check(base["document_id"] in groups, "MISSING_GROUP")
                if not page["layout_accepted"]:
                    raise ValueError("MNLM_PAGE_ATTACH_SOURCE_REFUSED:"+page["error"])
                result = attach(base, page, groups[base["document_id"]])
                counts.update(attached=1, pages=len(result["pages"]), edges=len(result["edges"]), interval_visits=result["interval_visits"])
            except (ValueError, KeyError, TypeError) as exc:
                result = dict(document_id=base["document_id"], attached=False, error=str(exc))
                counts["refused"] += 1; errors.append(result)
            out.write(canonical(result)+b"\n")
        check(next(l, None) is None, "EXTRA_LAYOUT")
    phases["join_publish_seconds"] = time.perf_counter()-t
    check(counts["cases"] == reports["layout"]["counts"]["cases"], "CASE_COVERAGE")
    report = dict(contract=CONTRACT, parent_report_sha256=pins, source_report_sha256=reports["layout"]["source_report_sha256"],
                  file=dict(name=dest.name, sha256=file_sha(dest), bytes=dest.stat().st_size), counts=dict(counts), errors=errors,
                  complete=not errors, scientific_admission=False, sealed_evaluation=False, served=False,
                  code_sha256=file_sha(Path(__file__)), timing=phases)
    wall = time.perf_counter()-started; phases.update(other_seconds=wall-sum(phases.values()), wall_seconds=wall)
    publish(output/"report.json", report)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("multiscale", "layout", "overlap", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    for name in ("multiscale", "layout", "overlap"):
        p.add_argument("--"+name+"-report-sha256", required=True)
    a = p.parse_args(); pins = {n: getattr(a, n+"_report_sha256") for n in ("multiscale", "layout", "overlap")}
    result = run(a.multiscale, a.layout, a.overlap, pins, a.output); print(json.dumps(result))
    if not result["complete"]:
        raise SystemExit(2)
