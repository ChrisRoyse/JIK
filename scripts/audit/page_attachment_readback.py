"""Audit every page/atom relation by binary-search enumeration of source spans."""
import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


def check(ok, code):
    if not ok:
        raise ValueError("MNLM_PAGE_READBACK_"+code)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def file_sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def audit_case(result, base, layout, group, exhaustive=False):
    graph = base["graph"]
    check(result["document_id"] == base["document_id"] == layout["document_id"] and result["source"] == base["source"]
          and result["base_graph_sha256"] == sha(canonical(graph)) and result["layout_sha256"] == sha(canonical(layout)), "PARENTS")
    check(result["source_characters"] == base["source_characters"] and result["overlap_group"] == group
          and result["group_keys"] == base["group_keys"], "GROUP_LENGTH")
    namespace = sha(canonical(["multiscale-physical-page-attachment-v1", base["source"], layout["artifacts"]]))
    expected = [dict(id=sha(canonical([namespace, p])), level="physical_page", page_order=p["page_order"], label=p["label"],
                     source_spans=p["source_spans"], part_sha256=p["part_sha256"], visible_after=max(b for _, b in p["source_spans"]))
                for p in layout["pages"]]
    check(result["pages"] == expected, "PAGES")
    joined = defaultdict(list); node_by_id = {n["id"]: n for n in graph["nodes"]}; counted_intersections = 0
    for level in ("sentence", "sentence_pair", "paragraph"):
        nodes = [n for n in graph["nodes"] if n["level"] == level]; ends = [n["end"] for n in nodes]
        for page in expected:
            for a, b in page["source_spans"]:
                candidates = nodes if exhaustive else nodes[bisect_right(ends, a):bisect_right(ends, b)+1]
                for node in candidates:
                    lo, hi = max(a, node["start"]), min(b, node["end"])
                    if lo < hi:
                        joined[(page["id"], node["id"])].append([lo, hi]); counted_intersections += 1
    sizes = {p["id"]: sum(b-a for a, b in p["source_spans"]) for p in expected}; edges = []
    for (page, key), spans in joined.items():
        spans.sort(); node = node_by_id[key]; length = sum(b-a for a, b in spans)
        page_complete = length == sizes[page]; atom_complete = length == node["end"]-node["start"]
        if page_complete and atom_complete:
            relation, left, right = "coextensive", page, key
        elif atom_complete:
            relation, left, right = "contains", page, key
        elif page_complete:
            relation, left, right = "contains", key, page
        else:
            relation, left, right = "overlaps", page, key
        edges.append(dict(relation=relation, **{"from": left, "to": right}, source_spans=spans))
    for a, b in zip(expected, expected[1:]):
        edges.append(dict(relation="next_physical_page", **{"from": a["id"], "to": b["id"]}))
    check(sorted(map(canonical, result["edges"])) == sorted(map(canonical, edges)), "EDGES")
    check(result["interval_visits"] == counted_intersections, "COST")
    check(result["contract"] == "multiscale-physical-page-attachment-v1" and result["field_name"] == "text" and result["attached"] is True
          and all(result[k] is False for k in ("old_slot_reuse_allowed", "scientific_admission", "sealed_evaluation"))
          and result["semantic_edges"] == "pending:structural relations only"
          and result["measurement_status"] == "pending:frozen-instrument-readings", "CLAIMS")
    return Counter(attached=1, pages=len(expected), edges=len(edges), interval_visits=counted_intersections)


def audit(root, multiscale, layout, overlap):
    report_bytes = (root/"report.json").read_bytes(); report = json.loads(report_bytes)
    parents = {}; roots = dict(multiscale=multiscale, layout=layout, overlap=overlap)
    for name, path in roots.items():
        raw = (path/"report.json").read_bytes(); check(sha(raw) == report["parent_report_sha256"][name], "PIN")
        parents[name] = json.loads(raw)
    check(all(r["source_report_sha256"] == report["source_report_sha256"] for r in parents.values()), "SOURCE")
    for path, item in ((multiscale/"legal.jsonl", parents["multiscale"]["files"]["legal"]),
                       (layout/"layouts.jsonl", parents["layout"]["file"]),
                       (overlap/"groups.jsonl", parents["overlap"]["files"]["groups.jsonl"]),
                       (root/"attachments.jsonl", report["file"])):
        check(path.stat().st_size == item["bytes"] and file_sha(path) == item["sha256"], "FILE_HASH")
    groups = {}
    with (overlap/"groups.jsonl").open("rb") as f:
        for line in f:
            row = json.loads(line)
            for doc in row["document_ids"]:
                check(doc not in groups, "DUPLICATE_GROUP"); groups[doc] = row["id"]
    counts = Counter(); errors = []
    with (multiscale/"legal.jsonl").open("rb") as mf, (layout/"layouts.jsonl").open("rb") as lf, (root/"attachments.jsonl").open("rb") as f:
        for line in mf:
            base = json.loads(line); pl = next(lf, None); out = next(f, None)
            check(pl is not None and out is not None, "MISSING_ROW")
            page, result = json.loads(pl), json.loads(out); counts["cases"] += 1
            check(result["document_id"] == page["document_id"] == base["document_id"], "ROW_ORDER")
            if page["layout_accepted"]:
                counts.update(audit_case(result, base, page, groups[base["document_id"]]))
            else:
                check(result == dict(document_id=base["document_id"], attached=False,
                                     error="MNLM_PAGE_ATTACH_SOURCE_REFUSED:"+page["error"]), "REFUSAL")
                errors.append(result); counts["refused"] += 1
        check(next(lf, None) is None and next(f, None) is None, "EXTRA_ROW")
    check(report["counts"] == dict(counts) and report["errors"] == errors and report["complete"] == (not errors), "COUNTS")
    check(report["contract"] == "multiscale-physical-page-attachment-v1" and
          all(report[k] is False for k in ("scientific_admission", "sealed_evaluation", "served")), "REPORT_CLAIMS")
    return dict(acceptance="author-checked", counts=dict(counts), every_relation_equal=True,
                report_sha256=sha(report_bytes), publication_complete=report["complete"], sealed_evaluation=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("publication", "multiscale", "layout", "overlap", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    a = p.parse_args(); result = audit(a.publication, a.multiscale, a.layout, a.overlap)
    with a.output.open("xb") as f:
        f.write(canonical(result)+b"\n")
    print(json.dumps(result))
