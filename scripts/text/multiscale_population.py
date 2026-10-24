"""Pin a complete real multiscale proof corpus; content reuse is not pooling."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from senses.trajectory_population import publish
from audit.evaluation_multiscale_readback import audit_field
from audit.page_attachment_readback import audit_case

SCHEMA = "multiscale-reading-population-v1"
DATA = ("sources.json", "fields.json", "layouts.json", "attachments.json",
        "groups.json", "units.json", "nodes.json", "edges.json")


def check(ok, code):
    if not ok:
        raise ValueError("MNLM_MULTI_READINGS_"+code)


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def rows(path):
    with path.open(encoding="utf-8") as f:
        yield from map(json.loads, f)


def load_pinned(root, meta):
    path = root/meta.get("file", meta.get("name"))
    check(path.stat().st_size == meta["bytes"] and sha(path) == meta["sha256"], "PARENT_FILE")
    return list(rows(path))


def derive(sources, fields, layouts, attachments, groups):
    source = {r["id"]: r for r in sources}
    check(len(source) == len(sources), "DUPLICATE_SOURCE")
    by_layout = {r["document_id"]: r for r in layouts}
    by_attach = {r["document_id"]: r for r in attachments}
    expected_fields = {(r["id"], f["name"]) for r in sources for f in r["fields"]}
    check({(f["document_id"], f["field_name"]) for f in fields} == expected_fields
          and len(fields) == len(expected_fields), "FIELD_COVERAGE")
    units, nodes, edges, unit_index = [], [], [], {}
    def add(base, node, spans, hashes):
        check(len(spans) == len(hashes), "PART_COUNT")
        document = base["document_id"]
        field = next(f for f in source[document]["fields"] if f["name"] == base["field_name"])
        indices = []
        for (a, b), expected_hash in zip(spans, hashes):
            text = field["text"][a:b]
            check(digest(text) == expected_hash, "PART_HASH")
            if not text.strip():
                indices.append(None); continue
            if expected_hash not in unit_index:
                unit_index[expected_hash] = len(units)
                units.append(dict(sha256=expected_hash, text=text, utf8_bytes=len(text.encode())))
            index = unit_index[expected_hash]
            check(units[index]["text"] == text, "HASH_COLLISION")
            indices.append(index)
        nodes.append(dict(id=node["id"], document_id=document, field_name=field["name"],
                          level=node["level"], source_spans=spans, unit_indices=indices,
                          overlap_group=groups[document], visible_after=max(b for _, b in spans)))
    for base in fields:
        document = base["document_id"]; row = source[document]
        field = next(f for f in row["fields"] if f["name"] == base["field_name"])
        audit_field(base, row, field)
        if base["graph"] is None:
            continue
        for node in base["graph"]["nodes"]:
            add(base, node, [[node["start"], node["end"]]], [node["content_sha256"]])
        edges.extend(dict(e) for e in base["graph"]["edges"])
        if document in by_attach:
            check(base["field_name"] == "text", "PAGE_FIELD")
            attachment = by_attach[document]
            audit_case(attachment, base, by_layout[document], groups[document], exhaustive=True)
            for page in attachment["pages"]:
                add(base, page, page["source_spans"], page["part_sha256"])
            edges.extend(dict(e) for e in attachment["edges"])
    by_node = {n["id"]: n for n in nodes}
    check(len(by_node) == len(nodes), "NODE_COLLISION")
    for e in edges:
        left, right = by_node[e["from"]], by_node[e["to"]]
        check(left["document_id"] == right["document_id"] and
              left["field_name"] == right["field_name"], "EDGE_SCOPE")
        e.update(overlap_group=left["overlap_group"],
                 visible_after=max(left["visible_after"], right["visible_after"]))
    return units, nodes, edges


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(); check(not a.output.exists(), "OUTPUT_EXISTS")
    started = time.perf_counter(); cfg = json.loads(a.config.read_bytes())
    roots, reports = {}, {}
    for name, entry in cfg["parents"].items():
        roots[name] = Path(entry["root"])
        check(sha(roots[name]/"report.json") == entry["sha256"], "PARENT_PIN")
        reports[name] = json.loads((roots[name]/"report.json").read_bytes())
    for name in ("multiscale", "layout", "attachment", "overlap"):
        check(reports[name]["source_report_sha256"] == cfg["parents"]["capture"]["sha256"], "PARENT_SOURCE")
    selection = {r["domain"]: r["document_id"] for r in cfg["selection"]}
    check(len(selection) == 6 and len(set(selection.values())) == 6, "SELECTION")
    sources, fields = [], []
    for domain, doc in sorted(selection.items()):
        sr = load_pinned(roots["capture"], reports["capture"]["sources"][domain])
        gr = load_pinned(roots["multiscale"], reports["multiscale"]["files"][domain])
        selected = [r for r in sr if r["id"] == doc]
        check(len(selected) == 1, "SOURCE_SELECTION")
        sources.extend(selected); fields.extend(r for r in gr if r["document_id"] == doc)
    layouts = [r for r in load_pinned(roots["layout"], reports["layout"]["file"])
               if r["document_id"] in selection.values() and r["layout_accepted"]]
    attachments = [r for r in load_pinned(roots["attachment"], reports["attachment"]["file"])
                   if r["document_id"] in selection.values() and r["attached"]]
    groups = {doc: r["id"] for r in load_pinned(roots["overlap"],
              dict(reports["overlap"]["files"]["groups.jsonl"], file="groups.jsonl"))
              for doc in r["document_ids"] if doc in selection.values()}
    check(len(groups) == 6 and len(layouts) == len(attachments) == 1, "GROUP_PAGE_COVERAGE")
    loaded = time.perf_counter()
    units, nodes, edges = derive(sources, fields, layouts, attachments, groups)
    built = time.perf_counter(); a.output.mkdir(parents=True)
    for name, value in zip(DATA, (sources, fields, layouts, attachments, groups, units, nodes, edges)):
        publish(a.output/name, value)
    for name, entry in cfg["parents"].items():
        check(sha(roots[name]/"report.json") == entry["sha256"], "PARENT_DRIFT")
    files = {n: dict(sha256=sha(a.output/n), bytes=(a.output/n).stat().st_size) for n in DATA}
    elapsed = time.perf_counter()-started
    report = dict(schema=SCHEMA, parents=cfg["parents"], selection=cfg["selection"],
                  files=files,
                  documents=len(sources), fields=len(fields), nodes=len(nodes), edges=len(edges),
                  units=len(units), unique_characters=sum(len(u["text"]) for u in units),
                  levels=dict(Counter(n["level"] for n in nodes)),
                  page_parts=sum(len(n["source_spans"]) for n in nodes if n["level"] == "physical_page"),
                  empty_parts=sum(i is None for n in nodes for i in n["unit_indices"]),
                  complete=True, scientific_admission=False, sealed_evaluation=False, served=False,
                  page_join="none:each-disjoint-source-part-retained", meaning_asserted=False,
                  timing=dict(input_seconds=loaded-started, derive_seconds=built-loaded,
                              write_seconds=elapsed-(built-started), wall_seconds=elapsed))
    publish(a.output/"report.json", report)
    print(json.dumps({k:v for k,v in report.items() if k not in ("parents", "selection", "files")}))


if __name__ == "__main__":
    main()
