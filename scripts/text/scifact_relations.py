"""Source-native claim, sentence, pair and context views with isolated judgments."""
import argparse
from collections import Counter, defaultdict
import copy
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from senses.contract import canonical
from senses.trajectory_population import publish
from text.evaluation_multiscale import file_sha
from text.multiscale_spans import build, identity
from text.scifact_source import require, rows

SCHEMA = "scifact-relations-v2"
FILES = ("documents.jsonl", "claims.jsonl", "contexts.jsonl", "cases.jsonl", "labels.jsonl", "groups.json")


def source(root, pin):
    require(file_sha(root/"report.json") == pin, "SOURCE_PIN")
    report = json.loads((root/"report.json").read_bytes())
    require(report["complete"] and report["schema"] == "scifact-source-v1"
            and not report["test_labels_used"] and not report["scientific_admission"], "SOURCE_CLAIM")
    for name, saved in report["files"].items():
        require(Path(name).name == name and (root/name).stat().st_size == saved["bytes"]
                and file_sha(root/name) == saved["sha256"], "SOURCE_FILE")
    docs = {d["doc_id"]: d for d in rows(root/"corpus.jsonl")}
    claims = [(s, c) for s in ("train", "dev") for c in rows(root/("claims_"+s+".jsonl"))]
    contexts = rows(root/"claims_with_citances.jsonl")
    return report, docs, claims, contexts


def lineage(docs, claims, contexts):
    refs = []; lookup = defaultdict(list); parent = {}; links = []
    for i, context in enumerate(contexts):
        for j, item in enumerate(context["claims"]):
            lookup[item["text"].strip()].append((i, j))

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x

    def link(a, b):
        links.append([a, b]); x, y = find(a), find(b)
        parent[max(x, y)] = min(x, y)

    for split, claim in claims:
        key = "claim:"+str(claim["id"])
        link(key, "claim-text:"+identity(claim["claim"].strip()))
        for d in claim["cited_doc_ids"]:
            doc = "document:"+str(d); link(key, doc)
            link(doc, "abstract:"+identity(docs[d]["abstract"]))
        matches = lookup[claim["claim"].strip()]
        for i, j in matches:
            context = contexts[i]; link(key, "context:"+str(i))
            link("context:"+str(i), "document:"+str(context["s2orc_id"]))
        refs.append([dict(row=i, variant=j, match="exact" if contexts[i]["claims"][j]["text"] == claim["claim"]
                          else "exterior-whitespace") for i, j in matches])
    components = defaultdict(list)
    for key in sorted(parent):
        components[find(key)].append(key)
    groups = {}; membership = {}
    for members in components.values():
        group = identity([SCHEMA, members]); bucket = int(identity(["scifact-development-25", group])[:8], 16)%100
        partition = "train" if bucket < 70 else "calibration" if bucket < 85 else "evaluation"
        groups[group] = dict(members=members, partition=partition)
        membership.update((member, group) for member in members)
    return refs, dict(links=sorted({tuple(v) for v in links}), groups=groups), membership


def abstract_views(doc):
    digest = identity(doc); n = len(doc["abstract"]); nodes = []; edges = []
    def node(level, start, end):
        value = dict(id=identity([digest, level, start, end]), level=level, sentence_range=[start, end],
                     content_sha256=identity(doc["abstract"][start:end]))
        nodes.append(value); return value
    sentences = [node("source_sentence", i, i+1) for i in range(n)]
    pairs = [node("sentence_pair", i, min(i+2, n)) for i in range(0, n, 2)]
    context = node("abstract_context", 0, n)
    def edge(a, b, kind):
        edges.append(dict(origin=a["id"], target=b["id"], kind=kind))
    for group in (sentences, pairs):
        for a, b in zip(group, group[1:]):
            edge(a, b, "next")
        for child in group:
            edge(context, child, "coextensive" if context["sentence_range"] == child["sentence_range"] else "contains")
    for pair in pairs:
        for child in sentences[slice(*pair["sentence_range"])]:
            edge(pair, child, "coextensive" if pair["sentence_range"] == child["sentence_range"] else "contains")
    return dict(nodes=nodes, edges=edges, paragraph_status="absent:provider-sentence-array",
                page_status="absent:no-source-pagination", source_sentences=doc["abstract"])


def project(root, pin, output):
    started = time.perf_counter(); output.mkdir(parents=True, exist_ok=False); phases = {}
    report, docs, claims, contexts = source(root, pin)
    phases["source_validation"] = time.perf_counter()-started; tick = time.perf_counter()
    refs, groups, membership = lineage(docs, claims, contexts)
    phases["component_closure"] = time.perf_counter()-tick; tick = time.perf_counter()
    counts = Counter(); used_docs = sorted({d for _, c in claims for d in c["cited_doc_ids"]})
    used_contexts = sorted({r["row"] for ref in refs for r in ref})
    def write(name, values):
        with (output/name).open("xb") as out:
            for value in values:
                out.write(canonical(value)+b"\n")
    doc_rows = []
    for key in used_docs:
        d = docs[key]; view = abstract_views(d)
        doc_rows.append(dict(id=key, title=d["title"], source_row_sha256=identity(d),
                             group=membership["document:"+str(key)], views=view))
        counts.update(documents=1, sentence_views=len(d["abstract"]), views=len(view["nodes"]), structural_edges=len(view["edges"]))
    write("documents.jsonl", doc_rows)
    context_rows = []
    for i in used_contexts:
        c = contexts[i]; group = membership["context:"+str(i)]; text = c["citation_paragraph"]
        graph = build(text, identity([SCHEMA, "context", i, identity(c)]), group)
        positions = []; offset = 0
        while (offset := text.find(c["citance"], offset)) >= 0:
            positions.append([offset, offset+len(c["citance"])]); offset += len(c["citance"])
        require(c["citance"] and positions, "CITANCE_SPAN")
        context_rows.append(dict(row=i, source=c, source_row_sha256=identity(c), group=group,
                                 citance_spans=positions, graph=graph, role="provenance_only_not_model_input"))
        counts.update(context_views=len(graph["nodes"]), context_edges=len(graph["edges"]))
    write("contexts.jsonl", context_rows)
    case_rows = []; label_rows = []; claim_rows = []; official = defaultdict(set)
    for (split, c), matches in zip(claims, refs):
        group = membership["claim:"+str(c["id"])]; partition = groups["groups"][group]["partition"]
        official[group].add(split)
        claim_rows.append(dict(id=c["id"], text=c["claim"], source_row_sha256=identity(c), official_split=split,
                               group=group, context_links=matches, context_status="linked" if matches else "unmatched"))
        counts["claims_"+partition] += 1; counts["claims_context_"+("linked" if matches else "unmatched")] += 1
        counts["citation_occurrences"] += len(c["cited_doc_ids"])
        counts["repeated_citation_occurrences"] += len(c["cited_doc_ids"])-len(set(c["cited_doc_ids"]))
        for doc_id in dict.fromkeys(c["cited_doc_ids"]):
            sets = c["evidence"].get(str(doc_id), [])
            values = sorted({s["label"] for s in sets})
            require(len(values) <= 1, "CONFLICTING_JUDGMENTS")
            label = values[0] if values else "NOT_ENOUGH_EVIDENCE"
            key = identity([SCHEMA, pin, c["id"], doc_id])
            case_rows.append(dict(id=key, claim=c["id"], document=doc_id, group=group, partition=partition,
                                  citation_occurrences=[i for i,d in enumerate(c["cited_doc_ids"]) if d == doc_id]))
            label_rows.append(dict(case=key, label=label, rationales=sets, semantics="OR-of-AND-source-sentence-sets",
                                   scope="provider-annotation-of-this-claim-and-abstract", world_truth=False))
            counts[label] += 1; counts["rationale_sets"] += len(sets)
            counts["joint_rationale_sets"] += sum(len(s["sentences"]) > 1 for s in sets)
    write("claims.jsonl", claim_rows); write("cases.jsonl", case_rows); write("labels.jsonl", label_rows)
    publish(output/"groups.json", groups)
    counts.update(claims=len(claims), cases=len(case_rows), contexts=len(used_contexts), groups=len(groups["groups"]),
                  official_split_crossing_groups=sum(len(s) > 1 for s in official.values()))
    phases["typed_projection_and_write"] = time.perf_counter()-tick; tick = time.perf_counter()
    require(file_sha(root/"report.json") == pin, "SOURCE_DRIFT")
    for name, saved in report["files"].items():
        require(file_sha(root/name) == saved["sha256"], "SOURCE_DRIFT")
    files = {name:dict(sha256=file_sha(output/name), bytes=(output/name).stat().st_size) for name in FILES}
    phases["drift_and_publication_hashes"] = time.perf_counter()-tick; wall = time.perf_counter()-started
    result = dict(schema=SCHEMA, source_sha256=pin, files=files, counts=counts, complete=True,
                  native_text_floor=5, scientific_admission=False, meaning_asserted=False, sealed_evaluation=False,
                  world_truth_adjudicated=False, gold_in_model_input=False, near_duplicate_separation=False,
                  exposure_separation=False, time_separation=False, benchmark_role="secondary-development-only",
                  phases_seconds=phases, wall_seconds=wall, phase_coverage=sum(phases.values())/wall,
                  code_sha256=file_sha(Path(__file__)))
    publish(output/"report.json", result); return result


class Inputs:
    """Accept only an independently audited publication. Never load gold/context files."""
    def __init__(self, root, pin):
        require(file_sha(root/"report.json") == pin, "INPUT_PIN")
        report = json.loads((root/"report.json").read_bytes())
        require(report["schema"] == SCHEMA and not report["gold_in_model_input"]
                and not report["scientific_admission"] and report["native_text_floor"] == 5, "INPUT_CLAIM")
        self.tables = {}
        for name in ("claims", "documents", "cases"):
            f = root/(name+".jsonl")
            require(file_sha(f) == report["files"][f.name]["sha256"], "INPUT_FILE")
            values = rows(f); self.tables[name] = {r["id"]:r for r in values}
            require(len(values) == len(self.tables[name]), "INPUT_DUPLICATE")

    def read(self, key, *, partition):
        require(key in self.tables["cases"], "INPUT_CASE")
        case = self.tables["cases"][key]
        require(case["partition"] == partition, "INPUT_PARTITION")
        claim = self.tables["claims"][case["claim"]]; doc = self.tables["documents"][case["document"]]
        require(case["group"] == claim["group"] == doc["group"], "INPUT_GROUP")
        return copy.deepcopy(dict(claim_text=claim["text"], title=doc["title"], source_sentences=doc["views"]["source_sentences"],
                    views=doc["views"]["nodes"], structural_edges=doc["views"]["edges"],
                    decision="refuse", code="MNLM_SCIFACT_UNADMITTED_PANEL", native_text_floor=5))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for key in ("source", "output"):p.add_argument("--"+key, type=Path, required=True)
    p.add_argument("--source-sha256", required=True); a = p.parse_args()
    result = project(a.source, a.source_sha256, a.output)
    print(json.dumps({k:result[k] for k in ("counts", "wall_seconds", "phase_coverage")}))
