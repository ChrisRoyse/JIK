"""Independent source/archive, view, rationale and component reconstruction."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
import tarfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.multiscale_readback import audit_spans


def check(ok, code):
    if not ok:
        raise ValueError("MNLM_SCIFACT_READBACK_"+code)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def ident(value):
    return sha(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode())


def jsonl(path):
    return [json.loads(s) for s in path.read_bytes().splitlines()]


def raw_source(root, pin):
    check(sha((root/"report.json").read_bytes()) == pin, "SOURCE_PIN")
    report = json.loads((root/"report.json").read_bytes())
    expected = {"corpus.jsonl", "claims_train.jsonl", "claims_dev.jsonl", "claims_with_citances.jsonl",
                "data.tar.gz", "LICENSE.md", "README.md", "doc__data.md", "doc__claims-with-citances.md", "capture.json"}
    check(set(report["files"]) == expected and report["complete"] is True
          and report["provider_commit"] == "68b98a56d93e0f9da0d2aab4e6c3294699a0f72e"
          and report["scientific_admission"] is report["test_labels_used"] is report["world_truth_adjudicated"] is False,
          "SOURCE_CLAIM")
    raw = {}
    for name, saved in report["files"].items():
        raw[name] = (root/name).read_bytes()
        check(len(raw[name]) == saved["bytes"] and sha(raw[name]) == saved["sha256"], "SOURCE_FILE")
    capture = json.loads(raw["capture.json"])
    check(not capture["errors"] and capture["provider_commit"] == report["provider_commit"], "SOURCE_CAPTURE")
    for name, saved in capture["files"].items():
        check(saved["sha256"] == sha(raw[name]) and saved["bytes"] == len(raw[name]), "SOURCE_CAPTURE")
    check(b"CC BY 4.0" in raw["LICENSE.md"] and b"ODC-By 1.0" in raw["LICENSE.md"], "RIGHTS")
    with tarfile.open(root/"data.tar.gz", "r:gz") as archive:
        for name in ("corpus.jsonl", "claims_train.jsonl", "claims_dev.jsonl"):
            check(archive.extractfile("data/"+name).read() == raw[name], "ARCHIVE_BINDING")
    docs = {r["doc_id"]:r for r in [json.loads(s) for s in raw["corpus.jsonl"].splitlines()]}
    claims = [(split, json.loads(s)) for split in ("train", "dev")
              for s in raw["claims_"+split+".jsonl"].splitlines()]
    contexts = [json.loads(s) for s in raw["claims_with_citances.jsonl"].splitlines()]
    return docs, claims, contexts


def components(docs, claims, contexts):
    graph = defaultdict(set); links = set(); matches = {}; by_text = defaultdict(list)
    for i, context in enumerate(contexts):
        for j, item in enumerate(context["claims"]):
            by_text[item["text"].strip()].append((i,j))
    def edge(a, b):
        graph[a].add(b); graph[b].add(a); links.add((a,b))
    for split, c in claims:
        key = "claim:"+str(c["id"]); edge(key, "claim-text:"+ident(c["claim"].strip()))
        for d in c["cited_doc_ids"]:
            edge(key, "document:"+str(d)); edge("document:"+str(d), "abstract:"+ident(docs[d]["abstract"]))
        matches[c["id"]] = []
        for i,j in by_text[c["claim"].strip()]:
            edge(key, "context:"+str(i)); edge("context:"+str(i), "document:"+str(contexts[i]["s2orc_id"]))
            matches[c["id"]].append(dict(row=i, variant=j, match="exact" if c["claim"] == contexts[i]["claims"][j]["text"]
                                         else "exterior-whitespace"))
    groups = {}; membership = {}; seen = set()
    for key in sorted(graph):
        if key in seen:
            continue
        queue = [key]; members = []; seen.add(key)
        while queue:
            k = queue.pop(); members.append(k)
            for neighbor in graph[k]:
                if neighbor not in seen:
                    seen.add(neighbor); queue.append(neighbor)
        members.sort(); group = ident(["scifact-relations-v2", members])
        bucket = int(ident(["scifact-development-25", group])[:8],16)%100
        groups[group] = dict(members=members, partition="train" if bucket < 70 else "calibration" if bucket < 85 else "evaluation")
        membership.update((k,group) for k in members)
    return dict(links=[list(v) for v in sorted(links)], groups=groups), membership, matches


def views(actual, doc):
    check(actual["source_sentences"] == doc["abstract"] and actual["paragraph_status"] == "absent:provider-sentence-array"
          and actual["page_status"] == "absent:no-source-pagination", "SOURCE_SENTENCES")
    n = len(doc["abstract"]); source = ident(doc)
    ranges = [("source_sentence",i,i+1) for i in range(n)] + [("sentence_pair",i,min(n,i+2)) for i in range(0,n,2)]
    ranges.append(("abstract_context",0,n))
    expected = [dict(id=ident([source,level,a,b]), level=level, sentence_range=[a,b],
                     content_sha256=ident(doc["abstract"][a:b])) for level,a,b in ranges]
    check(actual["nodes"] == expected, "NATIVE_VIEWS")
    levels = {k:[v for v in expected if v["level"] == k] for k in ("source_sentence","sentence_pair","abstract_context")}
    edges = set(); context = levels["abstract_context"][0]
    for level in ("source_sentence", "sentence_pair"):
        members = levels[level]
        edges.update((a["id"],b["id"],"next") for a,b in zip(members,members[1:]))
        for child in members:
            edges.add((context["id"],child["id"],"coextensive" if child["sentence_range"] == [0,n] else "contains"))
    for child in levels["source_sentence"]:
        pair = levels["sentence_pair"][child["sentence_range"][0]//2]
        edges.add((pair["id"],child["id"],"coextensive" if pair["sentence_range"] == child["sentence_range"] else "contains"))
    saved = [(e["origin"],e["target"],e["kind"]) for e in actual["edges"]]
    check(len(saved) == len(set(saved)) and set(saved) == edges, "NATIVE_EDGES")


def read(source, source_pin, root, pin, context=None):
    docs, claims, contexts = context if context is not None else raw_source(source, source_pin)
    check(sha((root/"report.json").read_bytes()) == pin, "PUBLICATION_PIN")
    report = json.loads((root/"report.json").read_bytes())
    check(report["schema"] == "scifact-relations-v2" and report["source_sha256"] == source_pin, "LINEAGE")
    flags = ("scientific_admission","meaning_asserted","sealed_evaluation","world_truth_adjudicated",
             "gold_in_model_input","near_duplicate_separation","exposure_separation","time_separation")
    check(all(report[k] is False for k in flags) and report["complete"] is True and report["native_text_floor"] == 5
          and report["benchmark_role"] == "secondary-development-only", "CLAIMS")
    expected = {"documents.jsonl","claims.jsonl","contexts.jsonl","cases.jsonl","labels.jsonl","groups.json"}
    check(set(report["files"]) == expected, "FILES")
    for name, saved in report["files"].items():
        data = (root/name).read_bytes()
        check(len(data) == saved["bytes"] and sha(data) == saved["sha256"], "FILE_HASH")
    groups, membership, matches = components(docs, claims, contexts)
    check(json.loads((root/"groups.json").read_bytes()) == groups, "COMPONENTS")
    counts = Counter(); official = defaultdict(set)
    dr = jsonl(root/"documents.jsonl"); cr = jsonl(root/"claims.jsonl"); ctx = jsonl(root/"contexts.jsonl")
    used = sorted({d for _,c in claims for d in c["cited_doc_ids"]})
    check([d["id"] for d in dr] == used, "DOCUMENT_POPULATION")
    for d in dr:
        raw = docs[d["id"]]
        check(set(d) == {"id","title","source_row_sha256","group","views"} and d["source_row_sha256"] == ident(raw)
              and d["title"] == raw["title"] and d["group"] == membership["document:"+str(d["id"])], "DOCUMENT_BINDING")
        views(d["views"], raw)
        counts.update(documents=1,sentence_views=len(raw["abstract"]),views=len(d["views"]["nodes"]),
                      structural_edges=len(d["views"]["edges"]))
    check([c["id"] for c in cr] == [c["id"] for _,c in claims], "CLAIM_POPULATION")
    cases = []; labels = []
    for row, (split, c) in zip(cr, claims):
        group = membership["claim:"+str(c["id"])]; partition = groups["groups"][group]["partition"]
        check(row == dict(id=c["id"], text=c["claim"], source_row_sha256=ident(c), official_split=split,
                          group=group, context_links=matches[c["id"]], context_status="linked" if matches[c["id"]] else "unmatched"),
              "CLAIM_BINDING")
        counts["claims_"+partition] += 1; official[group].add(split)
        counts["claims_context_"+("linked" if matches[c["id"]] else "unmatched")] += 1
        counts["citation_occurrences"] += len(c["cited_doc_ids"])
        counts["repeated_citation_occurrences"] += len(c["cited_doc_ids"])-len(set(c["cited_doc_ids"]))
        for doc_id in dict.fromkeys(c["cited_doc_ids"]):
            rationales = c["evidence"].get(str(doc_id),[])
            label = rationales[0]["label"] if rationales else "NOT_ENOUGH_EVIDENCE"
            check(all(r["label"] == label and r["sentences"] and len(set(r["sentences"])) == len(r["sentences"])
                      and all(type(i) is int and 0 <= i < len(docs[doc_id]["abstract"]) for i in r["sentences"])
                      for r in rationales), "ORACLE_SCOPE")
            key = ident(["scifact-relations-v2",source_pin,c["id"],doc_id])
            cases.append(dict(id=key,claim=c["id"],document=doc_id,group=group,partition=partition,
                              citation_occurrences=[i for i,d in enumerate(c["cited_doc_ids"]) if d == doc_id]))
            labels.append(dict(case=key,label=label,rationales=rationales,semantics="OR-of-AND-source-sentence-sets",
                               scope="provider-annotation-of-this-claim-and-abstract",world_truth=False))
            counts[label] += 1; counts["rationale_sets"] += len(rationales)
            counts["joint_rationale_sets"] += sum(len(r["sentences"]) > 1 for r in rationales)
    check(jsonl(root/"cases.jsonl") == cases, "CASE_BINDING")
    check(jsonl(root/"labels.jsonl") == labels, "RATIONALE_BINDING")
    used_contexts = sorted({r["row"] for v in matches.values() for r in v})
    check([r["row"] for r in ctx] == used_contexts, "CONTEXT_POPULATION")
    for r in ctx:
        c = contexts[r["row"]]; text = c["citation_paragraph"]; group = membership["context:"+str(r["row"])]
        check(r["source"] == c and r["source_row_sha256"] == ident(c) and r["group"] == group
              and r["role"] == "provenance_only_not_model_input", "CONTEXT_BINDING")
        expected_spans = []; start = 0
        while c["citance"] in text[start:]:
            a = text.index(c["citance"],start); start = a+len(c["citance"]); expected_spans.append([a,start])
        check(r["citance_spans"] == expected_spans, "CITANCE_SPANS")
        audit_spans(r["graph"], text, dict(text_sha256=sha(text.encode())), group,
                    source_id=ident(["scifact-relations-v2","context",r["row"],ident(c)]))
        counts.update(context_views=len(r["graph"]["nodes"]),context_edges=len(r["graph"]["edges"]))
    counts.update(claims=len(claims),cases=len(cases),contexts=len(ctx),groups=len(groups["groups"]),
                  official_split_crossing_groups=sum(len(s)>1 for s in official.values()))
    check(report["counts"] == dict(counts), "TOTALS")
    return dict(counts=dict(counts),source_sha256=source_pin,publication_sha256=pin,
                scientific_admission=False,meaning_asserted=False,all_source_bindings_equal=True,
                label_partitions=dict(Counter(c["partition"] for c in cases)))
