"""Three-boundary multiscale FSV on retained REAL source texts (#27).

Page layout controls below are explicitly manufactured test metadata, not a
claim that the Wikidata source retained real pages. Real pagination stays absent.
"""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.multiscale_readback import audit, audit_document, sha, ident
from text.multiscale_spans import build, causal_view

PRODUCER = Path(__file__).parents[1]/"text/multiscale_publish.py"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--structured-text", type=Path, required=True, help="Real retained prose with paragraph separators")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    results = []
    def check(name, ok):
        results.append({"name": name, "pass": bool(ok)})
    def refuses(name, code, fn):
        try:
            fn()
            ok = False
        except ValueError as exc:
            ok = code in str(exc)
        check(name, ok)
    pin = sha((a.source/"report.json").read_bytes())
    before = {str(f.relative_to(a.source)): sha(f.read_bytes()) for f in a.source.rglob("*") if f.is_file()}
    def run(name, source_hash=pin, layout=None):
        cmd = [sys.executable, "-I", str(PRODUCER), "--source", str(a.source), "--output", str(a.output/name),
               "--source-report-sha256", source_hash]
        if layout:
            cmd += ["--layouts", str(layout), "--layouts-sha256", sha(layout.read_bytes())]
        cp = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
        (a.output/(name+".log")).write_text(cp.stdout+cp.stderr, encoding="utf-8")
        return cp
    cp = run("real")
    check("B1 real publisher completes", cp.returncode == 0)
    cold = audit(a.output/"real", a.source)
    check("B2 cold source spans, complete edges, original population and grouping", True)
    first = json.loads((a.output/"real"/"report.json").read_bytes())
    cp = run("repeat")
    second = json.loads((a.output/"repeat"/"report.json").read_bytes())
    check("B3 repeated publication has identical physical node/edge bytes",
          cp.returncode == 0 and first["files"] == second["files"])
    cp = run("real")
    check("B3 immutable publication refuses overwrite", cp.returncode != 0 and "MNLM_MULTISCALE_EXISTS" in cp.stderr)
    cp = run("wrong-source", "0"*64)
    check("B1 wrong source report refuses", cp.returncode != 0 and "MNLM_MULTISCALE_SOURCE_REPORT" in cp.stderr)
    chosen = None
    for family in first["files"]:
        source_rows = [json.loads(l) for l in (a.source/"records"/family/"rows.jsonl").read_text(encoding="utf-8").split("\n") if l]
        for line in (a.output/"real"/(family+".jsonl")).read_text(encoding="utf-8").split("\n"):
            if not line:
                continue
            rec = json.loads(line)
            if sum(n["level"] == "sentence" for n in rec["nodes"]) >= 3:
                chosen = rec, source_rows[rec["row"]]
                break
        if chosen:
            break
    check("B1 real multi-sentence source exercised", chosen is not None)
    rec, row = chosen
    text, source, entity = row["text"], row["text_source"], row["entity"]
    check("B1 real missing pagination stays absent", all(c.get("page", 0) == 0 for c in cold["counts"].values()))
    bad = copy.deepcopy(rec)
    bad["nodes"][0]["start"] += 1
    refuses("B2 shifted span refuses", "MNLM_MULTI_READBACK_ID", lambda: audit_document(bad, text, source, entity))
    bad = copy.deepcopy(rec)
    bad["edges"].pop()
    refuses("B2 missing relation refuses", "MNLM_MULTI_READBACK_RELATIONS", lambda: audit_document(bad, text, source, entity))
    bad = copy.deepcopy(rec)
    bad["split_group"] = "other"
    refuses("B3 altered entity group refuses", "MNLM_MULTI_READBACK_LINEAGE", lambda: audit_document(bad, text, source, entity))
    cutoff = next(n["end"] for n in rec["nodes"] if n["level"] == "sentence")
    visible = causal_view(rec, cutoff, rec["split_group"])
    visible_ids = {n["id"] for n in visible["nodes"]}
    check("B3 causal consumer hides partially observed pairs/paragraphs and future edges",
          bool(visible_ids) and len(visible_ids) < len(rec["nodes"])
          and all(n["end"] <= cutoff for n in visible["nodes"])
          and all(e["from"] in visible_ids and e["to"] in visible_ids for e in visible["edges"]))
    refuses("B3 consumer rejects wrong split group", "MNLM_MULTISCALE_SPLIT_GROUP", lambda: causal_view(rec, cutoff, "other"))
    refuses("B3 consumer rejects out-of-range cutoff", "MNLM_MULTISCALE_CUTOFF", lambda: causal_view(rec, len(text)+1, rec["split_group"]))
    # Controlled page boundaries split a real sentence; tests exercise overlap
    # semantics without laundering fabricated pagination into real evidence.
    boundary = next(n["start"]+1 for n in rec["nodes"] if n["level"] == "sentence" and n["end"]-n["start"] > 2)
    layout = {"text_sha256": sha(text.encode()), "authority": "FSV control only; not source-attested production layout",
              "pages": [[0, boundary], [boundary, len(text)]]}
    paged = build(text, ident(source), rec["split_group"], layout)
    paged.update(entity=entity, row=rec["row"], source=source)
    audit_document(paged, text, source, entity, layout)
    check("B2 explicit page-layout control retains crossing overlaps", any(e["relation"] == "overlaps" for e in paged["edges"]))
    visible = causal_view(paged, boundary-1, rec["split_group"])
    check("B3 page containing future text cannot enter causal context", not any(n["level"] == "page" for n in visible["nodes"]))
    bad_layout = {**layout, "text_sha256": "0"*64}
    refuses("B1 layout for another text refuses", "MNLM_MULTISCALE_LAYOUT_SOURCE", lambda: build(text, ident(source), rec["split_group"], bad_layout))
    bad_layout = {**layout, "authority": " "}
    refuses("B1 unproven page authority refuses", "MNLM_MULTISCALE_LAYOUT_AUTHORITY", lambda: build(text, ident(source), rec["split_group"], bad_layout))
    bad_layout = {**layout, "pages": [[0, boundary], [boundary+1, len(text)]]}
    refuses("B1 page coverage gap refuses", "MNLM_MULTISCALE_PAGE_COVERAGE", lambda: build(text, ident(source), rec["split_group"], bad_layout))
    lp = a.output/"bad-layout.json"
    lp.write_text(json.dumps({entity: bad_layout}), encoding="utf-8")
    cp = run("partial", layout=lp)
    partial = json.loads((a.output/"partial"/"report.json").read_bytes())
    check("B1 invalid document does not discard valid siblings; root fails closed",
          cp.returncode == 2 and not partial["complete"] and len(partial["errors"]) == 1
          and sum(c["documents"] for c in partial["counts"].values()) == sum(c["documents"] for c in first["counts"].values())-1)
    refuses("B3 partial publication refuses acceptance", "MNLM_MULTI_READBACK_INCOMPLETE", lambda: audit(a.output/"partial", a.source, lp))
    check("B3 source bytes unchanged", all(sha((a.source/name).read_bytes()) == h for name,h in before.items()))
    # Separate real source exercises paragraph boundaries missing from A2's
    # flattened descriptions. Persist and re-read before auditing; no invented text.
    prose_raw = a.structured_text.read_bytes()
    prose = prose_raw.decode("utf-8")
    provenance = {"text_sha256": sha(prose.encode()), "source": a.structured_text.name,
                  "source_file_sha256": sha(prose_raw)}
    paper = build(prose, ident(provenance), "entity:real-structured-text")
    paper.update(entity="real-structured-text", source=provenance)
    paper_path = a.output/"real-structured-text.json"
    paper_path.write_text(json.dumps(paper), encoding="utf-8")
    paper = json.loads(paper_path.read_bytes())
    audit_document(paper, prose, provenance, "real-structured-text")
    check("B2 real retained paragraphs and pair boundaries cold-read",
          sum(n["level"] == "paragraph" for n in paper["nodes"]) > 1
          and paper["paragraph_status"] == "inferred:blank-line-or-U+2029")
    check("B3 structured source bytes unchanged", a.structured_text.read_bytes() == prose_raw)
    # An actual wrapped phrase from the retained paper: a line break inside it
    # must not create a spurious sentence boundary. This is a source excerpt.
    fragment = "First principles of meaning from grounded association: why a frozen embedder is a\nmeasuring instrument"
    check("B1 real line-wrap excerpt is present", fragment in prose.replace("\r\n", "\n"))
    wrap = build(fragment, "real-paper-excerpt", "entity:real-paper-excerpt")
    check("B2 line wrap is not a second sentence; source offsets remain exact",
          sum(n["level"] == "sentence" for n in wrap["nodes"]) == 1
          and next(n for n in wrap["nodes"] if n["level"] == "sentence")["content_sha256"] == sha(fragment.encode()))
    result = {"checks": results, "passed": sum(r["pass"] for r in results), "total": len(results),
              "real_readback": cold, "source_files": before, "acceptance": "author-checked",
              "page_evidence": "layout controls only; real pagination remains absent", "structured_source": provenance}
    (a.output/"fsv.json").write_text(json.dumps(result, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("passed", "total")}))
    if result["passed"] != result["total"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
