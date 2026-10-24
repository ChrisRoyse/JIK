"""Actual-source proof for page links, overlap groups and causal visibility."""
import argparse
import copy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from text.page_attachments import attach, causal_attachment, inputs, run, canonical, file_sha
from audit.page_attachment_readback import audit, audit_case


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("multiscale", "layout", "overlap", "publication", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=False); checks = []
    roots = {name: getattr(a, name) for name in ("multiscale", "layout", "overlap")}
    pins = {name: file_sha(path/"report.json") for name, path in roots.items()}
    _, paths, groups = inputs(a.multiscale, a.layout, a.overlap, pins)
    files = list(paths.values())+[p/"report.json" for p in roots.values()]

    def put(name, obj):
        with (a.output/(name+".json")).open("xb") as f:
            f.write(canonical(obj)+b"\n")

    def check(name, passed, **details):
        checks.append(dict(name=name, passed=bool(passed), **details))

    def refuses(name, code, fn):
        observed = None
        try:
            fn()
        except ValueError as exc:
            observed = str(exc)
        check(name, observed is not None and code in observed, expected=code, observed=observed)

    before = {str(path): file_sha(path) for path in files}; put("before", before)
    with paths["multiscale"].open("rb") as mf, paths["layout"].open("rb") as lf:
        cases = [(json.loads(next(mf)), json.loads(next(lf))) for _ in range(8)]
    proof = []
    for base, page in cases:
        result = attach(base, page, groups[base["document_id"]]); put(base["document_id"]+"-minimum", result)
        persisted = json.loads((a.output/(base["document_id"]+"-minimum.json")).read_bytes())
        proof.append(dict(audit_case(persisted, base, page, groups[base["document_id"]], exhaustive=True)))
    check("B1 eight real documents match exhaustive page/node intersection", len(proof) == 8, counts=proof)
    base, page = cases[4]; group = groups[base["document_id"]]; result = attach(base, page, group)
    bad = copy.deepcopy(result); bad["edges"].pop(); put("missing-edge", bad)
    refuses("B2 missing page relation refuses", "MNLM_PAGE_READBACK_EDGES", lambda: audit_case(bad, base, page, group))
    bad = copy.deepcopy(result); bad["edges"].append(bad["edges"][0]); put("duplicate-edge", bad)
    refuses("B2 duplicated relation refuses", "MNLM_PAGE_READBACK_EDGES", lambda: audit_case(bad, base, page, group))
    bad = copy.deepcopy(result); bad["pages"][0]["visible_after"] = 0; put("premature-parent", bad)
    refuses("B3 premature page visibility refuses readback", "MNLM_PAGE_READBACK_PAGES", lambda: audit_case(bad, base, page, group))
    refuses("B3 premature page visibility refuses consumer", "MNLM_PAGE_ATTACH_VISIBILITY", lambda: causal_attachment(bad, base["graph"], 0, group))
    bad = copy.deepcopy(result); bad["overlap_group"] = "0"*64; put("wrong-overlap-group", bad)
    refuses("B3 wrong overlap group refuses readback", "MNLM_PAGE_READBACK_GROUP_LENGTH", lambda: audit_case(bad, base, page, group))
    refuses("B3 wrong overlap group refuses consumer", "MNLM_PAGE_ATTACH_GROUP", lambda: causal_attachment(result, base["graph"], 0, "0"*64))
    badgraph = copy.deepcopy(base["graph"]); badgraph["nodes"][0]["end"] -= 1; put("changed-base", badgraph)
    refuses("B1 changed base graph refuses consumer", "MNLM_PAGE_ATTACH_BASE_HASH", lambda: causal_attachment(result, badgraph, 0, group))
    target = next(p for p in result["pages"] if len(p["source_spans"]) > 1); cutoff = target["source_spans"][0][1]
    visible = causal_attachment(result, base["graph"], cutoff, group); put("partial-page-consumer", dict(cutoff=cutoff, visible=visible))
    ids = {p["id"] for p in visible["pages"]} | {n["id"] for n in base["graph"]["nodes"] if n["end"] <= cutoff}
    check("B3 page with future footnotes remains hidden", target["id"] not in {p["id"] for p in visible["pages"]}
          and all(e["from"] in ids and e["to"] in ids for e in visible["edges"]))
    visible = causal_attachment(result, base["graph"], result["source_characters"], group)
    check("B3 fully observed case exposes all page links", visible["pages"] == result["pages"] and visible["edges"] == result["edges"])
    refuses("B3 invalid cutoff refuses", "MNLM_PAGE_ATTACH_CUTOFF", lambda: causal_attachment(result, base["graph"], -1, group))
    badpage = copy.deepcopy(page); badpage["text_sha256"] = "0"*64; put("wrong-source-text", badpage)
    refuses("B1 page source mismatch refuses", "MNLM_PAGE_ATTACH_LINEAGE", lambda: attach(base, badpage, group))
    wrong_pins = {**pins, "layout": "0"*64}
    refuses("B1 wrong parent pin refuses before publication", "MNLM_PAGE_ATTACH_PIN_LAYOUT",
            lambda: run(a.multiscale, a.layout, a.overlap, wrong_pins, a.output/"wrong-pin"))
    put("minimum-checks", checks)
    if not all(c["passed"] for c in checks):
        raise SystemExit("MNLM_PAGE_FSV_MINIMUM_FAILED")
    publication = run(a.multiscale, a.layout, a.overlap, pins, a.publication)
    cold = audit(a.publication, a.multiscale, a.layout, a.overlap); put("cold-readback", cold)
    check("B2 every real attachment equals independent binary-search join", cold["every_relation_equal"], counts=cold["counts"])
    check("B3 refusals keep publication incomplete", publication["complete"] == (not publication["errors"]))
    refuses("B3 immutable publication refuses overwrite", "MNLM_PAGE_ATTACH_EXISTS",
            lambda: run(a.multiscale, a.layout, a.overlap, pins, a.publication))
    after = {str(path): file_sha(path) for path in files}; put("after", after)
    check("B1/B3 all parent bytes unchanged", before == after)
    result = dict(checks=checks, passed=sum(c["passed"] for c in checks), total=len(checks),
                  acceptance="author-checked", cold_readback=cold, scientific_admission=False)
    put("fsv", result); print(json.dumps({k: result[k] for k in ("passed", "total", "cold_readback")}))
    if result["passed"] != result["total"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
