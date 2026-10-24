"""Three-boundary CAP source/page proof on actual retained cases, then full run."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from text.cap_layout import canonical, date_record, file_sha, map_case, run, sha
from audit.cap_layout_readback import audit, audit_case, raw_audit


def snapshot(root):
    paths = [root/"report.json", root/"cases.jsonl"]
    report = json.loads(paths[0].read_bytes())
    paths += [root/"raw"/v["file"] for c in report["cases"] for v in c["artifacts"].values()]
    paths += [root/"checkpoints"/f"{i:04d}.json" for i in range(1, len(report["cases"])+1)]
    return {str(p.relative_to(root)).replace("\\", "/"): file_sha(p) for p in paths}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "minimum-capture", "capture", "publication", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=False); checks = []

    def put(name, value):
        path = a.output/(name+".json")
        with path.open("xb") as f:
            f.write(canonical(value)+b"\n")

    def check(name, passed, **details):
        checks.append(dict(name=name, passed=bool(passed), **details))

    def refuses(name, code, fn):
        observed = None
        try:
            fn()
        except (ValueError, FileNotFoundError) as exc:
            observed = str(exc)
        check(name, observed is not None and code in observed, expected=code, observed=observed)

    before = dict(capture=snapshot(a.capture), minimum=snapshot(a.minimum_capture),
                  source={n: file_sha(a.source/n) for n in ("report.json", "legal.jsonl")})
    put("before", before)
    raw, cases = raw_audit(a.minimum_capture, a.source)
    check("B1 real minimum acquisition bytes, ledger and checkpoints agree", raw["cases"] == 8, evidence=raw)
    minimum = run(a.source, a.minimum_capture, file_sha(a.minimum_capture/"report.json"), a.output/"minimum-publication")
    cold = audit(a.output/"minimum-publication", a.minimum_capture, a.source); put("minimum-cold", cold)
    check("B2 independent XML and streaming HTML map every minimum character equally", cold["counts"]["accepted"] == 8)
    check("B2 actual footnotes exercise discontinuous pages", cold["counts"]["discontinuous_pages"] > 0)
    row, item, meta, html = cases[4]; result = map_case(row, meta, html)
    put("actual-discontinuous-case", result)
    bad = copy.deepcopy(result); page = next(p for p in bad["pages"] if len(p["source_spans"]) > 1)
    page["source_spans"] = [[page["source_spans"][0][0], page["source_spans"][-1][1]]]
    put("collapsed-page", bad)
    refuses("B2 collapsing discontinuous page refuses", "MNLM_CAP_READBACK_PAGE_SPANS", lambda: audit_case(bad, row, meta, html))
    bad = copy.deepcopy(result); bad["pages"][0]["source_spans"][0][1] -= 1; put("shifted-page", bad)
    refuses("B2 shifted physical page boundary refuses", "MNLM_CAP_READBACK_PAGE_SPANS", lambda: audit_case(bad, row, meta, html))
    bad = copy.deepcopy(result); bad["decision_date"]["value"] = "2000-01-01"; put("invented-date", bad)
    refuses("B3 invented decision date refuses", "MNLM_CAP_READBACK_DATE", lambda: audit_case(bad, row, meta, html))
    bad = copy.deepcopy(result); bad["court"]["id"] += 1; put("wrong-court", bad)
    refuses("B3 wrong court provenance refuses", "MNLM_CAP_READBACK_METADATA", lambda: audit_case(bad, row, meta, html))
    bad = copy.deepcopy(result); bad["scientific_admission"] = True; put("false-admission", bad)
    refuses("B3 layout cannot assert scientific admission", "MNLM_CAP_READBACK_CLAIMS", lambda: audit_case(bad, row, meta, html))
    badmeta = copy.deepcopy(meta); badmeta["file_name"] = "0000-01"; put("wrong-case", badmeta)
    refuses("B1 wrong original case refuses", "MNLM_CAP_LAYOUT_CASE_FILENAME", lambda: map_case(row, badmeta, html))
    badrow = copy.deepcopy(row); badrow["fields"][0]["text"] += "X"; put("changed-text", badrow)
    refuses("B1 changed retained bytes refuse", "MNLM_CAP_LAYOUT_SOURCE_TEXT_HASH", lambda: map_case(badrow, meta, html))
    badrow["fields"][0]["sha256"] = sha(badrow["fields"][0]["text"].encode()); badrow["fields"][0]["characters"] += 1
    put("rehash-changed-text", badrow)
    refuses("B2 even rehashed content mismatch refuses", "MNLM_CAP_LAYOUT_CONTENT_ALIGNMENT", lambda: map_case(badrow, meta, html))
    badhtml = html.replace('data-label="967"', 'data-label="968"'); (a.output/"wrong-folio.html").write_text(badhtml, encoding="utf-8")
    refuses("B2 changed source folio conflicts with metadata", "MNLM_CAP_LAYOUT_PAGE_LABEL_CONFLICT", lambda: map_case(row, meta, badhtml))
    check("B3 partial dates retain precision without filling days", date_record("1978")["value"] == "1978"
          and date_record("1978-04")["precision"] == "month" and date_record(None)["precision"] == "missing")
    refuses("B3 impossible calendar date refuses", "MNLM_CAP_LAYOUT_DATE_VALUE", lambda: date_record("1978-02-30"))
    clone = a.output/"raw-negative"; shutil.copytree(a.minimum_capture, clone)
    name = cases[0][1]["artifacts"]["html"]["file"]; path = clone/"raw"/name; original = path.read_bytes()
    path.write_bytes(original+b" ")
    put("raw-corruption", dict(file=name, before=sha(original), after=file_sha(path)))
    refuses("B1 corrupted downloaded original refuses", "MNLM_CAP_READBACK_RAW_BYTES", lambda: raw_audit(clone, a.source))
    path.write_bytes(original); missing = path.with_suffix(".missing"); path.rename(missing)
    refuses("B1 missing downloaded sibling refuses", name, lambda: raw_audit(clone, a.source)); missing.rename(path)
    refuses("B1 wrong acquisition pin refuses before publication", "MNLM_CAP_LAYOUT_CAPTURE_PIN",
            lambda: run(a.source, a.minimum_capture, "0"*64, a.output/"wrong-pin"))
    put("minimum-checks", checks)
    if not all(r["passed"] for r in checks):
        raise SystemExit("MNLM_CAP_FSV_MINIMUM_FAILED")
    full = run(a.source, a.capture, file_sha(a.capture/"report.json"), a.publication)
    # Refused cases stay explicit. Complete acceptance is never inferred from a batch return value.
    cold = audit(a.publication, a.capture, a.source); put("full-cold", cold)
    check("B1 full raw population cold-read", cold["raw"]["cases"] == full["counts"]["cases"], evidence=cold["raw"])
    check("B2 all accepted pages independently reconstructed", cold["all_accepted_character_assignments_equal"], counts=cold["counts"])
    check("B2 refused cases retained without complete success", full["complete"] == (not full["errors"]))
    refuses("B3 existing publication cannot be overwritten", "MNLM_CAP_LAYOUT_EXISTS",
            lambda: run(a.source, a.capture, file_sha(a.capture/"report.json"), a.publication))
    after = dict(capture=snapshot(a.capture), minimum=snapshot(a.minimum_capture),
                 source={n: file_sha(a.source/n) for n in ("report.json", "legal.jsonl")})
    put("after", after); check("B3 input originals and retained source unchanged", before == after)
    result = dict(checks=checks, passed=sum(c["passed"] for c in checks), total=len(checks),
                  publication_complete=full["complete"], acceptance="author-checked", counts=cold["counts"],
                  publication_report_sha256=file_sha(a.publication/"report.json"), scientific_admission=False)
    put("fsv", result); print(json.dumps(result))
    if result["passed"] != result["total"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
