"""Three-boundary FSV on the real complete-document capture (#14/#27).

Cold source truth is built separately once. Corrupt copies reuse that immutable
truth, so each control does not rescan the 7.3 GB raw inventory.
"""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.evaluation_capture_readback import encode, file_hash, digest, validate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("raw", "history", "contract", "publication", "truth", "proof", "producer", "output"):
        p.add_argument("--"+arg, type=Path, required=True)
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location("capture_under_audit", a.producer)
    producer = importlib.util.module_from_spec(spec); spec.loader.exec_module(producer)
    results = []

    def put(path, value):
        path.write_bytes(encode(value)+b"\n")

    def record(name, passed, **evidence):
        results.append(dict(name=name, passed=bool(passed), **evidence))

    def rejects(name, code, fn):
        error = None
        try:
            fn()
        except (ValueError, OSError) as exc:
            error = str(exc)
        record(name, error is not None and code in error, expected=code, observed=error)

    def inventory(root):
        return {str(f.relative_to(root)): file_hash(f) for f in root.rglob("*") if f.is_file()}

    before = inventory(a.publication); truth_before = inventory(a.truth)
    put(a.output/"before.json", dict(publication=before, truth=truth_before))
    cold = validate(a.truth, a.publication)
    record("B1 pinned full-source independent scan completed", cold["ok"], truth_report_sha256=cold["truth_report_sha256"])
    proof = json.loads((a.proof/"proof.json").read_bytes())
    record("B2 minimum actual-source corpus conserves fields and excludes prior IDs", len(proof) == 6 and
           all(all(v is True for k, v in r.items() if k != "domain") for r in proof))
    record("B2 full publication matches independently reconstructed source fields", cold["ok"], totals=cold["totals"])
    record("B3 publication remains unsealed", cold["sealed_evaluation"] is False)
    rejects("B3 existing output refuses overwrite", "MNLM_EVAL_CAPTURE_EXISTS",
            lambda: producer.run(a.raw, a.history, a.contract, a.publication))
    cfg = json.loads(a.contract.read_bytes())
    bad = copy.deepcopy(cfg); bad["raw_manifest_sha256"] = "0"*64
    path = a.output/"wrong-source.json"; put(path, bad)
    rejects("B1 wrong source manifest refuses", "MNLM_EVAL_CAPTURE_SOURCE_MANIFEST",
            lambda: producer.run(a.raw, a.history, path, a.output/"wrong-source-result"))
    bad = copy.deepcopy(cfg); bad["history"][next(iter(bad["history"]))] = "0"*64
    path = a.output/"wrong-history.json"; put(path, bad)
    rejects("B1 wrong historical exposure manifest refuses", "MNLM_EVAL_CAPTURE_HISTORY_MANIFEST",
            lambda: producer.run(a.raw, a.history, path, a.output/"wrong-history-result"))
    src = next(s for s in json.loads((a.raw/"manifest.json").read_bytes())["sources"] if s["domain"] == "general")
    proof_path = a.proof/"general.jsonl"
    bad = {**src, "bytes": proof_path.stat().st_size, "sha256": "0"*64}
    rejects("B1 changed source bytes refuse before selection", "MNLM_EVAL_CAPTURE_SOURCE_BYTES",
            lambda: producer.select(bad, proof_path, set(), 2, cfg["selection_seed"], a.output/"wrong-bytes-errors.jsonl"))

    def clone(name):
        dest = a.output/name; shutil.copytree(a.publication, dest)
        return dest, json.loads((dest/"report.json").read_bytes())

    def change_rows(name, domain, change):
        dest, report = clone(name); path = dest/(domain+".jsonl")
        rows = [json.loads(line) for line in path.read_bytes().split(b"\n") if line]
        change(rows)
        path.write_bytes(b"".join(encode(r)+b"\n" for r in rows))
        report["sources"][domain].update(sha256=file_hash(path), bytes=path.stat().st_size)
        put(dest/"report.json", report)
        rejects(name, "MNLM_EVAL_READBACK_SOURCE_CONTENT", lambda: validate(a.truth, dest))

    def truncate(rows):
        field = next(f for f in rows[0]["fields"] if f["text"])
        field["text"] = field["text"][:-1]
        field.update(characters=len(field["text"]), utf8_bytes=len(field["text"].encode()), sha256=digest(field["text"].encode()))

    change_rows("B2-truncated-tail-rehashed", "general", truncate)
    change_rows("B2-missing-finance-section-rehashed", "finance", lambda rows: rows[0]["fields"].pop())
    change_rows("B2-changed-source-metadata-rehashed", "general", lambda rows: rows[0]["metadata"].update(unattested=True))
    change_rows("B2-reordered-selection-rehashed", "science", lambda rows: rows.reverse())
    change_rows("B2-duplicate-document-rehashed", "legal", lambda rows: rows.__setitem__(0, copy.deepcopy(rows[1])))
    for name, modify, code in (
        ("B1-missing-source-inventory", lambda r: r["sources"].pop("general"), "SOURCE_INVENTORY"),
        ("B3-false-final-test-seal", lambda r: r.update(sealed_evaluation=True), "CLAIM"),
        ("B3-false-scientific-admission", lambda r: r.update(scientific_admission=True), "CLAIM"),
        ("B3-false-availability", lambda r: r.update(all_requested_source_counts_available=False), "AVAILABILITY"),
    ):
        dest, report = clone(name); modify(report); put(dest/"report.json", report)
        rejects(name, "MNLM_EVAL_READBACK_"+code, lambda: validate(a.truth, dest))
    dest, report = clone("B3-nonempty-error-ledger")
    path = dest/"general.errors.jsonl"; put(path, dict(error="injected-failure"))
    report["sources"]["general"]["errors_sha256"] = file_hash(path); put(dest/"report.json", report)
    rejects("B3 rehashed error ledger refuses", "MNLM_EVAL_READBACK_ERROR_LEDGER", lambda: validate(a.truth, dest))
    after = inventory(a.publication); truth_after = inventory(a.truth)
    put(a.output/"after.json", dict(publication=after, truth=truth_after))
    record("B3 original publication and source truth unchanged", before == after and truth_before == truth_after)
    result = dict(checks=results, passed=sum(r["passed"] for r in results), total=len(results),
                  cold_readback=cold, adjudication="author-checked", scientific_admission=False)
    put(a.output/"fsv.json", result)
    print(json.dumps({k: result[k] for k in ("passed", "total")}))
    if result["passed"] != result["total"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
