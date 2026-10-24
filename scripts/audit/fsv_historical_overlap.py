"""Full-source historical overlap replay with persisted rehashed negatives."""
import copy
import gzip
import json
import os
from pathlib import Path
import sys

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.historical_overlap_readback import parser, reference, validate_derived, header, rows, sha, check, canonical
from senses.trajectory_population import publish


def main():
    a = parser().parse_args(); check(not a.output.exists(), "FSV_OUTPUT_EXISTS")
    a.output.mkdir(parents=True)
    rep = json.loads((a.publication/"report.json").read_bytes())
    before = {name: sha(a.publication/name) for name in ["report.json", *rep["files"]]}
    publish(a.output/"before.json", before)
    expected = reference(a.publication, a.report_sha256, a.capture, a.overlap, a.history)
    publish(a.output/"cold-readback.json", {k: v for k, v in expected.items() if k not in ("links", "exposure")})
    checks = [dict(name="complete_literal_source_and_sparse_replay", passed=True,
                   original_histories_replayed=a.history is not None)]

    def refused(name, code, fn):
        error = None
        try:
            fn()
        except ValueError as exc:
            error = str(exc)
        checks.append(dict(name=name, expected=code, observed=error, passed=error == code))

    def mutant(name, filename=None, raw=None, update=None):
        root = a.output/name; root.mkdir()
        for item in before:
            os.link(a.publication/item, root/item)
        report = copy.deepcopy(rep)
        if filename:
            path = root/filename; path.unlink(); path.write_bytes(raw)
            report["files"][filename] = dict(sha256=sha(path), bytes=path.stat().st_size)
        if update:
            update(report)
        (root/"report.json").unlink(); publish(root/"report.json", report)
        return root

    def derived(root):
        header(root, sha(root/"report.json"), a.capture, a.overlap)
        validate_derived(root, expected)

    check(bool(expected["links"]), "FSV_NEEDS_REAL_LINK")
    root = mutant("missing-real-link", "links.jsonl", b"")
    refused("missing-real-link", "MNLM_HISTORY_READ_LINKS", lambda: derived(root))
    fabricated = copy.deepcopy(expected["links"]); fabricated[0]["intersection"] += 1
    root2 = mutant("fabricated-intersection", "links.jsonl", b"".join(canonical(x)+b"\n" for x in fabricated))
    refused("fabricated-intersection", "MNLM_HISTORY_READ_LINKS", lambda: derived(root2))
    root3 = mutant("changed-cardinality-summary", update=lambda r: r["counts"].update(historical_shingle_memberships=r["counts"]["historical_shingle_memberships"]+1))
    refused("changed-cardinality-summary", "MNLM_HISTORY_READ_COUNTS", lambda: derived(root3))
    groups = copy.deepcopy(expected["exposure"])
    hit = next(g for g in groups if g["historical_link_detected"])
    hit["historical_link_detected"] = False
    root4 = mutant("hidden-component-exposure", "component-exposure.jsonl", b"".join(canonical(g)+b"\n" for g in groups))
    refused("hidden-component-exposure", "MNLM_HISTORY_READ_COMPONENT_EXPOSURE", lambda: derived(root4))
    root5 = mutant("false-final-seal", update=lambda r: r.update(sealed_evaluation=True))
    refused("false-final-seal", "MNLM_HISTORY_READ_FALSE_CLAIM", lambda: derived(root5))
    root6 = mutant("false-clean-certificate", update=lambda r: r.update(absence_is_clean_certificate=True))
    refused("false-clean-certificate", "MNLM_HISTORY_READ_FALSE_CLAIM", lambda: derived(root6))
    refused("wrong-report-pin", "MNLM_HISTORY_READ_REPORT_PIN", lambda: header(a.publication, "0"*64, a.capture, a.overlap))

    # Fail early on first physical history record, after genuine cold loading.
    observed = a.output/"changed-first-text"; observed.mkdir()
    for item in before:
        os.link(a.publication/item, observed/item)
    (observed/"observed.jsonl.gz").unlink()
    with (observed/"observed.jsonl.gz").open("wb") as out:
        with gzip.GzipFile(filename="", fileobj=out, mode="wb", mtime=0) as z:
            for i, record in enumerate(rows(a.publication/"observed.jsonl.gz")):
                if i == 0:
                    record["texts"][0] += " changed observation"
                z.write(canonical(record)+b"\n")
    r = copy.deepcopy(rep)
    r["files"]["observed.jsonl.gz"] = dict(sha256=sha(observed/"observed.jsonl.gz"), bytes=(observed/"observed.jsonl.gz").stat().st_size)
    (observed/"report.json").unlink(); publish(observed/"report.json", r)
    refused("changed-first-text", "MNLM_HISTORY_READ_OBSERVED_TEXT",
            lambda: reference(observed, sha(observed/"report.json"), a.capture, a.overlap))
    pairs = np.fromfile(a.publication/"pairs.bin", dtype="<u4").reshape(-1, 3)
    pairs[0, 2] += 1
    root7 = mutant("rehashed-sparse-count", "pairs.bin", pairs.tobytes())
    refused("rehashed-sparse-count", "MNLM_HISTORY_READ_SPARSE_PAIRS",
            lambda: reference(root7, sha(root7/"report.json"), a.capture, a.overlap))
    after = {name: sha(a.publication/name) for name in before}; publish(a.output/"after.json", after)
    checks.append(dict(name="physical_source_unchanged", passed=before == after))
    result = dict(acceptance="author-checked", checks=checks, passed=sum(c["passed"] for c in checks), total=len(checks),
                  report_sha256=a.report_sha256, scientific_admission=False,
                  boundaries=["historical manifests/atoms to observed union", "literal shingles to exact sparse pairs", "cross-population links to all scale groups"])
    publish(a.output/"fsv.json", result); check(all(c["passed"] for c in checks), "FSV_FAILED")
    print(json.dumps(dict(passed=result["passed"], total=result["total"])))


if __name__ == "__main__":
    main()
