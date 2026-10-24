"""Bounded real proof, known-truth join controls and full-source overlap FSV."""
import argparse
import copy
import itertools
import json
from pathlib import Path
import re
import shutil
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from text.evaluation_overlap import Features, prefix_join, components, run, canonical, file_sha
from audit.evaluation_overlap_readback import audit, reference_groups, reference_candidates


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("source", "publication", "output"):
        p.add_argument("--"+arg, type=Path, required=True)
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=False); results = []

    def put(path, value):
        path.write_bytes(canonical(value)+b"\n")

    def check(name, passed, **evidence):
        results.append(dict(name=name, passed=bool(passed), **evidence))

    def refuses(name, code, fn):
        observed = None
        try:
            fn()
        except (ValueError, OSError) as exc:
            observed = str(exc)
        check(name, observed is not None and code in observed, expected=code, observed=observed)

    before = {f.name: file_sha(f) for f in a.source.iterdir() if f.is_file()}
    put(a.output/"source-before.json", before)
    source = json.loads((a.source/"report.json").read_bytes()); selected = []
    for domain in sorted(source["sources"]):
        with (a.source/(domain+".jsonl")).open("rb") as f:
            selected.extend(json.loads(line) for line in itertools.islice(f, 4))
    minimum = Features()
    for row in selected:
        minimum.add(row)
    dest = a.output/"minimum-real-24"; dest.mkdir(); minimum.save(dest)
    put(dest/"source-records.json", selected)
    vocabulary = json.loads((dest/"words.json").read_bytes())
    tuples = np.load(dest/"shingles.npy", allow_pickle=False)
    pointer = np.load(dest/"indptr.npy", allow_pickle=False); flat = np.load(dest/"members.npy", allow_pickle=False)
    true_sets = []
    for i, row in enumerate(json.loads((dest/"source-records.json").read_bytes())):
        exact = set()
        for field in row["fields"]:
            words = re.findall(r"\w+", field["text"].lower())
            exact.update(zip(words, words[1:], words[2:], words[3:], words[4:]))
        observed = {tuple(vocabulary[int(w)] for w in tuples[int(t)]) for t in flat[int(pointer[i]):int(pointer[i+1])]}
        check("B1 real document lexical tuples reconstruct source "+str(i), observed == exact)
        true_sets.append(exact)
    _, observed, stats = prefix_join(minimum.sets)
    expected = []
    for i, j in itertools.combinations(range(len(true_sets)), 2):
        common = len(true_sets[i] & true_sets[j]); union = len(true_sets[i] | true_sets[j])
        if union and 5*common >= 3*union:
            expected.append(dict(left=i, right=j, kind="near_exact_jaccard", intersection=common, union=union))
    check("B2 exhaustive real 24-document join equals prefix join", observed == expected, near_links=len(expected))
    # Known-truth method controls derived from actual shingles; never training
    # observations. The two four-member sets intersect in 3 of 5 union members.
    basis = sorted(next(s for s in minimum.sets if len(s) >= 10))[:10]
    controls = minimum.sets + [set(basis[:4]), set(basis[1:5]), set(basis[2:6]), set(), set(basis[:4])]
    pairs, actual, cstats = prefix_join(controls)
    expected = []
    for i, j in itertools.combinations(range(len(controls)), 2):
        common = len(controls[i] & controls[j]); union = len(controls[i] | controls[j])
        if union and 5*common >= 3*union:
            expected.append(dict(left=i, right=j, kind="near_exact_jaccard", intersection=common, union=union))
    check("B2 exact-threshold duplicate below-threshold and empty method controls", actual == expected)
    rpairs, rstats = reference_candidates(controls)
    check("B2 static bucket candidate enumeration agrees", pairs == rpairs and all(cstats[k] == v for k, v in rstats.items()))
    check("B3 union-find equals independent graph traversal", components(minimum.documents, observed) == reference_groups(minimum.documents, observed))
    changed = copy.deepcopy(minimum.documents); changed[1]["group_keys"] = changed[0]["group_keys"]
    changed[2]["exact_sha256"] = changed[0]["exact_sha256"]
    left, right = components(changed, observed), reference_groups(changed, observed)
    check("B3 planted source-key and exact-content unions remain together", left == right and
          any({0, 1, 2} <= set(g["members"]) for g in left[1]))
    put(dest/"join-proof.json", dict(real_stats=stats, control_stats=cstats, observed=actual, expected=expected,
                                     control_scope="known-truth method controls only; excluded from production"))
    bad = copy.deepcopy(selected[0]); bad["fields"][0]["text"] += "x"
    refuses("B1 changed source field refuses", "MNLM_OVERLAP_SOURCE_FIELD", lambda: Features().add(bad))
    put(a.output/"minimum-checks.json", results)
    if not all(r["passed"] for r in results):
        raise SystemExit("MNLM_OVERLAP_MINIMUM_FAILED")
    refuses("B1 wrong source pin refuses", "MNLM_OVERLAP_SOURCE_PIN", lambda: run(a.source, "0"*64, a.output/"wrong-source"))
    report = run(a.source, before["report.json"], a.publication)
    check("B1 complete real publication", report["complete"])
    cold = audit(a.publication, a.source); put(a.output/"cold-readback.json", cold)
    check("B2 all source tuples candidates edges and groups rebuilt", cold["ok"], counts=cold["counts"])
    refuses("B3 immutable generation refuses overwrite", "MNLM_OVERLAP_EXISTS", lambda: run(a.source, before["report.json"], a.publication))

    def clone(name):
        target = a.output/name; shutil.copytree(a.publication, target)
        return target, json.loads((target/"report.json").read_bytes())

    def rebind(target, rep, name):
        rep["files"][name] = dict(sha256=file_sha(target/name), bytes=(target/name).stat().st_size)
        put(target/"report.json", rep)

    target, rep = clone("false-final-seal"); rep["sealed_evaluation"] = True; put(target/"report.json", rep)
    refuses("B3 false final seal refuses", "MNLM_OVERLAP_READBACK_CLAIM", lambda: audit(target, a.source))
    target, rep = clone("missing-lexical-member")
    values = np.load(target/"members.npy", allow_pickle=False).copy(); values[0] = values[1]
    np.save(target/"members.npy", values); rebind(target, rep, "members.npy")
    refuses("B1 altered lexical membership with rebound hash refuses", "MNLM_OVERLAP_READBACK_SHINGLE_MEMBERSHIP", lambda: audit(target, a.source))
    target, rep = clone("changed-links")
    link_rows = [json.loads(line) for line in (target/"links.jsonl").read_bytes().split(b"\n") if line]
    if link_rows:
        link_rows.pop()
    else:
        link_rows.append(dict(left=0, right=1, kind="near_exact_jaccard", intersection=1, union=1))
    (target/"links.jsonl").write_bytes(b"".join(canonical(r)+b"\n" for r in link_rows)); rebind(target, rep, "links.jsonl")
    refuses("B2 missing or fabricated overlap edge with rebound hash refuses", "MNLM_OVERLAP_READBACK_LINKS", lambda: audit(target, a.source))
    target, rep = clone("changed-group")
    group_rows = [json.loads(line) for line in (target/"groups.jsonl").read_bytes().split(b"\n") if line]
    group_rows[0]["members"].append(group_rows[-1]["members"][0])
    (target/"groups.jsonl").write_bytes(b"".join(canonical(r)+b"\n" for r in group_rows)); rebind(target, rep, "groups.jsonl")
    refuses("B3 wrong component with rebound hash refuses", "MNLM_OVERLAP_READBACK_GROUPS", lambda: audit(target, a.source))
    after = {f.name: file_sha(f) for f in a.source.iterdir() if f.is_file()}; put(a.output/"source-after.json", after)
    check("B3 source bytes unchanged", before == after)
    result = dict(checks=results, passed=sum(r["passed"] for r in results), total=len(results),
                  cold_readback=cold, acceptance="author-checked", sealed_evaluation=False)
    put(a.output/"fsv.json", result)
    print(json.dumps({k: result[k] for k in ("passed", "total")}))
    if result["passed"] != result["total"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
