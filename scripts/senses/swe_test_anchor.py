"""Per-test grounded anchor for the SWE observation store (#12 senses, anchor tests-v1).

For every observation (same selection and supersession as swe_observations.build), the replayed
agent condition's recount.json gives the state of each required test in the pinned sandbox. This
writes test_labels.jsonl beside labels.jsonl: one row per (observation, required test) with
passed = 1 iff the test's recounted state is "passed". Like labels.jsonl, no instrument opens it.
`varies` marks tests whose state differs between the task's observations; a test constant within its
task carries no information given the (task, test) context.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from senses.contract import require
from senses.trajectory_population import publish

CODE = "MNLM_SWE_TEST_ANCHOR"
VERSION = "swe-test-anchor-v1"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--sources", type=Path, required=True, help="the sources JSON the observation store was built from")
    p.add_argument("--observations", type=Path, required=True, help="observation store directory")
    a = p.parse_args()
    where = {}
    for src in json.loads(a.sources.read_bytes()):
        report = a.root / src["report"]
        for x in json.loads(report.read_bytes())["records"]:
            if src.get("exclude") and x["task"] in src["exclude"]:
                continue
            uid = hashlib.sha256(f"{x['task']}|{x['attempt']['path']}|{x['attempt']['row']}".encode()).hexdigest()[:20]
            where[uid] = (report.parent / x["task"] / str(x["n"]) / "agent" / "recount.json", x)  # later supersede
    labels = [json.loads(l) for l in (a.observations / "labels.jsonl").read_text().splitlines()]
    rows, states = [], {}
    for lab in labels:
        path, x = where[lab["uid"]]
        rec = json.loads(path.read_bytes())
        require(rec["outcome"] == lab["outcome"] == x["outcome"], CODE, f"recount outcome differs for {lab['uid']}")
        for test, st in sorted(rec["states"].items()):
            rows.append({"uid": lab["uid"], "task": lab["task"], "test": test, "passed": int(st == "passed")})
            states.setdefault((lab["task"], test), set()).add(st == "passed")
    for r in rows:
        r["varies"] = len(states[(r["task"], r["test"])]) > 1
    out = a.observations / "test_labels.jsonl"
    with out.open("x", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    var = [r for r in rows if r["varies"]]
    summary = {"schema": "swe-test-anchor-v1", "version": VERSION, "pairs": len(rows), "varying_pairs": len(var),
               "varying_tests": len({(r["task"], r["test"]) for r in var}), "tasks_with_varying": len({r["task"] for r in var}),
               "varying_pass_rate": sum(r["passed"] for r in var) / max(1, len(var)),
               "labels_sha256": hashlib.sha256((a.observations / "labels.jsonl").read_bytes()).hexdigest(),
               "test_labels_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
               "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    publish(a.observations / "test_labels.manifest.json", summary)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
