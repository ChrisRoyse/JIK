"""Pre-evaluation observations of replayed SWE submissions, with labels kept apart (#12 senses plan).

One record per observed, provider-agreeing, non-empty attempt from the attempt replays. Inputs read:
benchmark `problem_statement` (the task as given to the agent), the attempt's `generated_patch`,
`trajectory` turns (what the agent did and saw before submitting), `model_name`, `exit_status`.
Never read: `eval_logs`, the test patch, FAIL_TO_PASS/PASS_TO_PASS, the reference patch.
The independently replayed outcome goes to a separate labels file that no instrument opens.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oracle.bound_rows import BoundRows
from senses.contract import require
from senses.trajectory_population import publish

CODE = "MNLM_SWE_OBSERVATIONS"
ATTEMPT = ["instance_id", "model_name", "generated_patch", "trajectory", "exit_status"]
ISSUE = ["instance_id", "repo", "problem_statement"]
FORBIDDEN = {"eval_logs", "test_patch", "FAIL_TO_PASS", "PASS_TO_PASS", "patch", "target"}


def build(root, sources, output):
    """sources: [{report, tasks, superseded_by: [report...]}]; returns the store digest."""
    require(not (set(ATTEMPT) | set(ISSUE)) & FORBIDDEN, CODE, "an evaluation field is in the input columns")
    output.mkdir(parents=True, exist_ok=False)
    records, plans = {}, {}
    for src in sources:
        for t in json.loads((root / src["tasks"]).read_bytes()):
            plans[t["task"]] = json.loads((root / t["plan"]["path"]).read_bytes())
        for x in json.loads((root / src["report"]).read_bytes())["records"]:
            if src.get("exclude") and x["task"] in src["exclude"]:
                continue
            records[(x["task"], x["attempt"]["path"], x["attempt"]["row"])] = x  # later sources supersede
    chosen = [x for x in records.values()
              if x["status"] == "observed" and x["agrees_with_provider"] and not x.get("empty_submission")]
    chosen.sort(key=lambda x: (x["task"], x["attempt"]["path"], x["attempt"]["row"]))
    shard = {}
    for p in plans.values():
        shard[p["attempt"]["path"]] = {k: p["attempt"][k] for k in p["attempt"] if k != "row"}
    reqs = []
    for x in chosen:
        reqs.append(({**shard[x["attempt"]["path"]], "row": x["attempt"]["row"]}, ATTEMPT))
    for t, p in sorted(plans.items()):
        reqs.append((p["benchmark"], ISSUE))
    seen, unique = set(), []
    for b, c in reqs:
        key = (b["path"], b["row"], tuple(c))
        if key not in seen:
            seen.add(key); unique.append((b, c))
    binders, owner = [], {}
    for j in range(0, len(unique), 256):
        b = BoundRows(root, unique[j:j + 256]); binders.append(b)
        for binding, cols in unique[j:j + 256]:
            owner[(binding["path"], binding["row"], tuple(cols))] = b
    read = lambda binding, cols: owner[(binding["path"], binding["row"], tuple(cols))].read(binding, cols)
    obs_path, lab_path = output / "observations.jsonl", output / "labels.jsonl"
    with obs_path.open("x", encoding="utf-8") as obs, lab_path.open("x", encoding="utf-8") as lab:
        for n, x in enumerate(chosen):
            plan = plans[x["task"]]
            a = read({**shard[x["attempt"]["path"]], "row": x["attempt"]["row"]}, ATTEMPT)
            issue = read(plan["benchmark"], ISSUE)
            require(a["instance_id"] == issue["instance_id"] == x["task"], CODE, "row identity differs")
            uid = hashlib.sha256(f"{x['task']}|{x['attempt']['path']}|{x['attempt']['row']}".encode()).hexdigest()[:20]
            turns = [{"role": t["role"], "text": t["text"] or ""} for t in a["trajectory"] if t["role"] != "system"]
            obs.write(json.dumps({"uid": uid, "task": x["task"], "group": issue["repo"], "model": a["model_name"],
                                  "exit_status": a["exit_status"], "issue": issue["problem_statement"],
                                  "patch": a["generated_patch"], "trajectory": turns,
                                  # pinned pre-patch source snapshot, for the structure and graph senses
                                  "source": {k: plan[k] for k in ("base_commit", "archive", "tree", "commit", "exact_blobs")
                                             if k in plan}},
                                 sort_keys=True) + "\n")
            lab.write(json.dumps({"uid": uid, "task": x["task"], "group": issue["repo"], "outcome": x["outcome"]}, sort_keys=True) + "\n")
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    return publish(output / "manifest.json", {"schema": "swe-observations-v1", "records": len(chosen),
        "tasks": len({x["task"] for x in chosen}), "observations": {"path": "observations.jsonl", "sha256": digest(obs_path)},
        "labels": {"path": "labels.jsonl", "sha256": digest(lab_path)}, "columns_read": {"attempt": ATTEMPT, "benchmark": ISSUE},
        "never_read": sorted(FORBIDDEN), "sources": sources, "row_bindings": [b.receipts for b in binders]})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--sources", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps({"sha256": build(a.root, json.loads(a.sources.read_bytes()), a.output)}))
