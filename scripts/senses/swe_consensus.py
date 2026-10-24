"""S8 consensus sense: how an attempt's change agrees with the other recorded attempts at the same task.

Computed over every recorded attempt of the task in the pinned trajectory shards (the replay census),
reading only `generated_patch`, so the observation set's label-agreement filter never shapes it.
Features per observation: Jaccard of its changed-line set and changed-file set with each other attempt
(mean, max, share >= 0.5), share of other attempts with an identical normalised change, whether it is a
modal change, its change size percentile within the task, and the task's attempt count.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oracle.bound_rows import BoundRows
from senses.swe_lenses import hunks
from senses.trajectory_population import publish

VERSION = "swe-consensus-sense-v1"


def signature(patch):
    hs, lines = hunks(patch or "")
    files = frozenset(h[0] for h in hs)
    changed = frozenset((f, s, re.sub(r"\s+", " ", t).strip()) for f, v in lines.items() for s in "+-" for t in v if t.strip())
    return files, changed


def jac(a, b):
    return len(a & b) / len(a | b) if (a or b) else 1.0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--observations", type=Path, required=True)
    p.add_argument("--census", type=Path, action="append", required=True, help="replay census.json files")
    p.add_argument("--scope", type=Path, required=True, help="trajectory shard scope (pins)")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    shards = {"raw/swe-agent-trajectories/" + s["path"]: s for s in json.loads(a.scope.read_bytes())["shards"]}
    rows = {}
    for c in a.census:
        for task, items in json.loads(c.read_bytes())["tasks"].items():
            rows.setdefault(task, set()).update((x["path"], x["row"]) for x in items)
    reqs = [({**shards[path], "path": path, "row": row}, ["instance_id", "generated_patch"])
            for task in sorted(rows) for path, row in sorted(rows[task])]
    binders = [BoundRows(a.root, reqs[j:j + 256]) for j in range(0, len(reqs), 256)]
    sigs = {}
    for j, (b, cols) in enumerate(reqs):
        r = binders[j // 256].read(b, cols)
        sigs.setdefault(r["instance_id"], []).append(((b["path"], b["row"]), signature(r["generated_patch"])))
    recs = [json.loads(l) for l in (a.observations / "observations.jsonl").read_text(encoding="utf-8").splitlines()]
    out = []
    for r in recs:
        files, changed = signature(r["patch"])
        others = [s for key, s in sigs[r["task"]]]
        # the observation itself is one of the task's attempts: drop exactly one identical signature
        idx = next((i for i, s in enumerate(others) if s == (files, changed)), None)
        if idx is not None:
            others = others[:idx] + others[idx + 1:]
        jl = [jac(changed, s[1]) for s in others] or [0.0]
        jf = [jac(files, s[0]) for s in others] or [0.0]
        same = sum(1 for s in others if s[1] == changed and changed)
        counts = {}
        for s in others + [(files, changed)]:
            counts[s[1]] = counts.get(s[1], 0) + 1
        modal = float(bool(changed) and counts[changed] == max(counts.values()))
        sizes = sorted(len(s[1]) for s in others + [(files, changed)])
        pct = sizes.index(len(changed)) / max(1, len(sizes) - 1)
        out.append([np.mean(jl), np.max(jl), np.mean(np.array(jl) >= 0.5), np.mean(jf), np.max(jf),
                    same / max(1, len(others)), modal, pct, np.log1p(len(others))])
    M = np.array(out, np.float32)
    np.save(a.output / "consensus.npy", M)
    publish(a.output / "manifest.json", {"schema": "swe-consensus-sense-v1", "version": VERSION,
        "senses": {"consensus": {"path": "consensus.npy", "shape": list(M.shape), "sha256": hashlib.sha256((a.output / "consensus.npy").read_bytes()).hexdigest()}},
        "attempts_read": len(reqs), "tasks": len(rows), "columns_read": ["instance_id", "generated_patch"],
        "uid_order_sha256": hashlib.sha256("\n".join(r["uid"] for r in recs).encode()).hexdigest(),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "row_bindings": [b.receipts for b in binders]})
    print(json.dumps({"shape": list(M.shape), "attempts_read": len(reqs)}))


if __name__ == "__main__":
    main()
