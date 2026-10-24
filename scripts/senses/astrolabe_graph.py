"""S3 graph sense: Astrolabe's code decomposer (codebase-memory engine) over each task's pinned base snapshot.

Per task: the exact Git snapshot is written to a scratch tree on the data drive, indexed with the
pinned Astrolabe/cbm binary (CLI mode, static parsing only), and its callables (Function/Method with
line spans, complexity, cognitive load, loops, recursion, exports, test flags) plus CALLS and TESTS
edges are kept. Per attempt: the callables whose base line spans overlap the diff's changed base
lines are located and their graph context is encoded as one fixed-width vector. Runs on Windows,
because the Astrolabe fork is Windows-native.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oracle.prepare_swe import source_files
from senses.swe_lenses import hunks
from senses.trajectory_population import publish

PROPS = ["file_path", "start_line", "end_line", "complexity", "cognitive", "loop_depth", "param_count", "recursive",
         "is_exported", "is_entry_point", "is_test"]
VERSION = "astrolabe-graph-sense-v2"
# The engine fails closed on ambiguous qualified names, which documentation and example trees produce
# (e.g. two tutorial files defining the same route). Scope: importable source and tests only.
EXCLUDED_TOP = {"docs", "doc", "examples", "example", "benchmarks", "benchmark", "demo", "demos", "site", "website"}


def cli(exe, tool, args, timeout=600):
    p = subprocess.run([str(exe), "cli", tool], input=json.dumps(args).encode(), capture_output=True, timeout=timeout)
    lines = [l for l in p.stdout.decode("utf-8", "replace").splitlines() if l.startswith("{")]
    if p.returncode != 0 or not lines:
        raise RuntimeError(f"MNLM_ASTROLABE_CLI {tool} rc={p.returncode}: {p.stderr.decode('utf-8', 'replace')[-300:]}")
    return json.loads(lines[-1])


def rows(exe, project, query):
    out = cli(exe, "query_graph", {"project": project, "query": query})
    if out.get("total") != len(out.get("rows", [])):
        raise RuntimeError("MNLM_ASTROLABE_TRUNCATED: query returned fewer rows than its total")
    return out["rows"]


def graph(exe, root, task, source, scratch):
    tree = scratch / hashlib.sha256(task.encode()).hexdigest()[:16]
    if tree.exists():
        shutil.rmtree(tree)
    excluded = 0
    for path, data in source_files(root, source).items():
        if path.split("/", 1)[0].lower() in EXCLUDED_TOP:
            excluded += 1; continue
        (tree / path).parent.mkdir(parents=True, exist_ok=True)
        (tree / path).write_bytes(data)
    info = cli(exe, "index_repository", {"repo_path": tree.as_posix()})
    project = info["project"]
    try:
        callables = []
        for label in ("Function", "Method"):
            fields = ", ".join(["f.qualified_name"] + [f"f.{k}" for k in PROPS])
            for r in rows(exe, project, f"MATCH (f:{label}) RETURN {fields} LIMIT 1000000"):
                callables.append([label, r[0].removeprefix(project + ".")] + r[1:])
        edges = {}
        for kind in ("CALLS", "TESTS"):
            edges[kind] = [[a.removeprefix(project + "."), b.removeprefix(project + ".")]
                           for a, b in rows(exe, project, f"MATCH (a)-[r:{kind}]->(b) RETURN a.qualified_name, b.qualified_name LIMIT 1000000")]
        return {"task": task, "nodes": info["nodes"], "edges": info["edges"], "callables": callables, "links": edges,
                "scope": {"excluded_top_level": sorted(EXCLUDED_TOP), "excluded_files": excluded}}
    finally:
        cli(exe, "delete_project", {"project": project})
        shutil.rmtree(tree, ignore_errors=True)


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return 1.0 if str(x).lower() == "true" else 0.0


def features(g, patch):
    """Graph context of the callables the patch changes (base line spans overlapping changed base lines)."""
    spans = {}
    for path, old_start, old_len, _, _ in hunks(patch)[0]:
        spans.setdefault(path, []).append((old_start, old_start + max(old_len, 1) - 1))
    cols = {k: i for i, k in enumerate(["label", "qname"] + PROPS)}
    touched = []
    for c in g["callables"]:
        f = c[cols["file_path"]]
        if f in spans:
            s, e = int(num(c[cols["start_line"]])), int(num(c[cols["end_line"]]))
            if any(a <= e and b >= s for a, b in spans[f]):
                touched.append(c)
    callers, callees, tested = {}, {}, set()
    for a, b in g["links"]["CALLS"]:
        callers.setdefault(b, set()).add(a); callees.setdefault(a, set()).add(b)
    for a, b in g["links"]["TESTS"]:
        tested.add(b)
    test_names = {c[cols["qname"]] for c in g["callables"] if num(c[cols["is_test"]]) > 0}
    names = [c[cols["qname"]] for c in touched]
    inn = [len(callers.get(n, ())) for n in names]; out = [len(callees.get(n, ())) for n in names]
    test_callers = [len(callers.get(n, set()) & test_names) for n in names]
    two_hop = [len(set().union(*[callers.get(m, set()) for m in callers.get(n, set())]) if callers.get(n) else set()) for n in names]
    agg = lambda v, f: f(v) if v else 0.0
    prop = lambda k: [num(c[cols[k]]) for c in touched]
    vec = [len(touched), len({c[cols["file_path"]] for c in touched}), float(bool(touched)),
           agg(inn, sum), agg(inn, max), agg(out, sum), agg(out, max), agg(test_callers, sum),
           float(any(n in tested for n in names)), agg(two_hop, max),
           agg(prop("complexity"), sum), agg(prop("complexity"), max), agg(prop("cognitive"), sum), agg(prop("cognitive"), max),
           agg(prop("loop_depth"), max), agg(prop("param_count"), lambda v: sum(v) / len(v)), agg(prop("recursive"), max),
           agg(prop("is_exported"), lambda v: sum(v) / len(v)), agg(prop("is_entry_point"), max), agg(prop("is_test"), max),
           sum(1 for c in touched if c[0] == "Method") / max(1, len(touched)),
           math.log1p(g["nodes"]), math.log1p(g["edges"]), math.log1p(len(g["links"]["TESTS"]))]
    v = np.array(vec, np.float32)
    return np.sign(v) * np.log1p(np.abs(v))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--exe", type=Path, required=True)
    p.add_argument("--scratch", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    started = time.perf_counter()
    a.output.mkdir(parents=True, exist_ok=True); (a.output / "graphs").mkdir(exist_ok=True); a.scratch.mkdir(parents=True, exist_ok=True)
    recs = [json.loads(l) for l in a.inputs.read_text(encoding="utf-8").splitlines()]
    tasks = {}
    for r in recs:
        tasks.setdefault(r["task"], r["source"])
    status = {}

    def one(item):
        task, source = item
        dest = a.output / "graphs" / (hashlib.sha256(task.encode()).hexdigest()[:16] + ".json.gz")
        if dest.exists():
            return task, "cached"
        try:
            g = graph(a.exe, a.root, task, source, a.scratch)
            with gzip.open(dest.with_suffix(".tmp"), "wt", encoding="utf-8") as f:
                json.dump(g, f)
            dest.with_suffix(".tmp").rename(dest)
            return task, "indexed"
        except Exception as exc:  # a task without a graph is a named gap for this sense
            return task, f"gap: {str(exc)[:200]}"
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for task, st in pool.map(one, sorted(tasks.items())):
            status[task] = st; print(task, st, flush=True)
    graphs = {}
    for task in tasks:
        dest = a.output / "graphs" / (hashlib.sha256(task.encode()).hexdigest()[:16] + ".json.gz")
        if dest.exists():
            with gzip.open(dest, "rt", encoding="utf-8") as f:
                graphs[task] = json.load(f)
    rows_ok = np.array([r["task"] in graphs for r in recs])
    M = np.stack([features(graphs[r["task"]], r["patch"]) if r["task"] in graphs else np.zeros(24, np.float32) for r in recs])
    np.save(a.output / "graph.npy", M.astype(np.float32)); np.save(a.output / "graph_rows.npy", np.flatnonzero(rows_ok))
    publish(a.output / "manifest.json", {"schema": "swe-graph-sense-v1", "version": VERSION, "excluded_top_level": sorted(EXCLUDED_TOP),
        "engine": {"binary": str(a.exe), "sha256": hashlib.sha256(a.exe.read_bytes()).hexdigest()},
        "shape": list(M.shape), "rows_with_graph": int(rows_ok.sum()), "tasks": len(tasks),
        "status": status, "graph_sha256": hashlib.sha256((a.output / "graph.npy").read_bytes()).hexdigest(),
        "uid_order_sha256": hashlib.sha256("\n".join(r["uid"] for r in recs).encode()).hexdigest(),
        "inputs_sha256": hashlib.sha256(a.inputs.read_bytes()).hexdigest(),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "wall_seconds": time.perf_counter() - started})
    print(json.dumps({"tasks": len(tasks), "with_graph": len(graphs), "rows_with_graph": int(rows_ok.sum())}))


if __name__ == "__main__":
    main()
