"""Create/read a finite native-time view population. One source parent resident."""
import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from native_time import build, canonical, digest, require
from native_time_readback import audit_graph
from senses.pipeline import file_hash, read_bound


def inputs(plan):
    require(plan.get("schema") == "native-time-plan-v1" and plan.get("exports"), "MNLM_TIME_PLAN", "pinned export plan required")
    for entry in plan["exports"]:
        path = Path(entry["path"])
        mp = path.with_suffix(".manifest.json")
        require(file_hash(mp) == entry["manifest_sha256"], "MNLM_TIME_INPUT", "source manifest changed")
        m = json.loads(mp.read_bytes())
        require(m["schema"] == "ucr-export-v1" and m["sha256"] == entry["sha256"]
                and m["count"] == entry["count"] and path.stat().st_size == entry["bytes"],
                "MNLM_TIME_INPUT", "source export scope changed")
    return plan["exports"]


def source_records(entries):
    for entry in entries:
        count = 0
        for r in read_bound(Path(entry["path"]), entry["sha256"]):
            count += 1
            yield r
        require(count == entry["count"], "MNLM_TIME_INPUT", "source row count changed")


def next_step(graph, previous):
    if not graph["clock"] or graph["clock"]["unit"] != "step":
        return None
    n = graph["nodes"][0]
    key = graph["clock"]["id"]
    before = previous.get(key)
    require(before is None or before["span"][1] <= n["span"][0], "MNLM_TIME_ORDER", "reordered or overlapping source step")
    previous[key] = {**n, "split_group": graph["split_group"]}
    if before and before["span"][1] == n["span"][0]:
        require(before["split_group"] == graph["split_group"], "MNLM_TIME_SPLIT", "episode crosses split groups")
        return {"origin": before["id"], "target": n["id"], "kind": "observed_next", "clock": key}
    return None


def publish(plan, output):
    start = time.perf_counter()
    entries = inputs(plan)
    output.mkdir(parents=True, exist_ok=False)
    (output / "plan.json").write_bytes(canonical(plan))
    preflight = time.perf_counter() - start
    elapsed = Counter()
    counts, previous = Counter(), {}
    t = time.perf_counter()
    with (output / "views.jsonl").open("wb") as nodes, (output / "next.jsonl").open("wb") as edges:
        for record in source_records(entries):
            elapsed["input"] += time.perf_counter() - t
            t = time.perf_counter()
            graph = build(record)
            next_edge = next_step(graph, previous)
            elapsed["projection"] += time.perf_counter() - t
            t = time.perf_counter()
            nodes.write(canonical(graph) + b"\n")
            if next_edge:
                edges.write(canonical(next_edge) + b"\n")
            counts.update({"parents": 1, "nodes": len(graph["nodes"]), "within_parent_edges": len(graph["edges"]),
                           "between_parent_edges": int(next_edge is not None), "untimed_parents": int(graph["clock"] is None)})
            counts.update(n["level"] for n in graph["nodes"])
            elapsed["write"] += time.perf_counter() - t
            t = time.perf_counter()
    elapsed["input"] += time.perf_counter() - t
    t = time.perf_counter()
    files = {name: {"bytes": (output / name).stat().st_size, "sha256": file_hash(output / name)} for name in ("views.jsonl", "next.jsonl")}
    elapsed["publication_hash"] += time.perf_counter() - t
    wall = time.perf_counter() - start
    report = {"schema": "native-time-publication-v1", "plan_sha256": digest(plan), "files": files, "counts": dict(counts),
              "code_sha256": {n: file_hash(Path(__file__).parent / n) for n in ("native_time.py", "native_time_population.py", "native_time_readback.py")},
              "timing": {"wall": wall, "preflight": preflight, **elapsed, "phase_coverage": (preflight + sum(elapsed.values())) / wall},
              "meaning_asserted": False, "scientific_admission": False, "synchronized_avt_proven": False}
    (output / "report.json").write_bytes(canonical(report))
    return report


def readback(plan, root):
    start = time.perf_counter()
    entries = inputs(plan)
    report = json.loads((root / "report.json").read_bytes())
    original = json.loads((root / "plan.json").read_bytes())
    require(report["plan_sha256"] == digest(original)
            and [{k: v for k, v in e.items() if k != "path"} for e in entries]
            == [{k: v for k, v in e.items() if k != "path"} for e in original["exports"]],
            "MNLM_TIME_READBACK_PLAN", "relocated inputs must preserve every pin")
    require(all(report[k] is False for k in ("meaning_asserted", "scientific_admission", "synchronized_avt_proven")),
            "MNLM_TIME_READBACK_CLAIM", "native views do not certify scientific claims")
    require(set(report["files"]) == {"views.jsonl", "next.jsonl"}, "MNLM_TIME_READBACK_FILES", "publication scope changed")
    for name, spec in report["files"].items():
        require((root / name).stat().st_size == spec["bytes"] and file_hash(root / name) == spec["sha256"],
                "MNLM_TIME_READBACK_DIGEST", "publication bytes changed")
    counts, last = Counter(), {}
    with (root / "views.jsonl").open("rb") as f, (root / "next.jsonl").open("rb") as e:
        for record, line in itertools.zip_longest(source_records(entries), f):
            require(record is not None and line is not None, "MNLM_TIME_READBACK_COVERAGE", "parent omitted or added")
            g = json.loads(line)
            result = audit_graph(record, g)
            counts.update({"parents": 1, "nodes": result["nodes"], "within_parent_edges": result["edges"], "untimed_parents": result["untimed_parents"]})
            counts.update(n["level"] for n in g["nodes"])
            if g["clock"] and g["clock"]["unit"] == "step":
                n, clock = g["nodes"][0], g["clock"]["id"]
                before = last.get(clock)
                if before:
                    require(before["span"][1] <= n["span"][0], "MNLM_TIME_READBACK_ORDER", "source reordered")
                    if before["span"][1] == n["span"][0]:
                        require(before["group"] == g["split_group"], "MNLM_TIME_READBACK_ORDER", "split crossed")
                        expected = {"origin": before["id"], "target": n["id"], "kind": "observed_next", "clock": clock}
                        raw = e.readline()
                        require(raw and json.loads(raw) == expected, "MNLM_TIME_READBACK_ORDER", "missing or fabricated next step")
                        counts["between_parent_edges"] += 1
                last[clock] = {"span": n["span"], "id": n["id"], "group": g["split_group"]}
        require(not e.read(1) and dict(counts) == report["counts"], "MNLM_TIME_READBACK_COUNTS", "extra edges or wrong denominator")
    return {"author_checked": True, "counts": dict(counts), "report_sha256": file_hash(root / "report.json"),
            "seconds": time.perf_counter() - start, "synchronized_avt_proven": False, "scientific_admission": False}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--readback", action="store_true")
    args = ap.parse_args()
    plan = json.loads(args.plan.read_bytes())
    result = readback(plan, args.output) if args.readback else publish(plan, args.output)
    print(canonical(result).decode())
