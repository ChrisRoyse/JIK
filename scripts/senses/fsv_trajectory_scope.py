"""Read selected tool exchanges back against real Parquet rows and scope changes."""
import argparse
from collections import defaultdict
import copy
import json
from pathlib import Path
import sys

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from senses.contract import SenseError, canonical, digest
from senses.pipeline import file_hash, ingest_file, read_bound, write_records
from senses.recorded import trajectory_records
from senses.store import Store
from senses.trajectory_scope import plan


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trajectory", type=Path, required=True)
    ap.add_argument("--scope", type=Path, required=True)
    ap.add_argument("--recordings", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    selection = json.loads(args.scope.read_bytes())
    sha = selection["raw_sha256"]
    src = args.recordings / "trajectories.ucr.jsonl"
    manifest = json.loads(src.with_suffix(".manifest.json").read_bytes())
    records = list(read_bound(src, manifest["sha256"]))
    checks = []

    def check(name, ok, **data):
        checks.append({"boundary": name, "pass": bool(ok), **data})

    check("scope_manifest_bound", manifest["input"]["trajectory_scope_sha256"] == file_hash(args.scope))
    check("deterministic_scope_replay", plan(args.trajectory, sha) == selection)
    byrow = defaultdict(list)
    for r in records:
        byrow[int(r["locator"]["shard"].rsplit("#trajectory=", 1)[1])].append(r)
    wanted = {r["row"]: r["task"] for r in selection["selection"]}
    check("all_and_only_selected_rows", set(byrow) == set(wanted))
    check("128_independent_task_units", len({r["unit"] for r in records}) == len(wanted) == 128)
    # Independent raw readback, with different batch boundaries. Never convert
    # unselected trajectories or outcome columns into Python records.
    offset, seen, exchanges, mismatches = 0, set(), 0, []
    for batch in pq.ParquetFile(args.trajectory).iter_batches(batch_size=37, columns=["instance_id", "model_name", "trajectory"]):
        indices = [i for i in range(batch.num_rows) if offset + i in wanted]
        chosen = batch.take(indices).to_pylist() if indices else []
        for i, raw in zip(indices, chosen):
            index = offset + i
            seen.add(index)
            turns, actual = raw["trajectory"], byrow[index]
            expected_steps = list(range(2, len(turns), 2))
            if raw["instance_id"] != wanted[index] or len(actual) != len(expected_steps):
                mismatches.append({"row": index, "kind": "identity_or_count"})
                continue
            for step, r in zip(expected_steps, actual):
                action = r["channels"]["surface"]["items"][0]["text"]
                b = r["channels"]["behaviour"]["items"]
                observed = next((f["value"] for f in b[0]["measurements"] if f["name"] == "tool_response"), None) if b else None
                expected_response = turns[step + 1]["text"] if step + 1 < len(turns) else None
                if action != turns[step]["text"] or observed != expected_response or r["locator"]["row"] != step:
                    mismatches.append({"row": index, "step": step, "kind": "action_or_response"})
                if r["unit"] != digest(["swe-task", raw["instance_id"]]):
                    mismatches.append({"row": index, "step": step, "kind": "sibling_unit"})
                exchanges += 1
        offset += batch.num_rows
    check("complete_raw_exchange_readback", seen == set(wanted) and not mismatches and exchanges == manifest["count"],
          exchanges=exchanges, tasks=len(seen), mismatches=mismatches)
    base_binding = manifest["input"]
    left = list(trajectory_records(args.trajectory, sha, {**base_binding, "trajectory_scope": "prefix"}, limit=1))
    right = list(trajectory_records(args.trajectory, sha, {**base_binding, "trajectory_scope": "overlap", "trajectory_scope_sha256": "f" * 64}, limit=1))
    check("overlap_preserves_native_identity", [canonical(r) for r in left] == [canonical(r) for r in right])
    store = Store(out / "channels.sqlite", create=True)
    for label, rows in (("first", left), ("overlap", right)):
        path = out / (label + ".ucr.jsonl")
        m = write_records(path, rows, {"scope": label})
        ingest_file(store, path, m, store.head)
        state = store.audit()
        check(label + "_physical_rows_not_duplicated", state["records"] == len(left) and state["slots"] == 9 * len(left), state=state)
    head = store.head
    for name, change in (("wrong_raw_identity", lambda s: s.update(raw_sha256="0" * 64)),
                         ("duplicate_task", lambda s: s["selection"][1].update(task=s["selection"][0]["task"])),
                         ("wrong_task_identity", lambda s: s["selection"][0].update(task="not-a-source-task")),
                         ("oracle_selected", lambda s: s.update(outcome_fields_read=["target"]))):
        bad = copy.deepcopy(selection)
        change(bad)
        path = out / (name + ".ucr.jsonl")
        error = None
        try:
            write_records(path, trajectory_records(args.trajectory, sha, base_binding, selection=bad), {"negative_control": name})
        except SenseError as exc:
            error = exc.code
        check(name + "_refused_before_publication", error == "MNLM_TRAJECTORY_SCOPE" and not path.exists()
              and not path.with_suffix(".manifest.json").exists() and not list(out.glob(path.name + ".tmp.*")), error=error)
    store.close()
    cold = Store(out / "channels.sqlite")
    check("cold_store_unchanged_by_refusals", cold.head == head and cold.audit()["records"] == len(left))
    cold.close()
    report = {"acceptance": "author-checked", "checks": checks, "passed": sum(c["pass"] for c in checks),
              "failed": sum(not c["pass"] for c in checks), "scope_sha256": file_hash(args.scope), "source_sha256": sha,
              "export_sha256": file_hash(src), "admitted": [], "served": False,
              "limitations": ["source-recorded actions/returns, not independently reexecuted", "128 task units; no claimed powered outcome study",
                              "old scope-dependent record identities remain historical; rebuild native exports for corrected identities"]}
    (out / "report.json").write_bytes(canonical(report))
    (out / "SHA256.json").write_bytes(canonical({str(p.relative_to(out)): file_hash(p) for p in out.rglob("*")
                                                if p.is_file() and not p.name.endswith(("-shm", "-wal"))}))
    print(json.dumps({k: report[k] for k in ("passed", "failed", "export_sha256")}))
    return int(report["failed"] != 0)


if __name__ == "__main__":
    sys.exit(main())
