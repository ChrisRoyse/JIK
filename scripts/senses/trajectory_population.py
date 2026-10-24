"""Global outcome-blind task selection over an explicitly pinned raw population.

PC-07/13: hash each named shard once; read only identifier columns in batches.
The old single-shard inspection sampler is deliberately a separate contract.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys
import tempfile

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from senses.contract import SenseError, canonical, digest, is_sha256, require

SCHEMA = "trajectory-population-v1"
SAMPLER = "global-one-attempt-per-task-v1"
SEED = "mnlm-global-task-units-20261017"
CODE = "MNLM_TRAJECTORY_POPULATION"


def validate_scope(scope):
    require(isinstance(scope, dict) and set(scope) == {
        "schema", "source", "revision", "approval", "shards", "distinct_tasks"
    }, CODE, "explicit population scope required")
    require(scope["schema"] == SCHEMA and scope["source"] == "swe-agent-trajectories"
            and isinstance(scope["revision"], str) and len(scope["revision"]) == 40
            and all(c in "0123456789abcdef" for c in scope["revision"])
            and isinstance(scope["approval"], str) and bool(scope["approval"]),
            CODE, "pin the provider revision and population decision")
    require(type(scope["distinct_tasks"]) is int and scope["distinct_tasks"] > 0
            and isinstance(scope["shards"], list) and scope["shards"],
            CODE, "measured task count and named shards required")
    names, hashes = set(), set()
    for shard in scope["shards"]:
        require(isinstance(shard, dict) and set(shard) == {"path", "sha256", "bytes", "rows"},
                CODE, "shard keys")
        name = shard["path"]
        require(isinstance(name, str) and name and "\\" not in name and ":" not in name,
                CODE, "portable relative shard path required")
        path = PurePosixPath(name)
        require(not path.is_absolute() and ".." not in path.parts and str(path) == name
                and path.suffix == ".parquet", CODE, "invalid shard path")
        require(name not in names and is_sha256(shard["sha256"]) and shard["sha256"] not in hashes
                and all(type(shard[k]) is int and shard[k] > 0 for k in ("bytes", "rows")),
                CODE, "duplicate shard or invalid byte/row/digest contract")
        names.add(name)
        hashes.add(shard["sha256"])
    require(scope["distinct_tasks"] <= sum(s["rows"] for s in scope["shards"]),
            CODE, "task count exceeds attempt count")
    return {**scope, "shards": sorted(scope["shards"], key=lambda s: s["path"])}


def bound_path(root, shard):
    root = Path(root).resolve()
    path = (root / shard["path"]).resolve()
    require(path.is_relative_to(root) and path != root, CODE, "shard escapes retained source root")
    return path


def open_bound(root, shard):
    """Return the same open file that was hashed, avoiding a second path open."""
    path = bound_path(root, shard)
    stream = path.open("rb")
    try:
        require(os.fstat(stream.fileno()).st_size == shard["bytes"], CODE, "raw byte count changed")
        require(hashlib.file_digest(stream, "sha256").hexdigest() == shard["sha256"],
                "MNLM_UCR_INPUT_DIGEST", "raw shard digest changed")
        stream.seek(0)
        return stream
    except BaseException:
        stream.close()
        raise


def select_metadata(scope, batches):
    """batches(shard) yields ID lists; never receives outcomes or trajectories."""
    scope = validate_scope(scope)
    best, counts, membership = {}, {}, {}
    for shard in scope["shards"]:
        row = 0
        for batch in batches(shard):
            for task in batch:
                require(isinstance(task, str) and bool(task), CODE, "empty task identity")
                rank = digest([SAMPLER, SEED, "attempt", task, shard["sha256"], row])
                key = (rank, shard["sha256"], row)
                if task not in best or key < best[task][0]:
                    best[task] = key, {"task": task, "shard": shard["path"], "row": row}
                membership.setdefault(task, set()).add(shard["path"])
                row += 1
        require(row == shard["rows"], CODE, "identifier census differs from declared rows")
        counts[shard["path"]] = row
    require(len(best) == scope["distinct_tasks"], CODE, "task census differs from declared population")
    selected = sorted((value[1] for value in best.values()), key=lambda v: (v["shard"], v["row"]))
    return {"schema": SCHEMA, "sampler": SAMPLER, "seed": SEED, "scope": scope,
            "scope_sha256": digest(scope), "selection": selected, "count": len(selected),
            "source_rows": sum(counts.values()),
            "cross_shard_tasks": sum(len(v) > 1 for v in membership.values()),
            "outcome_fields_read": [], "admitted": False}


def plan(root, scope):
    def batches(shard):
        with open_bound(root, shard) as stream:
            parquet = pq.ParquetFile(stream)
            require(parquet.metadata.num_rows == shard["rows"], CODE, "Parquet row count changed")
            for batch in parquet.iter_batches(batch_size=1024, columns=["instance_id"]):
                yield batch.column(0).to_pylist()
    return select_metadata(scope, batches)


def validate_selection(selection, expected):
    require(canonical(selection) == canonical(expected), CODE,
            "selection differs from independently re-derived identifier-only population")
    return selection


def publish(path, value):
    """Complete, fsynced bytes become visible exclusively; no partial acceptance."""
    path = Path(path)
    data = canonical(value) + b"\n"
    require(not path.exists(), "MNLM_UCR_OUTPUT_EXISTS", str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise SenseError("MNLM_UCR_OUTPUT_EXISTS", str(path)) from exc
        if os.name != "nt":
            directory = os.open(path.parent, os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        require(path.read_bytes() == data, CODE, "published bytes differ")
    finally:
        temporary.unlink(missing_ok=True)
    return hashlib.sha256(data).hexdigest()


def read_pinned(path, sha256):
    data = Path(path).read_bytes()
    require(is_sha256(sha256) and hashlib.sha256(data).hexdigest() == sha256,
            "MNLM_UCR_INPUT_DIGEST", "scope/selection file changed")
    return json.loads(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--scope", type=Path, required=True)
    parser.add_argument("--scope-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = plan(args.root, read_pinned(args.scope, args.scope_sha256))
    sha = publish(args.output, result)
    print(json.dumps({"output": str(args.output), "sha256": sha, "tasks": result["count"],
                      "cross_shard_tasks": result["cross_shard_tasks"], "admitted": False}))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (SenseError, OSError, ValueError) as exc:
        error = exc if isinstance(exc, SenseError) else SenseError(CODE, str(exc))
        print(json.dumps(error.envelope()), file=sys.stderr)
        sys.exit(2)
