"""Strict early recorded-tool prefixes; no outcome fields enter the input path.

Four complete exchanges plus the initial task prompt. A subsequent exchange must
exist, but its content and the trajectory's full length are never model inputs.
"""
from __future__ import annotations

import argparse
import copy
from bisect import bisect_left
from collections import Counter
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import time

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from senses.adapters import exact_ref
from senses.contract import SenseError, canonical, digest, field, present, require, seal_record
from senses.pipeline import file_hash, read_bound, write_records
from senses.recorded import base, exchange_records
from senses.trajectory_population import (CODE, open_bound, publish, read_pinned, select_metadata,
                                          validate_scope, validate_selection)

SCHEMA = "trajectory-prefix-v1"
EXCHANGES = 4
TURNS = 2 + EXCHANGES * 2
MAX_PREFIX_BYTES = 1024 * 1024


def source_binding(scope, shard):
    # An acquisition receipt or selection window must not change an observation.
    return {"schema": SCHEMA, "source": scope["source"], "revision": scope["revision"],
            "raw_sha256": shard["sha256"]}


def prefix_records(raw, shard, row, task, scope):
    require(isinstance(raw, dict) and set(raw) == {"instance_id", "model_name", "trajectory"},
            "MNLM_TRAJECTORY_PREFIX_FIELDS", "only prefix input columns are permitted")
    require(raw["instance_id"] == task, CODE, "selected task binding differs")
    turns = raw["trajectory"]
    require(isinstance(turns, list) and len(turns) == TURNS, "MNLM_TRAJECTORY_PREFIX_LENGTH",
            "exactly the declared prefix may enter the adapter")
    require(isinstance(raw["model_name"], str) and bool(raw["model_name"])
            and all(isinstance(t, dict) and isinstance(t.get("role"), str) for t in turns)
            and all(isinstance(t.get("text"), str) for t in turns[1:]),
            "MNLM_TRAJECTORY_FORMAT", "typed model, role and text fields are required")
    require(len(canonical(raw)) <= MAX_PREFIX_BYTES, "MNLM_TRAJECTORY_PREFIX_BUDGET",
            "retain the source and declare a smaller structural unit; no silent truncation")
    binding = source_binding(scope, shard)
    # The existing exchange adapter validates the preamble and role sequence.
    exchanges = list(exchange_records(raw, shard["path"], row, shard["sha256"], binding))
    require(len(exchanges) == EXCHANGES and all(r["channels"]["behaviour"]["state"] == "present" for r in exchanges),
            "MNLM_TRAJECTORY_PREFIX_LENGTH", "four complete recorded exchanges required")
    prompt = turns[1]["text"]
    require(isinstance(prompt, str) and prompt, "MNLM_TRAJECTORY_FORMAT", "nonempty initial task prompt required")
    unit = digest(["swe-task", task])
    record = base(scope["source"], Path(shard["path"]).name + f"#trajectory={row}", 1, [0, len(prompt)],
                  digest({"prompt": turns[1], "instance_id": task, "model_name": raw["model_name"]}), binding, unit)
    record["referents"] = [exact_ref("swe-task:" + task, "instance_id")]
    record["channels"]["surface"] = present([{"text": prompt}])
    record["channels"]["fields"] = present([field("turn_index", 1, "integer", "sequence_index")])
    record["channels"]["structure"] = present([field("kind", "task_prompt", "category")])
    record["channels"]["provenance"] = exchanges[0]["channels"]["provenance"]
    return [seal_record(record), *exchanges]


def bounded_prefix_rows(parquet, chosen):
    """Slice Arrow nested arrays before creating any Python trajectory object."""
    indices, offset = sorted(chosen), 0
    for batch in parquet.iter_batches(batch_size=16, columns=["instance_id", "model_name", "trajectory"]):
        end = offset + batch.num_rows
        wanted = indices[bisect_left(indices, offset):bisect_left(indices, end)]
        if wanted:
            taken = batch.take([i - offset for i in wanted])
            lengths = pc.list_value_length(taken.column(2)).to_pylist()
            clipped = pa.RecordBatch.from_arrays([taken.column(0), taken.column(1),
                                                  pc.list_slice(taken.column(2), 0, TURNS)],
                                                 names=["instance_id", "model_name", "trajectory"])
            for position, (index, length, raw) in enumerate(zip(wanted, lengths, clipped.to_pylist())):
                require(raw["instance_id"] == chosen[index], CODE, "source task binding differs")
                has_future = type(length) is int and length >= TURNS + 2
                if has_future:
                    # Inspect only two role scalars, never future text or payload.
                    roles = taken.column(2)[position].values.field("role")
                    has_future = roles[TURNS:TURNS + 2].to_pylist() == ["ai", "user"]
                # Only this Boolean leaves the future inspection, never a feature.
                yield index, has_future, raw
        offset = end


def build(root, selection_path, selection_sha256, output):
    started = time.perf_counter()
    selected = read_pinned(selection_path, selection_sha256)
    require(isinstance(selected, dict) and "scope" in selected, CODE, "selection object required")
    scope = validate_scope(selected["scope"])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    index, counts = [], Counter()
    with ExitStack() as stack:
        parquet = {s["path"]: pq.ParquetFile(stack.enter_context(open_bound(root, s))) for s in scope["shards"]}
        def batches(shard):
            p = parquet[shard["path"]]
            require(p.metadata.num_rows == shard["rows"], CODE, "Parquet row count changed")
            for batch in p.iter_batches(batch_size=1024, columns=["instance_id"]):
                yield batch.column(0).to_pylist()
        validate_selection(selected, select_metadata(scope, batches))
        metadata_seconds = time.perf_counter() - started

        def records():
            for shard in scope["shards"]:
                chosen = {r["row"]: r["task"] for r in selected["selection"] if r["shard"] == shard["path"]}
                for row, has_future, raw in bounded_prefix_rows(parquet[shard["path"]], chosen):
                    entry = {"task": chosen[row], "shard": shard["path"], "row": row,
                             "unit": digest(["swe-task", chosen[row]])}
                    if not has_future:
                        status, adapted = "MNLM_TRAJECTORY_NO_FUTURE_EXCHANGE", []
                    else:
                        try:
                            adapted = prefix_records(raw, shard, row, chosen[row], scope)
                            status = "prefix_retained"
                        except SenseError as exc:
                            status, adapted = exc.code, []
                    entry.update(status=status, past=[r["atom_id"] for r in adapted[:-1]],
                                 current=adapted[-1]["atom_id"] if adapted else None)
                    counts[status] += 1
                    index.append(entry)
                    yield from adapted

        record_manifest = write_records(output / "inputs.ucr.jsonl", records(),
                                        {"schema": SCHEMA, "selection_sha256": selection_sha256,
                                         "exchanges": EXCHANGES, "admitted": False})
    require(len(index) == selected["count"], CODE, "prefix export lost selected tasks")
    publish(output / "prefix-index.json", index)
    files = {name: {"sha256": file_hash(output / name), "bytes": (output / name).stat().st_size}
             for name in ("inputs.ucr.jsonl", "inputs.ucr.manifest.json", "prefix-index.json")}
    manifest = {"schema": SCHEMA, "selection_sha256": selection_sha256, "scope": scope,
                "count": len(index), "record_count": record_manifest["count"], "files": files,
                "status_counts": dict(sorted(counts.items())), "prefix_exchanges": EXCHANGES,
                "max_prefix_bytes": MAX_PREFIX_BYTES, "input_columns": ["instance_id", "model_name", "trajectory"],
                "excluded_columns": ["target", "eval_logs", "exit_status", "generated_patch"],
                "future_turns_retained": False, "admitted": False}
    manifest_sha = publish(output / "manifest.json", manifest)
    publish(output / "COMMITTED", {"manifest_sha256": manifest_sha})
    return {"manifest_sha256": manifest_sha, "records": manifest["record_count"], "status_counts": manifest["status_counts"],
            "input_bytes": files["inputs.ucr.jsonl"]["bytes"], "admitted": False,
            "timings": {"selection_seconds": metadata_seconds,
                        "prefix_export_seconds": time.perf_counter() - started - metadata_seconds,
                        "wall_seconds": time.perf_counter() - started}}


def load(root, manifest_sha256):
    root = Path(root)
    manifest = read_pinned(root / "manifest.json", manifest_sha256)
    require(manifest.get("schema") == SCHEMA and manifest.get("prefix_exchanges") == EXCHANGES
            and manifest.get("max_prefix_bytes") == MAX_PREFIX_BYTES and manifest.get("admitted") is False,
            "MNLM_TRAJECTORY_PREFIX_MANIFEST", "unknown prefix contract")
    scope = validate_scope(manifest["scope"])
    require(manifest["count"] == scope["distinct_tasks"] and manifest.get("future_turns_retained") is False
            and manifest.get("input_columns") == ["instance_id", "model_name", "trajectory"]
            and manifest.get("excluded_columns") == ["target", "eval_logs", "exit_status", "generated_patch"],
            "MNLM_TRAJECTORY_PREFIX_MANIFEST", "input boundary differs from the fixed contract")
    require(json.loads((root / "COMMITTED").read_bytes()) == {"manifest_sha256": manifest_sha256},
            "MNLM_TRAJECTORY_PREFIX_UNCOMMITTED", "publication not committed")
    require(set(manifest["files"]) == {"inputs.ucr.jsonl", "inputs.ucr.manifest.json", "prefix-index.json"},
            "MNLM_TRAJECTORY_PREFIX_MANIFEST", "unexpected input files")
    for name, binding in manifest["files"].items():
        require((root / name).stat().st_size == binding["bytes"]
                and (name == "inputs.ucr.jsonl" or file_hash(root / name) == binding["sha256"]),
                "MNLM_UCR_INPUT_DIGEST", "prefix file changed")
    from senses.contract import validate_record
    records = [validate_record(r) for r in read_bound(root / "inputs.ucr.jsonl", manifest["files"]["inputs.ucr.jsonl"]["sha256"])]
    indexed = {r["atom_id"]: r for r in records}
    require(len(indexed) == len(records) == manifest["record_count"], CODE, "duplicate or missing prefix records")
    rows = json.loads((root / "prefix-index.json").read_bytes())
    require(len(rows) == manifest["count"] and len({r["task"] for r in rows}) == len(rows), CODE, "task index count differs")
    require(dict(Counter(r["status"] for r in rows)) == manifest["status_counts"], CODE, "gap inventory differs")
    source_shards = {s["path"]: s for s in scope["shards"]}
    used = []
    for row in rows:
        require(set(row) == {"task", "shard", "row", "unit", "status", "past", "current"}
                and row["shard"] in source_shards and type(row["row"]) is int
                and 0 <= row["row"] < source_shards[row["shard"]]["rows"], CODE, "prefix source index differs")
        if row["status"] != "prefix_retained":
            require(row["past"] == [] and row["current"] is None, CODE, "refusal contains a model input")
            continue
        keys = [*row["past"], row["current"]]
        require(len(keys) == EXCHANGES + 1 and all(k in indexed for k in keys), CODE, "prefix index is incomplete")
        group = [indexed[k] for k in keys]
        require(all(r["unit"] == row["unit"] == digest(["swe-task", row["task"]]) for r in group)
                and [r["locator"]["row"] for r in group] == [1, 2, 4, 6, 8]
                and all(r["locator"]["shard"] == Path(row["shard"]).name + f"#trajectory={row['row']}"
                        and r["locator"]["source"] == scope["source"]
                        and r["locator"]["manifest_digest"] == digest(source_binding(scope, source_shards[row["shard"]]))
                        and [f["value"] for f in r["channels"]["fields"]["items"] if f["name"] == "turn_index"] == [r["locator"]["row"]]
                        for r in group),
                "MNLM_TRAJECTORY_PREFIX_ORDER", "past/current cross task or exceed the declared cutoff")
        used.extend(keys)
    require(len(used) == len(set(used)) and set(used) == set(indexed), CODE, "orphan or reused prefix record")
    return manifest, rows, indexed


class PrefixInputs:
    """Source adapter for the shared frozen-instrument prefix reader interface.

    Only the sealed current/past inputs are opened. This reader never opens the
    outcome audit, source trajectory, or any final label/patch/transcript file.
    """
    def __init__(self, root, expected_manifest_sha256):
        self.manifest, rows, self.records = load(root, expected_manifest_sha256)
        self.examples = {r["current"]: r for r in rows if r["status"] == "prefix_retained"}
        self.identity = expected_manifest_sha256
        self.reader_code_sha256 = file_hash(Path(__file__))

    def read(self, atom):
        require(atom in self.examples, "MNLM_PREFIX_NOT_FOUND", "only sealed current observations may open a prefix")
        e = self.examples[atom]
        current, past = self.records[atom], [self.records[key] for key in e["past"]]
        require(len(past) == EXCHANGES and [r["locator"]["row"] for r in [*past, current]] == [1, 2, 4, 6, 8]
                and all(r["unit"] == current["unit"] == e["unit"] for r in past),
                "MNLM_TRAJECTORY_PREFIX_ORDER", "prefix ordering or unit changed after loading")
        return copy.deepcopy(current), copy.deepcopy(past), {
            "input_manifest_sha256": self.identity, "current": atom, "visible_past": list(e["past"]),
            "cutoff": {"field": "turn_index", "unit": "sequence_index", "value": 8},
            "future_visible": False, "outcomes_visible": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--selection-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(build(args.root, args.selection, args.selection_sha256, args.output)))
    except (SenseError, OSError, ValueError) as exc:
        error = exc if isinstance(exc, SenseError) else SenseError(CODE, str(exc))
        print(json.dumps(error.envelope()), file=sys.stderr)
        sys.exit(2)
