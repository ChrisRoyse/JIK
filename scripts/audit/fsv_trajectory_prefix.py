"""Source-to-prefix replay, temporal-boundary controls and immutable input audit."""
import argparse
from bisect import bisect_left
import copy
import json
import os
from pathlib import Path
import sys
import time

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from senses.contract import SenseError, canonical, require
from senses.pipeline import file_hash
from senses.trajectory_population import publish
from senses.trajectory_prefix import (TURNS, bounded_prefix_rows, load,
                                       prefix_records)


def run(a):
    started = time.perf_counter()
    a.output.mkdir(parents=True, exist_ok=False)
    checks = []

    def check(name, ok, **details):
        checks.append({"name": name, "pass": bool(ok), **details})
        require(ok, "MNLM_TRAJECTORY_PREFIX_FSV", name)

    def refused(name, action, code):
        try:
            action()
        except SenseError as e:
            check(name, e.code == code, code=e.code)
        else:
            check(name, False)

    manifest, index, records = load(a.prefix, a.manifest_sha256)
    source_hashes, count, sample = {}, 0, None
    before = {name: file_hash(a.prefix / name) for name in (*manifest["files"], "manifest.json", "COMMITTED")}
    # B1: a separate complete source reader, different batch boundaries. Slice
    # selected Python rows here only for independent audit; never feed full rows
    # to an encoder or a training consumer.
    for shard in manifest["scope"]["shards"]:
        source = a.root / shard["path"]
        source_hashes[shard["path"]] = file_hash(source)
        check("raw_source_" + source.name, source_hashes[shard["path"]] == shard["sha256"])
        wanted = {r["row"]: r for r in index if r["shard"] == shard["path"]}
        indices, offset = sorted(wanted), 0
        for batch in pq.ParquetFile(source).iter_batches(batch_size=11, columns=["instance_id", "model_name", "trajectory"]):
            end = offset + batch.num_rows
            picked = indices[bisect_left(indices, offset):bisect_left(indices, end)]
            if picked:
                for row, raw in zip(picked, batch.take([i - offset for i in picked]).to_pylist()):
                    entry = wanted[row]
                    require(entry["task"] == raw["instance_id"], "MNLM_TRAJECTORY_PREFIX_FSV", "source task changed")
                    if entry["status"] == "prefix_retained":
                        require(len(raw["trajectory"]) >= TURNS + 2
                                and [t["role"] for t in raw["trajectory"][TURNS:TURNS + 2]] == ["ai", "user"],
                                "MNLM_TRAJECTORY_PREFIX_FSV", "no later recorded exchange")
                        cut = {**raw, "trajectory": raw["trajectory"][:TURNS]}
                        expected = prefix_records(cut, shard, row, entry["task"], manifest["scope"])
                        actual = [records[k] for k in [*entry["past"], entry["current"]]]
                        require(actual == expected, "MNLM_TRAJECTORY_PREFIX_FSV", "input differs from exact retained prefix")
                        # Independent content checks don't rely on adapter hashing.
                        require(actual[0]["channels"]["surface"]["items"] == [{"text": raw["trajectory"][1]["text"]}],
                                "MNLM_TRAJECTORY_PREFIX_FSV", "initial task prompt changed")
                        for position, r in zip((2, 4, 6, 8), actual[1:]):
                            response = next(f["value"] for f in r["channels"]["behaviour"]["items"][0]["measurements"] if f["name"] == "tool_response")
                            require(r["channels"]["surface"]["items"] == [{"text": raw["trajectory"][position]["text"]}]
                                    and response == raw["trajectory"][position + 1]["text"],
                                    "MNLM_TRAJECTORY_PREFIX_FSV", "action or response crossed the cutoff")
                        if sample is None:
                            sample = (raw, cut, shard, row, entry)
                        count += len(actual)
                    else:
                        require(entry["status"] == "MNLM_TRAJECTORY_NO_FUTURE_EXCHANGE"
                                and len(raw["trajectory"]) < TURNS + 2, "MNLM_TRAJECTORY_PREFIX_FSV", "unexplained source refusal")
            offset = end
    check("every_input_record_cold_rederived", count == len(records), records=count, tasks=len(index))
    check("input_files_exclude_outcome_columns", manifest["excluded_columns"] == ["target", "eval_logs", "exit_status", "generated_patch"]
          and manifest["future_turns_retained"] is False)

    # B2: source-derived negative controls exercise the real Arrow column/slice
    # reader. They are fault injections only, never extra training observations.
    raw, cut, shard, row, entry = sample
    baseline_path = a.output / "control-before.parquet"
    mutated_path = a.output / "control-after.parquet"
    altered = copy.deepcopy(raw)
    for turn in altered["trajectory"][TURNS:]:
        turn["text"] = "FUTURE_NEGATIVE_CONTROL_MUST_NOT_ENTER_INPUT"
    altered.update(target=True, eval_logs="ORACLE_NEGATIVE_CONTROL", exit_status="FINAL_STATUS", generated_patch="FINAL_PATCH")
    pq.write_table(pa.Table.from_pylist([raw]), baseline_path)
    pq.write_table(pa.Table.from_pylist([altered]), mutated_path)
    left = list(bounded_prefix_rows(pq.ParquetFile(baseline_path), {0: raw["instance_id"]}))
    right = list(bounded_prefix_rows(pq.ParquetFile(mutated_path), {0: raw["instance_id"]}))
    check("future_and_outcome_mutations_cannot_change_input", left == right and len(left) == 1
          and left[0][1] is True and len(left[0][2]["trajectory"]) == TURNS)
    altered["trajectory"][TURNS + 1]["role"] = "ai"
    role_path = a.output / "control-future-role.parquet"
    pq.write_table(pa.Table.from_pylist([altered]), role_path)
    role_result = list(bounded_prefix_rows(pq.ParquetFile(role_path), {0: raw["instance_id"]}))
    check("future_length_is_not_exchange_evidence", len(role_result) == 1 and role_result[0][1] is False
          and role_result[0][2] == left[0][2])
    refused("wrong_task", lambda: prefix_records(cut, shard, row, "wrong-task", manifest["scope"]), "MNLM_TRAJECTORY_POPULATION")
    refused("outcome_field_in_adapter", lambda: prefix_records({**cut, "target": True}, shard, row, entry["task"], manifest["scope"]), "MNLM_TRAJECTORY_PREFIX_FIELDS")
    refused("full_future_in_adapter", lambda: prefix_records(raw, shard, row, entry["task"], manifest["scope"]), "MNLM_TRAJECTORY_PREFIX_LENGTH")
    bad = copy.deepcopy(cut)
    bad["trajectory"][3]["role"] = "ai"
    refused("invalid_action_response_order", lambda: prefix_records(bad, shard, row, entry["task"], manifest["scope"]), "MNLM_TRAJECTORY_FORMAT")
    bad = copy.deepcopy(cut)
    bad["trajectory"][3] = None
    refused("malformed_turn_is_typed_refusal", lambda: prefix_records(bad, shard, row, entry["task"], manifest["scope"]), "MNLM_TRAJECTORY_FORMAT")
    bad = copy.deepcopy(cut)
    bad["trajectory"][1]["text"] = "x" * (1024 * 1024)
    refused("prefix_budget_no_truncation", lambda: prefix_records(bad, shard, row, entry["task"], manifest["scope"]), "MNLM_TRAJECTORY_PREFIX_BUDGET")

    # B3: resealed shadow indexes must still satisfy task/order/source semantics.
    # Input records are hardlinked read-only in intent and never mutated here.
    def shadow(name, change):
        dest = a.output / name
        dest.mkdir()
        for filename in ("inputs.ucr.jsonl", "inputs.ucr.manifest.json"):
            os.link(a.prefix / filename, dest / filename)
        rows, m = copy.deepcopy(index), copy.deepcopy(manifest)
        change(rows, m)
        publish(dest / "prefix-index.json", rows)
        m["files"]["prefix-index.json"] = {"sha256": file_hash(dest / "prefix-index.json"), "bytes": (dest / "prefix-index.json").stat().st_size}
        sha = publish(dest / "manifest.json", m)
        publish(dest / "COMMITTED", {"manifest_sha256": sha})
        return dest, sha
    good_indices = [i for i, r in enumerate(index) if r["status"] == "prefix_retained"]
    i, j = good_indices[:2]
    for name, change, code in [
        ("cross_task_past", lambda rows, m: rows[i]["past"].__setitem__(0, rows[j]["past"][0]), "MNLM_TRAJECTORY_PREFIX_ORDER"),
        ("reordered_past", lambda rows, m: rows[i]["past"].reverse(), "MNLM_TRAJECTORY_PREFIX_ORDER"),
        ("current_as_past", lambda rows, m: rows[i]["past"].__setitem__(0, rows[i]["current"]), "MNLM_TRAJECTORY_PREFIX_ORDER"),
        ("changed_source_identity", lambda rows, m: m["scope"]["shards"][0].update(sha256="0" * 64), "MNLM_TRAJECTORY_PREFIX_ORDER"),
        ("future_feature_claim", lambda rows, m: m.update(future_turns_retained=True), "MNLM_TRAJECTORY_PREFIX_MANIFEST"),
    ]:
        dest, sha = shadow(name, change)
        refused(name, lambda: load(dest, sha), code)
    dest, sha = shadow("uncommitted", lambda rows, m: None)
    (dest / "COMMITTED").write_bytes(canonical({"manifest_sha256": "0" * 64}))
    refused("commit_digest_mismatch", lambda: load(dest, sha), "MNLM_TRAJECTORY_PREFIX_UNCOMMITTED")
    check("original_prefix_unchanged", before == {name: file_hash(a.prefix / name) for name in before})
    check("raw_sources_unchanged", all(file_hash(a.root / s["path"]) == source_hashes[s["path"]] for s in manifest["scope"]["shards"]))
    report = {"schema": "trajectory-prefix-fsv-v1", "checks": checks, "author_checked": True,
              "independently_adjudicated": False, "admitted": False, "manifest_sha256": a.manifest_sha256,
              "records": len(records), "tasks": len(index), "status_counts": manifest["status_counts"],
              "code_sha256": {p.name: file_hash(p) for p in (Path(__file__).resolve(),
                                Path(__file__).resolve().parents[1] / "senses/trajectory_prefix.py",
                                Path(__file__).resolve().parents[1] / "senses/recorded.py")},
              "wall_seconds": time.perf_counter() - started}
    sha = publish(a.output / "report.json", report)
    print(json.dumps({"report_sha256": sha, "checks": len(checks), "records": len(records),
                      "wall_seconds": report["wall_seconds"], "admitted": False}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "prefix", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    run(parser.parse_args())
