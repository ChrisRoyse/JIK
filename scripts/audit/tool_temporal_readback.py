"""Independent raw-Parquet/alias/cutoff audit of the retained temporal source."""
from collections import Counter
import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from claims.episodic_memory import digest


def readback(raw_root, population):
    raw_root, population = Path(raw_root), Path(population)
    report = json.loads((population/"report.json").read_bytes())
    selection = json.loads((population/"selection.json").read_bytes())
    episodes = list(map(json.loads, (population/"episodes.jsonl").read_bytes().splitlines()))
    references = list(map(json.loads, (population/"references.jsonl").read_bytes().splitlines()))
    atoms = pq.read_table(population/"tool/chunks/atoms.parquet").to_pylist()
    units = list(map(json.loads, (population/"messages.jsonl").read_bytes().splitlines()))
    spans = list(map(json.loads, (population/"reading_spans.jsonl").read_bytes().splitlines()))
    by_message = {}
    for span in spans:
        by_message.setdefault(span["unit"], []).append(span)
    for i, unit in enumerate(units):
        char_end, byte_end, pieces = 0, 0, []
        for span in by_message[i]:
            start, end = span["char_span"]
            bs, be = span["utf8_span"]
            text = atoms[span["atom"]]["text"]
            assert start == char_end and bs == byte_end and text == unit["text"][start:end]
            assert be-bs == len(text.encode()) and hashlib.sha256(text.encode()).hexdigest() == span["text_sha256"]
            assert end == len(unit["text"]) or unit["text"][end-1] in "\n\r\v\f\x1c\x1d\x1e\x85\u2028\u2029"
            char_end, byte_end = end, be
            pieces.append(text)
        assert "".join(pieces) == unit["text"] and byte_end == len(unit["text"].encode())
    by_task = {e["task"]: e for e in episodes}
    refs = {(r["task"], r["turn"]): r for r in references}
    best, observed, n = {}, set(), 0
    for shard in selection["source"]["shards"]:
        path = raw_root/shard["path"]
        with path.open("rb") as f:
            assert path.stat().st_size == shard["bytes"]
            assert hashlib.file_digest(f, "sha256").hexdigest() == shard["sha256"]
        parquet = pq.ParquetFile(path)
        identifiers = parquet.read(columns=["instance_id"]).column(0).to_pylist()
        assert len(identifiers) == shard["rows"]
        for row, task in enumerate(identifiers):
            rank = digest(["global-one-attempt-per-task-v1", "mnlm-global-task-units-20261017",
                           "attempt", task, shard["sha256"], row])
            candidate = (rank, shard["sha256"], row)
            if task not in best or candidate < best[task][0]:
                best[task] = candidate, dict(task=task, shard=shard["path"], row=row)
        wanted = {e["row"]: e for e in episodes if e["shard"] == shard["path"]}
        offset = 0
        for batch in parquet.iter_batches(batch_size=128, columns=["instance_id", "trajectory"]):
            chosen = sorted(row for row in wanted if offset <= row < offset+batch.num_rows)
            if chosen:
                cut = batch.take(pa.array([row-offset for row in chosen]))
                prefixes = pc.list_slice(cut.column("trajectory"), 0, 16).to_pylist()
                for row, task, turns in zip(chosen, cut.column("instance_id").to_pylist(), prefixes):
                    ep = wanted[row]
                    assert task == ep["task"]
                    assert ep["clock"] == "recorded_turn_order_only"
                    assert ep["occurred_at"] is ep["place"] is ep["why"] is None
                    assert ep["independently_executed"] is False
                    if len(turns) < 16:
                        assert ep["status"] == "INSUFFICIENT_RECORDED_PREFIX"
                        assert not any(t == task for t, _ in refs)
                        continue
                    assert ep["status"] == "ready"
                    for turn in range(1, 16):
                        ref = refs[task, turn]
                        expected_role = "user" if turn % 2 else "ai"
                        assert turns[turn]["role"] == expected_role
                        text = turns[turn]["text"]
                        assert units[ref["unit"]]["text"] == text
                        assert ref["text_sha256"] == hashlib.sha256(text.encode()).hexdigest()
                        assert ref["utf8_span"] == [0, len(text.encode())]
                        assert ref["source_sha256"] == shard["sha256"] and ref["row"] == row
                        assert ref["partition"] == ep["partition"] and ref["derived"] == (turn % 2 == 0)
                        assert ref["role"] == ("task_report" if turn == 1 else
                            "generated_proposal" if turn % 2 == 0 else "recorded_response")
                        observed.add(ref["unit"])
                        n += 1
            offset += batch.num_rows
    ordered = sorted((value[1] for value in best.values()),
                     key=lambda e: digest([selection["seed"], e["task"]]))[:64]
    assert ordered == selection["selection"] and len(best) == 918
    for rank, e in enumerate(ordered):
        actual = by_task[e["task"]]
        assert actual["rank"] == rank and actual["partition"] == (
            "train" if rank < 32 else "development" if rank < 48 else "withheld")
    assert observed == set(range(len(units))) and n == len(references) == report["references"]
    assert len(units) == report["units"]
    return dict(episodes=len(episodes), ready=sum(e["status"] == "ready" for e in episodes),
                gaps=sum(e["status"] != "ready" for e in episodes), references=n, units=len(units),
                atoms=len(atoms), reading_spans=len(spans), exact_byte_conservation=True,
                ready_partitions=dict(Counter(e["partition"] for e in episodes if e["status"] == "ready")),
                independent_source_equality=True, selection_outcome_blind=True,
                world_truth_established=False)
