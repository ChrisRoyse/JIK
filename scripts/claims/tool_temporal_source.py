"""Outcome-blind, exact recorded-tool prefixes with explicit source-only grounding."""
from contextlib import ExitStack
import json
from pathlib import Path
import time

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from claims.codata_source import check, sha
from claims.episodic_memory import canonical, digest
from senses.trajectory_population import open_bound, select_metadata

REVISION = "68195a1450865274106246d0d0296a1d6807b88e"
SHARDS = (
    (89182938, "5a395e8c7bb8ddc4b8f4d268506b3a0e2cf9b5ec3922600117322fe788067a13"),
    (94796688, "fca106cce0f09891c2fadc032fb304da9ae5a7c31d2a39eb7ec70a7bdd4a9882"),
    (91137587, "fc28c2ab014c6c90d72026dda9cf8753e37b4b7b128c05c4232980cdcc99f3f7"))
SEED = "mnlm-foundation-temporal-v1"
CUTOFFS = (6, 10, 14)
FILES = ("tool/chunks/atoms.parquet", "tool/summary.json", "references.jsonl",
         "episodes.jsonl", "selection.json", "messages.jsonl", "reading_spans.jsonl")


def scope():
    return dict(schema="trajectory-population-v1", source="swe-agent-trajectories",
        revision=REVISION, approval="MNLM #3 retained source; #28 temporal learning increment",
        distinct_tasks=918, shards=[dict(path=f"data/train-{i:05}-of-00012.parquet",
        bytes=n, sha256=pin, rows=6670) for i, (n, pin) in enumerate(SHARDS)])


def write(path, value):
    with path.open("xb") as f:
        f.write(canonical(value))


class ToolSource:
    def __init__(self, root):
        start = time.perf_counter()
        self.scope = scope()
        self.pin = digest(self.scope)
        self.episodes, self.texts = [], {}
        with ExitStack() as stack:
            opened = {s["path"]: pq.ParquetFile(stack.enter_context(open_bound(root, s)))
                      for s in self.scope["shards"]}
            hashed = time.perf_counter()
            def batches(shard):
                p = opened[shard["path"]]
                check(p.metadata.num_rows == shard["rows"], "TOOL_RAW_ROWS")
                for batch in p.iter_batches(batch_size=1024, columns=["instance_id"]):
                    yield batch.column(0).to_pylist()
            population = select_metadata(self.scope, batches)
            self.selection = sorted(population["selection"],
                key=lambda x: digest([SEED, x["task"]]))[:64]
            ranks = {x["task"]: i for i, x in enumerate(self.selection)}
            selected_at = time.perf_counter()
            for shard in self.scope["shards"]:
                chosen = {x["row"]: x["task"] for x in self.selection if x["shard"] == shard["path"]}
                offset = 0
                for batch in opened[shard["path"]].iter_batches(batch_size=128,
                        columns=["instance_id", "model_name", "trajectory"]):
                    indices = sorted(i for i in chosen if offset <= i < offset+batch.num_rows)
                    if indices:
                        rows = batch.take(pa.array([i-offset for i in indices]))
                        trajectories = rows.column("trajectory")
                        prefixes = pc.list_slice(trajectories, 0, 16).to_pylist()
                        for row, task, provider, turns in zip(indices, rows.column("instance_id").to_pylist(),
                                rows.column("model_name").to_pylist(), prefixes):
                            check(task == chosen[row], "TOOL_SOURCE_ROW")
                            rank = ranks[task]
                            partition = "train" if rank < 32 else "development" if rank < 48 else "withheld"
                            ep = dict(task=task, shard=shard["path"], row=row, rank=rank,
                                partition=partition, provider=provider, source_sha256=shard["sha256"],
                                revision=REVISION, clock="recorded_turn_order_only",
                                occurred_at=None, place=None, why=None, independently_executed=False)
                            if len(turns) != 16:
                                ep.update(status="INSUFFICIENT_RECORDED_PREFIX", available_prefix_turns=len(turns))
                            elif (turns[0]["role"] != "system" or turns[1]["role"] != "user"
                                  or not all(t["role"] == ("ai" if i % 2 == 0 else "user")
                                  for i, t in enumerate(turns[2:], 2))):
                                ep.update(status="AMBIGUOUS_TURN_ROLES")
                            elif not all(isinstance(t["text"], str) and t["text"] for t in turns[1:]):
                                ep.update(status="EMPTY_SOURCE_TEXT")
                            else:
                                ep.update(status="ready")
                                for turn, value in enumerate(turns[1:], 1):
                                    self.texts[(task, turn)] = value["text"]
                            self.episodes.append(ep)
                    offset += batch.num_rows
                    if chosen and offset > max(chosen):
                        break
        self.episodes.sort(key=lambda e: e["rank"])
        check(len(self.episodes) == 64 and len({e["task"] for e in self.episodes}) == 64, "TOOL_SELECTION")
        self.by_task = {e["task"]: e for e in self.episodes}
        self.timings = dict(hash_open=hashed-start, selection=selected_at-hashed,
                            prefix_read=time.perf_counter()-selected_at)

    def reference(self, task, turn):
        check((task, turn) in self.texts, "TOOL_ADDRESS")
        ep, text = self.by_task[task], self.texts[task, turn]
        return dict(task=task, turn=turn, shard=ep["shard"], row=ep["row"],
            source_sha256=ep["source_sha256"], revision=REVISION, field="trajectory.text",
            utf8_span=[0, len(text.encode())], text_sha256=sha(text.encode()),
            role="task_report" if turn == 1 else "generated_proposal" if turn % 2 == 0 else "recorded_response",
            partition=ep["partition"], derived=(turn % 2 == 0), occurred_at=None,
            clock="recorded_turn_order_only", grounding_scope="pinned_artifact_contents_only")

    def receipt(self, task, turn):
        ref = self.reference(task, turn)
        check(ref["role"] == "recorded_response", "TOOL_ECHO_ROLE")
        body = dict(checker="pinned-tool-response-v1", source=ref,
                    scope="pinned_artifact_contents_only", independently_executed=False)
        return dict(id=digest(body), **body)


def expected(source):
    units, refs, seen = [], [], {}
    for ep in source.episodes:
        if ep["status"] != "ready":
            continue
        for turn in range(1, 16):
            text = source.texts[ep["task"], turn]
            pin = sha(text.encode())
            if pin not in seen:
                seen[pin] = len(units)
                units.append(dict(row=len(units), chunk=0, entity=pin, text=text, start=0, end=len(text)))
            check(units[seen[pin]]["text"] == text, "TOOL_HASH_COLLISION")
            refs.append(dict(unit=seen[pin], **source.reference(ep["task"], turn)))
    return units, refs


def reading_atoms(messages):
    """Lossless source-line blocks; a single long line remains intact."""
    atoms, spans, seen = [], [], {}
    for message, row in enumerate(messages):
        text = row["text"]
        start, end, byte_start = 0, 0, 0
        ranges = []
        for line in text.splitlines(keepends=True):
            if end > start and end+len(line)-start > 1200:
                ranges.append((start, end))
                start = end
            end += len(line)
        if end > start:
            ranges.append((start, end))
        check(end == len(text) and ranges, "TOOL_SEGMENT_COVERAGE")
        for start, end in ranges:
            segment = text[start:end]
            pin = sha(segment.encode())
            if pin not in seen:
                seen[pin] = len(atoms)
                atoms.append(dict(row=len(atoms), chunk=0, entity=pin, text=segment,
                                  start=0, end=len(segment)))
            check(atoms[seen[pin]]["text"] == segment, "TOOL_HASH_COLLISION")
            byte_end = byte_start+len(segment.encode())
            spans.append(dict(unit=message, atom=seen[pin], char_span=[start, end],
                              utf8_span=[byte_start, byte_end], text_sha256=pin))
            byte_start = byte_end
    return atoms, spans


def build(source, output):
    output = Path(output)
    check(not output.exists(), "TOOL_POPULATION_EXISTS")
    (output/"tool/chunks").mkdir(parents=True)
    start = time.perf_counter()
    units, refs = expected(source)
    atoms, spans = reading_atoms(units)
    pq.write_table(pa.Table.from_pylist(atoms), output/FILES[0])
    write(output/FILES[1], dict(chunks=len(atoms), chunks_sha256=sha((output/FILES[0]).read_bytes())))
    for name, records in (("references.jsonl", refs), ("episodes.jsonl", source.episodes),
                          ("messages.jsonl", units), ("reading_spans.jsonl", spans)):
        with (output/name).open("xb") as f:
            for r in records:
                f.write(canonical(r)+b"\n")
    write(output/"selection.json", dict(source=source.scope, seed=SEED, selection=source.selection))
    report = dict(schema="tool-temporal-population-v2", source_pin=source.pin, complete=True, errors=[],
        calibration_only=True, purpose="exposed_temporal_development_not_admission",
        families=dict(tool=dict(ready=True, summary_sha256=sha((output/FILES[1]).read_bytes()))),
        units=len(units), references=len(refs), episodes=len(source.episodes),
        atoms=len(atoms), reading_spans=len(spans), segmentation="complete_source_lines_1200_chars_v1",
        gaps=[e for e in source.episodes if e["status"] != "ready"],
        unique_utf8_bytes=sum(len(u["text"].encode()) for u in units), cutoffs=list(CUTOFFS),
        scientific_admission=False, world_truth_established=False,
        phases_seconds={**source.timings, "publish": time.perf_counter()-start},
        files={n: dict(sha256=sha((output/n).read_bytes()), bytes=(output/n).stat().st_size) for n in FILES})
    write(output/"report.json", report)
    pin = sha((output/"report.json").read_bytes())
    (output/"COMMITTED").write_text(pin, encoding="ascii")
    return pin


class TemporalPopulation:
    def __init__(self, root, source, expected_pin):
        root = Path(root)
        raw = (root/"report.json").read_bytes()
        report = json.loads(raw)
        check(sha(raw) == expected_pin == (root/"COMMITTED").read_text()
              and report["source_pin"] == source.pin and report["complete"]
              and report["scientific_admission"] is False
              and report["schema"] == "tool-temporal-population-v2", "TOOL_POPULATION_PIN")
        check(set(report["files"]) == set(FILES) and report["cutoffs"] == list(CUTOFFS), "TOOL_POPULATION_SCHEMA")
        for name, entry in report["files"].items():
            raw = (root/name).read_bytes()
            check(len(raw) == entry["bytes"] and sha(raw) == entry["sha256"], "TOOL_POPULATION_BYTES")
        units, refs = expected(source)
        atoms, spans = reading_atoms(units)
        check(pq.read_table(root/FILES[0]).to_pylist() == atoms
              and list(map(json.loads, (root/"messages.jsonl").read_bytes().splitlines())) == units
              and list(map(json.loads, (root/"reading_spans.jsonl").read_bytes().splitlines())) == spans
              and list(map(json.loads, (root/"references.jsonl").read_bytes().splitlines())) == refs,
              "TOOL_POPULATION_SOURCE")
        check(list(map(json.loads, (root/"episodes.jsonl").read_bytes().splitlines())) == source.episodes,
              "TOOL_POPULATION_EPISODES")
        check(json.loads((root/"selection.json").read_bytes()) ==
              dict(source=source.scope, seed=SEED, selection=source.selection), "TOOL_POPULATION_SELECTION")
        self.source, self.units, self.refs, self.pin = source, units, refs, expected_pin
        self.atoms, self.spans = atoms, spans
        self.lookup = {(r["task"], r["turn"]): r for r in refs}

    def prefix(self, task, cutoff, *, partition, observed_through):
        check(task in self.source.by_task and self.source.by_task[task]["status"] == "ready", "TOOL_TASK_UNAVAILABLE")
        check(partition == self.source.by_task[task]["partition"], "TOOL_SPLIT")
        check(type(cutoff) is int and cutoff in CUTOFFS and observed_through == cutoff, "TOOL_CUTOFF")
        return tuple(self.lookup[task, t] for t in range(1, cutoff+1))

    def echo(self, task, cutoff, *, receipt):
        check(type(cutoff) is int and cutoff in CUTOFFS, "TOOL_CUTOFF")
        check(receipt == self.source.receipt(task, cutoff+1), "TOOL_ECHO_RECEIPT")
        return self.lookup[task, cutoff+1]

    def require_wall_clock(self):
        check(False, "TOOL_OCCURRENCE_CLOCK_UNAVAILABLE")

    def initial_bank(self):
        # Only first two already-recorded training responses; aliases remain distinct.
        return tuple(r for r in self.refs if r["partition"] == "train" and r["turn"] in (3, 5))
