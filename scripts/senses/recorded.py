"""Format adapters for approved recorded tool interactions and audio.

These consume source bytes, never execute commands. Source provenance is distinct
from verification of a physical action. Labels remain outside the sense record.
"""
import io
import json
from bisect import bisect_left
from pathlib import Path

import blake3
import pyarrow.parquet as pq
import zstandard

from .adapters import exact_ref, relation
from .contract import CHANNELS, VERSION, absent, digest, field, present, require, seal_record
from .pipeline import file_hash


def base(source, shard, row, span, record_digest, binding, unit):
    return {"schema": VERSION, "locator": {"source": source, "shard": shard, "row": row, "span": span,
            "record_digest": record_digest, "manifest_digest": digest(binding)}, "unit": unit,
            "referents": [], "channels": {c: absent("not_supplied_by_source") for c in CHANNELS}}


def selected_rows(parquet, indices):
    """Select Arrow rows before materializing records; never parse outcome columns."""
    offset = 0
    for batch in parquet.iter_batches(batch_size=128, columns=["instance_id", "model_name", "trajectory"]):
        end = offset + batch.num_rows
        selected = indices[bisect_left(indices, offset):bisect_left(indices, end)]
        if selected:
            for index, row in zip(selected, batch.take([i - offset for i in selected]).to_pylist()):
                yield index, row
        offset = end
        if end > indices[-1]:
            break


def trajectory_records(path, sha256, binding, limit=128, selection=None):
    require(type(limit) is int and 0 < limit <= 128, "MNLM_TRAJECTORY_SCOPE", "approved inspection is at most 128 complete trajectories")
    require(file_hash(path) == sha256, "MNLM_UCR_INPUT_DIGEST", str(path))
    parquet = pq.ParquetFile(path)
    # Sampling is publication scope, not a property of the source observation.
    # Overlapping passes must reproduce the same identity for the same exchange.
    source_binding = {k: v for k, v in binding.items() if k not in ("trajectory_scope", "trajectory_scope_sha256")}
    if selection is None:
        count = min(limit, parquet.metadata.num_rows)
        require(count == limit, "MNLM_TRAJECTORY_SCOPE", "not enough complete source trajectories")
        expected = dict.fromkeys(range(count))
    else:
        from .trajectory_scope import metadata_plan, validate
        expected = validate(selection, sha256, parquet.metadata.num_rows, limit)
        require(selection == metadata_plan(parquet, sha256, len(expected)), "MNLM_TRAJECTORY_SCOPE",
                "selection differs from the deterministic outcome-blind metadata census")
    for index, raw in selected_rows(parquet, list(expected)):
        require(expected[index] is None or raw["instance_id"] == expected[index],
                "MNLM_TRAJECTORY_SCOPE", "selected row does not carry its declared task identity")
        yield from exchange_records(raw, path, index, sha256, source_binding)


def exchange_records(raw, path, index, sha256, source_binding):
    """Pure exchange adapter; callers bind the raw bytes and selection once."""
    turns = raw["trajectory"]
    require(len(turns) >= 3 and turns[0]["role"] == "system" and turns[1]["role"] == "user",
            "MNLM_TRAJECTORY_FORMAT", "unexpected task preamble")
    require(all(t["role"] == ("ai" if i % 2 == 0 else "user") for i, t in enumerate(turns[2:], 2)),
            "MNLM_TRAJECTORY_FORMAT", "ambiguous action/response order; do not infer tool returns")
    require(isinstance(raw["instance_id"], str) and raw["instance_id"], "MNLM_TRAJECTORY_FORMAT", "instance identity required")
    unit = digest(["swe-task", raw["instance_id"]])  # all attempts at one task are siblings
    for step in range(2, len(turns), 2):
        action = turns[step]["text"]
        response = turns[step + 1]["text"] if step + 1 < len(turns) else None
        require(isinstance(action, str) and action and (response is None or isinstance(response, str)),
                "MNLM_TRAJECTORY_FORMAT", "action/response text required")
        observed = {"action": turns[step], "response": turns[step + 1] if response is not None else None,
                    "instance_id": raw["instance_id"], "model_name": raw["model_name"]}
        r = base("swe-agent-trajectories", Path(path).name + f"#trajectory={index}", step, [0, len(action)],
                 digest(observed), source_binding, unit)
        r["referents"] = [exact_ref("swe-task:" + raw["instance_id"], "instance_id")]
        ch = r["channels"]
        ch["surface"] = present([{"text": action}])
        ch["fields"] = present([field("turn_index", step, "integer", "sequence_index"),
                                 field("response_recorded", response is not None, "boolean")])
        ch["structure"] = present([field("kind", "tool_exchange", "category"),
                                    field("action_role", "generated_agent_action", "category"),
                                    field("response_role", "provider_recorded_environment_response", "category")])
        ch["provenance"] = present([field("source", "swe-agent-trajectories", "reference"),
                                     field("model", raw["model_name"], "category"),
                                     field("raw_sha256", sha256, "reference"),
                                     field("evidence_level", "provider_recorded_not_independently_reexecuted", "category"),
                                     field("licence", "CC-BY-4.0 plus upstream repository and Llama conditions", "category")])
        ch["relations"] = present([relation("exchange_in_task", "swe-task:" + raw["instance_id"], "instance_id")])
        if response is not None:
            ch["behaviour"] = present([{"event": "tool_response", "mode": "observed",
                "evidence": {"raw_sha256": sha256, "parquet_row": index, "turn": step + 1,
                             "attestation": "source_recorded; not independently reexecuted"},
                "measurements": [field("tool_response", response, "text"),
                                 field("response_characters", len(response), "integer", "character")]}])
        else:
            ch["behaviour"] = absent("no_recorded_response_after_final_action")
        ch["history"] = absent("step_order_retained; observation_timestamps_not_supplied")
        yield seal_record(r)


def web_chapters(canon, manifest_path):
    rows = [json.loads(line) for line in Path(manifest_path).read_text(encoding="utf-8").splitlines()]
    accepted = [r for r in rows if r.get("shard") == "eng-web" and not r.get("error")]
    require(bool(accepted), "MNLM_AUDIO_TEXT", "no accepted eng-web canonical manifest")
    manifest = accepted[-1]
    raw = Path(canon).read_bytes()  # this explicitly bounded retained shard is 2,999,471 bytes
    require(len(raw) == manifest["out_bytes"] and blake3.blake3(raw).hexdigest() == manifest["out_blake3"],
            "MNLM_AUDIO_TEXT", "retained chapter corpus differs from manifest")
    chapters = {1: [], 2: []}
    with zstandard.ZstdDecompressor().stream_reader(io.BytesIO(raw)) as zr:
        for line in io.TextIOWrapper(zr, encoding="utf-8"):
            r = json.loads(line)
            f = r["f"]
            if f.get("book") == "PSA" and f.get("chapter") in chapters and f.get("role") == "verse":
                require(r["b3"] == blake3.blake3(r["text"].encode()).hexdigest()[:32], "MNLM_AUDIO_TEXT", "verse text digest changed")
                chapters[f["chapter"]].append(r)
    for chapter, expected in ((1, 6), (2, 12)):
        require([r["f"]["verse"] for r in chapters[chapter]] == [str(n) for n in range(1, expected + 1)],
                "MNLM_AUDIO_TEXT", "chapter correspondence is incomplete or duplicated")
    return chapters, manifest


def audio_records(files, chapters, text_manifest, binding):
    from .acoustics import frames
    for item in files:
        path, sha, chapter = Path(item["stored"]), item["sha256"], item["chapter"]
        require(path.name == f"Psalm{chapter:03d}.ogg" and chapter in chapters,
                "MNLM_AUDIO_ALIGNMENT", "no approved chapter correspondence")
        require(file_hash(path) == sha, "MNLM_UCR_INPUT_DIGEST", str(path))
        percepts = list(frames(path, sha))
        total = max(next(f["value"] for f in p["measurements"] if f["name"] == "sample_start") +
                    next(f["value"] for f in p["measurements"] if f["name"] == "sample_count") for p in percepts)
        key = f"web-chapter:PSA.{chapter}"
        unit = digest(["web-chapter", chapter])
        r = base("web-audio-mpj", path.name, 0, [0, total], sha, binding, unit)
        r["referents"] = [exact_ref(key, "source filename chapter correspondence; no word alignment", "attested_mapping")]
        ch = r["channels"]
        ch["perception"] = present(percepts)
        ch["structure"] = present([field("kind", "recorded_audio", "category"),
                                    field("span_unit", "sample", "category"), field("samples", total, "integer", "sample")])
        ch["fields"] = present([field("book", "PSA", "reference"), field("chapter", chapter, "integer")])
        ch["provenance"] = present([field("licence", "public_domain", "category"),
                                     field("reciter", "Michael Paul Johnson", "category"),
                                     field("raw_sha256", sha, "reference"), field("source", "web-audio-mpj", "reference")])
        ch["relations"] = present([relation("chapter_correspondence_not_transcript_alignment", key, "source filename")])
        yield seal_record(r)
        # Text is a separate observation, never an audio encoder's input or a forced transcript.
        text = "\n".join(v["text"] for v in chapters[chapter])
        r = base("ebible-english", f"eng-web#PSA.{chapter}", chapters[chapter][0]["row"], [0, len(text)],
                 digest(chapters[chapter]), text_manifest, unit)
        r["referents"] = [exact_ref(key, "explicit book/chapter fields", "attested_mapping")]
        r["channels"]["surface"] = present([{"text": text}])
        r["channels"]["structure"] = present([field("kind", "chapter_text", "category")])
        r["channels"]["provenance"] = present([field("licence", "public_domain", "category"),
                                                field("edition", "eng-web", "reference")])
        yield seal_record(r)
