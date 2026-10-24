"""Independent source/interval/member closure for native time views."""
from collections import defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from senses.contract import digest, require, validate_record


def audit_graph(record, graph):
    validate_record(record)
    require(set(graph) == {"schema", "parent", "parent_sha256", "split_group", "clock", "nodes", "edges",
                          "temporal_unavailable", "meaning_asserted", "scientific_admission"}
            and graph["schema"] == "native-time-views-v1" and graph["parent"] == record["atom_id"]
            and graph["parent_sha256"] == digest(record) and graph["split_group"] == record["unit"],
            "MNLM_TIME_READBACK_SOURCE", "time views are not bound to this complete retained observation")
    require(graph["meaning_asserted"] is False and graph["scientific_admission"] is False,
            "MNLM_TIME_READBACK_CLAIM", "source intervals are not admitted constellations")
    ch, loc = record["channels"], record["locator"]
    structure = {v["name"]: v["value"] for v in ch["structure"]["items"]}
    wanted, expected_edges = [], []
    if structure.get("kind") == "recorded_audio":
        require(ch["surface"]["state"] == "absent", "MNLM_TIME_READBACK_ALIGNMENT", "no timed transcript in this source")
        items = ch["perception"]["items"]
        rates, hashes, cursors = set(), set(), {}
        frames = []
        for i, item in enumerate(items):
            f = {v["name"]: v for v in item["measurements"]}
            names = ("sample_start", "sample_count", "sample_rate", "audio_channel")
            require(len(f) == len(item["measurements"]) and item["modality"] == "audio"
                    and all(n in f and f[n]["type"] == "integer" and type(f[n]["value"]) is int for n in names)
                    and f["sample_start"]["unit"] == f["sample_count"]["unit"] == "sample"
                    and f["sample_rate"]["unit"] == "Hz", "MNLM_TIME_READBACK_COORDINATE", "invalid native sample fields")
            a, length, rate, channel = (f[n]["value"] for n in names)
            b = a + length
            require(rate > 0 and channel >= 0 and length > 0 and a == cursors.get(channel, loc["span"][0])
                    and b <= loc["span"][1], "MNLM_TIME_READBACK_COORDINATE", "missing or overlapping native interval")
            rates.add(rate); hashes.add(item["sha256"]); cursors[channel] = b
            frames.append((i, a, b, channel))
        require(len(rates) == len(hashes) == 1 and hashes == {loc["record_digest"]}
                and all(v == loc["span"][1] for v in cursors.values()),
                "MNLM_TIME_READBACK_COORDINATE", "recording coverage or rate differs")
        expected_clock = {"id": digest(["recording-sample-clock", loc["record_digest"]]), "unit": "sample",
                          "seconds_per_tick": [1, rates.pop()], "origin": "recording_start", "cross_source_synchronization": False}
        wanted.append(("recording_context", loc["span"], [("perception", i, "native_audio") for i in range(len(items))]))
        wanted.extend(("audio_frame", [a, b], [("perception", i, "native_audio")]) for i, a, b, _ in frames)
        last = {}
        for i, _, _, channel in frames:
            expected_edges.append((0, i + 1, "contains"))
            if channel in last:
                expected_edges.append((last[channel], i + 1, "observed_next"))
            last[channel] = i + 1
    elif structure.get("kind") == "observation" and structure.get("observation_type") == "recorded_robot_step":
        prov = {v["name"]: v["value"] for v in ch["provenance"]["items"]}
        require(loc["source"] == "bridge-v2-rlds" and structure.get("span_unit") == "step"
                and loc["span"] == [loc["row"], loc["row"] + 1] and "episode" in prov
                and ch["history"]["state"] == "absent"
                and all(p["modality"] == "image" for p in ch["perception"]["items"]),
                "MNLM_TIME_READBACK_COORDINATE", "invalid ordinal source observation")
        expected_clock = {"id": digest(["ordinal-episode-clock", prov["episode"]]), "unit": "step",
                          "seconds_per_tick": None, "origin": "episode_order_only", "cross_source_synchronization": False}
        roles = [("perception", "native_image"), ("surface", "task_instruction_not_transcript"),
                 ("fields", "recorded_state_uncommissioned_units"), ("behaviour", "recorded_command_uncommissioned_alignment")]
        wanted.append(("recorded_step", loc["span"], [(c, i, role) for c, role in roles for i in range(len(ch[c]["items"]))]))
    else:
        require(graph["clock"] is None and not graph["nodes"] and not graph["edges"]
                and graph["temporal_unavailable"] == "source_has_no_attested_native_time_or_step_coordinates",
                "MNLM_TIME_READBACK_ALIGNMENT", "untimed correspondence cannot become temporal alignment")
        return {"nodes": 0, "edges": 0, "untimed_parents": 1}
    require(graph["clock"] == expected_clock and graph["temporal_unavailable"] is None,
            "MNLM_TIME_READBACK_CLOCK", "invented timebase, cross-source mapping or clock identity")
    require(len(graph["nodes"]) == len(wanted), "MNLM_TIME_READBACK_COVERAGE", "missing or extra native view")
    for n, (level, span, references) in zip(graph["nodes"], wanted):
        members = [{"channel": c, "index": i, "role": role} for c, i, role in references]
        body = {"level": level, "span": span, "available_after": span[1], "members": members}
        media = sorted({"text" if c == "surface" else ch[c]["items"][i]["modality"]
                        for c, i, _ in references if c in ("surface", "perception")})
        expected = {"id": digest(["native-time-views-v1", record["atom_id"], body]), **body, "required_native_media": media}
        require(n == expected, "MNLM_TIME_READBACK_MEMBERS", "view changed source members, role, cutoff or native requirements")
    nodes = graph["nodes"]
    require(graph["edges"] == [{"origin": nodes[a]["id"], "target": nodes[b]["id"], "kind": k}
                               for a, b, k in expected_edges], "MNLM_TIME_READBACK_EDGES", "missing or invented source relation")
    return {"nodes": len(nodes), "edges": len(expected_edges), "untimed_parents": 0}
