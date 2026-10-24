"""Source-bound native intervals, not inferred synchronization or admitted meaning.

Linear in retained members. Audio sample clocks and ordinal robot clocks cannot
be joined without an additional source-attested mapping. No encoder runs here.
"""
from collections import defaultdict
import copy
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from senses.contract import canonical, digest, require, validate_record

SCHEMA = "native-time-views-v1"


def fields(items):
    result = {f["name"]: f for f in items}
    require(len(result) == len(items), "MNLM_TIME_FIELDS", "duplicate coordinate names")
    return result


def audio_coordinate(item):
    f = fields(item["measurements"])
    names = ("sample_start", "sample_count", "sample_rate", "audio_channel")
    require(item["modality"] == "audio" and all(n in f and f[n]["type"] == "integer"
            and type(f[n]["value"]) is int for n in names), "MNLM_TIME_AUDIO", "native sample coordinates required")
    start, count, rate, channel = (f[n]["value"] for n in names)
    require(start >= 0 and count > 0 and rate > 0 and channel >= 0
            and f["sample_start"]["unit"] == f["sample_count"]["unit"] == "sample"
            and f["sample_rate"]["unit"] == "Hz", "MNLM_TIME_AUDIO", "invalid sample geometry")
    return start, start + count, rate, channel


def member(channel, index, role):
    return {"channel": channel, "index": index, "role": role}


def native_media(record, members):
    media = set()
    for m in members:
        if m["channel"] == "surface":
            media.add("text")
        elif m["channel"] == "perception":
            media.add(record["channels"]["perception"]["items"][m["index"]]["modality"])
    return sorted(media)


def node(record, level, span, members):
    body = {"level": level, "span": list(span), "available_after": span[1], "members": members}
    return {"id": digest([SCHEMA, record["atom_id"], body]), **body,
            "required_native_media": native_media(record, members)}


def edge(origin, target, kind):
    return {"origin": origin, "target": target, "kind": kind}


def build(record):
    validate_record(record)
    structure = fields(record["channels"]["structure"]["items"])
    kind = structure.get("kind", {}).get("value")
    loc, ch = record["locator"], record["channels"]
    graph = {"schema": SCHEMA, "parent": record["atom_id"], "parent_sha256": digest(record),
             "split_group": record["unit"], "clock": None, "nodes": [], "edges": [],
             "temporal_unavailable": None, "meaning_asserted": False, "scientific_admission": False}
    if kind == "recorded_audio":
        require(ch["surface"]["state"] == "absent" and ch["perception"]["items"],
                "MNLM_TIME_ALIGNMENT", "whole-record text cannot become a timed transcript")
        coordinates = [audio_coordinate(p) for p in ch["perception"]["items"]]
        rates = {c[2] for c in coordinates}
        blobs = {p["sha256"] for p in ch["perception"]["items"]}
        require(len(rates) == len(blobs) == 1 and next(iter(blobs)) == loc["record_digest"],
                "MNLM_TIME_AUDIO", "one retained recording and sample clock required")
        rate = next(iter(rates))
        graph["clock"] = {"id": digest(["recording-sample-clock", loc["record_digest"]]),
                          "unit": "sample", "seconds_per_tick": [1, rate],
                          "origin": "recording_start", "cross_source_synchronization": False}
        context = node(record, "recording_context", loc["span"],
                       [member("perception", i, "native_audio") for i in range(len(coordinates))])
        graph["nodes"].append(context)
        previous, cursor = {}, defaultdict(lambda: loc["span"][0])
        for i, (start, end, _, channel) in enumerate(coordinates):
            require(start == cursor[channel] and end <= loc["span"][1], "MNLM_TIME_COVERAGE",
                    "missing, overlapping or reordered native audio interval")
            child = node(record, "audio_frame", [start, end], [member("perception", i, "native_audio")])
            graph["nodes"].append(child)
            graph["edges"].append(edge(context["id"], child["id"], "contains"))
            if channel in previous:
                graph["edges"].append(edge(previous[channel], child["id"], "observed_next"))
            previous[channel], cursor[channel] = child["id"], end
        require(all(v == loc["span"][1] for v in cursor.values()), "MNLM_TIME_COVERAGE", "audio tail absent")
    elif kind == "observation" and structure.get("observation_type", {}).get("value") == "recorded_robot_step":
        prov = fields(ch["provenance"]["items"])
        require(loc["source"] == "bridge-v2-rlds" and "episode" in prov
                and structure.get("span_unit", {}).get("value") == "step"
                and loc["span"] == [loc["row"], loc["row"] + 1]
                and ch["history"]["state"] == "absent", "MNLM_TIME_STEP", "source ordinal step required")
        graph["clock"] = {"id": digest(["ordinal-episode-clock", prov["episode"]["value"]]),
                          "unit": "step", "seconds_per_tick": None, "origin": "episode_order_only",
                          "cross_source_synchronization": False}
        roles = {"perception": "native_image", "surface": "task_instruction_not_transcript",
                 "fields": "recorded_state_uncommissioned_units", "behaviour": "recorded_command_uncommissioned_alignment"}
        members = [member(c, i, role) for c, role in roles.items() for i in range(len(ch[c]["items"]))]
        require(all(p["modality"] == "image" for p in ch["perception"]["items"]),
                "MNLM_TIME_STEP", "this source supplies images, not synchronized audio")
        graph["nodes"] = [node(record, "recorded_step", loc["span"], members)]
    else:
        graph["temporal_unavailable"] = "source_has_no_attested_native_time_or_step_coordinates"
    return graph


class BoundViews:
    """Validate closure once, then read selected members without exposing parents."""

    def __init__(self, record, graph):
        from native_time_readback import audit_graph
        audit_graph(record, graph)
        self.record, self.graph = copy.deepcopy(record), copy.deepcopy(graph)
        self.nodes = {n["id"]: n for n in self.graph["nodes"]}

    def read(self, node_id, *, clock_id, through, split_group):
        require(self.graph["clock"] and clock_id == self.graph["clock"]["id"],
                "MNLM_TIME_CLOCK", "cutoff must use this observation's own source clock")
        require(split_group == self.graph["split_group"], "MNLM_TIME_SPLIT", "source group mismatch")
        require(type(through) is int and through >= 0 and node_id in self.nodes,
                "MNLM_TIME_SELECTION", "explicit native cutoff and retained view required")
        n = self.nodes[node_id]
        require(n["available_after"] <= through, "MNLM_TIME_FUTURE", "context includes future native observations")
        members = [{**m, "value": copy.deepcopy(self.record["channels"][m["channel"]]["items"][m["index"]])}
                   for m in n["members"]]
        return {"view": copy.deepcopy(n), "clock": copy.deepcopy(self.graph["clock"]),
                "parent": self.graph["parent"], "split_group": split_group, "members": members,
                "meaning_asserted": False, "scientific_admission": False}
