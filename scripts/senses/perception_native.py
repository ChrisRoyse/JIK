"""Source-bound native media prefixes and cutoff-limited consumption."""
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import numpy as np

from senses.perception_episodes import select, identity
from text.evaluation_multiscale import file_sha


SCHEMA = "perception-native-prefix-v1"
FLAGS = dict(meaning_asserted=False, scientific_admission=False, training_ready=False,
             world_facts_established=False, physical_sync_calibrated=False,
             transcript_available=False, sealed_evaluation=False,
             decoded_media_cutoff_enforced=True, online_causal_decode=False)


def check(ok, code):
    if not ok:
        raise ValueError("MNLM_NATIVE_PREFIX_"+code)


def fraction(value):
    check(isinstance(value, list) and len(value) == 2
          and all(type(v) is int for v in value) and value[1] > 0
          and value[0] >= 0, "CUTOFF")
    return Fraction(*value)


def source(root, report_pin, clip):
    root = Path(root)
    raw = (root/"report.json").read_bytes()
    check(hashlib.sha256(raw).hexdigest() == report_pin, "EPISODE_REPORT")
    report = json.loads(raw)
    check(report["complete"] and not report["errors"], "EPISODE_REPORT")
    check(clip in {e["clip"] for e in report["episodes"]}, "CLIP")
    names = [f"public/{clip}.json", f"probes/{clip}.json", f"media/{clip}.mp4"]
    for name in names:
        path = root/name
        check(path.is_file() and path.stat().st_size == report["files"][name]["bytes"]
              and file_sha(path) == report["files"][name]["sha256"], "SOURCE_BYTES")
    graph = json.loads((root/names[0]).read_bytes())
    probe = json.loads((root/names[1]).read_bytes())
    check(graph["blob_sha256"] == report["files"][names[2]]["sha256"]
          and graph["clip"] == clip and not graph["flags"]["meaning_asserted"], "SOURCE_IDENTITY")
    streams = {s["codec_type"]: s for s in probe["streams"]}
    check(set(streams) == {"video", "audio"}, "SOURCE_MEDIA")
    return graph, streams, dict(report=report_pin, graph=report["files"][names[0]]["sha256"],
        probe=report["files"][names[1]]["sha256"], media=graph["blob_sha256"])


def layout(graph, streams, through, media=("image", "audio")):
    check(isinstance(media, (list, tuple)) and media and len(set(media)) == len(media)
          and set(media) <= {"image", "audio"}, "MISSING_MODALITY")
    selected = select(graph, clock_id=graph["clock"]["id"], split_group=graph["split_group"],
                      through=through, required_media=list(media))
    result = {}
    for medium, kind in (("image", "video"), ("audio", "audio")):
        if medium not in media:
            continue
        stream = streams[kind]
        nodes = [n for n in selected["nodes"] if n["modality"] == medium]
        check([n["ordinal"] for n in nodes] == list(range(len(nodes))), "PREFIX_ORDER")
        check(nodes and all(n["stream"] == stream["index"] for n in nodes), "STREAM")
        check(all(fraction(n["available_after"]) <= fraction(through) for n in nodes), "FUTURE")
        if medium == "image":
            shape = [stream["height"], stream["width"], 3]
            sizes = [int(np.prod(shape))]*len(nodes)
            spec = dict(dtype="|u1", shape=shape, pixel_format="rgb24")
        else:
            channels, rate = stream["channels"], int(stream["sample_rate"])
            check(all(n["sample_count"] > 0 and fraction(n["span"][1])-fraction(n["span"][0])
                      == Fraction(n["sample_count"], rate) for n in nodes), "AUDIO_SPAN")
            sizes = [n["sample_count"]*channels*4 for n in nodes]
            spec = dict(dtype="<f4", channels=channels, sample_rate=rate, sample_format="flt")
        result[medium] = dict(nodes=nodes, sizes=sizes, spec=spec, stream=stream["index"])
    return result


class NativePrefix:
    """Only eligible, individually hash-checked native payloads reach the consumer."""
    def __init__(self, root, *, expected, episodes, episode_pin, media=("image", "audio")):
        self.root = Path(root)
        raw = (self.root/"manifest.json").read_bytes()
        check(hashlib.sha256(raw).hexdigest() == expected
              and (self.root/"COMMITTED").read_text() == expected, "MANIFEST_PIN")
        value = json.loads(raw)
        check(set(value) == {"schema", "clip", "source", "clock", "split_group", "provider_split",
              "through", "runtime", "records", "formats", "flags", "commands", "logs"},
              "MANIFEST_SCHEMA")
        check(value["schema"] == SCHEMA and value["flags"] == FLAGS, "FLAGS")
        graph, streams, bound = source(episodes, episode_pin, value["clip"])
        check(value["source"] == bound and value["clock"] == graph["clock"]
              and value["split_group"] == graph["split_group"]
              and value["provider_split"] == graph["provider_split"], "SOURCE_IDENTITY")
        declared = layout(graph, streams, value["through"], media)
        check(value["formats"] == {m: d["spec"] for m, d in declared.items()}, "FORMAT")
        check(set(value["records"]) == set(declared), "RECORD_ROSTER")
        for medium, d in declared.items():
            records = value["records"][medium]
            check(len(records) == len(d["nodes"]), "RECORD_ROSTER")
            for record, node, size in zip(records, d["nodes"], d["sizes"]):
                check(set(record) == {"node", "file", "bytes", "sha256"}
                      and record["node"] == node and record["bytes"] == size
                      and record["file"] == f"{medium}/{node['ordinal']:06d}.bin", "NATIVE_COORDINATES")
                path = self.root/record["file"]
                check(path.is_file() and path.stat().st_size == size
                      and file_sha(path) == record["sha256"], "PAYLOAD_BYTES")
        for medium in declared:
            check(file_sha(self.root/f"{medium}.log") == value["logs"][medium], "LOG_BYTES")
        self.value = value

    def nodes(self, *, clock_id, split_group, through, required_media):
        v = self.value
        check(clock_id == v["clock"]["id"], "CLOCK")
        check(split_group == v["split_group"], "SPLIT")
        check(fraction(through) <= fraction(v["through"]), "UNDECODED_FUTURE")
        check(isinstance(required_media, list) and required_media
              and len(set(required_media)) == len(required_media)
              and set(required_media) <= set(v["records"]), "MISSING_MODALITY")
        result = {m: [r for r in v["records"][m]
                      if fraction(r["node"]["available_after"]) <= fraction(through)]
                  for m in required_media}
        check(all(result.values()), "UNAVAILABLE_MODALITY")
        return result

    def consume(self, **request):
        """Yield read-only arrays in original source presentation order; never resample."""
        chosen = self.nodes(**request)
        records = sorted(((m, r) for m, rows in chosen.items() for r in rows),
                         key=lambda mr: (fraction(mr[1]["node"]["span"][0]),
                                         mr[1]["node"]["stream"], mr[1]["node"]["ordinal"]))
        for medium, record in records:
            raw = (self.root/record["file"]).read_bytes()
            check(len(raw) == record["bytes"]
                  and hashlib.sha256(raw).hexdigest() == record["sha256"], "CONSUMPTION_BYTES")
            spec = self.value["formats"][medium]
            shape = spec["shape"] if medium == "image" else [
                record["node"]["sample_count"], spec["channels"]]
            values = np.frombuffer(raw, dtype=spec["dtype"]).reshape(shape)
            check(bool(np.isfinite(values).all()), "NONFINITE_MEDIA")
            yield dict(modality=medium, node=record["node"], values=values,
                       source=self.value["source"], clock=self.value["clock"],
                       split_group=self.value["split_group"], through=request["through"],
                       interpretation="decoded_source_recording", meaning_asserted=False)
