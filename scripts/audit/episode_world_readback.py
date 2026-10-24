"""Read original episode bytes, time windows, traversals and snapshots independently."""
import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys
import time

STARTED = time.perf_counter()
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from audit.timed_world_readback import reference_hash
from senses.contract import canonical, digest
from senses.perception_corpus import pinned, sha
from senses.timed_speech import save


def check(ok, code):
    if not ok:
        raise ValueError("MNLM_EPISODE_READBACK_"+code)


def main(a):
    r = pinned(a.run/"report.json", a.pin)
    source = pinned(a.inputs/"report.json", r["input_pin"])
    bank = pinned(a.bank/"bank.json", r["bank_pin"])
    records = pinned(a.run/"episode-records.json", r["records"])
    trace = pinned(a.run/"episode-traversals.json", r["traversals"])
    census = pinned(a.run/"census.json", r["census"])
    moments = {m["id"]: m for m in source["moments"]}
    check(set(records) == set(census) == set(moments), "ROSTER")
    catalogue, expected_trace, total_bytes, total_rows, frames = {}, [], 0, 0, Counter()
    for row, lanes in records.items():
        moment = moments[row]
        check(set(lanes) == set(bank["readings"]), "LANES")
        catalogue[row] = {}
        for lane, record in lanes.items():
            entries = [e for e in bank["readings"][lane]
                       if e["input"]["segment"] == moment["source"]["segment"]]
            check(record["row"] == row and record["lane"] == lane and record["source"] == moment["source"]
                  and record["scope"] == "retrospective_complete_utterance"
                  and len(record["pieces"]) == len(entries), "EPISODE")
            cursor, byte_count = 0, 0
            h = hashlib.sha256()
            prior_frame = None
            for entry, timing in zip(entries, record["pieces"], strict=True):
                inp = entry["input"]
                path = a.bank/lane/entry["file"]
                raw = path.read_bytes()
                n, dims = entry["shape"]
                check(len(raw) == n*dims*4 and hashlib.sha256(raw).hexdigest() == entry["sha256"],
                      "ORIGINAL_BYTES")
                check(np.isfinite(np.frombuffer(raw, dtype="<f4")).all(), "FINITE")
                check(timing["source"] == moment["source"] and timing["row"] == row
                      and timing["instrument"] == lane and timing["measurement"] == entry["instrument_id"]
                      and timing["reading_sha256"] == entry["sha256"] and timing["bank"] == r["bank_pin"]
                      and timing["at"] == moment["content"]["at"] and timing["context"] == inp["support"]
                      and timing["through"] == timing["available_after"] == inp["available_after"]
                      and timing["timing"] == moment["content"]["timing"]
                      and timing["physical_sync_calibrated"] is False, "TIME_BINDING")
                if lane.startswith("image_"):
                    frame, tb = inp["frame"], Fraction(*inp["time_base"])
                    check(prior_frame is None or prior_frame < frame["ordinal"], "FRAME_ORDER")
                    prior_frame = frame["ordinal"]
                    expected = [(0, n, frame["pts"]*tb, (frame["pts"]+frame["duration"])*tb)]
                    if lane == "image_sam":
                        frames[inp["segment"]] += 1
                elif lane.startswith("audio_"):
                    audio = inp["source_audio"]
                    start = Fraction(audio["start_sample"], audio["rate"])
                    stop = Fraction(audio["stop_sample"], audio["rate"])
                    expected = [(w["token_start"], w["token_end"],
                        start+Fraction(w["resampled_start"], w.get("target_rate", 16000)),
                        min(stop, start+Fraction(w["resampled_end"], w.get("target_rate", 16000))))
                        for w in entry["native"]]
                else:
                    words, pos = [], 0
                    for word in inp["text"]["words"]:
                        if word["tag"] == "w":
                            end = pos+len(word["text"] or "")
                            words.append((pos, end, word))
                            pos = end+1
                    expected = []
                    for i, w in enumerate(entry["native"]["windows"]):
                        lo, hi = w["source_char_span"]
                        selected = [word for x, y, word in words if x < hi and y > lo]
                        spans = [word["span"] for word in selected]
                        check(timing["windows"][i]["components"] == spans, "WORD_SPANS")
                        expected.append((i, i+1, min(Fraction(*s[0]) for s in spans),
                                         max(Fraction(*s[1]) for s in spans)))
                check(len(expected) == len(timing["windows"]), "WINDOWS")
                for w, (lo, hi, start, stop) in zip(timing["windows"], expected, strict=True):
                    check(w["rows"] == [lo, hi] and [Fraction(*p) for p in w["interval"]] == [start, stop],
                          "TIME_INTERVAL")
                h.update(canonical(dict(file=entry["file"], sha256=entry["sha256"],
                                        rows=[cursor, cursor+n], timing=digest(timing))))
                cursor += n
                byte_count += len(raw)
            check(census[row][lane] == dict(rows=cursor, readings=len(entries), dims=entries[0]["shape"][1]),
                  "CENSUS")
            expected_trace.extend([dict(row=row, lane=lane, count=cursor, readings=len(entries),
                                        bytes=byte_count, traversal=h.hexdigest())]*2)
            catalogue[row][lane] = digest(record)
            total_bytes += byte_count
            total_rows += cursor
    check(Counter(canonical(x) for x in trace) == Counter(canonical(x) for x in expected_trace),
          "TWO_COMPLETE_MODEL_PASSES")
    check(digest(catalogue) == r["temporal_id"] and total_bytes == 2960381952
          and total_rows == 2099618 and sum(frames.values()) == 378, "COMPLETE_SOURCE")
    snapshots = {}
    for name, sequence in (("initial", 0), ("pending", 1)):
        root = a.run/name
        m = pinned(root/"manifest.json", r[name])
        check((root/"COMMITTED").read_text() == r[name] and m["binding"] == r["binding"]
              and m["temporal"] == catalogue and m["temporal_id"] == r["temporal_id"], "SNAPSHOT_BINDING")
        check((root/"state.pt").stat().st_size == m["bytes"] <= m["config"]["checkpoint_bytes"]
              and sha(root/"state.pt") == m["sha256"], "SNAPSHOT_BYTES")
        payload = torch.load(root/"state.pt", map_location="cpu", weights_only=True)
        weights = payload["model"]
        check(reference_hash(weights, m["instruments"], m["world_rows"], r["binding"], "world_") == r["world"],
              "WORLD")
        check(reference_hash(weights, m["identity_instruments"], m["identity_rows"],
              m["identity_source"], "core.identity.ref_") == r["standard"], "STANDARD")
        check(all(s["sequence"] == sequence for s in m["contexts"]), "SEQUENCE")
        if name == "pending":
            for i in range(2):
                output = torch.load(a.run/"outputs"/f"0-{i}.pt", weights_only=True)
                check(torch.equal(output["world:0"], payload["world"][i])
                      and torch.equal(output["identity:0"], payload["identity"][i]), "PENDING_CONTEXT")
        snapshots[name] = dict(pin=r[name], bytes=m["bytes"])
    for record in r["outputs"]:
        path = a.run/"outputs"/record["file"]
        check(sha(path) == record["sha256"], "OUTPUT_BYTES")
        output = torch.load(path, weights_only=True)
        check(all(torch.isfinite(x).all() for x in output.values()), "OUTPUT_FINITE")
    save(a.output, dict(acceptance="author-checked", report=a.pin, complete_frames=dict(frames),
        original_bytes=total_bytes, original_native_rows=total_rows, complete_traversals=len(trace),
        snapshots=snapshots, exact_native_support_readback=True, scientific_admission=False,
        learned_capability=False, code=sha(Path(__file__)), seconds=time.perf_counter()-STARTED))
    print(json.dumps(dict(readback=sha(a.output), frames=sum(frames.values()), bytes=total_bytes)))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "inputs", "bank", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--pin", required=True)
    main(p.parse_args())
