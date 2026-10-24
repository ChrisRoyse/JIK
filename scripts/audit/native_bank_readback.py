"""Cold readback of complete typed candidate rows and their execution evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path[:0] = [str(Path(__file__).resolve().parents[1]),
               str(Path(__file__).resolve().parents[1]/"lens")]
import numpy as np
from senses import visual_bank, audio_bank
from senses.perception_corpus import pinned, save, sha
from senses.contract import digest


def need(ok, code):
    if not ok:
        raise ValueError("MNLM_NATIVE_BANK_READBACK_"+code)


def main(a):
    started = time.perf_counter()
    report = pinned(a.bank/"report.json", a.report_pin)
    inputs = pinned(a.inputs/"report.json", report["input_pin"])
    need(report["author_checked"] and not report["scientific_admission"]
         and not report["training_ready"], "FLAGS")
    counts, phases = {}, dict(setup=time.perf_counter()-started)
    for medium, proof in report["banks"].items():
        start = time.perf_counter()
        need(medium in ("image","audio"), "MEDIUM")
        module = visual_bank if medium == "image" else audio_bank
        cls = visual_bank.VisualBank if medium == "image" else audio_bank.AudioBank
        input_pin = inputs["input_pins"][medium]
        root = a.bank/medium
        reader = cls(root,proof["bank_pin"],a.inputs/medium,input_pin,proof["instruments"])
        need(proof["input_pin"] == input_pin
             and pinned(root/"report.json",sha(root/"report.json")) == proof, "REPORT_BINDING")
        names = module.NAMES
        views = module.input_views(a.inputs/medium,input_pin)
        counts[medium] = {}
        for name in names:
            identity = reader.value["instruments"][name]
            state = proof["states"][name]
            need(digest(identity) == proof["instruments"][name]
                 and state["before"] == state["after"]
                 and state["frozen_parameters"] and not state["model_training"], "FROZEN_STATE")
            if name == "whisper":
                need(state["before"] == identity["encoder_state_sha256"], "WHISPER_STATE")
            elif name == "dinov2":
                # This established identity pins checkpoint bytes and the exact
                # bijective key map; its runtime state hash is in the witness.
                need(len(identity["weights_sha256"]) == 64
                     and identity["checkpoint_layout"]["tensors"] > 0, "DINO_CHECKPOINT")
            else:
                need(identity["frozen"] and state["before"] == identity["state_sha256"],
                     "CANDIDATE_STATE")
            total_bytes, total_rows = 0, 0
            stream_digest = hashlib.sha256()
            for entry in reader.value["readings"][name]:
                path = root/entry["file"]
                need(path.stat().st_size == entry["shape"][0]*entry["shape"][1]*4
                     and sha(path) == entry["sha256"], "PERSISTED_BYTES")
                raw = path.read_bytes()
                value = np.frombuffer(raw,dtype="<f4").reshape(entry["shape"])
                need(np.isfinite(value).all(), "FINITE")
                total_bytes += len(raw)
                total_rows += len(value)
                stream_digest.update(raw)
            need(total_bytes == state["value_bytes"] ==
                 inputs["projected_slot_bytes"]["dino" if name == "dinov2" else name]
                 and total_rows == state["rows"], "NATIVE_TOTALS")
            replayed = [x for x in proof["replays"] if x["instrument"] == name]
            need(len(replayed) == len(views) and all(x["exact"] for x in replayed)
                 and [x["file"] for x in replayed] ==
                     [x["file"] for x in reader.value["readings"][name]], "REPLAYS")
            frontend = [x for x in proof["frontends"] if x["instrument"] == name]
            need(len(frontend) >= len(views) and all(x["exact"] for x in frontend), "FRONTENDS")
            counts[medium][name] = dict(windows=len(views), rows=total_rows,
                bytes=total_bytes, dim=reader.value["readings"][name][0]["shape"][1],
                ordered_value_sha256=stream_digest.hexdigest(),
                instrument_id=proof["instruments"][name])
        need(sum(1 for _ in reader.consume()) == len(views)*len(names), "TYPED_CONSUMER")
        need(len(proof["controls"]) == 7 and
             {c["control"] for c in proof["controls"]} ==
             {"missing-instrument","swapped-instrument","pooled-native-rows","false-admission",
              "changed-instrument","premature-training","corrupt-values"}, "NEGATIVE_CONTROLS")
        phases[medium] = time.perf_counter()-start
    value = dict(author_checked=True, report_pin=a.report_pin, input_pin=report["input_pin"],
        source_readback_pin=report["source_readback_pin"], counts=counts, phases=phases,
        scientific_admission=False, training_ready=False, code_pin=sha(Path(__file__)),
        seconds=time.perf_counter()-started)
    value["phase_coverage"] = sum(phases.values())/value["seconds"]
    pin = save(a.output,value)
    print(json.dumps(dict(readback_pin=pin, counts=counts,
        seconds=value["seconds"], phase_coverage=value["phase_coverage"])),flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("bank","inputs","output"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--report-pin",required=True)
    main(p.parse_args())
