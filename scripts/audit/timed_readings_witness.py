"""Timed candidate bank -> existing model attention -> durable withdrawal/recovery."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
import copy
import json
from pathlib import Path
import shutil
import sys
import time
import subprocess

STARTED = time.perf_counter()
sys.path[:0] = [str(Path(__file__).resolve().parents[1]),
               str(Path(__file__).resolve().parents[1]/"lens")]
import torch
from torch import nn
from audit.native_stream_witness import branch, execution, compare
from audit.timed_speech_witness import request
from claims.timed_speech_memory import SpeechMemory
from lenscommon import take_lease, release_lease
from model.contracts import ModelConfig
from model.native_stream_input import descriptor
from senses.contract import digest
from senses.perception_corpus import pinned, sha
from senses.timed_speech import FLAGS, save
from senses.timed_readings import Bank, Inputs, ROSTER, need


def main(a):
    a.output.mkdir(parents=True, exist_ok=False)
    inputs = Inputs(a.inputs, a.input_pin, a.source)
    bank_data = pinned(a.bank/"bank.json", a.bank_pin)
    ids = {n: digest(c) for n, c in bank_data["instruments"].items()}
    bank = Bank(a.bank, a.bank_pin, inputs, ids)
    retained = pinned(a.inputs/"rollback-retained.json", a.retained_pin)
    need(sha(a.inputs/"rollback.sqlite") == a.memory_pin, "MEMORY_FILE")
    path = a.output/"memory.sqlite"
    shutil.copy2(a.inputs/"rollback.sqlite", path)
    memory = SpeechMemory(path, inputs.evidence, create=False, retained=retained)
    previous = pinned(a.previous/"report.json", a.previous_pin) if a.previous else None
    phases = dict(import_source_memory=time.perf_counter()-STARTED)
    resident = subprocess.check_output(["nvidia-smi",
        "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"], text=True)
    (a.output/"gpu-before.txt").write_text(subprocess.check_output(["nvidia-smi"], text=True))
    need(not resident.strip(), "GPU_BUSY")
    take_lease("28", "timed-native-model-consumer")
    results, controls, modules = [], [], nn.ModuleDict()
    config = ModelConfig()

    def deny(name, operation, expected):
        try:
            operation()
        except Exception as exc:
            actual = getattr(exc, "code", str(exc))
            need(actual == expected, "CONTROL_REASON:"+name+":"+actual)
            controls.append(dict(name=name, error=actual))
        else:
            raise ValueError("MNLM_TIMED_READINGS_CONTROL_ACCEPTED:"+name)

    try:
        tick = time.perf_counter()
        torch.set_num_threads(32)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.manual_seed(20261020)
        streams = []
        for moment in inputs.moments:
            streams.extend(bank.streams(memory, moment["id"], **request(moment)))
        need(len(streams) == len(inputs.moments)*len(ROSTER), "CONSUMER_ROSTER")
        for stream in streams:
            if stream.instrument not in modules:
                modules[stream.instrument] = branch(config, stream.values.shape[1])
        if previous:
            need(previous["bank_pin"] == a.bank_pin
                 and sha(a.previous/"weights.pt") == previous["weights_sha256"],
                 "MODEL_CHECKPOINT")
            modules.load_state_dict(torch.load(a.previous/"weights.pt",
                map_location="cpu", weights_only=True), strict=True)
        torch.save(modules.state_dict(), a.output/"weights.pt")
        modules.cuda()
        (a.output/"proofs").mkdir()
        phases["streams_and_model_load"] = time.perf_counter()-tick
        tick = time.perf_counter()
        for i, stream in enumerate(streams):
            module = modules[stream.instrument]
            with torch.no_grad():
                query = module["projector"](
                    torch.from_numpy(stream.values[:1].copy())[None].cuda()).cpu()
            actual, peak = execution(module, query, stream, config, "stream")
            if previous:
                expected = torch.load(a.previous/"proofs"/f"{i}.pt",
                                     map_location="cpu", weights_only=True)
                pairs = [(actual[k], expected[k]) for k in ("output", "query_gradient")]
                pairs.extend((v, expected["gradients"][k]) for k, v in actual["gradients"].items())
                need(all(torch.equal(left, right) for left, right in pairs), "COLD_MODEL_REPLAY")
                comparison = dict(exact=True)
            else:
                dense, _ = execution(module, query, stream, config, "dense")
                comparison = compare(actual, dense)
                need(not comparison["failed"], "MODEL_ARITHMETIC")
            torch.save(actual, a.output/"proofs"/f"{i}.pt")
            results.append(dict(stream=descriptor(stream), comparison=comparison,
                proof_sha256=sha(a.output/"proofs"/f"{i}.pt"), peak_cuda_bytes=peak))
        phases["model_execution_and_readback"] = time.perf_counter()-tick
        tick = time.perf_counter()
        first, last = inputs.moments[0], inputs.moments[-1]
        for name, change, expected in (
            ("unadmitted-training", dict(purpose="training"), "MNLM_TIMED_READINGS_ADMISSION_REQUIRED"),
            ("prospective-text", dict(through=first["content"]["at"]), "MNLM_TIMED_SPEECH_FUTURE_ANNOTATION"),
            ("wrong-clock", dict(clock_id="wrong"), "MNLM_TIMED_SPEECH_CLOCK"),
            ("wrong-speaker", dict(speaker={}), "MNLM_TIMED_SPEECH_SPEAKER")):
            deny(name, lambda change=change: bank.streams(memory, first["id"],
                 **(request(first) | change)), expected)
        bad_manifests = []
        value = copy.deepcopy(bank_data); value["readings"].pop(ROSTER[0])
        bad_manifests.append(("missing-instrument", value, "ROSTER"))
        value = copy.deepcopy(bank_data); value["flags"]["scientific_admission"] = True
        bad_manifests.append(("false-admission", value, "FLAGS"))
        value = copy.deepcopy(bank_data); value["readings"][ROSTER[0]][0]["shape"][0] = 1
        bad_manifests.append(("pooled-patches", value, "NATIVE_LAYOUT"))
        value = copy.deepcopy(bank_data)
        value["readings"][ROSTER[0]][0]["instrument_id"] = ids[ROSTER[1]]
        bad_manifests.append(("swapped-instrument", value, "INSTRUMENT"))
        value = copy.deepcopy(bank_data)
        value["readings"][ROSTER[0]][0]["input"]["frame"]["pts"] += 1
        bad_manifests.append(("shifted-time", value, "INPUT_ROSTER"))
        value = copy.deepcopy(bank_data); value["readings"][ROSTER[-1]].reverse()
        bad_manifests.append(("wrong-text-event", value, "INPUT_ROSTER"))
        for name, value, code in bad_manifests:
            dest = a.output/name
            dest.mkdir()
            save(dest/"bank.json", value)
            deny(name, lambda dest=dest: Bank(dest, sha(dest/"bank.json"), inputs, ids),
                 "MNLM_TIMED_READINGS_"+code)
        # Corruption control uses a private copy, never writes into the live bank.
        dest = a.output/"corrupt-reading"
        dest.mkdir()
        save(dest/"bank.json", bank_data)
        lane = ROSTER[0]
        (dest/lane).mkdir()
        entry = bank_data["readings"][lane][0]
        raw = (a.bank/lane/entry["file"]).read_bytes()
        (dest/lane/entry["file"]).write_bytes(bytes([raw[0]^1])+raw[1:])
        bad = Bank(dest, sha(dest/"bank.json"), inputs, ids)
        deny("corrupt-native-values", lambda: bad.read(lane, entry["input"]),
             "MNLM_TIMED_READINGS_READING_BYTES")
        phases["source_and_reading_controls"] = time.perf_counter()-tick
        tick = time.perf_counter()
        before = memory.snapshot()
        original = [descriptor(s) for s in bank.streams(memory, last["id"], **request(last))]
        memory.apply(dict(op="invalidate", episode=first["evidence"][1],
                          reason="FSV withdrawal control, not an actual provider retraction"),
                     expected_head=memory.retained()["head"])
        deny("stale-model-stream", lambda: list(streams[0].blocks(config)), "MNLM_NATIVE_STREAM_STALE")
        deny("withdrawn-moment", lambda: bank.streams(memory, first["id"], **request(first)),
             "MNLM_TIMED_SPEECH_SUPPORT_WITHDRAWN_OR_UNCHECKED")
        updated = bank.streams(memory, last["id"], **request(last))
        need([s.identity for s in updated] == [r["identity"] for r in original],
             "UNRELATED_INSTRUMENT_CHANGED")
        after = memory.snapshot()
        memory.close()
        memory = SpeechMemory(path, inputs.evidence, create=False, retained=after["retained"])
        need(memory.snapshot() == after, "RESTART_STATE")
        deny("restart-withdrawn", lambda: bank.streams(memory, first["id"], **request(first)),
             "MNLM_TIMED_SPEECH_SUPPORT_WITHDRAWN_OR_UNCHECKED")
        restored_path = a.output/"rollback.sqlite"
        shutil.copy2(a.inputs/"rollback.sqlite", restored_path)
        rollback = SpeechMemory(restored_path, inputs.evidence, create=False, retained=retained)
        need(rollback.snapshot() == before, "ROLLBACK_STATE")
        restored = bank.streams(rollback, first["id"], **request(first))
        need([descriptor(s) for s in restored] == [descriptor(s) for s in streams[:len(ROSTER)]],
             "ROLLBACK_NATIVE_STREAMS")
        rollback.close()
        identity = Path("/zfs/archive/mnlm/raw/identity-v1/profile.md")
        need(sha(identity) == inputs.report["identity_pin"], "IDENTITY_CHANGED")
        phases["withdraw_restart_rollback"] = time.perf_counter()-tick
    finally:
        memory.close()
        release_lease()
    result = dict(bank_pin=a.bank_pin, input_pin=a.input_pin, flags=FLAGS, results=results,
        controls=controls, before=before, after=after, rollback_exact=True,
        weights_sha256=sha(a.output/"weights.pt"), identity_pin=sha(identity),
        component="existing stream_attention.observe; also used by IdentityLoopPredictor",
        complete_native_predictor_connected=False, learned_update=False,
        code=sha(Path(__file__)), phases=phases, seconds=time.perf_counter()-STARTED,
        acceptance="author-checked only")
    result["phase_coverage"] = sum(phases.values())/result["seconds"]
    save(a.output/"report.json", result)
    need(result["phase_coverage"] >= .9, "PHASE_COVERAGE")
    print(json.dumps(dict(report=sha(a.output/"report.json"), streams=len(results),
        controls=len(controls), seconds=result["seconds"], phase_coverage=result["phase_coverage"])))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("inputs", "source", "bank", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    for name in ("input-pin", "bank-pin", "memory-pin", "retained-pin"):
        p.add_argument("--"+name, required=True)
    p.add_argument("--previous", type=Path)
    p.add_argument("--previous-pin")
    main(p.parse_args())
