"""Physical negative controls and unchanged single-prefix behavior for masked learning."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/"lens")]
import numpy as np
import torch
from audit.text_masked_learning import restore
from claims.codata_source import check
from lenscommon import take_lease, release_lease
from model.contracts import ModelConfig
from model.text_masked_associations import MaskedData, build_probe
from senses.trajectory_population import publish
from text.constellation_corpus import file_sha
from text.constellation_input import Corpus
from text.corpus_bank import Bank


def refuse(fn, code):
    try:
        fn()
    except ValueError as exc:
        check(str(exc) == "MNLM_CODATA_"+code, "MASKED_BOUNDARY_WRONG_CODE")
        return str(exc)
    raise ValueError("MNLM_CODATA_MASKED_BOUNDARY_ACCEPTED")


def run(a):
    start = time.perf_counter()
    a.output.mkdir(exist_ok=False)
    corpus = Corpus(a.corpus, a.corpus_pin)
    data = MaskedData(corpus, Bank(a.bank, a.bank_pin, corpus), ModelConfig())
    phases = dict(source=time.perf_counter()-start)
    tick = time.perf_counter()
    (a.output/"gpu-before.txt").write_text(subprocess.check_output(["nvidia-smi"], text=True))
    check(torch.cuda.is_available() and not subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip(), "MASKED_GPU_BUSY")
    take_lease("28", "masked-boundary-readback")
    try:
        torch.set_num_threads(32)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        phases["device_setup"] = time.perf_counter()-tick
        tick = time.perf_counter()
        probe = build_probe(data, a)
        report = json.loads((a.source/"report.json").read_bytes())
        restore(a.source/"cycle-2", report["checkpoints"]["cycle-2"], probe, data)
        phases["load"] = time.perf_counter()-tick
        tick = time.perf_counter()
        denied = {}
        denied["wrong_pin"] = refuse(lambda: restore(a.source/"cycle-2", "0"*64, probe, data), "MASKED_CHECKPOINT_PIN")
        bad = a.output/"corrupt-checkpoint"
        bad.mkdir()
        original = a.source/"cycle-2"
        (bad/"manifest.json").write_bytes((original/"manifest.json").read_bytes())
        raw = bytearray((original/"state.pt").read_bytes())
        raw[len(raw)//2] ^= 1
        (bad/"state.pt").write_bytes(raw)
        denied["persisted_corruption"] = refuse(lambda: restore(bad, report["checkpoints"]["cycle-2"], probe, data), "MASKED_CHECKPOINT_BYTES")
        keys = probe.keys_0.clone()
        with torch.no_grad():
            probe.keys_0[0, 0] += .01
            denied["frozen_key_mutation"] = refuse(probe.check_reference, "MASKED_FROZEN_REFERENCE_CHANGED")
            probe.keys_0.copy_(keys)
        probe.check_reference()
        dev = next(g["indices"] for g in data.groups if g["split"] == "development")
        denied["development_gradient"] = refuse(lambda: data.loss(probe, dev, shuffled=False), "MASKED_DEVELOPMENT_GRADIENT")
        denied["unknown_atom"] = refuse(lambda: data.observed([-1], data.lanes[0]), "MASKED_SOURCE_INDEX")
        # Exercise an actually mutated observed cache, then restore the whole checkpoint/data source.
        index, name = data.train_nodes[0], data.lanes[1]
        saved = data.cached[index][name].clone()
        with torch.no_grad():
            data.cached[index][name][0, 0] += .1
        denied["changed_observed_source"] = refuse(lambda: data.observed([index], data.lanes[0]), "MASKED_SOURCE_CHANGED")
        with torch.no_grad():
            data.cached[index][name].copy_(saved)
        # No continuation uses the invalidated cache version. Regression inputs are fresh source arrays.
        previous = subprocess.check_output(["git", "show", "3f1b2c0667567d87fc29ffdd69040533df1e7328:scripts/model/native_prefix_transformer.py"])
        old_path = a.output/"previous_prefix_transformer.py"
        old_path.write_bytes(previous)
        spec = importlib.util.spec_from_file_location("masked_previous_prefix", old_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with torch.random.fork_rng(devices=[]):
            old = module.NativePrefixTransformer(probe.config, probe.specs).cuda().eval()
        old.load_state_dict(probe.encoder.state_dict(), strict=True)
        frames = tuple({n: torch.from_numpy(data.native[n][i].copy()).cuda() for n in data.lanes}
                       for i in data.train_nodes[:6])
        with torch.no_grad():
            expected = old.encode(frames, probe.projectors)
            actual = probe.encoder.encode(frames, probe.projectors)
        check(torch.equal(expected, actual), "MASKED_EXISTING_PREFIX_CHANGED")
        np.savez(a.output/"prefix-regression.npz", expected=expected.cpu().numpy(), actual=actual.cpu().numpy())
        phases["physical_controls"] = time.perf_counter()-tick
        elapsed = time.perf_counter()-start
        result = dict(schema="masked-association-boundaries-v1", source=file_sha(a.source/"report.json"),
            refused=denied, frozen_keys_restored=True, existing_prefix_byte_exact=True,
            previous_source=file_sha(old_path), phases_seconds=phases, wall_seconds=elapsed,
            phase_coverage=sum(phases.values())/elapsed, acceptance="author-checked only")
        check(.9 <= result["phase_coverage"] <= 1.001, "MASKED_BOUNDARY_PHASE_COVERAGE")
        publish(a.output/"report.json", result)
        print(json.dumps(dict(pin=file_sha(a.output/"report.json"), refused=len(denied), prefix_exact=True)))
    finally:
        release_lease()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for n in ("source", "output", "corpus", "bank", "identity-population", "identity-native", "identity-source"):
        p.add_argument("--"+n, type=Path, required=True)
    for n in ("corpus-pin", "bank-pin"):
        p.add_argument("--"+n, required=True)
    run(p.parse_args())
