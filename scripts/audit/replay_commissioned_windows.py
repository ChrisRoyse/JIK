"""Fresh leased CUDA replay of fixed persisted commissioned-window batches."""
import argparse
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.commissioned_window_readback import read_measurement, digest, check
from senses.commissioned_windows import check_residents
from senses.train_embedder import Encoder
from senses.trajectory_population import publish
from lens.lenscommon import take_lease, run_cli


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("publication", "plan", "calibration", "instrument", "output"):
        p.add_argument("--"+arg, type=Path, required=True)
    p.add_argument("--reviewed-desktop-residents", type=Path)
    a = p.parse_args(); check(not a.output.exists(), "MNLM_CW_REPLAY_EXISTS")
    started = time.perf_counter(); phases = {}
    before = read_measurement(a.publication, a.plan, a.calibration, a.instrument)
    report = json.loads((a.publication/"report.json").read_bytes()); c = report["contract"]
    check(c["platform"] == sys.platform and c["torch"] == torch.__version__
          and c["cuda"] == torch.version.cuda, "MNLM_CW_REPLAY_RUNTIME")
    check(c["encoder_code_sha256"] == digest(Path(sys.modules[Encoder.__module__].__file__)), "MNLM_CW_REPLAY_ENCODER")
    phases["cold_source_and_static_runtime"] = time.perf_counter()-started; tick = time.perf_counter()
    residents = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv"], text=True)
    allowed = json.loads(a.reviewed_desktop_residents.read_bytes()) if a.reviewed_desktop_residents else None
    check_residents(residents, allowed); take_lease("12", "commissioned-window-cold-replay")
    check(torch.cuda.is_available() and c["device_name"] == torch.cuda.get_device_name(0), "MNLM_CW_REPLAY_RUNTIME")
    torch.use_deterministic_algorithms(True); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    phases["resident_check_lease_and_device"] = time.perf_counter()-tick; tick = time.perf_counter()
    cfg = json.loads((a.instrument/"config.json").read_bytes())
    model = Encoder(cfg["vocab"], cfg["width"], cfg["layers"], cfg["heads"], cfg["max_len"], 0.0)
    raw = (a.instrument/"model.pt").read_bytes()
    import hashlib
    check(hashlib.sha256(raw).hexdigest() == c["model_files"]["model.pt"], "MNLM_CW_REPLAY_WEIGHTS")
    model.load_state_dict(torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True))
    model.requires_grad_(False); model.eval(); model.cuda()
    ids = np.load(a.plan/"input_ids.npy", allow_pickle=False); expected = np.load(a.publication/"vectors.npy", allow_pickle=False)
    starts = sorted({0, len(ids)//2//32*32, (len(ids)-1)//32*32})
    rows, values = [], []
    phases["frozen_weights_and_native_ids"] = time.perf_counter()-tick; tick = time.perf_counter()
    with torch.inference_mode():
        for start in starts:
            actual = model(torch.tensor(ids[start:start+32], device="cuda", dtype=torch.long)).cpu().numpy()
            check(np.array_equal(actual, expected[start:start+len(actual)]), "MNLM_CW_REPLAY_DRIFT")
            rows.extend(range(start, start+len(actual))); values.append(actual)
        probe = np.load(a.plan/"probe_ids.npy", allow_pickle=False); probe_values = []
        for start in range(0, len(probe), 32):
            probe_values.append(model(torch.tensor(probe[start:start+32], device="cuda", dtype=torch.long)).cpu().numpy())
    probe_values = np.vstack(probe_values)
    check(np.array_equal(probe_values, np.load(a.publication/"probe-vectors.npy", allow_pickle=False)), "MNLM_CW_REPLAY_PROBE_DRIFT")
    torch.cuda.synchronize()
    phases["fixed_cuda_batches_and_probe"] = time.perf_counter()-tick; tick = time.perf_counter()
    after = read_measurement(a.publication, a.plan, a.calibration, a.instrument)
    check(before["report_sha256"] == after["report_sha256"] and before["plan"]["instrument_files"] == after["plan"]["instrument_files"], "MNLM_CW_REPLAY_SOURCE_DRIFT")
    phases["cold_source_model_and_publication_drift"] = time.perf_counter()-tick; tick = time.perf_counter()
    a.output.mkdir(parents=True)
    (a.output/"gpu-before.txt").write_text(residents, encoding="utf-8")
    np.savez(a.output/"replayed.npz", rows=np.asarray(rows, np.int64), vectors=np.vstack(values), probe_vectors=probe_values)
    artifact_pin = digest(a.output/"replayed.npz"); code_pin = digest(Path(__file__))
    phases["artifact_write_and_hash"] = time.perf_counter()-tick; wall = time.perf_counter()-started
    publish(a.output/"report.json", dict(acceptance="author-checked", source_report_sha256=before["report_sha256"],
            lens_id=before["lens_id"], fixed_batch_starts=starts, ordinary_windows_replayed=len(rows),
            probe_windows_replayed=len(probe_values), all_replayed_vectors_bitwise_equal=True,
            all_vectors_replayed=False, instrument_and_source_unchanged=True, scientific_admission=False,
            artifact_sha256=artifact_pin, code_sha256=code_pin,
            phases_seconds=phases, wall_seconds=wall, phase_coverage=sum(phases.values())/wall))
    print(json.dumps(dict(ordinary_windows_replayed=len(rows), probe_windows_replayed=len(probe_values), bitwise_equal=True,
                         wall_seconds=wall, phase_coverage=sum(phases.values())/wall)))


if __name__ == "__main__":
    sys.exit(run_cli(main))
