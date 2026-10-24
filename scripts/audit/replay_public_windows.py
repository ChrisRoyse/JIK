"""Fresh leased CUDA replay of fixed source batches; never claims full replay."""
import argparse
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from text import window_readings as runtime
from audit.window_readback import readback, check, sha
from senses.commissioned_windows import check_residents
from senses.trajectory_population import publish
from lens.lenscommon import run_cli, take_lease
import numpy as np
import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("publication", "calibration", "output"):
        p.add_argument("--"+arg, type=Path, required=True)
    p.add_argument("--reviewed-desktop-residents", type=Path)
    a = p.parse_args(); check(not a.output.exists(), "MNLM_PUBLIC_REPLAY_EXISTS")
    started = time.perf_counter(); phases = {}
    before = readback(a.publication, a.calibration)
    report = json.loads((a.publication/"report.json").read_bytes())
    records, _ = runtime.load_chunks(a.calibration, report["calibration_report_sha256"])
    lane_reports = {n: json.loads((a.publication/n/"report.json").read_bytes()) for n in report["lenses"]}
    for rep in lane_reports.values():
        c = rep["contract"]
        check(c["platform"] == sys.platform and c["torch"] == torch.__version__
              and c["cuda"] == torch.version.cuda and c["transformers"] == runtime.transformers.__version__
              and c["runtime_code_sha256"] == sha(Path(runtime.__file__).read_bytes()), "MNLM_PUBLIC_REPLAY_RUNTIME")
    phases["cold_source_native_and_static_runtime"] = time.perf_counter()-started; tick = time.perf_counter()
    residents = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv"], text=True)
    allowed = json.loads(a.reviewed_desktop_residents.read_bytes()) if a.reviewed_desktop_residents else None
    check_residents(residents, allowed); take_lease("25", "public-native-fixed-batch-replay")
    torch.use_deterministic_algorithms(True); torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    a.output.mkdir(parents=True); (a.output/"gpu-before.txt").write_text(residents, encoding="utf-8")
    phases["resident_check_and_lease"] = time.perf_counter()-tick
    starts = sorted({0, len(records)//2//32*32, (len(records)-1)//32*32}); results = {}
    for name, rep in lane_reports.items():
        tick = time.perf_counter(); dest = a.publication/name
        lens = runtime.LearnedLens(runtime.by_name(name), "cuda")
        contract = {**lens.contract(), "runtime": runtime.RUNTIME, "kind": "dense-window-multi", "shape": f"MultiWindows({lens.dim})",
                    "window_reduction": "none:ordered-window-vectors", "execution_unit": "fixed-source-batch-32-v1",
                    "batch": 32, "torch": torch.__version__, "transformers": runtime.transformers.__version__,
                    "cuda": torch.version.cuda, "device_name": torch.cuda.get_device_name(0), "platform": sys.platform,
                    "backend_code_sha256": sha(Path(sys.modules[runtime.LearnedLens.__module__].__file__).read_bytes()),
                    "runtime_code_sha256": sha(Path(runtime.__file__).read_bytes())}
        check(contract == rep["contract"], "MNLM_PUBLIC_REPLAY_INSTRUMENT")
        check(not lens.model.training and all(not v.requires_grad for v in lens.model.parameters()), "MNLM_PUBLIC_REPLAY_UNFROZEN")
        with np.load(dest/"slots.npz", allow_pickle=False) as z:
            expected, offsets = z["vectors"].copy(), z["offsets"].copy()
        with gzip.open(dest/"traces.jsonl.gz", "rt", encoding="utf-8") as f:
            traces = [json.loads(line)["trace"] for line in f]
        values = []; rows = []; source_rows = []
        for start in starts:
            batch = records[start:start+32]; stop = start+len(batch)
            actual, idx, segments = runtime.forward(lens, [r["text"] for r in batch])
            lo, hi = map(int, offsets[[start, stop]])
            check(np.array_equal(actual, expected[lo:hi]) and np.array_equal(idx, offsets[start:stop+1]-lo)
                  and segments == traces[start:stop], "MNLM_PUBLIC_REPLAY_DRIFT")
            values.append(actual); rows.extend(range(lo, hi)); source_rows.extend(range(start, stop))
        probe = json.loads((dest/"probe.json").read_bytes())
        text = "".join(r["text"] for r in records if (r["family"], r["row"]) == (probe["family"], probe["row"]))
        pv, po, ps = runtime.forward(lens, [text])
        with np.load(dest/"probe.npz", allow_pickle=False) as z:
            check(np.array_equal(pv, z["vectors"]) and np.array_equal(po, z["offsets"])
                  and ps == [probe["trace"]], "MNLM_PUBLIC_REPLAY_PROBE_DRIFT")
        check(runtime.weights_sha256(lens.snap)[0] == lens.sha256
              and runtime.preprocessing_sha256(lens.snap) == lens.preprocessing, "MNLM_PUBLIC_REPLAY_MODEL_DRIFT")
        np.savez(a.output/(name+".npz"), vector_rows=np.asarray(rows, np.int64), source_rows=np.asarray(source_rows, np.int64),
                 vectors=np.vstack(values), probe_vectors=pv)
        results[name] = dict(lens_id=rep["lens_id"], ordinary_windows_replayed=len(rows), probe_windows_replayed=len(pv),
                             source_inputs_replayed=len(source_rows), artifact_sha256=sha((a.output/(name+".npz")).read_bytes()),
                             all_replayed_vectors_bitwise_equal=True, all_vectors_replayed=False)
        del lens; torch.cuda.empty_cache()
        phases["load_replay_publish_"+name] = time.perf_counter()-tick
        print(json.dumps(dict(name=name, **results[name])), flush=True)
    tick = time.perf_counter(); after = readback(a.publication, a.calibration)
    check(before == after, "MNLM_PUBLIC_REPLAY_SOURCE_DRIFT")
    phases["cold_source_and_native_final_readback"] = time.perf_counter()-tick; wall = time.perf_counter()-started
    publish(a.output/"report.json", dict(acceptance="author-checked", publication_sha256=before["report_sha256"],
            fixed_source_batch_starts=starts, lanes=results, source_and_instruments_unchanged=True,
            all_vectors_replayed=False, scientific_admission=False, phases_seconds=phases, wall_seconds=wall,
            phase_coverage=sum(phases.values())/wall, code_sha256=sha(Path(__file__).read_bytes())))


if __name__ == "__main__":
    sys.exit(run_cli(main))
