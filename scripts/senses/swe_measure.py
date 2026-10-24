"""Measure the explicit SWE senses for every observation (CPU; parsing only, nothing executed).

Per task the pinned source snapshot is read once; per attempt only the touched files are written to a
scratch tree, the attempt's own patch is applied there (git apply, then the SWE-bench fuzz fallback),
and base/patched texts feed the structure and diagnostic senses. Output: one float32 matrix per sense,
rows in observation order, plus a manifest binding the observation store and the code.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oracle.prepare_swe import apply_patch, patch_paths, source_files
from senses import swe_lenses as L
from senses.contract import SenseError
from senses.trajectory_population import publish

SENSES = ("structure", "behaviour", "relations", "diagnostic", "provenance")


def per_task(args):
    root, task, records = args
    out, notes = [], []
    files = source_files(Path(root), records[0]["source"])
    for r in records:
        vecs, note = {}, {}
        patch = r["patch"]
        base, patched = {}, {}
        try:
            touched = sorted(patch_paths(patch))
            py = [p for p in touched if p.endswith(".py")]
            base = {p: files[p].decode("utf-8", "replace") for p in py if p in files}
            tmp = Path(tempfile.mkdtemp(prefix="mnlm-sense-"))
            try:
                for p in touched:
                    if p in files:
                        (tmp / p).parent.mkdir(parents=True, exist_ok=True); (tmp / p).write_bytes(files[p])
                note["apply"] = apply_patch(tmp, tmp.parent / (tmp.name + ".patch"), patch, touched, fallback=True)
                patched = {p: (tmp / p).read_text("utf-8", "replace") for p in py if (tmp / p).exists()}
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
                (tmp.parent / (tmp.name + ".patch")).unlink(missing_ok=True)
        except SenseError as exc:
            note["apply_error"] = exc.code  # structure/diagnostic then see base only; recorded, not hidden
        vecs["structure"] = L.structure(base, patched, patch)
        vecs["behaviour"] = L.behaviour(r["trajectory"])
        vecs["relations"] = L.relations(r["issue"], patch)
        vecs["diagnostic"] = L.diagnostic(base, patched) if patched else np.zeros(8, np.float32)
        vecs["provenance"] = L.provenance(r["model"], r["exit_status"])
        out.append((r["uid"], {k: v.astype(np.float32) for k, v in vecs.items()}))
        notes.append({"uid": r["uid"], **note})
    return out, notes


def run(root, obs_dir, output, workers):
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=False)
    records = [json.loads(l) for l in (obs_dir / "observations.jsonl").read_text(encoding="utf-8").splitlines()]
    order = [r["uid"] for r in records]
    by_task = {}
    for r in records:
        by_task.setdefault(r["task"], []).append(r)
    results, notes = {}, []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for out, nt in pool.map(per_task, [(str(root), t, rs) for t, rs in sorted(by_task.items())]):
            results.update(out); notes += nt
    files = {}
    for s in SENSES:
        M = np.stack([results[u][s] for u in order]).astype(np.float32)
        if not np.isfinite(M).all():
            raise SystemExit(f"MNLM_SENSE_NONFINITE {s}")
        np.save(output / f"{s}.npy", M)
        files[s] = {"path": f"{s}.npy", "shape": list(M.shape), "sha256": hashlib.sha256((output / f"{s}.npy").read_bytes()).hexdigest()}
    code = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__).parent / "swe_lenses.py", Path(__file__))}
    manifest = json.loads((obs_dir / "manifest.json").read_bytes())
    publish(output / "notes.json", {"notes": notes})
    return publish(output / "manifest.json", {"schema": "swe-explicit-senses-v1", "version": L.VERSION, "senses": files,
        "rows": len(order), "uid_order_sha256": hashlib.sha256("\n".join(order).encode()).hexdigest(),
        "observations_sha256": manifest["observations"]["sha256"], "code": code,
        "apply_errors": sum(1 for n in notes if "apply_error" in n), "wall_seconds": time.perf_counter() - started})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--observations", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=16)
    a = p.parse_args()
    print(json.dumps({"sha256": run(a.root, a.observations, a.output, a.workers)}))
