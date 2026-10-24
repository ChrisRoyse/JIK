"""Slots from the catalogue's non-dense trained instruments (#12/#25): learned sparse embedders and trained raters.

A sense is a frozen trained embedder that reads any text (operator, #1, 2026-10-18). measure_pilot.py reads the
dense ones; this reads the others through the Stage 3 loader (lens/learned.py: one atom per forward, the
instrument's own windows and window reduction, weights hash-pinned):
  sparse  vocabulary weights -> fixed seeded sign projection to PROJ dims / sqrt(PROJ), then unit norm
          (Johnson-Lindenstrauss: cosines preserved approximately; the projection is part of the declared
          reading and recorded in the manifest);
  rater   classifier logits, stored as they are (no norm).
Multi-vector (late-interaction) instruments are not read here: pooling them would make them another dense
embedder. Generative LMs are never read (doctrine). Output float16 .npy per lens beside the dense slots.
Cost: O(atoms x lenses) forwards; projection O(nnz x PROJ) per lens.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lens"))
from lens.catalogue import by_name
from lens.learned import LearnedLens
from lens.lenscommon import run_cli, take_lease
from senses.trajectory_population import publish

PROJ, SEED, CHUNK = 2048, 20261018, 8192


def project(X):
    """CSR [n, V] -> dense [n, PROJ] by a sign matrix generated in vocabulary chunks (seeded, reproducible)."""
    n, V = X.shape
    X = X.tocsc()
    out = np.zeros((n, PROJ), np.float64)
    for c0 in range(0, V, CHUNK):
        c1 = min(V, c0 + CHUNK)
        S = np.random.default_rng([SEED, c0]).integers(0, 2, (c1 - c0, PROJ), dtype=np.int8) * 2 - 1
        out += X[:, c0:c1] @ S.astype(np.float64)
    out /= np.sqrt(PROJ)
    nrm = np.linalg.norm(out, axis=1, keepdims=True)
    if (nrm == 0).any():
        raise SystemExit(f"MNLM_SLOT_EMPTY: {int((nrm == 0).sum())} atoms have an all-zero sparse reading")
    return out / nrm


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--atoms", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--lenses", required=True)
    a = p.parse_args()
    take_lease("25", "measure-catalogue")
    texts = pq.read_table(a.atoms / "atoms.parquet", columns=["text"]).column("text").to_pylist()
    a.output.mkdir(parents=True, exist_ok=True)
    lenses = {}
    for name in a.lenses.split(","):
        spec = by_name(name)
        if spec["kind"] not in ("sparse", "rater"):
            raise SystemExit(f"MNLM_PILOT_KIND: {name} is {spec['kind']}; this reads sparse and rater instruments only")
        dest = a.output / f"{name}.npy"
        if dest.exists():
            raise SystemExit(f"MNLM_PILOT_EXISTS: {dest} (slots are immutable; use a new output)")
        t0 = time.perf_counter()
        lens = LearnedLens(spec, "cuda")
        X = lens.embed(texts)
        X = project(X) if spec["kind"] == "sparse" else np.asarray(X, np.float64)
        if not np.isfinite(X).all():
            raise SystemExit(f"MNLM_SLOT_NONFINITE: {name}")
        np.save(dest, X.astype(np.float16))
        lenses[name] = {**lens.contract(), "slot_reading": f"sign-projection {PROJ} seed {SEED}, unit" if spec["kind"] == "sparse" else "logits",
                        "dims": int(X.shape[1]), "windows_max": int(max(lens.windows)), "seconds": time.perf_counter() - t0,
                        "slot_sha256": hashlib.sha256(dest.read_bytes()).hexdigest()}
        print(name, json.dumps({k: lenses[name][k] for k in ("dims", "windows_max", "seconds")}), flush=True)
        del lens
        import torch
        torch.cuda.empty_cache()
    publish(a.output / f"manifest-catalogue-{int(time.time())}.json", {"schema": "text-pilot-constellations-catalogue-v1",
        "runtime": "atom-local-v1", "atoms": len(texts), "atoms_sha256": hashlib.sha256((a.atoms / "atoms.parquet").read_bytes()).hexdigest(),
        "lenses": lenses, "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})


if __name__ == "__main__":
    run_cli(main)
