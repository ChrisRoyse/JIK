"""Deterministic surface encoders as slot files (#12/#25): explicit instruments beside the learned embedders.

Encoders (lens/encoders.py, identity = that file's BLAKE3): text-hash-tfidf (hashed word+bigram TF-IDF, idf
fitted on the given rows, label-free), text-char-trigram (hashed character trigrams), text-stylometry
(79 ratios: case, digits, punctuation, word length, function words). Sparse outputs are mapped to DIMS
dense dimensions by a fixed seeded Gaussian projection (Johnson-Lindenstrauss: cosines approximately
preserved), then unit-normalised and stored as float16 like the learned slots. An atom an encoder cannot
read (empty) is recorded in the manifest; the assay refuses absent rows.
Cost: O(characters) per encoder plus O(nnz * DIMS) for the projection.
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lens"))
import encoders
from senses.trajectory_population import publish

NAMES, DIMS, SEED = ("text-hash-tfidf", "text-char-trigram", "text-stylometry"), 1024, 20261018


def project(m, name):
    rng = np.random.default_rng(int(hashlib.sha256(f"{SEED}|{name}".encode()).hexdigest()[:8], 16))
    R = rng.standard_normal((m.shape[1], DIMS)).astype(np.float32) / np.sqrt(DIMS)
    return np.asarray(m @ R, dtype=np.float32)


def main():
    atoms, out = Path(sys.argv[1]), Path(sys.argv[2])
    texts = pq.read_table(atoms / "atoms.parquet", columns=["text"]).column("text").to_pylist()
    rows = [{"text": t} for t in texts]
    out.mkdir(parents=True, exist_ok=True)
    rec = {}
    for name in NAMES:
        dest = out / f"{name}.npy"
        if dest.exists():
            raise SystemExit(f"MNLM_PILOT_EXISTS: {dest}")
        slot = encoders.encode(name, rows)
        X = project(slot.X, name) if slot.sparse else np.asarray(slot.X, np.float32)
        if not slot.sparse:  # dense ratios: standardise columns (label-free) so no single ratio dominates cosine
            X = (X - X.mean(0)) / (X.std(0) + 1e-6)
        nrm = np.linalg.norm(X, axis=1, keepdims=True)
        absent = (nrm[:, 0] == 0) | slot.absent
        X = np.where(nrm > 0, X / np.where(nrm > 0, nrm, 1), 0)
        np.save(dest, X.astype(np.float16))
        rec[name] = {"dims": int(X.shape[1]), "absent": int(absent.sum()), "sparse_source": bool(slot.sparse), "fit": slot.fit,
                     "slot_sha256": hashlib.sha256(dest.read_bytes()).hexdigest()}
        print(name, rec[name]["dims"], "absent", rec[name]["absent"], flush=True)
    publish(out / "manifest-explicit.json", {"schema": "text-explicit-slots-v1", "encoders_blake3": encoders.ENCODERS_B3,
        "projection": {"kind": "gaussian", "dims": DIMS, "seed": SEED}, "atoms_sha256": hashlib.sha256((atoms / "atoms.parquet").read_bytes()).hexdigest(),
        "lenses": rec, "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})


if __name__ == "__main__":
    main()
