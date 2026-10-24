"""Slots from commissioned senses (commission-v1): frozen, hash-pinned embedders we trained (train_embedder.py).

Each sense is loaded from its directory; its weights sha256 is re-computed and must equal the one in its
report.json (a changed file is a different instrument: MNLM_COMMISSION_WEIGHTS). Runtime: eval mode, fp32,
atoms ordered by length in fixed batches of BATCH (deterministic geometry), outputs unit-normalised, stored
float16 as <name>.npy beside the other slots, plus a manifest. No CPU fallback. One GPU lease.
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
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lens.lenscommon import run_cli, take_lease
from senses.train_embedder import MAX_LEN, PAD, Encoder
from senses.trajectory_population import publish

BATCH = 512


def load(d: Path, dev):
    from tokenizers import Tokenizer
    rep = json.loads((d / "report.json").read_bytes())
    sha = hashlib.sha256((d / "model.pt").read_bytes()).hexdigest()
    if sha != rep["weights_sha256"]:
        raise SystemExit(f"MNLM_COMMISSION_WEIGHTS: {d} model.pt {sha} != report {rep['weights_sha256']}")
    cfg = rep["config"]
    m = Encoder(cfg["vocab"], cfg["width"], cfg["layers"], cfg["heads"], cfg["max_len"], 0.0).to(dev)
    m.load_state_dict(torch.load(d / "model.pt", map_location=dev)); m.eval()
    return m, Tokenizer.from_file(str(d / "tokenizer.json")), rep


CHUNK = 262144  # atoms per tokenisation chunk (bounded memory); batches are formed within a chunk


@torch.no_grad()
def embed(model, tok, texts, dev):
    if len(texts) > CHUNK:
        return np.concatenate([embed(model, tok, texts[i:i + CHUNK], dev) for i in range(0, len(texts), CHUNK)])
    ids = [e.ids[:MAX_LEN] for e in tok.encode_batch(texts)]
    order = np.argsort([len(x) for x in ids], kind="stable")
    out = np.zeros((len(texts), model.norm.normalized_shape[0]), np.float32)
    for j in range(0, len(order), BATCH):
        rows = order[j:j + BATCH]
        L = max(1, max(len(ids[r]) for r in rows))
        x = torch.full((len(rows), L), PAD, dtype=torch.long)
        for i, r in enumerate(rows):
            if ids[r]:
                x[i, : len(ids[r])] = torch.tensor(ids[r])
        out[rows] = model(x.to(dev)).float().cpu().numpy()
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--atoms", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--senses", required=True, help="comma list of commissioned sense directories")
    a = p.parse_args()
    take_lease("25", "measure-commissioned")
    dev = "cuda"
    texts = pq.read_table(a.atoms / "atoms.parquet", columns=["text"]).column("text").to_pylist()
    a.output.mkdir(parents=True, exist_ok=True)
    rec = {}
    for d in map(Path, a.senses.split(",")):
        model, tok, rep = load(d, dev)
        dest = a.output / f"{rep['name']}.npy"
        if dest.exists():
            raise SystemExit(f"MNLM_PILOT_EXISTS: {dest}")
        t0 = time.perf_counter()
        X = embed(model, tok, texts, dev)
        if not np.isfinite(X).all():
            raise SystemExit(f"MNLM_SLOT_NONFINITE: {rep['name']}")
        np.save(dest, X.astype(np.float16))
        rec[rep["name"]] = {"dir": str(d), "weights_sha256": rep["weights_sha256"], "tokenizer_sha256": rep["tokenizer_sha256"],
                            "dims": int(X.shape[1]), "seconds": time.perf_counter() - t0,
                            "slot_sha256": hashlib.sha256(dest.read_bytes()).hexdigest()}
        print(rep["name"], json.dumps({k: rec[rep["name"]][k] for k in ("dims", "seconds")}), flush=True)
    publish(a.output / f"manifest-commissioned-{int(time.time())}.json", {"schema": "commissioned-slots-v1", "atoms": len(texts),
        "atoms_sha256": hashlib.sha256((a.atoms / "atoms.parquet").read_bytes()).hexdigest(), "batch": BATCH, "senses": rec,
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})


if __name__ == "__main__":
    run_cli(main)
