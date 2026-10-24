"""Constellations for the text pilot: every atom read by K frozen embedders (#25 option 1).

Each instrument is a catalogue lens (lens/catalogue.py: repo, pooling, prefix, window) loaded from its local
snapshot, frozen, identified by the SHA-256 of its weight files. Runtime `batched-fixed-v1`: atoms are
processed in atom order in fixed blocks; inside a block they are ordered by length and cut into fixed
batches of 256, and a batch whose padded size exceeds TOKEN_BUDGET tokens is halved recursively, so a
replay of the same atom file reproduces the same batch geometry. That is a declared runtime,
so it is a different lens identity from the one-atom-per-forward Stage 3 runtime. Outputs are float32,
unit-normalised, stored as float16 (slot store). No CPU fallback. One GPU job: takes the host lease.
Cost: O(atoms x lenses) forwards; per lens the model loads once.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lens"))
from lens.catalogue import by_name
from lens.learned import snapshot_dir, weights_sha256
from lens.lenscommon import run_cli, take_lease
from senses.trajectory_population import publish

RUNTIME = "batched-fixed-v1"
BLOCK, BATCH = 8192, 256
TOKEN_BUDGET = 65536  # a batch whose padded tokens exceed this is halved (deterministically); smaller batches are unchanged


@torch.no_grad()
def embed(spec, texts, dtype, budget=TOKEN_BUDGET):
    snap = snapshot_dir(spec["repo"])
    tok = AutoTokenizer.from_pretrained(str(snap), trust_remote_code=False)
    if spec["pool"] == "last":
        tok.padding_side = "left"
    model = AutoModel.from_pretrained(str(snap), dtype=dtype, trust_remote_code=False).cuda().eval().requires_grad_(False)
    out = np.zeros((len(texts), model.config.hidden_size), np.float16)
    truncated = 0
    for b0 in range(0, len(texts), BLOCK):
        idx = list(range(b0, min(b0 + BLOCK, len(texts))))
        idx.sort(key=lambda i: (len(texts[i]), i))
        queue = [idx[j:j + BATCH] for j in range(0, len(idx), BATCH)]
        while queue:
            part = queue.pop(0)
            enc = tok([spec.get("prefix", "") + texts[i] for i in part], padding=True, truncation=True,
                      max_length=spec["max_len"], return_tensors="pt", return_length=True)
            if enc["input_ids"].numel() > budget and len(part) > 1:  # deterministic split of an oversized batch
                h_ = len(part) // 2
                queue[:0] = [part[:h_], part[h_:]]
                continue
            truncated += int((enc.pop("length") >= spec["max_len"]).sum())
            enc = {k: v.cuda() for k, v in enc.items() if k in ("input_ids", "attention_mask", "token_type_ids")}
            h = model(**enc).last_hidden_state.float()
            m = enc["attention_mask"].unsqueeze(-1).float()
            if spec["pool"] == "mean":
                v = (h * m).sum(1) / m.sum(1)
            elif spec["pool"] == "cls":
                v = h[:, 0]
            else:  # last (left-padded)
                v = h[:, -1]
            v = torch.nn.functional.normalize(v, dim=-1)
            if not torch.isfinite(v).all():
                raise SystemExit(f"MNLM_PILOT_NONFINITE {spec['name']}")
            out[part] = v.cpu().numpy().astype(np.float16)
    del model
    torch.cuda.empty_cache()
    return out, truncated, snap


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--atoms", type=Path, required=True, help="directory with atoms.parquet")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--lenses", required=True, help="comma-separated catalogue names")
    p.add_argument("--limit", type=int, default=0, help="first N atoms only (throughput probe)")
    p.add_argument("--dtype", choices=("float32", "float16"), default="float32")
    p.add_argument("--budgets", type=Path, help="JSON {lens: max padded tokens per batch}; default TOKEN_BUDGET")
    a = p.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("MNLM_PILOT_NO_GPU: a CPU run would be a different instrument")
    os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    take_lease("25", "text-pilot-measure")
    texts = pq.read_table(a.atoms / "atoms.parquet", columns=["text"]).column("text").to_pylist()
    if a.limit:
        texts = texts[:a.limit]
    a.output.mkdir(parents=True, exist_ok=True)
    dtype = getattr(torch, a.dtype)
    lenses = {}
    budgets = json.loads(a.budgets.read_bytes()) if a.budgets else {}
    for name in a.lenses.split(","):
        spec = by_name(name)
        if spec["kind"] != "dense":
            raise SystemExit(f"MNLM_PILOT_KIND: {name} is {spec['kind']}; this runtime reads dense lenses only")
        dest = a.output / f"{name}.npy"
        if dest.exists():
            raise SystemExit(f"MNLM_PILOT_EXISTS: {dest} (slots are immutable; use a new output)")
        t0 = time.perf_counter()
        budget = budgets.get(name, TOKEN_BUDGET)
        X, truncated, snap = embed(spec, texts, dtype, budget)
        np.save(dest, X)
        sha, files = weights_sha256(snap)
        lenses[name] = {"repo": spec["repo"], "revision": snap.name, "weights_sha256": sha, "group": spec.get("group"),
                        "pool": spec["pool"], "prefix": spec.get("prefix", ""), "max_len": spec["max_len"], "dims": int(X.shape[1]),
                        "truncated_atoms": truncated, "batch_token_budget": budget, "seconds": time.perf_counter() - t0,
                        "atoms_per_second": len(texts) / (time.perf_counter() - t0),
                        "slot_sha256": hashlib.sha256(dest.read_bytes()).hexdigest()}
        print(name, json.dumps({k: lenses[name][k] for k in ("dims", "truncated_atoms", "seconds", "atoms_per_second")}), flush=True)
    publish(a.output / f"manifest-{int(time.time())}.json", {"schema": "text-pilot-constellations-v1", "runtime": RUNTIME,
        "dtype": a.dtype, "block": BLOCK, "batch": BATCH, "atoms": len(texts), "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__, "atoms_sha256": hashlib.sha256((a.atoms / "atoms.parquet").read_bytes()).hexdigest(),
        "lenses": lenses, "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})


if __name__ == "__main__":
    run_cli(main)
