"""MNLM recognition codebook: named regions of constellation space (spec §15 semantic-ID term, §25.4; paper
"The Senses of a Machine" §8.2: synthesis of recognition, the schema).

The pilot's predictor regressed raw slots and its generations failed the guard (0.0 % accept, #25): it
predicted the mean of a multimodal next meaning, a point off the manifold of real constellations. This
builds the discrete layer the spec names: a hierarchical semantic identifier sigma(x) = (c_1, ..., c_L)
per atom, so the predictor can recognise WHICH region comes next (a proper likelihood, one mode) before it
places the slots inside it.

Construction (frozen, derived, hash-pinned; fitted on the training split only):
  1. per sense k: PCA of the centred slot to R dims (the quantiser's own latent, not the slot; no-flatten
     holds at the record), scaled so every sense's latent has unit total variance (no sense dominates);
  2. the K latents side by side form the quantiser input; residual k-means, L levels of C codes
     (level 1 = coarse region, the "category"; the last level = fine identity);
  3. schema: for every level-1 region and every sense, the unit mean of the member atoms' centred slots
     (how the region appears in each sense's form).
Reported: code usage (perplexity), within-region cosine of atoms to their schema on train and dev, next-atom
code predictability baselines on dev (unigram, copy-current, bigram from train), and NMI(level-1, source
domain) as a sanity check that regions are about content. Nothing here is tuned toward a gate.
Cost: O(L * iters * N_fit * C * K*R) for k-means on the GPU, O(N * K*R * C) for assignment; N = 960,075.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lens.lenscommon import run_cli
from model.constellation_lm import SPLIT, Data, bucket
from senses.trajectory_population import publish


def kmeans(X, C, iters, gen):
    """Lloyd's algorithm with k-means++-style seeding on a sample; empty clusters re-seeded from the worst-fit points."""
    n = len(X)
    first = torch.randint(n, (1,), generator=gen, device=X.device).item()
    cent = [X[first]]
    d2 = ((X - X[first]) ** 2).sum(1)
    for _ in range(1, C):  # k-means++ seeding (D^2 sampling), O(n*C)
        i = torch.multinomial(d2 / d2.sum(), 1, generator=gen).item()
        cent.append(X[i]); d2 = torch.minimum(d2, ((X - X[i]) ** 2).sum(1))
    cent = torch.stack(cent)
    for _ in range(iters):
        a, dist = assign(X, cent)
        cnt = torch.bincount(a, minlength=C)
        new = torch.zeros_like(cent).index_add_(0, a, X) / cnt.clamp(min=1)[:, None]
        empty = torch.nonzero(cnt == 0).flatten()
        if len(empty):
            new[empty] = X[dist.topk(len(empty)).indices]
        cent = new
    return cent


def assign(X, cent, chunk=65536):
    a, dist = [], []
    cn = (cent ** 2).sum(1)
    for j in range(0, len(X), chunk):
        x = X[j:j + chunk]
        d = (x ** 2).sum(1, keepdim=True) - 2 * x @ cent.T + cn[None]
        m = d.min(1)
        a.append(m.indices); dist.append(m.values)
    return torch.cat(a), torch.cat(dist)


def nmi(a, b):
    a = np.unique(a, return_inverse=True)[1]; b = np.unique(b, return_inverse=True)[1]
    j = np.zeros((a.max() + 1, b.max() + 1)); np.add.at(j, (a, b), 1); j /= j.sum()
    pa, pb = j.sum(1), j.sum(0)
    h = lambda p: -(p[p > 0] * np.log(p[p > 0])).sum()
    nz = j > 0
    mi = (j[nz] * np.log(j[nz] / (pa[:, None] * pb[None])[nz])).sum()
    return float(mi / max(1e-12, math.sqrt(h(pa) * h(pb))))


class Codebook:
    """The frozen recognition index: encode(slots) -> codes; schema[k] = [C1, d_k] unit vectors."""

    def __init__(self, path, device):
        z = np.load(path)
        self.L, self.K = int(z["levels"]), int(z["senses"])
        self.mean = [torch.from_numpy(z[f"mean{k}"]).to(device) for k in range(self.K)]
        self.comp = [torch.from_numpy(z[f"comp{k}"]).to(device) for k in range(self.K)]
        self.scale = torch.from_numpy(z["scale"]).to(device)
        self.cent = [torch.from_numpy(z[f"cent{l}"]).to(device) for l in range(self.L)]
        self.schema = [torch.from_numpy(z[f"schema{k}"]).to(device).float() for k in range(self.K)]
        self.sizes = [len(c) for c in self.cent]

    def latent(self, slots):
        return torch.cat([((s.float() - self.mean[k]) @ self.comp[k]) / self.scale[k] for k, s in enumerate(slots)], 1)

    def encode(self, slots):
        r = self.latent(slots); out = []
        for c in self.cent:
            a, _ = assign(r, c); out.append(a); r = r - c[a]
        return torch.stack(out, 1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--atoms", type=Path, required=True)
    p.add_argument("--slots", type=Path, required=True)
    p.add_argument("--lenses", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--rank", type=int, default=32, help="PCA dims per sense")
    p.add_argument("--levels", type=int, default=2)
    p.add_argument("--codes", type=int, default=256)
    p.add_argument("--fit", type=int, default=400000, help="training atoms sampled to fit the quantiser")
    p.add_argument("--iters", type=int, default=25)
    p.add_argument("--seed", type=int, default=20261019)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    if a.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("MNLM_RECOGNITION_NO_GPU")
    started = time.perf_counter()
    a.output.mkdir(parents=True, exist_ok=False)
    lenses = a.lenses.split(","); dev = a.device
    data = Data(a.atoms, a.slots, lenses, 24)  # centred on the training-split mean, unit rows: the predictor's view
    domain = pq.read_table(a.atoms / "atoms.parquet", columns=["domain"]).column("domain").to_pylist()
    split = np.array([bucket(d) for d in data.doc])
    train = np.flatnonzero((split >= SPLIT["train"][0]) & (split < SPLIT["train"][1]))
    devrows = np.flatnonzero((split >= SPLIT["dev"][0]) & (split < SPLIT["dev"][1]))
    rng = np.random.default_rng(a.seed); gen = torch.Generator(device=dev).manual_seed(a.seed)
    fit = np.sort(rng.choice(train, min(a.fit, len(train)), replace=False))
    z = {"levels": a.levels, "senses": len(lenses)}
    scale = []
    for k, x in enumerate(data.X):  # 1. per-sense PCA on the fit sample
        s = x[torch.as_tensor(fit)].to(dev, torch.float32)
        m = s.mean(0); U, S, V = torch.pca_lowrank(s - m, q=a.rank, center=False, niter=4)
        comp = V[:, : a.rank]
        var = ((s - m) @ comp).var(0).sum()
        z[f"mean{k}"], z[f"comp{k}"] = m.cpu().numpy(), comp.cpu().numpy()
        scale.append(float(var.sqrt()))
    z["scale"] = np.array(scale, dtype=np.float32)
    book = {"mean": [torch.from_numpy(z[f"mean{k}"]).to(dev) for k in range(len(lenses))],
            "comp": [torch.from_numpy(z[f"comp{k}"]).to(dev) for k in range(len(lenses))]}

    def latent(rows):
        return torch.cat([((data.X[k][torch.as_tensor(rows)].to(dev, torch.float32) - book["mean"][k]) @ book["comp"][k]) / scale[k]
                          for k in range(len(lenses))], 1)
    r = latent(fit)
    explained = [float(((data.X[k][torch.as_tensor(fit[:50000])].to(dev, torch.float32) - book["mean"][k]) @ book["comp"][k]).var(0).sum()
                       / (data.X[k][torch.as_tensor(fit[:50000])].to(dev, torch.float32).var(0).sum())) for k in range(len(lenses))]
    for l in range(a.levels):  # 2. residual k-means
        c = kmeans(r, a.codes, a.iters, gen)
        z[f"cent{l}"] = c.cpu().numpy()
        r = r - c[assign(r, c)[0]]
    codes = []
    cents = [torch.from_numpy(z[f"cent{l}"]).to(dev) for l in range(a.levels)]
    N = len(data.doc)
    for j in range(0, N, 131072):
        rr = latent(np.arange(j, min(N, j + 131072))); out = []
        for c in cents:
            q, _ = assign(rr, c); out.append(q); rr = rr - c[q]
        codes.append(torch.stack(out, 1).cpu())
    codes = torch.cat(codes).numpy().astype(np.int16)
    c1 = torch.as_tensor(codes[:, 0].astype(np.int64))
    tr = torch.as_tensor(train)
    within = {}
    for k, x in enumerate(data.X):  # 3. schema per level-1 region per sense (training members only)
        acc = torch.zeros(a.codes, x.shape[1], device=dev)
        for j in range(0, len(train), 131072):
            rows = tr[j:j + 131072]
            acc.index_add_(0, c1[rows].to(dev), x[rows].to(dev, torch.float32))
        sch = F.normalize(acc, dim=-1)
        z[f"schema{k}"] = sch.half().cpu().numpy()
        cos = lambda rows: float((x[torch.as_tensor(rows)].to(dev, torch.float32) * sch[c1[torch.as_tensor(rows)].to(dev)]).sum(-1).mean())
        within[lenses[k]] = {"train": cos(train[::max(1, len(train) // 200000)]), "dev": cos(devrows)}  # stride: rows are domain-ordered
    np.savez(a.output / "codebook.npz", **z)
    np.save(a.output / "codes.npy", codes)
    # predictability of the next atom's region (dev, consecutive atoms of one document)
    doc, pos = data.doc, data.pos
    nxt = np.array([i for i in devrows if i + 1 < N and doc[i + 1] == doc[i] and pos[i + 1] == pos[i] + 1])
    trn = np.array([i for i in train if i + 1 < N and doc[i + 1] == doc[i] and pos[i + 1] == pos[i] + 1])
    stats = {}
    for l in range(a.levels):
        cl = codes[:, l].astype(np.int64)
        uni = np.bincount(cl[train], minlength=a.codes)
        big = np.zeros((a.codes, a.codes)); np.add.at(big, (cl[trn], cl[trn + 1]), 1)
        y = cl[nxt + 1]
        pu = uni / uni.sum()
        stats[f"level{l + 1}"] = {
            "perplexity_train": float(np.exp(-(pu[pu > 0] * np.log(pu[pu > 0])).sum())), "used": int((uni > 0).sum()),
            "dev_next_pairs": int(len(nxt)), "unigram_acc": float((y == uni.argmax()).mean()),
            "copy_current_acc": float((y == cl[nxt]).mean()), "bigram_acc": float((y == big.argmax(1)[cl[nxt]]).mean()),
            "nmi_with_domain": nmi(cl[train], np.array(domain, dtype=object)[train].astype(str))}
    fine = codes[:, 0].astype(np.int64) * a.codes + codes[:, 1].astype(np.int64) if a.levels > 1 else None
    publish(a.output / "report.json", {"schema": "recognition-codebook-v1", "lenses": lenses,
        "config": vars(a) | {"atoms": str(a.atoms), "slots": str(a.slots), "output": str(a.output)},
        "atoms": N, "fit_atoms": int(len(fit)), "pca_explained_variance": dict(zip(lenses, explained)),
        "within_region_cos_to_schema": within, "levels": stats,
        "fine_regions_used": int(len(np.unique(fine))) if fine is not None else None,
        "codebook_sha256": hashlib.sha256((a.output / "codebook.npz").read_bytes()).hexdigest(),
        "codes_sha256": hashlib.sha256((a.output / "codes.npy").read_bytes()).hexdigest(),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "wall_seconds": time.perf_counter() - started})
    print(json.dumps({"within": within, "levels": stats}, indent=1))


if __name__ == "__main__":
    run_cli(main)
