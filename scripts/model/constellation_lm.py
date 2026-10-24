"""MNLM pilot, part 1: the constellation predictor with one decoder per sense (#25 option 1).

Input: documents as ordered atoms; each atom carries K frozen-embedder slots (measure_pilot.py). Each slot
enters through its own adapter (LayerNorm + Linear + sense-type embedding) as one typed sense token, so an
atom is K tokens and nothing is pooled or concatenated. A causal transformer runs over the K*T tokens with a
block-causal mask (every token of atom t sees every token of atoms <= t). Sense k's decoder reads the
output of sense k's token at atom t and predicts sense k's slot at atom t+1 (unit vector). Loss per slot:
in-batch InfoNCE (temperature TAU; negatives are the true next slots of other positions) plus (1 - cos).
Trained from scratch; the embedders never receive gradients (their outputs are fixed files).

Evaluation on a held-out document split (hash of the document id; never trained on):
  per sense: mean cosine and retrieval rank of the true next slot among the batch's next slots, against
  two baselines: copy-the-current-atom's slot and the training mean slot;
  ablation: each sense's input tokens masked out at evaluation, the change in every sense's loss.
Cost: O(steps * B * (K*T)^2 * H) attention; data in RAM, O(N * sum d_k) bytes as float16.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import pyarrow.parquet as pq
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lens.lenscommon import run_cli, take_lease
from senses.trajectory_population import publish

TAU = 0.05
SPLIT = {"train": (0, 90), "dev": (90, 95), "test": (95, 100)}  # percent buckets of sha256(doc)


def require_sequence_authority(atoms_dir, legacy=False):
    """Old filtered positions do not certify real next-sentence adjacency."""
    report = json.loads((atoms_dir / "manifest.json").read_bytes())
    if report.get("rules", {}).get("contiguous") is not True and not legacy:
        raise SystemExit("MNLM_PILOT_FILTERED_SEQUENCE: source gaps were renumbered; use the "
                         "lossless constellation corpus/native-window path. Historical diagnostic "
                         "reproduction requires --legacy-filtered-diagnostic; it is not admission.")


def bucket(doc: str) -> int:
    return int(hashlib.sha256(doc.encode()).hexdigest()[:8], 16) % 100


def text_hash(texts) -> np.ndarray:
    """60-bit hash of each atom's whitespace-stripped, lower-cased text: exact-duplicate detection across documents
    (2.8 % of text-pilot-v2 dev atoms recur verbatim in training documents; weave audit 2026-10-19, #25)."""
    return np.array([int(hashlib.sha1(t.strip().lower().encode()).hexdigest()[:15], 16) for t in texts], np.int64)


class Predictor(nn.Module):
    def __init__(self, dims, width, layers, heads, max_atoms, dropout=0.0, residual=False):
        super().__init__()
        self.K, self.residual = len(dims), residual
        self.gain = nn.Parameter(torch.full((len(dims),), 0.1))  # residual: s_t + gain * delta (starts at the copy baseline)
        self.adapters = nn.ModuleList([nn.Sequential(nn.LayerNorm(d), nn.Linear(d, width)) for d in dims])
        self.sense = nn.Parameter(torch.randn(self.K, width) * 0.02)
        self.pos = nn.Parameter(torch.randn(max_atoms, width) * 0.02)
        block = nn.TransformerEncoderLayer(width, heads, 4 * width, dropout=dropout, batch_first=True, norm_first=True, activation="gelu")
        self.trunk = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.decoders = nn.ModuleList([nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, d)) for d in dims])

    def forward(self, slots, absent=None):
        """slots: list of K tensors [B, T, d_k]; absent: optional [K] bool (sense masked out). Returns K predictions [B, T, d_k]."""
        B, T = slots[0].shape[:2]
        if absent is not None:  # a masked sense contributes no reading at all, not even to its own token
            slots = [torch.zeros_like(s) if bool(absent[k]) else s for k, s in enumerate(slots)]
        toks = torch.stack([self.adapters[k](slots[k]) + self.sense[k] for k in range(self.K)], 2)  # B,T,K,H
        toks = toks + self.pos[:T][None, :, None]
        x = toks.reshape(B, T * self.K, -1)
        atom = torch.arange(T, device=x.device).repeat_interleave(self.K)
        mask = atom[None, :] > atom[:, None]  # True = blocked: no token sees a later atom
        if absent is not None and bool(absent.any()):
            gone = absent.repeat(T)
            mask = mask | gone[None, :]
            mask[torch.arange(len(gone)), torch.arange(len(gone))] = False  # a token always sees itself (no empty rows)
        h = self.norm(self.trunk(x, mask=mask)).reshape(B, T, self.K, -1)
        if self.residual:
            return [F.normalize(slots[k] + self.gain[k] * self.decoders[k](h[:, :, k]), dim=-1) for k in range(self.K)]
        return [F.normalize(self.decoders[k](h[:, :, k]), dim=-1) for k in range(self.K)]


def self_excluded(sim, T1):
    """Candidate i-1 (the previous position's next) is the CURRENT atom: never a candidate for its own next."""
    i = torch.arange(sim.shape[0], device=sim.device); own = (i % T1) != 0
    sim = sim.clone(); sim[i[own], i[own] - 1] = float("-inf")
    return sim


def losses(pred, slots):
    """Per sense: InfoNCE over all (b, t<T-1) positions (own atom excluded) plus 1 - cos; returns (total, per-sense dict)."""
    out, total = {}, 0.0
    for k, (p, s) in enumerate(zip(pred, slots)):
        q = p[:, :-1].reshape(-1, p.shape[-1]); y = s[:, 1:].reshape(-1, s.shape[-1])
        logits = self_excluded(q @ y.T / TAU, s.shape[1] - 1)
        nce = F.cross_entropy(logits, torch.arange(len(q), device=q.device))
        cos = (q * y).sum(-1).mean()
        out[k] = (float(nce.detach()), float(cos.detach()))
        total = total + nce + (1 - cos)
    return total, out


class Data:
    def __init__(self, atoms_dir, slots_dir, lenses, window):
        t = pq.read_table(atoms_dir / "atoms.parquet", columns=["doc", "pos"]).to_pydict()
        self.doc, self.pos = t["doc"], np.array(t["pos"])
        self.X = [torch.from_numpy(np.load(slots_dir / f"{n}.npy")) for n in lenses]  # float16, unit rows
        assert all(len(x) == len(self.doc) for x in self.X), "MNLM_PILOT_ROWS: slot files differ in length"
        self.window = window
        starts = {}
        for i, (d, p) in enumerate(zip(self.doc, self.pos)):
            if p == 0:
                starts[d] = i
        runs = {}
        for d, i in starts.items():
            n = 1
            while i + n < len(self.doc) and self.doc[i + n] == d and self.pos[i + n] == n:
                n += 1
            runs[d] = (i, n)
        self.windows = {s: [] for s in SPLIT}
        train_rows = np.array([SPLIT["train"][0] <= bucket(d) < SPLIT["train"][1] for d in self.doc])
        # Frozen embedders are anisotropic (mean cosine ~0.8): centre each slot on its TRAINING-split mean and
        # re-normalise, a fixed label-free transform, so cosine and retrieval read content, not the common direction.
        self.centre = [x[torch.from_numpy(train_rows)].float().mean(0) for x in self.X]
        for k, c in enumerate(self.centre):  # in place, row chunks: peak RAM ~ one extra sense, not a second panel
            x = self.X[k]
            for j in range(0, len(x), 262144):  # row-wise ops, so identical to the whole-array transform
                x[j:j + 262144] = F.normalize(x[j:j + 262144].float() - c, dim=-1).half()
        for d, (i, n) in runs.items():
            b = bucket(d)
            split = next(k for k, (lo, hi) in SPLIT.items() if lo <= b < hi)
            for w in range(i, i + max(1, n - window + 1), window):  # non-overlapping windows inside a document
                if min(window, i + n - w) >= 4:
                    self.windows[split].append((w, min(window, i + n - w)))

    def batch(self, split, rows, device):
        T = min(L for _, L in rows)
        idx = torch.tensor([[w + t for t in range(T)] for w, _ in rows])
        return [x[idx].to(device, torch.float32) for x in self.X]


@torch.no_grad()
def evaluate(model, data, split, device, batch, lenses, rng, ablate=True):
    model.eval()
    wins = data.windows[split]
    order = rng.permutation(len(wins))[: min(len(wins), 64 * batch)]
    mean = [F.normalize(x[::max(1, len(x) // 50000)].float().mean(0), dim=0).to(device) for x in data.X]  # stride: rows are domain-ordered
    acc = {n: {"nce": 0.0, "cos": 0.0, "copy_cos": 0.0, "mean_cos": 0.0, "rank1": 0.0} for n in lenses}
    abl = {m: {n: 0.0 for n in lenses} for m in lenses} if ablate else None
    nb = 0
    for j in range(0, len(order), batch):
        rows = [wins[i] for i in order[j:j + batch]]
        if len(rows) < 8:
            break
        s = data.batch(split, rows, device)
        pred = model(s)
        _, per = losses(pred, s)
        for k, n in enumerate(lenses):
            q = pred[k][:, :-1].reshape(-1, pred[k].shape[-1]); y = s[k][:, 1:].reshape(-1, s[k].shape[-1])
            c = s[k][:, :-1].reshape(-1, s[k].shape[-1])
            acc[n]["nce"] += per[k][0]; acc[n]["cos"] += per[k][1]
            acc[n]["copy_cos"] += float((c * y).sum(-1).mean()); acc[n]["mean_cos"] += float((y @ mean[k]).mean())
            T1 = s[k].shape[1] - 1
            acc[n]["rank1"] += float((self_excluded(q @ y.T, T1).argmax(1) == torch.arange(len(q), device=device)).float().mean())
            acc[n]["copy_rank1"] = acc[n].get("copy_rank1", 0.0) + float((self_excluded(c @ y.T, T1).argmax(1) == torch.arange(len(q), device=device)).float().mean())
        if ablate:
            for m, gone in enumerate(lenses):
                absent = torch.zeros(len(lenses), dtype=torch.bool, device=device); absent[m] = True
                _, pa = losses(model(s, absent), s)
                for k, n in enumerate(lenses):
                    abl[gone][n] += pa[k][0] - per[k][0]
        nb += 1
    model.train()
    for n in lenses:
        for k in acc[n]:
            acc[n][k] /= max(nb, 1)
    if ablate:
        for m in abl:
            for n in abl[m]:
                abl[m][n] /= max(nb, 1)
    return {"batches": nb, "per_sense": acc, "ablation_nce_increase": abl,
            "chance_rank1": 1.0 / max(1, (batch * (data.window - 1)))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--atoms", type=Path, required=True)
    p.add_argument("--slots", type=Path, required=True)
    p.add_argument("--lenses", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--layers", type=int, default=8)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--window", type=int, default=24)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=25)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--residual", action="store_true")
    p.add_argument("--eval-every", type=int, default=0, help="dev check every N steps; the best state (mean dev NCE) is kept")
    p.add_argument("--legacy-filtered-diagnostic", action="store_true")
    a = p.parse_args()
    require_sequence_authority(a.atoms, a.legacy_filtered_diagnostic)
    if not torch.cuda.is_available():
        raise SystemExit("MNLM_PILOT_NO_GPU")
    take_lease("25", "constellation-predictor")
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    a.output.mkdir(parents=True, exist_ok=False)
    lenses = a.lenses.split(",")
    data = Data(a.atoms, a.slots, lenses, a.window)
    dims = [x.shape[1] for x in data.X]
    dev = "cuda"
    model = Predictor(dims, a.width, a.layers, a.heads, a.window, a.dropout, a.residual).to(dev)
    best, best_state, checks = float("inf"), None, []
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05, betas=(0.9, 0.95))
    warm = 500
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1, s / a.steps)))))
    log = (a.output / "train.jsonl").open("w")
    started = time.perf_counter()
    train = data.windows["train"]
    print(json.dumps({"windows": {k: len(v) for k, v in data.windows.items()}, "params": sum(q.numel() for q in model.parameters())}), flush=True)
    for step in range(a.steps):
        rows = [train[i] for i in rng.integers(0, len(train), a.batch)]
        s = data.batch("train", rows, dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = model(s)
        loss, per = losses([x.float() for x in pred], s)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if not torch.isfinite(loss):
            raise SystemExit("MNLM_PILOT_NONFINITE_LOSS")
        if step % 200 == 0 or step == a.steps - 1:
            rec = {"step": step, "loss": float(loss), "cos": {lenses[k]: round(v[1], 4) for k, v in per.items()}, "s": round(time.perf_counter() - started)}
            log.write(json.dumps(rec) + "\n"); log.flush(); print(json.dumps(rec), flush=True)
        if a.eval_every and (step + 1) % a.eval_every == 0:
            chk = evaluate(model, data, "dev", dev, a.batch, lenses, np.random.default_rng(1), ablate=False)
            m = float(np.mean([v["nce"] for v in chk["per_sense"].values()]))
            checks.append({"step": step + 1, "dev_nce": m}); print(json.dumps(checks[-1]), flush=True)
            if m < best:
                best, best_state = m, {k: v.detach().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), a.output / "predictor.pt")
    ev = evaluate(model, data, "dev", dev, a.batch, lenses, np.random.default_rng(0))
    publish(a.output / "report.json", {"schema": "constellation-predictor-v2", "dev_checks": checks,
        "selection": "best mean dev NCE (dev rng 1); final eval on dev rng 0" if checks else "last step", "split": "dev", "lenses": lenses, "dims": dims,
        "config": vars(a) | {"atoms": str(a.atoms), "slots": str(a.slots), "output": str(a.output)},
        "params": sum(q.numel() for q in model.parameters()), "windows": {k: len(v) for k, v in data.windows.items()},
        "eval": ev, "test_split_untouched": True, "wall_seconds": time.perf_counter() - started,
        "predictor_sha256": hashlib.sha256((a.output / "predictor.pt").read_bytes()).hexdigest(),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    print(json.dumps({n: {k: round(v, 4) for k, v in ev["per_sense"][n].items()} for n in lenses}, indent=1))


if __name__ == "__main__":
    run_cli(main)
