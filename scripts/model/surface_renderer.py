"""MNLM pilot, part 2: the byte-level surface renderer, conditioned on a constellation through one branch per sense.

Each sense slot s_k becomes its own typed memory token (its own LayerNorm + Linear + sense-type embedding; never
pooled or concatenated). Every decoder block cross-attends over the K sense tokens, so each byte position
chooses which senses to read, then a feed-forward. Vocabulary: 256 bytes + BOS + EOS;
the model has no tokenizer of its own. Trained from scratch by byte cross-entropy on (constellation -> atom text)
with the TRUE slots, plus Gaussian noise on each slot (re-normalised) so it can render from a predictor's noisy
slots (spec J3 solution 5). Evaluation on the dev document split: byte cross-entropy with true slots and with the
slots of a different atom (the conditioning gap; if it is near zero the renderer ignores the constellation), and
greedy renders kept for the guard step.
Cost: O(steps * B * L^2 * H) self-attention + O(steps * B * L * K * H) cross-attention.
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
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lens.lenscommon import run_cli, take_lease
from model.constellation_lm import SPLIT, bucket
from senses.trajectory_population import publish

BOS, EOS, VOCAB, MAXLEN = 256, 257, 258, 640


class Block(nn.Module):
    def __init__(self, K, width, heads):
        super().__init__()
        self.n1 = nn.LayerNorm(width); self.self_attn = nn.MultiheadAttention(width, heads, batch_first=True)
        # One cross-attention over the K typed sense memories: each byte position chooses which senses to read.
        # (A separate attention per sense over ONE memory vector is a constant projection, not attention.)
        self.cn = nn.LayerNorm(width); self.cross = nn.MultiheadAttention(width, heads, batch_first=True)
        self.n2 = nn.LayerNorm(width); self.ff = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))

    def forward(self, x, mem, causal, pad):
        q = self.n1(x)
        x = x + self.self_attn(q, q, q, attn_mask=causal, key_padding_mask=pad, need_weights=False)[0]
        q = self.cn(x)
        x = x + self.cross(q, mem, mem, need_weights=False)[0]
        return x + self.ff(self.n2(x))


class Renderer(nn.Module):
    def __init__(self, dims, width, layers, heads):
        super().__init__()
        self.K = len(dims)
        self.mem = nn.ModuleList([nn.Sequential(nn.LayerNorm(d), nn.Linear(d, width)) for d in dims])
        self.sense = nn.Parameter(torch.randn(self.K, width) * 0.02)
        self.emb = nn.Embedding(VOCAB, width); self.pos = nn.Parameter(torch.randn(MAXLEN + 2, width) * 0.02)
        self.blocks = nn.ModuleList([Block(self.K, width, heads) for _ in range(layers)])
        self.norm = nn.LayerNorm(width); self.out = nn.Linear(width, VOCAB)

    def forward(self, slots, ids, pad):
        mem = torch.stack([self.mem[k](slots[k]) + self.sense[k] for k in range(self.K)], 1)  # B, K, H: one typed token per sense
        L = ids.shape[1]
        x = self.emb(ids) + self.pos[:L]
        causal = torch.ones(L, L, dtype=torch.bool, device=ids.device).triu(1)
        for b in self.blocks:
            x = b(x, mem, causal, pad)
        return self.out(self.norm(x))

    @torch.no_grad()
    def render(self, slots, max_len=MAXLEN):
        B = slots[0].shape[0]
        ids = torch.full((B, 1), BOS, dtype=torch.long, device=slots[0].device)
        done = torch.zeros(B, dtype=torch.bool, device=ids.device)
        for _ in range(max_len):
            nxt = self(slots, ids, None)[:, -1].argmax(-1)
            nxt = torch.where(done, torch.full_like(nxt, EOS), nxt)
            ids = torch.cat([ids, nxt[:, None]], 1); done |= nxt == EOS
            if bool(done.all()):
                break
        out = []
        for row in ids[:, 1:].tolist():
            row = row[: row.index(EOS)] if EOS in row else row
            out.append(bytes(b for b in row if b < 256).decode("utf-8", "replace"))
        return out


def encode(texts, device):
    rows = [[BOS] + list(t.encode("utf-8")[:MAXLEN]) + [EOS] for t in texts]
    L = max(map(len, rows))
    ids = torch.full((len(rows), L), EOS, dtype=torch.long)
    pad = torch.ones((len(rows), L), dtype=torch.bool)
    for i, r in enumerate(rows):
        ids[i, :len(r)] = torch.tensor(r); pad[i, :len(r)] = False
    return ids.to(device), pad.to(device)


def ce(model, slots, texts, device):
    ids, pad = encode(texts, device)
    logits = model(slots, ids[:, :-1], pad[:, :-1])
    tgt = ids[:, 1:].masked_fill(pad[:, 1:], -100)
    return F.cross_entropy(logits.reshape(-1, VOCAB).float(), tgt.reshape(-1), ignore_index=-100)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--atoms", type=Path, required=True)
    p.add_argument("--slots", type=Path, required=True)
    p.add_argument("--lenses", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--layers", type=int, default=8)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--noise", type=float, default=0.3, help="slot noise std relative to a unit vector's per-dim scale")
    p.add_argument("--seed", type=int, default=25)
    a = p.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("MNLM_PILOT_NO_GPU")
    take_lease("25", "surface-renderer")
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    a.output.mkdir(parents=True, exist_ok=False)
    lenses = a.lenses.split(",")
    t = pq.read_table(a.atoms / "atoms.parquet", columns=["doc", "text"]).to_pydict()
    texts = t["text"]; split = np.array([next(k for k, (lo, hi) in SPLIT.items() if lo <= bucket(d) < hi) for d in t["doc"]])
    X = [torch.from_numpy(np.load(a.slots / f"{n}.npy")) for n in lenses]
    assert all(len(x) == len(texts) for x in X), "MNLM_PILOT_ROWS"
    tr, dv = np.flatnonzero(split == "train"), np.flatnonzero(split == "dev")
    # same fixed transform as the predictor: centre on the training-split mean, re-normalise
    X = [F.normalize(x.float() - x[torch.from_numpy(tr)].float().mean(0), dim=-1).half() for x in X]
    dev = "cuda"
    model = Renderer([x.shape[1] for x in X], a.width, a.layers, a.heads).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 500) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1, s / a.steps)))))
    take = lambda idx, noise=0.0: [F.normalize(x[idx].to(dev, torch.float32) + noise * torch.randn(len(idx), x.shape[1], device=dev) / math.sqrt(x.shape[1]), dim=-1) for x in X]
    log = (a.output / "train.jsonl").open("w"); started = time.perf_counter()
    print(json.dumps({"train": len(tr), "dev": len(dv), "params": sum(q.numel() for q in model.parameters())}), flush=True)
    for step in range(a.steps):
        idx = rng.choice(tr, a.batch, replace=False)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = ce(model, take(idx, a.noise), [texts[i] for i in idx], dev)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        if not torch.isfinite(loss):
            raise SystemExit("MNLM_PILOT_NONFINITE_LOSS")
        if step % 200 == 0 or step == a.steps - 1:
            rec = {"step": step, "bits_per_byte": float(loss) / math.log(2), "s": round(time.perf_counter() - started)}
            log.write(json.dumps(rec) + "\n"); log.flush(); print(json.dumps(rec), flush=True)
    torch.save(model.state_dict(), a.output / "renderer.pt")
    model.eval(); erng = np.random.default_rng(0)
    own, other = [], []
    with torch.no_grad():
        for _ in range(40):
            idx = erng.choice(dv, a.batch, replace=False); shuf = erng.permutation(idx)
            own.append(float(ce(model, take(idx), [texts[i] for i in idx], dev)))
            other.append(float(ce(model, take(shuf), [texts[i] for i in idx], dev)))
        sample = erng.choice(dv, 64, replace=False)
        renders = model.render(take(sample))
    rows = [{"atom_index": int(i), "text": texts[i], "render": r} for i, r in zip(sample, renders)]
    (a.output / "renders-dev.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    exact = sum(r["text"] == r["render"] for r in rows) / len(rows)
    publish(a.output / "report.json", {"schema": "surface-renderer-v1", "lenses": lenses, "split": "dev",
        "bits_per_byte_true_slots": float(np.mean(own)) / math.log(2), "bits_per_byte_other_atom_slots": float(np.mean(other)) / math.log(2),
        "conditioning_gap_bits_per_byte": (float(np.mean(other)) - float(np.mean(own))) / math.log(2),
        "greedy_exact_match_dev64": exact, "config": vars(a) | {"atoms": str(a.atoms), "slots": str(a.slots), "output": str(a.output)},
        "params": sum(q.numel() for q in model.parameters()), "renderer_sha256": hashlib.sha256((a.output / "renderer.pt").read_bytes()).hexdigest(),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "wall_seconds": time.perf_counter() - started})
    print(json.dumps({"bpb_true": np.mean(own) / math.log(2), "bpb_other": np.mean(other) / math.log(2), "exact": exact}))


if __name__ == "__main__":
    run_cli(main)
