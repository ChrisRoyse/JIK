"""Commission one sense (commission-v1, #12/#25; docs/senses_of_a_machine.txt P1): an embedder trained from
scratch on ONE corpus's associations, then frozen.

P1: training compresses a corpus's web of co-occurrence into a geometry. The objective here is that statement
made literal: two atoms that co-occur (same document, within +-WINDOW positions) are pulled together, atoms from
other documents in the batch are pushed apart (symmetric InfoNCE, temperature TAU). Nothing else is learned, so
the sense's perspective is exactly its corpus.
  tokenizer  byte-level BPE trained on this corpus only (VOCAB), so even segmentation is corpus-relative;
  encoder    pre-LN transformer, LAYERS x WIDTH, HEADS heads, MAX_LEN tokens, mean-pooled, unit-normalised;
  selection  validation documents (held out by hash): in-batch neighbour rank-1 every EVAL steps; best kept.
Output: model.pt, tokenizer.json, config.json, report.json (corpus files and hashes, atoms hash, code hash,
weights sha256 = the sense's identity). After this the weights never change (D3). GPU; one lease.
Cost: tokenise once O(atoms); STEPS x BATCH pairs of forwards. Memory: per-layer activation checkpointing
(measured: without it BATCH 1024 x MAX_LEN 128 x 6 layers OOMs at 31.4 GB, chain3l.log 2026-10-19T02:55Z).
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
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lens.lenscommon import run_cli, take_lease
from senses.trajectory_population import publish

VOCAB, MAX_LEN, WIDTH, LAYERS, HEADS = 32000, 128, 512, 6, 8
WINDOW, TAU, EVAL = 2, 0.05, 2000
PAD = 0
GROUPS = 8  # length-sorted sub-batches per contrastive batch (encode)


class Encoder(nn.Module):
    def __init__(self, vocab=VOCAB, width=WIDTH, layers=LAYERS, heads=HEADS, max_len=MAX_LEN, dropout=0.1):
        super().__init__()
        self.tok = nn.Embedding(vocab, width, padding_idx=PAD)
        self.pos = nn.Embedding(max_len, width)
        layer = nn.TransformerEncoderLayer(width, heads, 4 * width, dropout, activation="gelu", batch_first=True, norm_first=True)
        self.body = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)

    def forward(self, ids):
        mask = ids != PAD
        h = self.tok(ids) + self.pos(torch.arange(ids.shape[1], device=ids.device))[None]
        pad = ~mask
        for layer in self.body.layers:  # per-layer activation checkpointing: same objective and batch, bounded memory
            if self.training and torch.is_grad_enabled():
                h = checkpoint(layer, h, None, pad, use_reentrant=False)
            else:
                h = layer(h, src_key_padding_mask=pad)
        h = self.norm(h)
        m = mask.unsqueeze(-1).to(h.dtype)
        return F.normalize((h * m).sum(1) / m.sum(1).clamp(min=1), dim=-1)


def train_tokenizer(texts, path):
    from tokenizers import ByteLevelBPETokenizer
    tok = ByteLevelBPETokenizer()
    tok.train_from_iterator(texts, vocab_size=VOCAB, min_frequency=2, special_tokens=["<pad>", "<unk>"])
    tok.save(str(path))
    return tok


def encode_all(tok, texts):
    ids = []
    for i in range(0, len(texts), 200_000):  # bounded memory: Encoding objects are large
        ids += [e.ids[:MAX_LEN] for e in tok.encode_batch(texts[i:i + 200_000])]
    lens = np.array([len(x) for x in ids], np.int32)
    flat = np.zeros(int(lens.sum()), np.int32)
    off = np.r_[0, np.cumsum(lens)]
    for i, x in enumerate(ids):
        flat[off[i]:off[i + 1]] = x
    return flat, off


def batch_ids(flat, off, rows, device):
    L = int(max(off[r + 1] - off[r] for r in rows))
    out = np.zeros((len(rows), max(L, 1)), np.int64)
    for j, r in enumerate(rows):
        out[j, : off[r + 1] - off[r]] = flat[off[r]:off[r + 1]]
    return torch.from_numpy(out).to(device, non_blocking=True)


def encode(model, flat, off, rows, device, groups=GROUPS):
    """Embed rows in length-sorted groups, each padded only to its own longest atom, then restore the input order.
    The encoder masks padding in attention and pooling, so each embedding is the same as in one padded batch, and
    the contrastive batch (pairs, in-batch negatives, loss) is unchanged; measured: one 1,024 batch padded to its
    longest atom wastes 70-88 % of token positions on the commission corpora (padcheck 2026-10-19, #25)."""
    rows = np.asarray(rows); order = np.argsort(off[rows + 1] - off[rows], kind="stable"); g = -(-len(rows) // groups)
    out = torch.cat([model(batch_ids(flat, off, rows[order[i:i + g]], device)) for i in range(0, len(rows), g)])
    return out[torch.from_numpy(np.argsort(order)).to(device)]


def neighbours(doc, pos):
    """For each atom, the row range of its document (atoms are stored in document order)."""
    start = np.zeros(len(doc), np.int64); end = np.zeros(len(doc), np.int64)
    b = np.flatnonzero(np.r_[True, doc[1:] != doc[:-1], True])
    for s, e in zip(b[:-1], b[1:]):
        start[s:e] = s; end[s:e] = e
    return start, end


def sample_pairs(rng, rows, start, end, n):
    a = rows[rng.integers(0, len(rows), n)]
    lo, hi = np.maximum(start[a], a - WINDOW), np.minimum(end[a] - 1, a + WINDOW)
    ok = hi > lo
    a, lo, hi = a[ok], lo[ok], hi[ok]
    p = lo + rng.integers(0, (hi - lo), len(a))
    p = p + (p >= a)  # skip the anchor itself
    return a, p


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--atoms", type=Path, required=True, help="build_commission_atoms output (atoms.parquet, manifest.json)")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=25)
    ap.add_argument("--device", default="cuda", help="cpu only for a named smoke test; a sense of record is trained on cuda")
    a = ap.parse_args()
    dev = a.device
    if dev == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("MNLM_COMMISSION_NO_GPU")
        take_lease("25", f"commission-{a.name}")
    started = time.perf_counter()
    a.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    t = pq.read_table(a.atoms / "atoms.parquet")
    texts = t.column("text").to_pylist(); doc = t.column("doc").to_numpy(); split = np.array(t.column("split").to_pylist())
    sample = [texts[i] for i in rng.choice(np.flatnonzero(split == "train"), min(1_000_000, int((split == "train").sum())), replace=False)]
    tok = train_tokenizer(sample, a.output / "tokenizer.json")
    flat, off = encode_all(tok, texts)
    start, end = neighbours(doc, None)
    tr, va = np.flatnonzero(split == "train"), np.flatnonzero(split == "val")
    model = Encoder().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 1000) * 0.5 * (1 + np.cos(np.pi * min(1.0, s / a.steps))))
    vrng = np.random.default_rng(1)
    vpairs = [sample_pairs(vrng, va, start, end, a.batch) for _ in range(8)]
    best, best_state, log = -1.0, None, []

    def val():
        model.eval(); acc = []
        with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16):
            for x, y in vpairs:
                q, k = encode(model, flat, off, x, dev).float(), encode(model, flat, off, y, dev).float()
                acc.append(float(((q @ k.T).argmax(1) == torch.arange(len(q), device=dev)).float().mean()))
        model.train()
        return float(np.mean(acc))

    for step in range(a.steps):
        x, y = sample_pairs(rng, tr, start, end, a.batch)
        with torch.autocast(dev, dtype=torch.bfloat16):
            q, k = encode(model, flat, off, x, dev), encode(model, flat, off, y, dev)
            logits = (q.float() @ k.float().T) / TAU
            lab = torch.arange(len(q), device=dev)
            loss = 0.5 * (F.cross_entropy(logits, lab) + F.cross_entropy(logits.T, lab))
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        if not torch.isfinite(loss):
            raise SystemExit(f"MNLM_COMMISSION_NONFINITE_LOSS at step {step}")
        if step % 200 == 0:
            print(json.dumps({"step": step, "loss": round(float(loss), 4), "s": round(time.perf_counter() - started)}), flush=True)
        if (step + 1) % EVAL == 0 or step == a.steps - 1:
            v = val(); log.append({"step": step + 1, "loss": float(loss), "val_rank1": v, "s": round(time.perf_counter() - started)})
            print(json.dumps(log[-1]), flush=True)
            if v > best:
                best, best_state = v, {kk: vv.detach().clone() for kk, vv in model.state_dict().items()}
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), a.output / "model.pt")
    cfg = {"vocab": VOCAB, "max_len": MAX_LEN, "width": WIDTH, "layers": LAYERS, "heads": HEADS, "pool": "mean", "norm": "unit",
           "objective": f"co-occurrence InfoNCE, same document within +-{WINDOW} atoms, tau {TAU}, in-batch negatives"}
    (a.output / "config.json").write_text(json.dumps(cfg, indent=1, sort_keys=True))
    man = json.loads((a.atoms / "manifest.json").read_bytes())
    publish(a.output / "report.json", {"schema": "commissioned-sense-v1", "name": a.name, "config": cfg, "train": vars(a) | {"atoms": str(a.atoms), "output": str(a.output)},
        "corpus_files": man["files"], "atoms_sha256": man["atoms_sha256"], "train_atoms": int(len(tr)), "val_atoms": int(len(va)),
        "params": sum(p.numel() for p in model.parameters()), "log": log, "best_val_rank1": best, "chance_val_rank1": 1.0 / a.batch,
        "weights_sha256": hashlib.sha256((a.output / "model.pt").read_bytes()).hexdigest(),
        "tokenizer_sha256": hashlib.sha256((a.output / "tokenizer.json").read_bytes()).hexdigest(),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "device": dev, "wall_seconds": time.perf_counter() - started})
    print(json.dumps({"name": a.name, "best_val_rank1": best}))


if __name__ == "__main__":
    run_cli(main)
