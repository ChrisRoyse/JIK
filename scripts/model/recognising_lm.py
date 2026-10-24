"""MNLM predictor v3: recognise, then place (spec §15 semantic-ID term, §25.3-25.4; "The Senses of a Machine"
§8.2: apprehension = the frozen slots, reproduction = block-causal attention among all sense tokens within and
across atoms, recognition = the next atom's region, schema = the region's appearance in each sense).

Predictor v2 regressed the next slots directly; its generations failed the guard on every window (#25,
guard-v1: 0.0 %), because a regression to the mean of a multimodal next meaning lands off the manifold of
real constellations. Here the model first predicts the next atom's semantic identifier (codes of
recognition.py, autoregressive over levels: a proper likelihood whose argmax is one mode), then each
sense's decoder places the slot conditioned on that identifier:
  s_hat_k(t+1) = unit( a_k * schema_k[c1] + b_k * s_k(t) + decoder_k([h_k(t), E(c(t+1))]) ).
Input per atom: K typed sense tokens plus one concept token (the atom's own codes), so history is read both
as intuitions and as recognised concepts. Embedders never receive gradients (fixed files); the codebook is a
frozen derived index (its sha is recorded).
Training: code cross-entropy per level (teacher-forced over levels) + per-sense InfoNCE (own atom excluded)
+ (1 - cos), the decoders conditioned on the TRUE next codes except a fraction --sampled of positions that use
the model's own argmax codes (exposure to its own recognition).
Evaluation (dev; test untouched): code accuracy per level against the codebook's unigram/copy/bigram
baselines; slot rank-1 and cosine with predicted codes (the generation path) and with true codes (the
ceiling if recognition were perfect); on-manifold score: max joint cosine of a predicted constellation to a
pool of real training constellations, against the same score for the true next atom (and for predictor v2
when --baseline is given).
With --weave (weave.py output; predictor v4, schema recognising-predictor-v2): reproduction before
recognition (paper §8 Table 1, count before compose). Each atom gains a reproduction token: its association
vote (the level-1 regions that followed its >= 2-sense neighbours in other training documents, normalised)
read through the concept embedding, plus its 66 within-atom agreement scalars; and the vote enters the level-1
logits as a log-prior with a learned gain that starts at 0 (training decides how far to trust it).
Cost: O(steps * B * (T*(K+1))^2 * H) attention, as v2 with one extra token per atom (two with --weave).
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
from model.constellation_lm import (SPLIT, Data, Predictor, bucket, losses, self_excluded,
                                    text_hash, require_sequence_authority)
from model.recognition import Codebook
from senses.trajectory_population import publish


class Recogniser(nn.Module):
    def __init__(self, dims, sizes, schema, width, layers, heads, max_atoms, dropout=0.0, n_agree=0):
        super().__init__()
        self.K, self.L, self.reproduce = len(dims), len(sizes), n_agree > 0
        self.adapters = nn.ModuleList([nn.Sequential(nn.LayerNorm(d), nn.Linear(d, width)) for d in dims])
        # K sense types + the concept token (+ the reproduction token with --weave)
        self.kind = nn.Parameter(torch.randn(self.K + 1 + self.reproduce, width) * 0.02)
        if self.reproduce:
            self.agree_in = nn.Sequential(nn.LayerNorm(n_agree), nn.Linear(n_agree, width))
            self.prior_gain = nn.Parameter(torch.zeros(()))
        self.code_in = nn.ModuleList([nn.Embedding(c, width) for c in sizes])
        self.pos = nn.Parameter(torch.randn(max_atoms, width) * 0.02)
        block = nn.TransformerEncoderLayer(width, heads, 4 * width, dropout=dropout, batch_first=True, norm_first=True, activation="gelu")
        self.trunk = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.code_out = nn.ModuleList([nn.Embedding(c, width) for c in sizes])
        self.heads = nn.ModuleList([nn.Sequential(nn.LayerNorm(width), nn.Linear(width, c)) for c in sizes])
        self.decoders = nn.ModuleList([nn.Sequential(nn.Linear(2 * width, width), nn.GELU(), nn.Linear(width, d)) for d in dims])
        self.mix = nn.Parameter(torch.tensor([[1.0, 0.5]] * self.K))  # a_k (schema), b_k (current slot)
        for k, s in enumerate(schema):
            self.register_buffer(f"schema{k}", s.float(), persistent=False)

    def schema(self, k):
        return getattr(self, f"schema{k}")

    def encode(self, slots, codes, rep=None):
        """slots: K x [B,T,d_k]; codes: [B,T,L] (each atom's own codes); rep: (vote [B,T,C1], agree [B,T,P]) with
        --weave. Returns h [B,T,K+1(+1),H]."""
        B, T = slots[0].shape[:2]
        sense = [self.adapters[k](slots[k]) + self.kind[k] for k in range(self.K)]
        concept = sum(self.code_in[l](codes[..., l]) for l in range(self.L)) + self.kind[self.K]
        extra = []
        if self.reproduce:
            vote, agree = rep
            extra = [vote @ self.code_in[0].weight + self.agree_in(agree) + self.kind[self.K + 1]]
        n = self.K + 1 + self.reproduce
        toks = torch.stack(sense + [concept] + extra, 2) + self.pos[:T][None, :, None]
        x = toks.reshape(B, T * n, -1)
        atom = torch.arange(T, device=x.device).repeat_interleave(n)
        mask = atom[None, :] > atom[:, None]
        return self.norm(self.trunk(x, mask=mask)).reshape(B, T, n, -1)

    def head(self, l, x, rep):
        out = self.heads[l](x)
        if l == 0 and self.reproduce:  # the association vote as a log-prior on the next region
            out = out + self.prior_gain * torch.log(rep[0].to(out.dtype) + 1e-4)
        return out

    def code_logits(self, h, cond, rep=None):
        """Level-l logits from the concept token plus the embeddings of levels < l (cond: [B,T,L] codes)."""
        hc, out, acc = h[:, :, self.K], [], 0
        for l in range(self.L):
            out.append(self.head(l, hc + acc, rep))
            acc = acc + self.code_out[l](cond[..., l])
        return out

    @torch.no_grad()
    def recognise(self, h, rep=None):
        """Argmax autoregressively over levels: the mode, one region, never a mean."""
        hc, acc, codes = h[:, :, self.K], 0, []
        for l in range(self.L):
            c = self.head(l, hc + acc, rep).argmax(-1); codes.append(c)
            acc = acc + self.code_out[l](c)
        return torch.stack(codes, -1)

    def place(self, h, slots, codes):
        """Slots at t+1 given recognised codes of t+1: schema of the region + current slot + learned delta."""
        cond = sum(self.code_out[l](codes[..., l]) for l in range(self.L))
        out = []
        for k in range(self.K):
            delta = self.decoders[k](torch.cat([h[:, :, k], cond], -1))
            base = self.mix[k, 0] * self.schema(k)[codes[..., 0]] + self.mix[k, 1] * slots[k]
            out.append(F.normalize(base + delta, dim=-1))
        return out


def next_codes(codes, guess):
    """Target codes for position t are the codes of atom t+1; the last position has none, so use the guess."""
    return torch.cat([codes[:, 1:], guess[:, -1:]], 1)


def code_loss(logits, codes):
    tgt = codes[:, 1:]
    return [F.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]), tgt[..., l].reshape(-1)) for l, lg in enumerate(logits)]


def association_votes(edges, w, succ, c1, C, chunk=131072):
    """Per atom, the normalised weighted vote of its association neighbours for their successors' level-1 regions."""
    N = len(edges); out = np.zeros((N, C), np.float16)
    for j in range(0, N, chunk):
        e = edges[j:j + chunk].astype(np.int64); ww = w[j:j + chunk].astype(np.float32)
        ok = e >= 0; s = np.where(ok, succ[np.where(ok, e, 0)], -1); ok &= s >= 0
        v = np.zeros((len(e), C), np.float32)
        r = np.broadcast_to(np.arange(len(e))[:, None], e.shape)
        np.add.at(v, (r[ok], c1[s[ok]]), ww[ok])
        out[j:j + chunk] = v / np.maximum(v.sum(1, keepdims=True), 1e-9)
    return out


class CodedData(Data):
    def __init__(self, atoms, slots, lenses, window, codes, weave=None):
        super().__init__(atoms, slots, lenses, window)
        self.codes = torch.from_numpy(np.load(codes).astype(np.int64))
        assert len(self.codes) == len(self.doc), "MNLM_RECOGNISE_ROWS: codes and atoms differ in length"
        # seen: the atom's exact text occurs in the training split (a dev target that a duplicated passage answers)
        th = text_hash(pq.read_table(atoms / "atoms.parquet", columns=["text"]).column("text").to_pylist())
        tr = np.array([SPLIT["train"][0] <= bucket(d) < SPLIT["train"][1] for d in self.doc])
        self.seen = torch.from_numpy(np.isin(th, th[tr]))
        self.vote = self.agree = None
        if weave is not None:
            succ = np.load(weave / "succ.npy").astype(np.int64)
            c1 = self.codes[:, 0].numpy(); C = int(c1.max()) + 1
            self.vote = torch.from_numpy(association_votes(np.load(weave / "edges.npy"), np.load(weave / "edge_w.npy"), succ, c1, C))
            self.agree = torch.from_numpy(np.load(weave / "agree.npy"))
            assert len(self.vote) == len(self.agree) == len(self.doc), "MNLM_RECOGNISE_ROWS: weave and atoms differ in length"

    def coded_batch(self, rows, device):
        T = min(L for _, L in rows)
        idx = torch.tensor([[w + t for t in range(T)] for w, _ in rows])
        rep = None if self.vote is None else (self.vote[idx].to(device, torch.float32), self.agree[idx].to(device, torch.float32))
        return [x[idx].to(device, torch.float32) for x in self.X], self.codes[idx].to(device), rep

    def seen_batch(self, rows):
        T = min(L for _, L in rows)
        return self.seen[torch.tensor([[w + t for t in range(T)] for w, _ in rows])]


@torch.no_grad()
def evaluate(model, data, device, batch, lenses, rng, pool, baseline=None, nmax=64):
    model.eval()
    wins = data.windows["dev"]
    order = rng.permutation(len(wins))[: min(len(wins), nmax * batch)]
    K = len(lenses)
    acc = {n: {"rank1_pred_codes": 0.0, "cos_pred_codes": 0.0, "rank1_true_codes": 0.0, "cos_true_codes": 0.0,
               "copy_rank1": 0.0, "copy_cos": 0.0} for n in lenses}
    if baseline is not None:
        for n in lenses:
            acc[n] |= {"v2_rank1": 0.0, "v2_cos": 0.0}
    code_acc = [{"top1": 0.0, "top5": 0.0} for _ in range(model.L)]
    exact_codes = 0.0
    man = {"pred": [], "true": [], "v2": []}
    # pooled over positions (not mean of batch means), once over all positions and once over leak-free ones,
    # so the two differ only by the duplicated targets
    blank = lambda: {"positions": 0.0, "code_top1": [0.0] * model.L, "rank1_pred_codes": 0.0, "rank1_true_codes": 0.0,
                     "copy_rank1": 0.0} | ({"v2_rank1": 0.0} if baseline is not None else {})
    pooled = {"all": blank(), "leak_free": blank()}
    pool_x = [x[torch.as_tensor(pool)].to(device, torch.float32) for x in data.X]
    nb = 0
    for j in range(0, len(order), batch):
        rows = [wins[i] for i in order[j:j + batch]]
        if len(rows) < 8:
            break
        s, c, r = data.coded_batch(rows, device)
        h = model.encode(s, c, r)
        guess = model.recognise(h, r)
        logits = model.code_logits(h, next_codes(c, guess), r)
        fresh = ~data.seen_batch(rows)[:, 1:].to(device)  # target atoms whose text is not in training
        masks = {"all": torch.ones_like(fresh), "leak_free": fresh}
        for key, mk in masks.items():
            pooled[key]["positions"] += float(mk.sum())
        for l, lg in enumerate(logits):
            y = c[:, 1:, l]; q = lg[:, :-1]
            for key, mk in masks.items():
                pooled[key]["code_top1"][l] += float(((q.argmax(-1) == y) & mk).sum())
            code_acc[l]["top1"] += float((q.argmax(-1) == y).float().mean())
            code_acc[l]["top5"] += float((q.topk(5, -1).indices == y[..., None]).any(-1).float().mean())
        exact_codes += float((guess[:, :-1] == c[:, 1:]).all(-1).float().mean())
        routes = {"pred_codes": model.place(h, s, guess), "true_codes": model.place(h, s, next_codes(c, guess))}
        if baseline is not None:
            routes["v2"] = baseline(s)
        T1 = s[0].shape[1] - 1
        flat = {key: mk.reshape(-1) for key, mk in masks.items()}
        for k, n in enumerate(lenses):
            y = s[k][:, 1:].reshape(-1, s[k].shape[-1]); cur = s[k][:, :-1].reshape(-1, s[k].shape[-1])
            eye = torch.arange(len(y), device=device)
            for route, p in routes.items():
                q = p[k][:, :-1].reshape(-1, p[k].shape[-1])
                hit = self_excluded(q @ y.T, T1).argmax(1) == eye
                name = "v2_rank1" if route == "v2" else f"rank1_{route}"
                acc[n][name] += float(hit.float().mean())
                for key, fm in flat.items():
                    pooled[key][name] += float((hit & fm).sum()) / K
                acc[n]["v2_cos" if route == "v2" else f"cos_{route}"] += float((q * y).sum(-1).mean())
            hit = self_excluded(cur @ y.T, T1).argmax(1) == eye
            acc[n]["copy_rank1"] += float(hit.float().mean())
            for key, fm in flat.items():
                pooled[key]["copy_rank1"] += float((hit & fm).sum()) / K
            acc[n]["copy_cos"] += float((cur * y).sum(-1).mean())
        if nb < 8:  # on-manifold: best joint (mean over senses) cosine to a real training constellation
            last = lambda p: [p[k][:, -2] for k in range(K)]  # prediction for the last atom of the window
            truth = [s[k][:, -1] for k in range(K)]
            for name, v in [("pred", last(routes["pred_codes"])), ("true", truth)] + ([("v2", last(routes["v2"]))] if baseline is not None else []):
                score = sum(v[k] @ pool_x[k].T for k in range(K)) / K
                man[name] += score.max(1).values.cpu().tolist()
        nb += 1
    model.train()
    for n in lenses:
        for k in acc[n]:
            acc[n][k] /= max(nb, 1)
    for ca in code_acc:
        for k in ca:
            ca[k] /= max(nb, 1)
    def norm(d):
        m = max(d["positions"], 1.0)
        return {k: (int(v) if k == "positions" else [x / m for x in v] if isinstance(v, list) else v / m) for k, v in d.items()}
    pooled = {k: norm(v) for k, v in pooled.items()}
    return {"batches": nb, "per_sense": acc, "code_acc": code_acc, "all_levels_exact": exact_codes / max(nb, 1),
            "pooled_mean_over_senses": pooled,
            "on_manifold_max_joint_cos": {k: (float(np.mean(v)) if v else None) for k, v in man.items()},
            "on_manifold_n": len(man["true"]), "pool": int(len(pool)),
            "chance_rank1": 1.0 / max(1, (batch * (data.window - 1)))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--atoms", type=Path, required=True)
    p.add_argument("--slots", type=Path, required=True)
    p.add_argument("--lenses", required=True)
    p.add_argument("--codebook", type=Path, required=True, help="recognition.py output directory")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--baseline", type=Path, help="predictor v2 directory, evaluated on the same dev windows")
    p.add_argument("--weave", type=Path, help="weave.py output: reproduction token + association prior (predictor v4)")
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--layers", type=int, default=8)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--window", type=int, default=24)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--sampled", type=float, default=0.25, help="fraction of positions placed from the model's own codes")
    p.add_argument("--seed", type=int, default=25)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--pool", type=int, default=100000)
    p.add_argument("--legacy-filtered-diagnostic", action="store_true")
    a = p.parse_args()
    require_sequence_authority(a.atoms, a.legacy_filtered_diagnostic)
    if not torch.cuda.is_available():
        raise SystemExit("MNLM_RECOGNISE_NO_GPU")
    take_lease("25", "recognising-predictor")
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    a.output.mkdir(parents=True, exist_ok=False)
    lenses = a.lenses.split(","); dev = "cuda"
    crep = json.loads((a.codebook / "report.json").read_bytes())
    if crep["lenses"] != lenses:
        raise SystemExit("MNLM_RECOGNISE_LENSES: codebook lenses differ from --lenses")
    book = Codebook(a.codebook / "codebook.npz", dev)
    if hashlib.sha256((a.codebook / "codebook.npz").read_bytes()).hexdigest() != crep["codebook_sha256"]:
        raise SystemExit("MNLM_RECOGNISE_CODEBOOK_SHA")
    wrep = None
    if a.weave:
        wrep = json.loads((a.weave / "report.json").read_bytes())
        if wrep["codes_sha256"] != crep["codes_sha256"] or wrep["lenses"] != lenses:
            raise SystemExit("MNLM_RECOGNISE_WEAVE: weave built on other codes or lenses")
        for f in ("edges.npy", "edge_w.npy", "agree.npy", "succ.npy"):
            if hashlib.sha256((a.weave / f).read_bytes()).hexdigest() != wrep["files_sha256"][f]:
                raise SystemExit(f"MNLM_RECOGNISE_WEAVE_SHA: {f}")
    data = CodedData(a.atoms, a.slots, lenses, a.window, a.codebook / "codes.npy", a.weave)
    dims = [x.shape[1] for x in data.X]
    n_agree = 0 if data.agree is None else int(data.agree.shape[1])
    model = Recogniser(dims, book.sizes, book.schema, a.width, a.layers, a.heads, a.window, a.dropout, n_agree).to(dev)
    baseline = None
    if a.baseline:
        bc = json.loads((a.baseline / "report.json").read_bytes())["config"]
        baseline = Predictor(dims, bc["width"], bc["layers"], bc["heads"], bc["window"], 0.0, bc.get("residual", False)).to(dev)
        baseline.load_state_dict(torch.load(a.baseline / "predictor.pt", map_location=dev)); baseline.eval()
    split = np.array([bucket(d) for d in data.doc])
    train_rows = np.flatnonzero((split >= SPLIT["train"][0]) & (split < SPLIT["train"][1]))
    pool = np.sort(np.random.default_rng(7).choice(train_rows, min(a.pool, len(train_rows)), replace=False))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05, betas=(0.9, 0.95))
    warm = 500
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1, s / a.steps)))))
    log = (a.output / "train.jsonl").open("w")
    started = time.perf_counter()
    train = data.windows["train"]
    params = sum(q.numel() for q in model.parameters())
    print(json.dumps({"windows": {k: len(v) for k, v in data.windows.items()}, "params": params, "codes": book.sizes}), flush=True)
    best, best_state, checks = -1.0, None, []
    for step in range(a.steps):
        rows = [train[i] for i in rng.integers(0, len(train), a.batch)]
        s, c, r = data.coded_batch(rows, dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = model.encode(s, c, r)
            guess = model.recognise(h, r)
            cond = next_codes(c, guess)
            own = torch.rand(cond.shape[:2], device=dev) < a.sampled
            placed_codes = torch.where(own[..., None], guess, cond)  # guess at t is already the model's codes for t+1
            logits = model.code_logits(h, cond, r)
            pred = model.place(h, s, placed_codes)
        ce = code_loss([lg.float() for lg in logits], c)
        sl, per = losses([x.float() for x in pred], s)
        loss = sum(ce) + sl
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if not torch.isfinite(loss):
            raise SystemExit("MNLM_RECOGNISE_NONFINITE_LOSS")
        if step % 200 == 0 or step == a.steps - 1:
            rec = {"step": step, "loss": float(loss), "code_ce": [round(float(x), 4) for x in ce],
                   "cos": round(float(np.mean([v[1] for v in per.values()])), 4), "s": round(time.perf_counter() - started)}
            log.write(json.dumps(rec) + "\n"); log.flush(); print(json.dumps(rec), flush=True)
        if a.eval_every and (step + 1) % a.eval_every == 0:
            chk = evaluate(model, data, dev, a.batch, lenses, np.random.default_rng(1), pool[::max(1, len(pool) // 20000)], None, nmax=16)
            m = float(np.mean([v["rank1_pred_codes"] for v in chk["per_sense"].values()]))
            checks.append({"step": step + 1, "dev_rank1_pred_codes": m, "dev_code_top1": [x["top1"] for x in chk["code_acc"]]})
            print(json.dumps(checks[-1]), flush=True)
            if m > best:
                best, best_state = m, {k: v.detach().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), a.output / "predictor.pt")
    ev = evaluate(model, data, dev, a.batch, lenses, np.random.default_rng(0), pool, baseline)
    publish(a.output / "report.json", {"schema": "recognising-predictor-v2" if a.weave else "recognising-predictor-v1", "dev_checks": checks,
        "weave": None if wrep is None else {"dir": str(a.weave), "files_sha256": wrep["files_sha256"],
                                            "prior_gain": float(model.prior_gain.detach())},
        "selection": "best mean dev rank-1 with predicted codes (dev rng 1, 16 batches); final eval on dev rng 0" if checks else "last step",
        "split": "dev", "lenses": lenses, "dims": dims, "codes": book.sizes,
        "codebook_sha256": crep["codebook_sha256"], "codebook_report": crep["levels"],
        "config": vars(a) | {k: str(getattr(a, k)) for k in ("atoms", "slots", "codebook", "output", "baseline", "weave")},
        "params": params, "windows": {k: len(v) for k, v in data.windows.items()},
        "mix_schema_current": dict(zip(lenses, model.mix.detach().cpu().tolist())),
        "eval": ev, "test_split_untouched": True, "wall_seconds": time.perf_counter() - started,
        "predictor_sha256": hashlib.sha256((a.output / "predictor.pt").read_bytes()).hexdigest(),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    print(json.dumps({"code_acc": ev["code_acc"], "on_manifold": ev["on_manifold_max_joint_cos"],
                      "mean": {k: round(float(np.mean([v[k] for v in ev["per_sense"].values()])), 4) for k in next(iter(ev["per_sense"].values()))}}, indent=1))


if __name__ == "__main__":
    run_cli(main)
