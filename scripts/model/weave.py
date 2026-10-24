"""MNLM weave: synthesis of reproduction, the "count" stage (spec §21 Stage 5; "The Senses of a Machine" §8,
Table 1: apprehension = measure, reproduction = count, recognition = compose).

Recognition (recognising_lm.py) names the next atom's region at 30 % top-1 on dev, against 23 % for copying the
current region; given the true region the slot rank-1 rises from 0.30 to 0.72 (#25). So recognition is the
weak link, and Kant puts reproduction before it: the manifold must be run through and held together before a
concept can be applied. Spec §21 makes that literal, and this builds it for the pilot:
  between-atom   per sense, the exact top-K neighbours of every atom by that sense's own cosine, drawn ONLY
                 from training-split atoms of OTHER documents (no leakage: dev/test never see dev/test, an
                 atom never sees its own document). Brute force on the GPU (N = 960,075; no ANN, so recall is
                 exact up to f16 rounding, and checked against a float64 CPU recomputation on sampled queries).
                 Association graph: an edge a -> n when n is in a's top-K in at least MIN_LENSES senses
                 (Poly's rule, spec §21), weight = number of senses agreeing; the top EDGES edges per atom.
  within-atom    for every sense pair (i, j), the agreement scalar |N_i(a) & N_j(a)| / K: do the two senses
                 see the same neighbourhood around this atom. C(12,2) = 66 per atom; the agreement graph is
                 their mean over atoms (the panel's redundancy map).
Measured here, before any model consumes it (does reproduction carry the next region?): the reproduction
vote. For a dev atom a with a successor, each association neighbour n votes, with its weight, for the
level-1 region of n's own successor in n's document (what followed this association before). Reported
against the codebook baselines on the same pairs (unigram, copy, bigram) and against single-sense votes and
the union vote (>= 1 sense), with a negative control (successor regions permuted over training atoms; must
fall to unigram or below; smoke on the 20k slice: 0.03 vs unigram 0.09). Nothing is tuned toward a gate.
Outputs: nbr_<k>.npy int32 [N, K] per sense, edges.npy int32 [N, EDGES] (-1 = none), edge_w.npy uint8,
agree.npy float16 [N, 66], succ.npy int32 [N] (-1 = none), report.json. GPU; one lease.
Cost: per sense O(N * N_train * d) multiply-adds (about 6e14 at d = 768) plus top-K; edges O(N * S*K log S*K).
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time

import numpy as np
import pyarrow.parquet as pq
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lens.lenscommon import run_cli, take_lease
from model.constellation_lm import SPLIT, Data, bucket, text_hash
from senses.trajectory_population import publish

K_NBR, MIN_LENSES, EDGES, CHECK = 16, 2, 32, 256
SCORE_BYTES = 2 ** 32  # one f32 score block [rows, N_train] stays under 4 GiB, so the block shrinks as N grows
torch.backends.cuda.matmul.allow_tf32 = True


def knn(X, cand, cand_doc, qdoc, cand_h, qh, dev):
    """Exact top-K by cosine (rows are unit) over the candidate rows; the own document and any candidate with the
    query's exact text are excluded (a verbatim duplicate's successor is the answer, not an association).
    Returns int32 [N, K], f16 sims."""
    # slots are f16 (10-bit mantissa), so TF32 inputs are exact and products accumulate in f32
    C = X[torch.as_tensor(cand)].to(dev, torch.float32)
    cd = torch.as_tensor(cand_doc, device=dev); chh = torch.as_tensor(cand_h, device=dev)
    N = len(X); out = np.empty((N, K_NBR), np.int32); sims = np.empty((N, K_NBR), np.float16)
    cand_t = torch.as_tensor(cand, device=dev)
    block = max(64, min(1024, SCORE_BYTES // (4 * len(cand))))
    for j in range(0, N, block):
        q = X[j:j + block].to(dev, torch.float32)
        s = q @ C.T
        s[torch.as_tensor(qdoc[j:j + block], device=dev)[:, None] == cd[None, :]] = float("-inf")
        s[torch.as_tensor(qh[j:j + block], device=dev)[:, None] == chh[None, :]] = float("-inf")
        v, i = s.topk(K_NBR, 1)
        out[j:j + block] = cand_t[i].int().cpu().numpy(); sims[j:j + block] = v.half().cpu().numpy()
    return out, sims


def edges_from(nbrs, sims, dev):
    """Neighbours in >= MIN_LENSES senses; score = senses agreeing + mean sim tiebreak (< 1). Returns [N, EDGES] ids, weights."""
    S = len(nbrs); N = len(nbrs[0])
    ids_all = np.empty((N, EDGES), np.int32); w_all = np.zeros((N, EDGES), np.uint8)
    for j in range(0, N, 65536):
        ids = torch.as_tensor(np.concatenate([n[j:j + 65536] for n in nbrs], 1), device=dev).long()
        sm = torch.as_tensor(np.concatenate([s[j:j + 65536] for s in sims], 1), device=dev).float()
        srt, o = ids.sort(1); sm = sm.gather(1, o)
        start = torch.ones_like(srt, dtype=torch.bool); start[:, 1:] = srt[:, 1:] != srt[:, :-1]
        run = start.long().cumsum(1) - 1
        cnt = torch.zeros_like(srt, dtype=torch.float32).scatter_add_(1, run, torch.ones_like(sm)).gather(1, run)
        ssum = torch.zeros_like(sm).scatter_add_(1, run, sm).gather(1, run)
        score = torch.where(start & (cnt >= MIN_LENSES), cnt + ssum / (cnt * (S + 1)), torch.full_like(cnt, -1.0))
        v, i = score.topk(min(EDGES, score.shape[1]), 1)
        e = srt.gather(1, i); keep = v > 0  # valid scores are >= MIN_LENSES - 1/(S+1) > 0; invalid are -1
        ids_all[j:j + 65536] = torch.where(keep, e, torch.full_like(e, -1)).int().cpu().numpy()
        # weight = the count itself (floor(score) would drop to count-1 when the mean sim is negative)
        w_all[j:j + 65536] = torch.where(keep, cnt.gather(1, i), torch.zeros_like(v)).to(torch.uint8).cpu().numpy()
    return ids_all, w_all


def agreement(nbrs, dev):
    """|N_i & N_j| / K per atom per sense pair."""
    pairs = list(itertools.combinations(range(len(nbrs)), 2)); N = len(nbrs[0])
    out = np.empty((N, len(pairs)), np.float16)
    for j in range(0, N, 65536):
        t = [torch.as_tensor(n[j:j + 65536], device=dev) for n in nbrs]
        out[j:j + 65536] = torch.stack([(t[a][:, :, None] == t[b][:, None, :]).any(-1).float().mean(1) for a, b in pairs], 1).half().cpu().numpy()
    return out, pairs


def vote(ids, w, succ, c1, rows, C):
    """Weighted vote for the next level-1 region: each neighbour votes for its own successor's region."""
    e = ids[rows].astype(np.int64); ww = w[rows].astype(np.float64)
    ok = e >= 0
    s = np.where(ok, succ[np.where(ok, e, 0)], -1); ok &= s >= 0
    v = np.zeros((len(rows), C))
    r = np.repeat(np.arange(len(rows)), e.shape[1]).reshape(e.shape)
    np.add.at(v, (r[ok], c1[s[ok]]), ww[ok])
    return v


def score(v, y):
    has = v.sum(1) > 0
    top = np.argsort(-v, 1)[:, :5]
    return {"top1": float((has & (top[:, 0] == y)).mean()), "top5": float((has & (top == y[:, None]).any(1)).mean()),
            "coverage": float(has.mean()), "top1_when_covered": float((top[has, 0] == y[has]).mean()) if has.any() else None}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--atoms", type=Path, required=True)
    p.add_argument("--slots", type=Path, required=True)
    p.add_argument("--lenses", required=True)
    p.add_argument("--codebook", type=Path, required=True, help="recognition.py output (codes.npy, report.json)")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda", help="cpu only for a named smoke test; the run of record is cuda")
    a = p.parse_args()
    if a.device == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("MNLM_WEAVE_NO_GPU")
        take_lease("25", "weave")
    started = time.perf_counter(); dev = a.device
    a.output.mkdir(parents=True, exist_ok=False)
    lenses = a.lenses.split(",")
    crep = json.loads((a.codebook / "report.json").read_bytes())
    if hashlib.sha256((a.codebook / "codes.npy").read_bytes()).hexdigest() != crep["codes_sha256"]:
        raise SystemExit("MNLM_WEAVE_CODES_SHA")
    data = Data(a.atoms, a.slots, lenses, 24)
    N = len(data.doc)
    _, docid = np.unique(np.array(data.doc, dtype=object).astype(str), return_inverse=True)
    docid = docid.astype(np.int64)
    split = np.array([bucket(d) for d in data.doc])
    train = np.flatnonzero((split >= SPLIT["train"][0]) & (split < SPLIT["train"][1]))
    devrows = np.flatnonzero((split >= SPLIT["dev"][0]) & (split < SPLIT["dev"][1]))
    pos = data.pos
    succ = np.full(N, -1, np.int64)
    nx = np.flatnonzero((docid[1:] == docid[:-1]) & (pos[1:] == pos[:-1] + 1))
    succ[nx] = nx + 1
    th = text_hash(pq.read_table(a.atoms / "atoms.parquet", columns=["text"]).column("text").to_pylist())
    codes = np.load(a.codebook / "codes.npy").astype(np.int64); c1 = codes[:, 0]; C = int(c1.max()) + 1
    assert len(codes) == N, "MNLM_WEAVE_ROWS"
    nbrs, sims, recall = [], [], {}
    rng = np.random.default_rng(20261019)
    for k, n in enumerate(lenses):
        t0 = time.perf_counter()
        nb, sm = knn(data.X[k], train, docid[train], docid, th[train], th, dev)
        # boundary 2: float64 CPU brute force on sampled queries; a returned neighbour counts when its float64
        # cosine reaches the true K-th best (exact ties, e.g. duplicate sentences, may be broken either way)
        qs = rng.choice(N, CHECK, replace=False)
        Q = data.X[k][torch.as_tensor(qs)].double().numpy().T
        S = np.concatenate([data.X[k][torch.as_tensor(train[c:c + 262144])].double().numpy() @ Q for c in range(0, len(train), 262144)])
        S[(docid[train][:, None] == docid[qs][None, :]) | (th[train][:, None] == th[qs][None, :])] = -np.inf
        full = np.full(N, -np.inf); hit = 0
        for c, q in enumerate(qs):
            thr = np.partition(S[:, c], len(train) - K_NBR)[len(train) - K_NBR]
            full[train] = S[:, c]
            hit += int((full[nb[q]] >= thr - 1e-6).sum())
        recall[n] = hit / (CHECK * K_NBR)
        del S
        print(json.dumps({"sense": n, "knn_s": round(time.perf_counter() - t0, 1), "recall_vs_f64": round(recall[n], 4)}), flush=True)
        if recall[n] < 0.95:
            raise SystemExit(f"MNLM_WEAVE_RECALL: {n} top-{K_NBR} recall {recall[n]:.3f} vs float64 brute force")
        if (docid[nb] == docid[:, None]).any() or (th[nb] == th[:, None]).any() or not np.isin(nb[qs], train).all():
            raise SystemExit(f"MNLM_WEAVE_LEAK: {n} neighbour from own document, with the query's own text, or outside the training split")
        np.save(a.output / f"nbr_{k}.npy", nb); nbrs.append(nb); sims.append(sm)
    ids, w = edges_from(nbrs, sims, dev)
    agree, pairs = agreement(nbrs, dev)
    np.save(a.output / "edges.npy", ids); np.save(a.output / "edge_w.npy", w)
    np.save(a.output / "agree.npy", agree); np.save(a.output / "succ.npy", succ.astype(np.int32))
    # boundary 1: agreement scalars recomputed independently (python sets) on sampled atoms
    qs = rng.choice(N, 200, replace=False)
    worst = max(abs(float(agree[q, pi]) - len(set(nbrs[i][q]) & set(nbrs[j][q])) / K_NBR) for q in qs for pi, (i, j) in enumerate(pairs))
    if worst > 1e-3:
        raise SystemExit(f"MNLM_WEAVE_AGREE: agreement scalar differs from recomputation by {worst}")
    # the reproduction vote on dev pairs (same pairs as the codebook baselines)
    q = devrows[succ[devrows] >= 0]; y = c1[succ[q]]
    trn = train[succ[train] >= 0]
    uni = np.bincount(c1[train], minlength=C)
    big = np.zeros((C, C)); np.add.at(big, (c1[trn], c1[succ[trn]]), 1)
    perm = c1.copy(); perm[train] = rng.permutation(c1[train])
    ones = lambda nb: (nb, np.ones_like(nb, dtype=np.uint8))
    allids = np.concatenate(nbrs, 1)

    def table(q):
        y = c1[succ[q]]
        r = {"pairs": int(len(q)), "unigram": float((y == uni.argmax()).mean()), "copy": float((y == c1[q]).mean()),
             "bigram": float((y == big.argmax(1)[c1[q]]).mean())}
        r["association"] = score(vote(ids, w, succ, c1, q, C), y)
        r["single_sense"] = {n: score(vote(*ones(nbrs[k]), succ, c1, q, C), y) for k, n in enumerate(lenses)}
        r["union_any_sense"] = score(vote(*ones(allids), succ, c1, q, C), y)
        r["control_successor_regions_permuted"] = score(vote(ids, w, succ, perm, q, C), y)
        return r
    # leak-free: the true successor's text never occurs in the training split (else a duplicated passage can
    # answer by lookup); this subset is decided by the corpus alone, the same in recognising_lm.py
    seen = np.isin(th, th[train])
    res = table(q) | {"leak_free": table(q[~seen[succ[q]]]), "fraction_successor_text_in_train": float(seen[succ[q]].mean())}
    # coverage of the graph
    deg = (ids >= 0).sum(1)
    agree_mean = agree[train[::max(1, len(train) // 200000)]].astype(np.float32).mean(0)  # stride: rows are domain-ordered
    publish(a.output / "report.json", {"schema": "weave-v2", "lenses": lenses, "atoms": N, "train_atoms": int(len(train)),
        "k": K_NBR, "min_lenses": MIN_LENSES, "edges_per_atom_max": EDGES,
        "neighbour_pool": "training split, other documents only, never the query's exact text", "recall_vs_float64": recall,
        "degree": {"mean": float(deg.mean()), "zero_fraction": float((deg == 0).mean()), "p50": float(np.median(deg))},
        "edge_weight_hist": {int(v): int(c) for v, c in zip(*np.unique(w[w > 0], return_counts=True))},
        "agreement_graph": {f"{lenses[i]}|{lenses[j]}": float(agree_mean[pi]) for pi, (i, j) in enumerate(pairs)},
        "agreement_check_max_abs_diff": worst, "next_region_vote_dev": res,
        "codes_sha256": crep["codes_sha256"], "codebook_sha256": crep["codebook_sha256"],
        "device": dev, "files_sha256": {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(a.output.glob("*.npy"))},
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "wall_seconds": time.perf_counter() - started})
    print(json.dumps(res | {"single_sense": {n: v["top1"] for n, v in res["single_sense"].items()}}, indent=1))


if __name__ == "__main__":
    run_cli(main)
