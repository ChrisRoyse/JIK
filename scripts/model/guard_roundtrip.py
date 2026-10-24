"""MNLM pilot, part 3: re-measure and guard (#25 option 1). The same frozen embedders read what the model wrote.

On the TEST split (sealed until now; first opened here, recorded in the report):
  A. authentic round trip: render from an atom's TRUE constellation, re-embed the render with all K
     frozen embedders, per-slot cosine (centred, as the models see slots) between re-measured and source.
     Calibration half: tau_k = the 10th percentile of these cosines per slot (spec §27: calibrated to the
     10th-percentile authentic-sample cosine; never tuned toward a gate).
  B. generation: the predictor reads a document's atoms 0..t and predicts constellation t+1; the renderer
     writes it; the embedders re-measure; the guard ACCEPTS iff every slot clears its tau_k, else REFUSES
     and names the failing slots. Reported on the evaluation half: acceptance rate, per-slot pass rates,
     and, as the grounding check, cosine of the re-measured render to the TRUE next atom vs a random atom.
  Routes: direct (predictor v2 regresses the slots), kernel-first snap (--snap N), recognised (a
  recognising-predictor-v1 directory: the next region is recognised first, then the slots are placed in it;
  the history's codes come from the frozen codebook it names). Every route also reports the fraction of
  generated texts that exactly equal a training-split atom (retrieval by rendering, not composition).
GPU: models resident one at a time where possible; embedders are the same snapshots, runtime, centring.
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
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lens"))
from lens.catalogue import by_name
from lens.lenscommon import run_cli, take_lease
from model.constellation_lm import SPLIT, Data, Predictor, bucket
from model.recognising_lm import Recogniser, association_votes
from model.recognition import Codebook
from model.surface_renderer import Renderer
from senses.trajectory_population import publish
from text.measure_pilot import embed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--atoms", type=Path, required=True)
    p.add_argument("--slots", type=Path, required=True)
    p.add_argument("--lenses", required=True)
    p.add_argument("--predictor", type=Path, required=True)
    p.add_argument("--renderer", type=Path, required=True)
    p.add_argument("--budgets", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--n", type=int, default=1024, help="test windows used (half calibration, half evaluation)")
    p.add_argument("--snap", type=int, default=0, help="kernel-first: snap predictions to the nearest of N training constellations")
    a = p.parse_args()
    take_lease("25", "guard-roundtrip")
    started = time.perf_counter()
    a.output.mkdir(parents=True, exist_ok=False)
    lenses = a.lenses.split(","); dev = "cuda"
    budgets = json.loads(a.budgets.read_bytes())
    data = Data(a.atoms, a.slots, lenses, 24)  # centred slots (training-split mean), same transform as training
    texts = pq.read_table(a.atoms / "atoms.parquet", columns=["text"]).column("text").to_pylist()
    prep = json.loads((a.predictor / "report.json").read_bytes()); pcfg = prep["config"]
    rcfg = json.loads((a.renderer / "report.json").read_bytes())["config"]
    dims = [x.shape[1] for x in data.X]
    recognised = prep["schema"] in ("recognising-predictor-v1", "recognising-predictor-v2")
    vote = agree = None
    if recognised:
        if a.snap:
            raise SystemExit("MNLM_GUARD_ROUTE: --snap and a recognising predictor are separate routes")
        if prep["lenses"] != lenses:
            raise SystemExit("MNLM_GUARD_LENSES: predictor lenses differ from --lenses")
        cdir = Path(pcfg["codebook"])
        if hashlib.sha256((cdir / "codebook.npz").read_bytes()).hexdigest() != prep["codebook_sha256"]:
            raise SystemExit("MNLM_GUARD_CODEBOOK_SHA: the codebook differs from the one the predictor was trained on")
        book = Codebook(cdir / "codebook.npz", dev)
        codes_all = torch.from_numpy(np.load(cdir / "codes.npy").astype(np.int64))
        n_agree = 0
        if prep["schema"] == "recognising-predictor-v2":  # predictor v4: reproduction token + association prior
            wdir = Path(prep["weave"]["dir"])
            for f in ("edges.npy", "edge_w.npy", "agree.npy", "succ.npy"):
                if hashlib.sha256((wdir / f).read_bytes()).hexdigest() != prep["weave"]["files_sha256"][f]:
                    raise SystemExit(f"MNLM_GUARD_WEAVE_SHA: {f} differs from the weave the predictor was trained on")
            c1 = codes_all[:, 0].numpy()
            vote = torch.from_numpy(association_votes(np.load(wdir / "edges.npy"), np.load(wdir / "edge_w.npy"),
                                                      np.load(wdir / "succ.npy").astype(np.int64), c1, int(c1.max()) + 1))
            agree = torch.from_numpy(np.load(wdir / "agree.npy")); n_agree = int(agree.shape[1])
        pred = Recogniser(dims, book.sizes, book.schema, pcfg["width"], pcfg["layers"], pcfg["heads"], pcfg["window"], 0.0, n_agree).to(dev)
    else:
        pred = Predictor(dims, pcfg["width"], pcfg["layers"], pcfg["heads"], pcfg["window"], 0.0, pcfg.get("residual", False)).to(dev)
    pred.load_state_dict(torch.load(a.predictor / "predictor.pt", map_location=dev)); pred.eval()
    rend = Renderer(dims, rcfg["width"], rcfg["layers"], rcfg["heads"]).to(dev)
    rend.load_state_dict(torch.load(a.renderer / "renderer.pt", map_location=dev)); rend.eval()
    rng = np.random.default_rng(25)
    wins = [data.windows["test"][i] for i in rng.permutation(len(data.windows["test"]))[: a.n]]
    half = len(wins) // 2
    # each window: history = atoms w..w+L-2, target = atom w+L-1
    hist = [(w, L) for w, L in wins]
    target_idx = np.array([w + L - 1 for w, L in hist])
    true_slots = [x[torch.as_tensor(target_idx)].to(dev, torch.float32) for x in data.X]
    renders_true, renders_gen, gen_slots = [], [], [[] for _ in lenses]
    with torch.no_grad():
        for j in range(0, len(hist), 32):
            part = hist[j:j + 32]
            Lmin = min(L for _, L in part)
            idx = torch.tensor([[w + L - Lmin + t for t in range(Lmin - 1)] for w, L in part])  # last Lmin-1 history atoms
            s = [x[idx].to(dev, torch.float32) for x in data.X]
            if recognised:
                r = None if vote is None else (vote[idx].to(dev, torch.float32), agree[idx].to(dev, torch.float32))
                h = pred.encode(s, codes_all[idx].to(dev), r)
                out = pred.place(h, s, pred.recognise(h, r))
            else:
                out = pred(s)
            nxt = [o[:, -1] for o in out]
            for k in range(len(lenses)):
                gen_slots[k].append(nxt[k])
            renders_gen += rend.render(nxt)
            renders_true += rend.render([t[j:j + 32] for t in true_slots])
    gen_slots = [torch.cat(g) for g in gen_slots]
    snap_info = None
    if a.snap:
        # Kernel-first route (spec §28): snap each predicted constellation to the nearest REAL training-split
        # constellation, scored jointly as the sum of per-sense cosines; render and guard the snapped one.
        train_rows = np.flatnonzero(np.array([SPLIT["train"][0] <= bucket(d) < SPLIT["train"][1] for d in data.doc]))
        pool = np.sort(np.random.default_rng(7).choice(train_rows, min(a.snap, len(train_rows)), replace=False))
        score = torch.zeros(len(hist), len(pool), device=dev)
        for k, x in enumerate(data.X):
            score += gen_slots[k] @ x[torch.as_tensor(pool)].to(dev, torch.float32).T
        pick = pool[score.argmax(1).cpu().numpy()]
        gen_slots = [x[torch.as_tensor(pick)].to(dev, torch.float32) for x in data.X]
        with torch.no_grad():
            renders_gen = []
            for j in range(0, len(hist), 32):
                renders_gen += rend.render([g[j:j + 32] for g in gen_slots])
        snap_info = {"pool": int(len(pool)), "pool_seed": 7, "snapped_texts": [texts[i] for i in pick]}

    def remeasure(strings):
        out = []
        for k, n in enumerate(lenses):
            X, _, _ = embed(by_name(n), [s if s.strip() else "." for s in strings], torch.float32, budgets[n])
            X = torch.from_numpy(X).to(dev, torch.float32)
            out.append(F.normalize(X - data.centre[k].to(dev), dim=-1))
        return out
    m_true, m_gen = remeasure(renders_true), remeasure(renders_gen)
    cos = lambda A, B: (A * B).sum(-1)
    auth = torch.stack([cos(m_true[k], true_slots[k]) for k in range(len(lenses))], 1).cpu().numpy()  # N x K
    tau = np.percentile(auth[:half], 10, axis=0)
    ev = slice(half, None)
    gen_vs_pred = torch.stack([cos(m_gen[k], gen_slots[k]) for k in range(len(lenses))], 1).cpu().numpy()
    passed = gen_vs_pred[ev] >= tau
    accept = passed.all(1)
    perm = rng.permutation(len(hist))
    ground_true = torch.stack([cos(m_gen[k], true_slots[k]) for k in range(len(lenses))], 1).cpu().numpy()[ev].mean(0)
    ground_rand = torch.stack([cos(m_gen[k], true_slots[k][torch.as_tensor(perm, device=dev)]) for k in range(len(lenses))], 1).cpu().numpy()[ev].mean(0)
    auth_pass = (auth[ev] >= tau).all(1).mean()
    train_text = {texts[i] for i in np.flatnonzero(np.array([SPLIT["train"][0] <= bucket(d) < SPLIT["train"][1] for d in data.doc]))}
    verbatim = float(np.mean([g in train_text for g in renders_gen[half:]]))
    rows = [{"history_last": texts[w + L - 2], "true_next": texts[w + L - 1], "render_from_true": rt, "generated": rg,
             "accepted": bool(acc), "failed_slots": [lenses[k] for k in np.flatnonzero(~ps)]}
            for (w, L), rt, rg, acc, ps in zip(hist[half:], renders_true[half:], renders_gen[half:], accept, passed)]
    (a.output / "guard-test.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    if snap_info:
        for r, t in zip(rows, snap_info["snapped_texts"][half:]):
            r["snapped_atom"] = t
        (a.output / "guard-test.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    publish(a.output / "report.json", {"schema": "guard-roundtrip-v1", "split": "test", "test_split_first_opened": not (a.snap or recognised),
        "route": "kernel-first snap" if a.snap else ("recognised (region first, then slots)" if recognised else "direct prediction"),
        "generated_verbatim_training_atom_rate": verbatim, "snap": {k: v for k, v in (snap_info or {}).items() if k != "snapped_texts"},
        "windows": len(hist), "calibration": half, "evaluation": len(hist) - half, "lenses": lenses,
        "tau_10th_percentile": dict(zip(lenses, map(float, tau))),
        "authentic_roundtrip_mean_cos": dict(zip(lenses, map(float, auth[ev].mean(0)))),
        "authentic_eval_accept_rate": float(auth_pass),
        "generated_accept_rate": float(accept.mean()),
        "generated_slot_pass_rate": dict(zip(lenses, map(float, passed.mean(0)))),
        "grounding_cos_generated_vs_true_next": dict(zip(lenses, map(float, ground_true))),
        "grounding_cos_generated_vs_random_atom": dict(zip(lenses, map(float, ground_rand))),
        "predictor_sha256": hashlib.sha256((a.predictor / "predictor.pt").read_bytes()).hexdigest(),
        "renderer_sha256": hashlib.sha256((a.renderer / "renderer.pt").read_bytes()).hexdigest(),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "wall_seconds": time.perf_counter() - started})
    print(json.dumps({"authentic_accept": float(auth_pass), "generated_accept": float(accept.mean()),
                      "grounding_true": float(ground_true.mean()), "grounding_random": float(ground_rand.mean()), "verbatim": verbatim}))


if __name__ == "__main__":
    run_cli(main)
