"""Stage 3 FSV (issue #12, spec §19 verification): an independent restatement that reads the persisted registry, slots,
projections, readings, assay reports and panel back. B1 registration -> probe: lens_ids re-derived, weight digests
re-hashed (every encoder, a sample of learned lenses), the determinism probe replayed on a sample of learned lenses and
the replay compared with the stored slot rows (zero error), every registered lens's probe byte-identical. B2 assay inputs
-> recomputed: calibration digests, class histograms and H(Y); the seed-0 mixed-KSG point estimate recomputed from the
saved projections with this file's own numpy estimator for every lens whose seed-0 KSG estimated (the count is reported); projections re-derived from the slots with this
file's own sign function on a sample; readings and the Spearman matrix recomputed; the admission decisions re-derived
from the stored numbers; effective rank and sufficiency recomputed; the panel's invariants and repository copy
re-checked. B3 negatives: a byte-flipped weight file must refuse MNLM_LENS_FROZEN_VIOLATION; a duplicated lens under a
new name must be retired by the correlation gate; a label-shuffled control must not reach 0.05 bits. Cost: O(L × n²)
on the CPU for the point estimates (no intervals), one sampled replay per sampled lens.
    python -I scripts/lens/fsv_stage3.py --root R --out O --version p0-v1 --issue 12
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path

import blake3
import numpy as np
import scipy.sparse as sp
from scipy.special import digamma
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lenscommon import take_lease, FAMILIES, HOST, Paths, lens_root, read_jsonl, run_stamp, spec_family, write_json  # noqa: E402

CHECKS: list[dict] = []
K = 3
MIN_BITS, MAX_CORR, MIN_COV, MIN_LENSES = 0.05, 0.6, 0.9, 5


def check(name: str, ok: bool, **detail) -> bool:
    CHECKS.append({"check": name, "ok": bool(ok), **detail})
    print(f"  [{'ok' if ok else 'FAIL'}] {name} {json.dumps(detail, default=str)[:300]}", flush=True)
    return bool(ok)


def own_lens_id(c: dict) -> str:
    h = blake3.blake3()
    for part in (c["name"], c["weights_sha256"], c["corpus_hash"], f"dtype={c['dtype']};shape={c['shape']};norm={c['norm']}"):
        h.update(part.encode("utf-8"))
    return h.hexdigest()[:32]


def own_sha256_files(snap: Path, files: list[str]) -> str:
    h = hashlib.sha256()
    for f in files:
        h.update(f.encode("utf-8") + b"\x00")
        h.update((snap / f).read_bytes())
    return h.hexdigest()


def own_mixed_ksg(P: np.ndarray, y: np.ndarray, k: int = K) -> float:
    n = len(y)
    D = np.max(np.abs(P[:, None, :] - P[None, :, :]), axis=2)
    same = y[:, None] == y[None, :]
    np.fill_diagonal(same, False)
    tot = 0.0
    counts = np.bincount(y)
    for i in range(n):
        ds = np.sort(D[i][same[i]])
        r = ds[k - 1]
        same_c = int((ds <= r).sum())
        full_c = int((D[i] <= r).sum()) - 1
        tot += digamma(n) + digamma(same_c) - digamma(counts[y[i]]) - digamma(full_c)
    return max(0.0, tot / n / math.log(2))


def own_project(X, seed: int) -> np.ndarray:
    n, d = X.shape
    out = min(d, max(1, 2 * math.ceil(math.log2(n))))
    S = np.empty((d, out), np.float32)
    for i in range(d):
        for o in range(out):
            S[i, o] = 1.0 if blake3.blake3(seed.to_bytes(8, "big") + i.to_bytes(8, "big") + o.to_bytes(8, "big")).digest()[0] & 1 == 0 else -1.0
    P = np.asarray(X @ S, np.float64) / math.sqrt(out)
    mu, sd = P.mean(0), P.std(0)
    return np.where(sd > 0, (P - mu) / np.where(sd > 0, sd, 1), 0.0).astype(np.float32)


def own_entropy(y: np.ndarray) -> float:
    c = np.bincount(y)
    p = c[c > 0] / c.sum()
    return float(-(p * np.log2(p)).sum())


def load_slot(path: Path):
    z = np.load(path)
    absent = z["absent"].astype(bool)
    if "tokens" in z.files:
        import slotio

        return None, absent, "multi", {"tokens": slotio.npz_member(path, "tokens"), "offsets": np.asarray(slotio.npz_member(path, "offsets"))}
    if "data" in z.files:
        return sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"])), absent, "sparse", z
    return z["X"], absent, "dense", z


def own_cosine(X, i, j):
    if sp.issparse(X):
        a, b = X[i], X[j]
        na, nb = np.sqrt(a.multiply(a).sum()), np.sqrt(b.multiply(b).sum())
        return float(a.multiply(b).sum() / (na * nb)) if na > 0 and nb > 0 else 0.0
    a, b = np.asarray(X[i], np.float64), np.asarray(X[j], np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0


def b1_registration(root: Path, registry: dict, sample: int) -> None:
    ok = {n: e for n, e in registry["lenses"].items() if not e.get("error")}
    check("B1.registered_lenses_ge_40_with_probes", len(ok) >= 40 and all(e.get("probe", {}).get("byte_identical") for e in ok.values()), registered=len(ok),
          failed={n: e["error"] for n, e in registry["lenses"].items() if e.get("error")})
    bad = [n for n, e in ok.items() if own_lens_id(e["contract"]) != e["lens_id"] or e["contract"]["dtype"] != "f32"]
    check("B1.lens_ids_rederived", not bad, bad=bad[:5])
    stripped = {k: v for k, v in registry.items() if k not in ("ledger_seq", "ledger_head", "registry_blake3_before_ledger")}
    own = blake3.blake3((json.dumps(stripped, indent=2, sort_keys=True, ensure_ascii=False, default=str) + chr(10)).encode("utf-8")).hexdigest()
    check("B1.registry_digest_rederived_without_ledger_fields", own == registry.get("registry_blake3_before_ledger"), own=own[:16], recorded=str(registry.get("registry_blake3_before_ledger"))[:16])
    missing = [(n, fam) for n, e in ok.items() for fam, v in e["families"].items() if not v.get("digest")]
    cells = sorted(((n, fam) for n, e in ok.items() for fam in e["families"]), key=lambda c: blake3.blake3(f"{c[0]}|{c[1]}".encode()).digest())[:sample]
    bad = []
    for n, fam in cells:
        v = ok[n]["families"][fam]
        import slotio
        from slot_layout import check_seal
        from lenscommon import LensError
        try:
            check_seal(slotio.slot_digest(Path(v["slot"])), v, v["slot"])
        except LensError:
            bad.append((n, fam))
    check("B1.slot_digests_bound_and_sampled", not missing and not bad and bool(cells), missing=len(missing), sampled=[f"{n}/{fam}" for n, fam in cells], mismatched=bad)
    here = Path(__file__).resolve().parents[2]
    enc_sha = {m: hashlib.sha256((here / m).read_bytes()).hexdigest() for m in ("scripts/lens/encoders.py", "scripts/lens/encoders_ctx.py", "scripts/lens/encoders_kg.py", "scripts/lens/encoders_ctx2.py", "scripts/lens/encoders_ctx3.py", "scripts/lens/encoders_ctx4.py", "scripts/lens/encoders_ctx5.py", "scripts/lens/encoders_ctx6.py", "scripts/lens/encoders_ctx7.py")}
    bad = [n for n, e in ok.items() if e["contract"]["kind"] == "encoder" and e["contract"]["weights_sha256"] != enc_sha.get(e["contract"]["weight_files"][0])]
    check("B1.encoder_digest_is_current_encoders_py", not bad, bad=bad[:5], sha={m: v[:16] for m, v in enc_sha.items()})
    from learned import snapshot_dir

    learned = sorted(n for n, e in ok.items() if e["contract"]["kind"] != "encoder")
    pick = sorted(learned, key=lambda n: blake3.blake3(n.encode()).digest())[:sample]
    mism = []
    for n in pick:
        c = ok[n]["contract"]
        if own_sha256_files(snapshot_dir(c["repo"]), c["weight_files"]) != c["weights_sha256"]:
            mism.append(n)
    check("B1.weights_rehashed_sample", not mism and bool(pick), sampled=pick, mismatched=mism)
    return pick


def b1_replay(root: Path, registry: dict, pick: list[str], fam: str) -> None:
    import os

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    torch.use_deterministic_algorithms(True)
    from learned import LearnedLens

    rows = read_jsonl(root / "cal" / f"{fam}.jsonl")[:32]
    texts = [r["text"] for r in rows]
    results = {}
    for n in pick[:3]:
        spec = registry["lenses"][n]["spec"]
        lens = LearnedLens(spec, "cuda")
        a, b = lens.embed(texts), lens.embed(texts)
        X, absent, kind, z = load_slot(root / "slots" / fam / f"{n}.npz")
        if kind == "dense":
            same = np.array_equal(a, b)
            err = float(np.abs(a - X[:32]).max())
        elif kind == "sparse":
            same = np.array_equal(a.tocsr().data, b.tocsr().data) and np.array_equal(a.tocsr().indices, b.tocsr().indices)
            err = float(abs(a.tocsr() - X[:32]).max())
        else:
            same = np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
            toks, offs = z["tokens"], z["offsets"]
            err = float(np.abs(a[0] - toks[:offs[32]]).max()) if a[0].shape == toks[:offs[32]].shape else float("inf")
        results[n] = {"twice_byte_identical": bool(same), "replay_max_abs_error_vs_stored": err}
        del lens
        torch.cuda.empty_cache()
    check("B1.probe_replayed_and_stored_slots_reproduced", bool(results) and all(r["twice_byte_identical"] and r["replay_max_abs_error_vs_stored"] == 0.0 for r in results.values()), results=results)


def b2_family(root: Path, fam: str, registry: dict, sample: int) -> dict:
    rows = read_jsonl(root / "cal" / f"{fam}.jsonl")
    summ = json.loads((root / "cal" / f"{fam}.summary.json").read_text(encoding="utf-8"))
    h = blake3.blake3()
    with (root / "cal" / f"{fam}.jsonl").open("rb") as fh:
        for line in fh:
            h.update(line)
    y = np.asarray([r["y"] for r in rows], np.int64)
    hist = {str(int(k)): int(v) for k, v in zip(*np.unique(y, return_counts=True))}
    check(f"B2.{fam}.calibration_digest_histogram_entropy", h.hexdigest() == summ["file_blake3"] and hist == summ["classes"] and abs(own_entropy(y) - summ["H_Y_bits"]) < 1e-9
          and all(r["text_b3"] == blake3.blake3(r["text"].encode("utf-8")).hexdigest() for r in rows), n=len(rows), H=summ["H_Y_bits"])
    rep = json.loads((root / "assay" / f"{fam}.json").read_text(encoding="utf-8"))
    rd = np.load(root / "readings" / f"{fam}.npz", allow_pickle=False)
    pairs, R, C, order = rd["pairs"], rd["R"], rd["C"], [str(x) for x in rd["order"]]
    worst, recomputed = 0.0, 0
    for nm, s in rep["lenses"].items():
        pp = root / "proj" / fam / f"{nm}.npz"
        if not pp.exists() or not s["seeds"] or "error" in s["seeds"][0]:
            continue
        z = np.load(pp)
        bits = own_mixed_ksg(z["P"], z["y"])
        worst = max(worst, abs(bits - s["seeds"][0]["bits"]))
        recomputed += 1
    check(f"B2.{fam}.seed0_bits_recomputed_numpy", recomputed > 0 and worst < 1e-3, lenses=recomputed, max_abs_diff=worst)
    vi = sorted((nm for nm, s in rep["lenses"].items() if s.get("standalone_estimator") == "vinfo"), key=lambda n: blake3.blake3((fam + n).encode()).digest())[:3]
    if vi:
        import vinfo

        vworst = 0.0
        for nm in vi:
            X, absent, kind, z = load_slot(root / "slots" / fam / f"{nm}.npz")
            if kind == "multi":
                import slotio
                X, _ = slotio.maxsim_landmarks(root / "slots" / fam / f"{nm}.npz", absent, fam)
            used = np.flatnonzero(~absent)
            grp = np.unique([str(r["cluster"]) for r in rows], return_inverse=True)[1]
            prior, probe = vinfo.cv_cross_entropy(vinfo.block(X[used]), y[used], grp[used])
            s = rep["lenses"][nm]
            vworst = max(vworst, abs(float((prior - probe).mean()) - s["bits_mean"]), 0.0 if all(x.get("error") == "MNLM_ASSAY_DEGENERATE" for x in s["seeds"]) else float("inf"))
        check(f"B2.{fam}.degenerate_slots_heldout_standalone_replayed", vworst < 1e-6, lenses=vi, max_abs_diff=vworst)
    pick = sorted((nm for nm in order if (root / "proj" / fam / f"{nm}.npz").exists()), key=lambda n: blake3.blake3((fam + n).encode()).digest())[:sample]
    pworst, rworst = 0.0, 0.0
    for nm in pick:
        X, absent, kind, z = load_slot(root / "slots" / fam / f"{nm}.npz")
        zz = np.load(root / "proj" / fam / f"{nm}.npz")
        if kind == "multi":
            import slotio

            X, _lm = slotio.maxsim_landmarks(root / "slots" / fam / f"{nm}.npz", absent, fam)
        used = np.flatnonzero(~absent)
        P = own_project(X[used], 1000)
        pworst = max(pworst, float(np.abs(P - zz["P"]).max()), 0.0 if np.array_equal(used, zz["rows"]) else float("inf"))
        li = order.index(nm)
        sim = rep["lenses"][nm]["sim"]
        for t in range(0, len(pairs), 97):
            i, j = int(pairs[t, 0]), int(pairs[t, 1])
            if absent[i] or absent[j] or kind == "multi":
                continue
            if sim == "neg_cheb":
                D = np.asarray(X.todense()) if sp.issparse(X) else np.asarray(X, np.float64)
                sd = D.std(0)
                Z = np.where(sd > 0, (D - D.mean(0)) / np.where(sd > 0, sd, 1), 0.0)
                r = -float(np.abs(Z[i] - Z[j]).max()) if Z.shape[1] else 0.0
            else:
                r = own_cosine(X, i, j)
            rworst = max(rworst, abs(r - R[li, t]))
    check(f"B2.{fam}.projections_and_readings_rederived_sample", pworst < 1e-4 and rworst < 1e-5, sampled=pick, max_proj_diff=pworst, max_reading_diff=rworst)
    cworst = 0.0
    for a in range(len(order)):
        for b in range(a + 1, len(order)):
            ok = np.isfinite(R[a]) & np.isfinite(R[b])
            if ok.sum() >= 500 and R[a][ok].std() > 0 and R[b][ok].std() > 0:
                cworst = max(cworst, abs(spearmanr(R[a][ok], R[b][ok]).statistic - C[a, b]))
            elif np.isfinite(C[a, b]):
                cworst = float("inf")
    check(f"B2.{fam}.spearman_matrix_recomputed", cworst < 1e-9, max_abs_diff=cworst, lenses=len(order))
    panel = rep["admission"]["panel"]
    idx = {nm: i for i, nm in enumerate(order)}
    import assay
    import fsv_admit

    viol = fsv_admit.rules(rep, C, order, MIN_BITS, MAX_CORR, MIN_COV)
    served = False  # usable information is not a Shannon conditional certificate
    H = summ["H_Y_bits"]
    I = 0.0
    if panel:
        vz = np.load(root / "assay" / f"{fam}.vinfo.npz")
        lb = float((vz["prior"] - vz["probe"]).mean())
        I = lb
        if [str(x) for x in vz["panel"]] != panel or abs(lb - rep["joint"]["vinfo"]["usable_bits"]) > 1e-9 or abs(I - rep["joint"]["bits"]) > 1e-9:
            viol.append(f"panel diagnostic {rep['joint'].get('bits')} differs from held-out usable information {lb}")
        gains, gworst, gviol = fsv_admit.numbers(root, fam, rep, rows, y, load_slot, assay.strata_of, MIN_BITS)
        viol += gviol
        check(f"B2.{fam}.panel_ce_gains_and_five_seed_confirmations_replayed", gworst < fsv_admit.TOL and not gviol, gains=gains, max_abs_diff=gworst, violations=gviol)
    deficit = max(0.0, H - I)
    w = {nm: 1 / (rep["lenses"][nm]["bits_mean"] + 0.01) for nm in panel}
    tot = sum(w.values()) or 1.0
    split_ok = all(abs(rep["sufficiency"]["deficit_split"][nm] - deficit * w[nm] / tot) < 1e-9 for nm in panel)
    neff = None
    if panel:
        ii = [idx[p] for p in panel]
        G = np.abs(np.nan_to_num(C[np.ix_(ii, ii)], nan=0.0))
        np.fill_diagonal(G, 1.0)
        neff = float(np.trace(G) ** 2 / (G ** 2).sum())
    check(f"B2.{fam}.admission_served_neff_sufficiency_rederived", not viol and served == rep["served"] and (neff is None and rep["n_eff"] is None or abs(neff - rep["n_eff"]) < 1e-9)
          and abs(deficit - rep["sufficiency"]["deficit_bits"]) < 1e-9 and split_ok and not rep["sufficiency"]["sufficient"],
          panel=panel, violations=viol, n_eff=neff, I=I, H=H)
    return {"panel": panel, "R": R, "C": C, "order": order, "y": y, "rep": rep}


def b3_negatives(root: Path, registry: dict, pick: list[str], fam_state: dict) -> None:
    from learned import snapshot_dir
    from measure import check_pin

    from measure import pinned

    pins = pinned(root)  # the real frozen panel, read by the same function measure.py uses
    panel_learned = [n for n in pins if registry["lenses"][n]["contract"]["kind"] != "encoder"]
    name = panel_learned[0] if panel_learned else pick[0]
    if not panel_learned:
        pins = {name: registry["lenses"][name]["contract"]["weights_sha256"]}
    c = registry["lenses"][name]["contract"]
    tmp = root.parent.parent / "tmp" / f"fsv12-flip-{run_stamp()}"
    tmp.mkdir(parents=True)
    try:
        src = snapshot_dir(c["repo"])
        f = c["weight_files"][0]
        (tmp / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src / f, tmp / f)
        b = bytearray((tmp / f).read_bytes())
        b[len(b) // 2] ^= 0xFF
        (tmp / f).write_bytes(bytes(b))
        flipped = own_sha256_files(tmp, [f])
        code = None
        try:
            check_pin(pins, {"name": name, "weights_sha256": flipped})
        except Exception as exc:  # noqa: BLE001
            code = getattr(exc, "code", None)
        check("B3.byte_flipped_weights_refuse_frozen_violation", flipped != c["weights_sha256"] and code == "MNLM_LENS_FROZEN_VIOLATION" and pins.get(name) == c["weights_sha256"],
              lens=name, file=f, code=code, pinned_from_panel=bool(panel_learned))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    fam, st = next(((f, s) for f, s in fam_state.items() if s["panel"]), (None, None))
    if st is None:
        check("B3.duplicate_lens_retired_by_correlation_gate", False, reason="no family admitted a lens")
        check("B3.label_shuffled_control_not_admitted", False, reason="no family admitted a lens")
        return
    dup_of = st["panel"][0]
    li = st["order"].index(dup_of)
    # a real duplicate: the admitted lens's slot under a new name, its readings recomputed by assay.readings from the copied
    # file, appended to the readings matrix, and offered to assay.admit as an eligible candidate beside the real panel
    import assay as A

    X, absent, kind, z = load_slot(root / "slots" / fam / f"{dup_of}.npz")
    dup_path = root.parent.parent / "tmp" / f"fsv12-dup-{run_stamp()}.npz"
    dup_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(root / "slots" / fam / f"{dup_of}.npz", dup_path)
    try:
        sim = st["rep"]["lenses"][dup_of]["sim"]
        r_dup = A.readings(dup_path, X, absent, kind, sim, np.load(root / "readings" / f"{fam}.npz")["pairs"])
        R2 = np.vstack([st["R"], r_dup[None, :]])
        order2 = st["order"] + ["__duplicate__"]
        C2 = A.spearman_matrix(R2)
        rho = float(C2[len(order2) - 1, li])
        cands = {nm: dict(v) for nm, v in st["rep"]["lenses"].items()}
        cands["__duplicate__"] = dict(cands[dup_of])
        used = np.flatnonzero(~absent)
        Xv = X
        if kind == "multi":
            import slotio

            Xv, _lm = slotio.maxsim_landmarks(root / "slots" / fam / f"{dup_of}.npz", absent, fam)
        blk = __import__("vinfo").block(Xv[used])
        blocks = {dup_of: blk, "__duplicate__": blk}
        rows = {dup_of: used, "__duplicate__": used}
        panel2, trace2, _, _ = A.admit({dup_of: cands[dup_of], "__duplicate__": cands["__duplicate__"]}, blocks, rows, st["y"], C2, order2, 4)
        dup_step = next((s_["candidates"].get("__duplicate__") for s_ in trace2 if s_["panel"] == [dup_of]), None)
    finally:
        dup_path.unlink(missing_ok=True)
    check("B3.duplicate_lens_retired_by_correlation_gate", rho > MAX_CORR and "__duplicate__" not in panel2 and bool(dup_step) and dup_step.get("skipped") == "correlation",
          family=fam, duplicate_of=dup_of, spearman=rho, gate=MAX_CORR, admit_trace=dup_step, panel=panel2)
    z = np.load(root / "proj" / fam / f"{dup_of}.npz")
    rng = np.random.default_rng(12)
    ys = rng.permutation(z["y"])
    bits_shuffled = own_mixed_ksg(z["P"], ys)
    bits_true = own_mixed_ksg(z["P"], z["y"])
    check("B3.label_shuffled_control_not_admitted", bits_shuffled < MIN_BITS and bits_true >= MIN_BITS, family=fam, lens=dup_of, bits_true=bits_true, bits_shuffled=bits_shuffled)


def b2_panel(root: Path, registry: dict, repo: Path) -> None:
    panel = json.loads((root / "panel.json").read_text(encoding="utf-8"))
    pin = panel["invariants"].get("registry_blake3")
    if registry.get("registry_blake3_before_ledger") != pin and (root / f"registry-{pin}.json").exists():
        registry = json.loads((root / f"registry-{pin}.json").read_text(encoding="utf-8"))  # the content-addressed snapshot the panel was frozen against (readback H2c)
    body = dict(panel)
    for k in ("panel_blake3", "ledger_seq", "ledger_head"):
        body.pop(k, None)
    ids = sorted(l["lens_id"] for l in panel["lenses"])
    ok_ids = blake3.blake3("".join(ids).encode()).hexdigest() == panel["invariants"]["lens_ids_blake3"]
    reg_ok = all(registry["lenses"].get(l["name"], {}).get("lens_id") == l["lens_id"] and registry["lenses"][l["name"]]["contract"]["weights_sha256"] == l["contract"]["weights_sha256"] for l in panel["lenses"])
    copy = repo / "config" / f"{panel['panel_version']}.json"
    copy_ok = copy.exists() and blake3.blake3(copy.read_bytes()).hexdigest() == blake3.blake3((root / "panel.json").read_bytes()).hexdigest()
    from fsv_union import rederive_union

    from lenscommon import RETIRED_QUESTIONS
    admission = json.loads((root / "assay" / "admission.json").read_text(encoding="utf-8"))
    live = [f for f in FAMILIES if f in admission["digests"] and f not in RETIRED_QUESTIONS]
    union, excluded = rederive_union(root, live, MAX_CORR)  # question order, admission order, D1 across the union, unreadable fails closed
    from catalogue import by_name as _by_name

    served = []  # no Shannon conditional certificate from a usable-information probe
    union_ok = all(u == panel["anchor_families"][a]["admitted_union"] for a, u in union.items()) and all(
        sorted(excluded.get(a, [])) == sorted(x["lens"] for x in v["union_excluded"]) for a, v in panel["anchor_families"].items())
    lenses_ok = sorted(l["name"] for l in panel["lenses"]) == sorted({nm for u in union.values() for nm in u})
    gates_ok = sorted(panel["families"]) == sorted(live) and panel["gates"]["families_served_ge5_lenses"] == served and panel["gates"]["families_served_ge3"] == (len(served) >= 3) and union_ok and lenses_ok
    check("B2.panel_invariants_registry_repo_copy_gates", ok_ids and reg_ok and copy_ok and gates_ok, lenses=len(ids), served=served, repo_copy=str(copy), copy_identical=copy_ok,
          union_rederived=union, union_excluded=excluded, union_matches_panel=union_ok, panel_lenses_are_the_union=lenses_ok)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--version", default="p0-v1")
    ap.add_argument("--issue", default="12")
    ap.add_argument("--sample", type=int, default=6)
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[2]))
    args = ap.parse_args()
    take_lease(args.issue, "stage 3 FSV (probe replay on the GPU)")
    paths = Paths(args.root, args.out, args.version)
    root = lens_root(paths)
    from lenscommon import sealed_registry
    from admission_state import load_reports, validate_ledger

    registry = sealed_registry(root)
    admission = json.loads((root / "assay" / "admission.json").read_text(encoding="utf-8"))
    validate_ledger(paths.ledger, registry, admission)
    validated = load_reports(root, registry, admission)
    run = f"stage3-{args.version}-{run_stamp()}"
    pick = b1_registration(root, registry, args.sample)
    fams = [f for f in FAMILIES if f in validated]
    b1_replay(root, registry, pick, fams[0])
    state = {fam: b2_family(root, fam, registry, 3) for fam in fams}
    b2_panel(root, registry, Path(args.repo))
    b3_negatives(root, registry, pick, state)
    rdir = paths.evidence(args.issue, run)
    rdir.mkdir(parents=True, exist_ok=True)
    passed = sum(c["ok"] for c in CHECKS)
    report = {"run": run, "version": args.version, "host": HOST, "passed": passed, "total": len(CHECKS), "checks": CHECKS,
              "inputs": {"registry_file_blake3": blake3.blake3((root / "registry.json").read_bytes()).hexdigest(), "panel_file_blake3": blake3.blake3((root / "panel.json").read_bytes()).hexdigest(),
                         "panel_blake3_ledgered": json.loads((root / "panel.json").read_text(encoding="utf-8")).get("panel_blake3"),
                         "registry_blake3_ledgered": registry.get("registry_blake3_before_ledger")}}
    write_json(rdir / "stage3-fsv.json", report)
    print(f"{passed}/{len(CHECKS)} checks passed; report {rdir / 'stage3-fsv.json'}", flush=True)
    return 0 if passed == len(CHECKS) else 1


if __name__ == "__main__":
    from lenscommon import run_cli
    sys.exit(run_cli(main))
