"""Stage 1 ops 2-3 at document granularity (issue #8): population table, exact-document layer, global near-duplicate layer.

    python -I scripts/canon/dedup.py --root R --out O --version V --sources a,b,c [--step population|exact|near|all]

population: every doc of every sealed canonical shard of the named sources (DEDUP_EXCLUDED sources never enter), in
  keep-first order (SOURCE_PRIORITY, shard name, row). Arrays under <out>/dedup/<v>/population.npz; tables in population.json.
exact: BLAKE3-128 collisions over the population (one lexsort, O(n log n)); the first in population order is kept.
near: MinHash-LSH over valid prose docs that survived exact (per band: one argsort, runs of equal keys link every member
  to the run head: O(n log n) per band, never all-pairs); connected components; representative = first in population
  order; estimated Jaccard (signature agreement) recorded per removed doc.
Every removed doc is recorded with the index and ref of the doc it duplicates (npy arrays + a jsonl.zst listing); an
exact kept copy may itself be near-removed afterwards (p0-v1: 9,745 of 588,694), so resolving to the unique set means
following exact.kept_of then near.kept_of once (near kept copies are never removed). Band-collision partners that
ended up kept without a >= NEAR_FLOOR link are not persisted; only links at or above the floor are.
Nothing is deleted from canon or raw. Outputs are order-deterministic (PC-38).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import zstandard
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import HOST, Paths, digest_file, ledger_append, load_sidecars, now_iso, read_jsonl, sealed_shards, write_json  # noqa: E402
from contracts import DEDUP_EXCLUDED, PROSE, SOURCE_PRIORITY, near_removable  # noqa: E402
from minhash import BANDS, PERMS  # noqa: E402
from artifact_seals import Audit, refuse  # noqa: E402

NEAR_FLOOR, NEAR_K = 0.6, 32  # signature-agreement floor for a removal, and kept members consulted per component


# ------------------------------------------------------------------------------------------------------------ population
def build_population(paths: Paths, sources: list[str]) -> dict:
    if len(sources) != len(set(sources)) or len(sources) > 256:
        refuse("MNLM_DEDUP_SOURCES", "duplicate sources or uint8 source-index overflow")
    from training_scope import require_training_source
    for source in sources:
        require_training_source(source)
    srcs = sorted((s for s in sources if s not in DEDUP_EXCLUDED), key=lambda s: (SOURCE_PRIORITY.get(s, 99), s))
    unknown = [s for s in srcs if s not in SOURCE_PRIORITY]
    if unknown:
        raise SystemExit(f"MNLM_DEDUP_NO_PRIORITY: {unknown} have no keep-first priority (contracts.SOURCE_PRIORITY)")
    shards, src_i, shard_i, rows, b3s, ns, prose = [], [], [], [], [], [], []
    for si, s in enumerate(srcs):
        cdir = paths.canon(s)
        for r in sealed_shards(cdir):
            if r["kind"] == "alias":
                continue  # the alias layer is not a dedup population (spec §17 op 4)
            b3, rw, n = load_sidecars(cdir, r["shard"])
            if len(b3) != r["docs"]:
                raise SystemExit(f"MNLM_DEDUP_SIDECAR_COUNT {s}/{r['shard']}: {len(b3)} != manifest {r['docs']}")
            k = len(shards)
            shards.append({"source": s, "shard": r["shard"], "canon_blake3": r["out_blake3"], "docs": r["docs"], "kind": r["kind"]})
            src_i.append(np.full(len(b3), si, dtype=np.uint8))
            shard_i.append(np.full(len(b3), k, dtype=np.uint32))
            rows.append(rw)
            b3s.append(b3)
            ns.append(n)
            prose.append(np.full(len(b3), r["kind"] == PROSE, dtype=bool))
    if not shards:
        raise SystemExit("MNLM_DEDUP_EMPTY_POPULATION: no sealed shards for the named sources")
    empty = [s for s in srcs if not any(sh["source"] == s for sh in shards)]
    if empty:  # a named source with no sealed shard must not vanish from the population silently (run-11 class)
        raise SystemExit(f"MNLM_DEDUP_SOURCE_EMPTY: {empty} contribute no sealed shard")
    pop = {"src": np.concatenate(src_i), "shard": np.concatenate(shard_i), "row": np.concatenate(rows), "b3": np.concatenate(b3s),
           "n": np.concatenate(ns), "prose": np.concatenate(prose)}
    paths.dedup.mkdir(parents=True, exist_ok=True)
    audit = Audit(paths)
    parents = audit.begin("population")
    tmp = paths.dedup / f"population.npz.tmp.{HOST}"
    with tmp.open("wb") as f:
        np.savez(f, **pop)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, paths.dedup / "population.npz")
    meta = {"version": paths.version, "sources": srcs, "shards": shards, "docs": int(len(pop["n"])), "prose_docs": int(pop["prose"].sum()),
            "canonical_bytes": int(pop["n"].sum()), "population_blake3": digest_file(paths.dedup / "population.npz")[0], "host": HOST, "ts": now_iso()}
    write_json(paths.dedup / "population.json", meta)
    audit.finish("population", parents)
    return meta


def load_population(paths: Paths, audit=None) -> tuple[dict, dict]:
    """The arrays and the table are one artifact: the table names the arrays' digest, and a pair from two generations
    (a crash between the writes, or a rebuilt population with an older table) refuses instead of misaligning shards."""
    audit = audit or Audit(paths)
    audit.require("population")
    meta = json.loads((paths.dedup / "population.json").read_text(encoding="utf-8"))
    from training_scope import require_training_source
    for source in meta["sources"]:
        require_training_source(source)
    want = meta.get("population_blake3")
    sealed = json.loads((paths.dedup / "population.seal.json").read_text(encoding="utf-8"))
    if want != sealed["files"]["population.npz"]["blake3"]:
        raise SystemExit(f"MNLM_DEDUP_POPULATION_MISMATCH {paths.dedup}: population.npz is not the one population.json names")
    with np.load(paths.dedup / "population.npz", allow_pickle=False) as z:
        pop = {k: z[k] for k in z.files}
    from population_checks import validate
    validate(pop, meta, paths.version)
    return pop, meta


def ref_of(meta: dict, pop: dict, i: int) -> str:
    sh = meta["shards"][int(pop["shard"][i])]
    return f"{sh['source']}|{sh['shard']}|{int(pop['row'][i])}"


def write_listing(path: Path, meta: dict, pop: dict, removed: np.ndarray, kept: np.ndarray, extra: dict | None = None) -> None:
    with path.open("wb") as fh, zstandard.ZstdCompressor(level=6).stream_writer(fh) as zw:
        for j, (r, k) in enumerate(zip(removed.tolist(), kept.tolist())):
            rec = {"removed": r, "ref": ref_of(meta, pop, r), "dup_of": k, "dup_of_ref": ref_of(meta, pop, k)}
            if extra:
                rec.update({name: arr[j].item() for name, arr in extra.items()})
            zw.write((json.dumps(rec, sort_keys=True) + "\n").encode("utf-8"))


# ----------------------------------------------------------------------------------------------------------------- exact
def exact_layer(paths: Paths) -> dict:
    audit = Audit(paths)
    pop, meta = load_population(paths, audit)
    parents = audit.begin("exact")
    b3 = pop["b3"].view(np.uint64).reshape(-1, 2)
    order = np.lexsort((b3[:, 1], b3[:, 0]))  # population index ascending inside ties: lexsort is stable
    s = b3[order]
    same = np.all(s[1:] == s[:-1], axis=1)
    starts = np.flatnonzero(np.concatenate(([True], ~same)))
    sizes = np.diff(np.concatenate((starts, [len(s)])))
    grp = np.repeat(np.arange(len(starts)), sizes)
    head = order[starts][grp]  # smallest population index in each group (stable sort keeps ascending order)
    removed_mask = order != head
    removed, kept = order[removed_mask], head[removed_mask]
    o2 = np.argsort(removed, kind="stable")
    removed, kept = removed[o2], kept[o2]
    np.save(paths.dedup / "exact.removed.npy", removed)
    np.save(paths.dedup / "exact.kept_of.npy", kept)
    write_listing(paths.dedup / "exact.jsonl.zst", meta, pop, removed, kept)
    per_src = {s: int(np.sum(pop["src"][removed] == i)) for i, s in enumerate(meta["sources"])}
    summary = {"layer": "exact", "docs": int(len(b3)), "groups": int(len(starts)), "removed": int(len(removed)),
               "removed_bytes": int(pop["n"][removed].sum()), "removed_per_source": per_src, "largest_group": int(sizes.max()), "host": HOST, "ts": now_iso()}
    write_json(paths.dedup / "exact.summary.json", summary)
    audit.finish("exact", parents)
    return summary


# ------------------------------------------------------------------------------------------------------------------ near
def raise_nofile(want: int = 100000) -> None:
    """The near layer memory-maps one signature file per prose shard (9,787 on scripture-v1); the soft limit is raised
    here so the step does not depend on the calling shell's ulimit."""
    try:
        import resource

        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if soft < want:
            resource.setrlimit(resource.RLIMIT_NOFILE, (min(want, hard) if hard != resource.RLIM_INFINITY else want, hard))
    except (ImportError, ValueError, OSError):
        pass


def near_layer(paths: Paths) -> dict:
    raise_nofile()
    audit = Audit(paths)
    pop, meta = load_population(paths, audit)
    parents = audit.begin("near")
    n = len(pop["n"])
    alive = np.ones(n, dtype=bool)
    alive[np.load(paths.dedup / "exact.removed.npy")] = False
    cand = np.flatnonzero(alive & pop["prose"])  # population indices of near-dup candidates
    bands = np.zeros((len(cand), BANDS), dtype=np.uint64)
    valid = np.zeros(len(cand), dtype=bool)
    pos = 0
    sig_files: list[tuple[int, int, Path]] = []  # (cand start, cand end, sig path) per shard, for the Jaccard estimate
    sig_manifests: dict[Path, dict[tuple[str, str], dict]] = {}  # one read per source (9,473 Sefaria shards re-read it 9,473 times before)
    for k, sh in enumerate(meta["shards"]):
        if sh["kind"] != PROSE:
            continue
        sdir = paths.sig(sh["source"])
        if sdir not in sig_manifests:
            sig_manifests[sdir] = {(r["shard"], r.get("canon_blake3")): r for r in read_jsonl(sdir / "manifest.jsonl")}
        rows = [sig_manifests[sdir][(sh["shard"], sh["canon_blake3"])]] if (sh["shard"], sh["canon_blake3"]) in sig_manifests[sdir] else []
        if not rows:
            raise SystemExit(f"MNLM_DEDUP_SIG_MISSING {sh['source']}/{sh['shard']}: no minhash manifest row for canon {sh['canon_blake3'][:12]}")
        from minhash import accept_signature
        accept_signature(sdir, sh["shard"], sh["canon_blake3"])
        b = np.load(sdir / f"{sh['shard']}.bands.npy")
        v = np.load(sdir / f"{sh['shard']}.valid.npy")
        if len(b) != sh["docs"]:
            raise SystemExit(f"MNLM_DEDUP_SIG_COUNT {sh['source']}/{sh['shard']}: {len(b)} != {sh['docs']}")
        first = int(np.searchsorted(pop["shard"], k, side="left"))  # population index of this shard's first doc
        local = cand[(cand >= first) & (cand < first + sh["docs"])] - first
        m = len(local)
        bands[pos:pos + m] = b[local]
        valid[pos:pos + m] = v[local]
        sig_files.append((pos, pos + m, sdir / f"{sh['shard']}.sig.npy", first))
        pos += m
    assert pos == len(cand)
    edges_src, edges_dst = [], []
    vidx = np.flatnonzero(valid)
    for b in range(BANDS):
        if not len(vidx):
            edges_src.append(np.empty(0, dtype=np.int64))
            edges_dst.append(np.empty(0, dtype=np.int64))
            continue
        keys = bands[vidx, b]
        order = np.argsort(keys, kind="stable")
        sk = keys[order]
        same = sk[1:] == sk[:-1]
        starts = np.flatnonzero(np.concatenate(([True], ~same)))
        sizes = np.diff(np.concatenate((starts, [len(sk)])))
        grp = np.repeat(np.arange(len(starts)), sizes)
        head = order[starts][grp]
        link = order != head
        edges_src.append(vidx[order[link]])
        edges_dst.append(vidx[head[link]])
    es, ed = np.concatenate(edges_src), np.concatenate(edges_dst)
    g = coo_matrix((np.ones(len(es), dtype=np.int8), (es, ed)), shape=(len(cand), len(cand)))
    ncomp, label = connected_components(g, directed=False)
    order = np.argsort(label, kind="stable")  # within a component, ascending candidate index = keep-first order
    sl = label[order]
    starts = np.flatnonzero(np.concatenate(([True], sl[1:] != sl[:-1]))) if len(sl) else np.empty(0, dtype=np.int64)
    sizes = np.diff(np.concatenate((starts, [len(sl)])))
    sig_cache: dict[Path, np.ndarray] = {}
    chunk_starts = np.array([a for a, _, _, _ in sig_files], dtype=np.int64)  # ascending by construction

    def sig_of(ci: int) -> np.ndarray:
        # bisect over the chunk starts, not a linear scan of 9,787 chunks per call (measured 144 s of 287 s on scripture-v1)
        j = int(np.searchsorted(chunk_starts, ci, side="right")) - 1
        a, z, p, first = sig_files[j]
        if not (a <= ci < z):
            raise IndexError(ci)
        arr = sig_cache.get(p)
        if arr is None:
            arr = sig_cache[p] = np.load(p, mmap_mode="r")
        return np.asarray(arr[int(cand[ci] - first)])

    # Band collisions chain transitively (measured on p0micro-v1 before this floor: largest component 34,293 docs, median
    # signature agreement to the representative 0.55). A doc is removed only when its signature agrees with an already-kept
    # member of its component at or above NEAR_FLOOR (the first NEAR_K kept members are consulted); otherwise it is kept and
    # becomes a reference for later members. Every removal therefore names a kept doc it actually resembles.
    removable = np.array([near_removable(s) for s in meta["sources"]], dtype=bool)[pop["src"][cand]]  # per candidate
    rem_l, kept_l, est_l, unlinked = [], [], [], 0
    link_d, link_k, link_e, link_r = [], [], [], []  # every verified link, removed or exempt
    for s0, sz in zip(starts.tolist(), sizes.tolist()):
        if sz < 2:
            continue
        members = order[s0:s0 + sz]
        members = members[valid[members]]
        if len(members) < 2:
            continue
        kept_idx = [int(members[0])]
        kept_sigs = [sig_of(int(members[0]))]
        for ci in members[1:].tolist():
            sg = sig_of(ci)
            ag = (np.stack(kept_sigs[:NEAR_K]) == sg).mean(axis=1)
            j = int(np.argmax(ag))
            if ag[j] >= NEAR_FLOOR:
                link_d.append(ci)
                link_k.append(kept_idx[j])
                link_e.append(float(ag[j]))
                link_r.append(bool(removable[ci]))
                if removable[ci]:
                    rem_l.append(ci)
                    kept_l.append(kept_idx[j])
                    est_l.append(float(ag[j]))
            else:
                unlinked += 1
                if len(kept_idx) < NEAR_K:
                    kept_idx.append(ci)
                    kept_sigs.append(sg)
    rem_c = np.asarray(rem_l, dtype=np.int64)
    kept_c = np.asarray(kept_l, dtype=np.int64)
    est = np.asarray(est_l, dtype=np.float32)
    o2 = np.argsort(rem_c, kind="stable")
    rem_c, kept_c, est = rem_c[o2], kept_c[o2], est[o2]
    links = np.stack([cand[np.asarray(link_d, dtype=np.int64)], cand[np.asarray(link_k, dtype=np.int64)]], axis=1) if link_d else np.empty((0, 2), dtype=np.int64)
    link_est = np.asarray(link_e, dtype=np.float32)
    link_removed = np.asarray(link_r, dtype=bool)
    lo = np.argsort(links[:, 0], kind="stable")
    links, link_est, link_removed = links[lo], link_est[lo], link_removed[lo]
    np.save(paths.dedup / "near.links.npy", links)
    np.save(paths.dedup / "near.links_est.npy", link_est)
    np.save(paths.dedup / "near.links_removed.npy", link_removed)
    exempt_per_src = {s: int(np.sum((pop["src"][links[:, 0]] == i) & ~link_removed)) for i, s in enumerate(meta["sources"])}
    removed, kept = cand[rem_c], cand[kept_c]
    np.save(paths.dedup / "near.removed.npy", removed)
    np.save(paths.dedup / "near.kept_of.npy", kept)
    np.save(paths.dedup / "near.est_jaccard.npy", est)
    write_listing(paths.dedup / "near.jsonl.zst", meta, pop, removed, kept, {"est_jaccard": est})
    per_src = {s: int(np.sum(pop["src"][removed] == i)) for i, s in enumerate(meta["sources"])}
    summary = {"layer": "near", "candidates": int(len(cand)), "valid": int(valid.sum()), "too_short_kept": int((~valid).sum()),
               "band_links": int(len(es)), "components": int(ncomp), "clusters_ge2": int(np.sum(sizes >= 2)), "removed": int(len(removed)),
               "unlinked_kept": int(unlinked), "near_floor": NEAR_FLOOR, "near_k": NEAR_K, "links": int(len(links)),
               "exempt_linked_kept": int(np.sum(~link_removed)), "exempt_linked_kept_per_source": exempt_per_src,
               "removable_sources": [s for s in meta["sources"] if near_removable(s)],
               "removed_bytes": int(pop["n"][removed].sum()), "removed_per_source": per_src, "largest_cluster": int(sizes.max()) if len(sizes) else 0,
               "est_jaccard_mean": float(est.mean()) if len(est) else None, "est_jaccard_p10": float(np.percentile(est, 10)) if len(est) else None,
               "perms": PERMS, "bands": BANDS, "host": HOST, "ts": now_iso()}
    write_json(paths.dedup / "near.summary.json", summary)
    audit.finish("near", parents)
    return summary


# ---------------------------------------------------------------------------------------------------------------- unique
def unique_layer(paths: Paths) -> dict:
    audit = Audit(paths)
    pop, meta = load_population(paths, audit)
    parents = audit.begin("unique")
    n = len(pop["n"])
    alive = np.ones(n, dtype=bool)
    ex, nr = np.load(paths.dedup / "exact.removed.npy"), np.load(paths.dedup / "near.removed.npy")
    alive[ex] = False
    alive[nr] = False
    uniq = np.flatnonzero(alive)
    np.save(paths.dedup / "unique.npy", uniq)
    per_src = {}
    for i, s in enumerate(meta["sources"]):
        m = pop["src"] == i
        per_src[s] = {"docs": int(m.sum()), "exact_removed": int(np.sum(pop["src"][ex] == i)), "near_removed": int(np.sum(pop["src"][nr] == i)),
                      "unique": int(np.sum(m & alive)), "unique_bytes": int(pop["n"][m & alive].sum())}
    summary = {"version": paths.version, "docs": int(n), "exact_removed": int(len(ex)), "near_removed": int(len(nr)), "unique": int(len(uniq)),
               "unique_prose": int(np.sum(alive & pop["prose"])), "unique_bytes": int(pop["n"][uniq].sum()),
               "unique_prose_bytes": int(pop["n"][alive & pop["prose"]].sum()), "per_source": per_src, "host": HOST, "ts": now_iso()}
    write_json(paths.dedup / "unique.summary.json", summary)
    digests = {f: digest_file(paths.dedup / f)[0] for f in ("population.npz", "exact.removed.npy", "exact.kept_of.npy", "near.removed.npy", "near.kept_of.npy", "unique.npy")}
    e = ledger_append(paths, f"dedup@{HOST}", "dedup.layers", paths.version, {"summary": summary, "digests": digests})
    summary["ledger_seq"], summary["ledger_head"] = e["seq"], e["hash"]
    write_json(paths.dedup / "unique.summary.json", summary)
    audit.finish("unique", parents)
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--version", required=True)
    ap.add_argument("--sources", required=True)
    ap.add_argument("--step", default="all", choices=["population", "exact", "near", "unique", "all"])
    args = ap.parse_args()
    from training_scope import require_training_source
    for source in args.sources.split(","):
        require_training_source(source)
    paths = Paths(args.root, args.out, args.version)
    t0 = time.time()
    steps = ["population", "exact", "near", "unique"] if args.step == "all" else [args.step]
    for st in steps:
        t1 = time.time()
        if st == "population":
            r = build_population(paths, args.sources.split(","))
            print(f"population: {r['docs']} docs ({r['prose_docs']} prose), {r['canonical_bytes']} B over {len(r['shards'])} shards")
        elif st == "exact":
            r = exact_layer(paths)
            print(f"exact: {r['removed']} removed of {r['docs']} ({r['removed_bytes']} B); largest group {r['largest_group']}")
        elif st == "near":
            r = near_layer(paths)
            print(f"near: {r['removed']} removed of {r['candidates']} candidates; {r['clusters_ge2']} clusters; est J mean {r['est_jaccard_mean']}")
        else:
            r = unique_layer(paths)
            print(f"unique: {r['unique']} of {r['docs']} docs; prose unique {r['unique_prose']} ({r['unique_prose_bytes']} B); ledger seq {r['ledger_seq']}")
        print(f"  [{st} {round(time.time() - t1, 1)} s]", flush=True)
    print(f"total {round(time.time() - t0, 1)} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
