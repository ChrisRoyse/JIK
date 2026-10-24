"""Stage 2 typed edges (issue #11, spec §18: "citations and references are extracted as typed edges (atom -> cited
identity) and, when the cited item is outside the corpus, recorded as frontier edges, never resolved by guesswork").
Three kinds, each one pass, O(atoms + rows):
  cites      cuyahoga-eighth-district (p0-v1): an opinion's own WebCite comes from its "[Cite as ..., YYYY-Ohio-N.]" header;
             every WebCite mentioned in a sentence atom is an edge to the opinion that owns it, or a frontier edge when no
             opinion in the corpus owns it; reporter citations (Ohio St.3d, Ohio App.3d, N.E.2d/3d) are frontier edges.
  crossrefs  scripture-v1: the openbible-crossrefs rows (from_ref, to_ref, votes) resolved to verse keys of align/.
  anchors    identity-v1 -> scripture-v1: every anchor_refs / refs_expanded / ref of every identity record resolved to a
             verse key of the target version's align/; an unresolved ref refuses (MNLM_EDGES_ANCHOR_UNRESOLVED), because
             the builder guaranteed resolution.
Output: <out>/atoms/<version>/edges/<kind>.npz (+ .summary.json, + .frontier.jsonl for cites), one ledger entry `atoms.edges`.
    python -I scripts/canon/edges.py --root R --out O --version p0-v1 --kind cites
    python -I scripts/canon/edges.py --root R --out O --version identity-v1 --kind anchors --target-version scripture-v1
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

import blake3
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from align_verses import norm_book  # noqa: E402
from common import HOST, Paths, iter_canon_docs, ledger_append, now_iso, read_jsonl, write_json  # noqa: E402

CITES_SOURCE = "cuyahoga-eighth-district"
WEBCITE = re.compile(r"\b((?:19|20)\d{2}-Ohio-\d{1,5})\b")
HEADER = re.compile(r"\[Cite as [^\]]*?,\s*((?:19|20)\d{2}-Ohio-\d{1,5})\.?\]")
REPORTER = re.compile(r"\b(\d{1,3} (?:Ohio St\.(?:2d|3d)|Ohio App\.(?:2d|3d)|N\.E\.(?:2d|3d)) \d{1,4})\b")
KIND_WEBCITE, KIND_WEBCITE_FRONTIER, KIND_REPORTER_FRONTIER = 1, 2, 3
HEADER_WINDOW = 3000
OSIS_REF = re.compile(r"^([1-3]?[A-Za-z]{2,6})\.(\d+)\.(\d+)")  # Gen.1.1, 1Sam.2.3; a range Gen.1.1-Gen.1.3 keys on its first verse
FIELD_ANCHOR, FIELD_EXPANDED, FIELD_REF = 1, 2, 3


class EdgeError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code


def load_keys(align_dir: Path) -> tuple[dict[tuple, int], str]:
    """(book, chapter, verse) -> key_id from verse_keys.jsonl, and the file's blake3 (the identity of the target index)."""
    path = align_dir / "verse_keys.jsonl"
    keys = {(r["book"], r["chapter"], r["verse"]): r["key_id"] for r in read_jsonl(path)}
    return keys, blake3.blake3(path.read_bytes()).hexdigest()


def shards_of(paths: Paths, source: str) -> list[tuple[int, dict]]:
    meta = json.loads((paths.dedup / "population.json").read_text(encoding="utf-8"))
    return [(i, s) for i, s in enumerate(meta["shards"]) if s["source"] == source]


def atoms_of(adir: Path, source: str, shard: str):
    a = np.load(adir / source / f"{shard}.atoms.npz")
    return {k: a[k] for k in ("row", "start", "end", "role")}


def cites(paths: Paths, adir: Path) -> dict:
    shards = shards_of(paths, CITES_SOURCE)
    if not shards:
        raise EdgeError("MNLM_EDGES_SOURCE_MISSING", f"{CITES_SOURCE} not in {paths.version}")
    owner: dict[str, tuple[int, int]] = {}
    owners_dup = 0
    for si, sh in shards:  # pass 1: who owns which WebCite
        for rec in iter_canon_docs(paths.canon(CITES_SOURCE), sh["shard"]):
            m = HEADER.search(rec["text"][:HEADER_WINDOW])
            if m:
                if m.group(1) in owner:
                    owners_dup += 1
                owner.setdefault(m.group(1), (si, rec["row"]))
    cols = collections.defaultdict(list)
    frontier: dict[str, int] = {}
    counts = collections.Counter()
    for si, sh in shards:  # pass 2: mentions in atoms
        a = atoms_of(adir, CITES_SOURCE, sh["shard"])
        order = np.argsort(a["row"], kind="stable")
        rows = a["row"][order]
        bounds = np.flatnonzero(np.diff(rows)) + 1
        per_row = {int(rows[i]): order[i:j] for i, j in zip(np.concatenate([[0], bounds]), np.concatenate([bounds, [len(rows)]]))} if len(rows) else {}
        for rec in iter_canon_docs(paths.canon(CITES_SOURCE), sh["shard"]):
            idx = per_row.get(rec["row"])
            if idx is None:
                continue
            text = rec["text"]
            own = HEADER.search(text[:HEADER_WINDOW])
            own_cite = own.group(1) if own else None
            starts = a["start"][idx]
            for kind_re, resolved_kind, frontier_kind in ((WEBCITE, KIND_WEBCITE, KIND_WEBCITE_FRONTIER), (REPORTER, None, KIND_REPORTER_FRONTIER)):
                for m in kind_re.finditer(text):
                    cite = m.group(1)
                    if cite == own_cite:
                        counts["self"] += 1
                        continue
                    k = int(np.searchsorted(starts, m.start(), side="right") - 1)  # the atom whose span holds the match start
                    atom = int(idx[k])
                    tgt = owner.get(cite) if resolved_kind else None
                    if tgt:
                        cols["kind"].append(resolved_kind)
                        cols["target_shard"].append(tgt[0])
                        cols["target_row"].append(tgt[1])
                        cols["cite_id"].append(-1)
                        counts["resolved"] += 1
                    else:
                        cid = frontier.setdefault(cite, len(frontier))
                        cols["kind"].append(frontier_kind)
                        cols["target_shard"].append(-1)
                        cols["target_row"].append(-1)
                        cols["cite_id"].append(cid)
                        counts["frontier_webcite" if frontier_kind == KIND_WEBCITE_FRONTIER else "frontier_reporter"] += 1
                    cols["shard"].append(si)
                    cols["row"].append(rec["row"])
                    cols["atom"].append(atom)
                    cols["offset"].append(m.start())
    out = adir / "edges"
    out.mkdir(parents=True, exist_ok=True)
    arrays = {k: np.asarray(cols[k], dtype=np.int64) for k in ("shard", "row", "atom", "offset", "kind", "target_shard", "target_row", "cite_id")}
    _save(out / "cites.npz", arrays)
    with (out / "cites.frontier.jsonl").open("w", encoding="utf-8") as fh:
        for cite, cid in sorted(frontier.items(), key=lambda kv: kv[1]):
            fh.write(json.dumps({"cite_id": cid, "cite": cite}) + "\n")
    return {"kind": "cites", "source": CITES_SOURCE, "owners": len(owner), "owner_header_duplicates": owners_dup, "edges": int(len(arrays["shard"])),
            "resolved": counts["resolved"], "frontier_webcite": counts["frontier_webcite"], "frontier_reporter": counts["frontier_reporter"], "self_mentions": counts["self"],
            "distinct_frontier_cites": len(frontier), "distinct_targets": int(len(np.unique(arrays["target_row"][arrays["kind"] == KIND_WEBCITE]))) if len(arrays["shard"]) else 0,
            "arrays_blake3": _digest(arrays)}


def crossrefs(paths: Paths, adir: Path) -> dict:
    keys, keys_b3 = load_keys(adir / "align")
    src = "openbible-crossrefs"
    cols = collections.defaultdict(list)
    counts = collections.Counter()
    unresolved: collections.Counter = collections.Counter()
    for si, sh in shards_of(paths, src):
        for rec in iter_canon_docs(paths.canon(src), sh["shard"]):
            f = rec["f"]
            ids = []
            for name in ("from_ref", "to_ref"):
                m = OSIS_REF.match(f.get(name) or "")
                kid = keys.get((norm_book(m.group(1)), int(m.group(2)), int(m.group(3)))) if m else None
                if kid is None:
                    unresolved[(f.get(name) or "")[:12]] += 1
                ids.append(-1 if kid is None else kid)
            counts["resolved" if min(ids) >= 0 else "frontier"] += 1
            cols["shard"].append(si)
            cols["row"].append(rec["row"])
            cols["from_key"].append(ids[0])
            cols["to_key"].append(ids[1])
            cols["votes"].append(int(f.get("votes") or 0))
            cols["range"].append(int("-" in (f.get("to_ref") or "")))
    out = adir / "edges"
    out.mkdir(parents=True, exist_ok=True)
    arrays = {k: np.asarray(cols[k], dtype=np.int64) for k in ("shard", "row", "from_key", "to_key", "votes", "range")}
    _save(out / "crossrefs.npz", arrays)
    return {"kind": "crossrefs", "source": src, "rows": int(len(arrays["row"])), "resolved_both": counts["resolved"], "frontier": counts["frontier"],
            "unresolved_books_top": dict(unresolved.most_common(10)), "target_keys_blake3": keys_b3, "arrays_blake3": _digest(arrays)}


def anchors(paths: Paths, adir: Path, target: Paths) -> dict:
    keys, keys_b3 = load_keys(target.out / "atoms" / target.version / "align")
    src = "identity-v1"
    cols = collections.defaultdict(list)
    counts = collections.Counter()
    bad = []
    for si, sh in shards_of(paths, src):
        for rec in iter_canon_docs(paths.canon(src), sh["shard"]):
            f = rec["f"]
            for field, kind in (("anchor_refs", FIELD_ANCHOR), ("refs_expanded", FIELD_EXPANDED), ("ref", FIELD_REF)):
                vals = f.get(field)
                vals = [vals] if isinstance(vals, str) else (vals or [])
                for ref in vals:
                    parts = ref.split(".")
                    kid = keys.get((parts[0], int(parts[1]), int(parts[2]))) if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit() else None
                    if kid is None:
                        bad.append(f"{sh['shard']}/{rec['row']}:{ref}")
                        continue
                    cols["shard"].append(si)
                    cols["row"].append(rec["row"])
                    cols["key_id"].append(kid)
                    cols["field"].append(kind)
                    counts[field] += 1
    if bad:
        raise EdgeError("MNLM_EDGES_ANCHOR_UNRESOLVED", f"{len(bad)} refs resolve to no verse key: {bad[:5]}")
    out = adir / "edges"
    out.mkdir(parents=True, exist_ok=True)
    arrays = {k: np.asarray(cols[k], dtype=np.int64) for k in ("shard", "row", "key_id", "field")}
    _save(out / "anchors.npz", arrays)
    return {"kind": "anchors", "source": src, "target_version": target.version, "edges": int(len(arrays["row"])), "per_field": dict(counts),
            "distinct_keys": int(len(np.unique(arrays["key_id"]))) if len(arrays["row"]) else 0, "records_with_edges": int(len(set(zip(cols["shard"], cols["row"])))),
            "target_keys_blake3": keys_b3, "arrays_blake3": _digest(arrays)}


def _digest(arrays: dict) -> str:
    h = blake3.blake3()
    for k in sorted(arrays):
        h.update(k.encode())
        h.update(arrays[k].tobytes())
    return h.hexdigest()


def _save(path: Path, arrays: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as fh:
        np.savez(fh, **arrays)
    tmp.replace(path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--version", required=True)
    ap.add_argument("--kind", required=True, choices=["cites", "crossrefs", "anchors"])
    ap.add_argument("--target-version", default="")
    args = ap.parse_args()
    paths = Paths(args.root, args.out, args.version)
    adir = paths.out / "atoms" / args.version
    if args.kind == "cites":
        summary = cites(paths, adir)
    elif args.kind == "crossrefs":
        summary = crossrefs(paths, adir)
    else:
        if not args.target_version:
            raise EdgeError("MNLM_EDGES_TARGET_MISSING", "--target-version is required for anchors")
        summary = anchors(paths, adir, Paths(args.root, args.out, args.target_version))
    summary.update(version=args.version, host=HOST, ts=now_iso())
    write_json(adir / "edges" / f"{args.kind}.summary.json", summary)
    e = ledger_append(paths, f"edges@{HOST}", "atoms.edges", f"stage2/{args.version}/{args.kind}", summary)
    summary["ledger_seq"], summary["ledger_head"] = e["seq"], e["hash"]
    write_json(adir / "edges" / f"{args.kind}.summary.json", summary)
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
