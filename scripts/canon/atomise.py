"""Stage 2 (issue #11, spec §18): atomise the unique population of a dedup version into atoms stored as character
offsets into the canonical text. Per canon shard: `<out>/atoms/<version>/<source>/<shard>.atoms.npz` with arrays
row (int32: the canon record row), start, end (int32 chars), role (uint8: sentence/ws/row/cut/verse/code/struct),
unit (uint8: code unit kind, 0 outside code), cut_of (int64: the population index of the duplicate a cut span points to,
-1 otherwise), and a manifest line with counts and digests.
Rules (contract atoms-v3): prose records are segmented (segment.py) with fenced code blocks and markdown headings handled
by code_atoms.py (code units by the CPython ast, fence and heading lines structural); a verse, heading, title or
superscription record and an identity exemplar (one speaker span) are one atom (role verse); row/alias/other kinds are
one atom each (role row); exact-removed records are not atomised (they are reached through exact.kept_of); a recorded
exact-substring span is applied as a cut only when its duplicate lies in the same source (p0) or the same edition
(scripture), and an atom overlapping a cut is split at the cut boundaries. Conservation (spans tile the text) is checked
for every record and refuses with MNLM_ATOMS_CONSERVATION. An existing atom tree for the version refuses
(MNLM_ATOMS_TREE_EXISTS) unless --retire moves it aside, so a tree never mixes two runs.
Cost: one streaming pass per shard, O(chars + spans); shards in parallel.
    python -I scripts/canon/atomise.py --root R --out O --version scripture-v1 [--sources a,b] [--workers 24] [--retire]
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import blake3
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from code_atoms import CODE_ATOMIZER_ID, PY_VERSION, ROLE_CODE, ROLE_STRUCT, UNIT_NAMES, atomise_text  # noqa: E402,F401
from common import HOST, Paths, iter_canon_docs, ledger_append, now_iso, run_stamp, write_json  # noqa: E402
from segment import ROLE_CUT, ROLE_ROW, ROLE_VERSE, SEGMENTER_ID, conserved  # noqa: E402

CONTRACT_ID = "atoms-v3"
WRITER_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
ARRAYS = ("row", "start", "end", "role", "unit", "cut_of")
VERSE_ROLES = {"verse", "title", "heading", "superscription"}
MULTI_EDITION_SOURCES = {"ebible-english", "crosswire-sword-pd", "sefaria-export", "sefaria-export-nc"}


class AtomError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code

    def envelope(self):
        return {"code": self.code, "message": str(self),
                "remediation": "Rebuild the affected Stage 2 generation from accepted inputs in a fresh tree."}


def _byte_to_char(text: str, tb: bytes, b: int) -> int:
    """Char offset of byte offset b (b is on a UTF-8 boundary: Stage 1 snapped every span)."""
    if b <= 0:
        return 0
    if b >= len(tb):
        return len(text)
    return len(tb[:b].decode("utf-8"))


def _apply_cuts(text: str, tb: bytes, spans: list[tuple[int, int, int, int]], cuts: list[tuple[int, int, int]]) -> list[tuple[int, int, int, int, int]]:
    """Split atoms at cut boundaries; returns (start, end, role, unit, cut_of). cuts: (byte_start, byte_end, dup_doc)."""
    if not cuts:
        return [(s, e, r, u, -1) for s, e, r, u in spans]
    cc = sorted((_byte_to_char(text, tb, bs), _byte_to_char(text, tb, be), d) for bs, be, d in cuts)
    bounds = sorted({x for s, e, _ in cc for x in (s, e)})
    out: list[tuple[int, int, int, int, int]] = []
    for s, e, r, u in spans:
        pts = [s] + [b for b in bounds[bisect.bisect_right(bounds, s):bisect.bisect_left(bounds, e)]] + [e]
        for a, b in zip(pts, pts[1:]):
            inside = next((d for cs, ce, d in cc if cs <= a and b <= ce), -1)
            out.append((a, b, ROLE_CUT if inside >= 0 else r, u, inside))
    return out


def record_spans(rec: dict) -> tuple[list[tuple[int, int, int, int]], dict]:
    """(start, end, role, unit) spans for one canon record and the code statistics of the record."""
    text, f = rec["text"], rec["f"]
    if not text:
        return [], {}
    if rec["kind"] == "prose" and (f.get("role") in VERSE_ROLES or f.get("type") == "exemplar"):
        return [(0, len(text), ROLE_VERSE, 0)], {}
    if rec["kind"] == "prose":
        return atomise_text(text)
    return [(0, len(text), ROLE_ROW, 0)], {}


def atomise_shard(job: dict) -> dict:
    src, shard, canon_dir, out_dir = job["source"], job["shard"], Path(job["canon_dir"]), Path(job["out_dir"])
    from training_scope import require_training_source
    require_training_source(src)
    from atoms_publication import descriptor, identity, sync_dir
    binding = descriptor(job)
    unique_rows: set[int] = set(job["unique_rows"])
    cuts_by_row: dict[int, list] = {int(k): v for k, v in job["cuts"].items()}
    cols: dict[str, list] = {k: [] for k in ARRAYS}
    docs = atoms = chars = 0
    code = {"fences": 0, "unclosed": 0, "unparsed": 0}
    t0 = time.time()
    seen, previous = set(), -1
    for rec in iter_canon_docs(canon_dir, shard, job.get("canon_blake3")):
        row = rec["row"]
        if type(row) is not int or row <= previous:
            raise AtomError("MNLM_ATOMS_ROWS", f"{src}/{shard}: duplicate or unordered canonical row")
        previous = row
        if row not in unique_rows:
            continue
        seen.add(row)
        text = rec["text"]
        spans, st = record_spans(rec)
        for k in code:
            code[k] += st.get(k, 0)
        if not conserved(text, [(s, e, r) for s, e, r, _ in spans]):
            raise AtomError("MNLM_ATOMS_CONSERVATION", f"{src}/{shard} row {row}: spans do not tile the text")
        tb = text.encode("utf-8")
        for s, e, r, u, d in _apply_cuts(text, tb, spans, cuts_by_row.get(row, [])):
            for k, v in zip(ARRAYS, (row, s, e, r, u, d)):
                cols[k].append(v)
            atoms += 1
        docs += 1
        chars += len(text)
    if seen != unique_rows:
        raise AtomError("MNLM_ATOMS_ROWS_MISSING", f"{src}/{shard}: {len(unique_rows - seen)} requested rows missing; rebuild canonical inputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    dtypes = {"row": np.int32, "start": np.int32, "end": np.int32, "role": np.uint8, "unit": np.uint8, "cut_of": np.int64}
    arrays = {k: np.asarray(cols[k], dtype=dtypes[k]) for k in ARRAYS}
    tmp = out_dir / f"{shard}.atoms.npz.tmp"
    final = out_dir / f"{shard}.atoms.npz"
    with tmp.open("wb") as fh:
        np.savez(fh, **arrays)
        fh.flush()
        os.fsync(fh.fileno())
    tmp.replace(final)
    sync_dir(out_dir)
    h = blake3.blake3()
    for k in ARRAYS:
        h.update(arrays[k].tobytes())
    role_counts = {int(k): int(v) for k, v in zip(*np.unique(arrays["role"], return_counts=True))}
    unit_counts = {int(k): int(v) for k, v in zip(*np.unique(arrays["unit"], return_counts=True))}
    return {**binding, "rows_sha256": identity(sorted(seen)), "source": src, "shard": shard, "contract": CONTRACT_ID, "writer_sha256": WRITER_SHA256, "code_atomizer": CODE_ATOMIZER_ID, "segmenter": SEGMENTER_ID, "python": PY_VERSION, "docs": docs, "atoms": atoms, "chars": chars,
            "cut_atoms": int((arrays["role"] == ROLE_CUT).sum()), "roles": role_counts, "units": unit_counts, "code": code,
            "arrays_blake3": h.hexdigest(), "file_bytes": final.stat().st_size, "host": HOST, "ts": now_iso(), "secs": round(time.time() - t0, 3)}


def plan_jobs(paths: Paths, sources: list[str] | None) -> tuple[list[dict], dict]:
    d = paths.dedup
    from artifact_seals import Audit
    from dedup import load_population
    audit = Audit(paths)
    stage1 = audit.require("spans")
    pop, meta = load_population(paths, audit)
    unique = np.load(d / "unique.npy")
    spans = np.load(d / "spans.npy")  # columns: doc, byte_start, byte_end, dup_doc, dup_byte_start
    shards = meta["shards"]
    shard_idx = pop["shard"]
    row_arr = pop["row"]
    src_of_doc = np.array([s["source"] for s in shards])[shard_idx]
    # p0: a cut applies only within one source. scripture: only within one edition, which is the shard for the
    # multi-edition sources (one zip, module or version file per shard) and the source for the per-book sources.
    same = src_of_doc[spans[:, 0]] == src_of_doc[spans[:, 3]]
    if meta["version"].startswith("scripture"):
        multi = np.array([s["source"] in MULTI_EDITION_SOURCES for s in shards])[shard_idx]
        same &= ~multi[spans[:, 0]] | (shard_idx[spans[:, 0]] == shard_idx[spans[:, 3]])
    cuts_by_doc: dict[int, list] = {}
    for doc, bs, be, dup, _ in spans[same].tolist():
        cuts_by_doc.setdefault(doc, []).append((bs, be, dup))
    unique_by_shard: dict[int, list[int]] = {}
    for doc in unique.tolist():
        unique_by_shard.setdefault(int(shard_idx[doc]), []).append(int(row_arr[doc]))
    doc_of = {}  # (shard, row) -> doc index for cut lookup
    for doc in cuts_by_doc:
        doc_of[(int(shard_idx[doc]), int(row_arr[doc]))] = doc
    jobs = []
    from training_scope import allowed, require_training_source
    for source in sources or []:
        require_training_source(source)
    for i, sh in enumerate(shards):
        if not allowed(sh["source"]):
            continue
        if sources and sh["source"] not in sources:
            continue
        cuts = {}
        for row in unique_by_shard.get(i, []):
            doc = doc_of.get((i, row))
            if doc is not None:
                cuts[row] = cuts_by_doc[doc]
        jobs.append({"source": sh["source"], "shard": sh["shard"], "canon_dir": str(paths.canon(sh["source"])), "out_dir": str(paths.out / "atoms" / paths.version / sh["source"]),
                     "unique_rows": unique_by_shard.get(i, []), "cuts": cuts, "canon_blake3": sh["canon_blake3"]})
    if not jobs or (sources and set(sources) != {j["source"] for j in jobs}):
        raise AtomError("MNLM_ATOMS_SOURCE_MISSING", "requested source absent from accepted population; rebuild the intended scope")
    stats = {"population": int(len(shard_idx)), "unique": int(len(unique)), "spans": int(len(spans)), "spans_applicable": int(same.sum()), "shards": len(jobs), "stage1": stage1}
    return jobs, stats


def prepare_tree(adir: Path, retire: bool) -> str | None:
    """An existing tree is refused, or moved to <out>/tmp/retired/ when --retire is given; returns the retired path."""
    if not adir.exists() or not any(adir.iterdir()):
        return None
    if not retire:
        raise AtomError("MNLM_ATOMS_TREE_EXISTS", f"{adir} already holds build artifacts; pass --retire to move it aside")
    dest = adir.parents[1] / "tmp" / "retired" / f"atoms-{adir.name}-{run_stamp()}"
    boundary = adir.parents[1].resolve()
    if not adir.resolve().is_relative_to(boundary) or not dest.resolve().is_relative_to(boundary):
        raise AtomError("MNLM_ATOMS_RETIRE_FAILED", "resolved retirement paths leave the output workspace")
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        adir.rename(dest)
    except OSError as e:
        raise AtomError("MNLM_ATOMS_RETIRE_FAILED", f"{adir} -> {dest}: {e}") from e
    return str(dest)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--version", required=True)
    ap.add_argument("--sources", default="")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--retire", action="store_true")
    ap.add_argument("--ucr-output", help="export accepted atoms through universal channel adapters after a successful build")
    args = ap.parse_args()
    paths = Paths(args.root, args.out, args.version)
    sources = [s for s in args.sources.split(",") if s] or None
    from training_scope import require_training_source
    for source in sources or []:
        require_training_source(source)
    adir = paths.out / "atoms" / args.version
    jobs, stats = plan_jobs(paths, sources)
    retired = prepare_tree(adir, args.retire)
    from atoms_publication import begin, finish
    begin(paths, jobs, stats["stage1"])
    print(f"atomise {args.version}: {stats} retired={retired}", flush=True)
    import multiprocessing as mp

    adir.mkdir(parents=True, exist_ok=True)
    manifest = adir / "manifest.jsonl"
    totals = {"docs": 0, "atoms": 0, "chars": 0, "cut_atoms": 0, "shards": 0, "errors": 0}
    code = {"fences": 0, "unclosed": 0, "unparsed": 0}
    roles: dict[int, int] = {}
    units: dict[int, int] = {}
    t0 = time.time()
    with mp.Pool(min(args.workers, max(1, len(jobs)))) as pool, manifest.open("x", encoding="utf-8") as mf:
        for res in pool.imap_unordered(_safe, jobs):
            mf.write(json.dumps(res, sort_keys=True) + "\n")
            mf.flush()
            os.fsync(mf.fileno())
            if "error" in res:
                totals["errors"] += 1
                print(f"  REFUSED {res['source']}/{res['shard']}: {res['error']}", flush=True)
                continue
            for k in ("docs", "atoms", "chars", "cut_atoms"):
                totals[k] += res[k]
            for k in code:
                code[k] += res["code"][k]
            for k, v in res["roles"].items():
                roles[int(k)] = roles.get(int(k), 0) + v
            for k, v in res["units"].items():
                units[int(k)] = units.get(int(k), 0) + v
            totals["shards"] += 1
    summary = {**stats, **totals, "roles": dict(sorted(roles.items())), "units": dict(sorted(units.items())), "code": code, "retired": retired,
               "segmenter": SEGMENTER_ID, "code_atomizer": CODE_ATOMIZER_ID, "python": PY_VERSION, "contract": CONTRACT_ID, "version": args.version, "host": HOST, "ts": now_iso(), "secs": round(time.time() - t0, 1)}
    write_json(adir / "atoms.summary.json", summary)
    e = ledger_append(paths, f"atomise@{HOST}", "atoms.build", f"stage2/{args.version}", summary)
    summary["ledger_seq"], summary["ledger_head"] = e["seq"], e["hash"]
    write_json(adir / "atoms.summary.json", summary)
    if not totals["errors"]:
        finish(paths)
    print(json.dumps(summary), flush=True)
    if args.ucr_output and not totals["errors"]:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from senses.stage2 import export_stage2
        export_stage2(paths, args.ucr_output, sources)
    return 1 if totals["errors"] else 0


def _safe(job: dict) -> dict:
    try:
        return atomise_shard(job)
    except Exception as e:  # noqa: BLE001 - recorded as a typed row, never swallowed
        return {"source": job["source"], "shard": job["shard"], "error": f"{type(e).__name__}: {e}"[:300], "host": HOST, "ts": now_iso()}


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AtomError as exc:
        print(json.dumps(exc.envelope()), file=sys.stderr)
        sys.exit(2)
