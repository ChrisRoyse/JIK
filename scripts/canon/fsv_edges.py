"""Stage 2 FSV for the typed edges, the versification remap and the book classes (issue #11): an independent restatement
that reads the persisted edge arrays and alignment tables back against the canonical shards.
  cites      exhaustive: every record's WebCite and reporter mentions are re-found with this file's own regexes and must
             equal the stored edge count per record; every resolved target's header carries the cite; every frontier cite
             is owned by no record; the atom named by an edge contains the cite; a planted unknown cite is a frontier edge.
  crossrefs  exhaustive recount of rows and of resolved/frontier against the stored arrays; sample of 500 rows re-parsed.
  anchors    exhaustive: every ref of every identity record has exactly one edge whose key is that ref; a planted bad ref
             resolves to nothing.
  versify    scheme inference re-derived from the chapter table with this file's own scorer; mapped keys re-derived for a
             sample of 2,000 verse atoms from the raw mapping rows; an edition agreeing with no scheme is unresolved.
  classes    every verse key carries a testament and genre from config/book_classes.json or is listed as unclassified.
    python -I scripts/canon/fsv_edges.py --root R --out O --version V --issue 11 --kinds cites|crossrefs,versify,classes|anchors [--target-version scripture-v1]
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import HOST, Paths, iter_canon_docs, read_jsonl, run_stamp, write_json  # noqa: E402

CHECKS: list[dict] = []
WEB = re.compile(r"(?<!\w)((?:19|20)\d{2}-Ohio-\d{1,5})(?!\w)")  # a hyphen may adjoin a cite in extracted PDF text (measured: 3 of 14,537 opinions)
HDR = re.compile(r"\[Cite as [^\]]*?,\s*((?:19|20)\d{2}-Ohio-\d{1,5})\.?\]")
REP = re.compile(r"(?<!\w)(\d{1,3} (?:Ohio St\.(?:2d|3d)|Ohio App\.(?:2d|3d)|N\.E\.(?:2d|3d)) \d{1,4})(?!\w)")
OSIS = re.compile(r"^([1-3]?[A-Za-z]{2,6})\.(\d+)\.(\d+)")
MAPROW = re.compile(r"^([1-9A-Z]{2,3}) (\d+):(\d+)(?:-(\d+))? = ([1-9A-Z]{2,3}) (\d+):(\d+)(?:-(\d+))?$")


def check(name: str, ok: bool, **detail):
    CHECKS.append({"check": name, "ok": bool(ok), **detail})
    print(("  ok   " if ok else "  FAIL ") + name + ("" if ok else f"  {json.dumps(detail, default=str)[:240]}"), flush=True)


def shards_of(paths: Paths, source: str):
    meta = json.loads((paths.dedup / "population.json").read_text(encoding="utf-8"))
    return [(i, s) for i, s in enumerate(meta["shards"]) if s["source"] == source]


def fsv_cites(paths: Paths, adir: Path) -> None:
    src = "cuyahoga-eighth-district"
    e = np.load(adir / "edges" / "cites.npz")
    summary = json.loads((adir / "edges" / "cites.summary.json").read_text(encoding="utf-8"))
    frontier = {r["cite_id"]: r["cite"] for r in read_jsonl(adir / "edges" / "cites.frontier.jsonl")}
    owner_of: dict[tuple, str] = {}
    owners: dict[str, tuple] = {}
    texts_needed = collections.defaultdict(dict)
    for si, sh in shards_of(paths, src):  # pass 1: owners under this file's own header regex
        for rec in iter_canon_docs(paths.canon(src), sh["shard"]):
            m = HDR.search(rec["text"][:3000])
            if m:
                owner_of[(si, rec["row"])] = m.group(1)
                owners.setdefault(m.group(1), (si, rec["row"]))
    stored = collections.Counter(zip(e["shard"].tolist(), e["row"].tolist()))
    per_rec_bad = count_bad = atom_bad = target_bad = frontier_bad = 0
    recs = 0
    examples: list[dict] = []
    edge_idx = collections.defaultdict(list)
    for k, (si, r) in enumerate(zip(e["shard"].tolist(), e["row"].tolist())):
        edge_idx[(si, r)].append(k)
    for si, sh in shards_of(paths, src):  # pass 2: exhaustive recount and per-edge atom text
        a = np.load(adir / src / f"{sh['shard']}.atoms.npz")
        st, en = a["start"], a["end"]
        for rec in iter_canon_docs(paths.canon(src), sh["shard"]):
            key = (si, rec["row"])
            if key not in stored and not edge_idx.get(key) and not WEB.search(rec["text"]) and not REP.search(rec["text"]):
                continue
            recs += 1
            own = owner_of.get(key)
            expect = sum(1 for m in WEB.finditer(rec["text"]) if m.group(1) != own) + sum(1 for _ in REP.finditer(rec["text"]))
            if expect != stored.get(key, 0):
                count_bad += 1
                if len(examples) < 4:
                    examples.append({"src": src, "shard": sh["shard"], "row": rec["row"], "expected": expect, "stored": stored.get(key, 0),
                                     "found": [m.group(1) for m in WEB.finditer(rec["text"])][:6] + [m.group(1) for m in REP.finditer(rec["text"])][:6]})
            for k in edge_idx.get(key, []):
                atom, off = int(e["atom"][k]), int(e["offset"][k])
                kind = int(e["kind"][k])
                cite = owner_of.get((int(e["target_shard"][k]), int(e["target_row"][k]))) if kind == 1 else frontier.get(int(e["cite_id"][k]))
                at_offset = cite is not None and rec["text"][off:off + len(cite)] == cite and st[atom] <= off < en[atom]  # the cite sits at its offset inside the named atom
                if kind == 1 and not at_offset:
                    target_bad += 1
                elif kind in (2, 3):
                    if not at_offset:
                        atom_bad += 1
                    elif kind == 2 and cite in owners:
                        frontier_bad += 1
    check("cites.per_record_counts_recomputed", count_bad == 0, records_checked=recs, count_mismatch=count_bad, edges=int(len(e["row"])), examples=examples)
    check("cites.resolved_targets_own_the_cite_at_offset", target_bad == 0, bad=target_bad, resolved=int((e["kind"] == 1).sum()))
    check("cites.frontier_cites_in_atom_and_unowned", atom_bad == 0 and frontier_bad == 0, atom_bad=atom_bad, owned_frontier=frontier_bad, frontier=int((e["kind"] >= 2).sum()))
    check("cites.owners_equal_summary", len(owners) == summary["owners"], owners=len(owners), summary=summary["owners"])
    check("NEG.planted_unknown_cite_is_frontier", "1999-Ohio-99999" not in owners and WEB.search("see 1999-Ohio-99999 at 3") is not None)


def fsv_crossrefs(paths: Paths, adir: Path) -> None:
    from align_verses import norm_book

    e = np.load(adir / "edges" / "crossrefs.npz")
    keys = {(r["book"], r["chapter"], r["verse"]): r["key_id"] for r in read_jsonl(adir / "align" / "verse_keys.jsonl")}
    src = "openbible-crossrefs"
    rows = resolved = 0
    stored_from = e["from_key"]
    stored_to = e["to_key"]
    pos = {(int(s), int(r)): k for k, (s, r) in enumerate(zip(e["shard"].tolist(), e["row"].tolist()))}
    rng = np.random.default_rng(7)
    sample_bad = sampled = 0
    for si, sh in shards_of(paths, src):
        for rec in iter_canon_docs(paths.canon(src), sh["shard"]):
            rows += 1
            ids = []
            for name in ("from_ref", "to_ref"):
                m = OSIS.match(rec["f"].get(name) or "")
                ids.append(keys.get((norm_book(m.group(1)), int(m.group(2)), int(m.group(3))), -1) if m else -1)
            resolved += min(ids) >= 0
            if rng.random() < 0.0015 and sampled < 500:
                sampled += 1
                k = pos.get((si, rec["row"]))
                if k is None or int(stored_from[k]) != ids[0] or int(stored_to[k]) != ids[1] or int(e["votes"][k]) != int(rec["f"].get("votes") or 0):
                    sample_bad += 1
    check("crossrefs.rows_and_resolution_recomputed", rows == len(e["row"]) and resolved == int(((stored_from >= 0) & (stored_to >= 0)).sum()), rows=rows, stored=int(len(e["row"])), resolved=resolved)
    check("crossrefs.sampled_rows_match", sample_bad == 0 and sampled > 0, sampled=sampled, bad=sample_bad)


def fsv_anchors(paths: Paths, adir: Path, target: Paths) -> None:
    e = np.load(adir / "edges" / "anchors.npz")
    keys = {(r["book"], r["chapter"], r["verse"]): r["key_id"] for r in read_jsonl(target.out / "atoms" / target.version / "align" / "verse_keys.jsonl")}
    stored = collections.Counter(zip(e["shard"].tolist(), e["row"].tolist(), e["key_id"].tolist(), e["field"].tolist()))
    expect = collections.Counter()
    bad_refs = 0
    for si, sh in shards_of(paths, "identity-v1"):
        for rec in iter_canon_docs(paths.canon("identity-v1"), sh["shard"]):
            for field, kind in (("anchor_refs", 1), ("refs_expanded", 2), ("ref", 3)):
                vals = rec["f"].get(field)
                for ref in ([vals] if isinstance(vals, str) else (vals or [])):
                    p = ref.split(".")
                    kid = keys.get((p[0], int(p[1]), int(p[2]))) if len(p) == 3 and p[1].isdigit() and p[2].isdigit() else None
                    if kid is None:
                        bad_refs += 1
                    else:
                        expect[(si, rec["row"], kid, kind)] += 1
    check("anchors.every_ref_has_its_edge", expect == stored and bad_refs == 0, expected=sum(expect.values()), stored=sum(stored.values()), unresolved=bad_refs)
    check("NEG.planted_bad_ref_resolves_to_nothing", keys.get(("ZZZ", 99, 99)) is None)


def fsv_versify(paths: Paths, adir: Path) -> None:
    align = adir / "align"
    vers = json.loads((align / "versification.json").read_text(encoding="utf-8"))
    chapters = json.loads((align / "edition_chapters.json").read_text(encoding="utf-8"))
    shard_edition = {int(k): v for k, v in json.loads((align / "shard_editions.json").read_text(encoding="utf-8")).items()}
    maxv: dict = collections.defaultdict(lambda: collections.defaultdict(dict))
    mapping: dict = collections.defaultdict(dict)
    for si, sh in shards_of(paths, "ubsicap-versification"):
        for rec in iter_canon_docs(paths.canon("ubsicap-versification"), sh["shard"]):
            f, t = rec["f"], rec["text"].strip()
            if f.get("kind") == "maxverses":
                parts = t.split()
                for p in parts[1:]:
                    if ":" in p:
                        c, v = p.split(":")
                        maxv[f["scheme"]][parts[0]][int(c)] = int(v)
            elif f.get("kind") == "mapping":
                m = MAPROW.match(t)
                if m:
                    b1, c1, v1, v1b, b2, c2, v2, v2b = m.groups()
                    for lv, rv in zip(range(int(v1), int(v1b or v1) + 1), range(int(v2), int(v2b or v2) + 1)):
                        mapping[f["scheme"]].setdefault((b1, int(c1), lv), (b2, int(c2), rv))
    infer_bad = 0
    for ed, books in chapters.items():
        best = []
        for scheme, sb in maxv.items():
            agree = disagree = 0
            for book, chs in books.items():
                for c, mv in chs.items():
                    w = sb.get(book, {}).get(int(c))
                    if w is not None:
                        agree += mv == w
                        disagree += mv != w
            best.append((agree, -disagree, scheme))
        best.sort(reverse=True)
        top = best[0]
        compared = top[0] - top[1]
        ok = compared > 0 and top[0] / compared >= 0.95 and (len(best) < 2 or top[0] > best[1][0])
        if (top[2] if ok else "unresolved") != vers[ed]["scheme"]:
            infer_bad += 1
    check("versify.scheme_inference_recomputed", infer_bad == 0, editions=len(chapters), mismatch=infer_bad, by_scheme=dict(collections.Counter(v["scheme"] for v in vers.values())))
    keys = {r["key_id"]: (r["book"], r["chapter"], r["verse"]) for r in read_jsonl(align / "verse_keys.jsonl")}
    org = {r["org_key_id"]: (r["book"], r["chapter"], r["verse"]) for r in read_jsonl(align / "org_keys.jsonl")}
    vi = np.load(align / "verse_index.npz")
    oi = np.load(align / "org_index.npz")["org_key_id"]
    rng = np.random.default_rng(13)
    pick = rng.choice(len(oi), min(2000, len(oi)), replace=False)
    bad = 0
    for i in pick.tolist():
        ed = shard_edition[int(vi["shard"][i])]
        scheme = vers[ed]["scheme"]
        k = keys[int(vi["key_id"][i])]
        want = mapping.get(scheme, {}).get(k, k) if scheme not in ("unresolved", "org") else k
        if org[int(oi[i])] != want:
            bad += 1
    check("versify.sampled_remap_recomputed", bad == 0 and len(oi) == len(vi["key_id"]), sampled=int(len(pick)), bad=bad)
    planted = {"ed": {"GEN": {"1": 99, "2": 99, "3": 99}}}
    scores = sorted(((sum(1 for c, mv in planted["ed"]["GEN"].items() if sb.get("GEN", {}).get(int(c)) == mv), s) for s, sb in maxv.items()), reverse=True)
    check("NEG.planted_edition_agreeing_with_no_scheme_is_unresolved", scores[0][0] == 0)


def fsv_classes(adir: Path) -> None:
    classes = json.loads((Path(__file__).resolve().parents[2] / "config" / "book_classes.json").read_text(encoding="utf-8"))
    t_of = {b: t for t, bs in classes["testament"].items() for b in bs}
    g_of = {b: g for g, bs in classes["genre"].items() for b in bs}
    summary = json.loads((adir / "align" / "align.summary.json").read_text(encoding="utf-8"))
    bad = 0
    uncl = set()
    n = 0
    for r in read_jsonl(adir / "align" / "verse_keys.jsonl"):
        n += 1
        if r.get("testament") != t_of.get(r["book"], "unclassified") or r.get("genre") != g_of.get(r["book"], "unclassified"):
            bad += 1
        if r["book"] not in t_of:
            uncl.add(r["book"])
    check("classes.every_key_classified_or_reported", bad == 0 and sorted(uncl) == summary.get("unclassified_books"), keys=n, bad=bad, unclassified=sorted(uncl))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--version", required=True)
    ap.add_argument("--issue", required=True)
    ap.add_argument("--kinds", required=True)
    ap.add_argument("--target-version", default="")
    args = ap.parse_args()
    paths = Paths(args.root, args.out, args.version)
    adir = paths.out / "atoms" / args.version
    for kind in args.kinds.split(","):
        if kind == "cites":
            fsv_cites(paths, adir)
        elif kind == "crossrefs":
            fsv_crossrefs(paths, adir)
        elif kind == "anchors":
            fsv_anchors(paths, adir, Paths(args.root, args.out, args.target_version))
        elif kind == "versify":
            fsv_versify(paths, adir)
        elif kind == "classes":
            fsv_classes(adir)
        else:
            raise SystemExit(f"MNLM_FSV_KIND_UNKNOWN: {kind}")
    passed = sum(c["ok"] for c in CHECKS)
    run = f"stage2-edges-{args.version}-{run_stamp()}"
    rdir = paths.root / "evidence" / "fsv" / args.issue / run
    rdir.mkdir(parents=True, exist_ok=True)
    write_json(rdir / "stage2-edges-fsv.json", {"run": run, "version": args.version, "host": HOST, "kinds": args.kinds, "passed": passed, "total": len(CHECKS), "checks": CHECKS})
    print(f"{passed}/{len(CHECKS)} checks passed; report {rdir / 'stage2-edges-fsv.json'}", flush=True)
    return 0 if passed == len(CHECKS) else 1


if __name__ == "__main__":
    sys.exit(main())
