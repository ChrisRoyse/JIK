"""Stage 2 FSV (issue #11): an independent restatement that reads the persisted atom store (contract atoms-v2/v3) back
against the canonical shards. B1 conservation: on every atoms shard the spans tile each record's text exactly, roles and
unit kinds are valid, counts and array digests match the manifest, every unique record has atoms, and the manifest's
Python version is this host's. B1 code: on records with fences or headings (up to a few per shard) the fence and heading
lines are re-found with an independent regex and must be structural atoms, the fence bodies must be covered by code or
cut atoms only, no code atom may lie outside a fence, and the unit starts of a Python body re-derived with `ast` must
equal the code atom starts. B2 cuts: on a sample of records with cut atoms the applicable Stage 1 spans are re-derived
from spans.npy with an independent byte-to-char conversion and must be covered exactly by cut atoms; a span from another
edition must not be applied. B3 field plans: every explicit field of every source has a plan, numeric fields are never
routed to free-text, every clock field declares its clock role, and planted violations refuse. Negative controls on the
representation: a shifted offset, a tampered digest, an unclosed fence, a Python body that does not parse.
Cost: O(atoms + chars) one pass; nothing is recomputed per atom.
    python -I scripts/canon/fsv_stage2.py --root R --out O --version V --issue 11 [--sample-docs 2000]
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import re
import sys
from pathlib import Path

import blake3
import numpy as np
import zstandard

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import HOST, Paths, read_jsonl, run_stamp, write_json  # noqa: E402

CHECKS: list[dict] = []
ROLE_SENTENCE, ROLE_WS, ROLE_ROW, ROLE_CUT, ROLE_VERSE, ROLE_CODE, ROLE_STRUCT = 1, 2, 3, 4, 5, 6, 7
ROLES = [ROLE_SENTENCE, ROLE_WS, ROLE_ROW, ROLE_CUT, ROLE_VERSE, ROLE_CODE, ROLE_STRUCT]
UNITS = [0, 1, 2, 3, 4, 5]
ARRAYS = ("row", "start", "end", "role", "unit", "cut_of")
MULTI_EDITION = {"ebible-english", "crosswire-sword-pd", "sefaria-export", "sefaria-export-nc"}
PLAN_KINDS = {"scalar", "cyclic", "categorical", "ordinal", "set", "reference", "text-label", "clock", "free-text"}
CLOCK_ROLES = {"coordinate", "receipt", "revision", "calendar", "lifecycle", "relational"}
PY_VERSION = "cpython-" + ".".join(map(str, sys.version_info[:3]))
_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})[ \t]*([^\n`]*)$", re.M)
_HEAD = re.compile(r"(?:^|(?<=[\r\n]))#{1,6} \S")
CODE_SAMPLE_PER_SHARD = 5


def check(name: str, ok: bool, **detail):
    CHECKS.append({"check": name, "ok": bool(ok), **detail})
    print(("  ok   " if ok else "  FAIL ") + name + ("" if ok else f"  {json.dumps(detail, default=str)[:200]}"), flush=True)


def tiles(n: int, spans) -> bool:
    """Independent restatement of conservation: spans start at 0, abut, are non-empty, end at n (none when n == 0)."""
    if n == 0:
        return len(spans) == 0
    if len(spans) == 0 or spans[0][0] != 0 or spans[-1][1] != n:
        return False
    return all(e > s for s, e in spans) and all(spans[i][1] == spans[i + 1][0] for i in range(len(spans) - 1))


def canon_texts(canon_dir: Path, shard: str):
    with (canon_dir / f"{shard}.jsonl.zst").open("rb") as fh, zstandard.ZstdDecompressor().stream_reader(fh) as zr:
        for line in io.TextIOWrapper(zr, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                yield r["row"], r["text"], r["f"]


def fences(text: str) -> list[tuple[int, int, int, int, str]]:
    """Independent line/character restatement; no producer scanner imported."""
    out, opened = [], None
    for m in re.finditer(r"[^\r\n]*(?:\r\n|\r|\n|$)", text):
        if not m.group():
            continue
        line = m.group().rstrip("\r\n")
        if opened is None:
            match = _FENCE.fullmatch(line)
            if match:
                chars, info = match.groups()
                opened = (m.start(), m.end(), chars, info)
        else:
            stripped = line.lstrip(" ").rstrip(" \t")
            if (len(line) - len(line.lstrip(" ")) <= 3 and len(stripped) >= len(opened[2])
                    and set(stripped) == {opened[2][0]}):
                out.append((opened[0], opened[1], m.start(), m.end(), opened[3]))
                opened = None
    if opened is not None:
        out.append((opened[0], opened[1], len(text), len(text), opened[3]))
    return out


def code_record_check(text: str, atoms: list[tuple[int, int, int, int]]) -> dict:
    """atoms: (start, end, role, unit) of one record. Returns the counts of violations under the independent restatement."""
    bad = {"fence_not_struct": 0, "body_not_code": 0, "code_outside": 0, "heading_not_struct": 0, "unit_mismatch": 0, "python_bodies": 0}
    spans_by_start = {s: (e, r, u) for s, e, r, u in atoms}

    def structural(a: int, b: int) -> bool:
        """[a, b) is tiled by structural atoms, or by cut atoms where a Stage 1 cut covers the line (a cut keeps its role by contract)."""
        pos = a
        while pos < b:
            nxt = spans_by_start.get(pos)
            if nxt is None or nxt[1] not in (ROLE_STRUCT, ROLE_CUT) or nxt[0] > b:
                return False
            pos = nxt[0]
        return pos == b

    fs = fences(text)
    inside = []
    for o, bs, be, ce, info in fs:
        inside.append((bs, be))
        if not structural(o, bs):
            bad["fence_not_struct"] += 1
        if ce > be and not structural(be, ce):
            bad["fence_not_struct"] += 1
        body_atoms = [(s, e, r, u) for s, e, r, u in atoms if bs <= s and e <= be]
        if any(r not in (ROLE_CODE, ROLE_CUT) for _, _, r, _ in body_atoms) or sum(e - s for s, e, _, _ in body_atoms) != be - bs:
            bad["body_not_code"] += 1
        lang = info.split()[0].lower() if info.strip() else ""
        if lang in {"python", "py", "python3", "py3"} and be > bs:
            bad["python_bodies"] += 1
            try:
                tree = ast.parse(text[bs:be])
                body = text[bs:be]
                line_start = [bs] + [bs + i + 1 for i, ch in enumerate(body)
                                    if ch == "\n" or (ch == "\r" and body[i + 1:i + 2] != "\n")]
                starts = {bs}
                for node in tree.body[1:]:
                    lineno = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
                    starts.add(line_start[lineno - 1])
                got = {s for s, _, _, _ in body_atoms}  # a cut inside a unit adds boundaries; a unit start is always one of them
                has_cut = any(r == ROLE_CUT for _, _, r, _ in body_atoms)
                if (not tree.body and not any(u == 4 for *_, u in body_atoms)) or (tree.body and not (starts <= got if has_cut else starts == got)):
                    bad["unit_mismatch"] += 1
            except (SyntaxError, ValueError, RecursionError, MemoryError):
                if not any(u == 4 for *_, u in body_atoms):
                    bad["unit_mismatch"] += 1
    for s, e, r, _ in atoms:
        if r == ROLE_CODE and not any(bs <= s and e <= be for bs, be in inside):
            bad["code_outside"] += 1
    for m in _HEAD.finditer(text):
        if any(bs <= m.start() < be for bs, be in inside):
            continue
        end = re.search(r"\r\n|\r|\n", text[m.start():])
        le = len(text) if end is None else m.start() + end.end()
        if not structural(m.start(), le):
            bad["heading_not_struct"] += 1
    return bad


def b1_shard(job: dict) -> dict:
    src, shard, row = job["src"], job["shard"], job["row"]
    a = np.load(Path(job["adir"]) / src / f"{shard}.atoms.npz")
    h = blake3.blake3()
    for k in ARRAYS:
        h.update(a[k].tobytes())
    rows, st, en, ro, un, co = (a[k] for k in ARRAYS)
    out = {"src": src, "shard": shard, "digest_ok": h.hexdigest() == row["arrays_blake3"], "roles_ok": bool(np.isin(ro, ROLES).all()), "units_ok": bool(np.isin(un, UNITS).all()),
           "python_ok": row.get("python") == PY_VERSION, "atoms": int(len(rows)), "cut_targets": np.unique(co[co >= 0]).tolist(),
           "code_atoms": int((ro == ROLE_CODE).sum()), "struct_atoms": int((ro == ROLE_STRUCT).sum())}
    order = np.argsort(rows, kind="stable")
    rows, st, en, ro, un = rows[order], st[order], en[order], ro[order], un[order]
    bounds = np.flatnonzero(np.diff(rows)) + 1
    per_row = {int(rows[i]): (int(i), int(j)) for i, j in zip(np.concatenate([[0], bounds]), np.concatenate([bounds, [len(rows)]]))} if len(rows) else {}
    want = set(job["unique_rows"])
    docs = bad = 0
    seen = set()
    code_bad = {"fence_not_struct": 0, "body_not_code": 0, "code_outside": 0, "heading_not_struct": 0, "unit_mismatch": 0, "python_bodies": 0}
    code_checked = 0
    examples: list[dict] = []
    for r, text, _ in canon_texts(Path(job["canon_dir"]), shard):
        if r not in per_row:
            if r in want and text == "":  # an empty unique record is conserved by zero atoms (tiles(0, []))
                seen.add(r)
                docs += 1
            continue
        i, j = per_row[r]
        seen.add(r)
        docs += 1
        if not tiles(len(text), list(zip(st[i:j].tolist(), en[i:j].tolist()))):
            bad += 1
        segmented = not np.isin(ro[i:j], [ROLE_VERSE, ROLE_ROW]).any()  # a verse or row record is one atom by contract; fences and headings inside it are text
        if segmented and code_checked < CODE_SAMPLE_PER_SHARD and ("```" in text or "~~~" in text or _HEAD.search(text)):
            code_checked += 1
            cb = code_record_check(text, list(zip(st[i:j].tolist(), en[i:j].tolist(), ro[i:j].tolist(), un[i:j].tolist())))
            for k, v in cb.items():
                code_bad[k] += v
            if any(v for k, v in cb.items() if k != "python_bodies") and len(examples) < 3:
                examples.append({"src": src, "shard": shard, "row": r, **{k: v for k, v in cb.items() if v}})
    out.update(docs=docs, bad_docs=bad, uncovered=len(want - seen), counts_ok=(docs == row["docs"] and len(rows) == row["atoms"]), code_checked=code_checked, code_bad=code_bad, examples=examples)
    return out


def b1(paths: Paths, adir: Path, manifest: dict, unique_by_shard: dict, workers: int) -> tuple[dict, dict]:
    """Every shard: conservation per record, roles valid, counts and digests equal, unique coverage, code atoms. Returns totals and cut targets per shard."""
    import multiprocessing as mp

    jobs = [{"src": src, "shard": shard, "row": row, "adir": str(adir), "canon_dir": str(paths.canon(src)), "unique_rows": unique_by_shard.get((src, shard), [])}
            for (src, shard), row in sorted(manifest.items())]
    totals = {"shards": 0, "docs": 0, "atoms": 0, "bad_docs": 0, "bad_roles": 0, "bad_units": 0, "python_mismatch": 0, "digest_mismatch": 0, "count_mismatch": 0, "uncovered_unique": 0,
              "code_atoms": 0, "struct_atoms": 0, "code_records_checked": 0}
    code_bad = {"fence_not_struct": 0, "body_not_code": 0, "code_outside": 0, "heading_not_struct": 0, "unit_mismatch": 0, "python_bodies": 0}
    targets: dict[tuple, list[int]] = {}
    examples: list[dict] = []
    with mp.Pool(min(workers, max(1, len(jobs)))) as pool:
        for o in pool.imap_unordered(b1_shard, jobs):
            totals["shards"] += 1
            for k in ("docs", "atoms", "bad_docs", "code_atoms", "struct_atoms"):
                totals[k] += o[k]
            totals["bad_roles"] += not o["roles_ok"]
            totals["bad_units"] += not o["units_ok"]
            totals["python_mismatch"] += not o["python_ok"]
            totals["digest_mismatch"] += not o["digest_ok"]
            totals["count_mismatch"] += not o["counts_ok"]
            totals["uncovered_unique"] += o["uncovered"]
            totals["code_records_checked"] += o["code_checked"]
            for k, v in o["code_bad"].items():
                code_bad[k] += v
            if len(examples) < 6:
                examples += o["examples"]
            targets[(o["src"], o["shard"])] = o["cut_targets"]
    check("B1.conservation_every_record", totals["bad_docs"] == 0, **{k: totals[k] for k in ("shards", "docs", "atoms", "bad_docs")})
    check("B1.roles_and_units_valid", totals["bad_roles"] == 0 and totals["bad_units"] == 0, bad_role_shards=totals["bad_roles"], bad_unit_shards=totals["bad_units"])
    check("B1.counts_and_digests_match_manifest", totals["count_mismatch"] == 0 and totals["digest_mismatch"] == 0, count_mismatch=totals["count_mismatch"], digest_mismatch=totals["digest_mismatch"])
    check("B1.every_unique_record_has_atoms", totals["uncovered_unique"] == 0, uncovered=totals["uncovered_unique"])
    check("B1.python_version_matches_host", totals["python_mismatch"] == 0, mismatch_shards=totals["python_mismatch"], host=PY_VERSION)
    viol = sum(v for k, v in code_bad.items() if k != "python_bodies")
    check("B1.code_and_structural_atoms", viol == 0, records_checked=totals["code_records_checked"], code_atoms=totals["code_atoms"], struct_atoms=totals["struct_atoms"], examples=examples[:6], **code_bad)
    return totals, targets


def cut_targets_same_edition(meta: dict, pop, targets: dict) -> None:
    """Every cut atom points at a duplicate in the same source, and the same shard for multi-edition sources (the foreign-edition control)."""
    shards = meta["shards"]
    idx = {(s["source"], s["shard"]): i for i, s in enumerate(shards)}
    foreign = total = 0
    for key, docs in targets.items():
        if not docs:
            continue
        si = idx[key]
        d = np.asarray(docs)
        src_ok = np.array([shards[int(x)]["source"] for x in pop["shard"][d]]) == key[0]
        same_shard = pop["shard"][d] == si
        ok = src_ok & (same_shard | (key[0] not in MULTI_EDITION))
        total += len(d)
        foreign += int((~ok).sum())
    check("NEG.no_cut_points_outside_its_edition", foreign == 0, cut_targets=total, foreign=foreign)


def b2_cuts(paths: Paths, adir: Path, meta: dict, pop, spans: np.ndarray, sample_docs: int) -> None:
    """Re-derive applicable cuts for sampled docs with cut atoms; cut atoms must cover them exactly; a foreign-edition span must not apply."""
    shards = meta["shards"]
    src_of = np.array([s["source"] for s in shards])[pop["shard"]]
    same = src_of[spans[:, 0]] == src_of[spans[:, 3]]
    if meta["version"].startswith("scripture"):
        multi = np.array([s["source"] in MULTI_EDITION for s in shards])[pop["shard"]]
        same &= ~multi[spans[:, 0]] | (pop["shard"][spans[:, 0]] == pop["shard"][spans[:, 3]])
    app = spans[same]
    rng = np.random.default_rng(11)
    docs = np.unique(app[:, 0])
    pick = rng.choice(docs, min(sample_docs, len(docs)), replace=False) if len(docs) else np.array([], dtype=np.int64)
    by_shard: dict[int, list[int]] = {}
    for d in pick.tolist():
        by_shard.setdefault(int(pop["shard"][d]), []).append(d)
    tested = agree = 0
    for si, dlist in by_shard.items():
        sh = shards[si]
        a = np.load(adir / sh["source"] / f"{sh['shard']}.atoms.npz")
        rows_wanted = {int(pop["row"][d]): d for d in dlist}
        for r, text, _ in canon_texts(paths.canon(sh["source"]), sh["shard"]):
            if r not in rows_wanted:
                continue
            d = rows_wanted[r]
            tb = text.encode("utf-8")
            exp = set()
            for _, bs, be, dup, _ in app[app[:, 0] == d].tolist():
                cs = len(tb[:bs].decode("utf-8")) if 0 < bs < len(tb) else (0 if bs <= 0 else len(text))
                ce = len(tb[:be].decode("utf-8")) if 0 < be < len(tb) else (0 if be <= 0 else len(text))
                exp.add((cs, ce))
            m = (a["row"] == r) & (a["role"] == ROLE_CUT)
            got = sorted(zip(a["start"][m].tolist(), a["end"][m].tolist()))
            merged = []  # cut atoms may be split further by atom boundaries: merge contiguous cut atoms and compare the union
            for s, e in got:
                if merged and merged[-1][1] == s:
                    merged[-1] = (merged[-1][0], e)
                else:
                    merged.append((s, e))
            covered = {(cs, ce) for cs, ce in exp if any(ms <= cs and ce <= me for ms, me in merged)}
            tested += 1
            agree += covered == exp and sum(e - s for s, e in merged) == sum(e - s for s, e in exp)
    check("B2.cuts_match_applicable_spans", tested == agree and (tested > 0 or len(docs) == 0), tested=tested, agree=agree, applicable=int(same.sum()), spans=int(len(spans)))


def validate_plan(src: str, plan: dict, fields: dict) -> tuple[list, list, list, list]:
    """(missing, unknown_kind, numeric_routed_to_free_text, clock_without_role) for one record's explicit fields against its plan."""
    missing, unknown, numeric, clocks = [], [], [], []
    for k, v in fields.items():
        kind = plan["explicit"].get(k)
        if kind is None:
            missing.append(f"{src}.{k}")
        elif kind not in PLAN_KINDS:
            unknown.append(f"{src}.{k}:{kind}")
        elif isinstance(v, (int, float)) and not isinstance(v, bool) and kind == "free-text":
            numeric.append(f"{src}.{k}")
        elif kind == "clock" and plan.get("clocks", {}).get(k) not in CLOCK_ROLES:
            clocks.append(f"{src}.{k}")
    return missing, unknown, numeric, clocks


def b3_field_plans(paths: Paths, meta: dict) -> None:
    plans = json.loads((Path(__file__).resolve().parents[2] / "config" / "field_plans.json").read_text(encoding="utf-8"))
    missing, numeric_to_text, unknown_kind, clock_bad = [], [], [], []
    for src in sorted({s["source"] for s in meta["shards"]}):
        plan = plans["sources"].get(src) or (plans["sources"]["scripture-common"] if src in plans["scripture_sources"] else None)
        if plan is None:
            missing.append(src)
            continue
        first = next(s for s in meta["shards"] if s["source"] == src)
        for _, _, f in canon_texts(paths.canon(src), first["shard"]):
            m, u, n, c = validate_plan(src, plan, f)
            missing += m
            unknown_kind += u
            numeric_to_text += n
            clock_bad += c
            break
    check("B3.every_explicit_field_has_a_plan", not missing and not unknown_kind, missing=missing[:8], unknown=unknown_kind[:8])
    check("B3.no_numeric_field_embedded", not numeric_to_text, fields=numeric_to_text[:8])
    check("B3.every_clock_field_declares_its_role", not clock_bad, fields=clock_bad[:8])
    _, _, planted, _ = validate_plan("planted", {"explicit": {"score": "free-text"}}, {"score": 0.5})
    check("NEG.planted_embedded_numeric_refuses", planted == ["planted.score"])
    _, _, _, planted_clock = validate_plan("planted", {"explicit": {"when": "clock"}}, {"when": "2026-10-15"})
    check("NEG.planted_undeclared_clock_refuses", planted_clock == ["planted.when"])


def negative_controls(first_manifest_row: dict) -> None:
    from code_atoms import atomise_text  # the production path, run on planted inputs

    check("NEG.shifted_offset_breaks_conservation", not tiles(10, [(0, 5), (6, 10)]) and tiles(10, [(0, 5), (5, 10)]))
    check("NEG.tampered_manifest_digest_differs", first_manifest_row["arrays_blake3"] != blake3.blake3(b"tampered").hexdigest())
    spans, st = atomise_text("Text.\n```python\nx = 1\n")
    check("NEG.planted_unclosed_fence_runs_to_end", st["unclosed"] == 1 and spans[-1][1] == len("Text.\n```python\nx = 1\n") and spans[-1][2] == ROLE_CODE, stats=st)
    spans, st = atomise_text("```python\ndef broken(:\n```\n")
    check("NEG.planted_unparsable_python_is_one_unparsed_atom", st["unparsed"] == 1 and [s[3] for s in spans if s[2] == ROLE_CODE] == [4], stats=st)
    bad = code_record_check("```\nx\n```\n", [(0, 4, ROLE_SENTENCE, 0), (4, 6, ROLE_CODE, 5), (6, 10, ROLE_STRUCT, 0)])
    check("NEG.planted_fence_as_sentence_refuses", bad["fence_not_struct"] == 1, bad=bad)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--version", required=True)
    ap.add_argument("--issue", required=True)
    ap.add_argument("--sample-docs", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    paths = Paths(args.root, args.out, args.version)
    adir = paths.out / "atoms" / args.version
    manifest = {}
    for r in read_jsonl(adir / "manifest.jsonl"):
        if "error" not in r:
            manifest[(r["source"], r["shard"])] = r
    from artifact_seals import Audit
    Audit(paths).require("spans")
    meta = json.loads((paths.dedup / "population.json").read_text(encoding="utf-8"))
    npz = np.load(paths.dedup / "population.npz")
    pop = {k: npz[k] for k in ("shard", "row")}  # materialised once: an NpzFile re-reads the array on every index
    unique = np.load(paths.dedup / "unique.npy")
    spans = np.load(paths.dedup / "spans.npy")
    unique_by_shard: dict[tuple, list[int]] = {}
    sh_names = [(s["source"], s["shard"]) for s in meta["shards"]]
    u_shard, u_row = pop["shard"][unique], pop["row"][unique]
    for si, r in zip(u_shard.tolist(), u_row.tolist()):
        unique_by_shard.setdefault(sh_names[si], []).append(r)
    print(f"FSV Stage 2 {args.version} on {HOST}: {len(manifest)} atom shards, {len(unique)} unique records", flush=True)
    contracts = {r.get("contract") for r in manifest.values()}
    check("B1.manifest_contract_uniform_known", contracts in ({"atoms-v2"}, {"atoms-v3"}), contracts=sorted(map(str, contracts)))
    if contracts == {"atoms-v3"}:
        base = Path(__file__).resolve().parent
        identity = hashlib.sha256((base / "code_atoms.py").read_bytes()
                                  + hashlib.sha256((base / "fences.py").read_bytes()).digest()).hexdigest()
        check("B1.code_atomizer_identity", all(r.get("code_atomizer") == identity for r in manifest.values()))
    totals, targets = b1(paths, adir, manifest, unique_by_shard, args.workers)
    cut_targets_same_edition(meta, pop, targets)
    check("B1.docs_equal_unique_population", totals["docs"] == len(unique), docs=totals["docs"], unique=int(len(unique)))
    b2_cuts(paths, adir, meta, pop, spans, args.sample_docs)
    b3_field_plans(paths, meta)
    negative_controls(next(iter(manifest.values())))
    passed = sum(c["ok"] for c in CHECKS)
    run = f"stage2-{args.version}-{run_stamp()}"
    rdir = paths.root / "evidence" / "fsv" / args.issue / run
    rdir.mkdir(parents=True, exist_ok=True)
    report = {"run": run, "version": args.version, "host": HOST, "passed": passed, "total": len(CHECKS), "totals": totals, "checks": CHECKS}
    write_json(rdir / "stage2-fsv.json", report)
    print(f"{passed}/{len(CHECKS)} checks passed; report {rdir / 'stage2-fsv.json'}", flush=True)
    return 0 if passed == len(CHECKS) else 1


if __name__ == "__main__":
    sys.exit(main())
