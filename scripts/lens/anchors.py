"""Stage 3 calibration corpora (issue #12, spec §19.3 step 1): for each anchor family a stratified set of at least 500
anchored atoms drawn deterministically (BLAKE3 of seed and identity) from the Stage 2 atom stores, with the anchor read
from its oracle and the explicit fields kept beside the text for the fingerprint baseline. Families: A1 OpenCodeInstruct
tests executed here (sandbox.py), A2 Wikidata P31 class, A3/A5 GDELT (anchors_rows.py), A4 scripture genre, A6 Cuyahoga
in-corpus citation. Output: <out>/lenses/<version>/cal/<family>.jsonl + .summary.json, cal.summary.json, one ledger entry
`lenses.calibration`. Cost: one streaming pass per shard touched, O(records read); the A1 oracle is one sandboxed child per
candidate record (about 1.3 × target), parallel. A family under 50 samples in a class refuses MNLM_LENS_CAL_UNDERPOWERED.
    python -I scripts/lens/anchors.py --root R --out O --version p0-v1 [--families a,b] [--target 2000] [--seed 42]
"""
from __future__ import annotations

import argparse
import collections
import io
import json
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import zstandard

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lenscommon import (FAMILIES, HOST, LensError, Paths, atoms_by_row, fetch_records, hash_order, ledger_append, lens_root,  # noqa: E402
                        now_iso, population_shards, write_family, write_json)

ROLE_SENTENCE, ROLE_CODE, ROLE_STRUCT = 1, 6, 7
PY_UNITS = {1, 2, 3, 4}
MIN_CLASS = 50
FENCE_PY = re.compile(r"^```\s*(python|py|python3)\b", re.I)
EXCLUDED_P31 = {"Q4167410", "Q13406463", "Q4663903", "Q11266439", "Q17633526", "Q14204246", "Q4167836", "Q15184295", "Q13442814"}  # Wikimedia-internal classes and scholarly articles (no abstract)


def _check_classes(family: str, rows: list[dict]) -> None:
    counts = collections.Counter(r["y"] for r in rows)
    if len(counts) < 2 or min(counts.values()) < MIN_CLASS:
        raise LensError("MNLM_LENS_CAL_UNDERPOWERED", f"{family}: class counts {dict(counts)} (need >= 2 classes with >= {MIN_CLASS} each)")


# ---------------------------------------------------------------- A1: OpenCodeInstruct tests run by this stage
def _a1_scan(job: dict) -> dict:
    """Per shard: rows with a Python unit atom, and for the selected rows the solution body, prompt and tests."""
    src, shard, canon_dir, adir, seed, frac = job["source"], job["shard"], Path(job["canon_dir"]), Path(job["adir"]), job["seed"], job["frac"]
    arrays, per_row = atoms_by_row(adir, src, shard)
    py_rows = [r for r, idx in per_row.items() if bool((np.isin(arrays["unit"][idx], list(PY_UNITS))).any())]
    chosen = {r for r in py_rows if hash_order(seed, "a1", shard, r) / 2**64 < frac}
    out, skipped = [], collections.Counter()
    for row, rec in fetch_records(canon_dir, shard, chosen).items():
        text, f, idx = rec["text"], rec["f"], per_row[row]
        structs = [(int(arrays["start"][i]), int(arrays["end"][i])) for i in idx if arrays["role"][i] == ROLE_STRUCT]
        opens = [k for k, (s, e) in enumerate(structs) if FENCE_PY.match(text[s:e])]
        if len(opens) != 1 or opens[0] + 1 >= len(structs):
            skipped["fence_count"] += 1
            continue
        o = opens[0]
        body = text[structs[o][1]:structs[o + 1][0]].strip("\n")
        try:
            tests = json.loads(f.get("unit_tests") or "[]")
            provider = json.loads(f.get("tests_execution_status") or "[]")
        except json.JSONDecodeError:
            skipped["tests_unparsable"] += 1
            continue
        if not tests or not all(isinstance(t, str) for t in tests):
            skipped["no_tests"] += 1
            continue
        out.append({"source": src, "shard": shard, "row": row, "span": [structs[o][1], structs[o + 1][0]], "text": body, "derived": False,
                    "fields": {"domain": f.get("domain"), "generation_algorithm": f.get("generation_algorithm"), "n_tests": len(tests),
                               "body_chars": len(body), "prompt_chars": structs[o][0]},
                    "tests": tests, "provider_statuses": provider, "cluster": f.get("id") or f"{shard}:{row}"})
    return {"shard": shard, "py_rows": len(py_rows), "rows": out, "skipped": dict(skipped)}


def build_a1(paths: Paths, seed: int, target: int, workers: int) -> tuple[list[dict], dict]:
    import sandbox

    probe = sandbox.probe()
    src = "opencodeinstruct"
    adir = paths.out / "atoms" / paths.version
    shards = [s for s in population_shards(paths, paths.version) if s["source"] == src]
    counts = {}
    for s in shards:  # pass 0: how many records carry a Python unit (loads the atom arrays only)
        arrays, per_row = atoms_by_row(adir, src, s["shard"])
        counts[s["shard"]] = sum(1 for idx in per_row.values() if bool(np.isin(arrays["unit"][idx], list(PY_UNITS)).any()))
    total = sum(counts.values())
    frac = min(1.0, 1.6 * target / max(total, 1))
    jobs = [{"source": src, "shard": s["shard"], "canon_dir": str(paths.canon(src)), "adir": str(adir), "seed": seed, "frac": frac} for s in shards]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(_a1_scan, jobs))
    cands = sorted((r for res in results for r in res["rows"]), key=lambda r: hash_order(seed, "a1", r["shard"], r["row"]))
    skipped = collections.Counter()
    for res in results:
        skipped.update(res["skipped"])
    t0 = time.time()
    runs = sandbox.run_many([(r["text"], r["tests"]) for r in cands], workers)
    secs = time.time() - t0
    rows, agree_tests, agree_records, compared = [], 0, 0, 0
    outcomes = collections.Counter()
    for r, run in zip(cands, runs):
        st = run["statuses"]
        y = int(all(s == "pass" for s in st))
        r["y"], r["y_name"] = y, "all_pass" if y else "not_all_pass"
        r["statuses"], r["pass_fraction"], r["code_error"], r["timeout"] = st, sum(s == "pass" for s in st) / len(st), run["code_error"], run["timeout"]
        outcomes.update(st)
        prov = r.pop("provider_statuses")
        if len(prov) == len(st):
            compared += 1
            agree_tests += sum(a == b for a, b in zip(prov, st))
            agree_records += int(all(p == "pass" for p in prov) == bool(y))
        r["provider_all_pass"] = int(all(p == "pass" for p in prov)) if prov else None
        r.pop("tests")
        rows.append(r)
        if len(rows) >= target:
            break
    _check_classes("a1-oci-tests", rows)
    summary = {"sandbox": probe, "records_with_python_unit": total, "sample_fraction": frac, "candidates": len(cands), "executed_secs": round(secs, 1),
               "skipped": dict(skipped), "test_outcomes": dict(outcomes),
               "validity_audit": {"records_compared": compared, "per_test_agreement": agree_tests / max(1, sum(len(r["statuses"]) for r in rows[:compared])) if compared else None,
                                  "record_all_pass_agreement": agree_records / compared if compared else None,
                                  "note": "provider statuses are compared, never used as the anchor"}}
    return rows, summary


# ---------------------------------------------------------------- A2: Wikidata P31 class of the entity described by the abstract
def _iter_lines(path: Path):
    with path.open("rb") as fh, zstandard.ZstdDecompressor().stream_reader(fh) as zr:
        yield from io.TextIOWrapper(zr, encoding="utf-8")


def build_a2(paths: Paths, seed: int, target: int, per_class_classes: int = 12, prop: str = "P31", skip_classes: int = 0, fam: str = "a2-wikidata-p31") -> tuple[list[dict], dict]:
    """One Wikidata statement family: the value of `prop` held by the structured source, the `per_class_classes` most
    frequent values after skipping the first `skip_classes` (so a second question on the same property reads rarer classes)."""
    src = "wikidata5m"
    canon = paths.canon(src)
    adir = paths.out / "atoms" / paths.version
    p31: dict[str, str] = {}
    for shard in ("wikidata5m_transductive_train", "wikidata5m_transductive_valid", "wikidata5m_transductive_test"):
        for line in _iter_lines(canon / f"{shard}.jsonl.zst"):
            if f'"p": "{prop}"' not in line:
                continue
            f = json.loads(line)["f"]
            if f["p"] == prop:
                p31.setdefault(f["s"], f["o"])
    counts = collections.Counter(p31.values())
    classes = [c for c, _ in counts.most_common(200) if c not in EXCLUDED_P31][skip_classes:skip_classes + per_class_classes]
    cidx = {c: i for i, c in enumerate(classes)}
    per_class = max(MIN_CLASS, target // len(classes))
    want = collections.defaultdict(list)
    for s, o in p31.items():
        if o in cidx:
            want[o].append((hash_order(seed, "a2", s), s))
    cand: dict[str, str] = {}
    for o, lst in want.items():
        lst.sort()
        for _, s in lst[: per_class * 3]:
            cand[s] = o
    arrays, per_row = atoms_by_row(adir, src, "wikidata5m_text")
    found: dict[str, dict] = {}
    for line in _iter_lines(canon / "wikidata5m_text.jsonl.zst"):
        rec = json.loads(line)
        qid = rec["f"].get("entity_id")
        if qid in cand and rec["row"] in per_row:
            idx = per_row[rec["row"]]
            sent = [i for i in idx if arrays["role"][i] == ROLE_SENTENCE]
            if not sent:
                continue
            s, e = int(arrays["start"][sent[0]], ), int(arrays["end"][sent[0]])
            if e - s < 25 and len(sent) > 1:
                e = int(arrays["end"][sent[1]])
            found[qid] = {"source": src, "shard": "wikidata5m_text", "row": rec["row"], "span": [s, e], "text": rec["text"][s:e].strip(), "derived": False,
                          "fields": {"abstract_chars": len(rec["text"]), "atom_chars": e - s}, "cluster": qid}
    rows = []
    taken = collections.Counter()
    for qid, o in sorted(cand.items(), key=lambda kv: hash_order(seed, "a2", kv[0])):
        if qid in found and taken[o] < per_class and found[qid]["text"]:
            r = found[qid]
            r["y"], r["y_name"] = cidx[o], o
            rows.append(r)
            taken[o] += 1
    _check_classes(fam, rows)
    summary = {"property": prop, "skip_classes": skip_classes, "entities_with_p31": len(p31), "top_classes": [(c, counts[c]) for c in classes], "excluded_classes": sorted(EXCLUDED_P31 & set(counts)),
               "per_class_target": per_class, "candidates": len(cand), "with_text_record": len(found)}
    return rows, summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--version", default="p0-v1")
    ap.add_argument("--scripture-version", default="scripture-v1")
    ap.add_argument("--families", default=",".join(FAMILIES))
    ap.add_argument("--target", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--skip-existing", action="store_true", help="reuse a family already written under cal/ (its summary is re-read)")
    args = ap.parse_args()
    paths = Paths(args.root, args.out, args.version)
    spaths = Paths(args.root, args.out, args.scripture_version)
    spaths.canon_version = args.scripture_version  # the p0 override (MNLM_CANON_VERSION=p0micro-v1) never applies to the scripture partition
    cal_dir = lens_root(paths) / "cal"
    import anchors_q
    import anchors_rows

    builders = {"a1-oci-tests": lambda: build_a1(paths, args.seed, args.target, args.workers), "a2-wikidata-p31": lambda: build_a2(paths, args.seed, args.target),
                "a3-gdelt-dyad-7d": lambda: anchors_rows.build_gdelt(paths, args.seed, args.target, args.workers, "a3"),
                "a5-gdelt-sources": lambda: anchors_rows.build_gdelt(paths, args.seed, args.target, args.workers, "a5"),
                "a4-scripture-genre": lambda: anchors_rows.build_a4(spaths, args.seed, args.target),
                "a6-cuyahoga-cited": lambda: anchors_rows.build_a6(paths, args.seed, args.target),
                "a4-scripture-genre-keyfree": lambda: anchors_q.a4_keyfree(anchors_rows.build_a4(spaths, args.seed, args.target))}
    builders.update(anchors_q.builders(paths, args.seed, args.target, args.workers))
    summaries = {}
    for fam in args.families.split(","):
        from training_scope import POLICY
        if fam in POLICY["excluded_questions"]:
            print(f"{fam}: MNLM_SOURCE_EXCLUDED by operator", flush=True)
            continue
        if args.skip_existing and (cal_dir / f"{fam}.summary.json").exists():
            summaries[fam] = json.loads((cal_dir / f"{fam}.summary.json").read_text(encoding="utf-8"))
            print(f"{fam}: reused n={summaries[fam]['n']}", flush=True)
            continue
        t0 = time.time()
        rows, summary = builders[fam]()
        summary["secs"] = round(time.time() - t0, 1)
        summary["seed"] = args.seed
        summaries[fam] = write_family(cal_dir, fam, rows, summary)
        print(f"{fam}: n={summaries[fam]['n']} classes={summaries[fam]['classes']} H={summaries[fam]['H_Y_bits']:.3f} bits {summary['secs']} s", flush=True)
    top = {"version": args.version, "families": {f: {k: s[k] for k in ("n", "classes", "H_Y_bits", "file_blake3", "family", "oracle")} for f, s in summaries.items()},
           "seed": args.seed, "target": args.target, "host": HOST, "ts": now_iso()}
    e = ledger_append(paths, f"anchors@{HOST}", "lenses.calibration", f"stage3/{args.version}", top)
    top["ledger_seq"], top["ledger_head"] = e["seq"], e["hash"]
    write_json(cal_dir / "cal.summary.json", top)
    print(json.dumps({"ledger_seq": e["seq"], "ledger_head": e["hash"]}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
