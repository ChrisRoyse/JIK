"""Second questions of the Stage 3 anchor families (issue #12, spec §19.1: "lenses are admitted per anchor family and the
panel is the union"). The first run measured that a question one lens nearly answers leaves no room for a second lens to
add 0.05 bits (A2: 3.35 of 3.585 bits held out by one lexical lens), so each family gets questions with residual
uncertainty, read from the same oracles: A1 the same sandboxed run as all-pass / some-fail / any-error; A2 the P17 and
P106 statements and the rarer P31 classes; A3 the QuadClass of the pair's next coding within the horizon (or none); A6 the
number of in-corpus citations received as tiers. Rows are drawn by the same seeded hash order as the first questions;
class floors and the derived-text rule are unchanged. Called from anchors.py. Cost: A1 reads the frozen A1 rows; A2 one
pass over the triples per property; A3 one parallel pass over the 90 day shards; A6 the citation arrays in memory."""
from __future__ import annotations

import collections
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from anchors_rows import HORIZON_DAYS, MIN_CLASS, _gdelt_scan  # noqa: E402
from lenscommon import LensError, Paths, atoms_by_row, fetch_records, hash_order, lens_root, population_shards, read_family  # noqa: E402


def build_a1_outcome3(paths: Paths) -> tuple[list[dict], dict]:
    """The frozen A1 rows with the test run read as three outcomes; the row identity, text and fields are unchanged."""
    rows, summary = read_family(lens_root(paths) / "cal", "a1-oci-tests")
    out = []
    for r in rows:
        r = {k: v for k, v in r.items() if k not in ("i", "family", "text_b3")}
        st = r.get("statuses") or []
        err = bool(r.get("code_error")) or bool(r.get("timeout")) or any(s == "error" for s in st)
        y = 2 if err else (1 if any(s == "fail" for s in st) else 0)
        r["y"], r["y_name"] = y, ("all_pass", "some_fail", "any_error")[y]
        out.append(r)
    counts = collections.Counter(r["y"] for r in out)
    if len(counts) < 3 or min(counts.values()) < MIN_CLASS:
        raise LensError("MNLM_LENS_CAL_UNDERPOWERED", f"a1-oci-outcome3: {dict(counts)}")
    return out, {"from_family": "a1-oci-tests", "from_file_blake3": summary["file_blake3"], "outcome": "2 if any test errored, the code failed to load or the run timed out; 1 if any test failed; else 0"}


def build_a1_ran(paths: Paths) -> tuple[list[dict], dict]:
    """The A1 rows whose solution the oracle recorded as loaded (no code error, no timeout), read as all-pass or not: load
    failure is no longer part of the label, so an execution-trace lens on this question senses behaviour, not the oracle."""
    rows, summary = read_family(lens_root(paths) / "cal", "a1-oci-tests")
    out = []
    for r in rows:
        if r.get("code_error") or r.get("timeout"):
            continue
        r = {k: v for k, v in r.items() if k not in ("i", "family", "text_b3")}
        r["y"] = 1 if all(s == "pass" for s in (r.get("statuses") or [])) else 0
        r["y_name"] = ("not_all_pass", "all_pass")[r["y"]]
        out.append(r)
    counts = collections.Counter(r["y"] for r in out)
    if len(counts) < 2 or min(counts.values()) < MIN_CLASS:
        raise LensError("MNLM_LENS_CAL_UNDERPOWERED", f"a1-oci-ran-tests: {dict(counts)}")
    return out, {"from_family": "a1-oci-tests", "from_file_blake3": summary["file_blake3"], "population": "rows with code_error null and timeout false", "dropped": len(rows) - len(out)}


def build_a3_next_quad(paths: Paths, seed: int, target: int, workers: int) -> tuple[list[dict], dict]:
    """A3 as a five-class temporal realisation: the QuadClass of the earliest later coding of the same (Actor1, Actor2) within
    HORIZON_DAYS (the lowest QuadClass if several were coded on that day), or 0 when the pair is not coded again."""
    src = "gdelt-events-90d"
    adir = paths.out / "atoms" / paths.version
    shards = sorted(s["shard"] for s in population_shards(paths, paths.version) if s["source"] == src)
    frac = 2.5 * target / (max(1, len(shards) - HORIZON_DAYS) * 110_000)
    jobs = [{"source": src, "shard": sh, "canon_dir": str(paths.canon(src)), "adir": str(adir), "seed": seed, "frac": frac, "day": d} for d, sh in enumerate(shards)]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(_gdelt_scan, jobs))
    days_of: dict[str, int] = collections.defaultdict(int)
    for res in results:
        for k in res["keys"]:
            days_of[k] |= 1 << res["day"]
    last_day = len(shards) - 1
    cands = [c for res in results for c in res["cands"] if c["fields"]["day_index"] + HORIZON_DAYS <= last_day]
    cands.sort(key=lambda c: hash_order(seed, "gdelt", c["shard"], c["row"]))
    per_class = max(MIN_CLASS, target // 5)
    taken: collections.Counter = collections.Counter()
    rows = []
    for c in cands:
        d = c["fields"]["day_index"]
        pair = c["cluster"]
        first, quad = None, 0
        for q in ("1", "2", "3", "4"):
            mask = days_of.get(f"{pair}|{q}", 0) & (((1 << (d + HORIZON_DAYS + 1)) - 1) & ~((1 << (d + 1)) - 1))
            if mask:
                nxt = (mask & -mask).bit_length() - 1
                if first is None or nxt < first:
                    first, quad = nxt, int(q)
        if taken[quad] >= per_class:
            continue
        c = dict(c)
        c.pop("key", None)
        c.pop("key_event", None)
        c["y"], c["y_name"] = quad, ("no_recurrence_7d", "next_verbal_cooperation", "next_material_cooperation", "next_verbal_conflict", "next_material_conflict")[quad]
        rows.append(c)
        taken[quad] += 1
        if sum(taken.values()) >= target:
            break
    counts = collections.Counter(r["y"] for r in rows)
    if len(counts) < 5 or min(counts.values()) < MIN_CLASS:
        raise LensError("MNLM_LENS_CAL_UNDERPOWERED", f"a3-gdelt-next-quad: {dict(counts)}")
    return rows, {"days": len(shards), "horizon_days": HORIZON_DAYS, "per_class_target": per_class, "candidates": len(cands), "sample_fraction": frac,
                  "text": "derived: Actor1Name Actor2Name: SOURCEURL slug words (tagged derived; the row has no prose)",
                  "tie_rule": "several QuadClasses coded on the earliest later day: the lowest QuadClass", "window_note": "the realisation window had already closed when the anchor was read"}


def build_a6_cite_tier(paths: Paths, seed: int, target: int) -> tuple[list[dict], dict]:
    """A6 as tiers of in-corpus citations received (resolved edges from other opinions): none, one, two or more."""
    src = "cuyahoga-eighth-district"
    adir = paths.out / "atoms" / paths.version
    e = np.load(adir / "edges" / "cites.npz")
    resolved = e["kind"] == 1
    other = (e["shard"] != e["target_shard"]) | (e["row"] != e["target_row"])
    received: collections.Counter = collections.Counter()
    for ts, tr, os_, or_ in zip(e["target_shard"][resolved & other].tolist(), e["target_row"][resolved & other].tolist(), e["shard"][resolved & other].tolist(), e["row"][resolved & other].tolist()):
        received[(ts, tr)] += 1
    shards = population_shards(paths, paths.version)
    per_class = max(MIN_CLASS, target // 3)
    pools: dict[int, list] = {0: [], 1: [], 2: []}
    for si, s in enumerate(shards):
        if s["source"] != src:
            continue
        _, per_row = atoms_by_row(adir, src, s["shard"])
        for r in per_row:
            pools[min(2, received.get((si, r), 0))].append((hash_order(seed, "a6", s["shard"], r), si, r))
    picks = [p for tier in pools.values() for p in sorted(tier)[:per_class]]
    by_shard: dict[int, set[int]] = collections.defaultdict(set)
    for _, si, r in picks:
        by_shard[si].add(r)
    rows = []
    for si, rs in by_shard.items():
        shard = shards[si]["shard"]
        for r, rec in fetch_records(paths.canon(src), shard, rs).items():
            f = rec["f"]
            labels = {a["kind"]: a["value"] for a in f.get("anchors", []) if a.get("kind") != "label:cited"}
            y = min(2, received.get((si, r), 0))
            rows.append({"source": src, "shard": shard, "row": r, "span": [0, len(rec["text"])], "text": rec["text"], "derived": False,
                         "fields": {"year": int((f.get("date_filed") or "0000")[:4]), "precedential": labels.get("label:precedential"), "optype": labels.get("label:optype"),
                                    "era": labels.get("label:era"), "court": f.get("court_id"), "text_chars": len(rec["text"])},
                         "cluster": f"opinion:{f.get('opinion_id') or r}", "y": y, "y_name": ("uncited", "cited_once", "cited_twice_or_more")[y]})
    rows.sort(key=lambda r: hash_order(seed, "a6", r["shard"], r["row"]))
    counts = collections.Counter(r["y"] for r in rows)
    if len(counts) < 3 or min(counts.values()) < MIN_CLASS:
        raise LensError("MNLM_LENS_CAL_UNDERPOWERED", f"a6-cuyahoga-cite-tier: {dict(counts)}")
    return rows, {"population_by_tier": {t: len(p) for t, p in pools.items()}, "per_class_target": per_class, "note": "A6 is a conditioning anchor only (spec Table 2); label:cited is withheld"}


A4_KEY_FIELDS = ("chapter", "verse", "editions_of_key")


def a4_keyfree(built: tuple[list[dict], dict]) -> tuple[list[dict], dict]:
    """Decision D-3: the same A4 rows with the fields that identify the book (and so the genre) removed from what lenses read."""
    rows, summary = built
    for r in rows:
        r["fields"] = {k: v for k, v in r["fields"].items() if k not in A4_KEY_FIELDS}
    return rows, {**summary, "withheld_fields": list(A4_KEY_FIELDS), "decision": "D-3 (#12)"}


def build_a6_cohort(paths: Paths, seed: int, target: int) -> tuple[list[dict], dict]:
    """Decision D-4: cited / uncited by another opinion, sampled with equal counts of each class inside every filing year, so the
    year (the clock readback F2 found) carries zero bits about the label by construction; year and era are not fields."""
    src = "cuyahoga-eighth-district"
    adir = paths.out / "atoms" / paths.version
    e = np.load(adir / "edges" / "cites.npz")
    keep = (e["kind"] == 1) & ((e["shard"] != e["target_shard"]) | (e["row"] != e["target_row"]))
    cited = set(zip(e["target_shard"][keep].tolist(), e["target_row"][keep].tolist()))
    shards = population_shards(paths, paths.version)
    pools: dict[tuple[str, int], list] = collections.defaultdict(list)
    recs: dict[tuple[int, int], dict] = {}
    for si, s in enumerate(shards):
        if s["source"] != src:
            continue
        _, per_row = atoms_by_row(adir, src, s["shard"])
        for r, rec in fetch_records(paths.canon(src), s["shard"], set(per_row)).items():
            year = (rec["f"].get("date_filed") or "")[:4]
            if not year.isdigit():
                continue
            recs[(si, r)] = rec
            pools[(year, int((si, r) in cited))].append((hash_order(seed, "a6c", s["shard"], r), si, r))
    years = sorted({y for y, _ in pools})
    k = {y: min(len(pools[(y, 0)]), len(pools[(y, 1)])) for y in years}
    total = 2 * sum(k.values())
    if total > target:
        k = {y: (v * target) // total for y, v in k.items()}
    picks = [(si, r, c) for y in years for c in (0, 1) for _, si, r in sorted(pools[(y, c)])[:k[y]]]
    rows = []
    for si, r, c in picks:
        rec, f = recs[(si, r)], recs[(si, r)]["f"]
        labels = {a["kind"]: a["value"] for a in f.get("anchors", []) if a.get("kind") not in ("label:cited", "label:era")}
        rows.append({"source": src, "shard": shards[si]["shard"], "row": r, "span": [0, len(rec["text"])], "text": rec["text"], "derived": False,
                     "fields": {"precedential": labels.get("label:precedential"), "optype": labels.get("label:optype"), "court": f.get("court_id"), "text_chars": len(rec["text"])},
                     "cluster": f"opinion:{f.get('opinion_id') or r}", "y": c, "y_name": ("uncited_in_corpus", "cited_in_corpus")[c]})
    rows.sort(key=lambda x: hash_order(seed, "a6c", x["shard"], x["row"]))
    counts = collections.Counter(x["y"] for x in rows)
    if len(counts) < 2 or min(counts.values()) < MIN_CLASS:
        raise LensError("MNLM_LENS_CAL_UNDERPOWERED", f"a6-cuyahoga-cited-cohort: {dict(counts)}")
    return rows, {"per_year": {y: v for y, v in k.items() if v}, "years": len([v for v in k.values() if v]), "decision": "D-4 (#12)",
                  "balance": "equal cited and uncited counts inside every filing year; year and era withheld; label:cited withheld"}


def builders(paths: Paths, seed: int, target: int, workers: int) -> dict:
    from anchors import build_a2

    return {"a1-oci-outcome3": lambda: build_a1_outcome3(paths), "a1-oci-ran-tests": lambda: build_a1_ran(paths),
            "a2-wikidata-p17": lambda: build_a2(paths, seed, target, prop="P17", fam="a2-wikidata-p17"),
            "a2-wikidata-p106": lambda: build_a2(paths, seed, target, prop="P106", fam="a2-wikidata-p106"),
            "a2-wikidata-p31-rare": lambda: build_a2(paths, seed, target, skip_classes=12, fam="a2-wikidata-p31-rare"),
            "a3-gdelt-next-quad": lambda: build_a3_next_quad(paths, seed, target, workers),
            "a6-cuyahoga-cite-tier": lambda: build_a6_cite_tier(paths, seed, target), "a6-cuyahoga-cited-cohort": lambda: build_a6_cohort(paths, seed, target),
            **{f"a2-wikidata-{p.lower()}": (lambda p=p: build_a2(paths, seed, target, prop=p, fam=f"a2-wikidata-{p.lower()}")) for p in ("P27", "P19", "P69", "P641", "P136", "P495", "P54", "P166", "P463", "P108", "P1412", "P131")}}
