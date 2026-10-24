"""Calibration builders for the row and document families of Stage 3 (issue #12): A3/A5 from the GDELT event rows (one
parallel pass over the 90 daily shards builds the dyad-by-day index and draws the candidates), A4 from the Stage 2 verse
alignment (one edition per key, chosen by hash), A6 from the Stage 2 citation edges. Called from anchors.py."""
from __future__ import annotations

import collections
import datetime as dt
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lenscommon import LensError, Paths, atoms_by_row, fetch_records, hash_order, iter_canon_docs, population_shards, read_jsonl  # noqa: E402

MIN_CLASS = 50
HORIZON_DAYS = 7
GENRES = ("law", "history", "wisdom", "psalm", "prophecy", "gospel", "acts", "letter", "apocalypse")
_SLUG_SPLIT = re.compile(r"[-_+.]+")


def _slug(url: str) -> str:
    try:
        path = urlparse(url).path.rstrip("/")
    except ValueError:
        return ""
    seg = path.rsplit("/", 1)[-1]
    seg = re.sub(r"\.(html?|php|aspx?|cfm|jsp)$", "", seg, flags=re.I)
    words = [w for w in _SLUG_SPLIT.split(seg) if len(w) >= 2 and not w.isdigit()]
    return " ".join(words)


def _num(v):
    try:
        return float(v) if v not in (None, "") else None
    except ValueError:
        return None


def _gdelt_scan(job: dict) -> dict:
    src, shard, canon_dir, adir, seed, frac, day = job["source"], job["shard"], Path(job["canon_dir"]), Path(job["adir"]), job["seed"], job["frac"], job["day"]
    arrays, per_row = atoms_by_row(adir, src, shard)
    keys: set[str] = set()
    observed: set[tuple[str, str]] = set()  # (Actor1|Actor2|EventCode, source domain) seen on this coding day
    cands = []
    n = 0
    for rec in iter_canon_docs(canon_dir, shard):
        if rec["row"] not in per_row:
            continue
        n += 1
        f = rec["f"]
        a1, a2, q = f.get("Actor1Code") or "", f.get("Actor2Code") or "", f.get("QuadClass") or ""
        if not a1 or not a2 or not q:
            continue
        keys.add(f"{a1}|{a2}|{q}")
        observed.add((f"{a1}|{a2}|{f.get('EventCode') or ''}", (urlparse(f.get("SOURCEURL") or "").netloc or "").lower()))
        if hash_order(seed, "gdelt", shard, rec["row"]) / 2**64 >= frac:
            continue
        cols = rec["text"].split("\t")
        if len(cols) < 61:
            continue
        coding_day = dt.date(int(shard[6:10]), int(shard[10:12]), int(shard[12:14]))
        try:
            event_day = dt.datetime.strptime(f["Day"], "%Y%m%d").date()
        except ValueError:
            continue
        slug = _slug(f.get("SOURCEURL") or "")
        text = f"{cols[6].strip() or a1} {cols[16].strip() or a2}: {slug or '(no slug)'}"
        fields = {"GoldsteinScale": _num(f.get("GoldsteinScale")), "AvgTone": _num(f.get("AvgTone")), "NumMentions": _num(f.get("NumMentions")),
                  "NumSources": _num(f.get("NumSources")), "NumArticles": _num(cols[33]), "ActionGeo_Lat": _num(f.get("ActionGeo_Lat")), "ActionGeo_Long": _num(f.get("ActionGeo_Long")),
                  "knowledge_age_days": (coding_day - event_day).days, "weekday": coding_day.weekday(), "day_of_year": coding_day.timetuple().tm_yday,
                  "EventRootCode": cols[28], "EventBaseCode": cols[27], "EventCode": f.get("EventCode"), "QuadClass": q, "IsRootEvent": cols[25],
                  "Actor1Code": a1, "Actor2Code": a2, "Actor1CountryCode": cols[7], "Actor2CountryCode": cols[17], "ActionGeo_CountryCode": f.get("ActionGeo_CountryCode"),
                  "source_domain": (urlparse(f.get("SOURCEURL") or "").netloc or "").lower(), "day_index": day}
        cands.append({"source": src, "shard": shard, "row": rec["row"], "span": None, "text": text, "derived": True, "fields": fields,
                      "key": f"{a1}|{a2}|{q}", "key_event": f"{a1}|{a2}|{f.get('EventCode') or ''}", "cluster": f"{a1}|{a2}"})
    return {"day": day, "events": n, "keys": sorted(keys), "observed": sorted(observed), "cands": cands}


def build_gdelt(paths: Paths, seed: int, target: int, workers: int, which: str) -> tuple[list[dict], dict]:
    src = "gdelt-events-90d"
    adir = paths.out / "atoms" / paths.version
    shards = sorted(s["shard"] for s in population_shards(paths, paths.version) if s["source"] == src)
    frac = 2.5 * target / (max(1, len(shards) - HORIZON_DAYS) * 110_000)
    jobs = [{"source": src, "shard": sh, "canon_dir": str(paths.canon(src)), "adir": str(adir), "seed": seed, "frac": frac, "day": d} for d, sh in enumerate(shards)]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(_gdelt_scan, jobs))
    days_of: dict[str, int] = collections.defaultdict(int)
    seen_by: dict[str, list[tuple[int, str]]] = collections.defaultdict(list)  # event key -> (coding day, domain) observations
    for res in results:
        bit = 1 << res["day"]
        for k in res["keys"]:
            days_of[k] |= bit
        for k, dom in res["observed"]:
            seen_by[k].append((res["day"], dom))
    last_day = len(shards) - 1
    cands = [c for res in results for c in res["cands"] if c["fields"]["day_index"] + HORIZON_DAYS <= last_day]
    cands.sort(key=lambda c: hash_order(seed, "gdelt", c["shard"], c["row"]))
    rows = []
    for c in cands[:target]:
        d = c["fields"]["day_index"]
        mask = ((1 << (d + HORIZON_DAYS + 1)) - 1) & ~((1 << (d + 1)) - 1)
        c = dict(c)
        key, key_event = c.pop("key"), c.pop("key_event")
        if which == "a3":
            y = int(bool(days_of[key] & mask))
            c["y"], c["y_name"] = y, "recurs_7d" if y else "no_recurrence_7d"
        else:
            dom = c["fields"]["source_domain"]
            y = int(any(day > d and od != dom for day, od in seen_by.get(key_event, ())))
            c["y"], c["y_name"] = y, "reobserved_other_source" if y else "not_reobserved"
            c["fields"] = {k: v for k, v in c["fields"].items() if k not in ("NumMentions", "NumSources", "NumArticles")}
        rows.append(c)
    fam = "a3-gdelt-dyad-7d" if which == "a3" else "a5-gdelt-sources"
    if len(rows) < 2 * MIN_CLASS:
        raise LensError("MNLM_LENS_CAL_UNDERPOWERED", f"{fam}: {len(rows)} candidates")
    counts = collections.Counter(r["y"] for r in rows)
    if min(counts.values()) < MIN_CLASS:
        raise LensError("MNLM_LENS_CAL_UNDERPOWERED", f"{fam}: class counts {dict(counts)}")
    summary = {"days": len(shards), "events_unique": sum(r["events"] for r in results), "dyad_quad_keys": len(days_of), "horizon_days": HORIZON_DAYS,
               "candidate_days": f"0..{last_day - HORIZON_DAYS}", "sample_fraction": frac, "candidates": len(cands),
               "text": "derived: Actor1Name Actor2Name: SOURCEURL slug words (tagged derived; the row has no prose)",
               "window_note": "the realisation window had already closed when the anchor was read; no forecast was sealed before it opened" if which == "a3"
               else "Y: the same (Actor1, Actor2, EventCode) coded on a later day from a different source domain within the 90 days; NumMentions, NumSources and NumArticles are withheld from the fields"}
    return rows, summary


def build_a4(spaths: Paths, seed: int, target: int) -> tuple[list[dict], bool | dict]:
    adir = spaths.out / "atoms" / spaths.version
    align = adir / "align"
    vi = np.load(align / "verse_index.npz")
    keys = {k["key_id"]: k for k in read_jsonl(align / "verse_keys.jsonl")}
    gidx = {g: i for i, g in enumerate(GENRES)}
    eligible = [k for k in keys.values() if k["editions"] >= 2 and k["genre"] in gidx]
    per_class = max(MIN_CLASS, target // len(GENRES))
    chosen: dict[int, dict] = {}
    taken = collections.Counter()
    for k in sorted(eligible, key=lambda k: hash_order(seed, "a4", k["key_id"])):
        if taken[k["genre"]] < per_class:
            chosen[k["key_id"]] = k
            taken[k["genre"]] += 1
    best: dict[int, tuple[int, int, int]] = {}
    kid, sh, rw = vi["key_id"], vi["shard"], vi["row"]
    for i in np.flatnonzero(np.isin(kid, list(chosen))):
        k, s, r = int(kid[i]), int(sh[i]), int(rw[i])
        h = hash_order(seed, "a4e", k, s, r)
        if k not in best or h < best[k][0]:
            best[k] = (h, s, r)
    shards = population_shards(spaths, spaths.version)
    by_shard: dict[int, dict[int, int]] = collections.defaultdict(dict)
    for k, (_, s, r) in best.items():
        by_shard[s][r] = k
    rows = []
    for s, rowmap in by_shard.items():
        src, shard = shards[s]["source"], shards[s]["shard"]
        for r, rec in fetch_records(spaths.canon(src), shard, set(rowmap)).items():
            k = chosen[rowmap[r]]
            f = rec["f"]
            rows.append({"source": src, "shard": shard, "row": r, "span": [0, len(rec["text"])], "text": rec["text"], "derived": False,
                         "fields": {"edition": f.get("edition") or shard, "chapter": k["chapter"], "verse": k["verse"], "text_chars": len(rec["text"]), "editions_of_key": k["editions"]},
                         "cluster": f"key:{k['key_id']}", "y": gidx[k["genre"]], "y_name": k["genre"]})
    rows.sort(key=lambda r: hash_order(seed, "a4", int(r["cluster"][4:])))
    counts = collections.Counter(r["y"] for r in rows)
    if len(counts) < 2 or min(counts.values()) < MIN_CLASS:
        raise LensError("MNLM_LENS_CAL_UNDERPOWERED", f"a4-scripture-genre: {dict(counts)}")
    summary = {"eligible_keys_ge2_editions": len(eligible), "per_class_target": per_class, "genres": list(GENRES), "editions_in_sample": len({r["fields"]["edition"] for r in rows}),
               "edition_choice": "one edition per key: the (shard,row) with the smallest BLAKE3 ordinal under the seed"}
    return rows, summary


def build_a6(paths: Paths, seed: int, target: int) -> tuple[list[dict], dict]:
    src = "cuyahoga-eighth-district"
    adir = paths.out / "atoms" / paths.version
    e = np.load(adir / "edges" / "cites.npz")
    resolved = e["kind"] == 1
    other = (e["shard"] != e["target_shard"]) | (e["row"] != e["target_row"])
    cited = set(zip(e["target_shard"][resolved & other].tolist(), e["target_row"][resolved & other].tolist()))
    shards = population_shards(paths, paths.version)
    pos, neg = [], []
    for si, s in enumerate(shards):
        if s["source"] != src:
            continue
        _, per_row = atoms_by_row(adir, src, s["shard"])
        for r in per_row:
            (pos if (si, r) in cited else neg).append((hash_order(seed, "a6", s["shard"], r), si, r))
    per_class = max(MIN_CLASS, target // 2)
    picks = sorted(pos)[:per_class] + sorted(neg)[:per_class]
    by_shard: dict[int, set[int]] = collections.defaultdict(set)
    for _, si, r in picks:
        by_shard[si].add(r)
    rows = []
    for si, rs in by_shard.items():
        shard = shards[si]["shard"]
        for r, rec in fetch_records(paths.canon(src), shard, rs).items():
            f = rec["f"]
            labels = {a["kind"]: a["value"] for a in f.get("anchors", []) if a.get("kind") != "label:cited"}
            y = int((si, r) in cited)
            rows.append({"source": src, "shard": shard, "row": r, "span": [0, len(rec["text"])], "text": rec["text"], "derived": False,
                         "fields": {"year": int((f.get("date_filed") or "0000")[:4]), "precedential": labels.get("label:precedential"), "optype": labels.get("label:optype"),
                                    "era": labels.get("label:era"), "court": f.get("court_id"), "text_chars": len(rec["text"])},
                         "cluster": f"opinion:{f.get('opinion_id') or r}", "y": y, "y_name": "cited_in_corpus" if y else "uncited_in_corpus"})
    rows.sort(key=lambda r: hash_order(seed, "a6", r["shard"], r["row"]))
    counts = collections.Counter(r["y"] for r in rows)
    if len(counts) < 2 or min(counts.values()) < MIN_CLASS:
        raise LensError("MNLM_LENS_CAL_UNDERPOWERED", f"a6-cuyahoga-cited: {dict(counts)}")
    summary = {"opinions_cited_in_corpus": len(cited), "population_opinions": len(pos) + len(neg), "per_class_target": per_class,
               "note": "A6 is a conditioning anchor only (spec Table 2); label:cited of the Calyx anchors is withheld from the fields"}
    return rows, summary
