"""Second commissioning round of context lenses (issue #12, after admission run 6). Every question of every anchor family
has one or two carriers and the next candidate adds under 0.05 bits, and every second lens that was admitted reads an
input the first does not (fields beside text on A4, the module's static surface on A1, the entity's relations on A2, actor
activity beside event recurrence on A5). Each lens here reads an input no registered lens reads: the task prompt of the A1
record (the registered text lenses read the solution body only); the neighbouring verses of the same chapter in the same
edition for A4 (the verse itself excluded); the source domain's and the action country's prior-day coding history for
A3/A5; the in-citation counts of the opinions an A6 opinion cites (two-hop authority; the opinion's own in-citations are
the label and are never read) and the judge roster its header names; the asked property's values held by the entity's
neighbours for the A2 P17/P106 questions (the entity's own statements of that property, in either direction, are the label
and are excluded). Contexts are cached under <lens_root>/ctx/ as in encoders_ctx.py; encoders.py, encoders_ctx.py and
encoders_kg.py are byte-identical so no registered lens identity moves. Cost: one streamed pass per shard holding a
calibration row (prompts, verses), one process-pool pass over the GDELT day shards, one streamed pass over the Wikidata5m
triples, the citation arrays in memory; features are O(rows × context size)."""
from __future__ import annotations

import collections
import io
import math
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

import blake3
import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from encoders import Slot, _hash_index  # noqa: E402
from encoders_ctx import CITES_SOURCE, CODE_SOURCE, GDELT_SOURCE, KG_SOURCE, KIND_WEBCITE, WINDOWS, _cached, _cal_rows_of  # noqa: E402
from encoders_kg import _TRIPLE, LABEL_PROPERTY, SHARDS  # noqa: E402
from lenscommon import FAMILIES, FAMILY_SPEC, Paths, fetch_records, iter_canon_docs, lens_root, population_shards, read_family, read_jsonl  # noqa: E402

ENCODERS_CTX2_B3 = blake3.blake3(Path(__file__).read_bytes()).hexdigest()
SCRIPTURE_VERSION = "scripture-v1"
CHAPTER_RADIUS = 6
WORD_DIMS, TRIGRAM_DIMS, ROSTER_DIMS, VALUE_DIMS = 2048, 4096, 256, 512
_WORD = re.compile(r"[a-z0-9_]+")
_JUDGES = re.compile(r"^Judges?:\s*(.+)$", re.M)
_SPLIT_NAMES = re.compile(r",|;| and |&")


def _sparse_rows(counts: list[dict[int, float]], dims: int) -> tuple[sp.csr_matrix, np.ndarray]:
    ri, ci, v = [], [], []
    for i, tf in enumerate(counts):
        for j, x in tf.items():
            ri.append(i)
            ci.append(j)
            v.append(x)
    m = sp.csr_matrix((v, (ri, ci)), shape=(len(counts), dims), dtype=np.float32)
    nrm = np.sqrt(np.asarray(m.multiply(m).sum(1)).ravel())
    m = sp.diags(np.where(nrm > 0, 1 / np.where(nrm > 0, nrm, 1), 0.0)) @ m
    return m.tocsr(), nrm == 0


def _word_hist(text: str, dims: int) -> dict[int, float]:
    toks = _WORD.findall(text.lower())
    tf: dict[int, float] = {}
    for k, t in enumerate(toks):
        for g in ([t, f"{t} {toks[k + 1]}"] if k + 1 < len(toks) else [t]):
            j, s = _hash_index(g, dims=dims)
            tf[j] = tf.get(j, 0.0) + s
    return tf


def _trigram_hist(text: str, dims: int) -> dict[int, float]:
    t, tf = text.lower(), {}
    for k in range(len(t) - 2):
        j, s = _hash_index(t[k:k + 3], dims=dims)
        tf[j] = tf.get(j, 0.0) + s
    return tf


def _absent(rows: list[dict], dims: int, sim: str, why: str, sparse: bool = False) -> Slot:
    X = sp.csr_matrix((len(rows), dims), dtype=np.float32) if sparse else np.zeros((len(rows), dims), np.float32)
    return Slot(X, np.ones(len(rows), bool), {"absent": why}, sim, sparse=sparse)


# ------------------------------------------------------------------ A1: the task prompt of the record (text lenses read the body only)
def _prompt_ctx(paths: Paths) -> dict:
    rows = _cal_rows_of(paths, CODE_SOURCE)
    want = {(r["shard"], int(r["row"])): int(r["fields"]["prompt_chars"]) for r in rows}
    keys = sorted(f"{s}:{r}:{c}" for (s, r), c in want.items())

    def build() -> dict:
        by_shard: dict[str, set[int]] = collections.defaultdict(set)
        for s, r in want:
            by_shard[s].add(r)
        per = {}
        for s, rws in sorted(by_shard.items()):
            for r, rec in fetch_records(paths.canon(CODE_SOURCE), s, rws).items():
                per[f"{s}:{r}"] = rec["text"][: want[(s, r)]]  # the prompt only: the body and anything after it are never cached here
        return {"source": CODE_SOURCE, "read": "record text before the solution fence (fields.prompt_chars)", "prompts": per}

    return _cached(paths, "prompt", ["prompts", *keys], build)


PROMPT_SENSORS = ["prompt_chars", "prompt_lines", "code_fences", "bold_sections", "numbers", "bounds", "samples", "mentions_class", "mentions_function", "mentions_complexity",
                  "mentions_recursion", "mentions_edge_cases", "n_tests", "prompt_to_body"]
_SENSOR_WORDS = {"mentions_class": r"\bclass\b", "mentions_function": r"\bfunction\b|\bdef\b|\bmethod\b", "mentions_complexity": r"complexity|\bO\(|efficient|optimi",
                 "mentions_recursion": r"recurs", "mentions_edge_cases": r"edge case|empty|negative|zero|invalid|error"}


def _prompt_lens(name: str, rows: list[dict], paths: Paths) -> Slot:
    sparse = name == "code-prompt-signature"
    if not any(r["source"] == CODE_SOURCE for r in rows):
        return _absent(rows, WORD_DIMS if sparse else len(PROMPT_SENSORS), "cosine" if sparse else "neg_cheb", "no OpenCodeInstruct atom in this family", sparse)
    prompts = _prompt_ctx(paths)["prompts"]
    if sparse:
        counts = [_word_hist(prompts.get(f"{r['shard']}:{r['row']}", ""), WORD_DIMS) if r["source"] == CODE_SOURCE else {} for r in rows]
        m, empty = _sparse_rows(counts, WORD_DIMS)
        return Slot(m, empty, {"dims": WORD_DIMS, "features": "signed hashed word unigrams and bigrams of the task prompt, L2-normalised", "read": "the prompt only"}, "cosine", sparse=True)
    X, absent = np.zeros((len(rows), len(PROMPT_SENSORS)), np.float32), np.ones(len(rows), bool)
    for i, r in enumerate(rows):
        p = prompts.get(f"{r['shard']}:{r['row']}") if r["source"] == CODE_SOURCE else None
        if not p:
            continue
        low = p.lower()
        vals = [math.log1p(len(p)), math.log1p(p.count("\n") + 1), math.log1p(p.count("```") // 2), math.log1p(p.count("**") // 2), math.log1p(len(re.findall(r"\d+", p))),
                math.log1p(len(re.findall(r"≤|<=|>=|≥|\bat most\b|\bat least\b", low))), math.log1p(len(re.findall(r"sample|example", low)))]
        vals += [math.log1p(len(re.findall(_SENSOR_WORDS[k], low))) for k in PROMPT_SENSORS[7:12]]
        vals += [math.log1p(int(r["fields"].get("n_tests") or 0)), math.log(len(p) / max(1, int(r["fields"].get("body_chars") or len(r["text"]))))]
        X[i] = vals
        absent[i] = False
    return Slot(X, absent, {"layout": PROMPT_SENSORS, "log1p": "counts", "read": "the prompt only"}, "neg_cheb")


# ------------------------------------------------------------------ A4: the neighbouring verses of the same chapter in the same edition
def _spaths(paths: Paths) -> Paths:
    s = Paths(paths.root, paths.out, SCRIPTURE_VERSION)
    s.canon_version = SCRIPTURE_VERSION  # the p0 canon override never applies to the scripture partition (anchors.py)
    return s


def _scripture_rows(paths: Paths) -> list[dict]:
    """Calibration rows of the families read from the scripture partition (every edition source, Hebrew Tanach included)."""
    rows = []
    for fam in FAMILIES:
        if FAMILY_SPEC[fam]["version"] == SCRIPTURE_VERSION and (lens_root(paths) / "cal" / f"{fam}.jsonl").exists():
            rows += read_family(lens_root(paths) / "cal", fam)[0]
    return rows


def _is_verse(r: dict) -> bool:
    return FAMILY_SPEC.get(r.get("family", ""), {}).get("version") == SCRIPTURE_VERSION and str(r.get("cluster", "")).startswith("key:")


def _chapter_ctx(paths: Paths) -> dict:
    want = sorted({(r["source"], r["shard"], int(r["cluster"][4:])) for r in _scripture_rows(paths) if _is_verse(r)})

    def build() -> dict:
        spaths = _spaths(paths)
        align = spaths.out / "atoms" / SCRIPTURE_VERSION / "align"
        keys = {k["key_id"]: k for k in read_jsonl(align / "verse_keys.jsonl")}
        by_chapter: dict[tuple, list] = collections.defaultdict(list)
        for k in keys.values():
            by_chapter[(k["book"], k["chapter"])].append(k)
        vi = np.load(align / "verse_index.npz")
        loc: dict[int, dict[int, int]] = collections.defaultdict(dict)
        for k, s, r in zip(vi["key_id"].tolist(), vi["shard"].tolist(), vi["row"].tolist()):
            loc[k][s] = r
        shard_idx = {(s["source"], s["shard"]): i for i, s in enumerate(population_shards(spaths, SCRIPTURE_VERSION))}
        need: dict[tuple, set[int]] = collections.defaultdict(set)
        plan = {}
        for src, sh, kid in want:
            k, si = keys[kid], shard_idx[(src, sh)]
            chapter = by_chapter[(k["book"], k["chapter"])]
            nb = sorted((n["verse"], loc[n["key_id"]][si]) for n in chapter if n["key_id"] != kid and abs(n["verse"] - k["verse"]) <= CHAPTER_RADIUS and si in loc[n["key_id"]])
            plan[f"{src}:{sh}:{kid}"] = {"chapter_verses": len(chapter), "verse": k["verse"], "neighbours": nb}
            need[(src, sh)].update(r for _, r in nb)
        texts = {}
        for (src, sh), rws in sorted(need.items()):
            for r, rec in fetch_records(spaths.canon(src), sh, rws).items():
                texts[f"{src}:{sh}:{r}"] = rec["text"]
        entries = {}
        for key, p in plan.items():
            src, sh, _ = key.split(":", 2)
            entries[key] = {"chapter_verses": p["chapter_verses"], "verse": p["verse"], "neighbour_texts": [texts[f"{src}:{sh}:{r}"] for _, r in p["neighbours"] if f"{src}:{sh}:{r}" in texts]}
        return {"version": SCRIPTURE_VERSION, "radius": CHAPTER_RADIUS, "read": "verses of the same book and chapter within the radius, same edition shard, the verse itself excluded", "entries": entries}

    return _cached(paths, "chapter", ["radius", str(CHAPTER_RADIUS), *[f"{s}:{sh}:{k}" for s, sh, k in want]], build)


CHAPTER_SHAPE = ["chapter_verses", "verse_position", "neighbours_found", "neighbour_chars_mean", "neighbour_chars_cv", "own_to_neighbour_chars"]


def _chapter_lens(name: str, rows: list[dict], paths: Paths) -> Slot:
    sparse = name == "scripture-chapter-context"
    if not any(_is_verse(r) for r in rows):
        return _absent(rows, TRIGRAM_DIMS if sparse else len(CHAPTER_SHAPE), "cosine" if sparse else "neg_cheb", "no scripture atom in this family", sparse)
    entries = _chapter_ctx(paths)["entries"]
    if sparse:
        counts = []
        for r in rows:
            e = entries.get(f"{r['source']}:{r['shard']}:{r['cluster'][4:]}") if _is_verse(r) else None
            counts.append(_trigram_hist(" ".join(e["neighbour_texts"]), TRIGRAM_DIMS) if e and e["neighbour_texts"] else {})
        m, empty = _sparse_rows(counts, TRIGRAM_DIMS)
        return Slot(m, empty, {"dims": TRIGRAM_DIMS, "features": "signed hashed character trigrams of the neighbouring verses, L2-normalised", "radius": CHAPTER_RADIUS}, "cosine", sparse=True)
    X, absent = np.zeros((len(rows), len(CHAPTER_SHAPE)), np.float32), np.ones(len(rows), bool)
    for i, r in enumerate(rows):
        e = entries.get(f"{r['source']}:{r['shard']}:{r['cluster'][4:]}") if _is_verse(r) else None
        if not e or not e["neighbour_texts"]:
            continue
        lens = np.asarray([len(t) for t in e["neighbour_texts"]], np.float64)
        X[i] = [math.log1p(e["chapter_verses"]), e["verse"] / max(1, e["chapter_verses"]), math.log1p(len(lens)), math.log1p(lens.mean()), lens.std() / max(1.0, lens.mean()),
                math.log(max(1, len(r["text"])) / max(1.0, lens.mean()))]
        absent[i] = False
    return Slot(X, absent, {"layout": CHAPTER_SHAPE, "radius": CHAPTER_RADIUS}, "neg_cheb")


# ------------------------------------------------------------------ A3/A5: source-domain and action-country prior-day history
def _gdelt2_day(job: dict) -> dict:
    doms, geos, pairs = set(job["domains"]), set(job["geos"]), set(job["pairs"])
    dom_n, geo_n, pair_geo_n = collections.Counter(), collections.Counter(), collections.Counter()
    dom_pairs, dom_codes, dom_geos = collections.defaultdict(set), collections.defaultdict(set), collections.defaultdict(set)
    geo_pairs, geo_doms = collections.defaultdict(set), collections.defaultdict(set)
    geo_quad = collections.defaultdict(lambda: [0, 0, 0, 0])
    for rec in iter_canon_docs(Path(job["canon_dir"]), job["shard"]):
        f = rec["f"]
        dom, geo = (urlparse(f.get("SOURCEURL") or "").netloc or "").lower(), f.get("ActionGeo_CountryCode") or ""
        a1, a2 = f.get("Actor1Code") or "", f.get("Actor2Code") or ""
        p = f"{a1}|{a2}" if a1 and a2 else ""
        if dom in doms:
            dom_n[dom] += 1
            dom_codes[dom].add(f.get("EventCode") or "")
            dom_geos[dom].add(geo)
            if p:
                dom_pairs[dom].add(p)
        if geo in geos:
            geo_n[geo] += 1
            geo_doms[geo].add(dom)
            q = f.get("QuadClass") or ""
            if q in ("1", "2", "3", "4"):
                geo_quad[geo][int(q) - 1] += 1
            if p:
                geo_pairs[geo].add(p)
                if p in pairs:
                    pair_geo_n[f"{p}|{geo}"] += 1
    return {"day": job["day"], "shard": job["shard"], "dom_n": dict(dom_n), "dom_pairs": {k: len(v) for k, v in dom_pairs.items()}, "dom_codes": {k: len(v) for k, v in dom_codes.items()},
            "dom_geos": {k: len(v) for k, v in dom_geos.items()}, "geo_n": dict(geo_n), "geo_pairs": {k: len(v) for k, v in geo_pairs.items()}, "geo_doms": {k: len(v) for k, v in geo_doms.items()},
            "geo_quad": dict(geo_quad), "pair_geo_n": dict(pair_geo_n)}


def _gdelt2_ctx(paths: Paths) -> dict:
    rows = _cal_rows_of(paths, GDELT_SOURCE)
    doms = sorted({r["fields"].get("source_domain") or "" for r in rows} - {""})
    geos = sorted({r["fields"].get("ActionGeo_CountryCode") or "" for r in rows} - {""})
    pairs = sorted({f"{r['fields']['Actor1Code']}|{r['fields']['Actor2Code']}" for r in rows})
    shards = sorted(s["shard"] for s in population_shards(paths, paths.version) if s["source"] == GDELT_SOURCE)

    def build() -> dict:
        import os

        jobs = [{"shard": sh, "canon_dir": str(paths.canon(GDELT_SOURCE)), "day": d, "domains": doms, "geos": geos, "pairs": pairs} for d, sh in enumerate(shards)]
        with ProcessPoolExecutor(max_workers=min(28, os.cpu_count() or 1)) as ex:
            days = sorted(ex.map(_gdelt2_day, jobs), key=lambda r: r["day"])
        return {"source": GDELT_SOURCE, "clock": "coding day of the shard; features read days strictly before the atom's day_index", "distinct": "per-day distinct counts, summed over the window", "shards": shards, "days": days}

    return _cached(paths, "gdelt2", ["shards", *shards, "domains", *doms, "geos", *geos, "pairs", *pairs], build)


SOURCE_HISTORY = [f"{c}_{t}" for _, t in WINDOWS for c in ("dom", "dom_days", "dom_pairs", "dom_codes", "dom_geos")] + ["dom_sameday_other"]
GEO_HISTORY = [f"{c}_{t}" for _, t in WINDOWS for c in ("geo", "geo_days", "geo_pairs", "geo_doms", "pair_geo")] + ["geo_quad1_28", "geo_quad2_28", "geo_quad3_28", "geo_quad4_28", "geo_sameday_other"]


def _gdelt2_lens(name: str, rows: list[dict], paths: Paths) -> Slot:
    source = name == "gdelt-source-history"
    names = SOURCE_HISTORY if source else GEO_HISTORY
    if not any(r["source"] == GDELT_SOURCE for r in rows):
        return _absent(rows, len(names), "neg_cheb", "no GDELT atom in this family")
    days = _gdelt2_ctx(paths)["days"]
    X, absent = np.zeros((len(rows), len(names)), np.float32), np.ones(len(rows), bool)
    for i, r in enumerate(rows):
        f = r["fields"]
        key = (f.get("source_domain") or "") if source else (f.get("ActionGeo_CountryCode") or "")
        if r["source"] != GDELT_SOURCE or not key or f.get("day_index") is None:
            continue
        d = int(f["day_index"])
        if d >= len(days) or days[d]["shard"] != r["shard"]:
            continue
        pg = f"{f.get('Actor1Code') or ''}|{f.get('Actor2Code') or ''}|{key}"
        col = 0
        for W, _ in WINDOWS:
            n = nd = c1 = c2 = c3 = 0
            for dd in range(max(0, d - W), d):
                day = days[dd]
                c = (day["dom_n"] if source else day["geo_n"]).get(key, 0)
                n, nd = n + c, nd + (c > 0)
                if source:
                    c1, c2, c3 = c1 + day["dom_pairs"].get(key, 0), c2 + day["dom_codes"].get(key, 0), c3 + day["dom_geos"].get(key, 0)
                else:
                    c1, c2, c3 = c1 + day["geo_pairs"].get(key, 0), c2 + day["geo_doms"].get(key, 0), c3 + day["pair_geo_n"].get(pg, 0)
            X[i, col:col + 5] = [math.log1p(n), nd, math.log1p(c1), math.log1p(c2), math.log1p(c3)]
            col += 5
        same = days[d]
        if source:
            X[i, col] = math.log1p(max(0, same["dom_n"].get(key, 0) - 1))
        else:
            q = np.zeros(4)
            for dd in range(max(0, d - 28), d):
                if key in days[dd]["geo_quad"]:
                    q += days[dd]["geo_quad"][key]
            X[i, col:col + 4] = q / max(1, q.sum())
            X[i, col + 4] = math.log1p(max(0, same["geo_n"].get(key, 0) - 1))
        absent[i] = False
    return Slot(X, absent, {"layout": names, "clock": "days strictly before day_index (same-day columns exclude the atom itself)", "log1p": "counts", "key": "SOURCEURL domain" if source else "ActionGeo_CountryCode"}, "neg_cheb")


# ------------------------------------------------------------------ A6: two-hop citation authority and the judge roster
AUTHORITY = ["targets", "in_cites_sum", "in_cites_max", "in_cites_mean_log", "targets_cited_by_others", "targets_cited_ge10", "other_citing_opinions"]


def _authority_lens(rows: list[dict], paths: Paths) -> Slot:
    if not any(r["source"] == CITES_SOURCE for r in rows):
        return _absent(rows, len(AUTHORITY), "neg_cheb", "no Cuyahoga atom in this family")
    e = np.load(paths.out / "atoms" / paths.version / "edges" / "cites.npz")
    res = e["kind"] == KIND_WEBCITE
    links = {(int(s), int(r), int(ts), int(tr)) for s, r, ts, tr in zip(e["shard"][res], e["row"][res], e["target_shard"][res], e["target_row"][res])}
    citers: dict[tuple, set] = collections.defaultdict(set)
    out: dict[tuple, set] = collections.defaultdict(set)
    for s, r, ts, tr in links:
        if (s, r) != (ts, tr):
            citers[(ts, tr)].add((s, r))
            out[(s, r)].add((ts, tr))
    shard_index = {s["shard"]: i for i, s in enumerate(population_shards(paths, paths.version)) if s["source"] == CITES_SOURCE}
    X, absent = np.zeros((len(rows), len(AUTHORITY)), np.float32), np.ones(len(rows), bool)
    for i, r in enumerate(rows):
        if r["source"] != CITES_SOURCE:
            continue
        me = (shard_index.get(r["shard"]), int(r["row"]))
        targets = out.get(me, set())
        others = [citers[t] - {me} for t in targets]  # the in-citations of what this opinion cites, from opinions other than itself
        counts = np.asarray([len(o) for o in others], np.float64)
        X[i] = [math.log1p(len(targets)), math.log1p(counts.sum()), math.log1p(counts.max() if len(counts) else 0), float(np.log1p(counts).mean()) if len(counts) else 0.0,
                float((counts >= 1).mean()) if len(counts) else 0.0, float((counts >= 10).mean()) if len(counts) else 0.0, math.log1p(len(set().union(*others)) if others else 0)]
        absent[i] = False
    return Slot(X, absent, {"layout": AUTHORITY, "read": "resolved out-citations and the in-citations of their targets from other opinions; the opinion's own in-citations are the A6 label and are not read"}, "neg_cheb")


def _roster_lens(rows: list[dict]) -> Slot:
    if not any(r["source"] == CITES_SOURCE for r in rows):
        return _absent(rows, ROSTER_DIMS + 1, "cosine", "no Cuyahoga atom in this family", sparse=True)
    counts = []
    for r in rows:
        m = _JUDGES.search(r["text"][:600]) if r["source"] == CITES_SOURCE else None
        tf: dict[int, float] = {}
        if m:
            names = [n.strip().lower() for n in _SPLIT_NAMES.split(m.group(1)) if n.strip()]
            for n in names:
                j, _ = _hash_index("judge", n, dims=ROSTER_DIMS)
                tf[j] = 1.0
            tf[ROSTER_DIMS] = float(len(names))
        counts.append(tf)
    m2, empty = _sparse_rows(counts, ROSTER_DIMS + 1)
    return Slot(m2, empty, {"dims": ROSTER_DIMS, "features": "hashed multi-hot of the judge names in the header line, plus the roster size", "read": "the Judges: header line"}, "cosine", sparse=True)


# ------------------------------------------------------------------ A2 P17/P106: the asked property's values held by the neighbours
def _kgv_ctx(paths: Paths) -> dict:
    ents = sorted({r["cluster"] for r in _cal_rows_of(paths, KG_SOURCE)})
    props = sorted(set(LABEL_PROPERTY.values()) - {"P31"})

    def build() -> dict:
        import zstandard

        want = set(ents)
        value: dict[str, dict[str, str]] = {p: {} for p in props}
        edges: dict[str, list] = collections.defaultdict(list)  # wanted entity -> (property, neighbour, direction)
        triples = 0
        for sh in SHARDS:
            with (paths.canon(KG_SOURCE) / f"{sh}.jsonl.zst").open("rb") as fh, zstandard.ZstdDecompressor().stream_reader(fh) as zr:
                for line in io.TextIOWrapper(zr, encoding="utf-8"):
                    m = _TRIPLE.search(line)
                    if not m:
                        continue
                    o, p, s = m.groups()
                    triples += 1
                    if p in value:
                        value[p].setdefault(s, o)
                    if s in want:
                        edges[s].append((p, o))
                    if o in want:
                        edges[o].append((p, s))
        per = {}
        for e in ents:
            per[e] = {}
            for p in props:
                # neighbours reached by the asked property itself (either direction) carry the label's object and are excluded
                hist = collections.Counter(value[p][nb] for q, nb in edges[e] if q != p and nb in value[p])
                per[e][p] = dict(hist)
        return {"source": KG_SOURCE, "shards": SHARDS, "triples": triples, "properties": props, "excluded": "edges of the asked property in either direction", "entities": per}

    return _cached(paths, "kgv", ["shards", *SHARDS, "properties", *props, "entities", *ents], build)


def _values_lens(rows: list[dict], paths: Paths) -> Slot:
    if not any(r["source"] == KG_SOURCE and LABEL_PROPERTY.get(r.get("family", "")) not in (None, "P31") for r in rows):
        return _absent(rows, VALUE_DIMS, "cosine", "no A2 P17/P106 question in this family")
    ents = _kgv_ctx(paths)["entities"]
    X, absent = np.zeros((len(rows), VALUE_DIMS), np.float32), np.ones(len(rows), bool)
    for i, r in enumerate(rows):
        label = LABEL_PROPERTY.get(r.get("family", ""))
        e = ents.get(r["cluster"]) if (r["source"] == KG_SOURCE and label and label != "P31") else None
        hist = e.get(label) if e else None
        if not hist:
            continue
        for v, n in hist.items():
            j, _ = _hash_index("value", label, v, dims=VALUE_DIMS)
            X[i, j] += n
        X[i] /= X[i].sum()
        absent[i] = False
    return Slot(X, absent, {"hash_dims": VALUE_DIMS, "layout": "normalised histogram of the asked property's values over the neighbours", "excluded": "the entity's own statements of the asked property, in either direction"}, "cosine")


LENSES = {"code-prompt-signature": _prompt_lens, "code-prompt-sensors": _prompt_lens, "scripture-chapter-context": _chapter_lens, "scripture-chapter-shape": _chapter_lens,
          "gdelt-source-history": _gdelt2_lens, "gdelt-geo-history": _gdelt2_lens, "cite-authority": lambda _n, rows, paths: _authority_lens(rows, paths),
          "cite-roster": lambda _n, rows, _p: _roster_lens(rows), "kg-neighbour-values-q": lambda _n, rows, paths: _values_lens(rows, paths)}
SOURCE_OF = {"code-prompt-signature": CODE_SOURCE, "code-prompt-sensors": CODE_SOURCE, "scripture-chapter-context": "crosswire-sword-pd", "scripture-chapter-shape": "crosswire-sword-pd",
             "gdelt-source-history": GDELT_SOURCE, "gdelt-geo-history": GDELT_SOURCE, "cite-authority": CITES_SOURCE, "cite-roster": CITES_SOURCE, "kg-neighbour-values-q": KG_SOURCE}


def encode(name: str, rows: list[dict], paths: Paths) -> Slot:
    return LENSES[name](name, rows, paths)
