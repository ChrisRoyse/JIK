"""Context lenses of the Stage 3 catalogue (issue #12; spec §12.2(iii) temporal clocks, Table 3 "relational aggregates",
"graph-structural signatures", "entity and knowledge", "code structure / static-analysis findings"). A context lens is a
deterministic encoder that reads, beside the atom, the source-backed context the Stage 0–2 artifacts already hold: the
GDELT day shards strictly before the coding day (declared clock: the shard's coding day; nothing from day d or later), the
Wikidata5m triple store with the atom's own P31 edge excluded (that edge is the A2 label), the Stage 2 citation edges the
opinion itself emits (in-citations are the A6 label and are never read), and the Python module's static surface. The
context for a source is built once per lens root from every calibration family that reads it and cached under
<lens_root>/ctx/ keyed by the BLAKE3 of the keys it was built for, so the determinism probe and the full measurement read
the same bytes. Each lens identity carries this file's digest; encoders.py is byte-identical, so no existing lens identity
moves. Cost: one streamed pass over the GDELT population shards (process pool, O(events)), one streamed pass over the
Wikidata5m triples (O(triples), regex on the canonical line), the citation arrays in memory (O(edges)), one ast parse per
code atom; features are O(rows × window days)."""
from __future__ import annotations

import ast
import collections
import json
import math
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

import blake3
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from encoders import Slot, _hash_index, _parse  # noqa: E402
from lenscommon import FAMILIES, Paths, iter_canon_docs, lens_root, population_shards, read_family  # noqa: E402

ENCODERS_CTX_B3 = blake3.blake3(Path(__file__).read_bytes()).hexdigest()
GDELT_SOURCE = "gdelt-events-90d"
KG_SOURCE = "wikidata5m"
CITES_SOURCE = "cuyahoga-eighth-district"
CODE_SOURCE = "opencodeinstruct"
WINDOWS = ((7, "7"), (28, "28"), (10_000, "all"))
REL_DIMS = 256
CLASS_DIMS = 512
KIND_WEBCITE, KIND_WEBCITE_FRONTIER, KIND_REPORTER_FRONTIER = 1, 2, 3  # scripts/canon/edges.py
_TRIPLE = re.compile(r'"o": "(Q\d+)", "p": "(P\d+)", "s": "(Q\d+)"')
STDLIB = set(getattr(sys, "stdlib_module_names", ())) | {"__future__"}
IO_NAMES = {"open", "input", "socket", "urlopen", "requests", "subprocess", "os.system", "exec", "eval"}
NONDET = {"random", "time", "datetime", "uuid", "secrets", "os.urandom"}


def _cal_rows_of(paths: Paths, source: str) -> list[dict]:
    rows = []
    for fam in FAMILIES:
        p = lens_root(paths) / "cal" / f"{fam}.jsonl"
        if p.exists():
            rows += [r for r in read_family(p.parent, fam)[0] if r["source"] == source]
    return rows


def _ctx_path(paths: Paths, name: str, keys: list[str]) -> Path:
    h = blake3.blake3()
    for k in keys:
        h.update(k.encode("utf-8"))
        h.update(b"\x00")
    d = lens_root(paths) / "ctx"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{name}-{h.hexdigest()[:16]}.json"


def _cached(paths: Paths, name: str, keys: list[str], build) -> dict:
    p = _ctx_path(paths, name, keys)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    ctx = build()
    tmp = p.with_name(p.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(ctx, sort_keys=True), encoding="utf-8")
    os.replace(tmp, p)
    return ctx


# ------------------------------------------------------------------ GDELT history (A3, A5)
def _gdelt_day(job: dict) -> dict:
    pairs, actors, keys = set(job["pairs"]), set(job["actors"]), set(job["keys"])
    pair_n, actor_n, key_n = collections.Counter(), collections.Counter(), collections.Counter()
    pair_quad = collections.defaultdict(lambda: [0, 0, 0, 0])
    pair_gold = collections.defaultdict(float)
    pair_dom = collections.defaultdict(set)
    for rec in iter_canon_docs(Path(job["canon_dir"]), job["shard"]):
        f = rec["f"]
        a1, a2 = f.get("Actor1Code") or "", f.get("Actor2Code") or ""
        if a1 in actors:
            actor_n[a1] += 1
        if a2 in actors:
            actor_n[a2] += 1
        if not a1 or not a2:
            continue
        p = f"{a1}|{a2}"
        if p in pairs:
            pair_n[p] += 1
            q = f.get("QuadClass") or ""
            if q in ("1", "2", "3", "4"):
                pair_quad[p][int(q) - 1] += 1
            try:
                pair_gold[p] += float(f.get("GoldsteinScale") or 0.0)
            except ValueError:
                pass
            pair_dom[p].add((urlparse(f.get("SOURCEURL") or "").netloc or "").lower())
        k = f"{p}|{f.get('EventCode') or ''}"
        if k in keys:
            key_n[k] += 1
    return {"day": job["day"], "shard": job["shard"], "pair_n": dict(pair_n), "actor_n": dict(actor_n), "key_n": dict(key_n),
            "pair_quad": dict(pair_quad), "pair_gold": dict(pair_gold), "pair_dom": {k: sorted(v) for k, v in pair_dom.items()}}


def _gdelt_ctx(paths: Paths) -> dict:
    rows = _cal_rows_of(paths, GDELT_SOURCE)
    pairs = sorted({f"{r['fields']['Actor1Code']}|{r['fields']['Actor2Code']}" for r in rows})
    actors = sorted({r["fields"]["Actor1Code"] for r in rows} | {r["fields"]["Actor2Code"] for r in rows})
    keys = sorted({f"{r['fields']['Actor1Code']}|{r['fields']['Actor2Code']}|{r['fields'].get('EventCode') or ''}" for r in rows})
    shards = sorted(s["shard"] for s in population_shards(paths, paths.version) if s["source"] == GDELT_SOURCE)

    def build() -> dict:
        jobs = [{"shard": sh, "canon_dir": str(paths.canon(GDELT_SOURCE)), "day": d, "pairs": pairs, "actors": actors, "keys": keys} for d, sh in enumerate(shards)]
        with ProcessPoolExecutor(max_workers=min(28, os.cpu_count() or 1)) as ex:
            days = sorted(ex.map(_gdelt_day, jobs), key=lambda r: r["day"])
        return {"source": GDELT_SOURCE, "clock": "coding day of the shard; features read days strictly before the atom's day_index", "shards": shards, "days": days}

    return _cached(paths, "gdelt", ["shards", *shards, "pairs", *pairs, "actors", *actors, "keys", *keys], build)


def _gdelt_features(rows: list[dict], ctx: dict) -> tuple[np.ndarray, np.ndarray, list[str]]:
    days = ctx["days"]
    names = []
    for _, tag in WINDOWS:
        names += [f"pair_{tag}", f"pair_days_{tag}", f"pair_domains_{tag}", f"a1_{tag}", f"a2_{tag}", f"key_{tag}"]
    names += ["pair_sameday_other", "a1_sameday_other", "a2_sameday_other", "quad1_28", "quad2_28", "quad3_28", "quad4_28", "goldstein_mean_28"]
    X, absent = np.zeros((len(rows), len(names)), np.float64), np.ones(len(rows), bool)
    for i, r in enumerate(rows):
        f = r["fields"]
        if r["source"] != GDELT_SOURCE or not f.get("Actor1Code") or not f.get("Actor2Code") or f.get("day_index") is None:
            continue
        d = int(f["day_index"])
        if d >= len(days) or days[d]["shard"] != r["shard"]:
            continue
        a1, a2 = f["Actor1Code"], f["Actor2Code"]
        p, k = f"{a1}|{a2}", f"{a1}|{a2}|{f.get('EventCode') or ''}"
        col = 0
        for W, _ in WINDOWS:
            pn = pd = an1 = an2 = kn = 0
            doms: set[str] = set()
            for dd in range(max(0, d - W), d):
                day = days[dd]
                c = day["pair_n"].get(p, 0)
                pn, pd = pn + c, pd + (c > 0)
                an1, an2, kn = an1 + day["actor_n"].get(a1, 0), an2 + day["actor_n"].get(a2, 0), kn + day["key_n"].get(k, 0)
                doms.update(day["pair_dom"].get(p, ()))
            X[i, col:col + 6] = [math.log1p(pn), pd, math.log1p(len(doms)), math.log1p(an1), math.log1p(an2), math.log1p(kn)]
            col += 6
        same = days[d]
        X[i, col:col + 3] = [math.log1p(max(0, same["pair_n"].get(p, 0) - 1)), math.log1p(max(0, same["actor_n"].get(a1, 0) - 1)), math.log1p(max(0, same["actor_n"].get(a2, 0) - 1))]
        q, g, pn = np.zeros(4), 0.0, 0
        for dd in range(max(0, d - 28), d):
            day = days[dd]
            if p in day["pair_quad"]:
                q += day["pair_quad"][p]
                g += day["pair_gold"].get(p, 0.0)
                pn += day["pair_n"].get(p, 0)
        X[i, col + 3:col + 7] = q / max(1, q.sum())
        X[i, col + 7] = g / max(1, pn)
        absent[i] = False
    return X, absent, names


GDELT_LENSES = {  # lens -> the feature columns it reports (complementary views of the same history)
    "gdelt-pair-history": ["pair_7", "pair_days_7", "pair_domains_7", "pair_28", "pair_days_28", "pair_domains_28", "pair_all", "pair_days_all", "pair_domains_all", "pair_sameday_other"],
    "gdelt-actor-activity": ["a1_7", "a2_7", "a1_28", "a2_28", "a1_all", "a2_all", "a1_sameday_other", "a2_sameday_other"],
    "gdelt-event-recurrence": ["key_7", "key_28", "key_all"],
    "gdelt-pair-tone-mix": ["quad1_28", "quad2_28", "quad3_28", "quad4_28", "goldstein_mean_28"],
}


def _gdelt_lens(name: str, rows: list[dict], paths: Paths) -> Slot:
    if not any(r["source"] == GDELT_SOURCE for r in rows):
        return Slot(np.zeros((len(rows), len(GDELT_LENSES[name])), np.float32), np.ones(len(rows), bool), {"absent": "no GDELT atom in this family"}, "neg_cheb")
    X, absent, names = _gdelt_features(rows, _gdelt_ctx(paths))
    cols = [names.index(c) for c in GDELT_LENSES[name]]
    return Slot(X[:, cols].astype(np.float32), absent, {"layout": GDELT_LENSES[name], "clock": "days strictly before day_index (same-day columns exclude the atom itself)", "log1p": "counts"}, "neg_cheb")


# ------------------------------------------------------------------ Wikidata5m graph (A2)
def _kg_ctx(paths: Paths) -> dict:
    ents = sorted({r["cluster"] for r in _cal_rows_of(paths, KG_SOURCE)})
    shards = ["wikidata5m_transductive_train", "wikidata5m_transductive_valid", "wikidata5m_transductive_test"]

    def build() -> dict:
        import io

        import zstandard

        want = set(ents)
        out_rel, in_rel = collections.defaultdict(collections.Counter), collections.defaultdict(collections.Counter)
        out_nb, in_nb = collections.defaultdict(list), collections.defaultdict(list)
        p31: dict[str, str] = {}
        triples = 0
        for sh in shards:
            with (paths.canon(KG_SOURCE) / f"{sh}.jsonl.zst").open("rb") as fh, zstandard.ZstdDecompressor().stream_reader(fh) as zr:
                for line in io.TextIOWrapper(zr, encoding="utf-8"):
                    m = _TRIPLE.search(line)
                    if not m:
                        continue
                    o, p, s = m.groups()
                    triples += 1
                    if p == "P31":
                        p31.setdefault(s, o)
                        if s in want:
                            continue  # the atom's own class statement is the A2 label and is never read
                    if s in want:
                        out_rel[s][p] += 1
                        out_nb[s].append(o)
                    if o in want:
                        in_rel[o][p] += 1
                        in_nb[o].append(s)
        per = {}
        for e in ents:
            nb_classes = collections.Counter(p31[x] for x in out_nb[e] + in_nb[e] if x in p31)
            per[e] = {"out_rel": dict(out_rel[e]), "in_rel": dict(in_rel[e]), "out_deg": len(out_nb[e]), "in_deg": len(in_nb[e]), "nb_classes": dict(nb_classes)}
        return {"source": KG_SOURCE, "shards": shards, "triples": triples, "entities_with_p31": len(p31), "entities": per}

    return _cached(paths, "kg", ["shards", *shards, "entities", *ents], build)


def _kg_lens(name: str, rows: list[dict], paths: Paths) -> Slot:
    dims = 2 * REL_DIMS + 4 if name == "kg-relation-signature" else CLASS_DIMS
    X, absent = np.zeros((len(rows), dims), np.float32), np.ones(len(rows), bool)
    if not any(r["source"] == KG_SOURCE for r in rows):
        return Slot(X, absent, {"absent": "no Wikidata5m atom in this family"}, "cosine")
    ents = _kg_ctx(paths)["entities"]
    for i, r in enumerate(rows):
        e = ents.get(r["cluster"]) if r["source"] == KG_SOURCE else None
        if not e or (e["out_deg"] + e["in_deg"]) == 0:
            continue
        if name == "kg-relation-signature":
            for p, c in e["out_rel"].items():
                j, _ = _hash_index("out", p, dims=REL_DIMS)
                X[i, j] += c
            for p, c in e["in_rel"].items():
                j, _ = _hash_index("in", p, dims=REL_DIMS)
                X[i, REL_DIMS + j] += c
            X[i, :2 * REL_DIMS] = np.log1p(X[i, :2 * REL_DIMS])
            X[i, 2 * REL_DIMS:] = [math.log1p(e["out_deg"]), math.log1p(e["in_deg"]), math.log1p(len(e["out_rel"])), math.log1p(len(e["in_rel"]))]
        else:
            for c, n in e["nb_classes"].items():
                j, _ = _hash_index("class", c, dims=CLASS_DIMS)
                X[i, j] += n
            if X[i].sum() == 0:
                continue
            X[i] /= X[i].sum()
        absent[i] = False
    fit = {"hash_dims": REL_DIMS if name == "kg-relation-signature" else CLASS_DIMS, "excluded": "the entity's own P31 statements",
           "layout": "log1p out-relation histogram | log1p in-relation histogram | log1p out/in degree, distinct out/in relations" if name == "kg-relation-signature" else "P31 class histogram of out- and in-neighbours, normalised"}
    return Slot(X, absent, fit, "cosine")


# ------------------------------------------------------------------ citation out-edges (A6)
def _cites_ctx(paths: Paths) -> dict:
    e = np.load(paths.out / "atoms" / paths.version / "edges" / "cites.npz")
    per: dict[str, list] = {}
    for sh, rw, kind, ts, tr, off in zip(e["shard"].tolist(), e["row"].tolist(), e["kind"].tolist(), e["target_shard"].tolist(), e["target_row"].tolist(), e["offset"].tolist()):
        per.setdefault(f"{sh}:{rw}", []).append((kind, ts, tr, off))
    return per


def _cites_lens(rows: list[dict], paths: Paths) -> Slot:
    names = ["resolved", "frontier_reporter", "frontier_webcite", "distinct_resolved_targets", "self_target", "cites_per_kchar", "first_cite_frac", "resolved_share"]
    X, absent = np.zeros((len(rows), len(names)), np.float32), np.ones(len(rows), bool)
    if not any(r["source"] == CITES_SOURCE for r in rows):
        return Slot(X, absent, {"absent": "no Cuyahoga atom in this family"}, "neg_cheb")
    per = _cites_ctx(paths)
    shard_index = {s["shard"]: i for i, s in enumerate(population_shards(paths, paths.version)) if s["source"] == CITES_SOURCE}
    for i, r in enumerate(rows):
        if r["source"] != CITES_SOURCE:
            continue
        si = shard_index.get(r["shard"])
        edges = per.get(f"{si}:{r['row']}", [])
        chars = max(1, int(r["fields"].get("text_chars") or len(r["text"])))
        res = [(ts, tr, off) for k, ts, tr, off in edges if k == KIND_WEBCITE]
        rep = sum(1 for k, *_ in edges if k == KIND_REPORTER_FRONTIER)
        web = sum(1 for k, *_ in edges if k == KIND_WEBCITE_FRONTIER)
        targets = {(ts, tr) for ts, tr, _ in res}
        selfs = sum(1 for ts, tr, _ in res if ts == si and tr == r["row"])
        first = min((off for *_, off in edges), default=chars) / chars
        X[i] = [math.log1p(len(res)), math.log1p(rep), math.log1p(web), math.log1p(len(targets)), math.log1p(selfs), 1000 * len(edges) / chars, first, len(res) / max(1, len(edges))]
        absent[i] = False
    return Slot(X, absent, {"layout": names, "read": "edges the opinion emits (out-citations); in-citations are the A6 label and are not read"}, "neg_cheb")


# ------------------------------------------------------------------ static code sensors (A1)
def _dotted(node) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_dotted(node.value)}.{node.attr}"
    return ""


def _code_sensors(rows: list[dict]) -> Slot:
    names = ["parses", "lines", "imports", "nonstdlib_imports", "toplevel_statements", "toplevel_calls", "toplevel_prints", "main_guard", "input_calls", "io_calls",
             "nondeterminism", "while_true", "raise", "not_implemented", "try", "returns", "annotations", "docstrings", "globals", "exit_calls", "max_depth", "string_literals"]
    X, absent = np.zeros((len(rows), len(names)), np.float32), np.ones(len(rows), bool)
    for i, r in enumerate(rows):
        text = r["text"]
        if not text.strip():
            continue
        tree = _parse(text)
        c = collections.Counter()
        depth = 0
        if tree is not None:
            for node in tree.body:
                c["toplevel_statements"] += not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom))
                if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                    c["toplevel_calls"] += 1
                    c["toplevel_prints"] += _dotted(node.value.func) == "print"
                if isinstance(node, ast.If) and "__name__" in ast.dump(node.test):
                    c["main_guard"] = 1

            def walk(node, lvl):
                nonlocal depth
                depth = max(depth, lvl)
                for ch in ast.iter_child_nodes(node):
                    walk(ch, lvl + 1)

            walk(tree, 0)
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    c["imports"] += 1
                    mods = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
                    c["nonstdlib_imports"] += sum(1 for m in mods if m.split(".")[0] not in STDLIB)
                    c["nondeterminism"] += sum(1 for m in mods if m.split(".")[0] in NONDET)
                elif isinstance(node, ast.Call):
                    nm = _dotted(node.func)
                    c["input_calls"] += nm == "input"
                    c["io_calls"] += nm in IO_NAMES or nm.split(".")[-1] in ("open", "urlopen")
                    c["exit_calls"] += nm in ("exit", "sys.exit", "quit")
                    c["nondeterminism"] += nm.split(".")[0] in NONDET
                elif isinstance(node, ast.While):
                    c["while_true"] += isinstance(node.test, ast.Constant) and bool(node.test.value)
                elif isinstance(node, ast.Raise):
                    c["raise"] += 1
                    c["not_implemented"] += "NotImplemented" in ast.dump(node)
                elif isinstance(node, ast.Try):
                    c["try"] += 1
                elif isinstance(node, ast.Return):
                    c["returns"] += 1
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    c["annotations"] += sum(1 for a in node.args.args if a.annotation is not None) + (node.returns is not None)
                    c["docstrings"] += ast.get_docstring(node) is not None
                elif isinstance(node, ast.Global):
                    c["globals"] += 1
                elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                    c["string_literals"] += 1
        vals = [1.0 if tree is not None else 0.0, math.log1p(text.count("\n") + 1)] + [math.log1p(c[k]) for k in names[2:7]] + [float(c["main_guard"])] + \
               [math.log1p(c[k]) for k in names[8:20]] + [math.log1p(depth), math.log1p(c["string_literals"])]
        X[i] = vals
        absent[i] = False
    return Slot(X, absent, {"layout": names, "stdlib": "sys.stdlib_module_names of the measuring interpreter", "log1p": "counts"}, "neg_cheb")


LENSES = {**{n: _gdelt_lens for n in GDELT_LENSES}, "kg-relation-signature": _kg_lens, "kg-neighbour-classes": _kg_lens,
          "cite-out-signature": lambda _n, rows, paths: _cites_lens(rows, paths), "code-static-sensors": lambda _n, rows, _p: _code_sensors(rows)}
SOURCE_OF = {**{n: GDELT_SOURCE for n in GDELT_LENSES}, "kg-relation-signature": KG_SOURCE, "kg-neighbour-classes": KG_SOURCE, "cite-out-signature": CITES_SOURCE, "code-static-sensors": CODE_SOURCE}


def encode(name: str, rows: list[dict], paths: Paths) -> Slot:
    return LENSES[name](name, rows, paths)
