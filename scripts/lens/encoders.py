"""Algorithmic lenses of the Stage 3 catalogue (spec §19.2 "Structured encoders", "Temporal clocks", "Provenance", "Code
structure", and the hashed lexical and stylometric text encoders). Every encoder is a deterministic function of the
calibration rows and this file (the lens identity carries this file's BLAKE3); a per-family fit (normalisation
statistics, vocabularies, idf) is recorded beside the slots so a reader can replay it. An atom the encoder cannot read
(no numeric field, no clock, no parse) is Absent for that lens, never a zero vector."""
from __future__ import annotations

import ast
import math
import re
from pathlib import Path

import blake3
import numpy as np
import scipy.sparse as sp

ENCODERS_B3 = blake3.blake3(Path(__file__).read_bytes()).hexdigest()
LOWCARD_MAX = 64
HASH_DIMS = 512
TFIDF_DIMS = 1 << 14
TRIGRAM_DIMS = 1 << 12
FUNCTION_WORDS = ("the", "of", "and", "to", "a", "in", "that", "is", "was", "he", "for", "it", "with", "as", "his", "on", "be", "at", "by", "i", "this", "had",
                  "not", "are", "but", "from", "or", "have", "an", "they", "which", "one", "you", "were", "her", "all", "she", "there", "would", "their",
                  "we", "him", "been", "has", "when", "who", "will", "no", "more", "if")
PUNCT = ",.;:!?\"'()-[]{}/"
AST_TYPES = sorted(n for n in dir(ast) if isinstance(getattr(ast, n), type) and issubclass(getattr(ast, n), ast.AST))
_WORD = re.compile(r"\w+", re.U)
_SENT = re.compile(r"[.!?]+")


class Slot:
    """One lens's output over the calibration rows: dense X[n,d] or sparse CSR, an absent mask, the fit, the sim kind."""

    def __init__(self, X, absent: np.ndarray, fit: dict, sim: str, sparse: bool = False):
        self.X, self.absent, self.fit, self.sim, self.sparse = X, absent.astype(bool), fit, sim, sparse

    def to_arrays(self) -> dict:
        if self.sparse:
            m = sp.csr_matrix(self.X)
            return {"data": m.data.astype(np.float32), "indices": m.indices.astype(np.int32), "indptr": m.indptr.astype(np.int64), "shape": np.asarray(m.shape, dtype=np.int64), "absent": self.absent}
        return {"X": np.ascontiguousarray(self.X, dtype=np.float32), "absent": self.absent}


def _hash_index(*parts: str, dims: int) -> tuple[int, float]:
    h = blake3.blake3("\x00".join(parts).encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") % dims, 1.0 if h[8] & 1 == 0 else -1.0


def _numeric_fields(rows: list[dict]) -> list[str]:
    names = sorted({k for r in rows for k, v in r["fields"].items() if isinstance(v, (int, float)) and not isinstance(v, bool)})
    return [k for k in names if not all(isinstance(r["fields"].get(k), str) for r in rows)]


def _categorical_fields(rows: list[dict]) -> list[str]:
    return sorted({k for r in rows for k, v in r["fields"].items() if isinstance(v, str)})


def _numeric_matrix(rows: list[dict]) -> tuple[np.ndarray, list[str], dict]:
    cols = _numeric_fields(rows)
    if not cols:
        return np.zeros((len(rows), 0), dtype=np.float64), cols, {}
    X = np.full((len(rows), len(cols)), np.nan)
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            v = r["fields"].get(c)
            if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v)):
                X[i, j] = float(v)
    med = np.nanmedian(X, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    missing = int(np.isnan(X).sum())
    X = np.where(np.isnan(X), med, X)
    return X, cols, {"fields": cols, "fill_median": med.tolist(), "missing_cells": missing}


def _scalar(name: str, rows: list[dict]) -> Slot:
    X, cols, fit = _numeric_matrix(rows)
    n = len(rows)
    if not cols:
        return Slot(np.zeros((n, 0), np.float32), np.ones(n, bool), {"reason": "no numeric field"}, "neg_cheb")
    if name == "scalar-raw":
        Y = X
    elif name == "scalar-minmax":
        lo, hi = X.min(0), X.max(0)
        Y = np.where(hi > lo, (X - lo) / np.where(hi > lo, hi - lo, 1), 0.0)
        fit.update({"min": lo.tolist(), "max": hi.tolist()})
    elif name == "scalar-z":
        mu, sd = X.mean(0), X.std(0)
        Y = np.where(sd > 0, (X - mu) / np.where(sd > 0, sd, 1), 0.0)
        fit.update({"mean": mu.tolist(), "std": sd.tolist()})
    elif name == "scalar-log1p":
        Y = np.sign(X) * np.log1p(np.abs(X))
    elif name == "scalar-rank":
        from scipy.stats import rankdata

        Y = np.column_stack([rankdata(X[:, j], method="average") / n for j in range(X.shape[1])])
    elif name == "scalar-robust":
        med = np.median(X, 0)
        q1, q3 = np.percentile(X, 25, axis=0), np.percentile(X, 75, axis=0)
        iqr = q3 - q1
        Y = np.where(iqr > 0, (X - med) / np.where(iqr > 0, iqr, 1), 0.0)
        fit.update({"median": med.tolist(), "iqr": iqr.tolist()})
    elif name == "scalar-winsor":
        lo, hi = np.percentile(X, 1, axis=0), np.percentile(X, 99, axis=0)
        C = np.clip(X, lo, hi)
        Y = np.where(hi > lo, (C - lo) / np.where(hi > lo, hi - lo, 1), 0.0)
        fit.update({"p1": lo.tolist(), "p99": hi.tolist()})
    else:
        raise KeyError(name)
    return Slot(Y.astype(np.float32), np.zeros(n, bool), fit, "neg_cheb")


def _cyclic(rows: list[dict]) -> Slot:
    n = len(rows)
    X, absent = np.zeros((n, 4), np.float32), np.ones(n, bool)
    for i, r in enumerate(rows):
        wd, doy = r["fields"].get("weekday"), r["fields"].get("day_of_year")
        if isinstance(wd, int) and isinstance(doy, int):
            X[i] = [math.sin(2 * math.pi * wd / 7), math.cos(2 * math.pi * wd / 7), math.sin(2 * math.pi * doy / 366), math.cos(2 * math.pi * doy / 366)]
            absent[i] = False
    return Slot(X, absent, {"fields": ["weekday", "day_of_year"]}, "neg_cheb")


def _knowledge_age(rows: list[dict]) -> Slot:
    n = len(rows)
    X, absent = np.zeros((n, 2), np.float32), np.ones(n, bool)
    for i, r in enumerate(rows):
        a = r["fields"].get("knowledge_age_days")
        if isinstance(a, int):
            X[i] = [math.log1p(max(a, 0)), float(np.sign(a))]
            absent[i] = False
    return Slot(X, absent, {"fields": ["knowledge_age_days"], "role": "receipt minus coordinate clock, in days"}, "neg_cheb")


def _geo(rows: list[dict]) -> Slot:
    n = len(rows)
    X, absent = np.zeros((n, 6), np.float32), np.ones(n, bool)
    for i, r in enumerate(rows):
        la, lo = r["fields"].get("ActionGeo_Lat"), r["fields"].get("ActionGeo_Long")
        if isinstance(la, float) and isinstance(lo, float):
            X[i] = [la / 90, lo / 180, math.sin(math.radians(la)), math.cos(math.radians(la)), math.sin(math.radians(lo)), math.cos(math.radians(lo))]
            absent[i] = False
    return Slot(X, absent, {"fields": ["ActionGeo_Lat", "ActionGeo_Long"]}, "neg_cheb")


def _onehot(rows: list[dict]) -> Slot:
    n = len(rows)
    vocab = {}
    for c in _categorical_fields(rows):
        vals = sorted({r["fields"][c] for r in rows if isinstance(r["fields"].get(c), str)})
        if 2 <= len(vals) <= LOWCARD_MAX:
            vocab[c] = vals
    if not vocab:
        return Slot(np.zeros((n, 0), np.float32), np.ones(n, bool), {"reason": "no low-cardinality field"}, "cosine")
    offs, d = {}, 0
    for c, vals in vocab.items():
        offs[c] = d
        d += len(vals)
    X = np.zeros((n, d), np.float32)
    for i, r in enumerate(rows):
        for c, vals in vocab.items():
            v = r["fields"].get(c)
            if isinstance(v, str) and v in vals:
                X[i, offs[c] + vals.index(v)] = 1.0
    return Slot(X, np.zeros(n, bool), {"vocab": vocab}, "cosine")


def _hash_cat(rows: list[dict], dims: int) -> tuple[np.ndarray, list[str]]:
    cols = _categorical_fields(rows)
    X = np.zeros((len(rows), dims), np.float32)
    for i, r in enumerate(rows):
        for c in cols:
            v = r["fields"].get(c)
            if isinstance(v, str) and v:
                j, s = _hash_index(c, v, dims=dims)
                X[i, j] += s
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    return np.where(nrm > 0, X / np.where(nrm > 0, nrm, 1), 0.0), cols


def _highcard(rows: list[dict]) -> Slot:
    X, cols = _hash_cat(rows, HASH_DIMS)
    if not cols:
        return Slot(X[:, :0], np.ones(len(rows), bool), {"reason": "no categorical field"}, "cosine")
    return Slot(X, np.zeros(len(rows), bool), {"fields": cols, "dims": HASH_DIMS}, "cosine")


def _record_vector(rows: list[dict]) -> Slot:
    z = _scalar("scalar-z", rows)
    oh = _onehot(rows)
    hc, cols = _hash_cat(rows, 256)
    parts = [p for p in (z.X, oh.X, hc) if p.shape[1]]
    if not parts:
        return Slot(np.zeros((len(rows), 0), np.float32), np.ones(len(rows), bool), {"reason": "no explicit field"}, "cosine")
    return Slot(np.concatenate(parts, axis=1).astype(np.float32), np.zeros(len(rows), bool), {"numeric": z.fit, "onehot": oh.fit, "hashed_fields": cols}, "cosine")


def _provenance(rows: list[dict]) -> Slot:
    sources = sorted({r["source"] for r in rows})
    n, d = len(rows), len(sources) + 1 + 32
    X = np.zeros((n, d), np.float32)
    for i, r in enumerate(rows):
        X[i, sources.index(r["source"])] = 1.0
        X[i, len(sources)] = float(bool(r.get("derived")))
        j, s = _hash_index("shard", r["shard"], dims=32)
        X[i, len(sources) + 1 + j] = s
    return Slot(X, np.zeros(n, bool), {"sources": sources, "layout": "source one-hot | derived flag | shard hash 32"}, "cosine")


def _length(rows: list[dict]) -> Slot:
    X = np.zeros((len(rows), 6), np.float32)
    for i, r in enumerate(rows):
        t = r["text"]
        c, w, s = len(t), len(_WORD.findall(t)), max(1, len(_SENT.findall(t)))
        X[i] = [c, w, s, math.log1p(c), math.log1p(w), math.log1p(s)]
    return Slot(X, np.zeros(len(rows), bool), {"layout": "chars, words, sentences, log1p of each"}, "neg_cheb")


def _stylometry(rows: list[dict]) -> Slot:
    d = 12 + len(PUNCT) + len(FUNCTION_WORDS)
    X = np.zeros((len(rows), d), np.float32)
    for i, r in enumerate(rows):
        t = r["text"]
        n = max(1, len(t))
        words = _WORD.findall(t.lower())
        nw = max(1, len(words))
        wl = np.asarray([len(w) for w in words]) if words else np.zeros(1)
        sents = [s for s in _SENT.split(t) if s.strip()]
        lines = t.split("\n")
        feats = [sum(ch.isupper() for ch in t) / n, sum(ch.isdigit() for ch in t) / n, sum(ch in PUNCT for ch in t) / n, sum(ch.isspace() for ch in t) / n,
                 sum(ord(ch) > 127 for ch in t) / n, float(wl.mean()), float(wl.std()), nw / max(1, len(sents)), len(set(words)) / nw,
                 sum(1 for w in set(words) if words.count(w) == 1) / nw if nw <= 5000 else 0.0, math.log1p(len(lines)), float(np.mean([len(x) for x in lines]))]
        X[i, :12] = feats
        for j, p in enumerate(PUNCT):
            X[i, 12 + j] = t.count(p) / n
        base = 12 + len(PUNCT)
        counts = {}
        for w in words:
            counts[w] = counts.get(w, 0) + 1
        for j, fw in enumerate(FUNCTION_WORDS):
            X[i, base + j] = counts.get(fw, 0) / nw
    return Slot(X, np.zeros(len(rows), bool), {"layout": "12 ratios | punctuation per char | 50 function words"}, "cosine")


def _tfidf(rows: list[dict]) -> Slot:
    n = len(rows)
    rows_idx, cols_idx, vals = [], [], []
    df = np.zeros(TFIDF_DIMS)
    per_doc = []
    for i, r in enumerate(rows):
        toks = _WORD.findall(r["text"].lower())
        grams = toks + [a + " " + b for a, b in zip(toks, toks[1:])]
        tf: dict[int, float] = {}
        for g in grams:
            j, s = _hash_index(g, dims=TFIDF_DIMS)
            tf[j] = tf.get(j, 0.0) + s
        per_doc.append(tf)
        for j in tf:
            df[j] += 1
    idf = np.log((1 + n) / (1 + df)) + 1
    for i, tf in enumerate(per_doc):
        for j, v in tf.items():
            rows_idx.append(i)
            cols_idx.append(j)
            vals.append(float(np.sign(v) * (1 + math.log(abs(v))) * idf[j]) if v else 0.0)
    m = sp.csr_matrix((vals, (rows_idx, cols_idx)), shape=(n, TFIDF_DIMS), dtype=np.float32)
    nrm = np.sqrt(np.asarray(m.multiply(m).sum(1)).ravel())
    m = sp.diags(np.where(nrm > 0, 1 / np.where(nrm > 0, nrm, 1), 0.0)) @ m
    return Slot(m, nrm == 0, {"dims": TFIDF_DIMS, "idf_blake3": blake3.blake3(idf.astype(np.float32).tobytes()).hexdigest()}, "cosine", sparse=True)


def _trigram(rows: list[dict]) -> Slot:
    n = len(rows)
    rows_idx, cols_idx, vals = [], [], []
    for i, r in enumerate(rows):
        t = r["text"].lower()
        tf: dict[int, float] = {}
        for k in range(len(t) - 2):
            j, s = _hash_index(t[k:k + 3], dims=TRIGRAM_DIMS)
            tf[j] = tf.get(j, 0.0) + s
        for j, v in tf.items():
            rows_idx.append(i)
            cols_idx.append(j)
            vals.append(v)
    m = sp.csr_matrix((vals, (rows_idx, cols_idx)), shape=(n, TRIGRAM_DIMS), dtype=np.float32)
    nrm = np.sqrt(np.asarray(m.multiply(m).sum(1)).ravel())
    m = sp.diags(np.where(nrm > 0, 1 / np.where(nrm > 0, nrm, 1), 0.0)) @ m
    return Slot(m, nrm == 0, {"dims": TRIGRAM_DIMS}, "cosine", sparse=True)


def _parse(text: str):
    import warnings

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            return ast.parse(text)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return None


def _ast_hist(rows: list[dict]) -> Slot:
    n, d = len(rows), len(AST_TYPES)
    idx = {t: i for i, t in enumerate(AST_TYPES)}
    X, absent = np.zeros((n, d), np.float32), np.ones(n, bool)
    for i, r in enumerate(rows):
        tree = _parse(r["text"])
        if tree is None:
            continue
        for node in ast.walk(tree):
            X[i, idx[type(node).__name__]] += 1
        tot = X[i].sum()
        if tot:
            X[i] /= tot
            absent[i] = False
    return Slot(X, absent, {"types": AST_TYPES}, "cosine")


def _code_structure(rows: list[dict]) -> Slot:
    n = len(rows)
    X, absent = np.zeros((n, 18), np.float32), np.ones(n, bool)
    for i, r in enumerate(rows):
        tree = _parse(r["text"])
        if tree is None:
            continue
        c = {k: 0 for k in ("def", "cls", "loop", "if", "call", "ret", "imp", "try", "assert", "lambda", "comp", "str", "num", "bool")}
        names, depth = set(), 0

        def walk(node, lvl):
            nonlocal depth
            depth = max(depth, lvl)
            for ch in ast.iter_child_nodes(node):
                walk(ch, lvl + 1)

        walk(tree, 0)
        rec = 0
        for node in ast.walk(tree):
            t = type(node)
            if t in (ast.FunctionDef, ast.AsyncFunctionDef):
                c["def"] += 1
                names.add(node.name)
                if any(isinstance(x, ast.Call) and isinstance(x.func, ast.Name) and x.func.id == node.name for x in ast.walk(node)):
                    rec = 1
            elif t is ast.ClassDef:
                c["cls"] += 1
            elif t in (ast.For, ast.While, ast.AsyncFor):
                c["loop"] += 1
            elif t in (ast.If, ast.IfExp):
                c["if"] += 1
            elif t is ast.Call:
                c["call"] += 1
            elif t is ast.Return:
                c["ret"] += 1
            elif t in (ast.Import, ast.ImportFrom):
                c["imp"] += 1
            elif t is ast.Try:
                c["try"] += 1
            elif t is ast.Assert:
                c["assert"] += 1
            elif t is ast.Lambda:
                c["lambda"] += 1
            elif t in (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp):
                c["comp"] += 1
            elif t is ast.Constant:
                v = node.value
                c["str" if isinstance(v, str) else "bool" if isinstance(v, bool) else "num" if isinstance(v, (int, float)) else "bool"] += 1
            elif t is ast.BoolOp:
                c["bool"] += 1
        cyclo = 1 + c["if"] + c["loop"] + c["try"] + c["bool"]
        vals = [len(r["text"].split("\n")), *c.values(), depth, rec, cyclo]
        X[i] = [math.log1p(v) if k != 16 else v for k, v in enumerate(vals)]  # 0 lines, 1-14 counts, 15 depth, 16 recursion flag, 17 cyclomatic
        absent[i] = False
    return Slot(X, absent, {"layout": "lines, def, class, loop, if, call, return, import, try, assert, lambda, comprehension, str, num, bool, depth, recursion, cyclomatic (log1p except recursion)"}, "neg_cheb")


ENCODERS = {"scalar-raw": _scalar, "scalar-minmax": _scalar, "scalar-z": _scalar, "scalar-log1p": _scalar, "scalar-rank": _scalar, "scalar-robust": _scalar, "scalar-winsor": _scalar,
            "cyclic-calendar": lambda _n, r: _cyclic(r), "clock-knowledge-age": lambda _n, r: _knowledge_age(r), "geo-latlong": lambda _n, r: _geo(r),
            "onehot-lowcard": lambda _n, r: _onehot(r), "hash-highcard": lambda _n, r: _highcard(r), "record-vector": lambda _n, r: _record_vector(r),
            "provenance-source-class": lambda _n, r: _provenance(r), "length-signature": lambda _n, r: _length(r), "text-stylometry": lambda _n, r: _stylometry(r),
            "text-hash-tfidf": lambda _n, r: _tfidf(r), "text-char-trigram": lambda _n, r: _trigram(r), "code-ast-histogram": lambda _n, r: _ast_hist(r),
            "code-structure-signature": lambda _n, r: _code_structure(r)}


def encode(name: str, rows: list[dict]) -> Slot:
    return ENCODERS[name](name, rows)
