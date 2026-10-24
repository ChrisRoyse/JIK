"""Third commissioning round of context lenses (issue #12, Johari window 2, solution 2). Each reads an input no registered
lens reads: the A2 record's abstract outside the atom span (the row text is the first-sentence atom; the rest states the
occupations, places and countries the P106/P17/P27/... questions ask about); the A1 record's unit tests as text and the
solution's own execution trace in the bubblewrap sandbox with no tests run (spec Table 3 "execution traces"; the tests'
outcome is the label and is never computed here); the A6 opinion's typed disposition and party structure (Poly #1375's
lesson: typed indicators carried bits where embedders converged). The graph pair and the neighbour-values lens are
re-issued for the six third-round A2 questions with the asked property excluded in either direction (encoders_kg.py and
encoders_ctx2.py are frozen with their own family maps, so the same functions are registered under new names and
restricted to the new questions; a lens family counts once in a union because D1 fails closed across disjoint questions).
Contexts are cached under <lens_root>/ctx/; no earlier encoder module changes. Cost: one streamed pass per shard holding
a calibration row (abstracts, tests), one sandboxed child per A1 row (thread pool), one regex pass per opinion, the
kgq/kgv-style caches over the triples (O(triples))."""
from __future__ import annotations

import collections
import io
import json
import math
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import blake3
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from encoders import Slot, _hash_index  # noqa: E402
from encoders_ctx import CITES_SOURCE, CODE_SOURCE, KG_SOURCE, REL_DIMS, CLASS_DIMS, _cached, _cal_rows_of  # noqa: E402
from encoders_ctx2 import VALUE_DIMS, WORD_DIMS, _absent, _sparse_rows, _word_hist  # noqa: E402
from encoders_kg import _TRIPLE, SHARDS, _kg_ctx_by_property  # noqa: E402
from lenscommon import Paths, fetch_records  # noqa: E402
from sandbox import CPU_SECS, MEM_KB, WALL_SECS  # noqa: E402

ENCODERS_CTX3_B3 = blake3.blake3(Path(__file__).read_bytes()).hexdigest()
TRACE_HARNESS = Path(__file__).resolve().parent / "oci_trace.py"
LABEL2 = {"a2-wikidata-p27": "P27", "a2-wikidata-p19": "P19", "a2-wikidata-p69": "P69", "a2-wikidata-p641": "P641", "a2-wikidata-p136": "P136", "a2-wikidata-p495": "P495"}
A2_FAMILIES = {"a2-wikidata-p31", "a2-wikidata-p31-rare", "a2-wikidata-p17", "a2-wikidata-p106", *LABEL2}
EXC_DIMS = 16


# ------------------------------------------------------------------ A2: the abstract outside the atom span
def _abstract_ctx(paths: Paths) -> dict:
    rows = [r for r in _cal_rows_of(paths, KG_SOURCE) if r.get("family") in A2_FAMILIES]
    want = {(r["shard"], int(r["row"])): (int(r["span"][0]), int(r["span"][1])) for r in rows}
    keys = sorted(f"{s}:{r}:{a}:{b}" for (s, r), (a, b) in want.items())

    def build() -> dict:
        by_shard: dict[str, set[int]] = collections.defaultdict(set)
        for s, r in want:
            by_shard[s].add(r)
        per = {}
        for s, rws in sorted(by_shard.items()):
            for r, rec in fetch_records(paths.canon(KG_SOURCE), s, rws).items():
                a, b = want[(s, r)]
                per[f"{s}:{r}"] = (rec["text"][:a] + " " + rec["text"][b:]).strip()  # everything the atom span does not cover
        return {"source": KG_SOURCE, "read": "record abstract outside the row's atom span", "rest": per}

    return _cached(paths, "abstract", ["rest", *keys], build)


def _abstract_lens(rows: list[dict], paths: Paths) -> Slot:
    if not any(r["source"] == KG_SOURCE and r.get("family") in A2_FAMILIES for r in rows):
        return _absent(rows, WORD_DIMS + 1, "cosine", "no A2 question in this family", sparse=True)
    rest = _abstract_ctx(paths)["rest"]
    counts = []
    for r in rows:
        key = f"{r['shard']}:{r['row']}"
        if r["source"] != KG_SOURCE or key not in rest:
            counts.append({})
            continue
        tf = _word_hist(rest[key], WORD_DIMS)
        tf[WORD_DIMS] = 1.0 if rest[key] else -1.0  # the record was read: an abstract with no sentence beyond the atom is a reading, not an absence
        counts.append(tf)
    m, empty = _sparse_rows(counts, WORD_DIMS + 1)
    return Slot(m, empty, {"dims": WORD_DIMS, "features": "signed hashed word unigrams and bigrams of the abstract outside the atom span, L2-normalised; last dim +1 when a rest exists, -1 when the abstract is the atom alone"}, "cosine", sparse=True)


# ------------------------------------------------------------------ A1: the unit tests as text; the solution's execution trace
def _record_ctx(paths: Paths) -> dict:
    rows = _cal_rows_of(paths, CODE_SOURCE)
    want = {(r["shard"], int(r["row"])): r["text_b3"] for r in rows}
    keys = sorted(f"{s}:{r}:{b}" for (s, r), b in want.items())

    def build() -> dict:
        by_shard: dict[str, set[int]] = collections.defaultdict(set)
        for s, r in want:
            by_shard[s].add(r)
        tests = {}
        for s, rws in sorted(by_shard.items()):
            for r, rec in fetch_records(paths.canon(CODE_SOURCE), s, rws).items():
                try:
                    t = json.loads(rec["f"].get("unit_tests") or "[]")
                except json.JSONDecodeError:
                    t = []
                tests[f"{s}:{r}"] = [x for x in t if isinstance(x, str)]
        return {"source": CODE_SOURCE, "read": "the unit_tests field of the record (text only; never executed here)", "tests": tests}

    return _cached(paths, "tests", ["tests", *keys], build)


def _tests_lens(rows: list[dict], paths: Paths) -> Slot:
    if not any(r["source"] == CODE_SOURCE for r in rows):
        return _absent(rows, WORD_DIMS, "cosine", "no OpenCodeInstruct atom in this family", sparse=True)
    tests = _record_ctx(paths)["tests"]
    counts = [_word_hist("\n".join(tests.get(f"{r['shard']}:{r['row']}", [])), WORD_DIMS) if r["source"] == CODE_SOURCE else {} for r in rows]
    m, empty = _sparse_rows(counts, WORD_DIMS)
    return Slot(m, empty, {"dims": WORD_DIMS, "features": "signed hashed word unigrams and bigrams of the unit tests, L2-normalised", "read": "tests as text"}, "cosine", sparse=True)


def _trace_cmd() -> list[str]:
    binds = []
    for d in ("/usr", "/lib", "/lib64", "/bin", "/etc/alternatives"):
        if Path(d).exists():
            binds += ["--ro-bind", d, d]
    return ["timeout", "-s", "KILL", str(WALL_SECS), "bwrap", *binds, "--ro-bind", str(TRACE_HARNESS), "/sandbox/trace.py", "--tmpfs", "/tmp",
            "--unshare-all", "--die-with-parent", "--new-session", "--", "/bin/sh", "-c", f"ulimit -v {MEM_KB} -t {CPU_SECS}; exec /usr/bin/python3 -I /sandbox/trace.py"]


def _trace_one(code: str) -> dict:
    try:
        p = subprocess.run(_trace_cmd(), input=json.dumps({"code": code}).encode("utf-8"), capture_output=True, timeout=WALL_SECS + 5)
    except subprocess.TimeoutExpired:
        return {"ran": False, "exception": "WallClock", "timeout": True}
    if p.returncode in (137, -9):
        return {"ran": False, "exception": "WallClock", "timeout": True, "rc": p.returncode}
    lines = p.stdout.decode("utf-8", "replace").strip().splitlines()
    try:
        out = json.loads(lines[-1]) if lines else {}
    except json.JSONDecodeError:
        out = {}
    if "ran" not in out:
        return {"ran": False, "exception": "HarnessNoOutput", "timeout": False, "rc": p.returncode}
    out.update({"timeout": False, "rc": p.returncode})
    return out


def _exec_ctx(paths: Paths) -> dict:
    rows = _cal_rows_of(paths, CODE_SOURCE)
    by_hash = {r["text_b3"]: r["text"] for r in rows}
    keys = sorted(by_hash)

    def build() -> dict:
        import shutil

        if shutil.which("bwrap") is None:
            raise RuntimeError("MNLM_SANDBOX_UNAVAILABLE: bwrap not on PATH")
        probe = _trace_one("x = 1\nprint(x)\n")
        if not probe.get("ran") or probe.get("stdout_chars") != 2:
            raise RuntimeError(f"MNLM_SANDBOX_UNAVAILABLE: trace probe {probe}")
        with ThreadPoolExecutor(max_workers=16) as ex:
            traces = list(ex.map(_trace_one, [by_hash[k] for k in keys]))
        return {"source": CODE_SOURCE, "harness_blake3": blake3.blake3(TRACE_HARNESS.read_bytes()).hexdigest(), "probe": probe, "read": "the solution executed alone; no test is run", "traces": dict(zip(keys, traces))}

    return _cached(paths, "exec", ["traces", *keys], build)


EXEC_LAYOUT = ["ran", "timeout", "stdout_chars", "stderr_chars", "wall_ms", "functions", "classes", "stdin_read", "names"] + [f"exc_{i}" for i in range(EXC_DIMS)]


def _exec_lens(rows: list[dict], paths: Paths) -> Slot:
    if not any(r["source"] == CODE_SOURCE for r in rows):
        return _absent(rows, len(EXEC_LAYOUT), "neg_cheb", "no OpenCodeInstruct atom in this family")
    traces = _exec_ctx(paths)["traces"]
    X, absent = np.zeros((len(rows), len(EXEC_LAYOUT)), np.float32), np.ones(len(rows), bool)
    for i, r in enumerate(rows):
        t = traces.get(r["text_b3"]) if r["source"] == CODE_SOURCE else None
        if not t:
            continue
        X[i, :9] = [float(t.get("ran", False)), float(t.get("timeout", False)), math.log1p(t.get("stdout_chars", 0)), math.log1p(t.get("stderr_chars", 0)), math.log1p(t.get("wall_ms", 0.0)),
                    math.log1p(t.get("functions", 0)), math.log1p(t.get("classes", 0)), float(t.get("stdin_read", False)), math.log1p(t.get("names", 0))]
        if t.get("exception"):
            j, _ = _hash_index("exc", str(t["exception"]), dims=EXC_DIMS)
            X[i, 9 + j] = 1.0
        absent[i] = False
    return Slot(X, absent, {"layout": EXEC_LAYOUT, "log1p": "sizes and counts", "read": "execution trace of the solution alone (sandboxed); the tests are never run"}, "neg_cheb")


# ------------------------------------------------------------------ A6: typed disposition and party structure
DISPOSITION = {"affirmed": r"\baffirm", "reversed": r"\brevers", "remanded": r"\bremand", "dismissed": r"\bdismiss", "vacated": r"\bvacat", "modified": r"\bmodif",
               "assignments_of_error": r"assignments? of error", "appellant": r"\bappellant", "appellee": r"\bappellee", "plaintiff": r"\bplaintiff", "defendant": r"\bdefendant",
               "pro_se": r"\bpro se\b", "sentence": r"\bsentenc", "en_banc": r"\ben banc\b", "concur": r"\bconcur", "dissent": r"\bdissent", "motion": r"\bmotion\b", "summary_judgment": r"summary judgment",
               "guilty": r"\bguilty\b", "custody": r"\bcustody\b", "contract": r"\bcontract", "negligen": r"\bnegligen", "constitution": r"\bconstitution"}
_DISP_RE = {k: re.compile(v, re.I) for k, v in DISPOSITION.items()}
DISP_LAYOUT = list(DISPOSITION) + ["last_affirmed", "last_reversed", "judges_named", "opinion_kchars"]
_JUDGES = re.compile(r"^Judges?:\s*(.+)$", re.M)


def _disposition_lens(rows: list[dict]) -> Slot:
    if not any(r["source"] == CITES_SOURCE for r in rows):
        return _absent(rows, len(DISP_LAYOUT), "neg_cheb", "no Cuyahoga atom in this family")
    X, absent = np.zeros((len(rows), len(DISP_LAYOUT)), np.float32), np.ones(len(rows), bool)
    for i, r in enumerate(rows):
        if r["source"] != CITES_SOURCE:
            continue
        t = r["text"]
        X[i, :len(DISPOSITION)] = [math.log1p(len(_DISP_RE[k].findall(t))) for k in DISPOSITION]
        tail = t[-1500:].lower()
        X[i, len(DISPOSITION)] = float("affirm" in tail)
        X[i, len(DISPOSITION) + 1] = float("revers" in tail)
        m = _JUDGES.search(t[:600])
        X[i, len(DISPOSITION) + 2] = math.log1p(len(re.split(r",|;| and |&", m.group(1))) if m else 0)
        X[i, len(DISPOSITION) + 3] = math.log1p(len(t) / 1000)
        absent[i] = False
    return Slot(X, absent, {"layout": DISP_LAYOUT, "log1p": "counts", "read": "regex counts over the opinion text; the mandate tail; the Judges header"}, "neg_cheb")


# ------------------------------------------------------------------ A2 third-round questions: the graph pair and neighbour values
def _kg_q2_lens(name: str, rows: list[dict], paths: Paths) -> Slot:
    relation = name == "kg-relation-signature-q2"
    dims = 2 * REL_DIMS + 4 if relation else CLASS_DIMS
    if not any(r["source"] == KG_SOURCE and r.get("family") in LABEL2 for r in rows):
        return _absent(rows, dims, "cosine", "no third-round A2 question in this family")
    ents = _kg_ctx_by_property(paths)["entities"]
    X, absent = np.zeros((len(rows), dims), np.float32), np.ones(len(rows), bool)
    for i, r in enumerate(rows):
        label = LABEL2.get(r.get("family", ""))
        e = ents.get(r["cluster"]) if (r["source"] == KG_SOURCE and label) else None
        if not e:
            continue
        out = {p: v for p, v in e["out"].items() if p != label}
        inn = {p: v for p, v in e["in"].items() if p != label}
        out_deg, in_deg = sum(v["n"] for v in out.values()), sum(v["n"] for v in inn.values())
        if out_deg + in_deg == 0:
            continue
        if relation:
            for p, v in out.items():
                j, _ = _hash_index("out", p, dims=REL_DIMS)
                X[i, j] += v["n"]
            for p, v in inn.items():
                j, _ = _hash_index("in", p, dims=REL_DIMS)
                X[i, REL_DIMS + j] += v["n"]
            X[i, :2 * REL_DIMS] = np.log1p(X[i, :2 * REL_DIMS])
            X[i, 2 * REL_DIMS:] = [math.log1p(out_deg), math.log1p(in_deg), math.log1p(len(out)), math.log1p(len(inn))]
        else:
            for v in list(out.values()) + list(inn.values()):
                for c, n in v["classes"].items():
                    j, _ = _hash_index("class", c, dims=CLASS_DIMS)
                    X[i, j] += n
            if X[i].sum() == 0:
                continue
            X[i] /= X[i].sum()
        absent[i] = False
    return Slot(X, absent, {"hash_dims": REL_DIMS if relation else CLASS_DIMS, "excluded": "edges of the asked property, in either direction"}, "cosine")


def _kgv2_ctx(paths: Paths) -> dict:
    ents = sorted({r["cluster"] for r in _cal_rows_of(paths, KG_SOURCE) if r.get("family") in LABEL2})
    props = sorted(set(LABEL2.values()))

    def build() -> dict:
        import zstandard

        want = set(ents)
        value: dict[str, dict[str, str]] = {p: {} for p in props}
        edges: dict[str, list] = collections.defaultdict(list)
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
        per = {e: {p: dict(collections.Counter(value[p][nb] for q, nb in edges[e] if q != p and nb in value[p])) for p in props} for e in ents}
        return {"source": KG_SOURCE, "shards": SHARDS, "triples": triples, "properties": props, "excluded": "edges of the asked property in either direction", "entities": per}

    return _cached(paths, "kgv2", ["shards", *SHARDS, "properties", *props, "entities", *ents], build)


def _values2_lens(rows: list[dict], paths: Paths) -> Slot:
    if not any(r["source"] == KG_SOURCE and r.get("family") in LABEL2 for r in rows):
        return _absent(rows, VALUE_DIMS, "cosine", "no third-round A2 question in this family")
    ents = _kgv2_ctx(paths)["entities"]
    X, absent = np.zeros((len(rows), VALUE_DIMS), np.float32), np.ones(len(rows), bool)
    for i, r in enumerate(rows):
        label = LABEL2.get(r.get("family", ""))
        hist = (ents.get(r["cluster"]) or {}).get(label) if (r["source"] == KG_SOURCE and label) else None
        if not hist:
            continue
        for v, n in hist.items():
            j, _ = _hash_index("value", label, v, dims=VALUE_DIMS)
            X[i, j] += n
        X[i] /= X[i].sum()
        absent[i] = False
    return Slot(X, absent, {"hash_dims": VALUE_DIMS, "layout": "normalised histogram of the asked property's values over the neighbours", "excluded": "the entity's own statements of the asked property, in either direction"}, "cosine")


LENSES = {"a2-abstract-rest": lambda _n, rows, paths: _abstract_lens(rows, paths), "code-tests-signature": lambda _n, rows, paths: _tests_lens(rows, paths),
          "code-exec-trace": lambda _n, rows, paths: _exec_lens(rows, paths), "opinion-disposition-sensors": lambda _n, rows, _p: _disposition_lens(rows),
          "kg-relation-signature-q2": _kg_q2_lens, "kg-neighbour-classes-q2": _kg_q2_lens, "kg-neighbour-values-q2": lambda _n, rows, paths: _values2_lens(rows, paths)}
SOURCE_OF = {"a2-abstract-rest": KG_SOURCE, "code-tests-signature": CODE_SOURCE, "code-exec-trace": CODE_SOURCE, "opinion-disposition-sensors": CITES_SOURCE,
             "kg-relation-signature-q2": KG_SOURCE, "kg-neighbour-classes-q2": KG_SOURCE, "kg-neighbour-values-q2": KG_SOURCE}


def encode(name: str, rows: list[dict], paths: Paths) -> Slot:
    return LENSES[name](name, rows, paths)
