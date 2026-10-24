"""Explicit (encoded) senses over pre-evaluation SWE observations (#12 senses plan).

Each function maps one observation to a fixed-width float vector for exactly one perspective:
  structure   S2  AST delta of the touched Python files (base vs patched), parsed only, never run
  behaviour   S4a what the agent did and saw before submitting (trajectory turns)
  relations   S5a identifiers and paths shared by the issue text and the patch
  diagnostic  S6  static findings introduced by the patch (syntax, pyflakes), parsed only
  provenance  S7  agent model and termination status
Vectors are never concatenated across senses; each sense is stored and measured as its own block.
"""
from __future__ import annotations

import ast
import io
import re
import tempfile
from pathlib import Path

import numpy as np

VERSION = "swe-explicit-senses-v1"
AST_TYPES = sorted(n for n in dir(ast) if isinstance(getattr(ast, n), type) and issubclass(getattr(ast, n), ast.AST)
                   and getattr(ast, n) not in (ast.AST,) and not n.startswith("_"))
COMMANDS = ["open", "goto", "scroll_down", "scroll_up", "create", "edit", "search_dir", "search_file", "find_file",
            "python", "pytest", "ls", "cat", "grep", "find", "submit", "rm", "pip", "git"]
OBS_SIGNS = {"traceback": r"Traceback \(most recent call last\)", "error": r"\b\w*Error\b", "syntax": r"SyntaxError",
             "passed": r"\bpassed\b", "failed": r"\bfailed\b", "ok": r"^OK$|\bOK\b", "no_output": r"did not produce any output",
             "edit_ok": r"File updated", "edit_err": r"Your proposed edit has introduced new syntax error"}
MODELS = ["swe-agent-llama-8b", "swe-agent-llama-70b", "swe-agent-llama-405b"]
EXITS = ["submitted", "submitted (exit_context)", "submitted (exit_cost)", "submitted (exit_format)",
         "submitted (exit_error)", "early_exit", "exit_context", "exit_cost", "exit_format", "exit_error"]
IDENT = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\b")


def hunks(patch):
    """[(path, old_start, old_len, new_start, new_len)] and per-file added/removed line texts."""
    out, lines, cur = [], {}, None
    for line in patch.splitlines():
        m = re.match(r"^diff --git a/(\S+) b/(\S+)", line)
        if m:
            cur = m[2]; lines.setdefault(cur, {"+": [], "-": []}); continue
        h = re.match(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
        if h and cur:
            out.append((cur, int(h[1]), int(h[2] or 1), int(h[3]), int(h[4] or 1))); continue
        if cur and line[:1] in "+-" and not line.startswith(("+++", "---")):
            lines[cur][line[0]].append(line[1:])
    return out, lines


def ast_hist(text):
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return None
    counts = np.zeros(len(AST_TYPES), np.float32); index = {n: i for i, n in enumerate(AST_TYPES)}
    for node in ast.walk(tree):
        i = index.get(type(node).__name__)
        if i is not None:
            counts[i] += 1
    return counts


def defs(text):
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return {}
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out[node.name] = ast.get_source_segment(text, node) or ""
    return out


def structure(base, patched, patch):
    """S2: base/patched are {path: text} for touched .py files."""
    hs, lines = hunks(patch)
    delta = np.zeros(len(AST_TYPES), np.float32); broken = 0
    added = removed = modified = 0
    for path in sorted(set(base) | set(patched)):
        b, p = base.get(path, ""), patched.get(path, "")
        hb, hp = ast_hist(b) if b else np.zeros(len(AST_TYPES), np.float32), ast_hist(p) if p else np.zeros(len(AST_TYPES), np.float32)
        if hp is None:
            broken += 1; continue
        if hb is not None:
            delta += hp - hb
        db, dp = defs(b), defs(p)
        added += len(dp.keys() - db.keys()); removed += len(db.keys() - dp.keys())
        modified += sum(1 for k in db.keys() & dp.keys() if db[k] != dp[k])
    files = {h[0] for h in hs}
    tests = sum(1 for f in files if re.search(r"(^|/)(tests?|testing)(/|_)|test_[^/]*\.py$|_test\.py$", f))
    plus = sum(len(v["+"]) for v in lines.values()); minus = sum(len(v["-"]) for v in lines.values())
    scalars = np.array([len(files), sum(1 for f in files if f.endswith(".py")), tests, len(hs), plus, minus,
                        added, removed, modified, broken], np.float32)
    return np.concatenate([np.sign(delta) * np.log1p(np.abs(delta)), np.log1p(scalars)])


def behaviour(turns):
    """S4a: counts over the agent's own actions and the environment's replies before submission."""
    ai = [t["text"] for t in turns if t["role"] == "ai"]
    env = [t["text"] for t in turns if t["role"] == "user"][1:]  # the first user turn is the task statement
    cmd = np.zeros(len(COMMANDS), np.float32)
    for text in ai:
        blocks = re.findall(r"```\n?(.*?)```", text, re.S) or [text.strip().splitlines()[-1] if text.strip() else ""]
        first = (blocks[-1].strip().split() or [""])[0]
        if first in COMMANDS:
            cmd[COMMANDS.index(first)] += 1
    signs = np.array([[1.0 if re.search(p, e, re.M) else 0.0 for p in OBS_SIGNS.values()] for e in env] or [[0.0] * len(OBS_SIGNS)], np.float32)
    last = signs[-3:].mean(0)
    # was there a successful run of a script after the last edit?
    order = [("edit" if re.search(r"```\n?edit ", t) else "run" if re.search(r"```\n?python ", t) else "") for t in ai]
    last_edit = max((i for i, o in enumerate(order) if o == "edit"), default=-1)
    ran_after = float(any(o == "run" for o in order[last_edit + 1:]))
    scalars = np.array([len(turns), len(ai), len(env), sum(len(t["text"]) for t in turns) / 1000.0, ran_after], np.float32)
    return np.concatenate([np.log1p(cmd), signs.mean(0), last, np.log1p(scalars)])


def identifiers(text):
    return set(IDENT.findall(text or ""))


def relations(issue, patch):
    """S5a: lexical alignment between the task statement and the submitted change."""
    hs, lines = hunks(patch)
    changed = " ".join(" ".join(v["+"] + v["-"]) for v in lines.values())
    a, b = identifiers(issue), identifiers(changed)
    paths = {h[0] for h in hs}
    mentioned = sum(1 for p in paths if Path(p).name in issue or Path(p).stem in issue)
    inter = len(a & b)
    return np.array([np.log1p(len(a)), np.log1p(len(b)), np.log1p(inter), inter / max(1, len(a)), inter / max(1, len(b)),
                     mentioned / max(1, len(paths)), float("```" in issue or "Traceback" in issue)], np.float32)


def diagnostic(base, patched):
    """S6: findings the patch adds, by pyflakes message class (parsing only)."""
    from pyflakes.api import check
    from pyflakes.reporter import Reporter
    classes = ["UndefinedName", "UnusedImport", "UnusedVariable", "RedefinedWhileUnused", "ImportStarUsed",
               "UndefinedLocal", "DuplicateArgument", "syntax"]

    def findings(files):
        counts = np.zeros(len(classes), np.float32)
        for path, text in files.items():
            out, err = io.StringIO(), io.StringIO()

            class R(Reporter):
                def flake(self, message):
                    name = type(message).__name__
                    if name in classes:
                        counts[classes.index(name)] += 1

                def syntaxError(self, *a):
                    counts[classes.index("syntax")] += 1

                def unexpectedError(self, *a):
                    counts[classes.index("syntax")] += 1
            check(text, path, R(out, err))
        return counts
    db, dp = findings(base), findings(patched)
    d = dp - db
    return np.sign(d) * np.log1p(np.abs(d))


def provenance(model, exit_status):
    """S7: which agent produced the submission and how its session ended."""
    m = np.array([1.0 if model == x else 0.0 for x in MODELS] + [0.0 if model in MODELS else 1.0], np.float32)
    e = np.array([1.0 if exit_status == x else 0.0 for x in EXITS] + [0.0 if exit_status in EXITS else 1.0], np.float32)
    return np.concatenate([m, e])
