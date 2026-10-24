"""Execution-trace perspective without the anchor's own clause (issue #12, readback H1/H2 of runs 8-9). `code-exec-trace`
(encoders_ctx3.py) is retired: its `ran` feature is the A1 oracle's "the code failed to load" clause, so on the two A1
questions it re-derived part of the label. `code-run-trace` reads the same sandboxed trace (the solution executed alone,
no test run) but only on the question whose population the oracle says ran (a1-oci-ran-tests), and only the behavioural
columns that are reproducible: stdout and stderr size, functions and classes bound, whether stdin was read, names bound.
No timing column (wall time is not reproducible under load) and a harness failure is an absence, not a reading. Cached
trace shared with encoders_ctx3 (its builder re-runs the sandbox for the new row set). Cost: one sandboxed child per
row the first time, O(rows) after."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import blake3
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from encoders import Slot  # noqa: E402
from encoders_ctx import CODE_SOURCE  # noqa: E402
from encoders_ctx2 import _absent  # noqa: E402
from encoders_ctx3 import _exec_ctx  # noqa: E402
from lenscommon import Paths  # noqa: E402

ENCODERS_CTX5_B3 = blake3.blake3(Path(__file__).read_bytes()).hexdigest()
RUN_LAYOUT = ["stdout_chars", "stderr_chars", "functions", "classes", "stdin_read", "names"]
RAN_FAMILY = "a1-oci-ran-tests"


def _run_trace_lens(rows: list[dict], paths: Paths) -> Slot:
    if not any(r["source"] == CODE_SOURCE and r.get("family") == RAN_FAMILY for r in rows):
        return _absent(rows, len(RUN_LAYOUT), "neg_cheb", "not the ran-population A1 question")
    traces = _exec_ctx(paths)["traces"]
    X, absent = np.zeros((len(rows), len(RUN_LAYOUT)), np.float32), np.ones(len(rows), bool)
    for i, r in enumerate(rows):
        t = traces.get(r["text_b3"]) if (r["source"] == CODE_SOURCE and r.get("family") == RAN_FAMILY) else None
        if not t or not t.get("ran") or t.get("exception"):
            continue  # the population ran by the oracle's own record; a trace that did not is a harness failure and an absence
        X[i] = [math.log1p(t.get("stdout_chars", 0)), math.log1p(t.get("stderr_chars", 0)), math.log1p(t.get("functions", 0)), math.log1p(t.get("classes", 0)), float(t.get("stdin_read", False)), math.log1p(t.get("names", 0))]
        absent[i] = False
    return Slot(X, absent, {"layout": RUN_LAYOUT, "log1p": "sizes and counts", "read": "behaviour of the solution executed alone; no load-failure, timing or exception column; the tests are never run"}, "neg_cheb")


LENSES = {"code-run-trace": lambda _n, rows, paths: _run_trace_lens(rows, paths)}
SOURCE_OF = {"code-run-trace": CODE_SOURCE}


def encode(name: str, rows: list[dict], paths: Paths) -> Slot:
    return LENSES[name](name, rows, paths)
