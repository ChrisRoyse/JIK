"""Independent count/tail/typed-state readback for #12 exact-context experiment.

Does not call the bound's interval/counting/serialization implementation. Beta
quantiles are checked by binomial-tail inversion; full native rows are reopened
with NumPy rather than slot_layout/memmap. These are author checks, not acceptance.
"""
from __future__ import annotations

import base64
from collections import Counter
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import binom


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


@lru_cache(maxsize=16384)
def tail_interval(k, n, alpha):
    if n == 0:
        return (0.0, 1.0)
    def root(k, increasing):
        lo, hi = 0.0, 1.0
        for _ in range(58):
            p = (lo + hi) / 2
            tail = float(binom.sf(k - 1, n, p) if increasing else binom.cdf(k, n, p))
            if (tail < alpha / 2) == increasing:
                lo = p
            else:
                hi = p
        return (lo + hi) / 2
    return (0.0 if k == 0 else root(k, True), 1.0 if k == n else root(k, False))


def check_native(directory, payload):
    """Exact typed/value reconstruction without the production serializer."""
    native = []
    for item in payload["panel"]:
        path = directory / item["path"]
        assert sha(path) == item["sha256"]
        with np.load(path, allow_pickle=False) as z:
            native.append((item["id"], {k: z[k].copy() for k in z.files}))
    for i, encoded in enumerate(payload["states"]):
        state = json.loads(base64.b64decode(encoded))
        assert [s["id"] for s in state] == payload["panel_ids"]
        for s, (name, arrays) in zip(state, native):
            assert s["id"] == name
            kind = "dense" if "X" in arrays else "multi" if "tokens" in arrays else "sparse"
            assert s["kind"] == kind
            expected = {"absent": arrays["absent"][i:i + 1]}
            if "windows" in arrays:
                expected["windows"] = arrays["windows"][i:i + 1]
            if kind == "dense":
                expected["X"] = arrays["X"][i]
            else:
                o = arrays["offsets" if kind == "multi" else "indptr"]
                start, end = int(o[i]), int(o[i + 1])
                key = "tokens" if kind == "multi" else "data"
                expected[key] = arrays[key][start:end]
                if kind == "sparse":
                    expected["indices"] = arrays["indices"][start:end]
                    expected["vocabulary_extent"] = arrays["shape"][1:2]
            assert set(s["members"]) == set(expected)
            for key, array in expected.items():
                part = s["members"][key]
                assert part["shape"] == list(array.shape) and part["dtype"] == array.dtype.str
                assert base64.b64decode(part["bytes"]) == array.tobytes(order="C")


def replay(directory, plan):
    directory = Path(directory)
    payload = json.loads((directory / "inputs.json").read_bytes())
    result = json.loads((directory / "result.json").read_bytes())
    assert result["production_admission"] is False
    plan_bytes = json.dumps(plan, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    assert result["plan_sha256"] == hashlib.sha256(plan_bytes).hexdigest()
    spec = plan["comparisons"][result["comparison"]]
    assert payload["panel_ids"] == spec["panel_ids"]
    candidate = directory / payload["candidate"]["path"]
    outcome = directory / payload["outcome"]["path"]
    assert sha(candidate) == payload["candidate"]["sha256"] and sha(outcome) == payload["outcome"]["sha256"]
    with np.load(candidate, allow_pickle=False) as q:
        assert payload["u"] == ((q["X"][:, 0] > 0) & ~q["absent"]).tolist()
    assert payload["v"] == np.load(outcome, allow_pickle=False).tolist()
    if not payload["artificial_boundary_only"]:
        labels = [json.loads(s) for s in (directory.parent / "real-labels.jsonl").read_text(encoding="utf-8").splitlines()]
        assert payload["v"] == [r["y"] == 0 for r in labels]
        assert payload["units"] == [r["cluster"] for r in labels]
    check_native(directory, payload)
    units = payload["units"]
    assert len(units) == len(set(units))
    states = [base64.b64decode(s) for s in payload["states"]]
    split = plan["design_units"]
    chosen = set(states[:split])
    n = len(states) - split
    assert n == spec["evaluation_units"]
    assert len(result["contexts"]) == len(chosen)
    alpha = plan["alpha"] / len(plan["seeds"]) / len(plan["comparisons"]) / 4 / len(chosen)
    assert abs(alpha - result["interval_alpha"]) < 1e-18
    counts = Counter((states[i], payload["u"][i], payload["v"][i]) for i in range(split, len(states)))
    total = 0.0
    matched = 0
    for state, c in zip(sorted(chosen), result["contexts"]):
        assert c["state_sha256"] == hashlib.sha256(state).hexdigest() and c["state_bytes"] == len(state)
        table = [[counts[(state, a, b)] for b in (False, True)] for a in (False, True)]
        ns, us, vs, both = sum(map(sum, table)), sum(table[1]), table[0][1] + table[1][1], table[1][1]
        assert [c[k] for k in ("n", "u", "v", "both")] == [ns, us, vs, both]
        intervals = [tail_interval(k, m, alpha) for k, m in ((ns, n), (us, ns), (vs, ns), (both, ns))]
        for key, pair in zip(("mass_interval", "u_interval", "v_interval", "both_interval"), intervals):
            assert max(abs(x - y) for x, y in zip(c[key], pair)) < 2e-11
        mass, a, b, joint = intervals
        gap = max(0.0, joint[0] - a[1] * b[1], a[0] * b[0] - joint[1])
        value = mass[0] * 8 * gap ** 2 / math.log(2) if ns >= 50 and min(vs, ns - vs) >= 50 else 0.0
        assert abs(value - c["weighted_bits_lcb"]) < 2e-10
        total += value
        matched += ns
    assert n - matched == result["unselected_evaluation_units"]
    assert abs(total - result["bits_lcb"]) < 1e-9
    return {"case": directory.name, "contexts": len(chosen), "units": n, "bits_lcb": result["bits_lcb"]}


def coverage_grid(interval_function):
    """Enumerate all k at declared p values; no Monte Carlo coverage claim."""
    results = []
    for n in (5, 20, 50, 200):
        for alpha in (0.05, 0.0001):
            bounds = [interval_function(k, n, alpha) for k in range(n + 1)]
            for k, pair in enumerate(bounds):
                reference = tail_interval(k, n, alpha)
                assert max(abs(x - y) for x, y in zip(pair, reference)) < 2e-11
            worst = 1.0
            for p in np.linspace(0, 1, 101):
                prob = math.fsum(float(binom.pmf(k, n, p)) for k, (lo, hi) in enumerate(bounds) if lo <= p <= hi)
                assert prob >= 1 - alpha - 1e-12
                worst = min(worst, prob)
            results.append({"n": n, "alpha": alpha, "minimum_grid_coverage": worst})
    return results
