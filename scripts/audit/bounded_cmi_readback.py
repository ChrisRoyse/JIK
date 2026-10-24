"""Independent full-state/count/tail/inequality readback for bounded CMI (#12)."""
import base64
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from finite_cmi_readback import check_native, tail_interval


def replay(directory, plan):
    payload = json.loads((directory / "inputs.json").read_bytes())
    result = json.loads((directory / "result.json").read_bytes())
    check_native(directory, payload)
    with np.load(directory / "witnesses.npz", allow_pickle=False) as ar:
        u, v = ar["u"].tolist(), ar["v"].tolist()
    assert u == payload["u"] and v == payload["v"]
    with (directory / "witnesses.npz").open("rb") as f:
        assert hashlib.file_digest(f, "sha256").hexdigest() == payload["witnesses_sha256"]
    with np.load(directory / "panel.npz", allow_pickle=False) as ar:
        z = ar["X"]
    spec = plan["comparisons"][result["comparison"]]
    alpha = .05/(5*4*sum(len(c["variation"]) for c in plan["comparisons"].values()))
    assert alpha == result["interval_alpha"] and len(z) == spec["evaluation_units"]
    if spec["partition"]["kind"] == "one-cell":
        cells = ["all"]*len(z)
    else:
        p = spec["partition"]
        cells = ["below" if row[p["coordinate"]] < p["cut"] else "at_or_above" for row in z]
    counts = Counter(zip(cells, u, v))
    total = 0.
    for cell in spec["variation"]:
        table = [[counts[cell, a, b] for b in (False, True)] for a in (False, True)]
        n = sum(map(sum, table)); us = sum(table[1]); vs = table[0][1]+table[1][1]; both=table[1][1]
        c = result["cells"][cell]
        assert [c[k] for k in ("n", "u", "v", "both")] == [n, us, vs, both]
        mass = tail_interval(n, len(z), alpha)
        raw = dict(zip(("u", "v", "both"), [tail_interval(k, n, alpha) for k in (us, vs, both)]))
        allowance = spec["variation"][cell]
        assert c["allowance"] == allowance
        expanded = {k:(max(0.,pair[0]-allowance[k]),min(1.,pair[1]+allowance[k])) for k,pair in raw.items()}
        for k in raw:
            assert max(abs(x-y) for x,y in zip(raw[k],c["cell_intervals"][k])) < 2e-11
            assert max(abs(x-y) for x,y in zip(expanded[k],c["full_z_intervals"][k])) < 2e-11
        a,b,joint = expanded["u"],expanded["v"],expanded["both"]
        gap=max(0.,joint[0]-a[1]*b[1],a[0]*b[0]-joint[1])
        weight=mass[0]*8*gap**2/math.log(2) if n >= 50 and min(vs,n-vs) >= 50 else 0.
        assert abs(weight-c["weighted_bits_lcb"]) < 2e-10
        total += weight
    assert abs(total-result["bits_lcb"]) < 1e-9
    h=hashlib.sha256()
    for unit,state,a,b in zip(payload["units"],payload["states"],u,v):
        for part in (unit.encode(),base64.b64decode(state),bytes([a,b])):
            h.update(len(part).to_bytes(8,"big"));h.update(part)
    assert h.hexdigest() == result["input_sha256"]
    assert result["production_admission"] is False and result["assumptions_proven_by_this_estimator"] is False
    return dict(case=directory.name,bits_lcb=total,full_states=len(z),unique_states=len(set(payload["states"])))


def probability_inequalities():
    """Enumerate valid binary tables; exact MI exceeds the Pinsker covariance bound.

This checks arithmetic, not a proof by finite grid of arbitrary population laws.
The proof is TV(Puv,PuPv)=2*|p11-pu*pv| and Pinsker KL_bits>=2*TV^2/ln(2).
"""
    count, worst = 0, math.inf
    for a in range(21):
        for b in range(21-a):
            for c in range(21-a-b):
                p=np.asarray([[a,b],[c,20-a-b-c]],float)/20
                x,y=p.sum(1),p.sum(0)
                mi=math.fsum(p[i,j]*math.log2(p[i,j]/(x[i]*y[j])) for i in range(2) for j in range(2) if p[i,j])
                gap=p[1,1]-x[1]*y[1]
                lower=8*gap*gap/math.log(2)
                assert mi+1e-12>=lower
                worst=min(worst,mi-lower);count+=1
    return dict(binary_tables=count,min_margin_bits=worst)


def analytical_law_check(laws):
    """Full-Z population arguments are explicit; numeric checks support their algebra."""
    z=np.linspace(-1.,1.,4097)
    q=.1+.02*z
    # U independent fair bit, V flipped with q(z): both marginals are exactly 1/2.
    joint=.5*(1-q)
    assert max(abs(joint-.45)) <= laws["bounded-positive"]["variation"]["all"]["both"]+1e-15
    entropy=-(q*np.log2(q)+(1-q)*np.log2(1-q))
    return dict(bounded_positive_cmi_min=float((1-entropy).min()),
                bounded_positive_cmi_max=float((1-entropy).max()),grid_points=len(z),
                proof="marginals=1/2; p11=.45-.01*z; |z|<=1 implies joint deviation<=.01; all other laws have exact stated probabilities")
