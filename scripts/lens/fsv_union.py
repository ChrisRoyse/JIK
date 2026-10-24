"""Independent re-derivation of the per-anchor-family union for fsv_stage3.py (issue #12): spec §19.1 (the panel is the union
over a family's questions) under Def. 13.3 D1 applied across the union: question order, then admission order; a lens whose
|Spearman| with any lens already in the union exceeds MAX_CORR on any question of the family where both were read is a
duplicate and is excluded; a pair no question could read fails closed. Written separately from panel.union_panel so the
verifier does not share the code it checks. Cost: O(questions × union² × lenses) over the stored correlation matrices."""
from __future__ import annotations

import json
import math
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lenscommon import FAMILIES, spec_family  # noqa: E402


def rederive_union(root: Path, families: list[str], max_corr: float) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    families = [f for f in FAMILIES if f in families]  # declared question order, the order panel.py forms the union in (a JSON file sorts its keys)
    reps = {f: json.loads((root / "assay" / f"{f}.json").read_text(encoding="utf-8")) for f in families}
    union: dict[str, list[str]] = {}
    excluded: dict[str, list[str]] = {}
    for f, rep in reps.items():
        a = spec_family(f)
        qs = [q for q in families if spec_family(q) == a]
        u = union.setdefault(a, [])
        for nm in rep["admission"]["panel"]:
            if nm in u:
                continue
            worst = []
            for p in u:
                vals = []
                undefined = False
                for q in qs:
                    order, M = reps[q]["correlation"]["order"], reps[q]["correlation"]["matrix"]
                    if nm in order and p in order:
                        value = M[order.index(nm)][order.index(p)]
                        if value is None or not math.isfinite(float(value)):
                            undefined = True
                        else:
                            vals.append(abs(float(value)))
                worst.append(max(vals) if vals and not undefined else None)
            if any(w is None or w > max_corr for w in worst):
                excluded.setdefault(a, []).append(nm)
                continue
            u.append(nm)
    return union, excluded
