"""Linear validation of accepted population geometry; no per-shard full-array scan."""
import numpy as np
from artifact_seals import refuse


def validate(pop, meta, version):
    n = meta["docs"]
    if meta["version"] != version or n <= 0 or set(pop) != {"src", "shard", "row", "b3", "n", "prose"}:
        refuse("MNLM_DEDUP_POPULATION_SCHEMA", "version, count or array inventory")
    types = {"src": "u", "shard": "u", "row": "iu", "n": "iu", "prose": "b", "b3": "u"}
    for k, a in pop.items():
        shape = (n, 16) if k == "b3" else (n,)
        if a.shape != shape or a.dtype.kind not in types[k] or (k == "b3" and a.dtype.itemsize != 1):
            refuse("MNLM_DEDUP_POPULATION_SCHEMA", f"{k}: shape/dtype")
    if int(pop["n"].sum()) != meta["canonical_bytes"] or int(pop["prose"].sum()) != meta["prose_docs"]:
        refuse("MNLM_DEDUP_POPULATION_SCHEMA", "byte/prose census")
    pos = 0
    srcs = meta["sources"]
    if len(srcs) != len(set(srcs)):
        refuse("MNLM_DEDUP_POPULATION_SCHEMA", "duplicate source mapping")
    for k, sh in enumerate(meta["shards"]):
        end = pos + sh["docs"]
        if sh["source"] not in srcs or end > n or sh["docs"] <= 0:
            refuse("MNLM_DEDUP_POPULATION_SCHEMA", "shard inventory")
        sl = slice(pos, end)
        if (not np.all(pop["shard"][sl] == k) or not np.all(pop["src"][sl] == srcs.index(sh["source"]))
                or not np.all(np.diff(pop["row"][sl]) > 0) or np.any(pop["row"][sl] < 0)
                or np.any(pop["n"][sl] <= 0) or not np.all(pop["prose"][sl] == (sh["kind"] == "prose"))):
            refuse("MNLM_DEDUP_POPULATION_SCHEMA", f"shard {k}: index/order/geometry")
        pos = end
    if pos != n:
        refuse("MNLM_DEDUP_POPULATION_SCHEMA", "shard/document census")
