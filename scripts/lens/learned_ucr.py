"""Frozen learned surface instruments consuming only bound, masked UCR views.

All models read ONE sense. Absent observations never enter a tokenizer. Native
Dense/Sparse/Multi layouts survive, with explicit absence and window provenance.
The input snapshot, masking policy, runtime and preprocessing are hash-bound.
"""
from __future__ import annotations

import hashlib
import platform
import time
from pathlib import Path

import blake3
import numpy as np

from lenscommon import LensError, write_json
from slotio import TokenSpool, save_slot
import slot_binding
import text_windows
from universal import ChannelReader
from senses.contract import SenseError, canonical, digest, require


def specs(learned):
    return [{**s, "name": "ucr-" + s["name"], "inputs": "ucr", "channel": "surface",
             "perspective": "surface", "role": "universal_instrument", "module": "learned_ucr"}
            for s in learned]


def visible_batch(reader, policy, rows):
    texts, bindings = [], []
    for row in rows:
        r, _, binding = reader.read(row, policy)
        items = r["channels"]["surface"]["items"]
        require(len(items) <= 1, "MNLM_UCR_SURFACE_LAYOUT", "one atom surface required; split separate observations into atoms")
        texts.append(items[0]["text"] if items else None)
        bindings.append(binding)
    return texts, bindings


def expand_rows(out, indices, n, kind, dim):
    """Restore absent rows without manufacturing a zero observation (FSV seam)."""
    absent = np.ones(n, bool)
    absent[indices] = False
    if kind == "multi":
        toks, offsets = out
        require(np.isfinite(toks).all(), "MNLM_LENS_NONFINITE", "token measurement")
        counts = np.zeros(n, np.int64)
        counts[indices] = np.diff(offsets)
        return {"tokens": toks, "offsets": np.r_[0, np.cumsum(counts)], "absent": absent | (counts == 0)}
    if kind == "sparse":
        import scipy.sparse as sp
        m = out.tocoo()
        require(np.isfinite(m.data).all(), "MNLM_LENS_NONFINITE", "sparse measurement")
        full = sp.csr_matrix((m.data, (np.asarray(indices)[m.row], m.col)), shape=(n, dim))
        return {"data": full.data.astype(np.float32), "indices": full.indices.astype(np.int32),
                "indptr": full.indptr.astype(np.int64), "shape": np.asarray(full.shape, np.int64),
                "absent": absent | (np.diff(full.indptr) == 0)}
    require(np.isfinite(out).all(), "MNLM_LENS_NONFINITE", "dense measurement")
    full = np.zeros((n, dim), np.float32)
    full[indices] = out
    return {"X": full, "absent": absent | (np.linalg.norm(full, axis=1) == 0)}


def embed_batch(lens, texts):
    indices = np.asarray([i for i, t in enumerate(texts) if t is not None and t.strip()], np.int64)
    lens.windows.clear()
    lens.segments = []
    if len(indices):
        out = lens.embed([texts[i] for i in indices])
    elif lens.kind == "multi":
        out = (np.empty((0, lens.dim), np.float32), np.zeros(1, np.int64))
    elif lens.kind == "sparse":
        import scipy.sparse as sp
        out = sp.csr_matrix((0, lens.dim), dtype=np.float32)
    else:
        out = np.empty((0, lens.dim), np.float32)
    require(len(lens.windows) == len(indices), "MNLM_LENS_WINDOWS_MISMATCH", lens.spec["name"])
    arrays = expand_rows(out, indices, len(texts), lens.kind, lens.dim)
    arrays["windows"] = np.zeros(len(texts), np.int64)
    arrays["windows"][indices] = lens.windows
    require(len(lens.segments) == len(indices), "MNLM_LENS_WINDOW_PROVENANCE", "window trace/observed-row count mismatch")
    lens.batch_segments = [text_windows.empty(t, lens.spec.get("prefix", "")) for t in texts]
    for i, trace in zip(indices, lens.segments):
        lens.batch_segments[i] = trace
    return arrays


def measure_learned_ucr(spec, fams, sdir, pins, store_path, policies_path, batch):
    from learned import LearnedLens, check_gpu
    from measure import probe_equal, probe_prefix
    import torch
    import transformers
    reader, lens = ChannelReader(store_path, policies_path), None
    try:
        require(batch > 0, "MNLM_LENS_BATCH", "positive batch size required")
        policies = {f: reader.policy(f) for f in fams}
        probe = []
        for fam, rows in fams.items():
            for i in range(0, len(rows), batch):
                texts, _ = visible_batch(reader, policies[fam], rows[i:i + batch])
                probe.extend(t for t in texts if t is not None and t.strip())
                if len(probe) >= 64:
                    break
            if len(probe) >= 64:
                break
        require(probe, "MNLM_UCR_CHANNEL_UNOBSERVED", "surface absent in the declared calibration scope")
        runtime = {**check_gpu(), "python": platform.python_version(), "torch_version": torch.__version__,
                   "transformers": transformers.__version__, "platform": platform.platform()}
        started = time.monotonic()
        lens = LearnedLens(spec, "cuda")
        identity = {**lens.contract(), "inputs": "ucr", "channel": "surface", "runtime": runtime,
                    "code_sha256": {p: hashlib.sha256((Path(__file__).parent / p).read_bytes()).hexdigest()
                                    for p in ("learned.py", "learned_ucr.py", "universal.py", "text_windows.py")}}
        identity["view_code_sha256"] = hashlib.sha256((Path(__file__).parents[1] / "senses" / "views.py").read_bytes()).hexdigest()
        contract = {**identity, "ucr_generation": reader.generation, "policies_sha256": reader.policy_hash}
        require(spec["name"] not in pins or pins[spec["name"]] == contract["weights_sha256"],
                "MNLM_LENS_FROZEN_VIOLATION", spec["name"])
        lens.windows.clear()
        small_n = min(8, max(1, min(64, len(probe)) // 2))
        a = lens.embed(probe[:64])
        window_probe = text_windows.probe_digest(lens.windows, lens.segments)
        prefix_probe = text_windows.probe_digest(lens.windows[:small_n], lens.segments[:small_n])
        lens.windows.clear()
        b = lens.embed(probe[:64])
        require(probe_equal(a, b), "MNLM_LENS_NONDETERMINISTIC", spec["name"])
        text_windows.check_probe(window_probe, text_windows.probe_digest(lens.windows, lens.segments), spec["name"])
        lens.windows.clear()
        small = lens.embed(probe[:small_n])
        require(probe_equal(probe_prefix(a, small_n), small), "MNLM_LENS_BATCH_DEPENDENT", spec["name"])
        text_windows.check_probe(prefix_probe, text_windows.probe_digest(lens.windows, lens.segments), spec["name"])
        entry = {"contract": contract, "lens_id": blake3.blake3(canonical(identity)).hexdigest()[:32], "families": {},
                 "load_secs": round(time.monotonic() - started, 3),
                 "probe": {"n": min(64, len(probe)), "present": min(64, len(probe)), "byte_identical": True,
                           "window_provenance_byte_identical": True, "window_provenance_sha256": window_probe,
                           "batch_invariance_byte_identical": True, "batch_invariance_rows": small_n}}
        for fam, rows in fams.items():
            path = sdir / fam / (spec["name"] + ".npz")
            bindings, parts, windows, details = [], [], [], []
            spool = TokenSpool(path, lens.dim) if lens.kind == "multi" else None
            try:
                for i in range(0, len(rows), batch):
                    texts, bound = visible_batch(reader, policies[fam], rows[i:i + batch])
                    arrays = embed_batch(lens, texts)
                    bindings.extend(bound)
                    windows.extend(arrays["windows"].tolist())
                    details.extend({"text_windows": trace} for trace in lens.batch_segments)
                    if spool:
                        spool.add(arrays["tokens"], arrays["offsets"])
                    else:
                        parts.append(arrays)
                if spool:
                    dg, counts = spool.finish(np.asarray(windows, np.int64))
                else:
                    arrays = combine(parts, lens.kind, lens.dim)
                    dg = save_slot(path, arrays)
                    counts = {"n": len(rows), "absent": int(arrays["absent"].sum())}
            finally:
                if spool:
                    spool.fh.close()
                    spool.raw.unlink(missing_ok=True)
            ent = {"slot": str(path), "digest": dg, "dims": lens.dim, **counts, "ucr_generation": reader.generation,
                   "view_bindings_sha256": digest(bindings), "policy_sha256": digest(policies[fam]),
                   "windows_max": max(windows, default=0), "windows_gt1": sum(w > 1 for w in windows)}
            provenance = {"generation": reader.generation, "policy": policies[fam], "bindings": bindings, "details": details}
            write_json(path.with_suffix(".views.json"), provenance)
            ent["window_provenance_sha256"] = digest(provenance)
            text_windows.validate_slot(path, contract, provenance)
            slot_binding.seal(path, contract, ent)
            entry["families"][fam] = ent
        return entry
    except SenseError as exc:
        raise LensError(exc.code, exc.message) from exc
    finally:
        reader.close()
        del lens
        torch.cuda.empty_cache()


def combine(parts, kind, dim):
    require(parts, "MNLM_UCR_EMPTY", "no calibration rows")
    if kind != "sparse":
        return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    import scipy.sparse as sp
    full = sp.vstack([sp.csr_matrix((p["data"], p["indices"], p["indptr"]), shape=tuple(p["shape"])) for p in parts]).tocsr()
    return {"data": full.data.astype(np.float32), "indices": full.indices.astype(np.int32), "indptr": full.indptr.astype(np.int64),
            "shape": np.asarray(full.shape, np.int64), "absent": np.concatenate([p["absent"] for p in parts]),
            "windows": np.concatenate([p["windows"] for p in parts])}
