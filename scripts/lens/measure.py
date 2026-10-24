"""Stage 3 lens registration and calibration measurement (issue #12, spec §19.2–19.3, Def. 12.2): every catalogue lens is
registered with a frozen contract (name, weights digest, repository commit, shape, modality, dtype f32, norm) and a
lens_id binds the complete frozen contract, including preprocessing and atom-local execution. Repeated batches and
per-atom bytes under smaller learned batches must be identical, and the instrument is then measured over the
calibration atoms of every family. Slots are persisted per (family, lens) under <out>/lenses/<version>/slots/ and the
registry under registry.json; one ledger entry `lenses.measure`. A pinned panel (panel.json) is checked at load: a
weights digest that differs refuses MNLM_LENS_FROZEN_VIOLATION. Learned lenses need the GPU and a lease; no CPU fallback.
Cost: per (lens × family × atom) forward passes; reuse requires a matching contract, calibration and content binding; context slots are remeasured.
    python -I scripts/lens/measure.py --root R --out O --version p0-v1 [--families a,b] [--lenses a,b] [--batch 32]
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import blake3  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from catalogue import CATALOGUE, by_name  # noqa: E402
from lenscommon import HOST, LensError, Paths, families_present, ledger_append, lens_root, now_iso, read_family, take_lease, write_json  # noqa: E402

PROBE_N = 64
PROBE_SMALL = 8


def lens_id(contract: dict) -> str:
    raw = json.dumps(contract, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    return blake3.blake3(b"frozen-contract-v2\0" + raw).hexdigest()[:32]


from slotio import TokenSpool, arrays_digest, save_slot, slot_digest  # noqa: E402,F401  (bounded-memory slot I/O; issue #5)


def nvidia_smi() -> str:
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv"], capture_output=True, text=True, timeout=20).stdout.strip()
        lines = out.splitlines()
        kept = lines[:1] + [ln for ln in lines[1:] if "[N/A]" not in ln][:40]
        note = f"({len(lines) - 1} processes listed, those without a memory figure omitted)" if len(lines) > 1 else ""
        return "\n".join(kept + [note])
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"unavailable: {exc}"


def pinned(root: Path) -> dict:
    p = root / "panel.json"
    if not p.exists():
        return {}
    panel = json.loads(p.read_text(encoding="utf-8"))
    return {c["name"]: c["contract"]["weights_sha256"] for c in panel.get("lenses", [])}


def check_pin(pins: dict, contract: dict) -> None:
    if contract["name"] in pins and pins[contract["name"]] != contract["weights_sha256"]:
        raise LensError("MNLM_LENS_FROZEN_VIOLATION", f"{contract['name']}: weights {contract['weights_sha256'][:16]} differ from the pinned {pins[contract['name']][:16]}")


def probe_equal(a, b) -> bool:
    if isinstance(a, tuple):
        return isinstance(b, tuple) and len(a) == len(b) and all(probe_equal(x, y) for x, y in zip(a, b))
    if hasattr(a, "tocsr"):
        if not hasattr(b, "tocsr") or a.shape != b.shape:
            return False
        a, b = a.tocsr(), b.tocsr()
        return all(probe_equal(x, y) for x, y in ((a.data, b.data), (a.indices, b.indices), (a.indptr, b.indptr)))
    return (isinstance(b, np.ndarray) and a.dtype == b.dtype and a.shape == b.shape
            and np.isfinite(a).all() and a.tobytes() == b.tobytes())


def probe_prefix(a, n):
    return (a[0][:a[1][n]], a[1][:n + 1]) if isinstance(a, tuple) else a[:n]


def to_arrays(out, absent_from_norm: bool) -> dict:
    if isinstance(out, tuple):
        toks, offs = out
        return {"tokens": toks, "offsets": offs, "absent": (np.diff(offs) == 0)}
    if hasattr(out, "tocsr"):
        m = out.tocsr()
        return {"data": m.data.astype(np.float32), "indices": m.indices.astype(np.int32), "indptr": m.indptr.astype(np.int64), "shape": np.asarray(m.shape, np.int64),
                "absent": np.diff(m.indptr) == 0}
    absent = (np.linalg.norm(out, axis=1) == 0) if absent_from_norm else np.zeros(len(out), bool)
    return {"X": out.astype(np.float32), "absent": absent}


def measure_algorithmic(spec: dict, fams: dict, sdir: Path, pins: dict, skip: bool, paths: Paths) -> dict:
    """An algorithmic lens is a function of encoders.py (inputs fields/text/code) or of encoders_ctx.py plus the Stage 0-2
    context it reads (inputs context); the identity carries the module's digest and the probe runs on a family it can read."""
    import slot_binding
    if spec["inputs"] == "context":
        import importlib

        modname = spec.get("module", "encoders_ctx")
        mod = importlib.import_module(modname)
        encode, src = (lambda name, rows: mod.encode(name, rows, paths)), mod.SOURCE_OF[spec["name"]]
        module, b3 = f"scripts/lens/{modname}.py", getattr(mod, {"encoders_ctx": "ENCODERS_CTX_B3", "encoders_kg": "ENCODERS_KG_B3", "encoders_ctx2": "ENCODERS_CTX2_B3", "encoders_ctx3": "ENCODERS_CTX3_B3", "encoders_ctx4": "ENCODERS_CTX4_B3", "encoders_ctx5": "ENCODERS_CTX5_B3", "encoders_ctx6": "ENCODERS_CTX6_B3", "encoders_ctx7": "ENCODERS_CTX7_B3"}[modname])
        only = spec.get("only_families")
        fams = {f: rows for f, rows in fams.items() if not only or f in only}  # a lens restricted to the questions it may read (label-free by construction)
        first = next((rows for rows in fams.values() if any(r["source"] == src for r in rows)), next(iter(fams.values())))
    else:
        import encoders as mod

        encode, module, b3 = mod.encode, "scripts/lens/encoders.py", mod.ENCODERS_B3
        first = next(iter(fams.values()))
    sha = hashlib.sha256(Path(mod.__file__).read_bytes()).hexdigest()
    contract = {"name": spec["name"], "kind": "encoder", "inputs": spec["inputs"], "weights_sha256": sha, "weight_files": [module], "corpus_hash": spec.get("source", "none"),
                "shape": "Sparse(var)" if spec.get("sparse") else "Dense(var)", "modality": "record" if spec["inputs"] in ("fields", "context") else "text", "dtype": "f32",
                "norm": "declared(per lens)", "sim": spec["sim"], "encoders_blake3": b3}
    check_pin(pins, contract)
    entry = {"contract": contract, "lens_id": lens_id(contract), "families": {}, "probe": None}
    a, b = encode(spec["name"], first[:PROBE_N]), encode(spec["name"], first[:PROBE_N])
    entry["probe"] = {"n": min(PROBE_N, len(first)), "byte_identical": probe_equal(a.X if not a.sparse else a.X, b.X if not b.sparse else b.X), "batch_invariance_max_abs": 0.0}
    if not entry["probe"]["byte_identical"]:
        entry["error"] = "MNLM_LENS_NONDETERMINISTIC"
        return entry
    for fam, rows in fams.items():
        path = sdir / fam / f"{spec['name']}.npz"
        t0 = time.time()
        if skip and path.exists():
            cached = slot_binding.reuse(path, contract)
            if cached is not None:
                entry["families"][fam] = cached
                continue
        slot = encode(spec["name"], rows)
        arrays = slot.to_arrays()
        arrays["windows"] = np.ones(len(rows), np.int64)
        dims = int(arrays["X"].shape[1]) if "X" in arrays else int(arrays["shape"][1])
        entry["families"][fam] = {"slot": str(path), "digest": save_slot(path, arrays), "dims": dims, "absent": int(arrays["absent"].sum()), "n": len(rows), "secs": round(time.time() - t0, 2)}
        write_json(sdir / fam / f"{spec['name']}.fit.json", slot.fit)
        slot_binding.seal(path, contract, entry["families"][fam])
    return entry


def measure_learned(spec: dict, fams: dict, sdir: Path, pins: dict, skip: bool, batch: int) -> dict:
    import slot_binding
    import text_windows
    import torch

    from learned import LearnedLens

    t0 = time.time()
    lens = LearnedLens(spec, "cuda")
    contract = lens.contract()
    check_pin(pins, contract)
    entry = {"contract": contract, "lens_id": lens_id(contract), "families": {}, "load_secs": round(time.time() - t0, 1)}
    first = [r["text"] for r in next(iter(fams.values()))[:PROBE_N]]
    small_n = min(PROBE_SMALL, max(1, len(first) // 2))
    a = lens.embed(first)
    window_probe = text_windows.probe_digest(lens.windows, lens.segments)
    prefix_probe = text_windows.probe_digest(lens.windows[:small_n], lens.segments[:small_n])
    lens.windows.clear()
    b = lens.embed(first)
    text_windows.check_probe(window_probe, text_windows.probe_digest(lens.windows, lens.segments), spec["name"])
    lens.windows.clear()
    small = lens.embed(first[:small_n])
    text_windows.check_probe(prefix_probe, text_windows.probe_digest(lens.windows, lens.segments), spec["name"])
    lens.windows.clear()
    if isinstance(a, tuple):  # multi-vector: the token rows of the first PROBE_SMALL texts
        inv = float(np.abs(a[0][: a[1][small_n]] - small[0]).max()) if a[1][small_n] == len(small[0]) else float("inf")
    elif hasattr(a, "tocsr"):
        inv = float(abs(a.tocsr()[:small_n] - small.tocsr()).max())
    else:
        inv = float(np.abs(a[:small_n] - small).max())
    if not probe_equal(probe_prefix(a, small_n), small):
        raise LensError("MNLM_LENS_BATCH_DEPENDENT", f"{spec['name']}: per-atom native bytes change with batch layout")
    entry["probe"] = {"n": len(first), "byte_identical": probe_equal(a, b), "batch_invariance_max_abs": inv,
                      "window_provenance_byte_identical": True, "window_provenance_sha256": window_probe,
                      "batch_invariance_byte_identical": True, "batch_invariance_rows": small_n}
    if not entry["probe"]["byte_identical"]:
        entry["error"] = "MNLM_LENS_NONDETERMINISTIC"
        return entry
    for fam, rows in fams.items():
        path = sdir / fam / f"{spec['name']}.npz"
        t1 = time.time()
        if skip and path.exists():
            cached = slot_binding.reuse(path, contract)
            if cached is not None:
                entry["families"][fam] = cached
                continue
        lens.windows.clear()
        details = []
        if lens.kind == "multi":  # token rows streamed to disk batch by batch: O(batch) resident, not O(tokens) (measure-run15 OOM)
            spool = TokenSpool(path, lens.dim)
            for i in range(0, len(rows), batch):
                toks, offs = lens.embed([r["text"] for r in rows[i:i + batch]])
                details.extend({"text_windows": trace} for trace in lens.segments)
                spool.add(toks, offs)
                del toks
            if len(lens.windows) != len(rows):
                raise LensError("MNLM_LENS_WINDOWS_MISMATCH", f"{spec['name']}/{fam}: {len(lens.windows)} window counts for {len(rows)} atoms")
            windows = np.asarray(lens.windows, np.int64)
            digest, meta = spool.finish(windows)
            entry["families"][fam] = {"slot": str(path), "digest": digest, "dims": lens.dim, "absent": meta["absent"], "n": meta["n"],
                                      "windows_max": int(windows.max()), "windows_gt1": int((windows > 1).sum()), "secs": round(time.time() - t1, 1)}
            write_json(path.with_suffix(".views.json"), {"details": details})
            slot_binding.seal(path, contract, entry["families"][fam])
            continue
        parts = []
        for i in range(0, len(rows), batch):
            parts.append(lens.embed([r["text"] for r in rows[i:i + batch]]))
            details.extend({"text_windows": trace} for trace in lens.segments)
        if hasattr(parts[0], "tocsr"):
            import scipy.sparse as sp

            out = sp.vstack(parts).tocsr()
        else:
            out = np.concatenate(parts)
        arrays = to_arrays(out, absent_from_norm=lens.kind in ("dense", "static"))
        arrays["windows"] = np.asarray(lens.windows, np.int64)
        if len(arrays["windows"]) != len(rows):
            raise LensError("MNLM_LENS_WINDOWS_MISMATCH", f"{spec['name']}/{fam}: {len(arrays['windows'])} window counts for {len(rows)} atoms")
        entry["families"][fam] = {"slot": str(path), "digest": save_slot(path, arrays), "dims": lens.dim, "absent": int(arrays["absent"].sum()), "n": len(rows),
                                  "windows_max": int(arrays["windows"].max()), "windows_gt1": int((arrays["windows"] > 1).sum()), "secs": round(time.time() - t1, 1)}
        write_json(path.with_suffix(".views.json"), {"details": details})
        slot_binding.seal(path, contract, entry["families"][fam])
    del lens
    torch.cuda.empty_cache()
    return entry


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--version", default="p0-v1")
    ap.add_argument("--families", default="")
    ap.add_argument("--lenses", default="")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--issue", default="12")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--ucr-store")
    ap.add_argument("--ucr-policies")
    ap.add_argument("--ucr-blobs", help="local content-addressed raw-percept manifest")
    ap.add_argument("--ucr-blobs-sha256", help="expected SHA-256 of the raw-percept manifest")
    ap.add_argument("--legacy-adapters", action="store_true", help="explicitly run historical source-specific comparisons")
    args = ap.parse_args()
    paths = Paths(args.root, args.out, args.version)
    root = lens_root(paths)
    sdir = root / "slots"
    requested = args.families.split(",") if args.families else families_present(root)
    from training_scope import POLICY
    excluded = set(requested) & set(POLICY["excluded_questions"])
    if excluded and args.families:
        raise LensError("MNLM_SOURCE_EXCLUDED", f"excluded training questions: {sorted(excluded)}")
    fams = {f: read_family(root / "cal", f)[0] for f in requested if f not in excluded}
    specs = [by_name(n) for n in args.lenses.split(",")] if args.lenses else [s for s in CATALOGUE if (s.get("role") != "universal_instrument") == args.legacy_adapters]
    if any(s.get("inputs") == "ucr" for s in specs) and not (args.ucr_store and args.ucr_policies):
        raise LensError("MNLM_UCR_REQUIRED", "provide --ucr-store and --ucr-policies; historical comparisons require --legacy-adapters")
    pins = pinned(root)
    if any(s["kind"] != "encoder" for s in specs):
        import torch
        torch.use_deterministic_algorithms(True)
        from learned import check_gpu
        runtime = check_gpu()
    else:
        runtime = {"device": "algorithmic-cpu", "cpu_fallback": False}
    runtime["resident_before"] = nvidia_smi()
    take_lease(args.issue, "lens measurement")
    registry_path = root / "registry.json"
    from lenscommon import sealed_registry
    from admission_state import validate_ledger
    registry = sealed_registry(root) if registry_path.exists() and args.skip_existing else {"lenses": {}}
    if args.skip_existing and registry_path.exists():
        validate_ledger(paths.ledger, registry)
    for spec in specs:
        t0 = time.time()
        if spec.get("retired"):  # a lens retired by a readback finding stays in the record with its reason; its slots stay on disk; it is never assayed
            registry["lenses"][spec["name"]] = {"error": "MNLM_LENS_RETIRED", "detail": spec["retired"], "spec": spec, "secs": 0.0, "families": {}}
            print(f"{spec['name']}: retired ({spec['retired'][:60]})", flush=True)
            continue
        try:
            if spec.get("inputs") == "ucr":
                if spec.get("module") == "image_ucr":
                    from image_ucr import measure_image_ucr
                    entry = measure_image_ucr(spec, fams, sdir, pins, args.ucr_store, args.ucr_policies,
                                              args.ucr_blobs, args.ucr_blobs_sha256)
                elif spec.get("module") == "vision_ucr":
                    from vision_ucr import measure_vision_ucr
                    entry = measure_vision_ucr(spec, fams, sdir, pins, args.ucr_store, args.ucr_policies,
                                              args.ucr_blobs, args.ucr_blobs_sha256)
                elif spec["kind"] == "encoder":
                    from universal import measure_universal
                    entry = measure_universal(spec, fams, sdir, pins, args.ucr_store, args.ucr_policies)
                elif spec.get("module") == "audio_ucr":
                    from audio_ucr import measure_audio_ucr
                    entry = measure_audio_ucr(spec, fams, sdir, pins, args.ucr_store, args.ucr_policies,
                                              args.ucr_blobs, args.ucr_blobs_sha256)
                else:
                    from learned_ucr import measure_learned_ucr
                    entry = measure_learned_ucr(spec, fams, sdir, pins, args.ucr_store, args.ucr_policies, args.batch)
            else:
                entry = measure_algorithmic(spec, fams, sdir, pins, args.skip_existing, paths) if spec["kind"] == "encoder" else measure_learned(spec, fams, sdir, pins, args.skip_existing, args.batch)
        except LensError as exc:
            if exc.code == "MNLM_LENS_FROZEN_VIOLATION":
                raise
            entry = {"error": exc.code, "detail": str(exc)[:300]}
        except Exception as exc:  # noqa: BLE001 - a lens that cannot be loaded or run is a recorded registration failure, never a silent skip
            entry = {"error": "MNLM_LENS_LOAD_FAILED", "detail": f"{type(exc).__name__}: {exc}"[:400]}
        entry["secs"] = round(time.time() - t0, 1)
        entry["spec"] = spec
        registry["lenses"][spec["name"]] = entry
        registry.update({"version": args.version, "runtime": runtime, "host": HOST, "ts": now_iso(), "families": list(fams)})
        write_json(root / "registry.partial.json", registry)  # progress only; registry.json is written once, sealed, at the end
        print(f"{spec['name']}: {entry.get('error') or 'ok'} {entry['secs']} s" + (f" probe={entry['probe']}" if entry.get("probe") else ""), flush=True)
    ok = {n: e for n, e in registry["lenses"].items() if not e.get("error")}
    registry["summary"] = {"candidates": len(registry["lenses"]), "registered": len(ok), "failed": {n: e["error"] for n, e in registry["lenses"].items() if e.get("error")},
                           "probe_all_byte_identical": all(e["probe"]["byte_identical"] for e in ok.values()), "resident_after": nvidia_smi()}
    for k in ("ledger_seq", "ledger_head", "registry_blake3_before_ledger"):  # the digest covers the registry without its own ledger fields
        registry.pop(k, None)
    body = (json.dumps(registry, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n").encode("utf-8")
    digest = blake3.blake3(body).hexdigest()
    e = ledger_append(paths, f"measure@{HOST}", "lenses.measure", f"stage3/{args.version}", {"registry_blake3": digest, **registry["summary"], "families": list(fams)})
    registry["ledger_seq"], registry["ledger_head"], registry["registry_blake3_before_ledger"] = e["seq"], e["hash"], digest
    write_json(registry_path, registry)
    write_json(root / f"registry-{digest}.json", registry)  # content-addressed snapshot: a panel pinned to this digest stays verifiable after the next registration (readback H2c)
    print(json.dumps({**registry["summary"], "ledger_seq": e["seq"]}), flush=True)
    return 0


if __name__ == "__main__":
    from lenscommon import run_cli
    sys.exit(run_cli(main))
