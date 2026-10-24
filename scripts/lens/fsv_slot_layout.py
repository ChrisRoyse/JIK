"""Physical native-slot replay and corrupt-layout controls, no admission claim.

Read real retained Dense/Sparse/Multi/Absent measurements, write through the real
slot writers and bindings, then independently decode their bytes. No GPU needed.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import time
import warnings
import zipfile

import blake3
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lenscommon import LensError
from slotio import save_slot, TokenSpool, slot_digest, npz_member
from slot_layout import check_seal
import slot_binding


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def independent(path):
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    value, descriptors = blake3.blake3(), {}
    for key, arr in sorted(arrays.items()):
        value.update(key.encode())
        value.update(arr.tobytes(order="C"))
        descriptors[key] = {"shape": list(arr.shape), "dtype": arr.dtype.str, "order": "C"}
    body = {"schema": "slot-layout-v1", "members": descriptors, "value_digest": value.hexdigest()}
    layout = blake3.blake3(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return arrays, value.hexdigest(), layout


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--registry", required=True, help="retained GPU proof registry (name -> measured entry)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    reg = json.loads(Path(args.registry).read_text())
    checks, sources, sealed, byte_count = [], {}, {}, 0

    def check(name, ok, **detail):
        checks.append({"name": name, "pass": bool(ok), **detail})
        (out / "checks.json").write_text(json.dumps(checks, indent=2) + "\n")
        if not ok:
            raise RuntimeError(name)

    def refuses(name, fn, codes=None):
        try:
            fn()
        except LensError as exc:
            check(name, codes is None or exc.code in codes, code=exc.code)
            return
        check(name, False)

    for name, e in reg.items():
        for family, record in e["families"].items():
            source = Path(record["slot"])
            sources[str(source)] = sha(source)
            byte_count += source.stat().st_size
            arrays, legacy, layout = independent(source)
            check(f"{name}/{family}/original_record", legacy == record["digest"])
            cal_source = source.parent.parent.parent / "cal" / (family + ".jsonl")
            cal = out / "replay" / "cal" / cal_source.name
            cal.parent.mkdir(parents=True, exist_ok=True)
            if not cal.exists():
                shutil.copyfile(cal_source, cal)
            dest = out / "replay" / "slots" / family / source.name
            if "tokens" in arrays:
                spool = TokenSpool(dest, arrays["tokens"].shape[1])
                try:
                    split = len(arrays["absent"]) // 2
                    for a, b in ((0, split), (split, len(arrays["absent"]))):
                        lo, hi = arrays["offsets"][[a, b]]
                        spool.add(arrays["tokens"][lo:hi], arrays["offsets"][a:b + 1] - lo)
                    dg, _ = spool.finish(arrays["windows"])
                finally:
                    spool.fh.close()
                    spool.raw.unlink(missing_ok=True)
            else:
                dg = save_slot(dest, arrays)
            ent = {k: record[k] for k in ("n", "dims", "absent")}
            ent.update(slot=str(dest), digest=dg)
            slot_binding.seal(dest, e["contract"], ent)
            cached = slot_binding.reuse(dest, e["contract"])
            check_seal(cached, ent, dest)
            again, cold_value, cold_layout = independent(dest)
            check(f"{name}/{family}/writer_cache_cold_readback",
                  cold_value == dg == legacy and cold_layout == layout == ent["layout_digest"]
                  and all(np.array_equal(again[k], arrays[k]) for k in arrays))
            key = "tokens" if "tokens" in arrays else "X" if "X" in arrays else "data"
            mm = npz_member(dest, key)
            check(f"{name}/{family}/native_memmap", np.array_equal(mm, arrays[key]))
            if isinstance(mm, np.memmap):
                mm._mmap.close()
            sealed[f"{name}/{family}"] = ent

    dense_key = "ucr-all-minilm-l6-v2/integration-prose"
    dense = Path(sealed[dense_key]["slot"])
    dense_arrays = independent(dense)[0]
    multi = independent(Path(sealed["ucr-answerai-colbert-small/integration-prose"]["slot"]))[0]
    sparse = independent(Path(sealed["ucr-opensearch-sparse-doc-v2/integration-prose"]["slot"]))[0]
    negative = out / "negative"
    negative.mkdir()

    def bad_arrays(name, base, modify):
        arrays = {k: v.copy() for k, v in base.items()}
        modify(arrays)
        path = negative / (name + ".npz")
        # Persist the exact hostile source separately from the guarded writer.
        np.savez(path, **arrays)
        refuses(name + "/reader", lambda: slot_digest(path))
        dest = negative / (name + "-writer.npz")
        refuses(name + "/writer", lambda: save_slot(dest, arrays))
        check(name + "/no_publication", not dest.exists() and not dest.with_name(dest.name + ".tmp").exists())

    bad_arrays("dtype", dense_arrays, lambda a: a.update(X=a["X"].view(np.int32)))
    bad_arrays("shape", dense_arrays, lambda a: a.update(X=a["X"].reshape(16, 192, 2)))
    bad_arrays("nan", dense_arrays, lambda a: a["X"].__setitem__((0, 0), np.nan))
    bad_arrays("infinity", dense_arrays, lambda a: a["X"].__setitem__((0, 0), np.inf))
    bad_arrays("absence_type", dense_arrays, lambda a: a.update(absent=a["absent"].view(np.uint8)))
    bad_arrays("absence_nonzero", dense_arrays, lambda a: a["absent"].__setitem__(0, True))
    bad_arrays("window_count", dense_arrays, lambda a: a.update(windows=a["windows"][:-1]))
    bad_arrays("window_negative", dense_arrays, lambda a: a["windows"].__setitem__(0, -1))
    bad_arrays("offset_origin", multi, lambda a: a["offsets"].__setitem__(0, 1))
    bad_arrays("offset_end", multi, lambda a: a["offsets"].__setitem__(-1, len(a["tokens"]) - 1))
    bad_arrays("offset_backward", multi, lambda a: a["offsets"].__setitem__(2, a["offsets"][1] - 1))
    bad_arrays("sparse_range", sparse, lambda a: a["indices"].__setitem__(0, a["shape"][1]))
    bad_arrays("sparse_shape", sparse, lambda a: a["shape"].__setitem__(0, 17))
    bad_arrays("sparse_pointer", sparse, lambda a: a["indptr"].__setitem__(-1, 0))

    with zipfile.ZipFile(dense) as z:
        parts = {k: z.read(k) for k in z.namelist()}
    for name in ("trailing", "duplicate", "extra", "compressed", "fortran", "torn"):
        path = negative / (name + ".npz")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED if name == "compressed" else zipfile.ZIP_STORED) as z:
                for k, content in parts.items():
                    if k == "X.npy":
                        if name == "trailing": content += b"unbound-trailing-bytes"
                        if name == "torn": content = content[:-1]
                        if name == "fortran":
                            buf = io.BytesIO()
                            np.save(buf, np.asfortranarray(dense_arrays["X"]), allow_pickle=False)
                            content = buf.getvalue()
                    z.writestr(k, content)
                if name == "duplicate": z.writestr("X.npy", parts["X.npy"])
                if name == "extra": z.writestr("unexpected.txt", b"not a slot")
        refuses(name + "/reader", lambda: slot_digest(path))
        refuses(name + "/memmap", lambda: npz_member(path, "X"))

    scale = {k: v.copy() for k, v in dense_arrays.items()}
    scale["X"] *= 2
    path = negative / "norm.npz"
    np.savez(path, **scale)
    refuses("declared_norm_refused", lambda: slot_digest(path, norm="unit"), {"MNLM_SLOT_NORM"})
    refuses("modified_values_refused", lambda: check_seal(slot_digest(path), sealed[dense_key], path), {"MNLM_SLOT_STALE"})
    historical = {k: v for k, v in sealed[dense_key].items() if not k.startswith("layout_")}
    refuses("historical_seal_cannot_authorize", lambda: check_seal(slot_digest(dense), historical, dense), {"MNLM_SLOT_LAYOUT_UNSEALED"})
    binding = dense.with_suffix(".binding.json")
    bound = binding.read_bytes()
    binding.write_text(json.dumps({k: v for k, v in json.loads(bound).items() if not k.startswith("layout_")}))
    refuses("historical_cache_refused", lambda: slot_binding.reuse(dense, reg["ucr-all-minilm-l6-v2"]["contract"]), {"MNLM_SLOT_LAYOUT_UNSEALED"})
    binding.write_bytes(bound)
    protected = sha(dense)
    refuses("bad_overwrite_refused", lambda: save_slot(dense, {**dense_arrays, "X": dense_arrays["X"].view(np.int32)}))
    check("accepted_destination_unchanged", sha(dense) == protected)
    scratch = dense.with_name(dense.name + ".tmp")
    scratch.write_bytes(b"labelled existing-run scratch")
    scratch_before = sha(scratch)
    refuses("existing_scratch_refused", lambda: save_slot(dense, dense_arrays), {"MNLM_SLOT_SCRATCH_EXISTS"})
    check("existing_scratch_not_deleted", sha(scratch) == scratch_before and sha(dense) == protected)
    spool = TokenSpool(negative / "spool.npz", multi["tokens"].shape[1])
    try:
        before = spool.raw.stat().st_size
        refuses("spool_bad_offsets_before_write", lambda: spool.add(multi["tokens"], multi["offsets"] + 1))
        refuses("spool_no_silent_cast", lambda: spool.add(multi["tokens"].astype(np.float64), multi["offsets"]))
        check("spool_failed_batch_no_write", spool.raw.stat().st_size == before and spool.rows == 0)
        spool.add(multi["tokens"], multi["offsets"])
        refuses("spool_bad_windows", lambda: spool.finish(np.asarray([1], np.int64)))
        check("spool_failed_finish_no_publication", not spool.path.exists() and not spool.path.with_name(spool.path.name + ".tmp").exists())
    finally:
        spool.fh.close()
    check("original_sources_unchanged", all(sha(Path(p)) == h for p, h in sources.items()))
    report = {"acceptance": "author-checked only", "metric_kind": "native storage replay, not model or information admission",
              "sources": sources, "source_files": len(sources), "source_bytes": byte_count,
              "records": sealed, "checks": checks, "passed": sum(c["pass"] for c in checks),
              "failed": sum(not c["pass"] for c in checks), "seconds": round(time.monotonic() - start, 3), "served": False}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    paths = [p for p in out.rglob("*") if p.is_file()]
    (out / "SHA256.json").write_text(json.dumps({str(p.relative_to(out)): sha(p) for p in sorted(paths)}, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("source_files", "source_bytes", "passed", "failed", "seconds", "served")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
