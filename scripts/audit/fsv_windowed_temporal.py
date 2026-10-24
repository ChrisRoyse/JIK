"""Cold all-token replay and segmentation/freeze/publication fault controls."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1] / "lens")]
import numpy as np
from senses.contract import SenseError, require
from senses.pipeline import file_hash
from senses.trajectory_population import publish
from prefix_panel import load as load_panel
from windowed_temporal import WindowedTemporal, load
from temporal_encoder import FrozenTemporal, CONFIG
from model.checkpoint import state_hash
from slotio import npz_member
from lenscommon import take_lease, release_lease


def run(a):
    started = time.perf_counter()
    a.output.mkdir(parents=True, exist_ok=False)
    checks = []

    def check(name, ok, **details):
        checks.append({"name": name, "pass": bool(ok), **details})
        require(ok, "MNLM_TEMPORAL_WINDOW_FSV", name)

    def refused(name, action, code):
        try:
            action()
        except SenseError as e:
            check(name, e.code == code, code=e.code)
        else:
            check(name, False)

    pm, source_rows = load_panel(a.panel, a.panel_sha256)
    manifest, rows = load(a.native, a.native_sha256)
    check("source_bound_without_outcomes", manifest["source_panel_manifest_sha256"] == a.panel_sha256
          and manifest["plan"]["checkpoint_id"] == a.checkpoint_id
          and [r["atom"] for r in rows] == [r["atom"] for r in source_rows]
          and all(r["binding"] == s["binding"] and r["input"] == s["slots"]["history"] for r, s in zip(rows, source_rows)))
    before = {p.name: file_hash(p) for p in a.native.iterdir() if p.is_file()}
    checkpoint_before = {p.name: file_hash(p) for p in a.checkpoint.iterdir() if p.is_file()}
    source_before = file_hash(a.panel / "history.npz")
    (a.output / "gpu-before.txt").write_bytes(subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv"]))
    take_lease(12, "window-preserving-temporal-cold-replay")
    arrays = {}
    try:
        base = FrozenTemporal(a.checkpoint, a.checkpoint_id)
        wrapped = WindowedTemporal(base)
        check("runtime_instrument_bound", wrapped.contract == manifest["contract"]
              and wrapped.contract["lens_id"] != base.contract["lens_id"]
              and wrapped.contract["channel"] == base.contract["channel"] == "history")
        for name, path in (("source", a.panel), ("output", a.native)):
            arrays[name] = {k: npz_member(path / "history.npz", k) for k in ("tokens", "offsets", "windows", "absent")}
        check("all_native_offsets_retained", np.array_equal(arrays["source"]["offsets"], arrays["output"]["offsets"])
              and np.array_equal(arrays["source"]["absent"], arrays["output"]["absent"]))
        n_tokens, n_windows = 0, 0
        for i, row in enumerate(rows):
            lo, hi = arrays["source"]["offsets"][i:i + 2]
            values, stored = arrays["source"]["tokens"][lo:hi], arrays["output"]["tokens"][lo:hi]
            # Reconstruct boundaries independently, then invoke the original frozen
            # encoder directly (not the wrapper or publication function).
            expected_boundaries = [0]
            for start in range(0, len(values), CONFIG.max_tokens):
                end = min(start + CONFIG.max_tokens, len(values))
                require(np.array_equal(base.encode(values[start:end]), stored[start:end]),
                        "MNLM_TEMPORAL_WINDOW_FSV", "cold per-window forward differs from persisted output")
                expected_boundaries.append(end)
                n_windows += 1
            require(row["window_offsets"] == expected_boundaries and int(arrays["output"]["windows"][i]) == len(expected_boundaries) - 1,
                    "MNLM_TEMPORAL_WINDOW_FSV", "window boundaries/count differ")
            n_tokens += len(values)
        check("all_tokens_cold_replayed", n_tokens == manifest["plan"]["token_rows"], tokens=n_tokens)
        check("all_windows_cold_replayed", n_windows == manifest["plan"]["windows"], windows=n_windows)
        first_hi = arrays["source"]["offsets"][1]
        values = np.array(arrays["source"]["tokens"][:first_hi], copy=True)
        require(len(values) > CONFIG.max_tokens, "MNLM_TEMPORAL_WINDOW_FSV", "proof needs a real long observation")
        refused("original_context_limit_still_enforced", lambda: base.encode(values[:257]), "MNLM_TEMPORAL_INPUT")
        short, boundaries = wrapped.encode(values[:256], token_budget=256)
        check("single_window_exact_original", np.array_equal(short, base.encode(values[:256])) and boundaries == [0, 256])
        empty, boundaries = wrapped.encode(values[:0], token_budget=0)
        check("absence_stays_absent", empty.shape == (0, CONFIG.width) and boundaries == [0])
        encoded, boundaries = wrapped.encode(values, token_budget=len(values))
        check("repeat_wrapper_exact", np.array_equal(encoded, arrays["output"]["tokens"][:first_hi]))
        changed = values.copy()
        changed[256:, 0] += 1
        later, _ = wrapped.encode(changed, token_budget=len(values))
        check("later_windows_cannot_change_earlier", np.array_equal(later[:256], encoded[:256])
              and not np.array_equal(later[256:], encoded[256:]))
        refused("short_budget_refuses_no_truncation", lambda: wrapped.encode(values, token_budget=len(values) - 1), "MNLM_TEMPORAL_WINDOW_BUDGET")
        refused("dtype_refuses_no_cast", lambda: wrapped.encode(values.astype(np.float64), token_budget=len(values)), "MNLM_TEMPORAL_WINDOW_INPUT")
        bad = values.copy()
        bad[0, 0] = np.nan
        refused("nonfinite_refused", lambda: wrapped.encode(bad, token_budget=len(values)), "MNLM_TEMPORAL_WINDOW_INPUT")
        base.model.train()
        try:
            refused("mutable_training_state_refused", lambda: wrapped.encode(values[:0], token_budget=0), "MNLM_TEMPORAL_RUNTIME_DRIFT")
        finally:
            base.model.eval()
        check("frozen_weights_unchanged", state_hash(base.model) == base.manifest["encoder_state_sha256"])
    finally:
        for ar in arrays.values():
            for v in ar.values():
                if hasattr(v, "_mmap"):
                    v._mmap.close()
        release_lease()

    # Durable shadow controls only copy small metadata; source native files stay
    # read-only. A resealed manifest alone cannot justify wrong window semantics.
    def shadow(name, change_rows=None, change_manifest=None):
        dest = a.output / name
        dest.mkdir()
        os.link(a.native / "history.npz", dest / "history.npz")
        rr, mm = copy.deepcopy(rows), copy.deepcopy(manifest)
        if change_rows:
            change_rows(rr)
        mm["rows_sha256"] = publish(dest / "rows.json", rr)
        if change_manifest:
            change_manifest(mm)
        sha = publish(dest / "manifest.json", mm)
        publish(dest / "COMMITTED", {"manifest_sha256": sha})
        return dest, sha

    for name, rr, mm, code in [
        ("wrong_window_boundary", lambda r: r[0]["window_offsets"].__setitem__(1, 255), None, "MNLM_TEMPORAL_WINDOW_BINDING"),
        ("pooled_output_claim", lambda r: r[0].update(output_shape=[1, CONFIG.width]), None, "MNLM_TEMPORAL_WINDOW_BINDING"),
        ("policy_identity_changed", None, lambda m: m["contract"]["segmentation"].update(overlap=1), "MNLM_TEMPORAL_WINDOW_MANIFEST"),
        ("value_seal_changed", None, lambda m: m.update(file_sha256="0" * 64), "MNLM_TEMPORAL_WINDOW_DIGEST"),
        ("admission_claim", None, lambda m: m.update(production_admission=True), "MNLM_TEMPORAL_WINDOW_MANIFEST"),
    ]:
        dest, sha = shadow(name, rr, mm)
        refused(name, lambda: load(dest, sha), code)
    dest, sha = shadow("uncommitted")
    (dest / "COMMITTED").write_text("{}", encoding="utf-8")
    refused("uncommitted", lambda: load(dest, sha), "MNLM_TEMPORAL_WINDOW_UNCOMMITTED")
    check("input_native_unchanged", source_before == file_hash(a.panel / "history.npz"))
    check("checkpoint_unchanged", checkpoint_before == {n: file_hash(a.checkpoint / n) for n in checkpoint_before})
    check("published_native_unchanged", before == {n: file_hash(a.native / n) for n in before})
    report = {"schema": "windowed-temporal-fsv-v1", "checks": checks, "native_manifest_sha256": a.native_sha256,
              "source_panel_manifest_sha256": a.panel_sha256, "checkpoint_id": a.checkpoint_id,
              "rows": len(rows), "tokens": n_tokens, "windows": n_windows,
              "author_checked": True, "independently_adjudicated": False,
              "production_admission": False, "stage4_accepted": False,
              "code_sha256": file_hash(Path(__file__)), "wall_seconds": time.perf_counter() - started}
    sha = publish(a.output / "report.json", report)
    print(json.dumps({"report_sha256": sha, "checks": len(checks), "wall_seconds": report["wall_seconds"]}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("panel", "native", "checkpoint", "output"):
        p.add_argument("--" + n, type=Path, required=True)
    for n in ("panel-sha256", "native-sha256", "checkpoint-id"):
        p.add_argument("--" + n, required=True)
    run(p.parse_args())
