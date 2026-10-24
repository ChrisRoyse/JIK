"""A separately identified, window-preserving view of a frozen temporal encoder.

Every input row has one output row. Windows use the original context bound and
reset position/attention explicitly; this is not full-history attention. It is
one history sense, regardless of window count. No learning or admission occurs.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
from senses.contract import canonical, digest, require
from senses.pipeline import file_hash
from senses.trajectory_population import publish as put, read_pinned
from prefix_panel import load as load_panel, native_state
from slotio import TokenSpool, npz_member, slot_digest
from temporal_encoder import CONFIG, FrozenTemporal
from model.checkpoint import state_hash
from lenscommon import take_lease, release_lease

SCHEMA = "windowed-temporal-native-v1"
POLICY = {"schema": "consecutive-native-windows-v1", "max_tokens": CONFIG.max_tokens,
          "overlap": 0, "positions": "reset_per_window", "attention": "causal_within_window_only",
          "output": "all_token_rows_in_original_order", "pooling": False}


def window_offsets(n):
    require(type(n) is int and n >= 0, "MNLM_TEMPORAL_WINDOW_INPUT", "nonnegative row count required")
    return [*range(0, n, CONFIG.max_tokens), n] if n else [0]


class WindowedTemporal:
    def __init__(self, frozen):
        require(isinstance(frozen, FrozenTemporal), "MNLM_TEMPORAL_WINDOW_INSTRUMENT", "hash-bound frozen base required")
        self.base, self.dim = frozen, frozen.dim
        self.contract = {**frozen.contract, "name": "ucr-temporal-windowed-v1",
                         "base_lens_id": frozen.contract["lens_id"], "segmentation": dict(POLICY),
                         "wrapper_code_sha256": file_hash(Path(__file__))}
        self.contract.pop("lens_id")
        self.contract["preprocessing"] = {**frozen.contract["preprocessing"],
            "order": "consecutive bounded CUDA forwards; every native row retained; explicit window boundaries"}
        self.contract["lens_id"] = digest(self.contract)

    def encode(self, values, *, token_budget):
        require(type(token_budget) is int and token_budget >= 0 and isinstance(values, np.ndarray)
                and values.dtype == np.float32 and values.ndim == 2 and values.shape[1] == CONFIG.dims
                and bool(np.isfinite(values).all()), "MNLM_TEMPORAL_WINDOW_INPUT", "finite native f32 rows required")
        require(len(values) <= token_budget, "MNLM_TEMPORAL_WINDOW_BUDGET", "input exceeds preflight; never truncate")
        offsets = window_offsets(len(values))
        output = np.empty((len(values), self.dim), np.float32)
        if not len(values):
            # Even absence checks the original frozen runtime/weight state.
            self.base.encode(values)
        for lo, hi in zip(offsets, offsets[1:]):
            output[lo:hi] = self.base.encode(values[lo:hi])
        return output, offsets


def census(root, manifest_sha256, checkpoint_id):
    manifest, rows = load_panel(root, manifest_sha256)
    offsets = npz_member(root / "history.npz", "offsets")
    try:
        lengths = np.diff(offsets)
        full, tail = lengths // CONFIG.max_tokens, lengths % CONFIG.max_tokens
        return {"schema": "temporal-window-count-v1", "source_panel_manifest_sha256": manifest_sha256,
                "rows": len(rows), "token_rows": int(lengths.sum()), "max_tokens": int(lengths.max()),
                "window_tokens": CONFIG.max_tokens, "windows": int(((lengths + CONFIG.max_tokens - 1) // CONFIG.max_tokens).sum()),
                "max_windows_per_observation": int(((lengths + CONFIG.max_tokens - 1) // CONFIG.max_tokens).max()),
                "over_context_observations": int((lengths > CONFIG.max_tokens).sum()),
                "attention_pairs": int((full * CONFIG.max_tokens ** 2 + tail ** 2).sum()),
                "projected_output_values_bytes": int(lengths.sum()) * CONFIG.width * 4,
                "checkpoint_id": checkpoint_id, "admitted": False}
    finally:
        if hasattr(offsets, "_mmap"):
            offsets._mmap.close()


def publish(root, manifest_sha256, frozen, plan, out):
    started = time.perf_counter()
    require(plan == census(root, manifest_sha256, frozen.base.contract["checkpoint_id"]),
            "MNLM_TEMPORAL_WINDOW_PLAN", "cold native census differs from fixed preflight")
    manifest, rows = load_panel(root, manifest_sha256)
    upstream = manifest["slots"]["history"]["contract"]
    require(upstream["weights_sha256"] == frozen.base.manifest["input_weights_sha256"]
            and upstream["contract"] == frozen.base.manifest["input_contract"],
            "MNLM_TEMPORAL_WINDOW_INSTRUMENT", "native history preprocessor differs from training")
    out.mkdir(parents=True, exist_ok=False)
    arrays = {k: npz_member(root / "history.npz", k) for k in ("tokens", "offsets")}
    spool, bindings, windows = None, [], []
    try:
        spool = TokenSpool(out / "history.npz", frozen.dim)
        for i, row in enumerate(rows):
            lo, hi = arrays["offsets"][i:i + 2]
            values = arrays["tokens"][lo:hi]
            source = row["slots"]["history"]
            require(native_state(values) == source["state_sha256"] and list(values.shape) == source["shape"],
                    "MNLM_TEMPORAL_WINDOW_INPUT", "source binding/payload mismatch")
            encoded, boundaries = frozen.encode(values, token_budget=plan["max_tokens"])
            spool.add(encoded, np.asarray([0, len(encoded)], np.int64))
            windows.append(len(boundaries) - 1)
            bindings.append({"atom": row["atom"], "binding": row["binding"], "input": source,
                             "window_offsets": boundaries, "output_shape": list(encoded.shape),
                             "output_state_sha256": native_state(encoded)})
        require(sum(windows) == plan["windows"] and spool.rows == plan["token_rows"],
                "MNLM_TEMPORAL_WINDOW_COUNT", "window/token count changed")
        spool.finish(np.asarray(windows, np.int64))
    finally:
        if spool is not None:
            spool.fh.close()
            spool.raw.unlink(missing_ok=True)
        for ar in arrays.values():
            if hasattr(ar, "_mmap"):
                ar._mmap.close()
    require(state_hash(frozen.base.model) == frozen.base.manifest["encoder_state_sha256"],
            "MNLM_TEMPORAL_RUNTIME_DRIFT", "frozen weights changed during measurement")
    m = {"schema": SCHEMA, "contract": frozen.contract, "source_panel_manifest_sha256": manifest_sha256,
         "plan": plan, "plan_sha256": digest(plan), "rows_sha256": put(out / "rows.json", bindings),
         "slot": slot_digest(out / "history.npz"), "file_sha256": file_hash(out / "history.npz"),
         "production_admission": False, "stage4_accepted": False}
    sha = put(out / "manifest.json", m)
    put(out / "COMMITTED", {"manifest_sha256": sha})
    result = {"manifest_sha256": sha, "rows": len(rows), "token_rows": plan["token_rows"],
              "windows": sum(windows), "wall_seconds": time.perf_counter() - started,
              "production_admission": False, "stage4_accepted": False}
    put(out / "run.json", result)
    return result


def load(root, expected_sha256):
    m = read_pinned(root / "manifest.json", expected_sha256)
    require(m.get("schema") == SCHEMA and m.get("production_admission") is False and m.get("stage4_accepted") is False,
            "MNLM_TEMPORAL_WINDOW_MANIFEST", "engineering candidate required")
    require(json.loads((root / "COMMITTED").read_bytes()) == {"manifest_sha256": expected_sha256},
            "MNLM_TEMPORAL_WINDOW_UNCOMMITTED", "uncommitted window publication")
    contract = {k: v for k, v in m["contract"].items() if k != "lens_id"}
    require(digest(contract) == m["contract"]["lens_id"] and contract["segmentation"] == POLICY
            and contract["wrapper_code_sha256"] == file_hash(Path(__file__)) and contract["channel"] == "history"
            and digest(m["plan"]) == m["plan_sha256"],
            "MNLM_TEMPORAL_WINDOW_MANIFEST", "instrument/policy/census identity changed")
    require(file_hash(root / "history.npz") == m["file_sha256"] and slot_digest(root / "history.npz") == m["slot"],
            "MNLM_TEMPORAL_WINDOW_DIGEST", "native values/layout changed")
    rows = read_pinned(root / "rows.json", m["rows_sha256"])
    require(len(rows) == m["slot"]["n"] == m["plan"]["rows"] and len({r["atom"] for r in rows}) == len(rows),
            "MNLM_TEMPORAL_WINDOW_COUNT", "row binding count changed")
    arrays = {k: npz_member(root / "history.npz", k) for k in ("tokens", "offsets", "windows")}
    try:
        for i, row in enumerate(rows):
            lo, hi = arrays["offsets"][i:i + 2]
            values = arrays["tokens"][lo:hi]
            require(row["window_offsets"] == window_offsets(len(values))
                    and int(arrays["windows"][i]) == len(row["window_offsets"]) - 1
                    and list(values.shape) == row["output_shape"] == [row["input"]["shape"][0], CONFIG.width]
                    and native_state(values) == row["output_state_sha256"],
                    "MNLM_TEMPORAL_WINDOW_BINDING", "token/window output differs from row binding")
    finally:
        for ar in arrays.values():
            if hasattr(ar, "_mmap"):
                ar._mmap.close()
    return m, rows


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("panel", "checkpoint", "output"):
        p.add_argument("--" + n, type=Path, required=True)
    for n in ("panel-sha256", "checkpoint-id"):
        p.add_argument("--" + n, required=True)
    p.add_argument("--count-only", action="store_true")
    p.add_argument("--plan", type=Path)
    p.add_argument("--plan-sha256")
    a = p.parse_args()
    if a.count_only:
        plan = census(a.panel, a.panel_sha256, a.checkpoint_id)
        print(json.dumps({"plan_sha256": put(a.output, plan), "plan": plan}))
    else:
        require(a.plan is not None and a.plan_sha256 is not None, "MNLM_TEMPORAL_WINDOW_PLAN", "pin a count preflight")
        plan = read_pinned(a.plan, a.plan_sha256)
        resident = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv"])
        a.output.parent.mkdir(parents=True, exist_ok=True)
        with a.output.with_name(a.output.name + "-gpu-before.txt").open("xb") as inventory:
            inventory.write(resident)
        take_lease(12, "window-preserving-frozen-temporal-publication")
        try:
            frozen = WindowedTemporal(FrozenTemporal(a.checkpoint, a.checkpoint_id))
            result = publish(a.panel, a.panel_sha256, frozen, plan, a.output)
            print(json.dumps(result))
        finally:
            release_lease()
