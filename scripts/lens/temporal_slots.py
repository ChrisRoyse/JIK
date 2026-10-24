"""Persist/read complete learned temporal slots with distinct input/output identities."""
import hashlib
import json
from pathlib import Path

import numpy as np

from senses.contract import canonical, require
from senses.forecast import put
from senses.instruments import slot_digest as observation_digest
from senses.pipeline import file_hash
from model.checkpoint import state_hash
from production.publication import sync_dir
from slotio import TokenSpool, slot_digest


def publish(frozen, values, rows, out):
    require(len(values) == len(rows), "MNLM_TEMPORAL_SLOT_BINDING", "one binding per native observation required")
    out.mkdir()
    spool = TokenSpool(out / "history.npz", frozen.dim)
    bindings, windows = [], []
    try:
        for v, r in zip(values, rows):
            a = frozen.encode(v)
            require(np.array_equal(a, frozen.encode(v)), "MNLM_TEMPORAL_NONDETERMINISTIC", "frozen repeat differs")
            require(list(v.shape) == r["shape"] and (r["kind"] == "Absent") == (len(v) == 0),
                    "MNLM_TEMPORAL_SLOT_BINDING", "input shape/absence differs from bound row")
            source = {"tokens": v, "kind": r["kind"], "reason": r["reason"], "shape": r["shape"],
                      "channel": "history", "dependencies": r["binding"]["visible_past"]}
            require(observation_digest(source) == r["slot_digest"], "MNLM_TEMPORAL_SLOT_BINDING", "input slot digest mismatch")
            output = {**source, "tokens": a, "shape": list(a.shape)}
            spool.add(a, np.asarray([0, len(a)], np.int64))
            bindings.append({"atom": r["atom"], "binding": r["binding"],
                             "input": {k: r[k] for k in ("kind", "reason", "shape", "slot_digest")},
                             "output": {"kind": output["kind"], "reason": output["reason"], "shape": output["shape"],
                                        "slot_digest": observation_digest(output)}})
            windows.append(len(r["binding"]["visible_past"]))
        value_digest, counts = spool.finish(np.asarray(windows, np.int64))
    finally:
        spool.fh.close()
        spool.raw.unlink(missing_ok=True)
    require(state_hash(frozen.model) == frozen.manifest["encoder_state_sha256"],
            "MNLM_TEMPORAL_RUNTIME_DRIFT", "encoder mutated during measurement")
    put(out / "rows.json", canonical(bindings))
    m = {"schema": "trained-temporal-slots-v1", "contract": frozen.contract,
         "rows_sha256": file_hash(out / "rows.json"), "publication_code_sha256": file_hash(Path(__file__)),
         "slot": slot_digest(out / "history.npz"), "value_digest": value_digest, "counts": counts,
         "production_admission": False, "stage4_accepted": False}
    put(out / "manifest.json", canonical(m))
    sync_dir(out)
    return m


def load(root, expected_manifest_sha256):
    raw = (root / "manifest.json").read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected_manifest_sha256,
            "MNLM_TEMPORAL_SLOT_DIGEST", "unbound learned slot manifest")
    m = json.loads(raw)
    require(m["schema"] == "trained-temporal-slots-v1" and m["production_admission"] is False
            and m["stage4_accepted"] is False and m["publication_code_sha256"] == file_hash(Path(__file__)),
            "MNLM_TEMPORAL_SLOT_MANIFEST", "unbound diagnostic publication definition")
    actual = slot_digest(root / "history.npz")
    require(actual == m["slot"] and actual["digest"] == m["value_digest"],
            "MNLM_TEMPORAL_SLOT_DIGEST", "native values/absence/layout changed")
    raw = (root / "rows.json").read_bytes()
    require(hashlib.sha256(raw).hexdigest() == m["rows_sha256"], "MNLM_TEMPORAL_SLOT_DIGEST", "binding bytes changed")
    rows = json.loads(raw)
    require(len(rows) == actual["n"] == m["counts"]["n"], "MNLM_TEMPORAL_SLOT_MANIFEST", "native/binding count mismatch")
    return m, rows
