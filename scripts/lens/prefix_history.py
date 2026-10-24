"""Native diagnostic history slots from sealed training prefixes; never admission."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
from senses.contract import canonical, require
from senses.forecast import put
from senses.forecast_prefix import PrefixInputs
from senses.instruments import DIM, contracts, measure, slot_digest as observation_digest
from senses.pipeline import file_hash
from slotio import TokenSpool, slot_digest


def publish(reader, out):
    out.mkdir(parents=True, exist_ok=False)
    contract = next(c for c in contracts() if c["channel"] == "history")
    spool = TokenSpool(out / "history.npz", DIM)
    bindings, windows = [], []
    try:
        for atom in reader.examples:
            current, past, binding = reader.read(atom)
            slot = measure(current, "history", past)
            require(observation_digest(slot) == observation_digest(measure(current, "history", past)),
                    "MNLM_PREFIX_NONDETERMINISTIC", "temporal instrument repeat drift")
            spool.add(slot["tokens"], np.asarray([0, len(slot["tokens"])], np.int64))
            bindings.append({"atom": atom, "binding": binding, "slot_digest": observation_digest(slot),
                             "kind": slot["kind"], "reason": slot["reason"], "shape": slot["shape"]})
            windows.append(len(past))
        value_digest, counts = spool.finish(np.asarray(windows, np.int64))
    finally:
        spool.fh.close()
        spool.raw.unlink(missing_ok=True)
    put(out / "rows.json", canonical(bindings))
    manifest = {"schema": "training-prefix-slots-v1", "contract": contract, "input_manifest_sha256": reader.identity,
                "reader_code_sha256": reader.reader_code_sha256, "rows": len(bindings), "counts": counts,
                "slot": slot_digest(out / "history.npz"), "value_digest": value_digest,
                "rows_sha256": file_hash(out / "rows.json"), "stage3_admitted": False, "stage4_accepted": False}
    put(out / "manifest.json", canonical(manifest))
    return manifest


def load(root, expected_manifest_sha256):
    raw = (root / "manifest.json").read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected_manifest_sha256, "MNLM_PREFIX_SLOT_DIGEST", "unbound slot manifest")
    m = json.loads(raw)
    require(m["schema"] == "training-prefix-slots-v1" and m["stage3_admitted"] is False and m["stage4_accepted"] is False,
            "MNLM_PREFIX_SLOT_MANIFEST", "diagnostic native history required")
    require(slot_digest(root / "history.npz") == m["slot"] and m["value_digest"] == m["slot"]["digest"],
            "MNLM_PREFIX_SLOT_DIGEST", "native history values/layout changed")
    raw = (root / "rows.json").read_bytes()
    require(hashlib.sha256(raw).hexdigest() == m["rows_sha256"], "MNLM_PREFIX_SLOT_DIGEST", "row bindings changed")
    rows = json.loads(raw)
    require(len(rows) == m["rows"] == m["slot"]["n"], "MNLM_PREFIX_SLOT_MANIFEST", "row count mismatch")
    return m, rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prefix", type=Path, required=True)
    ap.add_argument("--sha256", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    m = publish(PrefixInputs(a.prefix, a.sha256), a.out)
    print(json.dumps({"rows": m["rows"], "value_digest": m["value_digest"], "stage4_accepted": False}))
