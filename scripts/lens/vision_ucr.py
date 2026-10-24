"""Source-neutral learned image assay with the same mask boundary as raw pixels."""
from pathlib import Path

import numpy as np

from image_ucr import visible_image
from vision_encoder import SPEC
from lenscommon import LensError, write_json
from senses.blobs import Blobs
from senses.contract import SenseError, digest, require
from universal import ChannelReader
from slotio import TokenSpool
from production.session import Session
from production.input_plan import code_hashes
import slot_binding


def measure_vision_ucr(spec, fams, sdir, pins, store_path, policies_path, blobs_path, blobs_sha256):
    from vision_assets import FILES
    reader, session = ChannelReader(store_path, policies_path), None
    try:
        require(spec == SPEC, "MNLM_LENS_FROZEN_VIOLATION", "unsupported vision instrument")
        sha = FILES["model.safetensors"][1]
        require(spec["name"] not in pins or pins[spec["name"]] == sha,
                "MNLM_LENS_FROZEN_VIOLATION", "visual weight drift")
        session = Session({"spec": spec, "revision": spec["revision"], "weights_sha256": sha}, reader,
                          code_hashes(), Blobs(blobs_path, blobs_sha256))
        entry = {"contract": session.contract, "lens_id": session.lens_id, "families": {},
                 "probe": {"n": 0, "present": 0, "byte_identical": True}}
        for family, rows in fams.items():
            policy = reader.policy(family)
            path = Path(sdir) / family / (spec["name"] + ".npz")
            spool = TokenSpool(path, session.dim)
            bindings, regions, windows = [], [], []
            try:
                for row in rows:
                    record, binding = visible_image(reader, policy, row)
                    result = session.lens.measure(record, session.blobs)
                    t = result["tokens"]
                    if len(t) and entry["probe"]["present"] < 64:
                        again = session.lens.measure(record, session.blobs)
                        require(t.tobytes() == again["tokens"].tobytes() and result["regions"] == again["regions"],
                                "MNLM_LENS_NONDETERMINISTIC", spec["name"])
                        entry["probe"]["n"] += 1
                        entry["probe"]["present"] += 1
                    spool.add(t, np.asarray([0, len(t)], np.int64))
                    bindings.append(binding)
                    regions.append(result["regions"])
                    windows.append(len(result["regions"]))
                dg, sizes = spool.finish(np.asarray(windows, np.int64))
            finally:
                spool.fh.close()
                spool.raw.unlink(missing_ok=True)
            provenance = {"generation": reader.generation, "policy": policy, "bindings": bindings,
                          "image_regions": regions, "blobs_manifest_sha256": session.blobs.sha256}
            write_json(path.with_suffix(".views.json"), provenance)
            ent = {"slot": str(path), "digest": dg, "dims": session.dim, **sizes,
                   "ucr_generation": reader.generation, "view_bindings_sha256": digest(bindings),
                   "window_provenance_sha256": digest(provenance), "policy_sha256": digest(policy)}
            slot_binding.seal(path, session.contract, ent)
            entry["families"][family] = ent
        require(entry["probe"]["present"] > 0, "MNLM_UCR_CHANNEL_UNOBSERVED", "no image exercised the encoder")
        entry["frozen_encoder_state_sha256"] = session.initial_state
        return entry
    except SenseError as exc:
        raise LensError(exc.code, exc.message) from exc
    finally:
        try:
            if session is not None:
                session.close()
        finally:
            reader.close()
