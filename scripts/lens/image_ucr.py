"""Source-neutral raw-image measurement with a mask-bypass refusal boundary."""
from pathlib import Path

import numpy as np

from senses.contract import SenseError, canonical, digest, require
from senses.blobs import Blobs
from lenscommon import LensError, write_json
from slotio import TokenSpool
from universal import ChannelReader
import slot_binding

SPEC = {"name": "ucr-rgb-patches-v1", "module": "image_ucr", "kind": "encoder", "inputs": "ucr",
        "channel": "perception", "perspective": "perception", "role": "universal_instrument",
        "group": "universal", "sim": "maxsim", "sparse": False}


def visible_image(reader, policy, row):
    record, _, binding = reader.read(row, policy)
    images = [i for i in record["channels"]["perception"]["items"] if i["modality"] == "image"]
    if images:
        require(not policy["label_aliases"], "MNLM_IMAGE_MASK_UNSUPPORTED",
                "exclude raw perception or commission a pixel-level visibility policy; text masking cannot hide image labels")
        retained = reader.store.get(binding["atom"], reader.generation)
        original = [i for i in retained["channels"]["perception"]["items"] if i["modality"] == "image"]
        require(canonical(images) == canonical(original), "MNLM_IMAGE_MASK_UNSUPPORTED",
                "reopening image bytes would bypass a masked measurement")
    return record, binding


def measure_image_ucr(spec, fams, sdir, pins, store_path, policies_path, blobs_path, blobs_sha256):
    from senses import pixels
    reader = ChannelReader(store_path, policies_path)
    try:
        require(spec == SPEC, "MNLM_LENS_FROZEN_VIOLATION", "unsupported image instrument")
        identity = {**pixels.contract(), **SPEC}
        require(spec["name"] not in pins or pins[spec["name"]] == identity["weights_sha256"],
                "MNLM_LENS_FROZEN_VIOLATION", "pixel encoder drift")
        blobs = Blobs(blobs_path, blobs_sha256)
        contract = {**identity, "ucr_generation": reader.generation, "policies_sha256": reader.policy_hash,
                    "blobs_manifest_sha256": blobs.sha256}
        entry = {"contract": contract, "lens_id": identity["lens_id"], "families": {},
                 "probe": {"n": 0, "present": 0, "byte_identical": True}}
        for family, rows in fams.items():
            policy = reader.policy(family)
            path = Path(sdir) / family / (spec["name"] + ".npz")
            spool = TokenSpool(path, pixels.DIM)
            bindings, regions, windows = [], [], []
            try:
                for row in rows:
                    record, binding = visible_image(reader, policy, row)
                    slot = pixels.measure(record, blobs)
                    t = slot["tokens"]
                    if len(t) and entry["probe"]["present"] < 64:
                        again = pixels.measure(record, blobs)
                        require(t.tobytes() == again["tokens"].tobytes() and slot["regions"] == again["regions"],
                                "MNLM_LENS_NONDETERMINISTIC", spec["name"])
                        entry["probe"]["n"] += 1
                        entry["probe"]["present"] += 1
                    spool.add(t, np.asarray([0, len(t)], np.int64))
                    bindings.append(binding)
                    regions.append(slot["regions"])
                    windows.append(len(slot["regions"]))
                dg, sizes = spool.finish(np.asarray(windows, np.int64))
            finally:
                spool.fh.close()
                spool.raw.unlink(missing_ok=True)
            provenance = {"generation": reader.generation, "policy": policy, "bindings": bindings,
                          "image_regions": regions, "blobs_manifest_sha256": blobs.sha256}
            write_json(path.with_suffix(".views.json"), provenance)
            ent = {"slot": str(path), "digest": dg, "dims": pixels.DIM, **sizes,
                   "ucr_generation": reader.generation, "view_bindings_sha256": digest(bindings),
                   "window_provenance_sha256": digest(provenance), "policy_sha256": digest(policy)}
            slot_binding.seal(path, contract, ent)
            entry["families"][family] = ent
        if not entry["probe"]["present"]:
            entry.update(error="MNLM_UCR_CHANNEL_UNOBSERVED", detail="No nonempty raw-image observation exercised this encoder.")
        return entry
    except SenseError as exc:
        raise LensError(exc.code, exc.message) from exc
    finally:
        reader.close()
