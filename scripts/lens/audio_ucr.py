"""UCR-to-audio measurement. One source-neutral perception instrument (#15).

Policy, source generation, raw-blob manifest, temporal provenance and native slot
bytes are bound separately. Registration does not constitute information admission.
"""
from pathlib import Path

import blake3
import numpy as np

from audio_encoder import AudioEncoder, SPEC, spans
from lenscommon import LensError, write_json
from senses.blobs import Blobs
from senses.contract import SenseError, canonical, digest, require
from slotio import TokenSpool
import slot_binding
from universal import ChannelReader


def visible_audio(reader, policy, row):
    record, _, binding = reader.read(row, policy)
    items = record["channels"]["perception"]["items"]
    audio = [p for p in items if p["modality"] == "audio"]
    if audio:
        # A label alias cannot be redacted in a waveform by a string replacement.
        # Reopening raw sound after dropping a derived field would restore that
        # forbidden information. Require unmodified audio observations or refuse.
        require(not policy["label_aliases"], "MNLM_AUDIO_MASK_UNSUPPORTED", "exclude raw perception or supply an independently reviewed audio mask")
        retained = reader.store.get(binding["atom"], reader.generation)
        original = [p for p in retained["channels"]["perception"]["items"] if p["modality"] == "audio"]
        require(canonical(audio) == canonical(original), "MNLM_AUDIO_MASK_UNSUPPORTED", "raw audio would bypass masked perception measurements")
    return spans(record), binding


def measure_audio_ucr(spec, fams, sdir, pins, store_path, policies_path, blobs_path, blobs_sha256):
    reader, lens = ChannelReader(store_path, policies_path), None
    try:
        policies = {f: reader.policy(f) for f in fams}
        probe = None
        for fam, rows in fams.items():
            for row in rows:
                regions, _ = visible_audio(reader, policies[fam], row)
                if regions:
                    probe = regions
                    break
            if probe:
                break
        require(probe, "MNLM_UCR_CHANNEL_UNOBSERVED", "audio absent in the declared calibration scope")
        blobs = Blobs(blobs_path, blobs_sha256)
        lens = AudioEncoder(spec)
        identity = lens.identity
        require(spec["name"] not in pins or pins[spec["name"]] == identity["weights_sha256"],
                "MNLM_LENS_FROZEN_VIOLATION", spec["name"])
        contract = {**identity, "ucr_generation": reader.generation, "policies_sha256": reader.policy_hash,
                    "blobs_manifest_sha256": blobs.sha256}
        before = lens.state_digest()
        a, aw = lens.embed(probe, blobs)
        b, bw = lens.embed(probe, blobs)
        require(a.tobytes() == b.tobytes() and aw == bw, "MNLM_LENS_NONDETERMINISTIC", spec["name"])
        entry = {"contract": contract, "lens_id": blake3.blake3(canonical(identity)).hexdigest()[:32], "families": {},
                 "probe": {"n": 1, "present": 1, "byte_identical": True, "windows": len(aw), "tokens": len(a)}}
        del a, b
        for fam, rows in fams.items():
            path = Path(sdir) / fam / (spec["name"] + ".npz")
            spool = TokenSpool(path, lens.dim)
            bindings, segments, counts = [], [], []
            try:
                for row in rows:
                    regions, binding = visible_audio(reader, policies[fam], row)
                    tokens, windows = lens.embed(regions, blobs)
                    spool.add(tokens, np.asarray([0, len(tokens)], np.int64))
                    bindings.append(binding)
                    segments.append(windows)
                    counts.append(len(windows))
                dg, sizes = spool.finish(np.asarray(counts, np.int64))
            finally:
                spool.fh.close()
                spool.raw.unlink(missing_ok=True)
            provenance = {"generation": reader.generation, "policy": policies[fam], "bindings": bindings,
                          "windows": segments, "blobs_manifest_sha256": blobs.sha256}
            write_json(path.with_suffix(".views.json"), provenance)
            ent = {"slot": str(path), "digest": dg, "dims": lens.dim, **sizes, "ucr_generation": reader.generation,
                   "view_bindings_sha256": digest(bindings), "window_provenance_sha256": digest(provenance),
                   "policy_sha256": digest(policies[fam]), "windows_max": max(counts, default=0),
                   "windows_gt1": sum(c > 1 for c in counts)}
            slot_binding.seal(path, contract, ent)
            entry["families"][fam] = ent
        require(before == lens.state_digest() and not any(p.requires_grad for p in lens.model.parameters()),
                "MNLM_LENS_FROZEN_VIOLATION", "audio encoder changed during measurement")
        entry["frozen_encoder_state_sha256"] = before
        return entry
    except SenseError as exc:
        raise LensError(exc.code, exc.message) from exc
    finally:
        reader.close()
        if lens is not None:
            import torch
            del lens
            torch.cuda.empty_cache()
