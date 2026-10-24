"""Stage 3 consumer of UCR: masked views -> native Multi slots -> existing assay.

No source branches, new model per source, pooled panel or label-fitting here.
The snapshot and mask policy are bound into the measurement contract.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from senses.contract import SenseError, digest, require
from senses.instruments import DIM, contracts, measure, slot_digest
from senses.store import Store
from senses.views import validate_policy, view
from lenscommon import LensError, write_json
from slotio import TokenSpool
import slot_binding


def specs():
    return [{**c, "group": "universal", "module": "universal", "sparse": False} for c in contracts()]


class ChannelReader:
    """One read snapshot and policy parse per instrument, not per atom (PC-07)."""
    def __init__(self, store_path, policies_path, generation=None):
        require(store_path and policies_path, "MNLM_UCR_REQUIRED", "provide a channel store and reviewed policies")
        raw = Path(policies_path).read_bytes()
        self.policies = json.loads(raw)
        self.policy_hash = hashlib.sha256(raw).hexdigest()
        self.store = Store(store_path)
        self.store.db.execute("BEGIN")
        self.generation = self.store.head if generation is None else generation
        if generation is not None and not self.store.db.execute("SELECT 1 FROM generations WHERE hash=?", (generation,)).fetchone():
            self.close()
            raise SenseError("MNLM_UCR_NOT_IN_GENERATION", "restore the exact frozen observation generation")
        if self.generation is None:
            self.close()
            raise SenseError("MNLM_UCR_EMPTY", "ingest channel records first")

    def close(self):
        self.store.close()

    def policy(self, family):
        require(family in self.policies, "MNLM_UCR_POLICY", f"no reviewed visibility policy for {family}")
        return validate_policy(self.policies[family], family)

    def read(self, row, policy):
        from senses.scope import check_source
        check_source(row["source"])
        span = row.get("ucr_span", row.get("span"))
        r = self.store.find(row["source"], row["shard"], row["row"], span, self.generation)
        return view(self.store, r, policy, self.generation, surface=row["text"])


def measure_universal(spec, fams, sdir, pins, store_path, policies_path):
    require(store_path and policies_path, "MNLM_UCR_REQUIRED", "universal measurement needs --ucr-store and --ucr-policies")
    reader = ChannelReader(store_path, policies_path)
    try:
        generation = reader.generation
        contract = {**spec, "shape": f"Multi(var,{DIM})", "modality": "record", "ucr_generation": generation,
                    "policies_sha256": reader.policy_hash,
                    "context_code_sha256": hashlib.sha256((Path(__file__).parents[1] / "senses" / "store.py").read_bytes()).hexdigest(),
                    "view_code_sha256": hashlib.sha256((Path(__file__).parents[1] / "senses" / "views.py").read_bytes()).hexdigest()}
        require(spec["name"] not in pins or pins[spec["name"]] == contract["weights_sha256"], "MNLM_LENS_FROZEN_VIOLATION", spec["name"])
        entry = {"contract": contract, "lens_id": spec["lens_id"], "families": {},
                 "probe": {"n": 0, "present": 0, "byte_identical": True, "batch_invariance_max_abs": 0.0}}
        for fam, rows in fams.items():
            policy = reader.policy(fam)
            path = sdir / fam / (spec["name"] + ".npz")
            spool = TokenSpool(path, DIM)
            bindings = []
            try:
                for i, row in enumerate(rows):
                    v, contexts, binding = reader.read(row, policy)
                    slot = measure(v, spec["channel"], contexts.get(spec["channel"]))
                    if i < 64 or (len(slot["tokens"]) and entry["probe"]["present"] < 64):
                        repeat = measure(v, spec["channel"], contexts.get(spec["channel"]))
                        require(slot_digest(slot) == slot_digest(repeat), "MNLM_LENS_NONDETERMINISTIC", spec["name"])
                        entry["probe"]["n"] += 1
                        entry["probe"]["present"] += bool(len(slot["tokens"]))
                    t = slot["tokens"]
                    spool.add(t, np.asarray([0, len(t)], np.int64))
                    bindings.append(binding)
                dg, counts = spool.finish(np.ones(len(rows), np.int64))
            finally:
                spool.fh.close()
                spool.raw.unlink(missing_ok=True)
            ent = {"slot": str(path), "digest": dg, "dims": DIM, **counts,
                   "ucr_generation": generation, "view_bindings_sha256": digest(bindings), "policy_sha256": digest(policy)}
            write_json(path.with_suffix(".views.json"), {"generation": generation, "policy": policy, "bindings": bindings})
            slot_binding.seal(path, contract, ent)
            entry["families"][fam] = ent
        if not entry["probe"]["present"]:
            entry["error"] = "MNLM_UCR_CHANNEL_UNOBSERVED"
            entry["detail"] = "Absence was recorded, but no nonempty observation exercised this instrument; supply that channel before registration."
        return entry
    except SenseError as exc:
        raise LensError(exc.code, exc.message) from exc
    finally:
        reader.close()


def coverage_report(registry, reports):
    """Per-row coverage (instruments present per atom; each instrument is counted, channels are reported) and one roster."""
    names = [n for n, e in registry["lenses"].items() if e.get("spec", {}).get("role") == "universal_instrument" and not e.get("error")]
    questions = {}
    for q, report in reports.items():
        n = report["n"]
        present, channels = {}, {}
        for nm in names:
            e = registry["lenses"][nm]
            if q not in e.get("families", {}):
                continue
            with np.load(e["families"][q]["slot"], allow_pickle=False) as z:
                absent = z["absent"]
                require(absent.dtype == np.bool_ and absent.shape == (n,), "MNLM_UCR_COVERAGE", "boolean absence vector required")
                import blake3
                require(blake3.blake3(absent.tobytes()).hexdigest() == e["families"][q].get("absent_blake3"),
                        "MNLM_SLOT_STALE", "coverage absence differs from measured registry")
                mask = ~absent
            require(mask.shape == (n,), "MNLM_UCR_COVERAGE", q)
            present[nm] = mask
            channels.setdefault(e["spec"]["perspective"], np.zeros(n, bool))
            channels[e["spec"]["perspective"]] |= mask
        per_atom = np.sum(list(present.values()), axis=0) if present else np.zeros(n, int)
        questions[q] = {"n": n, "measured_channels": sorted(channels), "available_channels": sorted(c for c, m in channels.items() if m.any()),
                        "available_instruments": sorted(i for i, m in present.items() if m.any()),
                        "per_atom_available_histogram": {str(k): int(v) for k, v in zip(*np.unique(per_atom, return_counts=True))},
                        "atoms_ge5_available": int((per_atom >= 5).sum()), "atoms_ge5_admitted": 0,
                        "served": False, "sufficient": False}
    return {"contract": "universal-panel-v1", "candidate_roster": sorted(names), "admitted": [],
            "questions": questions, "served": False, "status": "MNLM_SHANNON_CONDITIONAL_UNMEASURED"}
