"""Read accepted native engineering shards once; join tiers by exact observation identity."""
import json

from senses.contract import canonical, digest, require
from production.publication import Publication
from production.native_rows import mapped_rows, one, joined, value_hash
from model.contracts import Instrument, validate_bank
from model.native import from_arrays


def read_publication(root, groups, config):
    """groups explicitly names aligned shard keys, never an implicit corpus scan.

    Each group supplies one observation set across tiers; groups are appended
    along the batch axis. No two instruments' values are ever concatenated.
    """
    require(isinstance(groups, list) and groups and all(isinstance(g, list) and g for g in groups),
            "MNLM_MODEL_SCOPE", "explicit nonempty shard groups required")
    require(len({key for group in groups for key in group}) == sum(map(len, groups)),
            "MNLM_MODEL_SCOPE", "duplicate shard key")
    pub = Publication(root)
    specs, pieces, all_rows, witnesses = {}, {}, [], []
    padded_elements = 0
    try:
        for group in groups:
            rows, slots = None, {}
            for key in group:
                shard, _ = pub.read(key)
                current = [json.loads(line) for line in pub.blob(shard["rows_sha256"]).read_bytes().splitlines()]
                require(rows is None or canonical(rows) == canonical(current), "MNLM_MODEL_ROWS",
                        "tiers must describe exactly the same ordered observations")
                rows = current
                for name, slot in shard["slots"].items():
                    require(name not in slots, "MNLM_MODEL_DUPLICATE", "instrument appears twice in a group")
                    spec = Instrument.from_slot(name, slot)
                    require(name not in specs or specs[name] == spec, "MNLM_MODEL_IDENTITY",
                            "instrument changed between observation batches")
                    specs[name], slots[name] = spec, slot
                witnesses.append({"key": key, "shard_sha256": digest(shard), "rows_sha256": shard["rows_sha256"],
                                  "slot_blobs": {n: s["sha256"] for n, s in shard["slots"].items()}})
            if all_rows:
                require(set(slots) == set(pieces), "MNLM_MODEL_ROSTER", "all groups must declare the same instrument bank")
            all_rows.extend(rows)
            for name, slot in slots.items():
                spec = specs[name]
                with mapped_rows(pub.blob(slot["sha256"])) as arrays:
                    target = pieces.setdefault(name, [])
                    # Bound retained native copies before allocation as well as
                    # the padded conversion. No unbounded full-corpus loader.
                    entries = arrays["X"].size if spec.layout == "dense" else arrays["tokens" if spec.layout == "multi" else "data"].size
                    padded_elements += entries
                    require(padded_elements <= config.max_native_elements, "MNLM_MODEL_INPUT_BUDGET",
                            "selected native payload exceeds fixed batch budget")
                    target.extend(one(arrays, i) for i in range(len(rows)))
        ordered = tuple(specs[n] for n in sorted(specs))
        validate_bank(ordered)
        identities = tuple(r["atom"] for r in all_rows)
        require(len(set(identities)) == len(identities), "MNLM_MODEL_ROWS", "repeated observation across groups")
        inputs, hashes, total = {}, {}, 0
        # Whole-bank padding preflight precedes every padded allocation.
        for spec in ordered:
            part = pieces[spec.name]
            width = 1 if spec.layout == "dense" else max(1, max(len(r["tokens" if spec.layout == "multi" else "data"]) for r in part))
            require(width <= config.max_native_tokens, "MNLM_MODEL_INPUT_BUDGET", "native sequence exceeds token budget")
            total += len(identities) * width * (1 if spec.layout == "sparse" else spec.dims)
        require(total <= config.max_native_elements, "MNLM_MODEL_INPUT_BUDGET", "padded bank exceeds fixed batch budget before allocation")
        for spec in ordered:
            arrays = joined(pieces[spec.name], spec.layout, spec.dims)
            inputs[spec.name] = from_arrays(spec, arrays, identities, config)
            hashes[spec.name] = value_hash(arrays)
        receipt = {"schema": "model-native-input-v1", "publication_plan_sha256": pub.plan_sha256,
                   "publication_state": pub.state(), "shards": witnesses, "native_value_hashes": hashes,
                   "rows_sha256": digest(all_rows), "rows": len(all_rows), "instruments": len(specs),
                   "native_elements": padded_elements, "padded_elements": total, "stage4_accepted": False}
        return ordered, inputs, all_rows, receipt
    finally:
        pub.close()
