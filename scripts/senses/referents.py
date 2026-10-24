"""Explicit referent mappings, including unresolved entries; no fuzzy joining.

Mappings are an input artifact, not a discovered truth. The caller pins its digest;
each resolved mapping must name its evidence and exact source identity. Gold
held-out mappings can be checked without teaching the index a model per source.
"""
from __future__ import annotations

from .adapters import exact_ref
from .contract import digest, require


class ReferentMap:
    def __init__(self, artifact, expected_digest):
        require(digest(artifact) == expected_digest and artifact.get("schema") == "ucr-referents-v1", "MNLM_UCR_REFERENT_MAP", "mapping digest/schema mismatch")
        self.digest, self.rows = expected_digest, {}
        for r in artifact["rows"]:
            require(set(r) == {"source", "shard", "row", "status", "key", "evidence"}, "MNLM_UCR_REFERENT_MAP", "mapping row keys")
            key = (r["source"], r["shard"], r["row"])
            require(key not in self.rows and r["status"] in ("resolved", "unresolved"), "MNLM_UCR_REFERENT_MAP", "duplicate identity or unknown status")
            require((r["status"] == "resolved" and isinstance(r["key"], str) and bool(r["key"]) and bool(r["evidence"]))
                    or (r["status"] == "unresolved" and r["key"] is None), "MNLM_UCR_REFERENT_MAP", "resolution requires evidence")
            self.rows[key] = r

    def apply(self, prepared):
        key = (prepared["source"], prepared["shard"], prepared["rec"]["row"])
        row = self.rows.get(key)
        if row and row["status"] == "resolved":
            prepared["refs"].append(exact_ref(row["key"], {"mapping_digest": self.digest, "evidence": row["evidence"]}, "attested_mapping"))
        return prepared

    def evaluate(self, held_out):
        require(bool(held_out), "MNLM_UCR_REFERENT_GOLD", "held-out gold required")
        correct = unresolved = wrong = 0
        for g in held_out:
            row = self.rows.get((g["source"], g["shard"], g["row"]))
            if row is None or row["status"] == "unresolved":
                unresolved += 1
            elif row["key"] == g["key"]:
                correct += 1
            else:
                wrong += 1
        return {"n": len(held_out), "correct": correct, "wrong": wrong, "unresolved": unresolved,
                "precision": correct / (correct + wrong) if correct + wrong else None,
                "coverage": (correct + wrong) / len(held_out)}
