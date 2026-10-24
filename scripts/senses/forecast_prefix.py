"""Training-only, source-bound recorded prefixes for reusable temporal instruments.

Split groups remain intact. This is a separate future-target contract, not an
exception to Store.context's label-sibling guards. O(source bytes + selected
prefix rows); no image decoding, fitting, test evaluation or physical-time claim.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from senses.contract import CHANNELS, absent, canonical, require, seal_record, validate_record
from senses.forecast import load_bundle, put, split_for, validate_plan
from senses.pipeline import file_hash, read_bound
from senses.views import masked
from senses.scope import check_source
from training_scope import POLICY_SHA256

SCHEMA = "ucr-training-prefix-v1"
WINDOW = 8
FILES = {"plan.json", "policy.json", "records.jsonl", "examples.jsonl", "training-targets.jsonl"}


def series(r):
    return r["locator"]["source"], r["locator"]["shard"]


def validate_prefix(current, past, plan):
    validate_record(current)
    check_source(current["locator"]["source"])
    start = current["locator"]["row"]
    require(type(start) is int and start >= 0 and len(past) <= WINDOW,
            "MNLM_PREFIX_ORDER", "invalid current position or prefix extent")
    require([r["locator"]["row"] for r in past] == list(range(max(0, start - WINDOW), start)),
            "MNLM_PREFIX_ORDER", "need the complete ordered strict prefix; never pad/truncate/substitute")
    signature = None
    for r in [current, *past]:
        validate_record(r)
        pos = r["locator"]["row"]
        clocks = [f for f in r["channels"]["structure"]["items"] if f["name"] == plan["ordinal_field"]]
        require(r["locator"]["span"] == [pos, pos + 1] and len(clocks) == 1
                and clocks[0]["type"] == "integer" and clocks[0]["unit"] == plan["ordinal_unit"]
                and clocks[0]["value"] == pos, "MNLM_PREFIX_CLOCK", "retained ordinal differs from locator")
        selected = [[f for f in r["channels"]["fields"]["items"] if f["name"] == name] for name in plan["fields"]]
        require(all(len(fs) == 1 and fs[0]["type"] in ("number", "integer") for fs in selected),
                "MNLM_PREFIX_FIELDS", "observed coordinates missing, duplicated or not numeric")
        actual = [[fs[0][k] for k in ("name", "type", "unit", "clock_role")] for fs in selected]
        require(signature is None or actual == signature, "MNLM_PREFIX_FIELDS", "coordinate types/units drift within prefix")
        signature = actual
    for r in past:
        require(series(r) == series(current) and r["unit"] == current["unit"]
                and r["locator"]["manifest_digest"] == current["locator"]["manifest_digest"]
                and r["atom_id"] != current["atom_id"],
                "MNLM_PREFIX_DOMAIN", "prefix crosses a series, group or source contract")
    require(current["locator"]["source"] == plan["source"]
            and split_for(current["unit"], plan["seed"]) == "train",
            "MNLM_PREFIX_SPLIT", "training consumer cannot open held-out groups")


def build(export, bundle, out):
    started = time.perf_counter()
    original, data = load_bundle(bundle)
    plan = data["plan.json"]
    chosen = [(r, label) for r, label in zip(data["inputs.jsonl"], data["labels.jsonl"])
              if label["split"] == "train"]
    require(len(chosen) == 239 and plan["rows"] == 23702, "MNLM_PREFIX_SCOPE", "recommission changed scope/cost")
    wanted = {}
    for current, _ in chosen:
        key = series(current)
        require(key not in wanted, "MNLM_PREFIX_SCOPE", "one target per series/group required")
        wanted[key] = current
    retained, prefixes, count = {}, {r["atom_id"]: [] for r, _ in chosen}, 0
    for r in read_bound(export, original["source_sha256"]):
        count += 1
        current = wanted.get(series(r))
        if current is None:
            continue
        pos, end = r["locator"]["row"], current["locator"]["row"]
        if max(0, end - WINDOW) <= pos <= end:
            validate_record(r)
            require(r["atom_id"] not in retained, "MNLM_PREFIX_DUPLICATE", "repeated source record")
            retained[r["atom_id"]] = r
            if pos < end:
                prefixes[current["atom_id"]].append(r)
            else:
                require(r == current, "MNLM_PREFIX_SOURCE", "selected input differs from authoritative export")
    require(count == plan["rows"] and len(retained) == 1734, "MNLM_PREFIX_SCOPE", "source population/prefix count changed")
    examples, targets = [], []
    for current, label in chosen:
        past = prefixes[current["atom_id"]]
        validate_prefix(current, past, plan)
        require(current["atom_id"] in retained and label["future_atom"] not in retained,
                "MNLM_PREFIX_TARGET", "future target entered the input store")
        examples.append({"current": current["atom_id"], "past": [r["atom_id"] for r in past],
                         "unit": current["unit"], "split": "train"})
        targets.append(label)
    policy = data["policies.json"]["recorded-future-state"]
    # Current visibility is unchanged. Historical readings expose only the
    # declared observed coordinates, not repeated instructions or annotations.
    payloads = {"plan.json": canonical(plan), "policy.json": canonical(policy),
                "records.jsonl": b"".join(canonical(retained[k]) + b"\n" for k in sorted(retained)),
                "examples.jsonl": b"".join(canonical(r) + b"\n" for r in examples),
                "training-targets.jsonl": b"".join(canonical(r) + b"\n" for r in targets)}
    out.mkdir(parents=True, exist_ok=False)
    files = {}
    for name, raw in payloads.items():
        put(out / name, raw)
        files[name] = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    manifest = {"schema": SCHEMA, "purpose": "train_only_recorded_order_diagnostic", "window": WINDOW,
                "source_sha256": original["source_sha256"], "forecast_manifest_sha256": file_hash(bundle / "manifest.json"),
                "training_scope_sha256": POLICY_SHA256, "files": files, "examples": len(examples),
                "records": len(retained), "past_records": sum(len(e["past"]) for e in examples),
                "absent_history": sum(not e["past"] for e in examples), "new_outcome_evaluation": False,
                "physical_causality_asserted": False, "stage3_admitted": False, "stage4_accepted": False,
                "code_sha256": file_hash(Path(__file__)), "retention_owner": "ChrisRoyse/MNLM#12",
                "wall_seconds": time.perf_counter() - started}
    put(out / "manifest.json", canonical(manifest))
    return manifest


class PrefixInputs:
    """A fitting consumer sees only bound training input records, never targets.

    The supervised trainer loads training-targets.jsonl separately. It cannot
    request a held-out group, arbitrary history, or an unbound current record.
    Frozen machinery remains shared across sources; no source-name dispatch.
    """
    def __init__(self, root, expected_manifest_sha256):
        raw = (root / "manifest.json").read_bytes()
        require(hashlib.sha256(raw).hexdigest() == expected_manifest_sha256,
                "MNLM_PREFIX_DIGEST", "pin the complete training artifact before consuming it")
        m = json.loads(raw)
        require(m.get("schema") == SCHEMA and m.get("window") == WINDOW
                and m.get("purpose") == "train_only_recorded_order_diagnostic"
                and m.get("training_scope_sha256") == POLICY_SHA256
                and all(m.get(k) is False for k in ("new_outcome_evaluation", "physical_causality_asserted", "stage3_admitted", "stage4_accepted"))
                and set(m["files"]) == FILES, "MNLM_PREFIX_MANIFEST", "unsupported prefix/claim contract")
        data = {}
        for name in sorted(FILES - {"training-targets.jsonl"}):
            raw = (root / name).read_bytes()
            entry = m["files"][name]
            require(len(raw) == entry["bytes"] and hashlib.sha256(raw).hexdigest() == entry["sha256"],
                    "MNLM_PREFIX_DIGEST", name)
            data[name] = [json.loads(line) for line in raw.splitlines()] if name.endswith(".jsonl") else json.loads(raw)
        self.plan = validate_plan(data["plan.json"])
        require(self.plan["export_sha256"] == m["source_sha256"], "MNLM_PREFIX_SOURCE", "source binding changed")
        self.records = {r["atom_id"]: r for r in data["records.jsonl"]}
        self.examples = {e["current"]: e for e in data["examples.jsonl"]}
        require(len(self.records) == len(data["records.jsonl"]) == m["records"]
                and len(self.examples) == len(data["examples.jsonl"]) == m["examples"],
                "MNLM_PREFIX_COUNT", "duplicate or missing records/examples")
        self.policy = data["policy.json"]
        require(self.policy["allow_channels"] == self.plan["allow_channels"]
                and self.policy["question"] == "recorded-future-state",
                "MNLM_PREFIX_POLICY", "current-input visibility changed")
        from senses.forecast import policy_for
        # Original policy carries all 416 split groups; the fixed byte hash is
        # pinned by the caller, and each current group must remain excluded.
        expected_policy = policy_for(self.policy["blocked_units"])
        require(self.policy == expected_policy, "MNLM_PREFIX_POLICY", "baseline label/visibility guard changed")
        used, groups = set(), []
        for e in self.examples.values():
            require(set(e) == {"current", "past", "unit", "split"} and e["split"] == "train"
                    and isinstance(e["past"], list), "MNLM_PREFIX_EXAMPLE", "explicit current/past/train roles required")
            ids = [e["current"], *e["past"]]
            require(all(a in self.records for a in ids), "MNLM_PREFIX_NOT_FOUND", "missing input record")
            current = self.records[e["current"]]
            require(e["unit"] == current["unit"] and current["unit"] in self.policy["blocked_units"],
                    "MNLM_PREFIX_SPLIT", "split unit or original label exclusion changed")
            validate_prefix(current, [self.records[a] for a in e["past"]], self.plan)
            used.update(ids)
            groups.append(e["unit"])
        require(len(set(groups)) == len(groups) and used == set(self.records)
                and sum(len(e["past"]) for e in self.examples.values()) == m["past_records"]
                and sum(not e["past"] for e in self.examples.values()) == m["absent_history"],
                "MNLM_PREFIX_COUNT", "orphan, repeated group or changed history extent")
        self.manifest, self.identity = m, expected_manifest_sha256
        self.reader_code_sha256 = file_hash(Path(__file__))

    def read(self, atom):
        require(atom in self.examples, "MNLM_PREFIX_NOT_FOUND", "only frozen training targets can open prefixes")
        e = self.examples[atom]
        validate_prefix(self.records[atom], [self.records[a] for a in e["past"]], self.plan)
        current = masked(self.records[atom], self.policy)
        past = []
        for key in e["past"]:
            raw = self.records[key]
            # Keep exact coordinates and their field identities; do not feed
            # provenance IDs, instructions, commands, reward or terminal flags.
            r = {**raw, "channels": {c: absent("temporal_coordinate_policy_excluded") for c in CHANNELS}}
            values = copy.deepcopy([f for f in raw["channels"]["fields"]["items"] if f["name"] in self.plan["fields"]])
            require(len(values) == len(self.plan["fields"]), "MNLM_PREFIX_FIELDS", "missing observed history coordinates")
            r["channels"]["fields"] = {"state": "present", "reason": None, "items": values}
            past.append(seal_record(r))
        return current, past, {"manifest_sha256": self.identity, "current": atom, "past": list(e["past"]),
                               "visible_past": [r["atom_id"] for r in past], "reader_code_sha256": self.reader_code_sha256,
                               "scope": "recorded_order; not physical causality"}


def training_targets(root, inputs):
    """Explicit supervised side; the sensing-input reader never opens this file."""
    entry = inputs.manifest["files"]["training-targets.jsonl"]
    raw = (root / "training-targets.jsonl").read_bytes()
    require(len(raw) == entry["bytes"] and hashlib.sha256(raw).hexdigest() == entry["sha256"],
            "MNLM_PREFIX_DIGEST", "training-targets.jsonl")
    rows = [json.loads(line) for line in raw.splitlines()]
    require([r["atom"] for r in rows] == list(inputs.examples), "MNLM_PREFIX_TARGET", "supervised target alignment changed")
    for row in rows:
        current = inputs.records[row["atom"]]
        require(row["split"] == "train" and row["unit"] == current["unit"]
                and row["future_atom"] not in inputs.records
                and row["current_locator"] == current["locator"]
                and row["future_locator"]["row"] - current["locator"]["row"] == inputs.plan["horizon"]
                and all(row["future_locator"][k] == current["locator"][k] for k in ("source", "shard")),
                "MNLM_PREFIX_TARGET", "input/target separation or future order changed")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("export", "bundle", "out"):
        ap.add_argument("--" + name, type=Path, required=True)
    a = ap.parse_args()
    result = build(a.export, a.bundle, a.out)
    print(json.dumps({k: result[k] for k in ("examples", "records", "past_records", "absent_history", "wall_seconds")}))
