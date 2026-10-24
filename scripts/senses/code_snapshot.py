"""Byte-bound file transitions -> existing UCR channels, without executing code.

One immutable snapshot pair is read once. Older versions are history of the same
file, never another task's solution. Static file relations and local patch events
are not inferred runtime behavior. All slots remain diagnostic until admission.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from senses.adapters import exact_ref, relation
from senses.code_syntax import contract as syntax_contract, syntax_fields
from senses.contract import (CHANNELS, VERSION, SenseError, absent, digest, field,
                             is_sha256, present, require, seal_record, validate_record)
from senses.pipeline import file_hash, write_records
from senses.scope import check_source
from training_scope import POLICY_SHA256
from senses.trajectory_population import publish, read_pinned

CODE = "MNLM_CODE_SNAPSHOT"
CUTOFF = "submitted_patch_before_external_evaluation"
KEYS = {"schema", "cutoff", "unit", "source", "source_binding", "model", "licence", "snapshots",
        "changed", "patch_sha256", "order", "clock", "evidence", "admitted"}
MAX_FILES, MAX_BYTES = 128, 8 << 20


def relative(name):
    require(isinstance(name, str) and name and "\\" not in name and ":" not in name,
            CODE, "portable relative file path required")
    p = PurePosixPath(name)
    require(not p.is_absolute() and ".." not in p.parts and str(p) == name, CODE, "unsafe snapshot path")
    return name


def validate(m):
    require(isinstance(m, dict) and set(m) == KEYS and m["schema"] == "code-snapshot-transition-v1"
            and m["cutoff"] == CUTOFF and m["order"] == ["base", "submitted"]
            and m["clock"] == "declared_snapshot_order_not_wall_time" and m["admitted"] is False,
            CODE, "explicit pre-evaluation transition required; no future snapshot or admission")
    check_source(m["source"])
    require(all(isinstance(m[k], str) and m[k] for k in ("source", "model", "licence", "evidence"))
            and is_sha256(m["unit"]) and is_sha256(m["patch_sha256"]), CODE, "provenance and unit required")
    require(set(m["snapshots"]) == {"base", "submitted"}, CODE, "exactly two ordered snapshots required")
    total = 0
    for inv in m["snapshots"].values():
        require(isinstance(inv, dict) and 0 < len(inv) <= MAX_FILES, CODE, "bounded complete inventory required")
        for name, b in inv.items():
            relative(name)
            require(isinstance(b, dict) and set(b) == {"bytes", "sha256"} and is_sha256(b["sha256"])
                    and type(b["bytes"]) is int and b["bytes"] >= 0, CODE, "exact file binding required")
            total += b["bytes"]
    require(total <= MAX_BYTES, CODE, "snapshot byte budget exceeded; no truncation")
    old, new = (m["snapshots"][n] for n in m["order"])
    require(not old.keys() - new.keys() and m["changed"] == sorted(p for p in new if old.get(p) != new[p]),
            CODE, "file transition inventory differs")


def read_files(root, m):
    """Exact inventory before reading; links and undeclared extras fail closed."""
    result = {}
    for phase in m["order"]:
        folder = root / phase
        require(folder.is_dir() and not folder.is_symlink(), CODE, "missing/linked snapshot")
        paths = list(folder.rglob("*"))
        require(all(not p.is_symlink() and not p.is_junction() and p.resolve().is_relative_to(folder.resolve())
                    for p in paths), CODE, "snapshot links or escaped paths refused")
        actual = {p.relative_to(folder).as_posix() for p in paths if p.is_file()}
        require(actual == m["snapshots"][phase].keys(), CODE, "missing or undeclared snapshot file")
        data = {}
        for name, b in sorted(m["snapshots"][phase].items()):
            p = folder / name
            require(p.stat().st_size == b["bytes"], CODE, "snapshot size changed")
            raw = p.read_bytes()
            require(hashlib.sha256(raw).hexdigest() == b["sha256"], CODE, "snapshot bytes changed")
            data[name] = raw
        result[phase] = data
    patch = root / "submitted.patch"
    require(not patch.is_symlink() and file_hash(patch) == m["patch_sha256"], CODE, "patch binding changed")
    return result


def record(m, manifest_sha, name, raw, phase, previous=None):
    # No newline conversion, lossy decode, source execution or implicit language guess.
    text = raw.decode("utf-8")
    ordinal = m["order"].index(phase)
    language = "python" if PurePosixPath(name).suffix in (".py", ".pyi") else "unsupported"
    r = {"schema": VERSION, "locator": {"source": m["source"], "shard": phase + "/" + name,
         "row": ordinal, "span": [0, len(raw)], "record_digest": hashlib.sha256(raw).hexdigest(),
         "manifest_digest": manifest_sha}, "unit": m["unit"],
         "referents": [exact_ref(digest([m["unit"], name]), "source-bound same file path")],
         "channels": {c: absent("not_observed") for c in CHANNELS}}
    ch = r["channels"]
    ch["surface"] = present([{"text": text}])
    ch["fields"] = present([field("file.path", name, "reference"), field("file.bytes", len(raw), "integer", "byte"),
                            field("snapshot.ordinal", ordinal, "integer", "sequence_index")])
    ch["structure"] = present([field("kind", "source_file", "category"), field("span_unit", "utf8_byte", "category"),
                               *syntax_fields(text, language)])
    ch["relations"] = present([relation("file_in_snapshot", digest([manifest_sha, phase]), "snapshot inventory")])
    ch["provenance"] = present([field("source", m["source"], "reference"), field("licence", m["licence"], "category"),
                                field("evidence", "pinned_base" if ordinal == 0 else "submitted_patch_applied_locally", "category"),
                                field("model", m["model"], "category"), field("snapshot.binding", manifest_sha, "reference")])
    if phase == "submitted" and name in m["changed"]:
        before = m["snapshots"]["base"].get(name)
        ch["behaviour"] = present([{"event": "file_patch_applied", "mode": "observed",
            "evidence": {"patch_sha256": m["patch_sha256"], "file": name, "scope": m["evidence"]},
            "measurements": [field("file.created", before is None, "boolean"),
                             field("file.before_bytes", before["bytes"] if before else 0, "integer", "byte"),
                             field("file.after_bytes", len(raw), "integer", "byte")]}])
    if previous is not None:
        validate_record(previous)
        require(phase == "submitted" and previous["unit"] == m["unit"]
                and previous["locator"]["shard"] == "base/" + name
                and previous["locator"]["manifest_digest"] == manifest_sha
                and previous["locator"]["record_digest"] == m["snapshots"]["base"][name]["sha256"],
                CODE, "history must be this exact file's pinned earlier version")
        ch["history"] = present([{"atom_id": previous["atom_id"], "unit": previous["unit"],
                                  "source": previous["locator"]["source"], "observed_at": None}])
        ch["relations"]["items"].append(relation("previous_file_version", previous["atom_id"], "declared snapshot order"))
    return seal_record(r)


class SnapshotInputs:
    """Same native-reader protocol as trajectory/physical prefixes, no new lens."""

    def __init__(self, root, manifest_sha256):
        self.root = Path(root)
        self.manifest = read_pinned(self.root / "snapshot.json", manifest_sha256)
        validate(self.manifest)
        files = read_files(self.root, self.manifest)
        self.reader_code_sha256 = {"code_snapshot.py": file_hash(Path(__file__)),
                                   "code_syntax.py": syntax_contract()["code_sha256"],
                                   "scope.py": file_hash(Path(__file__).with_name("scope.py"))}
        self.identity = digest({"snapshot_sha256": manifest_sha256, "syntax": syntax_contract(),
                                "reader_code_sha256": self.reader_code_sha256,
                                "training_scope_sha256": POLICY_SHA256})
        self.records, self.examples, self.bindings, self.omitted = {}, {}, {}, []
        self.seals = {}
        previous = {}
        for phase in self.manifest["order"]:
            for name, raw in files[phase].items():
                if not raw:
                    self.omitted.append({"phase": phase, "path": name, "reason": "empty_file_no_nonempty_atom"})
                    continue
                try:
                    r = record(self.manifest, manifest_sha256, name, raw, phase, previous.get(name))
                except UnicodeDecodeError:
                    self.omitted.append({"phase": phase, "path": name, "reason": "non_utf8_bytes_retained_in_snapshot"})
                    continue
                atom = r["atom_id"]
                self.records[atom] = r
                if phase == "base":
                    previous[name] = r
                else:
                    self.examples[atom] = name
                    self.bindings[atom] = {"current": atom, "unit": r["unit"], "cutoff": CUTOFF,
                        "visible_past": [previous[name]["atom_id"]] if name in previous else [],
                        "future_visible": False, "outcomes_visible": False, "snapshot_sha256": manifest_sha256}
                    self.seals[atom] = digest([name, self.bindings[atom]])

    def read(self, atom):
        require(atom in self.examples, CODE, "only submitted file observations are model targets")
        b = self.bindings[atom]
        require(digest([self.examples[atom], b]) == self.seals[atom], CODE, "loaded cutoff/history binding changed")
        require(all(k in self.records for k in [*b["visible_past"], atom]), CODE, "missing retained observation")
        records = [validate_record(self.records[k]) for k in [*b["visible_past"], atom]]
        require([r["atom_id"] for r in records] == [*b["visible_past"], atom]
                and all(r["unit"] == b["unit"] for r in records), CODE, "loaded observation identity changed")
        return deepcopy(records[-1]), deepcopy(records[:-1]), deepcopy(b)

    def export(self, output):
        output.mkdir(parents=True, exist_ok=False)
        manifest = write_records(output / "inputs.ucr.jsonl", self.records.values(),
                                 {"snapshot": self.identity, "cutoff": CUTOFF, "admitted": False})
        publish(output / "records.json", self.records)
        publish(output / "index.json", {"bindings": self.bindings, "omitted": self.omitted,
                                         "input_sha256": manifest["sha256"], "admitted": False})
        return manifest


def native_plan(reader):
    """Fixed commissioning envelope, checked before allocating native arrays."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lens"))
    from prefix_panel import census
    plan = census(reader)
    require(plan["projected_values_bytes"] <= 32 << 20
            and max(p["max_token_rows"] for p in plan["channels"].values()) <= 8192,
            "MNLM_CODE_SNAPSHOT_NATIVE_BUDGET", "choose smaller structural units; no pooling or truncation")
    return plan


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("snapshot", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--snapshot-sha256", required=True)
    p.add_argument("--native-panel", action="store_true", help="write all separate diagnostic native slots")
    a = p.parse_args()
    try:
        r = SnapshotInputs(a.snapshot, a.snapshot_sha256)
        m = r.export(a.output)
        if a.native_panel:
            sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lens"))
            from prefix_panel import publish as publish_panel
            plan = native_plan(r)
            publish(a.output / "native-plan.json", plan)
            publish_panel(r, plan, a.output / "panel")
        print(json.dumps({"records": m["count"], "current": len(r.examples), "omitted": r.omitted, "admitted": False}))
    except (SenseError, OSError, ValueError, KeyError, TypeError) as exc:
        err = exc if isinstance(exc, SenseError) else SenseError(CODE, str(exc))
        print(json.dumps(err.envelope()), file=sys.stderr)
        sys.exit(2)
