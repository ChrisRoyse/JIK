"""Source-neutral structural units from complete, pre-test code snapshots."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path, PurePosixPath
from types import MappingProxyType

from .adapters import exact_ref, relation
from .code_snapshot import CUTOFF, read_files, validate as validate_snapshot
from .code_units import contract, partition, summary
from .contract import (CHANNELS, VERSION, absent, canonical, digest, field, present,
                       require, seal_record, validate_record)
from .pipeline import file_hash, write_records
from .trajectory_population import publish, read_pinned
from training_scope import POLICY_SHA256

CODE = "MNLM_CODE_UNIT_READER"


def unit_record(m, manifest_sha, phase, name, raw, u, context_sha, previous):
    lo, hi = u["span"]
    ordinal = m["order"].index(phase)
    channels = {c: absent("not_observed") for c in CHANNELS}
    channels["surface"] = present([{"text": raw[lo:hi].decode("utf-8")}])
    channels["fields"] = present([field("file.path", name, "reference"),
        field("file.bytes", len(raw), "integer", "byte"), field("unit.bytes", hi-lo, "integer", "byte"),
        field("unit.start", lo, "integer", "utf8_byte"), field("unit.end", hi, "integer", "utf8_byte"),
        field("snapshot.ordinal", ordinal, "integer", "sequence_index")])
    channels["structure"] = present([field("kind", "source_code_unit", "category"),
        field("unit.kind", u["kind"], "category"), field("span_unit", "utf8_byte", "category"), *u["syntax"]])
    channels["relations"] = present([relation("unit_in_file", context_sha, "complete source/AST inventory"),
        *[relation("syntax_parent_outside_unit", digest([context_sha, n]), "full-file AST edge",
                   qualifiers=[field("parent.node", n, "integer", "node")]) for n in u["parents"]]])
    channels["provenance"] = present([field("source", m["source"], "reference"),
        field("licence", m["licence"], "category"), field("model", m["model"], "category"),
        field("evidence", "pinned_base" if ordinal == 0 else "submitted_patch_applied_locally", "category"),
        field("file.context", context_sha, "reference"), field("snapshot.binding", manifest_sha, "reference")])
    if phase == "submitted" and name in m["changed"]:
        # This event describes the file operation, not execution or inferred unit behavior.
        channels["behaviour"] = present([{"event": "file_patch_applied", "mode": "observed",
            "evidence": {"file": name, "patch_sha256": m["patch_sha256"], "scope": m["evidence"]},
            "measurements": [field("file.created", name not in m["snapshots"]["base"], "boolean")]}])
    if previous is not None:
        channels["history"] = present([{"atom_id": previous["atom_id"], "unit": m["unit"],
                                       "source": m["source"], "observed_at": None}])
        channels["relations"]["items"].append(relation("previous_source_unit", previous["atom_id"], u["history_reason"]))
    else:
        channels["history"] = absent(u["history_reason"] if phase == "submitted" else "no_previous_snapshot")
    return seal_record({"schema": VERSION, "locator": {"source": m["source"], "shard": phase+"/"+name,
        "row": ordinal, "span": [lo, hi], "record_digest": m["snapshots"][phase][name]["sha256"],
        "manifest_digest": manifest_sha}, "unit": m["unit"],
        "referents": [exact_ref(digest([m["unit"], name, u["key"]]), "unique source unit across snapshots")]
                     if u["key"] is not None else [], "channels": channels})


class CodeUnitInputs:
    def __init__(self, root, manifest_sha256):
        self.root = Path(root)
        self.manifest = read_pinned(self.root / "snapshot.json", manifest_sha256)
        validate_snapshot(self.manifest)
        files = read_files(self.root, self.manifest)
        self.reader_code_sha256 = {p.name: file_hash(p) for p in (
            Path(__file__), Path(__file__).with_name("code_units.py"), Path(__file__).with_name("code_snapshot.py"),
            Path(__file__).with_name("code_syntax.py"), Path(__file__).with_name("scope.py"))}
        self.identity = digest({"snapshot_sha256": manifest_sha256, "partition": contract(),
                                "reader_code_sha256": self.reader_code_sha256, "training_scope_sha256": POLICY_SHA256})
        self.records, self.examples, self.bindings, self.seals = {}, {}, {}, {}
        self.omitted, self.files = [], []
        cache, contexts, previous = {}, {}, {}
        for phase in self.manifest["order"]:
            for name, raw in files[phase].items():
                if not raw:
                    self.omitted.append({"phase": phase, "path": name, "reason": "empty_file_no_nonempty_atom"})
                    continue
                file_sha = self.manifest["snapshots"][phase][name]["sha256"]
                language = "python" if PurePosixPath(name).suffix in (".py", ".pyi") else "unsupported"
                cache_key = (file_sha, language)
                if cache_key not in cache:
                    try:
                        cache[cache_key] = partition(raw, language)
                    except UnicodeDecodeError:
                        self.omitted.append({"phase": phase, "path": name, "reason": "non_utf8_bytes_retained_in_snapshot"})
                        continue
                units, context = cache[cache_key]
                context_sha = digest(context)
                contexts[context_sha] = canonical(context)
                keyed = Counter(digest(u["key"]) for u in units if u["key"] is not None)
                file_atoms = []
                for original in units:
                    u = dict(original)
                    key = digest(u["key"]) if u["key"] is not None else None
                    if key is not None and keyed[key] != 1:
                        u.update(key=None, history_reason="ambiguous_unit_identity")
                        key = None
                    old = previous.get((name, key)) if key is not None else None
                    r = unit_record(self.manifest, manifest_sha256, phase, name, raw, u, context_sha, old)
                    atom = r["atom_id"]
                    self.records[atom] = r
                    file_atoms.append(atom)
                    if phase == "base" and key is not None:
                        previous[(name, key)] = r
                    if phase == "submitted":
                        self.examples[atom] = {"path": name, "span": u["span"], "context": context_sha}
                        self.bindings[atom] = {"current": atom, "unit": r["unit"], "cutoff": CUTOFF,
                            "visible_past": [old["atom_id"]] if old is not None else [],
                            "future_visible": False, "outcomes_visible": False, "snapshot_sha256": manifest_sha256,
                            "file_context_sha256": context_sha, "history_rule": u["history_reason"]}
                        self.seals[atom] = digest([self.examples[atom], self.bindings[atom]])
                self.files.append({"phase": phase, "path": name, "sha256": file_sha,
                                   "context_sha256": context_sha, "atoms": file_atoms, "bytes": len(raw)})
        self.contexts = MappingProxyType(contexts)
        self.counts = {"file_versions": len(self.files), "distinct_parses_or_declarations": len(cache),
                       "records": len(self.records), "current_units": len(self.examples),
                       "task_grouping_units": len({r["unit"] for r in self.records.values()})}

    def read(self, atom):
        require(atom in self.examples, CODE, "only current submitted units may be measured")
        b = self.bindings[atom]
        require(digest([self.examples[atom], b]) == self.seals[atom], CODE, "loaded context/cutoff binding changed")
        require(b["file_context_sha256"] in self.contexts, CODE, "complete file context missing")
        keys = [*b["visible_past"], atom]
        require(all(k in self.records for k in keys), CODE, "missing source unit/history")
        rs = [validate_record(self.records[k]) for k in keys]
        require([r["atom_id"] for r in rs] == keys and all(r["unit"] == b["unit"] for r in rs),
                CODE, "record identity or task grouping changed")
        return deepcopy(rs[-1]), deepcopy(rs[:-1]), deepcopy(b)

    def context(self, sha):
        require(sha in self.contexts and hashlib.sha256(self.contexts[sha]).hexdigest() == sha,
                CODE, "context digest changed")
        return json.loads(self.contexts[sha])

    def export(self, output):
        output.mkdir(parents=True, exist_ok=False)
        m = write_records(output / "inputs.ucr.jsonl", self.records.values(),
                          {"snapshot": self.identity, "cutoff": CUTOFF, "admitted": False})
        publish(output / "records.json", self.records)
        publish(output / "index.json", {"bindings": self.bindings, "files": self.files, "counts": self.counts,
            "omitted": self.omitted, "input_sha256": m["sha256"], "admitted": False})
        folder = output / "contexts"
        folder.mkdir()
        for sha in self.contexts:
            publish(folder / (sha + ".json"), self.context(sha))
        return m
