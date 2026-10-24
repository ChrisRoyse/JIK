"""Versioned structural view over any sealed UCR prefix-reader interface.

One transformation per unique record, reused across prefixes. No raw source,
future, label or filesystem named in a tool response is opened by this reader.
"""
import copy
import os
from pathlib import Path

from .contract import canonical, digest, require, validate_record
from .numbered_code import CONTRACT, enrich
from .pipeline import file_hash


class PrefixInputs:
    def __init__(self, source, selected=None):
        self.records, self.receipts, self.examples, self.bindings = {}, {}, {}, {}
        parents = list(source.examples) if selected is None else list(selected)
        require(parents and len(parents) == len(set(parents)) and all(k in source.examples for k in parents),
                "MNLM_PREFIX_SELECTION", "nonempty unique current observations required")
        self.seals = {}
        transformed = {}
        for parent in parents:
            current, past, binding = source.read(parent)
            require(isinstance(binding, dict) and all(k not in binding or binding[k] is False
                    for k in ("future_visible", "outcomes_visible")),
                    "MNLM_PREFIX_VISIBILITY", "structural view cannot repair an input exposing future/outcomes")
            require(current["atom_id"] == parent, "MNLM_PREFIX_ID", "source current identity differs from selection")
            ids = []
            for record in [*past, current]:
                validate_record(record)
                key = record["atom_id"]
                if key not in transformed:
                    value, receipt = enrich(record)
                    transformed[key] = value["atom_id"]
                    self.records[value["atom_id"]] = value
                    self.receipts[key] = receipt
                ids.append(transformed[key])
            key = ids[-1]
            require(key not in self.examples, "MNLM_PREFIX_DUPLICATE", "duplicate transformed observation")
            self.examples[key] = {"current": key, "past": ids[:-1], "parent": parent, "unit": current["unit"]}
            self.bindings[key] = {"source": binding, "current": key, "visible_past": ids[:-1],
                                  "reads": "supplied-records-only",
                                  **{k: binding[k] for k in ("future_visible", "outcomes_visible") if k in binding}}
            self.seals[key] = digest([self.examples[key], self.bindings[key]])
        self.reader_code_sha256 = file_hash(Path(__file__))
        self.descriptor = {"schema": "structured-prefix-view-v1", "source_manifest_sha256": source.identity,
            "source_reader_code_sha256": source.reader_code_sha256, "adapter": CONTRACT,
            "reader_code_sha256": self.reader_code_sha256, "examples_sha256": digest(self.examples),
            "bindings_sha256": digest(self.bindings), "selection_sha256": digest(parents),
            "records": len(self.records), "admitted": False}
        self.identity = digest(self.descriptor)

    def read(self, atom):
        require(atom in self.examples, "MNLM_PREFIX_NOT_FOUND", "unknown structural observation")
        example = self.examples[atom]
        require(digest([example, self.bindings[atom]]) == self.seals[atom],
                "MNLM_PREFIX_BINDING", "prefix ordering or binding changed after loading")
        records = [validate_record(self.records[k]) for k in [*example["past"], atom]]
        require(all(r["unit"] == example["unit"] for r in records)
                and [r["atom_id"] for r in records] == [*example["past"], atom],
                "MNLM_PREFIX_UNIT", "structural view crossed source units")
        return copy.deepcopy(records[-1]), copy.deepcopy(records[:-1]), {
            **copy.deepcopy(self.bindings[atom]), "input_manifest_sha256": self.identity}

    def write_descriptor(self, path):
        with Path(path).open("xb") as output:
            output.write(canonical(self.descriptor))
            output.flush()
            os.fsync(output.fileno())
        require(file_hash(path) == self.identity, "MNLM_PREFIX_DESCRIPTOR", "descriptor readback differs")
