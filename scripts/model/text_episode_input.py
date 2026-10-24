"""Whole native text windows at exact source intervals; no fictitious audio clock."""
from dataclasses import dataclass
import math
from pathlib import Path

import torch

from model.contracts import Instrument
from model.episode_input import EpisodeBatch
from model.native import NativeInput
from model.stream_attention import NativeBlock, BLOCK_ROWS
from senses.contract import canonical, digest, require
from text.corpus_bank import ROSTER
from text.constellation_corpus import file_sha


@dataclass(frozen=True)
class TextLane:
    owner: object
    index: int
    identity: str
    name: str
    rows: tuple[str, ...]
    count: int
    dims: int
    temporal_pin: str

    def blocks(self, config):
        source, global_index = self.owner.sources[self.index]
        values, trace = self.owner.bank.row(self.name, source, global_index)
        require(len(values) == self.count and values.shape[1] == self.dims,
                "MNLM_TEXT_STREAM_COVERAGE", "restore complete native windows")
        positions = []
        for i, window in enumerate(trace["windows"]):
            lo, hi = window["source_char_span"]
            # Source-order geometry, no wall time or claim of physical synchrony.
            positions.append([math.log1p(v) for v in
                (lo, hi, hi-lo, source["char_start"], source["char_end"], source["pos"], i)]+[1.])
        for start in range(0, len(values), min(BLOCK_ROWS, config.max_native_tokens)):
            end = min(len(values), start+BLOCK_ROWS, start+config.max_native_tokens)
            require((end-start)*self.dims <= config.max_native_elements,
                    "MNLM_TEXT_STREAM_BUDGET", "stream bounded complete blocks")
            tensor = torch.from_numpy(values[start:end].copy())[None]
            yield NativeBlock(start, tensor, torch.zeros(tensor.shape[:2], dtype=torch.bool),
                              torch.tensor(positions[start:end], dtype=torch.float32)[None])


class TextInputs:
    def __init__(self, bank, sources, config):
        self.bank, self.sources, self.config = bank, tuple(sources), config
        self.specs, self.lanes, self.moments, self.temporal_catalogue = [], [], [], {}
        for name in ROSTER:
            r = bank.reports[name]
            definition = dict(schema="source-text-window-instrument-v1", name=name, channel="language",
                layout="multi", dims=r["dimensions"], frozen=True, weights_sha256=r["contract"]["weights_sha256"],
                training_lineage={"status": "upstream_not_independently_attested"},
                measurement_contract=r["contract"], lens_id=r["lens_id"])
            self.specs.append(Instrument(name, digest(definition), "language", "multi", r["dimensions"],
                                         canonical(definition).decode()).validate())
        self.specs = tuple(self.specs)
        require(len({r["atom_id"] for r, _ in sources}) == len(sources), "MNLM_TEXT_STREAM_DUPLICATE",
                "each selected source span has one address")
        for i, (row, global_index) in enumerate(self.sources):
            self.moments.append(dict(id=row["atom_id"]))
            lanes, records = {}, {}
            for spec in self.specs:
                values, trace = bank.row(spec.name, row, global_index)
                record = dict(schema="text-ordinal-support-v1", row=row["atom_id"], instrument=spec.identity,
                    corpus=bank.corpus.pin, bank=bank.pin, sequence=row["sequence"], source_position=row["pos"],
                    source_span=[row["char_start"], row["char_end"]], byte_span=[row["byte_start"], row["byte_end"]],
                    available_after=row["char_end"], clock="source-field-codepoint-order",
                    windows=trace["windows"], physical_timestamp=None)
                records[spec.name] = digest(record)
                lanes[spec.name] = TextLane(self, i, spec.identity, spec.name, (row["atom_id"],),
                                            len(values), spec.dims, records[spec.name])
            self.lanes.append(lanes)
            self.temporal_catalogue[row["atom_id"]] = records
        self.binding = digest(dict(schema="text-native-model-input-v1", bank=bank.pin,
            corpus=bank.corpus.pin, catalogue=self.temporal_catalogue,
            code_sha256=file_sha(Path(__file__)),
            sources=[dict(atom=r["atom_id"], index=i) for r, i in sources]))

    def fresh(self):
        self.bank.fresh()

    def batch(self, index, *, through=None, purpose="engineering_boundary"):
        require(purpose == "engineering_boundary", "MNLM_TEXT_STREAM_ADMISSION_REQUIRED",
                "native readings do not constitute admitted senses")
        require(type(index) is int and 0 <= index < len(self.sources), "MNLM_TEXT_STREAM_INDEX",
                "source-bound observation required")
        row = self.sources[index][0]
        require(type(through) is int and through >= row["char_end"], "MNLM_TEXT_STREAM_FUTURE",
                "the complete atom is unavailable before its source cutoff")
        return EpisodeBatch(self, index).validate(self.specs, self.config)

    def references(self, indices, device):
        require(indices and len(set(indices)) == len(indices), "MNLM_TEXT_STREAM_REFERENCES",
                "distinct frozen source references required")
        result = {}
        rows = tuple(self.sources[i][0]["atom_id"] for i in indices)
        for spec in self.specs:
            arrays = [self.bank.row(spec.name, *self.sources[i])[0] for i in indices]
            width = max(map(len, arrays))
            require(width <= self.config.max_native_tokens
                    and len(indices)*width*spec.dims <= self.config.max_native_elements,
                    "MNLM_TEXT_STREAM_BUDGET", "reference window budget exceeded")
            values = torch.zeros(len(indices), width, spec.dims, dtype=torch.float32, device=device)
            padding = torch.ones(len(indices), width, dtype=torch.bool, device=device)
            for i, a in enumerate(arrays):
                values[i, :len(a)] = torch.from_numpy(a.copy()).to(device)
                padding[i, :len(a)] = False
            result[spec.name] = NativeInput(spec.identity, rows, values, padding,
                                            torch.zeros(len(indices), dtype=torch.bool, device=device))
        return result
