"""Source-bound temporal views over the intrinsic model's unchanged native instruments."""
import json
from pathlib import Path

import numpy as np

from audit.window_readback import readback as window_readback
from claims.codata_source import check, sha
from claims.episodic_memory import canonical, digest
from claims.tool_temporal_source import TemporalPopulation
from model.constellation import Constellation
from model.contracts import Instrument
from model.loop_input import LANES, LoopInputs


class ToolInputs:
    def __init__(self, source, population, native, *, population_pin, native_pin):
        population, native = Path(population), Path(native)
        self.population = TemporalPopulation(population, source, population_pin)
        self.window_readback = window_readback(native, population)
        raw = (native/"report.json").read_bytes()
        report = json.loads(raw)
        check(sha(raw) == native_pin and report["complete"] and not report["errors"]
              and report["calibration_report_sha256"] == population_pin
              and tuple(sorted(report["lenses"])) == LANES, "TOOL_NATIVE_JOIN")
        self.refs = [dict(text_sha256=sha(r["text"].encode())) for r in self.population.units]
        self.specs, self.arrays, self.window_atoms = [], {}, {}
        groups = {}
        for span in self.population.spans:
            groups.setdefault(span["unit"], []).append(span["atom"])
        for name in LANES:
            root = native/name
            raw = (root/"report.json").read_bytes()
            check(sha(raw) == report["lenses"][name]["report_sha256"], "TOOL_LANE_PIN")
            lane = json.loads(raw)
            check(sha((root/"slots.npz").read_bytes()) == lane["files"]["slots.npz"]["sha256"], "TOOL_SLOT_PIN")
            with np.load(root/"slots.npz", allow_pickle=False) as z:
                x, ptr = z["vectors"].copy(), z["offsets"].copy()
            check(len(ptr) == len(self.population.atoms)+1 and ptr[0] == 0 and ptr[-1] == len(x)
                  and np.all(np.diff(ptr) > 0), "TOOL_NATIVE_ROSTER")
            contract = lane["contract"]
            definition = dict(schema="loop-instrument-v1", name=name, channel="language",
                layout="multi", dims=x.shape[1], frozen=True, weights_sha256=contract["weights_sha256"],
                training_lineage={"status": "upstream_not_independently_attested"},
                measurement_contract=contract, lens_id=lane["lens_id"])
            self.specs.append(Instrument(name, digest(definition), "language", "multi", x.shape[1],
                                         canonical(definition).decode()).validate())
            # Reassemble ordered native windows, never average vectors. Keep the
            # atom membership of every window alongside each message offset.
            pieces, offsets, membership = [], [0], []
            for unit in range(len(self.refs)):
                for atom in groups[unit]:
                    pieces.append(x[ptr[atom]:ptr[atom+1]])
                    membership.extend([atom]*int(ptr[atom+1]-ptr[atom]))
                offsets.append(len(membership))
            self.arrays[name] = np.concatenate(pieces), np.asarray(offsets, dtype=np.int64)
            self.window_atoms[name] = tuple(membership)
        self.specs = tuple(self.specs)
        self.source_id = digest(dict(population=population_pin, native=native_pin))
        self.world_units = sorted({r["unit"] for r in self.population.initial_bank()})

    batch = LoopInputs.batch

    def observed(self, reference, config, device):
        address = reference.get("task"), reference.get("turn")
        check(reference == self.population.lookup.get(address), "TOOL_OBSERVATION_JOIN")
        return Constellation(self.batch([reference["unit"]], config, device), "observed").validate(self.specs, config)

    def prefix(self, task, cutoff, *, partition, observed_through, config, device, references=None):
        expected = self.population.prefix(task, cutoff, partition=partition, observed_through=observed_through)
        check(references is None or tuple(references) == expected, "TOOL_PREFIX_JOIN")
        return tuple(self.observed(r, config, device) for r in expected)

    def feedback(self, task, cutoff, *, receipt, config, device):
        return self.observed(self.population.echo(task, cutoff, receipt=receipt), config, device)

    def world_bank(self, config, device, *, references=None):
        expected = self.population.initial_bank()
        check(references is None or tuple(references) == expected, "TOOL_BANK_CUTOFF")
        return self.batch(self.world_units, config, device)

    def require_meaning(self):
        check(False, "TOOL_SENSE_ADMISSION_MISSING")
