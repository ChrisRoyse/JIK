"""Frozen native condition windows, retaining every field and instrument boundary."""
import json
from pathlib import Path

import numpy as np
import torch

from audit.window_readback import readback
from claims.codata_source import check, sha
from claims.condition_population import ConditionPopulation
from claims.episodic_memory import canonical, digest
from model.constellation import Constellation
from model.contracts import Instrument
from model.loop_input import LoopInputs
from model.native import NativeInput


class ConditionNative:
    def for_condition(self, key):
        self.conditions.for_condition(key)
        return self

    def __init__(self, conditions, inputs, population, native, *, population_pin, native_pin):
        self.conditions, self.base = conditions, inputs
        self.native = Path(native)
        self.population = ConditionPopulation(population, conditions, inputs.population, population_pin)
        self.readback = readback(self.native, self.population.root)
        raw = (self.native/"report.json").read_bytes()
        report = json.loads(raw)
        check(sha(raw) == native_pin and report["calibration_report_sha256"] == population_pin
              and report["complete"] and not report["errors"]
              and set(report["lenses"]) == {s.name for s in inputs.specs}, "CONDITION_NATIVE_PIN")
        self.specs, self.arrays, self.membership = inputs.specs, {}, {}
        self.keys = sorted(conditions.inputs)
        self.lookup = {k:i for i,k in enumerate(self.keys)}
        self.refs = [dict(text_sha256=k) for k in self.keys]
        self.identity = dict(commitment=conditions.pin, inputs=conditions.commitment["inputs"],
                             population=population_pin, native=native_pin)
        self.files = {"report.json":native_pin}
        groups = {k: [r for r in self.population.fields if r["condition"] == k] for k in self.keys}
        for spec in self.specs:
            name = spec.name
            raw = (self.native/name/"report.json").read_bytes()
            check(sha(raw) == report["lenses"][name]["report_sha256"], "CONDITION_NATIVE_LANE")
            lane = json.loads(raw)
            c = lane["contract"]
            definition = dict(schema="loop-instrument-v1", name=name, channel="language", layout="multi",
                dims=spec.dims, frozen=True, weights_sha256=c["weights_sha256"],
                training_lineage=dict(status="upstream_not_independently_attested"),
                measurement_contract=c, lens_id=lane["lens_id"])
            check(Instrument(name, digest(definition), "language", "multi", spec.dims,
                             canonical(definition).decode()).validate() == spec, "CONDITION_NATIVE_INSTRUMENT")
            self.files[name+"/report.json"] = sha(raw)
            self.files[name+"/slots.npz"] = lane["files"]["slots.npz"]["sha256"]
            with np.load(self.native/name/"slots.npz", allow_pickle=False) as z:
                values, ptr = z["vectors"].copy(), z["offsets"].copy()
            pieces, offsets, membership = [], [0], {}
            for key in self.keys:
                spans, cursor = [], 0
                for r in groups[key]:
                    unit = r["unit"]
                    check(unit != self.population.probe["unit"], "CONDITION_NATIVE_PROBE_LEAK")
                    value = values[ptr[unit]:ptr[unit+1]]
                    pieces.append(value)
                    spans.append(dict(**r, window_start=cursor, window_end=cursor+len(value)))
                    cursor += len(value)
                offsets.append(offsets[-1]+cursor)
                membership[key] = spans
            self.arrays[name] = np.concatenate(pieces), np.asarray(offsets, dtype=np.int64)
            for value in self.arrays[name]:
                value.flags.writeable = False
            self.membership[name] = membership

    batch = LoopInputs.batch

    def checked(self, key):
        self.conditions.read(key)
        check(sha((self.population.root/"report.json").read_bytes()) == self.population.pin,
              "CONDITION_POPULATION_PIN")
        for name, pin in self.files.items():
            check(sha((self.native/name).read_bytes()) == pin, "CONDITION_NATIVE_BYTES")
        return dict(**self.identity, condition=key, unit=self.lookup[key],
                    fields={n: v[key] for n,v in self.membership.items()})

    def merge(self, reference, binding, config, device):
        check(binding == self.checked(binding["condition"]), "CONDITION_NATIVE_BINDING")
        check(reference == self.base.population.lookup.get((reference["task"], reference["turn"])),
              "CONDITION_NATIVE_ACTION")
        action = self.base.batch([reference["unit"]], config, device)
        extra = self.batch([binding["unit"]], config, device)
        slots = {}
        row = digest(dict(action=reference, conditions=binding))
        for spec in self.specs:
            a, b = action[spec.name], extra[spec.name]
            av, bv = a.values[:, ~a.padding[0]], b.values[:, ~b.padding[0]]
            size = av.shape[1]+bv.shape[1]
            check(size <= config.max_native_tokens and size*spec.dims <= config.max_native_elements,
                  "CONDITION_NATIVE_BUDGET")
            slots[spec.name] = NativeInput(spec.identity, (row,), torch.cat((av,bv), dim=1),
                torch.zeros((1,size), dtype=torch.bool, device=device),
                torch.zeros(1, dtype=torch.bool, device=device))
        return Constellation(slots, "observed").validate(self.specs, config)
