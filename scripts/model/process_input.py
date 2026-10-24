"""Observed process output as complete, typed, frozen native constellations."""
import json
from pathlib import Path

import numpy as np

from audit.window_readback import readback
from claims.codata_source import check, sha
from claims.episodic_memory import canonical, digest
from claims.process_population import ProcessPopulation
from model.constellation import Constellation
from model.contracts import Instrument
from model.loop_input import LoopInputs


class ProcessInputs:
    def __init__(self, observations, inputs, population, native, *, population_pin, native_pin):
        self.observations, self.base = observations, inputs
        self.native, self.pin = Path(native), native_pin
        self.population = ProcessPopulation(population, observations, inputs.population, population_pin)
        self.readback = readback(self.native, self.population.root)
        raw = (self.native/"report.json").read_bytes()
        report = json.loads(raw)
        check(sha(raw) == native_pin and report["calibration_report_sha256"] == population_pin
              and set(report["lenses"]) == {s.name for s in inputs.specs}, "PROCESS_NATIVE_PIN")
        self.specs, self.arrays = inputs.specs, {}
        self.keys = sorted(observations.entries)
        self.lookup = {key:i for i,key in enumerate(self.keys)}
        self.refs = [dict(text_sha256=digest([r for r in self.population.refs if r["observation"] == key]))
                     for key in self.keys]
        self.files = {"report.json":native_pin}
        self.identity = dict(observations=observations.pin, population=population_pin, native=native_pin)
        for spec in self.specs:
            name = spec.name
            lane_raw = (self.native/name/"report.json").read_bytes()
            check(sha(lane_raw) == report["lenses"][name]["report_sha256"], "PROCESS_LANE_PIN")
            lane = json.loads(lane_raw)
            contract = lane["contract"]
            definition = dict(schema="loop-instrument-v1", name=name, channel="language",
                layout="multi", dims=spec.dims, frozen=True, weights_sha256=contract["weights_sha256"],
                training_lineage={"status":"upstream_not_independently_attested"},
                measurement_contract=contract, lens_id=lane["lens_id"])
            actual = Instrument(name, digest(definition), "language", "multi", spec.dims,
                                canonical(definition).decode()).validate()
            check(actual == spec, "PROCESS_INSTRUMENT_MISMATCH")
            self.files[name+"/report.json"] = sha(lane_raw)
            self.files[name+"/slots.npz"] = lane["files"]["slots.npz"]["sha256"]
            with np.load(self.native/name/"slots.npz", allow_pickle=False) as data:
                x, ptr = data["vectors"].copy(), data["offsets"].copy()
            pieces, offsets = [], [0]
            for key in self.keys:
                units = [r["unit"] for r in self.population.refs
                         if r["observation"] == key and r["unit"] is not None]
                check(units and self.population.probe["unit"] not in units, "PROCESS_EMPTY_FEEDBACK")
                for unit in units:
                    pieces.append(x[ptr[unit]:ptr[unit+1]])
                offsets.append(offsets[-1]+sum(int(ptr[u+1]-ptr[u]) for u in units))
            self.arrays[name] = np.concatenate(pieces), np.asarray(offsets, dtype=np.int64)
            for values in self.arrays[name]:
                values.flags.writeable = False

    batch = LoopInputs.batch

    def for_observation(self, key):
        check(key in self.lookup, "PROCESS_NATIVE_UNKNOWN")
        return self

    def checked(self, key):
        check(key in self.lookup, "PROCESS_NATIVE_UNKNOWN")
        self.observations.read(key)
        check(sha((self.population.root/"report.json").read_bytes()) == self.population.pin,
              "PROCESS_POPULATION_PIN")
        for name, pin in self.files.items():
            check(sha((self.native/name).read_bytes()) == pin, "PROCESS_NATIVE_BYTES")
        return dict(**self.identity, observation=key, unit=self.lookup[key], scope="captured_output_surfaces_only")

    def observed(self, key, config, device):
        self.checked(key)
        return Constellation(self.batch([self.lookup[key]], config, device), "observed").validate(self.specs, config)
