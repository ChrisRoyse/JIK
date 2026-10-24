"""Use the unchanged native output materializer with the next observation source."""
from pathlib import Path

import numpy as np

from audit.window_readback import readback
from claims.episodic_memory import digest
from claims.next_echo_population import NextEchoPopulation
from model.proposal_echo_input import ProposalEchoInputs
from model.proposal_input import load


class NextEchoInputs(ProposalEchoInputs):
    def __init__(self, population, population_pin, native, native_pin):
        self.population = NextEchoPopulation(population,population_pin)
        self.observations = self.population.observations
        self.native = Path(native)
        self.readback = readback(self.native,self.population.root)
        self.source_arrays, specs, self.files = load(self.native,native_pin,population_pin)
        self.specs = tuple(specs[n] for n in sorted(specs))
        self.keys = sorted(self.observations.entries)
        self.lookup = {key:i for i,key in enumerate(self.keys)}
        self.refs = [dict(text_sha256=digest([r for r in self.population.refs if r['observation'] == key]))
                     for key in self.keys]
        self.identity = dict(observations=self.observations.pin,population=population_pin,native=native_pin)
        self.arrays, self.streams = {}, {}
        for spec in self.specs:
            pieces, offsets, streams = [], [0], {}
            for key in self.keys:
                values, fields = self.materialize(key,spec.name)
                pieces.append(values)
                offsets.append(offsets[-1]+len(values))
                streams[key] = fields
            self.arrays[spec.name] = np.concatenate(pieces),np.asarray(offsets,dtype=np.int64)
            self.streams[spec.name] = streams
            for value in self.arrays[spec.name]:
                value.flags.writeable = False
        self.memory_pin = self.memory_hash()
