"""Observed output windows preserve stream provenance and original instruments."""
from pathlib import Path

import numpy as np

from audit.window_readback import readback
from claims.codata_source import check, sha
from claims.episodic_memory import digest
from claims.proposal_echo_population import ProposalEchoPopulation
from model.constellation import Constellation
from model.loop_input import LoopInputs
from model.proposal_input import load


class ProposalEchoInputs:
    def __init__(self, population, population_pin, native, native_pin):
        self.population = ProposalEchoPopulation(population,population_pin)
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

    def materialize(self, key, name, *, refs=None):
        selected = [r for r in (self.population.refs if refs is None else refs) if r['observation'] == key]
        check([r['stream'] for r in selected] == ['stdout','stderr'],'PROPOSAL_ECHO_STREAM_ORDER')
        expected = self.observations.read(key)['captured']
        pieces, fields, cursor = [], [], 0
        x, ptr = self.source_arrays[name]
        for item in selected:
            unit, text = item['unit'],expected[item['stream']]
            check(unit != self.population.probe['unit'],'PROPOSAL_ECHO_PROBE_LEAK')
            check((unit is None) == (text == '') and sha(text.encode()) == item['text_sha256']
                  and len(text.encode()) == item['bytes'],'PROPOSAL_ECHO_SURFACE')
            count = 0
            if unit is not None:
                check(type(unit) is int and 0 <= unit < len(self.population.rows)
                      and self.population.rows[unit]['text'] == text,'PROPOSAL_ECHO_UNIT')
                piece = x[ptr[unit]:ptr[unit+1]]
                check(len(piece) > 0,'PROPOSAL_ECHO_EMPTY_READING')
                pieces.append(piece)
                count = len(piece)
            fields.append(dict(stream=item['stream'],source_unit=unit,start=cursor,end=cursor+count,
                               bytes=item['bytes'],text_sha256=item['text_sha256']))
            cursor += count
        check(pieces,'PROPOSAL_ECHO_NO_OUTPUT_SURFACE')
        return np.concatenate(pieces),fields

    def memory_hash(self):
        return digest({n:[sha(a.tobytes()) for a in values] for n,values in self.arrays.items()})

    batch = LoopInputs.batch

    def checked(self, key):
        check(key in self.lookup,'PROPOSAL_ECHO_UNKNOWN')
        self.observations.read(key)
        check(sha((self.population.root/'report.json').read_bytes()) == self.population.pin,
              'PROPOSAL_ECHO_POPULATION_PIN')
        for name,pin in self.files.items():
            check(sha((self.native/name).read_bytes()) == pin,'PROPOSAL_ECHO_NATIVE_BYTES')
        check(self.memory_hash() == self.memory_pin,'PROPOSAL_ECHO_MEMORY_CHANGED')
        return dict(**self.identity,observation=key,unit=self.lookup[key],
            streams={n:v[key] for n,v in self.streams.items()},scope='captured_output_surfaces_only')

    def observed(self, key, config, device):
        self.checked(key)
        return Constellation(self.batch([self.lookup[key]],config,device),'observed').validate(self.specs,config)
