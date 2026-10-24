"""Ordered native source windows, one typed branch per frozen instrument."""
import copy
from pathlib import Path

import numpy as np

from audit.window_readback import readback
from claims.episodic_memory import digest
from claims.git_source_population import SourcePopulation
from model.constellation import Constellation
from model.contracts import validate_bank
from model.native import from_arrays
from model.proposal_input import load
from oracle.git_source_slice import digest as sha, require


class SourceNative:
    def __init__(self, population, population_pin, native, native_pin):
        self.population = SourcePopulation(population, population_pin)
        self.root = Path(native)
        self.readback = readback(self.root, self.population.root)
        arrays, specs, self.files = load(self.root, native_pin, population_pin)
        self.specs = tuple(specs[n] for n in sorted(specs))
        validate_bank(self.specs)
        # Bytes-backed arrays cannot be made writable after binding.
        self.arrays = {n: tuple(np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)
                               for a in pair) for n, pair in arrays.items()}
        self.identity = dict(population=population_pin, native=native_pin,
                             source=self.population.source.pin)

    def fresh(self):
        for name, pin in self.files.items():
            require(sha((self.root/name).read_bytes()) == pin, 'NATIVE_CHANGED')
        require(sha((self.population.root/'report.json').read_bytes()) == self.population.pin,
                'POPULATION_CHANGED')
        for name, entry in self.population.report['files'].items():
            require(sha((self.population.root/name).read_bytes()) == entry['sha256'], 'POPULATION_CHANGED')

    def materialize(self, name, *, scientific_admission=False, require_execution=False):
        require(scientific_admission is False, 'NATIVE_UNADMITTED')
        require(require_execution is False, 'NO_EXECUTION_OUTCOME')
        require(name in self.population.occurrences, 'UNKNOWN_FILE')
        occurrence = self.population.occurrences[name]
        values, spans = {}, {}
        for spec in self.specs:
            x, ptr = self.arrays[spec.name]
            pieces, parts, cursor = [], [], 0
            for part in occurrence['parts']:
                unit = part['unit']
                piece = x[ptr[unit]:ptr[unit+1]]
                require(len(piece) > 0, 'MISSING_NATIVE_WINDOW')
                pieces.append(piece)
                parts.append(dict(**part, window_start=cursor, window_end=cursor+len(piece)))
                cursor += len(piece)
            values[spec.name] = np.concatenate(pieces) if pieces else np.empty((0, spec.dims), np.float32)
            spans[spec.name] = parts
        description = dict(schema='git-source-native-selection-v1', pins=self.identity,
            occurrence=copy.deepcopy(occurrence), windows=spans, scientific_admission=False,
            execution_outcome=None, metadata_used_as_native_values=False)
        return values, description

    def observed(self, name, config, device, specs):
        require(tuple(specs) == self.specs, 'NATIVE_INSTRUMENT')
        arrays, description = self.materialize(name)
        row, slots = digest(description), {}
        for spec in specs:
            x = arrays[spec.name]
            slots[spec.name] = from_arrays(spec, dict(tokens=x,
                offsets=np.asarray([0, len(x)], np.int64), absent=np.asarray([not len(x)])),
                (row,), config).to(device)
        return Constellation(slots, 'observed').validate(specs, config), description

    def receipt(self, name, config, device, specs):
        value, description = self.observed(name, config, device, specs)
        slots = {}
        for spec in specs:
            slot = value.slots[spec.name]
            x = slot.values[0, ~slot.padding[0]].detach().cpu().contiguous().numpy().astype('<f4')
            slots[spec.name] = dict(instrument=spec.identity, shape=list(x.shape),
                                    sha256=sha(x.tobytes()), dtype='float32-le', absent=bool(slot.absent[0]))
        return dict(description=description, slots=slots)
