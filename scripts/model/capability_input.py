"""Complete typed native fields for capability tasks; private targets stay separate."""
import json
from pathlib import Path

import numpy as np

from claims.capability_population import CapabilityPopulation, need
from claims.episodic_memory import digest
from model.native import from_arrays
from model.proposal_input import load
from text.evaluation_multiscale import file_sha


class CapabilityNative:
    def __init__(self, population, population_pin, plan, plan_pin, native, native_pin):
        self.population = CapabilityPopulation(Path(population), population_pin)
        self.plan, self.native = Path(plan), Path(native)
        need(file_sha(self.plan / 'report.json') == plan_pin, 'NATIVE_PLAN_PIN')
        report = json.loads((self.plan / 'report.json').read_bytes())
        need(report['complete'] and not report['errors'] and not report['scientific_admission'], 'NATIVE_PLAN')
        need(file_sha(self.plan / 'mapping.json') == report['mapping_sha256'], 'NATIVE_MAPPING_PIN')
        mapping = json.loads((self.plan / 'mapping.json').read_bytes())
        family = self.population.family
        need(mapping[family]['pin'] == population_pin
             and Path(mapping[family]['population']) == self.population.root, 'NATIVE_POPULATION_JOIN')
        arrays, specs, files = load(self.native, native_pin, plan_pin)
        self.specs = tuple(specs[n] for n in sorted(specs))
        self.banks = {'fresh': arrays}
        self.files = {str(self.native): files}
        self.field_cache = {}
        self.mapping = mapping[family]
        if family == 'coding':
            by_root = {}
            for field in self.population.fields.values():
                s = field['source']
                expected = {k: s[k] for k in ('native_path', 'native_pin', 'native_parts',
                                             'population', 'population_pin')}
                need(mapping['coding']['field_reading_sources'][field['id']] == expected, 'NATIVE_CODE_MAP')
                key = s['native_path']
                record = (s['native_pin'], s['population_pin'])
                need(key not in by_root or by_root[key] == record, 'NATIVE_CODE_ALIAS')
                by_root[key] = record
            for root, (pin, source_pin) in by_root.items():
                bank, old_specs, old_files = load(Path(root), pin, source_pin)
                need(old_specs == specs, 'NATIVE_INSTRUMENT_CHANGED')
                old = json.loads((Path(root) / 'report.json').read_bytes())
                need(set(old['sources']) == {'repository'}, 'NATIVE_CODE_ROW_LAYOUT')
                self.banks[root], self.files[root] = bank, old_files
            self.offset = None
        else:
            offset = 0
            found = False
            for name in sorted(report['families']):
                summary = self.plan / name / 'summary.json'
                need(file_sha(summary) == report['families'][name]['summary_sha256'], 'NATIVE_FAMILY_PIN')
                counts = json.loads(summary.read_bytes())
                if name == family:
                    need(counts['chunks'] == len(self.population.units)
                         and file_sha(self.plan / name / 'chunks/atoms.parquet')
                         == self.population.report['files'][family + '/chunks/atoms.parquet']['sha256'],
                         'NATIVE_FRESH_ROWS')
                    self.offset, found = offset, True
                offset += counts['chunks']
            need(found, 'NATIVE_FRESH_FAMILY')
        self.binding = dict(population=population_pin, plan=plan_pin, native=native_pin,
                            instruments={s.name: s.identity for s in self.specs},
                            scientific_admission=False, serving=False)
        self.pin = digest(self.binding)

    def field(self, key, name):
        cache_key = (key, name)
        if cache_key in self.field_cache: return self.field_cache[cache_key]
        f = self.population.fields[key]
        if self.population.family == 'coding':
            origin, parts = f['source']['native_path'], f['source']['native_parts']
            offset = 0
        else:
            origin, parts, offset = 'fresh', f['parts'], self.offset
        vectors, indices = self.banks[origin][name]
        pieces, spans, cursor = [], [], 0
        for p in parts:
            row = p['unit'] + offset
            need(0 <= row < len(indices) - 1, 'NATIVE_FIELD_ROW')
            lo, hi = map(int, indices[row:row + 2])
            need(0 <= lo < hi <= len(vectors), 'NATIVE_FIELD_WINDOW')
            piece = vectors[lo:hi]
            pieces.append(piece)
            spans.append(dict(origin=origin, row=row, source_byte_span=[p['start'], p['end']],
                              source_sha256=p['sha256'], source_window_span=[lo, hi],
                              field_window_span=[cursor, cursor + len(piece)]))
            cursor += len(piece)
        values = np.concatenate(pieces)
        values.flags.writeable = False
        result = (values, dict(field=key, role=f['role'], parts=spans, windows=len(values)))
        self.field_cache[cache_key] = result
        return result

    def batch(self, key, *, side, config, device, purpose='engineering_boundary'):
        need(purpose == 'engineering_boundary', 'NATIVE_ADMISSION_REQUIRED')
        selection = self.population.selection(key, side=side)
        need(selection['fields'], 'NATIVE_EMPTY_SELECTION')
        descriptions, arrays = {}, {}
        for spec in self.specs:
            pieces, records, cursor = [], [], 0
            for selected in selection['fields']:
                values, desc = self.field(selected['id'], spec.name)
                records.append(dict(desc, start=cursor, end=cursor + len(values)))
                pieces.append(values)
                cursor += len(values)
            arrays[spec.name] = np.concatenate(pieces)
            descriptions[spec.name] = records
        description = dict(binding=self.pin, selection=selection, fields=descriptions,
                           numerical_source_identifiers_are_model_inputs=False)
        row = digest(description)
        slots = {s.name: from_arrays(s, dict(tokens=arrays[s.name],
                    offsets=np.array([0, len(arrays[s.name])], dtype=np.int64), absent=np.array([False])),
                    (row,), config).to(device) for s in self.specs}
        for s in self.specs: slots[s.name].check(s, config)
        return slots, description

    def fresh(self):
        self.population.fresh()
        need(file_sha(self.plan / 'report.json') == self.binding['plan'], 'NATIVE_STALE_PLAN')
        for root, files in self.files.items():
            for name, pin in files.items():
                need(file_sha(Path(root) / name) == pin, 'NATIVE_STALE_BYTES')
