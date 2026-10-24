"""Shared native stores for new source versions; preserve every typed window and parent field."""
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from claims.capability_population import need
from claims.capability_revision_population import RevisionPopulation
from claims.episodic_memory import digest
from model.capability_input import CapabilityNative
from model.proposal_input import load
from text.capability_delta_plan import retained_cache
from text.evaluation_multiscale import file_sha


class DeltaStore:
    """One authenticated store per immutable plan/native pair, shared by all populations."""
    def __init__(self, plan, plan_pin, native, native_pin):
        self.plan, self.native = Path(plan), Path(native)
        self.plan_pin, self.native_pin = plan_pin, native_pin
        need(file_sha(self.plan / 'report.json') == plan_pin, 'DELTA_PLAN_PIN')
        report = json.loads((self.plan / 'report.json').read_bytes())
        need(report['schema'] == 'capability-delta-measurement-plan-v1' and report['complete']
             and not report['errors'] and report['scientific_admission'] is False, 'DELTA_PLAN_SCOPE')
        need(file_sha(self.plan / 'mapping.json') == report['mapping_sha256'], 'DELTA_MAPPING_PIN')
        mapping = json.loads((self.plan / 'mapping.json').read_bytes())
        for name in ('retained_config', 'delta_config'):
            need(file_sha(Path(mapping[name]['path'])) == mapping[name]['sha256'], 'DELTA_CONFIG_PIN')
        old = json.loads(Path(mapping['retained_config']['path']).read_bytes())['populations']
        cache, banks = retained_cache(old)
        source_specs = json.loads(Path(mapping['delta_config']['path']).read_bytes())
        self.populations = {s['population_pin']: RevisionPopulation(s['population'], s['population_pin'])
                            for s in source_specs.values()}
        need(set(mapping['populations']) == set(self.populations), 'DELTA_POPULATION_ROSTER')
        summary = self.plan / 'delta/summary.json'
        need(file_sha(summary) == report['families']['delta']['summary_sha256'], 'DELTA_SUMMARY_PIN')
        counts = json.loads(summary.read_bytes())
        path = self.plan / 'delta/chunks/atoms.parquet'
        need(file_sha(path) == counts['chunks_sha256'], 'DELTA_CHUNKS_PIN')
        rows = pq.read_table(path).to_pylist()
        need(len(rows) == counts['chunks'] == report['inputs'], 'DELTA_ROWS')
        probe = json.loads((self.plan / 'probe.json').read_bytes())
        need(file_sha(self.plan / 'probe.json') == report['probe_sha256']
             and probe['row'] == len(rows)-1, 'DELTA_PROBE_PIN')
        used_banks, fresh_rows = set(), set()
        for pin, p in self.populations.items():
            entry = mapping['populations'][pin]
            need(entry['population'] == str(p.root) and entry['family'] == p.family
                 and len(entry['units']) == len(p.units), 'DELTA_SOURCE_JOIN')
            for unit, ref in zip(p.units, entry['units']):
                need(ref['sha256'] == unit['entity'], 'DELTA_UNIT_HASH')
                if ref['bank'] == 'fresh':
                    row = ref['row']
                    need(type(row) is int and 0 <= row < probe['row']
                         and rows[row]['text'] == unit['text'] and rows[row]['entity'] == unit['entity']
                         and unit['entity'] not in cache, 'DELTA_FRESH_ROW')
                    fresh_rows.add(row)
                else:
                    original = cache.get(unit['entity'])
                    need(original is not None and original['text'] == unit['text']
                         and {k: ref[k] for k in ('bank', 'row', 'source')} ==
                         {k: original[k] for k in ('bank', 'row', 'source')}, 'DELTA_CACHE_ROW')
                    used_banks.add(ref['bank'])
        need(fresh_rows == set(range(probe['row'])) and set(mapping['banks']) == used_banks,
             'DELTA_UNUSED_OR_MISSING_ROW')
        arrays, specs, files = load(self.native, native_pin, plan_pin)
        need(all(len(offsets) == len(rows)+1 for _, offsets in arrays.values()), 'DELTA_NATIVE_ROWS')
        self.banks, self.files = {'fresh': arrays}, {str(self.native): files}
        self.roots = {'fresh': self.native}
        for bank in sorted(used_banks):
            need(mapping['banks'][bank] == banks[bank], 'DELTA_CACHE_BANK')
            item = banks[bank]
            values, previous_specs, previous_files = load(Path(bank), item['native_pin'], item['source_pin'])
            need(previous_specs == specs, 'DELTA_INSTRUMENT_CHANGED')
            self.banks[bank], self.files[bank], self.roots[bank] = values, previous_files, Path(bank)
        self.specs = tuple(specs[n] for n in sorted(specs))
        self.mapping = mapping
        self.mapping_pin = digest(mapping)
        self.manifest_files = {str(self.plan / n): file_sha(self.plan / n) for n in
            ('report.json', 'mapping.json', 'probe.json', 'delta/summary.json', 'delta/chunks/atoms.parquet')}
        self.field_cache = {}


class DeltaNative(CapabilityNative):
    def __init__(self, population, population_pin, *, store):
        need(population_pin in store.populations, 'DELTA_UNKNOWN_POPULATION')
        self.store, self.population = store, store.populations[population_pin]
        need(self.population.root == Path(population), 'DELTA_POPULATION_PATH')
        self.plan, self.native, self.specs = store.plan, store.native, store.specs
        self.banks, self.files = store.banks, store.files
        self.mapping = store.mapping['populations'][population_pin]
        self.binding = dict(population=population_pin, plan=store.plan_pin, native=store.native_pin,
            instruments={s.name: s.identity for s in self.specs},
            adapter='capability-delta-native-v1', scientific_admission=False, serving=False)
        self.pin = digest(self.binding)

    def field(self, key, name):
        cache_key = (self.population.pin, key, name)
        if cache_key in self.store.field_cache:
            return self.store.field_cache[cache_key]
        field = self.population.fields[key]
        pieces, spans, cursor = [], [], 0
        for part in field['parts']:
            ref = self.mapping['units'][part['unit']]
            origin, row = ref['bank'], ref['row']
            vectors, indices = self.banks[origin][name]
            need(0 <= row < len(indices)-1, 'DELTA_FIELD_ROW')
            lo, hi = map(int, indices[row:row+2])
            need(0 <= lo < hi <= len(vectors), 'DELTA_FIELD_WINDOW')
            pieces.append(vectors[lo:hi])
            spans.append(dict(origin=origin, row=row, source_byte_span=[part['start'], part['end']],
                source_sha256=part['sha256'], source_window_span=[lo, hi],
                field_window_span=[cursor, cursor+hi-lo]))
            cursor += hi-lo
        values = np.concatenate(pieces)
        values.flags.writeable = False
        result = values, dict(field=key, role=field['role'], parts=spans, windows=len(values))
        self.store.field_cache[cache_key] = result
        return result

    def fresh(self):
        super().fresh()
        need(digest(self.store.mapping) == self.store.mapping_pin, 'DELTA_MUTATED_MAP')
        for path, pin in self.store.manifest_files.items():
            need(file_sha(Path(path)) == pin, 'DELTA_STALE_MAP')
