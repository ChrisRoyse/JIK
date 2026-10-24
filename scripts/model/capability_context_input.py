"""Typed repository-context windows; historical banks retain their instrument identities."""
import json
from pathlib import Path

import pyarrow.parquet as pq

from claims.capability_population import need
from claims.capability_context_population import ContextPopulation
from claims.episodic_memory import digest
from model.capability_delta_input import DeltaNative
from model.proposal_input import load
from senses.trajectory_population import read_pinned
from text.capability_context_plan import historical_cache
from text.evaluation_multiscale import file_sha


class ContextStore:
    def __init__(self, plan, plan_pin, native, native_pin):
        self.plan, self.plan_pin, self.native, self.native_pin = Path(plan), plan_pin, Path(native), native_pin
        report = read_pinned(self.plan / 'report.json', plan_pin)
        need(report['schema'] == 'capability-context-measurement-plan-v1' and report['complete']
             and not report['errors'] and not report['scientific_admission'], 'CONTEXT_PLAN_SCOPE')
        mapping = read_pinned(self.plan / 'mapping.json', report['mapping_sha256'])
        config = read_pinned(Path(mapping['config']['path']), mapping['config']['pin'])
        cache, banks = historical_cache(config)
        pop = ContextPopulation(config['population'], config['population_pin'])
        self.populations = {pop.pin:pop}
        need(set(mapping['populations']) == {pop.pin}, 'CONTEXT_MAP_POPULATION')
        summary = read_pinned(self.plan / 'delta/summary.json', report['families']['delta']['summary_sha256'])
        path = self.plan / 'delta/chunks/atoms.parquet'
        need(file_sha(path) == summary['chunks_sha256'], 'CONTEXT_PLAN_ROWS')
        rows = pq.read_table(path).to_pylist()
        need(len(rows) == summary['chunks'] == report['inputs'], 'CONTEXT_PLAN_COUNT')
        probe = read_pinned(self.plan / 'probe.json', report['probe_sha256'])
        need(probe['row'] == len(rows)-1 and rows[-1]['entity'] == probe['sha256'], 'CONTEXT_PROBE')
        entry = mapping['populations'][pop.pin]
        need(entry['population'] == str(pop.root) and entry['family'] == pop.family
             and len(entry['units']) == len(pop.units), 'CONTEXT_UNIT_ROSTER')
        used, fresh = set(), set()
        for unit, ref in zip(pop.units, entry['units']):
            need(ref['sha256'] == unit['entity'], 'CONTEXT_UNIT_HASH')
            if ref['bank'] == 'fresh':
                row = ref['row']
                need(type(row) is int and 0 <= row < probe['row'] and rows[row]['text'] == unit['text']
                     and rows[row]['entity'] == unit['entity'] and unit['entity'] not in cache, 'CONTEXT_FRESH_ROW')
                fresh.add(row)
            else:
                old = cache.get(unit['entity'])
                need(old is not None and old['text'] == unit['text'] and
                     all(ref[k] == old[k] for k in ('bank', 'row', 'source')), 'CONTEXT_REUSED_ROW')
                used.add(ref['bank'])
        need(fresh == set(range(probe['row'])) and used == set(mapping['banks']), 'CONTEXT_ROW_COVERAGE')
        arrays, specs, files = load(self.native, native_pin, plan_pin)
        need(all(len(offsets) == len(rows)+1 for _, offsets in arrays.values()), 'CONTEXT_NATIVE_ROWS')
        self.banks, self.files, self.roots = {'fresh':arrays}, {str(self.native):files}, {'fresh':self.native}
        for bank in sorted(used):
            need(mapping['banks'][bank] == banks[bank], 'CONTEXT_BANK_ALIAS')
            item = banks[bank]
            values, old_specs, old_files = load(Path(bank), item['native_pin'], item['source_pin'])
            need(old_specs == specs, 'CONTEXT_INSTRUMENT_DRIFT')
            self.banks[bank], self.files[bank], self.roots[bank] = values, old_files, Path(bank)
        self.specs = tuple(specs[n] for n in sorted(specs))
        self.mapping, self.mapping_pin = mapping, digest(mapping)
        self.manifest_files = {str(self.plan / n):file_sha(self.plan / n) for n in
            ('report.json', 'mapping.json', 'probe.json', 'delta/summary.json', 'delta/chunks/atoms.parquet')}
        self.field_cache = {}


class ContextNative(DeltaNative):
    def __init__(self, population, population_pin, *, store):
        super().__init__(population, population_pin, store=store)
        self.binding['adapter'] = 'capability-context-native-v1'
        self.pin = digest(self.binding)
