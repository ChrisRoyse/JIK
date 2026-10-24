"""Typed new action/program windows joined to existing frozen wheel readings."""
import json
from pathlib import Path

import numpy as np
import torch

from audit.window_readback import readback
from claims.codata_source import check, sha
from claims.episodic_memory import canonical, digest
from claims.proposal_population import ProposalPopulation, cached
from model.contracts import Instrument
from model.field_native import FieldBatch, ROLES
from model.native import NativeInput


def definition(name, lane, dims):
    contract = lane['contract']
    value = dict(schema='loop-instrument-v1', name=name, channel='language', layout='multi', dims=dims,
        frozen=True, weights_sha256=contract['weights_sha256'],
        training_lineage=dict(status='upstream_not_independently_attested'),
        measurement_contract=contract, lens_id=lane['lens_id'])
    return Instrument(name, digest(value), 'language', 'multi', dims, canonical(value).decode()).validate()


def load(root, pin, population_pin):
    raw = (root/'report.json').read_bytes()
    report = json.loads(raw)
    check(sha(raw) == pin and report['calibration_report_sha256'] == population_pin
          and report['complete'] and not report['errors'] and report['scientific_admission'] is False,
          'PROPOSAL_NATIVE_PIN')
    arrays, specs, files = {}, {}, {'report.json':pin}
    for name, item in report['lenses'].items():
        raw = (root/name/'report.json').read_bytes()
        check(sha(raw) == item['report_sha256'], 'PROPOSAL_NATIVE_LANE')
        lane = json.loads(raw)
        files[name+'/report.json'] = sha(raw)
        for filename, entry in lane['files'].items():
            check(Path(filename).name == filename, 'PROPOSAL_NATIVE_PATH')
            value = (root/name/filename).read_bytes()
            check(len(value) == entry['bytes'] and sha(value) == entry['sha256'], 'PROPOSAL_NATIVE_BYTES')
            files[name+'/'+filename] = entry['sha256']
        with np.load(root/name/'slots.npz', allow_pickle=False) as archive:
            x, ptr = archive['vectors'].copy(), archive['offsets'].copy()
        specs[name] = definition(name, lane, x.shape[1])
        x.flags.writeable = ptr.flags.writeable = False
        arrays[name] = x, ptr
    return arrays, specs, files


class ProposalNative:
    def __init__(self, population, population_pin, native, native_pin, cache_native, cache_native_pin):
        self.population = ProposalPopulation(population, population_pin)
        p = self.population
        self.roots = {'new':Path(native), 'cache':Path(cache_native)}
        self.pins = dict(population=population_pin, native=native_pin, cached_population=p.cache_pin,
                         cached_native=cache_native_pin, conditions=p.prepared.conditions.pin)
        self.readbacks = dict(new=readback(self.roots['new'], p.root),
                              cache=readback(self.roots['cache'], p.cache_root))
        new, specs, new_files = load(self.roots['new'], native_pin, population_pin)
        old, old_specs, old_files = load(self.roots['cache'], cache_native_pin, p.cache_pin)
        check(specs == old_specs, 'PROPOSAL_NATIVE_INSTRUMENT_CHANGED')
        self.specs = tuple(specs[n] for n in sorted(specs))
        self.arrays, self.files = dict(new=new, cache=old), dict(new=new_files, cache=old_files)
        old_rows, _, old_probe = cached(p.cache_root, p.cache_pin)
        self.rows = dict(new=p.rows, cache=old_rows)
        self.probes = dict(new=p.probe['unit'], cache=old_probe['unit'])
        self.fields = p.fields
        self.field_pin = digest(self.fields)

    def fresh(self):
        check(digest(self.fields) == self.field_pin, 'PROPOSAL_NATIVE_FIELDS_CHANGED')
        check(sha((self.population.root/'report.json').read_bytes()) == self.population.pin,
              'PROPOSAL_POPULATION_PIN')
        for origin, files in self.files.items():
            for name, pin in files.items():
                check(sha((self.roots[origin]/name).read_bytes()) == pin, 'PROPOSAL_NATIVE_BYTES')

    def materialize(self, condition, *, fields=None):
        """Metadata indexes windows; its numbers never enter measured values."""
        self.fresh()
        self.population.prepared.conditions.read(condition)
        fields = self.fields if fields is None else fields
        selected = [r for r in fields if r['condition'] == condition]
        public = self.population.prepared.conditions.read(condition)
        expected_roles = ['action','program'] + ['wheel_filename']*len(public['wheels'])
        check([r['role'] for r in selected] == expected_roles
              and [r['ordinal'] for r in selected] == list(range(len(selected))), 'PROPOSAL_NATIVE_ROLES')
        arrays, spans = {}, {}
        for spec in self.specs:
            pieces, descriptors, cursor = [], [], 0
            for item in selected:
                role, origin, unit = item['role'], item['origin'], item['unit']
                check(origin in self.rows and type(unit) is int and 0 <= unit < len(self.rows[origin])
                      and unit != self.probes[origin], 'PROPOSAL_NATIVE_PROBE_LEAK')
                check((origin == 'cache') == (role == 'wheel_filename'), 'PROPOSAL_NATIVE_ORIGIN')
                row = self.rows[origin][unit]
                expected_text = (self.population.prepared.source.action.decode() if role == 'action' else
                    self.population.prepared.source.program.decode() if role == 'program' else
                    sorted(public['wheels'])[item['ordinal']-2])
                check(row['text'] == expected_text and sha(row['text'].encode()) == item['text_sha256'],
                      'PROPOSAL_NATIVE_SURFACE')
                x, ptr = self.arrays[origin][spec.name]
                piece = x[ptr[unit]:ptr[unit+1]]
                check(len(piece) > 0 and piece.shape[1] == spec.dims, 'PROPOSAL_NATIVE_WINDOWS')
                pieces.append(piece)
                descriptors.append(dict(role=role, start=cursor, end=cursor+len(piece),
                    text_sha256=item['text_sha256'], origin=origin, source_unit=unit))
                cursor += len(piece)
            arrays[spec.name] = np.concatenate(pieces)
            spans[spec.name] = descriptors
        description = dict(schema='proposed-native-fields-v1', pins=self.pins, condition=condition,
            source=self.population.prepared.source.reference(), fields=spans, role_order=list(ROLES),
            numerical_identifiers=False, historical_occurrence_established=False,
            scope='recorded_proposal_and_declared_runtime_for_new_execution')
        return arrays, description

    def batch(self, condition, config, device, specs):
        check(tuple(specs) == self.specs, 'PROPOSAL_NATIVE_MODEL_INSTRUMENT')
        arrays, description = self.materialize(condition)
        row = digest(description)
        slots = {}
        for spec in self.specs:
            values = torch.from_numpy(arrays[spec.name].copy()).unsqueeze(0).to(device)
            slot = NativeInput(spec.identity, (row,), values,
                torch.zeros(values.shape[:2], dtype=torch.bool, device=device),
                torch.zeros(1, dtype=torch.bool, device=device))
            slot.check(spec, config)
            slots[spec.name] = slot
        result = FieldBatch(slots, description)
        result.checked(specs)
        return result
