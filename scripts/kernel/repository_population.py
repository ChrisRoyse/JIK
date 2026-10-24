"""Authenticate the intake journal once; recover every typed native reading."""
import copy
import json
from pathlib import Path

import numpy as np

from audit.repository_memory_readback import chain
from audit.window_readback import readback
from claims.codata_source import sha
from claims.episodic_memory import digest
from claims.repository_cohort_population import CohortPopulation
from claims.repository_depth_population import DepthPopulation
from claims.repository_demonstration_population import DemonstrationPopulation
from kernel.native_route import check
from model.proposal_input import load
from model.repository_input import utc_ns


KINDS = dict(repository_cohort_intake=CohortPopulation,
    repository_depth_intake=DepthPopulation,
    repository_demonstration_intake=DemonstrationPopulation)


class RepositoryPopulation:
    def __init__(self, journal, expected):
        self.path = Path(journal).resolve()
        check(sha(self.path.read_bytes()) == expected, 'REPOSITORY_JOURNAL_PIN')
        _, rows = chain(self.path)
        self.journal_pin, self.rows = expected, rows
        self.groups, self.records, self.values, self.files = [], {}, {}, {str(self.path):expected}
        loaded, specs = {}, None
        for seq,(body,payload,event) in enumerate(rows):
            kind = body.get('kind')
            if kind not in KINDS:
                continue
            spec = body['spec']; root = Path(spec['native'])
            identity = digest({k:v for k,v in spec.items() if k != 'partition'})
            check(spec['partition'] in ('learn_1','learn_2'), 'REPOSITORY_PARTITION')
            if identity not in loaded:
                population = KINDS[kind](spec['population'], spec['population_pin'])
                readback(root, population.root)
                arrays,current,files = load(root, spec['native_pin'], spec['population_pin'])
                check(specs is None or specs == current, 'REPOSITORY_INSTRUMENT_CHANGE')
                specs = current
                self.files.update({str(root/n):pin for n,pin in files.items()})
                self.files[str(population.root/'report.json')] = spec['population_pin']
                self.files.update({str(population.root/n):v['sha256'] for n,v in population.report['files'].items()})
                loaded[identity] = population, arrays
            population,arrays = loaded[identity]
            expected_conditions = {k for k,f in population.fields.items() if f['partition'] == spec['partition']}
            check(set(payload['bindings']) == expected_conditions == set(payload['episodes']), 'REPOSITORY_ROSTER')
            group = dict(seq=seq, event=event, spec=spec, kind=kind, ids=[])
            for condition in sorted(expected_conditions):
                source = population.fields[condition]
                for side in ('input','echo'):
                    receipt = payload['bindings'][condition][side]
                    values,spans = {},{}
                    for name,instrument in sorted(specs.items()):
                        x,ptr = arrays[name]
                        pieces,fields,cursor = [],[],0
                        for field in source[side]:
                            windows = []
                            for part in field['parts']:
                                unit = part['unit']
                                check(sha(population.rows[unit]['text'].encode()) == part['sha256'], 'REPOSITORY_SPAN')
                                piece = x[ptr[unit]:ptr[unit+1]]
                                check(len(piece) > 0, 'REPOSITORY_EMPTY_PART')
                                pieces.append(piece)
                                windows.append(dict(**part,window_start=cursor,window_end=cursor+len(piece)))
                                cursor += len(piece)
                            fields.append(dict(**copy.deepcopy(field), windows=windows))
                        a = np.concatenate(pieces) if pieces else np.empty((0,instrument.dims),np.float32)
                        a.flags.writeable = False
                        slots = dict(instrument=instrument.identity,shape=list(a.shape),sha256=sha(a.astype('<f4').tobytes()),
                                     dtype='float32-le',absent=not len(a))
                        check(receipt['slots'][name] == slots, 'REPOSITORY_SLOT_BYTES')
                        values[name],spans[name] = a,fields
                    available = source['input_available_by' if side == 'input' else 'echo_available_after']
                    description = receipt['description']
                    check(set(receipt['slots']) == set(specs) and description['fields'] == spans
                        and description['condition'] == condition and description['side'] == side
                        and description['task'] == source['task'] and description['container'] == source['container']
                        and description['available'] == available and utc_ns(description['cutoff']) >= utc_ns(available)
                        and description['pins'] == dict(population=spec['population_pin'], native=spec['native_pin'],
                            observations=population.observations.pin,partition=spec['partition'])
                        and description['scientific_admission'] is False, 'REPOSITORY_BINDING')
                    key = digest(dict(episode=payload['episodes'][condition], receipt=receipt))
                    check(key not in self.records, 'REPOSITORY_DUPLICATE')
                    self.records[key] = dict(seq=seq, event=event, episode=payload['episodes'][condition],
                        condition=condition, side=side, task=source['task'], available=available,
                        receipt=receipt, absence=all(not len(a) for a in values.values()))
                    self.values[key] = values; group['ids'].append(key)
            self.groups.append(group)
        check(specs is not None, 'REPOSITORY_EMPTY')
        self.specs = specs

    def fresh(self):
        for name,pin in self.files.items():
            check(sha(Path(name).read_bytes()) == pin, 'REPOSITORY_SOURCE_CHANGED')

    def prefix(self, groups, excluded=(), withdrawn=()):
        check(type(groups) is int and 0 < groups <= len(self.groups), 'REPOSITORY_PREFIX')
        ids = [k for g in self.groups[:groups] for k in g['ids']]
        check(set(withdrawn) <= set(ids), 'REPOSITORY_WITHDRAWAL')
        return [k for k in ids if self.records[k]['task'] not in excluded and k not in withdrawn]

    def source(self, groups):
        seq = self.groups[groups-1]['seq']
        return dict(journal=str(self.path), journal_sha256=self.journal_pin,
                    event_count=seq+1, head=self.rows[seq][2], supported_intakes=groups)

    def holdout(self, initial):
        tasks = {self.records[k]['task'] for k in self.prefix(initial)}
        check(len(tasks) > 2, 'REPOSITORY_QUERY_TASKS')
        selected = sorted(tasks,key=lambda t:(sha(t.encode()),t))[:2]
        keys = [k for k in self.prefix(initial) if self.records[k]['task'] in selected
                and not self.records[k]['absence']]
        check(keys, 'REPOSITORY_QUERY_EMPTY')
        return selected,sorted(keys)
