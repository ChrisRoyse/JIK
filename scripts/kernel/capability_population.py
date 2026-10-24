"""Journal-backed task contexts; targets remain typed, separately addressed relations."""
import copy

import numpy as np

from claims.capability_population import need, sha
from claims.episodic_memory import digest


class TaskPopulation:
    def __init__(self, native, tasks, fields, relationships, states, split):
        self.native, self.population = native, native.population
        p = self.population
        self.specs = {s.name: s for s in native.specs}
        self.split = copy.deepcopy(split)
        need(split['population'] == p.pin and split['family'] == p.family, 'TASK_SPLIT_SOURCE')
        self.records, self.values, self.links = {}, {}, {}
        self.eligible, self.statuses = [], {}
        for key, t in sorted(tasks.items()):
            if t['population'] != p.pin: continue
            original = p.tasks[key]
            need(t['binding'] == native.pin and t['family'] == p.family
                 and t['split_group'] == original['split_group'], 'TASK_SOURCE_BINDING')
            need(t['selections'] == {s: p.selection(key, side=s)
                 for s in ('context', 'targets', 'consequences')}, 'TASK_ROLE_MASK')
            selected = {k for side in ('context', 'targets', 'consequences') for k in original[side]}
            receipts = {k: fields[p.pin + ':' + k] for k in selected}
            required = [e for e in p.edges if e['origin'] in selected and e['target'] in selected]
            need(all(relationships.get(digest(e)) == e for e in required), 'TASK_RELATION_MISSING')
            need(original['targets'] and required, 'TASK_TARGET_MISSING')
            episodes = [t['episode']] + [r['episode'] for r in receipts.values()] + t['related_executions']
            status = {e: states[e] for e in episodes}
            self.statuses[key] = digest(status)
            if all(s['active'] and s['grounded_scope'] is not None for s in status.values()):
                self.eligible.append(key)
            self.links[key] = dict(task=t, fields=receipts, relationships=required)
            values, slots = {}, {}
            for name, spec in self.specs.items():
                pieces, descriptions, cursor = [], [], 0
                for field in original['context']:
                    array, description = native.field(field, name)
                    descriptions.append(dict(description, start=cursor, end=cursor+len(array)))
                    cursor += len(array); pieces.append(array)
                array = np.concatenate(pieces); array.flags.writeable = False
                values[name] = array
                slots[name] = dict(instrument=spec.identity, shape=list(array.shape),
                                  sha256=sha(array.astype('<f4', copy=False).tobytes()),
                                  fields=descriptions)
            self.values[key] = values
            self.records[key] = dict(task=t['split_group'], episode=t['episode'], family=p.family,
                                     receipt=dict(slots=slots), absence=False)
        need(set(self.records) <= set(split['train'] + split['queries']), 'TASK_SPLIT_UNKNOWN')
        self.files = dict(population=p.pin, native=native.binding, split=digest(split))

    def fresh(self): self.native.fresh()

    def ids(self, groups):
        need(groups == sorted(set(groups)) and set(groups) <= set(self.split['groups']), 'TASK_GROUPS')
        result = [k for k in self.eligible if self.records[k]['task'] in groups
                  and self.records[k]['task'] not in self.split['excluded']]
        need(len(result) >= 17, 'TASK_POOL_INSUFFICIENT')
        return result

    def source(self, groups):
        ids = self.ids(groups)
        return dict(family=self.population.family, files=self.files, groups=groups,
                    ids=ids, statuses={k: self.statuses[k] for k in sorted(self.records)},
                    connections={k: digest(self.links[k]) for k in ids},
                    query_contexts={k: digest(self.records[k]['receipt']) for k in self.split['queries']})

    def queries(self):
        need(set(self.split['queries']) <= set(self.eligible), 'TASK_QUERY_INELIGIBLE')
        return self.split['queries']

    def target(self, key):
        need(key in self.eligible, 'TASK_TARGET_INELIGIBLE')
        link = self.links[key]
        return dict(task=key, episode=link['task']['episode'],
                    targets=link['task']['selections']['targets'],
                    consequences=link['task']['selections']['consequences'],
                    connections=digest(link), related_executions=link['task']['related_executions'])


class SharedTasks:
    """One association universe; capability families are evaluation slices only."""
    def __init__(self, populations):
        self.populations = dict(populations)
        need(set(populations) == {'coding', 'conversation', 'reasoning'}, 'TASK_SHARED_FAMILIES')
        self.specs = next(iter(populations.values())).specs
        need(all(p.specs == self.specs for p in populations.values()), 'TASK_SHARED_INSTRUMENTS')
        self.records, self.values, self.links, self.owners, self.statuses = {}, {}, {}, {}, {}
        self.eligible = []
        self.split = dict(schema='shared-capability-split-v1', families={f: p.split for f, p in populations.items()})
        for key in ('groups', 'excluded', 'train', 'queries'):
            self.split[key] = sorted(x for p in populations.values() for x in p.split[key])
        for family, p in populations.items():
            need(not set(p.records) & set(self.records), 'TASK_SHARED_ID_ALIAS')
            for name in ('records', 'values', 'links', 'statuses'): getattr(self, name).update(getattr(p, name))
            self.owners.update({k: p for k in p.records}); self.eligible.extend(p.eligible)
        self.eligible.sort()
        self.files = {f: p.files for f, p in populations.items()}

    def fresh(self):
        for p in self.populations.values(): p.fresh()

    def ids(self, groups):
        need(groups == sorted(set(groups)) and set(groups) <= set(self.split['groups']), 'TASK_SHARED_GROUPS')
        return sorted(k for p in self.populations.values()
                      for k in p.ids(sorted(set(groups) & set(p.split['groups']))))

    def source(self, groups):
        ids = self.ids(groups)
        return dict(family='shared', files=self.files, groups=groups, ids=ids,
                    families={f: p.source(sorted(set(groups) & set(p.split['groups'])))
                              for f, p in self.populations.items()})

    def queries(self): return sorted(k for p in self.populations.values() for k in p.queries())

    def target(self, key):
        need(key in self.owners, 'TASK_SHARED_TARGET')
        return self.owners[key].target(key)
