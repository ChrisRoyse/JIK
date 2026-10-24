"""Authenticated task sources without loading model weights or native reading banks."""
from pathlib import Path
import json

from claims.capability_population import CapabilityPopulation, need, sha
from claims.capability_context_population import ContextPopulation
from claims.capability_revision_population import RevisionPopulation


def pinned(ref):
    raw = Path(ref['path']).read_bytes()
    need(sha(raw) == ref['pin'], 'ANSWER_SOURCE_PIN')
    return json.loads(raw)


def stamp(path):
    s = Path(path).stat()
    return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns


class AnswerSources:
    def __init__(self, ref):
        self.ref = dict(ref); self.config = c = pinned(ref)
        self.snapshot, self.kernel = s, k = pinned(c['snapshot']), pinned(c['kernel'])
        audit, census = pinned(c['state_readback']), pinned(c['census'])
        need(sha(Path(c['journal']['path']).read_bytes()) == c['journal']['pin'] == census['journal'],
             'ANSWER_JOURNAL')
        need(census['snapshot_sha256'] == c['snapshot']['pin'] and s['head'] == audit['head']
             and audit['active'] == s['task_kernel']['checkpoint'] and k['recall_passes']
             and k['source'] == audit['active']['source'] and not s['task_kernel']['pending'],
             'ANSWER_SNAPSHOT')
        self.train = sorted({x for g in k['graphs'].values() for x in g['entries']})
        self.query = k['split']['queries']
        need(not set(self.train) & set(self.query) and set(self.train) <= set(k['ids']), 'ANSWER_SPLIT')
        self.tasks = {key:s['capability_memory']['tasks'][key] for key in self.train + self.query}
        need(not {self.tasks[x]['split_group'] for x in self.train} &
             {self.tasks[x]['split_group'] for x in self.query}, 'ANSWER_GROUP_SPLIT')
        wanted = {t['population'] for t in self.tasks.values()}; self.populations = {}
        self.specs = {}; self.paths = [Path(ref['path'])]
        self.paths += [Path(c[n]['path']) for n in ('snapshot', 'kernel', 'state_readback', 'census', 'journal')]
        for intake in s['capability_memory']['intakes'].values():
            spec = intake['body']['spec']; pin = spec['population_pin']
            if pin not in wanted or pin in self.populations: continue
            schema = spec.get('adapter', {}).get('schema')
            need(schema in (None, 'capability-context-native-v1', 'capability-delta-native-v1'),
                 'ANSWER_ADAPTER')
            cls = {None:CapabilityPopulation, 'capability-context-native-v1':ContextPopulation,
                   'capability-delta-native-v1':RevisionPopulation}[schema]
            pop = cls(spec['population'], pin)
            self.populations[pin], self.specs[pin] = pop, spec
            self.paths += [pop.root/n for n in (*pop.report['files'], 'report.json', 'COMMITTED')]
        need(set(self.populations) == wanted, 'ANSWER_POPULATIONS')
        for key, task in self.tasks.items():
            pop = self.populations[task['population']]; original = pop.tasks[key]
            need(task['kind'] == original['kind'] and task['split_group'] == original['split_group'],
                 'ANSWER_TASK_BINDING')
            episodes = [task['episode'], *task['related_executions']]
            for side in ('context', 'targets', 'consequences'):
                need(task['selections'][side] == pop.selection(key, side=side), 'ANSWER_SELECTION')
                episodes += [s['capability_memory']['fields'][pop.pin+':'+f]['episode']
                             for f in original[side]]
            need(all(s['memory']['states'][ep]['active'] and
                     s['memory']['states'][ep]['grounded_scope'] is not None for ep in episodes),
                 'ANSWER_WITHDRAWN')
        self.stamps = {str(p):stamp(p) for p in self.paths}

    def selected(self, key):
        task = self.tasks[key]; pop = self.populations[task['population']]
        return pop, pop.tasks[key], 'train' if key in self.train else 'query'

    def fresh(self):
        need(all(stamp(p) == value for p, value in self.stamps.items()), 'ANSWER_SOURCE_CHANGED')
