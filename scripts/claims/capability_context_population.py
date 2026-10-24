"""Strict coding-context consumer: prior targets and outcomes stay private and unchanged."""
from collections import Counter
import json
from pathlib import Path

from claims.capability_population import CapabilityPopulation, need
from claims.repository_context import POLICY, RepositoryContext
from senses.trajectory_population import read_pinned


ROLES = dict(python_source='base_api_source', public_base_test='public_base_test',
             documentation='base_documentation')


class ContextPopulation(CapabilityPopulation):
    def __init__(self, root, pin):
        report = read_pinned(Path(root) / 'report.json', pin)
        source = report['source']
        need(source['adapter'] == 'repository-context-population-v1' and source['policy'] == POLICY,
             'CONTEXT_ADAPTER')
        p = source['parent']
        self.parent = CapabilityPopulation(p['path'], p['pin'])
        c = source['census']
        self.context = RepositoryContext(c['path'], c['pin'], source['source_root'], self.parent)
        super().__init__(root, pin)
        need(len(self.tasks) == len(self.parent.tasks)
             and {t['provenance']['parent_task'] for t in self.tasks.values()} == self.parent.tasks.keys(),
             'CONTEXT_PARENT_TASK_ROSTER')
        for key, original in self.parent.fields.items():
            need(key in self.fields and {k:v for k,v in self.fields[key].items() if k != 'parts'} ==
                 {k:v for k,v in original.items() if k != 'parts'}, 'CONTEXT_PARENT_FIELD_CHANGED')
        parent_edges = Counter(json.dumps(e, sort_keys=True) for e in self.parent.edges)
        need(not parent_edges - Counter(json.dumps(e, sort_keys=True) for e in self.edges),
             'CONTEXT_PARENT_EDGE_LOST')

    def coding(self, task):
        need(task['kind'] == 'repository_context_to_recorded_patch', 'CONTEXT_KIND')
        p = task['provenance']
        need(p['parent_population'] == self.parent.pin and p['parent_task'] in self.parent.tasks,
             'CONTEXT_PARENT_TASK')
        old = self.parent.tasks[p['parent_task']]
        for key in ('targets', 'consequences', 'split_group', 'checks', 'target_origin'):
            need(task[key] == old[key], 'CONTEXT_CHANGED_'+key.upper())
        need(task['context'][:len(old['context'])] == old['context'], 'CONTEXT_PREFIX')
        name = old['split_group'].removeprefix('repository-task:')
        selection = self.context.selections[name]
        added = [self.fields[k] for k in task['context'][len(old['context']):]]
        expected = sorted(selection['selected'])
        need([f['source']['path'] for f in added] == expected and added, 'CONTEXT_SELECTED_ROSTER')
        for f in added:
            path = f['source']['path']
            need(f['source'] == self.context.address(name, path)
                 and f['text'].encode() == self.context.files[name][path]
                 and f['role'] == ROLES[self.context.catalogs[name][path]['kind']]
                 and f['clock'] == self.context.clock(name), 'CONTEXT_SOURCE_OR_ROLE')
            self.edge(f['id'], task['targets'][0], 'base_context_available_for_patch')
        need(p == dict(parent_population=self.parent.pin, parent_task=old['id'],
                       parent_kind=old['kind'], context_policy=POLICY,
                       localization='inherited patch-touched original paths; supplied, not autonomous',
                       benchmark_test_patch_read=False, current_outcome_used_for_selection=False),
             'CONTEXT_PROVENANCE')
