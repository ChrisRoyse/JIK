"""Journal -> union of typed neighbors -> retained region-first meaning predictor.

Only training-neighbor targets form the frozen reference regions. The query's
own next constellation is opened separately, after prediction, for supervision.
"""
import copy

import numpy as np
import torch

from claims.capability_population import need
from claims.episodic_memory import digest
from model.constellation_world import ConstellationWorld
from model.native import from_arrays
from model.stream_surface import ConnectedSurface
from model.tool_context_regions import attach


def evidence_for(session, key):
    task = session.capability_tasks[key]
    matches = [e for e in session.capability_sources.values() if e.native.pin == task['binding']]
    need(len(matches) == 1, 'CAPABILITY_PREDICTION_SOURCE')
    return matches[0]


class CapabilityPrediction:
    def __init__(self, session, key, *, kernel):
        self.session, self.key, self.kernel = session, key, kernel
        self.head = session.head
        route = session.read_task(key, kernel=kernel)
        population = session._task_population()
        need(key in population.queries(), 'CAPABILITY_PREDICTION_HOLDOUT')
        task = session.capability_tasks[key]
        evidence = evidence_for(session, key)
        config, device = session.model.config, session.device
        self.native, description = evidence.native.batch(key, side='context', config=config, device=device)
        need(description['selection'] == session.capability_selection(key, side='context',
             binding=task['binding']), 'CAPABILITY_PREDICTION_CONTEXT')
        self.route = copy.deepcopy(route)
        support = {k for trace in route['traces'].values() for k, _ in trace['hits']}
        need(support == set(route['targets']) and key not in support, 'CAPABILITY_PREDICTION_UNION')
        need(all(session.capability_tasks[k]['split_group'] != task['split_group'] for k in support),
             'CAPABILITY_PREDICTION_FUTURE')
        regions, readings = {}, {}
        for other in sorted(support):
            target = route['targets'][other]
            source = evidence_for(session, other)
            selection = session.capability_selection(other, side='targets', binding=source.native.pin)
            need(target['targets'] == selection, 'CAPABILITY_PREDICTION_TARGET_CLOSURE')
            for field in selection['fields']:
                region = digest(dict(binding=source.native.pin, field=field['id']))
                if region not in regions:
                    regions[region] = dict(binding=source.native.pin, field=field, tasks=[])
                    readings[region] = {s.name: source.native.field(field['id'], s.name)[0]
                                        for s in source.native.specs}
                regions[region]['tasks'].append(other)
        self.regions = dict(sorted(regions.items()))
        need(self.regions, 'CAPABILITY_PREDICTION_NO_REFERENCE')
        private = {(task['binding'], f['id']) for f in task['selections']['targets']['fields']}
        need(not private & {(r['binding'], r['field']['id']) for r in regions.values()},
             'CAPABILITY_PREDICTION_TARGET_LEAK')
        references = {}
        for spec in session.model.specs:
            pieces = [readings[k][spec.name] for k in self.regions]
            references[spec.name] = from_arrays(spec, dict(tokens=np.concatenate(pieces),
                offsets=np.asarray([0, *np.cumsum([len(x) for x in pieces])], dtype=np.int64),
                absent=np.zeros(len(pieces), dtype=bool)), tuple(self.regions), config).to(device)
        self.definition = dict(schema='kernel-capability-prediction-v1', head=self.head, query=key,
            kernel=kernel, context=description, regions=self.regions,
            neighbor_policy='union_of_instrument_neighbors', query_targets_in_bank=False,
            label_source='native_constellation_geometry', scientific_admission=False)
        with torch.random.fork_rng(devices=[]):
            self.view = ConstellationWorld(session.model.core, references,
                                           world_source_id=digest(self.definition))
        self.view.to(device).eval()
        attach(self.view, session.model)
        self.view.check_reference()
        self.event = digest(dict(context=description, kernel=kernel))
        self.output = None

    def fresh(self):
        self.session._fresh()
        need(self.session.head == self.head and self.session.task_intent is None
             and self.session.task_checkpoint['kernel']['pin'] == self.kernel,
             'CAPABILITY_PREDICTION_STALE')
        self.view.check_reference()

    def forward(self):
        self.fresh()
        state = self.view.start('capability:' + self.key)
        self.output = self.view(self.native, (state,),
            allowed=torch.ones(len(self.regions), dtype=torch.bool, device=self.session.device),
            phase=0, event_ids=[self.event], evidence_origin=['source_observation'], advance_phase=False)
        return self.output

    def targets(self):
        self.fresh()
        need(self.output is not None, 'CAPABILITY_PREDICTION_TARGET_BEFORE_FORWARD')
        task = self.session.capability_tasks[self.key]
        source = ConnectedSurface(self.session, self.key, binding=task['binding'])
        return [(field, *source.field(field, self.view.config, self.session.device)) for field in source.fields]
