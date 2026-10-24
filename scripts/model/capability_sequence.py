"""Attested reasoning prefixes enter the existing typed causal-history path.

This adapter accepts source document order only. It does not turn repository
file-list order or overlapping dialogue intervals into a chronological prefix.
No new parameters, instrument identities, targets, or inferred clocks enter it.
"""
import numpy as np
import torch

from claims.capability_population import need
from claims.episodic_memory import digest
from model.constellation import Constellation
from model.constellation_history import CausalHistory
from model.native import from_arrays


def source_prefix(corpus, key, config, device, *, selection=None):
    need(key in corpus.train or key in corpus.query, 'SEQUENCE_TASK_SCOPE')
    task = corpus.tasks[key]
    need(task['kind'] == 'continue_recorded_steps' and task['family'] == 'reasoning',
         'SEQUENCE_OPERATION')
    population = corpus.sources[task['population']].population
    population.fresh()
    expected = population.selection(key, side='context')
    need(task['selections']['context'] == expected, 'SEQUENCE_RETAINED_SELECTION')
    selection = expected if selection is None else selection
    need(selection == expected, 'SEQUENCE_SELECTION')
    fields = [population.fields[item['id']] for item in selection['fields']]
    need(fields[0]['role'] == 'problem' and all(f['role'] == 'recorded_step' for f in fields[1:]),
         'SEQUENCE_ROLES')
    need(all(f['clock'] is None for f in fields), 'SEQUENCE_UNDECLARED_CLOCK')
    # The population validates the complete earlier prefix against source spans.
    # These are document positions; no timestamp or task identity becomes a feature.
    frames, rows = [], []
    for field in fields:
        row = digest(dict(population=task['population'], field=field['id']))
        slots = {}
        for spec in corpus.specs:
            values = corpus.field(task['population'], field['id'], spec.name)
            slots[spec.name] = from_arrays(spec, dict(tokens=values,
                offsets=np.array([0, len(values)], np.int64), absent=np.array([False])),
                (row,), config).to(device)
        rows.append(row)
        frames.append(Constellation(slots, 'observed').validate(corpus.specs, config))
    return CausalHistory('capability:' + key, tuple(rows), tuple(frames))


def predict_sequence(world, request, corpus, key):
    """Use the authenticated complete prefix; callers cannot supply future frames."""
    history = source_prefix(corpus, key, world.config, world.core.initial.device)
    task = corpus.tasks[key]
    native = corpus.batch(key, 'context', world.config, world.core.initial.device)
    state = request(world.start('capability:' + key), operation=task['kind'])
    return world(native, (state,), allowed=torch.ones(len(corpus.regions), dtype=torch.bool,
        device=world.core.initial.device), phase=0,
        event_ids=[digest(dict(key=key, operation=task['kind']))],
        evidence_origin=['source_observation'], advance_phase=False, history=history)
