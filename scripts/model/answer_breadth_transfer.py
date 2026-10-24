"""Named warm-start from a corpus region predictor into complete-answer branches.

This transfers weights, not optimizer continuation. The independent attention
layouts adapt a shared donor; only the shared layout preserves its function.
"""
import torch

from model.native_population_training import fingerprint
from text.constellation_corpus import need


def mapping(model, donor):
    predictor = model.forecaster.predictor; architecture = model.forecaster.architecture
    need(architecture in ('shared', 'own_slot', 'whole_proposals'), 'BREADTH_TRANSFER_ARCHITECTURE')
    need(tuple(model.roster) == tuple(donor.roster) and predictor.config == donor.config,
         'BREADTH_TRANSFER_GEOMETRY')
    result = {}; prefix = 'forecaster.predictor.'
    for name in donor.state_dict():
        if name.startswith('counts.'): continue
        need(name.startswith(('encoder.', 'projectors.', 'region.')), 'BREADTH_TRANSFER_SOURCE_FIELD')
        if name.startswith('encoder.') and architecture != 'shared':
            suffix = name.removeprefix('encoder.')
            for index, lane in enumerate(model.roster):
                result[prefix+'encoder.lanes.'+lane+'.'+suffix] = dict(source=name,
                    row=index if suffix in ('kind', 'query') else None)
        elif name.startswith('region.') and architecture == 'whole_proposals':
            for lane in model.roster: result[prefix+'readouts.'+lane+'.'+name] = dict(source=name, row=None)
        else: result[prefix+name] = dict(source=name, row=None)
    return result


def install(model, donor, *, source_reference, destination_reference, expected_source):
    need(source_reference == destination_reference and isinstance(source_reference, dict)
         and set(source_reference) == {'pin', 'state', 'rows'}, 'BREADTH_TRANSFER_TARGET_IDENTITY')
    need(donor.target_pin == source_reference['pin'], 'BREADTH_TRANSFER_DONOR_TARGET')
    original = donor.state_dict(); need(fingerprint(original) == expected_source, 'BREADTH_TRANSFER_SOURCE_STATE')
    weights = model.state_dict(); source_before = fingerprint(original); before = fingerprint(weights)
    pairs = mapping(model, donor); prepared = {}
    for destination, ref in pairs.items():
        need(destination in weights and ref['source'] in original, 'BREADTH_TRANSFER_FIELD')
        value = original[ref['source']]
        if ref['row'] is not None: value = value[ref['row']:ref['row']+1]
        need(value.shape == weights[destination].shape and value.dtype == weights[destination].dtype
             and bool(torch.isfinite(value).all()), 'BREADTH_TRANSFER_LAYOUT')
        prepared[destination] = value.detach().clone().to(weights[destination].device)
    untouched = {n: fingerprint(v) for n, v in weights.items() if n not in pairs}
    # Validate the complete map before the first mutation; load into model-owned storage.
    model.load_state_dict(dict(weights, **prepared), strict=True)
    actual = model.state_dict(); parameters = dict(model.named_parameters())
    need(len({actual[n].untyped_storage().data_ptr() for n in prepared}) == len(prepared),
         'BREADTH_TRANSFER_DESTINATION_ALIAS')
    for name, value in prepared.items():
        need(torch.equal(actual[name].cpu(), value.cpu()), 'BREADTH_TRANSFER_COPY')
        source = original[pairs[name]['source']]
        need(actual[name].untyped_storage().data_ptr() != source.untyped_storage().data_ptr(), 'BREADTH_TRANSFER_ALIAS')
    need(all(fingerprint(actual[n]) == h for n, h in untouched.items()), 'BREADTH_TRANSFER_UNRELATED_CHANGED')
    need(fingerprint(donor.state_dict()) == source_before, 'BREADTH_TRANSFER_SOURCE_MUTATED')
    return dict(schema='answer-breadth-weight-transfer-v1', source_state=expected_source,
        source_reference=source_reference, architecture=model.forecaster.architecture, mapping=pairs,
        before=before, after=fingerprint(actual), preserved=untouched,
        transferred_tensors=len(pairs), transferred_parameters=sum(parameters[n].numel() for n in pairs if n in parameters),
        discarded_source_count_tensors=sum(n.startswith('counts.') for n in original),
        optimizer_policy='new_task_warm_start_reset_adamw_not_continuation',
        shared_function_preservation=model.forecaster.architecture == 'shared', retention_guarantee=False)
