"""Extend a diagnostic reference view with complete captured proposal outputs."""
import copy

import numpy as np
import torch

from claims.codata_source import check
from claims.episodic_memory import digest
from model.constellation import ordered_distance
from model.field_world import FieldWorld
from model.native import NativeInput
from model import tool_context_regions


def extend(prior, model, provider, observations, provenance):
    check(observations and len(observations) == len(set(observations)), 'PROPOSAL_REGION_ROSTER')
    targets = [provider.observed(k,model.config,model.core.initial.device) for k in observations]
    check(all(t.origin == 'observed' for t in targets), 'PROPOSAL_REGION_OBSERVED')
    references = {}
    for spec in model.specs:
        old = prior.references()[spec.name]
        pieces = [old]+[t.slots[spec.name] for t in targets]
        width = max(p.values.shape[1] for p in pieces)
        count = sum(len(p.rows) for p in pieces)
        values = old.values.new_zeros(count,width,spec.dims)
        padding = torch.ones(count,width,dtype=torch.bool,device=values.device)
        rows, cursor = [], 0
        for piece in pieces:
            end = cursor+len(piece.rows)
            values[cursor:end,:piece.values.shape[1]] = piece.values
            padding[cursor:end,:piece.padding.shape[1]] = piece.padding
            rows.extend(piece.rows)
            cursor = end
        references[spec.name] = NativeInput(spec.identity,tuple(rows),values,padding,padding.all(1))
        references[spec.name].check(spec,model.config)
    desc = dict(schema='proposal-training-regions-v1',prior=prior.world_id,
        observations=[provider.checked(k) for k in observations],provenance=provenance,
        scope='post_observation_training_only',scientific_admission=False)
    with torch.random.fork_rng(devices=[]):
        view = FieldWorld(model.core,references,world_source_id=digest(desc))
    view.to(model.core.initial.device).eval()
    tool_context_regions.attach(view,model)
    view.context_history = prior.context_history
    bank = copy.deepcopy(prior.execution_anchor_bank)
    first = len(bank['rows'])
    next_unit = max(r['unit'] for r in bank['rows'])+1
    for i,observation in enumerate(observations):
        actual = provider.observations.read(observation)
        code = actual['result']['returncode']
        check(type(code) is int,'PROPOSAL_REGION_RETURNCODE')
        bank['rows'].append(dict(unit=next_unit+i,unit_scope=digest(desc),outcome=int(code != 0),
            observation=observation,observation_set=provider.observations.pin,native=provider.checked(observation),
            returncode=code,finished=actual['finished']))
    bank['pin'] = digest({k:v for k,v in bank.items() if k != 'pin'})
    view.execution_anchor_bank = bank
    view.check_reference()
    return view, dict(definition=desc,world=view.world_id,first=first,bank=bank,
                      regions=len(bank['rows']))


def costs(view, target):
    result = {}
    for spec in view.specs:
        slot = target[spec.name]
        values = slot.values[0,~slot.padding[0]].cpu().numpy().astype(np.float64)
        values /= np.linalg.norm(values,axis=1,keepdims=True)
        reference = view.references()[spec.name]
        distances = []
        for i in range(len(reference.rows)):
            old = reference.values[i,~reference.padding[i]].cpu().numpy().astype(np.float64)
            check(len(values)*len(old) <= view.config.max_attention_elements,'PROPOSAL_REGION_COST_BUDGET')
            old /= np.linalg.norm(old,axis=1,keepdims=True)
            distances.append(ordered_distance(np.clip(values@old.T,-1,1)))
        result[spec.name] = (spec.identity,torch.tensor(distances,dtype=torch.float32,device=slot.values.device))
    return result
