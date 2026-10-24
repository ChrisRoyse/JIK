"""Owned decoder inputs from a predicted answer atom; no reference target API."""
import re

import torch

from model.constellation import Constellation
from model.native import NativeInput
from text.constellation_corpus import canonical, digest, need


def predicted_atom(event, specs, config, *, region_pin, region_sizes, trace_pin, position):
    need(set(event) == {'kind','proposal','index','slots','codes','counts','stop_logits',
                       'feedback_index','source','validity'}, 'ANSWER_SURFACE_EVENT_FIELDS')
    need(event['kind'] == 'constellation' and event['source'] == 'predicted'
         and event['validity'] == 'unassessed' and type(position) is int
         and position >= 0 and type(event['index']) is int and event['index'] == position,
         'ANSWER_SURFACE_POSITION')
    need(type(event['proposal']) is str and bool(event['proposal'])
         and type(event['feedback_index']) is int and event['feedback_index'] > position,
         'ANSWER_SURFACE_LINEAGE')
    need(all(type(x) is str and re.fullmatch('[0-9a-f]{64}', x) for x in [region_pin, trace_pin]),
         'ANSWER_SURFACE_BINDING')
    names = {s.name for s in specs}
    need(set(event['slots']) == set(event['counts']) == names, 'ANSWER_SURFACE_ROSTER')
    codes = event['codes']; stop = event['stop_logits']
    need(codes.dtype == torch.int64 and codes.shape == (1, len(region_sizes))
         and all(type(n) is int and n > 0 and 0 <= int(codes[0,i]) < n
                 for i,n in enumerate(region_sizes)), 'ANSWER_SURFACE_REGION')
    need(stop.shape == (1,2) and bool(torch.isfinite(stop).all()) and int(stop.argmax(-1)) == 0,
         'ANSWER_SURFACE_STOP_CONFLICT')
    row = digest(canonical(dict(trace=trace_pin, proposal=event['proposal'], position=position)))
    region = region_pin+':'+'.'.join(str(x) for x in codes[0].tolist())
    slots = {}
    for spec in specs:
        v, count = event['slots'][spec.name], event['counts'][spec.name]
        need(v.dtype == torch.float32 and v.ndim == 2 and v.shape[1] == spec.dims
             and not v.requires_grad and bool(torch.isfinite(v).all()), 'ANSWER_SURFACE_VALUES')
        need(count.dtype == torch.int64 and count.shape == (1,) and int(count[0]) == len(v) > 0,
             'ANSWER_SURFACE_COUNT')
        slots[spec.name] = NativeInput(spec.identity, (row,), v.detach().clone()[None],
            torch.zeros((1,len(v)), dtype=torch.bool, device=v.device),
            torch.zeros(1, dtype=torch.bool, device=v.device))
    return Constellation(slots, 'predicted', (region,)).validate(specs, config)
