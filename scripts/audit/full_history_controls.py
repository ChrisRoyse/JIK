"""Failure atomicity, saved-storage semantics and directional derivative checks."""
from copy import deepcopy
import gc

import torch
from audit.answer_surface_stream import rejected
from audit import answer_history_source as prior
from model import answer_full_history as learner
from model.native_population_training import cpu, fingerprint
from model.native_request_input import NativeRequest, legal
from model.saved_gradient_storage import SavedStorage
from text.constellation_corpus import need


def controls(state):
    model, opt = prior.make(state, 'shared'); value, request = prior.item(state, state['train'][0])
    initial = fingerprint((model.state_dict(), opt.state_dict())); out = {}
    def run(v=value, q=request, policy=None):
        return learner.update(model, opt, v, 5, policy or state['policy'], request=q, target=state['target'], full=True)
    held = next(k for k,d in state['items'].items() if d['side'] != 'train')
    out['query_fit'] = rejected(lambda:run(v=prior.item(state, held)[0]), 'CONDITIONED_NONTRAIN')
    out['wrong_request'] = rejected(lambda:run(q=NativeRequest(dict(legal(2), operation='respond_to_source_question'))),
                                    'FULL_HISTORY_REQUEST')
    for field, code in (('native', 'CONDITIONED_NATIVE_PIN'), ('labels', 'CONDITIONED_LABEL_PIN'),
                        ('binding', 'CONDITIONED_SOURCE_BINDING')):
        bad = deepcopy(value)
        if field == 'native': next(iter(bad['answers'][0].values())).add_(1)
        elif field == 'labels': bad['codes'][0,0] = (bad['codes'][0,0]+1) % 4
        else: bad['provenance']['observed'] += 1
        out[field] = rejected(lambda:run(v=bad), code)
    bad = deepcopy(state['initial']['shared']); bad['optimizer']['state'] = {}
    out['moments'] = rejected(lambda:prior.make(state, 'shared', bad),
                             'PLACEMENT_OPTIMIZER_INCOMPLETE_STATE_RESTORE_FULL_CHECKPOINT')
    out['memory'] = rejected(lambda:run(policy=dict(state['policy'], working_bytes=1)), 'GRADIENT_STORAGE_MEMORY')
    need(initial == fingerprint((model.state_dict(), opt.state_dict())), 'FULL_HISTORY_CONTROL_UPDATE')
    need(all(p.grad is None for p in model.parameters()), 'FULL_HISTORY_REFUSAL_GRADIENTS')
    # Owned copy of an actual native reading, not a replacement training example.
    tensor = next(iter(value['context'][0].values())).clone().requires_grad_()
    storage = SavedStorage(state['policy']['working_bytes'], 'cpu')
    one = storage.pack(tensor); two = storage.pack(tensor.view(-1))
    need(storage.current == tensor.untyped_storage().nbytes(), 'FULL_HISTORY_STORAGE_ALIAS')
    need(torch.equal(storage.unpack(one), tensor), 'FULL_HISTORY_STORAGE_VALUES')
    del one, two; need(storage.current == 0, 'FULL_HISTORY_STORAGE_RELEASE')
    with storage.hooks(): result = tensor.square().sum()
    with torch.no_grad(): tensor.add_(1)
    out['saved_mutation'] = rejected(result.backward, 'GRADIENT_STORAGE_MUTATED')
    del result, tensor; gc.collect()
    need(storage.current == 0 and not storage.storages, 'FULL_HISTORY_STORAGE_LEAK')
    out['storage'] = storage.report(); out['no_update_on_refusal'] = True
    return out


def finite_difference(state, architecture):
    model, opt = prior.make(state, architecture)
    doc = min(state['train'], key=lambda k:(len(state['items'][k]['frames']), k))
    value, request = prior.item(state, doc); name = model.roster[0]; first = []
    projection = model.forecaster.predictor.projectors[name]
    def capture(module, args, output):
        if not first: first.append(output)
    handle = projection.register_forward_hook(capture)
    try:
        total, end, _ = learner.forward(model, value, 5, state['policy'], request, state['target'])
        gradient, = torch.autograd.grad(total, first[0])
        direction = gradient.detach()/gradient.detach().norm()
        analytic = float((gradient*direction).sum())
        origin = float(total.detach())
    finally: handle.remove(); first.clear()
    del total, end, gradient
    probes = []
    # Declared before measurements; every step size must agree. f32 forward noise
    # is reported explicitly, not used to silently expand a tolerance.
    for epsilon in (.01, .003, .001):
        values = []
        for sign in (-1, 1):
            seen = []
            def intervene(module, args, output):
                if not seen:
                    seen.append(True)
                    return output+(sign*epsilon)*direction
            handle = projection.register_forward_hook(intervene)
            try:
                with torch.no_grad():
                    loss, _, _ = learner.forward(model, value, 5, state['policy'], request, state['target'])
                values.append(float(loss))
            finally: handle.remove()
        central = (values[1]-values[0])/(2*epsilon)
        tolerance = .0002+.03*abs(analytic)
        need(abs(central-analytic) <= tolerance, 'FULL_HISTORY_DIRECTIONAL_DERIVATIVE')
        probes.append(dict(epsilon=epsilon, losses=values, central=central, analytic=analytic,
                           absolute_error=abs(central-analytic), tolerance=tolerance))
    need(fingerprint((model.state_dict(), opt.state_dict())) ==
         fingerprint((state['initial'][architecture]['model'], state['initial'][architecture]['optimizer'])),
         'FULL_HISTORY_PROBE_CHANGED_MODEL')
    return dict(document=doc, instrument=name, direction=cpu(direction), origin=origin, probes=probes)
