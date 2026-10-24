"""Complete-answer teacher learning with credit through every earlier position.

The frozen observations remain immutable. Only the learner's projections, causal
attention and region/native/END heads receive gradients. Prior recipes stay intact.
"""
import time

import torch
from model.native_answer_sequence import versions
from model.native_conditioned_answer import check
from model.native_device_transport import transfer
from model.native_placement_checked import validate_optimizer
from model.native_population_training import cpu, fingerprint
from model.native_requested_answer import update as detached_update
from model.native_request_input import NativeRequest
from model.saved_gradient_storage import SavedStorage
from text.constellation_corpus import need


def forward(model, item, step, policy, request, target):
    schedule = check(item)
    need(isinstance(request, NativeRequest) and request.observed == len(item['context'])
         and request.pin == item['provenance']['request'], 'FULL_HISTORY_REQUEST')
    device = next(model.parameters()).device
    frames = transfer((*item['context'], *item['answers']), device)
    c, a = len(item['context']), len(item['answers'])
    codes = item['codes'].to(device); counts = {n:x.to(device) for n,x in item['counts'].items()}
    noise = torch.randn(a, policy['samples'], policy['noise_width'], device=device,
        generator=torch.Generator(device=device).manual_seed(policy['noise_seed']+step))
    stream = model.stream(request); state = None; losses = []; records = []
    for position, frame in enumerate(frames):
        hidden, state = stream.step((frame,), context='requested-answer:'+str(item['document']),
                                   offset=position, state=state, detach_history=False)
        if position < c-1: continue
        index = position-(c-1); row = schedule[index]
        stop = model.stop_loss(hidden, row['end']); loss = stop['loss']/(a+1); continuation = None
        if not row['end']:
            continuation = model.forecaster.losses(hidden, codes[index:index+1],
                {n:x[index:index+1] for n,x in counts.items()}, frames[position+1:position+2],
                noise[index:index+1], target=target)
            loss = loss+continuation['loss']/a
        need(bool(torch.isfinite(loss)), 'FULL_HISTORY_NONFINITE')
        losses.append(loss)
        records.append(dict(**row, stop=cpu(stop), continuation=None if continuation is None else
                            float(continuation['loss'].detach()), weighted_loss=float(loss.detach())))
    need(len(losses) == a+1 and sum(r['end'] for r in records) == 1, 'FULL_HISTORY_COVERAGE')
    return torch.stack(losses).sum(), losses[-1], records


def backward(model, opt, item, step, policy, request, target, full):
    first = {}; handles = []; phases = {}; start = time.perf_counter()
    def capture(name):
        def hook(module, args, output):
            if name not in first: first[name] = output
        return hook
    for name, projection in model.forecaster.predictor.projectors.items():
        handles.append(projection.register_forward_hook(capture(name)))
    try:
        if full:
            total, end, records = forward(model, item, step, policy, request, target)
            phases['forward'] = time.perf_counter()-start; start = time.perf_counter()
            need(set(first) == set(model.roster) and all(t.requires_grad for t in first.values()),
                 'FULL_HISTORY_EARLY_GRAPH')
            ends = torch.autograd.grad(end, tuple(first.values()), retain_graph=True)
            early = torch.autograd.grad(total, tuple(first.values()), retain_graph=True)
            credit = {n:dict(value=cpu(t), requires_grad=True, end_gradient=cpu(e), gradient=cpu(g))
                      for (n,t),e,g in zip(first.items(), ends, early)}
            need(all(bool(r['end_gradient'].abs().any()) and bool(r['gradient'].abs().any())
                     for r in credit.values()), 'FULL_HISTORY_DISCONNECTED_CONTEXT')
            phases['credit_probe'] = time.perf_counter()-start; start = time.perf_counter()
            total.backward()
            gradients = cpu({n:p.grad for n,p in model.named_parameters()})
            row = dict(records=records)
        else:
            row, gradients = detached_update(model, opt, item, step, policy, request=request, target=target)
            credit = {n:dict(value=cpu(t), requires_grad=t.requires_grad,
                            end_gradient=None, gradient=None) for n,t in first.items()}
            need(set(first) == set(model.roster) and all(not t.requires_grad for t in first.values()),
                 'FULL_HISTORY_DETACHED_CONTEXT')
        phases['backward' if full else 'detached_update'] = time.perf_counter()-start
        return row, gradients, credit, phases
    finally:
        for handle in handles: handle.remove()
        first.clear()


def update(model, optimizer, item, step, policy, *, request, target, full):
    need(type(full) is bool, 'FULL_HISTORY_MODE')
    check(item)
    need(isinstance(request, NativeRequest) and request.observed == len(item['context'])
         and request.pin == item['provenance']['request'], 'FULL_HISTORY_REQUEST')
    validate_optimizer(model, optimizer, policy, step)
    before = fingerprint(model.state_dict()); frozen = versions(target)
    optimizer.zero_grad(set_to_none=True)
    storage = SavedStorage(policy['working_bytes'], next(model.parameters()).device)
    try:
        with storage.hooks():
            row, gradients, credit, phases = backward(model, optimizer, item, step, policy, request, target, full)
        need(storage.current == 0, 'FULL_HISTORY_RETAINED_GRAPH')
        need(all(g is not None and bool(g.isfinite().all()) for g in gradients.values()),
             'FULL_HISTORY_GRADIENTS')
        prefixes = ['forecaster.predictor.projectors.'+n+'.' for n in model.roster]
        prefixes += ['request_projection.lanes.'+n+'.' for n in model.roster]
        prefixes += ['ending.'+p+'.'+n+'.' for p in model.proposals for n in model.roster]
        branches = {}
        for prefix in prefixes:
            group = {n:g for n,g in gradients.items() if n.startswith(prefix)}
            need(group and any(bool(g.abs().any()) for g in group.values()), 'FULL_HISTORY_BRANCH:'+prefix)
            branches[prefix] = fingerprint(group)
        need(versions(target) == frozen, 'FULL_HISTORY_FROZEN_CHANGED'); check(item)
        if full:
            tick = time.perf_counter(); optimizer.step(); phases['optimizer'] = time.perf_counter()-tick
        validate_optimizer(model, optimizer, policy, step+1)
        need(all(bool(x.isfinite().all()) for x in model.state_dict().values()), 'FULL_HISTORY_PARAMETERS')
    except BaseException:
        optimizer.zero_grad(set_to_none=True)
        raise
    return dict(document=item['document'], request=request.pin, native=item['native_pin'],
        labels=item['label_pin'], binding=item['binding'], full=full, before=before,
        after=fingerprint(model.state_dict()), optimizer=fingerprint(optimizer.state_dict()),
        gradients=fingerprint(gradients), branch_gradients=branches, records=row['records'],
        decisions=len(row['records']), storage=storage.report(), phases=phases), gradients, credit
