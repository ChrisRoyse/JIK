"""Exact-arithmetic sibling of native_document_training, with typed GPU transport."""
import torch

from model.native_document_training import check_item, branches
from model.native_device_transport import transfer, EfficientNativePrefixStream, validate_optimizer
from model.native_population_training import cpu, fingerprint
from text.constellation_corpus import need


def update(model, optimizer, item, step, policy):
    check_item(item)
    validate_optimizer(model, optimizer, policy, step)
    before = cpu(model.state_dict()); device = model.encoder.query.device
    frames = transfer(item['frames'], device)
    codes = item['codes'].to(device); counts = {n:x.to(device) for n,x in item['counts'].items()}
    optimizer.zero_grad(set_to_none=True)
    stream = EfficientNativePrefixStream(model.encoder, model.projectors, cache_bytes=model.cache_bytes)
    state = None; total = len(codes); loss_sum = 0.
    context = 'document:'+str(item['document']); consumed = 0
    for lo in range(0, len(frames), policy['tile']):
        hi = min(lo+policy['tile'], len(frames))
        hidden, state = stream.step(frames[lo:hi], context=context, offset=lo, state=state, detach_history=True)
        stop = min(hi, total); targets = stop-lo
        if targets > 0:
            out = model.loss(hidden[:targets], codes[lo:stop], {n:x[lo:stop] for n,x in counts.items()})
            need(bool(torch.isfinite(out['loss'])), 'DOCUMENT_TRAIN_NONFINITE_LOSS')
            weighted = out['loss']*(targets/total); weighted.backward()
            loss_sum += float(weighted.detach()); consumed += targets
    need(consumed == total and state.next_atom == len(frames), 'DOCUMENT_TRAIN_INCOMPLETE_HISTORY')
    gradients = cpu({n:p.grad for n,p in model.named_parameters()})
    need(all(x is not None and bool(torch.isfinite(x).all()) for x in gradients.values()), 'DOCUMENT_TRAIN_GRADIENT')
    need(all(any(bool(x.abs().max()>0) for n,x in gradients.items() if n.startswith('projectors.'+lane+'.'))
             for lane in model.roster), 'DOCUMENT_TRAIN_DISCONNECTED_INSTRUMENT')
    optimizer.step(); validate_optimizer(model, optimizer, policy, step+1)
    after = cpu(model.state_dict()); moments = cpu(optimizer.state_dict())
    need(all(bool(torch.isfinite(x).all()) for x in after.values()), 'DOCUMENT_TRAIN_PARAMETER')
    need(len(moments['state']) == len(list(model.parameters())), 'DOCUMENT_TRAIN_MOMENT_COVERAGE')
    need(fingerprint(transfer(frames, 'cpu')) == item['native_pin'] == fingerprint(item['frames']),
         'DOCUMENT_TRAIN_NATIVE_CHANGED')
    try: stream.step(frames[-1:], context=context, offset=len(frames), state=state, detach_history=True)
    except (ValueError, RuntimeError) as error: stale = str(error)
    else: raise AssertionError('DOCUMENT_TRAIN_STALE_CACHE_ACCEPTED')
    need('WEIGHTS_CHANGED' in stale, 'DOCUMENT_TRAIN_STALE_CACHE_REASON')
    return dict(step=step+1, document=item['document'], atoms=len(frames), targets=total, loss=loss_sum,
                before=fingerprint(before), after=fingerprint(after), gradient=fingerprint(gradients),
                optimizer=fingerprint(moments), branch_before=branches(before,model.roster),
                branch_after=branches(after,model.roster), branch_gradient=branches(gradients,model.roster),
                native=item['native_pin'], labels=item['label_pin'], stale_cache_refusal=stale,
                complete_forward_history=True, backward_history='detached_prior_tiles')
