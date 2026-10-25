"""Placement updates with identical arithmetic and bounded batched validation."""
import torch
from model import native_placement_training as base
from model.native_placement_efficient import stream_for
from model.native_device_transport import transfer
from model.native_batched_validation import finite_tensors,validate_optimizer
from model.native_population_training import cpu,fingerprint
from text.constellation_corpus import need

def update(model, optimizer, item, step, policy, *, target):
    base.check_item(item); validate_optimizer(model, optimizer, policy, step)
    device = next(model.parameters()).device; total = len(item['codes']); samples = policy['samples']
    plan = [(lo, min(lo+policy['tile'], len(item['frames'])),
             list(base.microtiles(item['frames'], lo, min(lo+policy['tile'], total), model,
                                  samples, policy['working_bytes'])))
            for lo in range(0, len(item['frames']), policy['tile'])]
    before = fingerprint(model.state_dict()); optimizer.zero_grad(set_to_none=True)
    frames = transfer(item['frames'], device)
    codes = item['codes'].to(device); counts = {n: x.to(device) for n, x in item['counts'].items()}
    noise_seed = policy['noise_seed']+step
    noise = torch.randn(total, samples, policy['noise_width'],
                        generator=torch.Generator(device=device).manual_seed(noise_seed), device=device)
    stream = stream_for(model); state = None; context = 'placement-document:'+str(item['document'])
    losses = []; used = 0
    for lo, hi, tiles in plan:
        hidden, state = stream.step(frames[lo:hi], context=context, offset=lo, state=state, detach_history=True)
        for index, (start, stop) in enumerate(tiles):
            out = model.losses(hidden[start-lo:stop-lo], codes[start:stop], {n: x[start:stop] for n, x in counts.items()},
                               frames[start+1:stop+1], noise[start:stop], target=target)
            weighted = out['loss']*((stop-start)/total)
            need(bool(torch.isfinite(weighted)), 'PLACEMENT_TRAIN_NONFINITE_LOSS')
            weighted.backward(retain_graph=index+1 < len(tiles)); used += stop-start
            losses.append(dict(start=start, stop=stop, loss=float(weighted.detach()),
                discrete={n: float(r['discrete']['loss'].detach()) for n, r in out['proposals'].items()},
                native={n: float(r['native']['loss'].detach()) for n, r in out['proposals'].items()}))
    need(used == total and base._complete(state, len(frames)), 'PLACEMENT_TRAIN_COVERAGE')
    gradients = cpu({n: p.grad for n, p in model.named_parameters()})
    need(all(g is not None and bool(torch.isfinite(g).all()) for g in gradients.values()), 'PLACEMENT_TRAIN_GRADIENT')
    for lane in model.roster:
        need(any(bool(g.abs().max() > 0) for n, g in gradients.items()
                 if n.startswith('predictor.projectors.'+lane+'.')), 'PLACEMENT_TRAIN_DISCONNECTED_HISTORY')
        for proposal in model.proposals:
            need(any(bool(g.abs().max() > 0) for n, g in gradients.items()
                     if n.startswith('placement.'+proposal+'.residual.'+lane+'.')),
                 'PLACEMENT_TRAIN_DISCONNECTED_PLACEMENT')
    optimizer.step(); validate_optimizer(model, optimizer, policy, step+1)
    need(finite_tensors(model.state_dict().values()), 'PLACEMENT_TRAIN_PARAMETER')
    base.check_item(item)
    need(fingerprint(transfer(frames, 'cpu')) == item['native_pin'], 'PLACEMENT_TRANSFER_CHANGED')
    try: stream.step(frames[-1:], context=context, offset=len(frames), state=state, detach_history=True)
    except (ValueError, RuntimeError) as error: stale = str(error)
    else: raise AssertionError('PLACEMENT_TRAIN_STALE_HISTORY_ACCEPTED')
    need('WEIGHTS_CHANGED' in stale, 'PLACEMENT_TRAIN_STALE_HISTORY_REASON')
    return dict(step=step+1, document=item['document'], targets=total, noise_seed=noise_seed,
                noise=fingerprint(noise), loss=sum(x['loss'] for x in losses), tiles=losses,
                before=before, after=fingerprint(model.state_dict()), gradient=fingerprint(gradients),
                optimizer=fingerprint(optimizer.state_dict()), branch_gradient=base.branch_pins(gradients, model),
                stale_cache=stale, native=item['native_pin'], labels=item['label_pin'],
                backward_history='detached_prior_tiles'), gradients, noise
