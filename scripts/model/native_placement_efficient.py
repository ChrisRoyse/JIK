"""Same placement updates with typed batched transport and fewer host synchronizations."""
import torch
from torch import nn

from model import native_placement_training as base
from model.native_branch_predictor import BranchCache
from model.native_device_transport import transfer, finite_tensors, EfficientNativePrefixStream
from model.native_device_transport import validate_optimizer as device_optimizer
from model.native_placement_checked import state_census
from model.native_population_training import cpu, fingerprint
from text.constellation_corpus import need


class IndependentStream:
    def __init__(self, encoder, projectors, *, cache_bytes):
        need(type(cache_bytes) is int and cache_bytes >= len(encoder.lanes), 'BRANCH_CACHE_ALLOWANCE')
        self.encoder = encoder
        self.streams = {n: EfficientNativePrefixStream(branch, nn.ModuleDict({n: projectors[n]}),
                       cache_bytes=cache_bytes//len(encoder.lanes)) for n, branch in encoder.lanes.items()}

    def step(self, frames, *, context, offset, state=None, detach_history):
        names = {s.name for s in self.encoder.specs}
        need(isinstance(frames, (list, tuple)) and frames and all(set(f) == names for f in frames),
             'BRANCH_NATIVE_ROSTER')
        values = []
        for f in frames:
            for spec in self.encoder.specs:
                x = f[spec.name]
                need(isinstance(x, torch.Tensor) and x.ndim == 2 and len(x) > 0
                     and x.shape[1] == spec.dims and x.dtype == torch.float32
                     and x.device == self.encoder.lanes[spec.name].query.device and not x.requires_grad,
                     'BRANCH_NATIVE_VALUES')
                values.append(x)
        need(finite_tensors(values), 'BRANCH_NATIVE_VALUES')
        need(state is None or (isinstance(state, BranchCache) and state.owner == id(self)
             and len(state.states) == len(self.streams)), 'BRANCH_CACHE_OWNER')
        caches = (None,)*len(self.streams) if state is None else state.states
        for stream, cache in zip(self.streams.values(), caches):
            stream.check_cache(cache, context, offset, detach_history)
        outputs, updated = [], []
        for (name, stream), cache in zip(self.streams.items(), caches):
            value, new = stream.step(tuple({name: f[name]} for f in frames), context=context,
                                      offset=offset, state=cache, detach_history=detach_history)
            outputs.append(value); updated.append(new)
        return torch.cat(outputs, dim=1), BranchCache(id(self), tuple(updated))


def stream_for(model):
    p = model.predictor
    cls = EfficientNativePrefixStream if model.architecture == 'shared' else IndependentStream
    return cls(p.encoder, p.projectors, cache_bytes=p.cache_bytes)


def validate_optimizer(model, optimizer, policy, step):
    state_census(model, optimizer, step)
    device_optimizer(model, optimizer, policy, step)


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
