"""Batch transfers within each instrument, preserving every frame/window byte."""
import torch

from claims.codata_source import check
from model.native_prefix_stream import NativePrefixStream


def transfer(frames, device):
    check(frames, 'NATIVE_TRANSFER_EMPTY')
    names = tuple(frames[0]); device = torch.device(device)
    check(all(tuple(f) == names for f in frames), 'NATIVE_TRANSFER_ROSTER')
    result = [dict() for _ in frames]
    for name in names:
        values = [f[name] for f in frames]
        check(all(x.ndim == 2 and x.shape[1] == values[0].shape[1] and len(x) > 0
                  and x.dtype == values[0].dtype and x.device == values[0].device
                  and not x.requires_grad for x in values), 'NATIVE_TRANSFER_LAYOUT')
        joined = torch.cat(values, dim=0)
        if joined.device.type == 'cpu' and device.type == 'cuda':
            joined = joined.pin_memory().to(device, non_blocking=True)
        else:
            joined = joined.to(device)
        for frame, value in zip(result, joined.split([len(x) for x in values])):
            frame[name] = value
    return tuple(result)


def finite_tensors(values):
    """All elements checked on their resident devices, one host read per device."""
    groups = {}
    for value in values:
        groups.setdefault(value.device, []).append(torch.isfinite(value).all())
    return all(bool(torch.stack(flags).all()) for flags in groups.values())


class EfficientNativePrefixStream(NativePrefixStream):
    """Same learned arithmetic; coalesce Boolean checks and reuse fixed positions."""
    def pack(self, frames, offset):
        encoder = self.encoder; config = encoder.config; device = encoder.query.device
        check(frames and all(set(f) == set(self.names) for f in frames), 'PREFIX_STREAM_ROSTER')
        windows = sum(len(x) for f in frames for x in f.values())
        elements = sum(x.numel() for f in frames for x in f.values())
        size = windows+len(frames)*len(self.names)
        check(size <= config.max_native_tokens and elements <= config.max_native_elements,
              'PREFIX_STREAM_CURRENT_MEMORY')
        for frame in frames:
            for spec in encoder.specs:
                value = frame[spec.name]
                check(value.ndim == 2 and value.shape[1] == spec.dims and len(value) > 0
                      and value.device == device and value.dtype == torch.float32
                      and not value.requires_grad, 'PREFIX_STREAM_NATIVE')
        # Keep instrument buffers separate. This is exact Boolean validation,
        # never an average or a shared representation of native readings.
        check(finite_tensors([torch.cat([f[n] for f in frames]) for n in self.names]), 'PREFIX_STREAM_NATIVE')
        check(not encoder.frequency.requires_grad, 'PREFIX_STREAM_TRAINABLE_FREQUENCY')
        positions = encoder.position(torch.arange(offset, offset+len(frames), device=device))
        local = {count:encoder.position(torch.arange(count, device=device))
                 for count in sorted({len(x) for f in frames for x in f.values()})}
        tokens, atoms, queries, cursor = [], [], [], 0
        for atom, frame in enumerate(frames):
            selected = []
            for lane, spec in enumerate(encoder.specs):
                values = frame[spec.name]
                # Preserve each original GEMM shape and every addition order.
                tokens.append(self.projectors[spec.name](values)+encoder.kind[lane]+positions[atom]+local[len(values)])
                atoms.extend([offset+atom]*len(values)); cursor += len(values)
                selected.append(cursor)
                tokens.append((encoder.query[lane]+encoder.kind[lane]+positions[atom])[None])
                atoms.append(offset+atom); cursor += 1
            queries.append(selected)
        return torch.cat(tokens, 0)[None], torch.tensor(atoms, device=device), torch.tensor(queries, device=device)


def validate_optimizer(model, optimizer, plan, step):
    check(type(optimizer) is torch.optim.AdamW and len(optimizer.param_groups) == 1, 'OPTIMIZER_RECIPE')
    group = optimizer.param_groups[0]; params = list(model.parameters())
    check(len(group['params']) == len(params) and all(a is b for a,b in zip(group['params'],params)), 'OPTIMIZER_ORDER')
    expected = dict(lr=plan['learning_rate'], weight_decay=plan['weight_decay'], betas=(.9,.999), eps=1e-8,
                    amsgrad=False, maximize=False, foreach=False, capturable=False, differentiable=False, fused=False)
    check(all(group[k] == v for k,v in expected.items()), 'OPTIMIZER_RECIPE')
    values = []
    for p, state in optimizer.state.items():
        check(set(state) == {'step','exp_avg','exp_avg_sq'} and float(state['step']) == step, 'OPTIMIZER_CURSOR')
        check(all(state[k].shape == p.shape and state[k].dtype == p.dtype and state[k].device == p.device
                  for k in ('exp_avg','exp_avg_sq')), 'OPTIMIZER_STATE')
        values.extend(state.values())
    check(finite_tensors(values), 'OPTIMIZER_STATE')
    check((step == 0) == (not optimizer.state), 'OPTIMIZER_CURSOR')
