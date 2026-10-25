"""Bounded resident-device validation; model values and arithmetic are unchanged."""
import torch

from claims.codata_source import check
from model.native_placement_checked import state_census


def finite_tensors(values, *, tile_bytes=8<<20):
    """Coalesce Boolean validation work without casting or moving tensor values.

    The temporary joined buffer is a validation workspace, never a model input.
    Noncontiguous tensors keep the original shape-preserving validation path.
    """
    check(type(tile_bytes) is int and tile_bytes>0,'VALIDATION_TILE_BYTES')
    groups={};flags={}
    for value in values:
        check(isinstance(value,torch.Tensor),'VALIDATION_TENSOR')
        if not value.is_contiguous():
            flags.setdefault(value.device,[]).append(torch.isfinite(value).all())
        else:groups.setdefault((value.device,value.dtype),[]).append(value)
    for (device,_),group in groups.items():
        capacity=tile_bytes//group[0].element_size()
        check(capacity>0,'VALIDATION_TILE_BYTES')
        pending=[];count=0
        def flush():
            nonlocal pending,count
            if pending:
                flags.setdefault(device,[]).append(torch.isfinite(torch.cat(pending)).all())
                pending=[];count=0
        for value in group:
            flat=value.detach().view(-1);offset=0
            while offset<flat.numel():
                size=min(capacity-count,flat.numel()-offset)
                pending.append(flat[offset:offset+size]);count+=size;offset+=size
                if count==capacity:flush()
        flush()
    return all(bool(torch.stack(group).all()) for group in flags.values())


def validate_optimizer(model,optimizer,plan,step):
    """The existing complete AdamW census/recipe/cursor checks, batched finiteness."""
    state_census(model,optimizer,step)
    check(type(optimizer) is torch.optim.AdamW and len(optimizer.param_groups)==1,'OPTIMIZER_RECIPE')
    group=optimizer.param_groups[0];params=list(model.parameters())
    check(len(group['params'])==len(params) and all(a is b for a,b in zip(group['params'],params)),'OPTIMIZER_ORDER')
    expected=dict(lr=plan['learning_rate'],weight_decay=plan['weight_decay'],betas=(.9,.999),eps=1e-8,
                  amsgrad=False,maximize=False,foreach=False,capturable=False,differentiable=False,fused=False)
    check(all(group[k]==v for k,v in expected.items()),'OPTIMIZER_RECIPE')
    values=[]
    for p,state in optimizer.state.items():
        check(set(state)=={'step','exp_avg','exp_avg_sq'} and float(state['step'])==step,'OPTIMIZER_CURSOR')
        check(all(state[k].shape==p.shape and state[k].dtype==p.dtype and state[k].device==p.device
                  for k in ('exp_avg','exp_avg_sq')),'OPTIMIZER_STATE')
        values.extend(state.values())
    check(finite_tensors(values),'OPTIMIZER_STATE')
    check((step==0)==(not optimizer.state),'OPTIMIZER_CURSOR')
