"""Experimental native-only checkpoints; region/count gradients stay connected."""
from contextlib import contextmanager
from types import MethodType

import torch
from torch.utils.checkpoint import checkpoint

from model import answer_full_history as learner
from model import captured_answer_backward as captured
from model.native_placement_training import NativePlacementForecaster
from model.native_stochastic_placement import energy_loss
from model.scalar_answer_checkpoint import STORAGE,Replay,fresh,versions
from text.constellation_corpus import need


def native_checkpoint(model,placement,context,codes,counts,targets,noise,target,owner,pin):
    roster=model.roster;n=len(roster);b=len(targets);replay=Replay(owner)
    def calculate(h,c,z,*values):
        fresh(model,pin);first=not torch.is_grad_enabled();replay.calls+=1
        need(replay.calls==(1 if first else 2),'SCALAR_HEAD_REPLAY_ORDER')
        count=dict(zip(roster,values[:n]))
        frames=tuple(dict(zip(roster,values[n+i*n:n+(i+1)*n])) for i in range(b))
        h=h.detach().requires_grad_(True) if first else h
        pack,unpack=(replay.record,replay.impossible) if first else (replay.retain,owner.unpack)
        with torch.enable_grad(),torch.autograd.graph.saved_tensors_hooks(pack,unpack):
            samples=placement(h,c,count,z,target=target)
            value=energy_loss(samples,frames,placement.variances())['loss']
        if first:
            replay.primal=value.detach().clone();return value.detach()
        replay.finish(value);return value
    value=checkpoint(calculate,context,codes,noise,*(counts[k] for k in roster),
        *(row[k] for row in targets for k in roster),use_reentrant=True,preserve_rng_state=True)
    def before_backward(gradient):
        fresh(model,pin);return gradient
    value.register_hook(before_backward)
    return value,replay


@contextmanager
def scalar_heads(model):
    need(captured.ACTIVE and learner.backward is captured.backward,'SCALAR_HEAD_CAPTURE_REQUIRED')
    need(learner.SavedStorage is STORAGE,'SCALAR_HEAD_ALREADY_INSTALLED')
    head=model.forecaster;original=vars(head).get('losses')
    need(original is not None and getattr(getattr(original,'__func__',None),'__module__',None)=='model.guarded_transient_answer'
         and type(head).losses is NativePlacementForecaster.losses,'SCALAR_HEAD_RUNTIME')
    owners=[];replays=[]
    def owner(*args,**kwargs):
        need(not owners,'SCALAR_HEAD_MULTIPLE_UPDATES')
        value=STORAGE(*args,**kwargs);owners.append(value);return value
    def loss(this,hidden,codes,counts,targets,noise,*,target):
        need(this is head and len(owners)==1 and hidden.requires_grad,'SCALAR_HEAD_CONTEXT')
        pin=versions(this);rows=[]
        for name,(readout,context) in this.contexts(hidden).items():
            # Preserve the original discrete graph and original arithmetic order.
            discrete=readout.loss(context,codes,counts)
            native,replay=native_checkpoint(this,this.placement[name],context,codes,counts,targets,
                noise,target,owners[0],pin)
            replays.append(replay);rows.append(discrete['loss']+native)
        return dict(loss=torch.stack(rows).mean())
    replacement=MethodType(loss,head);learner.SavedStorage=owner;head.losses=replacement
    try:yield replays
    finally:
        intact=learner.SavedStorage is owner and vars(head).get('losses') is replacement
        learner.SavedStorage=STORAGE;head.losses=original
        need(intact,'SCALAR_HEAD_OWNERSHIP_CHANGED')
