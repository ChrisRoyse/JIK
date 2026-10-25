"""Experimental scalar continuation checkpoints with explicit metadata replay.

Only the captured-backward complete-answer learner may consume this scope.
The normal typed forecaster API and all previously pinned recipes stay intact.
"""
from contextlib import contextmanager
from types import MethodType

import torch
from torch.utils.checkpoint import checkpoint

from model import answer_full_history as learner
from model import captured_answer_backward as captured
from model.native_answer_sequence import versions
from model.native_placement_training import NativePlacementForecaster
from text.constellation_corpus import need

STORAGE=learner.SavedStorage


def fresh(model,pin):
    need(versions(model)==pin,'SCALAR_HEAD_WEIGHTS_CHANGED')


def metadata(tensor):
    return tuple(tensor.shape),tensor.dtype,tensor.device


class Replay:
    def __init__(self,owner):
        self.owner=owner;self.expected=[];self.cursor=0;self.calls=0;self.primal=None

    def record(self,tensor):
        self.expected.append(metadata(tensor))
        return None

    @staticmethod
    def impossible(saved):
        raise ValueError('SCALAR_HEAD_FORWARD_GRAPH_REUSED')

    def retain(self,tensor):
        need(self.cursor<len(self.expected) and metadata(tensor)==self.expected[self.cursor],
             'SCALAR_HEAD_METADATA_CHANGED')
        self.cursor+=1
        return self.owner.pack(tensor)

    def finish(self,value):
        need(self.cursor==len(self.expected),'SCALAR_HEAD_METADATA_COUNT')
        need(value.shape==self.primal.shape and value.dtype==self.primal.dtype and value.device==self.primal.device
             and torch.equal(value.detach().reshape(-1).view(torch.uint8),
                             self.primal.reshape(-1).view(torch.uint8)),'SCALAR_HEAD_VALUE_CHANGED')

    def report(self):
        return dict(saved_tensors=len(self.expected),replayed_tensors=self.cursor,calls=self.calls,
                    value_checked=self.primal is not None and self.calls==2)


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
        roster=this.roster;n=len(roster);b=len(targets);pin=versions(this)
        replay=Replay(owners[0]);replays.append(replay)
        def calculate(h,c,z,*values):
            fresh(this,pin);first=not torch.is_grad_enabled();replay.calls+=1
            need(replay.calls==(1 if first else 2),'SCALAR_HEAD_REPLAY_ORDER')
            count=dict(zip(roster,values[:n]))
            frames=tuple(dict(zip(roster,values[n+i*n:n+(i+1)*n])) for i in range(b))
            # First pass records the same saved-tensor metadata but keeps no
            # head graph or intermediate arrays alive after the scalar returns.
            h=h.detach().requires_grad_(True) if first else h
            pack,unpack=(replay.record,replay.impossible) if first else (replay.retain,replay.owner.unpack)
            with torch.enable_grad(),torch.autograd.graph.saved_tensors_hooks(pack,unpack):
                value=NativePlacementForecaster.losses(this,h,c,count,frames,z,target=target)['loss']
            if first:
                replay.primal=value.detach().clone();return value.detach()
            replay.finish(value);return value
        value=checkpoint(calculate,hidden,codes,noise,*(counts[k] for k in roster),
                         *(row[k] for row in targets for k in roster),use_reentrant=True,preserve_rng_state=True)
        def before_backward(gradient):
            fresh(this,pin);return gradient
        value.register_hook(before_backward)
        return dict(loss=value)
    replacement=MethodType(loss,head);learner.SavedStorage=owner;head.losses=replacement
    try:yield replays
    finally:
        intact=learner.SavedStorage is owner and vars(head).get('losses') is replacement
        learner.SavedStorage=STORAGE;head.losses=original
        need(intact,'SCALAR_HEAD_OWNERSHIP_CHANGED')
