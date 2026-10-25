"""Capture the existing training derivative instead of traversing its graph twice.

The distinct final-END derivative remains an independent autograd call. Identity
hooks observe the total backward and return its tensors unchanged. This candidate
does not change the original learner or any already-pinned training recipe.
"""
from contextlib import contextmanager
import time

import torch
from model import answer_full_history as learner
from model.native_population_training import cpu
from text.constellation_corpus import need

ORIGINAL=learner.backward
ACTIVE=False


def credit(first,ends,gradients,roster):
    need(set(first)==set(ends)==set(gradients)==set(roster),'CAPTURE_BACKWARD_MISSING_CONTEXT')
    result={}
    for name,value in first.items():
        e,g=ends[name],gradients[name]
        need(value.requires_grad and isinstance(e,torch.Tensor) and isinstance(g,torch.Tensor)
             and e.shape==g.shape==value.shape and e.dtype==g.dtype==value.dtype,
             'CAPTURE_BACKWARD_CONTEXT_LAYOUT')
        need(bool(e.isfinite().all()) and bool(g.isfinite().all()) and bool(e.abs().any())
             and bool(g.abs().any()),'FULL_HISTORY_DISCONNECTED_CONTEXT')
        result[name]=dict(value=cpu(value),requires_grad=True,end_gradient=cpu(e),gradient=cpu(g))
    return result


def backward(model,opt,item,step,policy,request,target,full):
    need(ACTIVE and learner.backward is backward,'CAPTURE_BACKWARD_NOT_INSTALLED')
    if not full:return ORIGINAL(model,opt,item,step,policy,request,target,full)
    first={};captured={};handles=[];phases={};start=time.perf_counter()
    def projected(name):
        def hook(module,args,value):
            if name not in first:first[name]=value
        return hook
    def derivative(name):
        def hook(gradient):
            need(name not in captured,'CAPTURE_BACKWARD_DUPLICATE_CONTEXT')
            captured[name]=cpu(gradient)
            return gradient
        return hook
    for name,projection in model.forecaster.predictor.projectors.items():
        handles.append(projection.register_forward_hook(projected(name)))
    try:
        total,end,records=learner.forward(model,item,step,policy,request,target)
        phases['forward']=time.perf_counter()-start;start=time.perf_counter()
        need(set(first)==set(model.roster) and all(t.requires_grad for t in first.values()),'FULL_HISTORY_EARLY_GRAPH')
        end_values=torch.autograd.grad(end,tuple(first.values()),retain_graph=True)
        ends={n:cpu(v) for n,v in zip(first,end_values,strict=True)};del end_values
        for name,value in first.items():handles.append(value.register_hook(derivative(name)))
        phases['credit_probe']=time.perf_counter()-start;start=time.perf_counter()
        total.backward()
        gradients=cpu({n:p.grad for n,p in model.named_parameters()})
        context=credit(first,ends,captured,model.roster)
        phases['backward']=time.perf_counter()-start
        return dict(records=records),gradients,context,phases
    finally:
        for handle in handles:handle.remove()
        first.clear();captured.clear()


@contextmanager
def captured_backward():
    global ACTIVE
    need(not ACTIVE and learner.backward is ORIGINAL,'CAPTURE_BACKWARD_ALREADY_INSTALLED')
    ACTIVE=True;learner.backward=backward
    try:yield
    finally:
        intact=learner.backward is backward
        learner.backward=ORIGINAL;ACTIVE=False
        need(intact,'CAPTURE_BACKWARD_OWNERSHIP_CHANGED')
