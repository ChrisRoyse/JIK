"""One exact scalar read per frozen placement head at each update boundary.

The loss still performs its original arithmetic and validation. Every lookup checks
live tensor metadata; the closing value read catches mutations without a version bump.
This optional scope does not change an already-pinned training recipe.
"""
from contextlib import contextmanager
import math
import struct
from types import MethodType

import torch
from model.native_stochastic_placement import NativeStochasticPlacement
from text.constellation_corpus import need


class VarianceScalars:
    def __init__(self,head):
        need(isinstance(head,NativeStochasticPlacement) and 'variances' not in vars(head)
             and type(head).variances is NativeStochasticPlacement.variances,'VARIANCE_SCOPE_METHOD')
        self.head=head;self.roster=tuple(head.roster);self.closed=False
        self.calls=self.batch_reads=self.scalar_reads=0
        values=self.tensors();self.identity=self.stamp(values);self.values=self.batch(values)
        need(all(math.isfinite(v) and v>0 for v in self.values),'VARIANCE_SCOPE_INVALID_VARIANCE')
        self.seal=tuple(struct.pack('!d',v) for v in self.values)

    def tensors(self):
        need(tuple(self.head.roster)==self.roster and self.roster,'VARIANCE_SCOPE_ROSTER_CHANGED')
        keys=tuple('variance_'+str(i) for i in range(len(self.roster)))
        need(all(k in self.head._buffers for k in keys),'VARIANCE_SCOPE_MISSING_BUFFER')
        values=tuple(self.head._buffers[k] for k in keys)
        need(all(isinstance(x,torch.Tensor) and x.numel()==1 and not x.requires_grad
                 and x.dtype in (torch.float32,torch.float64) and x.device.type in ('cpu','cuda')
                 for x in values),'VARIANCE_SCOPE_LAYOUT')
        need(len({(x.device,x.dtype) for x in values})==1,'VARIANCE_SCOPE_LAYOUT')
        return values

    @staticmethod
    def stamp(values):
        return tuple((id(x),x._version,x.data_ptr(),tuple(x.shape),tuple(x.stride()),x.dtype,x.device)
                     for x in values)

    def batch(self,values):
        with torch.no_grad():result=tuple(torch.stack([x.reshape(()) for x in values]).cpu().tolist())
        self.batch_reads+=1;self.scalar_reads+=len(values)
        return result

    def fresh(self):
        need(not self.closed,'VARIANCE_SCOPE_CLOSED')
        values=self.tensors()
        need(self.stamp(values)==self.identity,'VARIANCE_SCOPE_BUFFER_CHANGED')
        return values

    def read(self):
        self.fresh();self.calls+=1
        return dict(zip(self.roster,self.values))

    def close(self):
        try:
            actual=self.batch(self.fresh())
            need(tuple(struct.pack('!d',v) for v in actual)==self.seal,'VARIANCE_SCOPE_VALUES_CHANGED')
        finally:self.closed=True

    def report(self):
        need(self.closed,'VARIANCE_SCOPE_NOT_CLOSED')
        return dict(roster=self.roster,values=self.values,calls=self.calls,batch_reads=self.batch_reads,
                    scalar_reads=self.scalar_reads,closed=self.closed)


def bound(cache):
    def variances(head):
        need(head is cache.head,'VARIANCE_SCOPE_WRONG_HEAD')
        return cache.read()
    return MethodType(variances,cache.head)


@contextmanager
def frozen_variances(model):
    heads=tuple(model.forecaster.placement.values())
    need(heads,'VARIANCE_SCOPE_NO_HEADS')
    # Complete every entry check before replacing any method.
    caches=tuple(VarianceScalars(head) for head in heads)
    methods=tuple(bound(cache) for cache in caches)
    for head,method in zip(heads,methods):head.variances=method
    try:yield caches
    finally:
        intact=all(vars(head).get('variances') is method for head,method in zip(heads,methods))
        failures=[]
        for cache in caches:
            try:cache.close()
            except BaseException as exc:failures.append(exc)
        for head in heads:
            if 'variances' in vars(head):del head.variances
        need(intact,'VARIANCE_SCOPE_OWNERSHIP_CHANGED')
        if failures:raise failures[0]
