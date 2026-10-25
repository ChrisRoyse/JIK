"""Recompute a typed energy score only for its first-order vector-Jacobian product.

Every needed gradient returns to the original sample tensor. Nothing is detached
from answer history. This optional eager runtime establishes the buffer lifetime
required by a later CUDA replay implementation; it creates no CUDA graphs.
"""
from contextlib import contextmanager

import torch

from model.energy_score_program import ORIGINAL, prepare, training
from text.constellation_corpus import need

ACTIVE = False


def outputs(value, roster):
    return (value['loss'], value['joint'], *(value['lanes'][n] for n in roster))


def result(values, roster):
    return dict(loss=values[0], joint=values[1], lanes=dict(zip(roster, values[2:])))


class EnergyVJP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, owner, program, record, *inputs):
        ctx.owner, ctx.program, ctx.record = owner, program, record
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(*inputs)
        return outputs(program(*inputs), program.roster)

    @staticmethod
    def backward(ctx, *derivatives):
        need(not ctx.owner.closed, 'ENERGY_TRANSIENT_CLOSED')
        need(not torch.is_grad_enabled(), 'ENERGY_TRANSIENT_HIGHER_ORDER_UNSUPPORTED')
        needed = ctx.needs_input_grad[3:]
        # Detached temporary leaves terminate only this local reconstruction.
        # Its returned VJPs reconnect to every original input through this node.
        inputs = tuple(x.detach().requires_grad_(yes) for x, yes in zip(ctx.saved_tensors, needed))
        with torch.enable_grad():
            values = outputs(ctx.program(*inputs), ctx.program.roster)
            selected = tuple((v, g) for v, g in zip(values, derivatives) if g is not None)
            need(selected, 'ENERGY_TRANSIENT_NO_DERIVATIVE')
            gradients = torch.autograd.grad(tuple(v for v, _ in selected),
                tuple(x for x, yes in zip(inputs, needed) if yes),
                tuple(g for _, g in selected), create_graph=False, allow_unused=True)
        cursor = iter(gradients)
        ctx.record['backwards'] += 1
        return (None, None, None, *(next(cursor) if yes else None for yes in needed))


class CheckedEnergy:
    def __init__(self):
        self.records = []; self.closed = False

    def __call__(self, samples, targets, variances, **kwargs):
        need(not self.closed, 'ENERGY_TRANSIENT_CLOSED')
        program, inputs = prepare(samples, targets, variances, **kwargs)
        transient = torch.is_grad_enabled() and any(x.requires_grad for x in inputs)
        record = dict(program=program.description(), completed=False, transient=transient, backwards=0)
        self.records.append(record)
        value = result(EnergyVJP.apply(self, program, record, *inputs), program.roster) if transient else program(*inputs)
        record['completed'] = True
        return value


@contextmanager
def checked_energy():
    global ACTIVE
    need(not ACTIVE and training.energy_loss is ORIGINAL, 'ENERGY_TRANSIENT_ALREADY_INSTALLED')
    value = CheckedEnergy(); ACTIVE = True; training.energy_loss = value
    try: yield value
    finally:
        intact = training.energy_loss is value
        training.energy_loss = ORIGINAL; ACTIVE = False; value.closed = True
        need(intact, 'ENERGY_TRANSIENT_OWNERSHIP_CHANGED')
