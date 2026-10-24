"""Complete-byte inversion steps, with source-bound optimizer continuation."""
from dataclasses import replace
import io
import os

import torch

from model.native_population_training import cpu, fingerprint
from model.stream_surface import StreamSurfaceDecoder
from model.surface_device_runtime import evaluate
from senses.contract import digest
from text.constellation_corpus import file_sha, need

POLICY = dict(epochs=1, atoms=128, mode='gathered', rows=128, lr=1e-4,
              checkpoint_steps=256, weighting='mean_of_complete_atom_byte_means',
              order='ascending_TRAIN_occurrence', initialization='scratch_20261023',
              dtype='float32', tf32=False, label='source_reconstruction_not_world_truth')


def create(config, specs, initial, device):
    model = StreamSurfaceDecoder(config, specs).to(device)
    model.load_state_dict(initial, strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=POLICY['lr'], foreach=False)
    return model, optimizer


def check_state(model, optimizer, step):
    values = cpu(model.state_dict()); moments = cpu(optimizer.state_dict())
    need(all(bool(x.isfinite().all()) for x in values.values()), 'INVERSION_PARAMETER_FINITE')
    need(len(moments['state']) == (len(list(model.parameters())) if step else 0),
         'INVERSION_MOMENT_COVERAGE')
    for v in moments['state'].values():
        need(set(v) == {'step', 'exp_avg', 'exp_avg_sq'} and float(v['step']) == step
             and all(bool(x.isfinite().all()) for x in v.values()), 'INVERSION_MOMENT_STATE')
    return values, moments


def update(model, optimizer, examples, step):
    need(0 < len(examples) <= POLICY['atoms'] and all(e.side == 'train' for e in examples),
         'INVERSION_TRAIN_ONLY')
    before, _ = check_state(model, optimizer, step)
    native = fingerprint([{n:x.values for n,x in e.native.items()} for e in examples])
    model.train(); optimizer.zero_grad(set_to_none=True)
    scores, profile = evaluate(model, examples, POLICY['mode'], POLICY['rows'], backward=True)
    gradients = cpu({n:p.grad for n,p in model.named_parameters()})
    need(all(v is not None and bool(v.isfinite().all()) for v in gradients.values()),
         'INVERSION_GRADIENT_FINITE')
    branches = {}
    for s in model.specs:
        for prefix in ['projectors.'+s.name+'.', 'blocks.0.branches.'+s.name+'.']:
            values = {n:v for n,v in gradients.items() if n.startswith(prefix)}
            need(values and any(bool(v.abs().any()) for v in values.values()), 'INVERSION_DISCONNECTED')
            branches[prefix] = fingerprint(values)
    optimizer.step(); after, moments = check_state(model, optimizer, step+1)
    need(native == fingerprint([{n:x.values for n,x in e.native.items()} for e in examples]),
         'INVERSION_NATIVE_CHANGED')
    row = dict(step=step+1, occurrences=[e.occurrence for e in examples],
        input=digest([dict(occurrence=e.occurrence, uid=e.uid, raw=e.raw.hex()) for e in examples]),
        native=native, before=fingerprint(before), after=fingerprint(after),
        gradient=fingerprint(gradients), branch_gradients=branches, optimizer=fingerprint(moments),
        scores=fingerprint(scores), positions=sum(len(e.raw)+1 for e in examples), objective=profile['objective'])
    return row, profile


def checkpoint(path, model, optimizer, binding, cursor, step, ledger):
    values, moments = check_state(model, optimizer, step)
    state = dict(schema='surface-inversion-state-v1', model=values, optimizer=moments,
                 binding=binding, cursor=cursor, step=step, ledger=ledger)
    buffer = io.BytesIO(); torch.save(state, buffer)
    with path.open('xb') as f:
        f.write(buffer.getvalue()); f.flush(); os.fsync(f.fileno())
    need(fingerprint(torch.load(path, map_location='cpu', weights_only=True)) == fingerprint(state),
         'INVERSION_CHECKPOINT_READBACK')
    return dict(path=str(path), pin=file_sha(path), state=fingerprint(state), cursor=cursor, step=step, ledger=ledger)


def restore(ref, model, optimizer, binding, *, size):
    need(file_sha(ref['path']) == ref['pin'], 'INVERSION_CHECKPOINT_PIN')
    state = torch.load(ref['path'], map_location='cpu', weights_only=True)
    need(state['schema'] == 'surface-inversion-state-v1' and state['binding'] == binding,
         'INVERSION_CHECKPOINT_BINDING')
    need(type(state['cursor']) is int and 0 <= state['cursor'] <= size
         and type(state['step']) is int and 0 <= state['step']
         and all(state[n] == ref[n] for n in ['cursor', 'step', 'ledger'])
         and fingerprint(state) == ref['state'], 'INVERSION_CHECKPOINT_CURSOR')
    model.load_state_dict(state['model'], strict=True); optimizer.load_state_dict(state['optimizer'])
    values, moments = check_state(model, optimizer, state['step'])
    need(fingerprint(values) == fingerprint(state['model']) and fingerprint(moments) == fingerprint(state['optimizer']),
         'INVERSION_CONTINUATION_CHANGED')
    return state['cursor'], state['step'], state['ledger']


def paired(model, examples, donors):
    need(len(examples) == len(donors) and all(d.side == 'train' and e.uid != d.uid
         for e,d in zip(examples, donors)), 'INVERSION_DONOR')
    model.eval()
    with torch.no_grad():
        correct, a = evaluate(model, examples, POLICY['mode'], POLICY['rows'])
        shuffled, b = evaluate(model, [replace(e, native=d.native) for e,d in zip(examples, donors)],
                               POLICY['mode'], POLICY['rows'])
    return dict(correct=correct, shuffled=shuffled), dict(correct=a, shuffled=b)
