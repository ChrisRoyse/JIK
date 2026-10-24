"""Identical surface objectives with explicit device transfers and phase attribution."""
from dataclasses import replace
import time

import torch
from torch.nn import functional as F

from model import gathered_surface_batches as gathered
from model import mixed_surface_batches as mixed
from model.native_population_training import cpu
from model.stream_surface import StreamSurfaceDecoder, batches as original_batches
from text.constellation_corpus import need

COHORTS = (8, 32, 128, 512)
ARMS = [('reference', 'reference', None)] + [(f'{mode}-{n}', mode, n) for n in COHORTS for mode in ['mixed', 'gathered']]


def batches(examples, config, specs, mode, rows):
    need(mode in {'reference', 'mixed', 'gathered'}, 'SURFACE_DEVICE_BATCH_MODE')
    if mode == 'reference':
        need(rows is None, 'SURFACE_DEVICE_REFERENCE_ROWS')
        for i, e in enumerate(examples):
            for prefix, labels, offsets, native, items in original_batches(e.raw, e.native, config):
                records = tuple(dict(atom=i, occurrence=e.occurrence, uid=e.uid, offset=x['offset'],
                    scored=x['scored'], prefix_length=len(x['prefix'])) for x in items)
                yield mixed.MixedSurfaceBatch(prefix, labels, offsets,
                    torch.full((len(items),), len(e.raw)+1, dtype=torch.float32), native, records, len(examples))
    else:
        module = mixed if mode == 'mixed' else gathered
        for lo in range(0, len(examples), 128):
            for b in module.batches(examples[lo:lo+128], config, specs, rows=rows):
                yield replace(b, records=tuple(dict(r, atom=r['atom']+lo) for r in b.records), atoms=len(examples))


def packed(b):
    return dict(prefix=b.prefix, labels=b.labels, offsets=b.offsets, denominators=b.denominators,
        native={n: dict(identity=x.identity, rows=x.rows, values=x.values, padding=x.padding, absent=x.absent)
                for n, x in b.native.items()}, records=b.records, atoms=b.atoms)


def sync(device):
    if device.type == 'cuda': torch.cuda.synchronize(device)


def evaluate(model, examples, mode, rows, *, backward=False):
    if backward: need(all(e.side == 'train' for e in examples), 'SURFACE_DEVICE_TRAIN_ONLY')
    device = next(model.parameters()).device
    result = [dict(losses=torch.full((len(e.raw)+1,), float('nan')),
                   argmax=torch.full((len(e.raw)+1,), -1, dtype=torch.int64)) for e in examples]
    sync(device); began = tick = time.perf_counter(); objective = 0.; calls = 0
    phases = dict(packing=0., transfer=0., forward=0., scoring_backward=0., readout=0.)
    for host in batches(examples, model.config, model.specs, mode, rows):
        phases['packing'] += time.perf_counter()-tick; tick = time.perf_counter()
        b = host.to(device); sync(device)
        phases['transfer'] += time.perf_counter()-tick; tick = time.perf_counter()
        logits = model(b.prefix, b.native, offsets=b.offsets)['logits']; sync(device)
        phases['forward'] += time.perf_counter()-tick; tick = time.perf_counter()
        if mode == 'reference':
            # Keep the original reduction order, including its per-atom denominator.
            loss = F.cross_entropy(logits.reshape(-1, model.config.vocabulary), b.labels.reshape(-1),
                ignore_index=-100, reduction='sum')/float(host.denominators[0])/len(examples)
            token = F.cross_entropy(logits.reshape(-1, model.config.vocabulary), b.labels.reshape(-1),
                ignore_index=-100, reduction='none').reshape(b.labels.shape)
        else: loss, token = mixed.loss(logits, b)
        if backward: loss.backward()
        sync(device); phases['scoring_backward'] += time.perf_counter()-tick; tick = time.perf_counter()
        # One readback per tensor per batch, rather than a device synchronization per row.
        token = token.detach().cpu(); predicted = logits.detach().argmax(-1).cpu()
        objective += float(loss.detach()); calls += 1
        for row, r in enumerate(b.records):
            i = r['atom']; a, end = r['scored']; left = a-r['offset']; right = end-r['offset']
            need(bool(torch.isnan(result[i]['losses'][a:end]).all()), 'SURFACE_DEVICE_DUPLICATE_SCORE')
            result[i]['losses'][a:end] = token[row, left:right]
            result[i]['argmax'][a:end] = predicted[row, left:right]
        del logits, loss, token, predicted, b, host
        sync(device); phases['readout'] += time.perf_counter()-tick; tick = time.perf_counter()
    phases['packing'] += time.perf_counter()-tick
    need(all(bool(v['losses'].isfinite().all()) and bool((v['argmax'] >= 0).all()) for v in result),
         'SURFACE_DEVICE_MISSING_SCORE')
    seconds = time.perf_counter()-began
    need(.9 <= sum(phases.values())/seconds <= 1.000001, 'SURFACE_DEVICE_PHASE_COVERAGE')
    return result, dict(objective=objective, calls=calls, positions=sum(len(e.raw)+1 for e in examples),
                       seconds=seconds, phases=phases)


def execute(initial, config, specs, train, examples, shuffled, mode, rows, device):
    model = StreamSurfaceDecoder(config, specs).to(device); model.load_state_dict(initial, strict=True); model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, foreach=False); model.zero_grad(set_to_none=True)
    fitting, fit = evaluate(model, train, mode, rows, backward=True)
    gradients = cpu({n: p.grad for n, p in model.named_parameters()})
    need(all(v is not None and bool(v.isfinite().all()) for v in gradients.values()), 'SURFACE_DEVICE_GRADIENT_FINITE')
    need(all(any(bool(v.abs().any()) for n, v in gradients.items() if n.startswith(prefix)) for s in specs
        for prefix in ['projectors.'+s.name+'.', 'blocks.0.branches.'+s.name+'.']), 'SURFACE_DEVICE_DISCONNECTED_BRANCH')
    optimizer.step(); after = cpu(model.state_dict()); moments = cpu(optimizer.state_dict())
    with torch.no_grad():
        evaluation, score = evaluate(model, examples, mode, rows); control, shuf = evaluate(model, shuffled, mode, rows)
    artifact = dict(training=fitting, evaluation=evaluation, shuffled=control, gradients=gradients, after=after, optimizer=moments)
    del model, optimizer
    return artifact, dict(train=fit, evaluation=score, shuffled=shuf)


def differences(a, b):
    def values(x, y):
        return dict(maximum=max(float((u-v).abs().max()) for u, v in zip(x, y)),
                    differing_elements=sum(int((u != v).sum()) for u, v in zip(x, y)))
    result = {}
    for name in ['training', 'evaluation', 'shuffled']:
        result[name] = {k: values([v[k] for v in a[name]], [v[k] for v in b[name]]) for k in ['losses', 'argmax']}
    for name in ['gradients', 'after']:
        need(a[name].keys() == b[name].keys(), 'SURFACE_DEVICE_STATE_KEYS')
        result[name] = values(list(a[name].values()), list(b[name].values()))
    return result
