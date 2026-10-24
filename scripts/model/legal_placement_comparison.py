"""Matched legal-document evaluation for separately retained native proposals.

This module imports production paths only: importing it cannot hide CUDA.
Conditional energy is explicitly separate from autonomous ragged predictions.
"""
from collections import defaultdict
import hashlib
import os
from pathlib import Path

import numpy as np
import torch

from model.constellation import ordered_distance
from model.native_device_transport import transfer
from model.native_placement_efficient import stream_for
from model.native_population_training import cpu, fingerprint
from model.native_stochastic_placement import energy_loss
from text.constellation_corpus import file_sha, need


def persist(path, value):
    value = cpu(value)
    with Path(path).open('xb') as f:
        torch.save(value, f); f.flush(); os.fsync(f.fileno())
    restored = torch.load(path, map_location='cpu', weights_only=True)
    need(fingerprint(restored) == fingerprint(value), 'PLACEMENT_COMPARISON_PERSISTED_STATE')
    return dict(pin=file_sha(path), state=fingerprint(restored))


def metadata(items):
    """One input-byte fingerprint per frame, reused for all donor decisions."""
    result = {}
    for item in items:
        need(item['document'] not in result, 'PLACEMENT_COMPARISON_DUPLICATE_DOCUMENT')
        prefix = bytes(32); shapes = []
        for frame in item['frames']:
            shape = repr(tuple((n, tuple(x.shape)) for n, x in frame.items())).encode()
            prefix = hashlib.sha256(prefix+shape).digest(); shapes.append(prefix.hex())
        result[item['document']] = dict(side=item['side'], component=item['component'],
            source=item['source_collection'], layouts=shapes,
            identities=[fingerprint(f) for f in item['frames']])
    return result


def donor_plan(meta):
    # Indexed prefix buckets, not a quadratic scan of every pair of histories.
    buckets = defaultdict(list)
    for document, row in meta.items():
        for offset, layout in enumerate(row['layouts'][:-1]):
            buckets[(row['side'], row['source'], offset, layout)].append(document)
    result = {}
    for document, row in meta.items():
        chosen = []
        for offset, layout in enumerate(row['layouts'][:-1]):
            candidates = [other for other in buckets[(row['side'], row['source'], offset, layout)]
                if meta[other]['component'] != row['component']
                and row['identities'][offset+1] not in meta[other]['identities'][:offset+1]]
            selected = min(candidates, key=lambda other: hashlib.sha256(
                f'legal-placement-donor:{document}:{offset}:{other}'.encode()).digest()) if candidates else -1
            chosen.append(selected)
        result[str(document)] = chosen
    return result


def validate_donors(meta, plan):
    need(set(plan) == {str(d) for d in meta}, 'PLACEMENT_COMPARISON_DONOR_DOCUMENTS')
    for d, row in meta.items():
        need(len(plan[str(d)]) == len(row['identities'])-1, 'PLACEMENT_COMPARISON_DONOR_ROWS')
        for offset, other in enumerate(plan[str(d)]):
            if other == -1: continue
            need(other in meta, 'PLACEMENT_COMPARISON_DONOR_ID')
            donor = meta[other]
            need(donor['side'] == row['side'] and donor['source'] == row['source']
                 and donor['component'] != row['component'], 'PLACEMENT_COMPARISON_DONOR_SPLIT')
            need(offset+1 < len(donor['layouts']) and donor['layouts'][offset] == row['layouts'][offset],
                 'PLACEMENT_COMPARISON_DONOR_LAYOUT')
            need(row['identities'][offset+1] not in donor['identities'][:offset+1],
                 'PLACEMENT_COMPARISON_DONOR_TARGET_LEAK')
    need(plan == donor_plan(meta), 'PLACEMENT_COMPARISON_DONOR_SELECTION_CHANGED')


@torch.no_grad()
def histories(model, items, tile):
    device = next(model.parameters()).device; result = {}
    for item in items:
        stream = stream_for(model); state = None; values = []
        for lo in range(0, len(item['frames']), tile):
            frames = transfer(item['frames'][lo:lo+tile], device)
            h, state = stream.step(frames, context='comparison:'+str(item['document']), offset=lo,
                                   state=state, detach_history=True)
            values.append(cpu(h))
        need(sum(len(x) for x in values) == len(item['frames']), 'PLACEMENT_COMPARISON_HISTORY_COVERAGE')
        result[item['document']] = torch.cat(values)
    return result


def categorical(logits, codes):
    probability = []; ce = []; brier = []
    for level, tensor in enumerate(logits):
        x = tensor.detach().cpu().double().numpy(); x = np.exp(x-x.max(1, keepdims=True)); x /= x.sum(1, keepdims=True)
        p = x[np.arange(len(x)), codes[:, level].cpu().numpy()]
        need(np.isfinite(x).all() and bool((p > 0).all()), 'PLACEMENT_COMPARISON_PROBABILITIES')
        probability.append(torch.from_numpy(x.copy()))
        ce.append(torch.from_numpy(-np.log(p))); brier.append(torch.from_numpy((x*x).sum(1)-2*p+1))
    return dict(probabilities=probability, ce=torch.stack(ce, 1), brier=torch.stack(brier, 1))


def distance(left, right):
    a = left.detach().cpu().double().numpy(); b = right.detach().cpu().double().numpy()
    an = np.linalg.norm(a, axis=1, keepdims=True); bn = np.linalg.norm(b, axis=1, keepdims=True)
    need(np.isfinite(a).all() and np.isfinite(b).all() and bool((an > 0).all()) and bool((bn > 0).all()),
         'PLACEMENT_COMPARISON_NATIVE_DISTANCE')
    return ordered_distance(np.clip((a/an) @ (b/bn).T, -1, 1))


def numpy_energy(samples, target, variances):
    """Independent complete pair-matrix recount of the distinct-pair energy."""
    td = []; pd = []; lanes = {}
    for name in variances:
        x = samples[name].detach().cpu().double().numpy(); y = target[name].detach().cpu().double().numpy()
        scale = y.size*float(variances[name]); s = len(x)
        a = np.sqrt(np.square(x-y).sum((1, 2))/scale)
        b = np.array([[np.sqrt(np.square(u-v).sum()/scale) for v in x] for u in x])
        lanes[name] = float(a.mean()-.5*b.sum()/(s*(s-1))); td.append(a); pd.append(b)
    joint = float(np.sqrt(np.mean(np.square(td), axis=0)).mean()
                  -.5*np.sqrt(np.mean(np.square(pd), axis=0)).sum()/(s*(s-1)))
    return dict(lanes=lanes, joint=joint, loss=float(np.mean(list(lanes.values())))+joint)


@torch.no_grad()
def evaluate(model, target, labels, items, plan, stats, output, *, tile=8):
    output.mkdir(); hidden = histories(model, items, tile); records = []; payloads = {}
    device = next(model.parameters()).device
    frozen_before = fingerprint((target.state_dict(), labels.model.state_dict(), labels.runtime.state_dict()))
    for item in items:
        d = item['document']; nrows = len(item['codes']); rows = []; donor = plan[str(d)]
        for lo in range(0, nrows, tile):
            stop = min(lo+tile, nrows); h = hidden[d][lo:stop].to(device)
            # Entire autonomous generation precedes access to the next-atom targets.
            noise = torch.randn(stop-lo, 4, 8, generator=torch.Generator(device=device).manual_seed(
                20261103+d*104729+lo), device=device)
            autonomous = {}
            for name, (head, context) in model.contexts(h).items():
                forecast = head.forecast(context)
                samples = tuple(model.placement[name](context[j:j+1], forecast['codes'][j:j+1],
                    {k:v[j:j+1] for k,v in forecast['counts'].items()}, noise[j:j+1, :1], target=target)[0]
                    for j in range(stop-lo))
                autonomous[name] = dict(forecast=forecast, samples=samples)
            codes = item['codes'][lo:stop].to(device)
            counts = {k:v[lo:stop].to(device) for k,v in item['counts'].items()}
            future = transfer(item['frames'][lo+1:stop+1], device)
            for name, (head, context) in model.contexts(h).items():
                placement = model.placement[name]; forced = head.loss(context, codes, counts)
                tile_payload = []
                discrete = categorical(forced['logits'], codes)
                saved_forecast = cpu(autonomous[name]['forecast'])
                for level, actual in enumerate(forced['region']):
                    need(abs(float(discrete['ce'][:,level].mean())-float(actual)) <= 5e-7,
                         'PLACEMENT_COMPARISON_CE_RECOUNT')
                for j in range(stop-lo):
                    i = lo+j; c = codes[j:j+1]; w = {k:v[j:j+1] for k,v in counts.items()}
                    generated = placement(context[j:j+1], c, w, noise[j:j+1], target=target)
                    score = energy_loss(generated, future[j:j+1], placement.variances())
                    recount = numpy_energy(generated[0], future[j], placement.variances())
                    need(abs(float(score['loss'])-recount['loss']) <= 1e-10, 'PLACEMENT_COMPARISON_ENERGY_RECOUNT')
                    base = target.reconstruct(target.values(c), [{k:int(v[0]) for k,v in w.items()}])[0]
                    baseline = numpy_energy({k:x[None].expand(4,*x.shape) for k,x in base.items()}, future[j], placement.variances())
                    wrong = None; wrong_metrics = None
                    if donor[i] != -1:
                        wh = hidden[donor[i]][i:i+1].to(device); wc = model.contexts(wh)[name][1]
                        wf = head.forecast(wc)
                        wrong = placement(wc, c, w, noise[j:j+1], target=target)
                        wrong_energy = numpy_energy(wrong[0], future[j], placement.variances())
                        wl = head.loss(wc, c, w); wd = categorical(wl['logits'], c)
                        wrong_metrics = dict(energy=wrong_energy, categorical=wd, forecast=cpu(wf))
                    auto = autonomous[name]; sample = {k:x[0].detach().cpu() for k,x in auto['samples'][j].items()}
                    measured, measured_counts = labels.encode((sample,), canonical=True)
                    forecast = saved_forecast; predicted = forecast['codes'][j]
                    lanes = {k:dict(model=distance(future[j][k], sample[k]),
                        copy_current=distance(future[j][k], item['frames'][i][k]),
                        train_mean=distance(future[j][k], stats['centers'][k].float()[None])) for k in model.roster}
                    record = dict(document=d, side=item['side'], source=item['source_collection'],
                        component=item['component'], offset=i, proposal=name, target=cpu(c[0]), predicted=predicted,
                        target_count={k:int(v[0]) for k,v in w.items()},
                        predicted_count={k:int(v[j]) for k,v in forecast['counts'].items()},
                        ce=discrete['ce'][j], brier=discrete['brier'][j], conditional_energy=recount,
                        baselines={k:v[i].clone() for k,v in item['baselines'].items()},
                        conditional_frozen_baseline=baseline, donor=donor[i], shuffled=wrong_metrics,
                        native_distance=lanes, measured_codes=torch.tensor(measured[0].astype(np.int64)),
                        native_pin=item['native_pin'], label_pin=item['label_pin'])
                    rows.append(record); records.append(record)
                    # Native outputs and all fixed-shape score inputs are independently readable.
                    tile_payload.append(cpu(dict(record=record, auto=sample, conditional=generated,
                        target=future[j], frozen_base=base, wrong=wrong, noise=noise[j],
                        probabilities=[p[j] for p in discrete['probabilities']], variances=placement.variances())))
                key=f'document-{d}-rows-{lo}-{stop}-{name}.pt'
                payloads[key]=persist(output/key, tile_payload)
        need(len(rows) == nrows*len(model.proposals), 'PLACEMENT_COMPARISON_ROW_COVERAGE')
    need(frozen_before == fingerprint((target.state_dict(), labels.model.state_dict(), labels.runtime.state_dict())),
         'PLACEMENT_COMPARISON_TARGET_CHANGED')
    return records, payloads


def summary(records, roster):
    groups = defaultdict(list)
    for row in records: groups[(row['side'], row['source'], row['proposal'])].append(row)
    result = {}
    for key, rows in groups.items():
        exact = torch.stack([r['predicted'] == r['target'] for r in rows])
        roundtrip = torch.stack([r['predicted'] == r['measured_codes'] for r in rows])
        paired = [r for r in rows if r['shuffled'] is not None]
        result['|'.join(key)] = dict(rows=len(rows), documents=len({r['document'] for r in rows}),
            code_accuracy=exact.double().mean(0).tolist(), prefix_accuracy=exact.cumprod(1).double().mean(0).tolist(),
            roundtrip_prefix=roundtrip.cumprod(1).double().mean(0).tolist(),
            ce=torch.stack([r['ce'] for r in rows]).mean(0).tolist(),
            brier=torch.stack([r['brier'] for r in rows]).mean(0).tolist(),
            conditional_energy=float(np.mean([r['conditional_energy']['loss'] for r in rows])),
            conditional_frozen_baseline=float(np.mean([r['conditional_frozen_baseline']['loss'] for r in rows])),
            paired_rows=len(paired), missing_donor_rows=len(rows)-len(paired),
            paired_energy_advantage=float(np.mean([r['shuffled']['energy']['loss']-r['conditional_energy']['loss'] for r in paired])) if paired else None,
            paired_ce_advantage=torch.stack([r['shuffled']['categorical']['ce'][0]-r['ce'] for r in paired]).mean(0).tolist() if paired else None,
            count_accuracy={n:sum(r['target_count'][n]==r['predicted_count'][n] for r in rows)/len(rows) for n in roster},
            baseline_accuracy={name:torch.stack([r['baselines'][name]==r['target'] for r in rows]).double().mean(0).tolist()
                for name in rows[0]['baselines']},
            baseline_available={name:torch.stack([r['baselines'][name]>=0 for r in rows]).sum(0).tolist()
                for name in rows[0]['baselines']},
            native_distance={n:{arm:float(np.mean([r['native_distance'][n][arm] for r in rows]))
                for arm in ('model','copy_current','train_mean')} for n in roster})
    return result
