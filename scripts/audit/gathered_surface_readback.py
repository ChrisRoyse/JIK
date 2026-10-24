"""Independent source/packing recount and cold replay for the owned gather path."""
import argparse
from copy import deepcopy
from dataclasses import replace
import gc
import json
import os
from pathlib import Path
import sys
import time
import weakref

os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/'lens')]
import torch
from audit import gathered_surface_proof as producer
from audit import mixed_surface_readback as independent
from claims.process_conditions import write_committed
from model import gathered_surface_batches as gathered
from model import mixed_surface_batches as mixed
from model.native_population_training import fingerprint
from model.stream_surface import StreamSurfaceDecoder, schedule
from text.constellation_corpus import file_sha, need


def source_buffers(examples, config, specs):
    rows = elements = owned_bytes = 0
    for lo in range(0, len(examples), 128):
        part = examples[lo:lo+128]
        with gathered.SourceTile(part, config, specs) as tile:
            observed = tile.surface.tolist()
            start = 0
            for i, e in enumerate(part):
                expected = [256, *e.raw, 257]
                need(int(tile.starts[i]) == start and observed[start:start+len(expected)] == expected
                     and float(tile.denominators[i]) == len(e.raw)+1, 'GATHERED_READBACK_SURFACE_BUFFER')
                start += len(expected)
                for spec in specs:
                    n, local = tile.addresses[spec.name][i]
                    value = tile.banks[spec.name][n][local]
                    need(torch.equal(value, e.native[spec.name].values[0]), 'GATHERED_READBACK_NATIVE_BUFFER')
                    elements += value.numel()
                rows += 1
            need(start == len(observed), 'GATHERED_READBACK_BYTE_BUFFER_TAIL')
            owned_bytes += tile.owned_bytes
            pointers = [weakref.ref(t) for t, _ in tile.owned_stamps]
            del value
        gc.collect()
        need(tile.closed and not tile.banks and not tile.examples and tile.surface is None
             and all(p() is None for p in pointers), 'GATHERED_READBACK_BUFFER_RELEASE')
    return dict(atoms=rows, native_elements=elements, summed_tile_owned_bytes=owned_bytes, released=True)


def controls(selected, config, specs, good):
    results = {}
    for mode in ['source_value', 'source_mask', 'source_rebound', 'cache_value', 'cache_rebound', 'closed']:
        isolated = deepcopy(selected)
        with gathered.SourceTile(isolated, config, specs) as tile:
            if mode == 'source_value': isolated[0].native[specs[0].name].values[0, 0, 0] += 1
            elif mode == 'source_mask': isolated[0].native[specs[0].name].padding[0, 0] = True
            elif mode == 'source_rebound':
                x = isolated[0].native[specs[0].name]
                isolated[0].native[specs[0].name] = replace(x, identity='0'*64)
            elif mode == 'cache_value': tile.surface[0] += 1
            elif mode == 'cache_rebound': tile.surface = tile.surface.clone()
            else: tile.close()
            results[mode] = independent.rejected(tile.fresh)
    for mode in ['target', 'missing_tail', 'native_identity']:
        states = deepcopy(good)
        if mode == 'target': states[0]['labels'][0, 0] += 1
        elif mode == 'missing_tail': states.pop()
        else: states[0]['native'][specs[0].name]['identity'] = '0'*64
        results[mode] = independent.rejected(lambda: independent.check(states, selected, config, specs))
    for limit in [-1, 0, True, 513]:
        results['rows_'+str(limit)] = independent.rejected(
            lambda v=limit: list(gathered.batches(selected, config, specs, rows=v)))
    tiny = replace(config, max_attention_elements=1)
    results['memory'] = independent.rejected(lambda: list(gathered.batches(selected, tiny, specs, rows=32)))
    # Known-negative enlarged bytes, never accepted as a training/source observation.
    raw_limit = sum(x.values.numel() for x in selected[0].native.values())
    long = replace(selected[0], raw=selected[0].raw*(raw_limit//len(selected[0].raw)+2))
    results['byte_memory'] = independent.rejected(lambda: gathered.SourceTile(
        [long], replace(config, max_native_elements=raw_limit), specs))
    iterator = gathered.batches(selected, config, specs, rows=32)
    next(iterator); held = iterator.gi_frame.f_locals['tile']
    refs = [weakref.ref(t) for t, _ in held.owned_stamps]
    iterator.close(); gc.collect()
    need(held.closed and all(p() is None for p in refs), 'GATHERED_READBACK_EARLY_CLOSE_RELEASE')
    results['generator_early_close'] = dict(closed=True, buffers_released=True)
    isolated = deepcopy(selected)
    with gathered.SourceTile(isolated, config, specs) as tile:
        isolated[-1].native[specs[0].name].values[0, 0, 0] += 1
        first = [(0, next(schedule(isolated[0].raw, width=config.max_prefix)))]
        a = tile.assemble(first); b = mixed.assemble(first, isolated, config, specs)
        need(fingerprint(producer.previous.packed_state(a)) == fingerprint(producer.previous.packed_state(b)),
             'GATHERED_READBACK_UNCONSUMED_CHANGE')
        later = [(len(isolated)-1, next(schedule(isolated[-1].raw, width=config.max_prefix)))]
        results['changed_later_source'] = independent.rejected(lambda: tile.assemble(later))
        results['full_fresh_after_later_change'] = independent.rejected(tile.fresh)
    results['scoped_freshness'] = dict(unchanged_consumed_rows_exact=True, changed_later_rows_refused=True)
    return results


def run(a):
    began = time.perf_counter(); a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    need(file_sha(a.input/'report.json') == a.input_pin, 'GATHERED_READBACK_PIN')
    r = json.loads((a.input/'report.json').read_bytes())
    need(r['complete'] and r['code'] == producer.code(), 'GATHERED_READBACK_CODE')
    for n, h in r['files'].items(): need(file_sha(a.input/n) == h, 'GATHERED_READBACK_PHYSICAL')
    _, (_, config, specs, examples, metadata, initial) = producer.load(a.root)
    saved = json.loads((a.input/'results.json').read_bytes())
    train = [e for e in examples if e.side == 'train']
    selected = [next(e for e in train if e.occurrence == i) for i in saved['selected']]
    buffers = source_buffers(examples, config, specs); packing = {}; replay = {}
    for rows in producer.previous.COHORTS:
        states = torch.load(a.input/f'packed-{rows}.pt', map_location='cpu', weights_only=True)
        packing[str(rows)] = independent.check(states, selected, config, specs)
        observed = []
        def full():
            for b in producer.batches(examples, config, specs, rows):
                state = producer.previous.packed_state(b); observed.append(fingerprint(state)); yield state
        packing[str(rows)]['full'] = independent.check(full(), examples, config, specs)
        need(observed == json.loads((a.input/f'schedule-{rows}.json').read_bytes()), 'GATHERED_READBACK_FULL_REPLAY')
    for label, sequence, cfg in [('reversed', list(reversed(selected)), config),
                                  ('duplicate', [*selected, selected[0]], config),
                                  ('reduced_physical_batch', selected, replace(config, max_native_elements=40000))]:
        states = [producer.previous.packed_state(b) for b in gathered.batches(sequence, cfg, specs, rows=128)]
        old = [producer.previous.packed_state(b) for b in mixed.batches(sequence, cfg, specs, rows=128)]
        need(fingerprint(states) == fingerprint(old), 'GATHERED_READBACK_BOUNDARY_DIFFERENCE')
        packing[label] = independent.check(states, sequence, cfg, specs)
    good = torch.load(a.input/'packed-32.pt', map_location='cpu', weights_only=True)
    negative = controls(selected, config, specs, good)
    negative['held_gradient'] = independent.rejected(lambda: producer.evaluate(StreamSurfaceDecoder(config, specs),
        [next(e for e in examples if e.side == 'query')], 32, backward=True))
    shuffled = [replace(e, native=examples[j].native) for e, j in zip(examples, saved['donors'])]
    for i, j in enumerate(saved['donors']):
        need(examples[j].side == 'train' and examples[i].uid != examples[j].uid
             and metadata[i]['document'] != metadata[j]['document'], 'GATHERED_READBACK_DONOR')
    for rows in producer.previous.COHORTS:
        actual, _ = producer.execute(initial, config, specs, train, examples, shuffled, rows)
        prior = torch.load(a.input/f'rows-{rows}.pt', map_location='cpu', weights_only=True)
        baseline = torch.load(a.root/producer.SOURCE[0]/f'rows-{rows}.pt', map_location='cpu', weights_only=True)
        need(fingerprint(actual) == fingerprint(prior) == fingerprint(baseline)
             == saved['results'][str(rows)]['state'], 'GATHERED_READBACK_MODEL_STATE')
        replay[str(rows)] = dict(state=fingerprint(actual), all_tensors_exact=True)
        print(json.dumps(dict(rows=rows, cold_exact=True)), flush=True)
    need(producer.code() == r['code'] and not torch.cuda.is_initialized(), 'GATHERED_READBACK_ENVIRONMENT')
    result = dict(complete=True, input_pin=a.input_pin, source=producer.SOURCE, code=r['code'], buffers=buffers,
        packing=packing, controls=negative, replay=replay, seconds=time.perf_counter()-began,
        same_cohort_exact=True, original_per_atom_equivalence=False, promoted_to_training=False,
        scientific_admission=False, acceptance='author-checked', number_status='measured')
    pin = write_committed(a.output/'report.json', result); print(json.dumps(dict(complete=True, pin=pin)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('/zfs/archive/mnlm/evidence/fsv/29'))
    p.add_argument('--input', type=Path, required=True); p.add_argument('--input-pin', required=True)
    p.add_argument('--output', type=Path, required=True); a = p.parse_args()
    try: run(a)
    except BaseException as e:
        if a.output.exists(): write_committed(a.output/'failure.json', dict(error=repr(e)))
        raise
