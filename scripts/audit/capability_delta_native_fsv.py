"""Cold native/source/typed-tensor joins and cross-population split checks for new experience."""
import argparse
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'lens'))
from audit.capability_revision_fsv import rejected
from audit.window_readback import readback
from claims.capability_population import need
from claims.episodic_memory import digest
from claims.process_conditions import write_committed
from kernel.capability_registry import freeze_together, load_sources
from model.capability_delta_input import DeltaNative, DeltaStore
from model.contracts import ModelConfig
from text.evaluation_multiscale import file_sha


def registry_check(registry, populations, history):
    for pin, pop in populations.items():
        split = registry['splits'][pin]
        groups = {t['split_group'] for t in pop.tasks.values()}
        need(set(split['groups']) == groups, 'DELTA_REGISTRY_GROUPS')
        for side, label in (('train', 'train'), ('queries', 'query')):
            expected = {k for k, t in pop.tasks.items() if registry['assignments'][t['split_group']] == label}
            need(set(split[side]) == expected, 'DELTA_REGISTRY_SHARED_MEMBERSHIP')
        need(set(split['excluded']) == {g for g in groups if registry['assignments'][g] == 'query'},
             'DELTA_REGISTRY_EXCLUDED')
    for pin, old in history.items():
        for side in ('train', 'queries', 'excluded'):
            need(old[side] == registry['splits'][pin][side], 'DELTA_REGISTRY_HISTORY')


def build(a):
    start = tick = time.perf_counter()
    a.output.mkdir(parents=True, exist_ok=False)
    window_receipt = readback(a.native, a.plan)
    write_committed(a.output / 'window-readback.json', window_receipt)
    phases = dict(all_native_tokens_and_offsets=time.perf_counter() - tick)
    tick = time.perf_counter()
    store = DeltaStore(a.plan, file_sha(a.plan / 'report.json'), a.native, file_sha(a.native / 'report.json'))
    native = {pin: DeltaNative(str(p.root), pin, store=store) for pin, p in store.populations.items()}
    phases['authenticated_shared_store'] = time.perf_counter() - tick
    tick = time.perf_counter()
    arrays, traces = {}, {}
    for origin, root in store.roots.items():
        for spec in store.specs:
            with np.load(root / spec.name / 'slots.npz', allow_pickle=False) as z:
                arrays[origin, spec.name] = z['vectors'].copy(), z['offsets'].copy()
            with gzip.open(root / spec.name / 'traces.jsonl.gz', 'rt', encoding='utf-8') as stream:
                traces[origin, spec.name] = [json.loads(line) for line in stream]
    fields, tensor_cases, bindings = {}, {}, {}
    config = ModelConfig()
    for pin, provider in native.items():
        p = provider.population
        fields[pin] = {}
        for key, f in p.fields.items():
            fields[pin][key] = {}
            for spec in provider.specs:
                actual, desc = provider.field(key, spec.name)
                pieces = []
                for part in desc['parts']:
                    bank, row = part['origin'], part['row']
                    values, offsets = arrays[bank, spec.name]
                    lo, hi = map(int, offsets[row:row+2])
                    trace = traces[bank, spec.name][row]
                    need(trace['vector_span'] == part['source_window_span'] == [lo, hi]
                         and trace['trace']['surface_sha256'] == part['source_sha256'], 'DELTA_TRACE_JOIN')
                    source = f['text'].encode()[slice(*part['source_byte_span'])]
                    need(hashlib.sha256(source).hexdigest() == part['source_sha256'], 'DELTA_FIELD_BYTES')
                    pieces.append(values[lo:hi])
                need(np.array_equal(actual, np.concatenate(pieces)), 'DELTA_NATIVE_FIELD_CONSERVATION')
                fields[pin][key][spec.name] = dict(windows=len(actual), dimensions=actual.shape[1],
                    sha256=hashlib.sha256(actual.astype('<f4').tobytes()).hexdigest(), description=desc)
        def width(key, side):
            return max(sum(fields[pin][f][s.name]['windows'] for f in p.tasks[key][side]) for s in provider.specs)
        selected = {next(iter(p.tasks)), max(p.tasks, key=lambda k: width(k, 'context')),
                    max(p.tasks, key=lambda k: width(k, 'targets'))}
        selected.update(v['task'] for v in p.report['coverage'].get('witnesses', {}).values())
        unknowns = [k for k, t in p.tasks.items() if any(p.fields[f]['text'] == "DON'T KNOW" for f in t['targets'])]
        if unknowns:
            selected.add(unknowns[0])
        tensor_cases[pin] = {}
        for case_index, key in enumerate(sorted(selected)):
            saved, sides = {}, {}
            for side in ('context', 'targets'):
                slots, description = provider.batch(key, side=side, config=config, device='cpu')
                for spec in provider.specs:
                    actual = slots[spec.name].values.numpy()[0]
                    expected = np.concatenate([provider.field(f, spec.name)[0] for f in p.tasks[key][side]])
                    need(np.array_equal(actual, expected), 'DELTA_TENSOR_CONSERVATION')
                    saved[side + '__' + spec.name] = actual
                sides[side] = description
            name = pin + '-tensor-' + str(case_index) + '.npz'
            np.savez(a.output / name, **saved)
            with np.load(a.output / name, allow_pickle=False) as z:
                need(set(z.files) == set(saved) and all(np.array_equal(z[n], v) for n, v in saved.items()),
                     'DELTA_TENSOR_PERSISTENCE')
            tensor_cases[pin][key] = dict(file=name, descriptions=sides)
        bindings[pin] = provider.binding
    for provider in native.values():
        provider.fresh()
    write_committed(a.output / 'field-receipts.json', fields)
    write_committed(a.output / 'tensor-cases.json', tensor_cases)
    write_committed(a.output / 'bindings.json', bindings)
    phases['independent_arrays_and_typed_tensors'] = time.perf_counter() - tick
    tick = time.perf_counter()
    populations, history = load_sources(a.old_config, a.delta_config, a.historical_splits)
    registry = json.loads(a.registry.read_bytes())
    replayed = freeze_together(populations, history)
    for key, value in replayed.items():
        if key != 'wall_seconds':
            need(registry[key] == value, 'DELTA_REGISTRY_COLD_REPLAY')
    registry_check(registry, populations, history)
    broken = deepcopy(registry)
    pin = next(k for k in store.populations if populations[k].family == 'reasoning')
    split = broken['splits'][pin]
    key = split['queries'].pop()
    split['train'].append(key)
    controls = dict(cross_family_split_leak=rejected(lambda: registry_check(broken, populations, history)))
    first = next(iter(native.values()))
    entry = next(iter(store.mapping['populations'].values()))['units'][0]
    original = entry['row']
    try:
        entry['row'] = original + 1
        controls['mutated_native_row_map'] = rejected(first.fresh)
    finally:
        entry['row'] = original
    phases['global_split_replay_and_negative_controls'] = time.perf_counter() - tick
    tick = time.perf_counter()
    files = {p.name: dict(bytes=p.stat().st_size, sha256=file_sha(p)) for p in a.output.iterdir() if p.is_file()}
    phases['final_hash_census'] = time.perf_counter() - tick
    wall = time.perf_counter() - start
    result = dict(schema='capability-delta-native-fsv-v1', complete=True, author_checked=True,
        plan=file_sha(a.plan / 'report.json'), native=file_sha(a.native / 'report.json'), registry=file_sha(a.registry),
        populations={pin: dict(fields=len(p.fields), tasks=len(p.tasks), binding=bindings[pin],
            tensor_cases=len(tensor_cases[pin]),
            field_windows={s.name: sum(v[s.name]['windows'] for v in fields[pin].values()) for s in store.specs})
            for pin, p in store.populations.items()},
        instruments={s.name: s.identity for s in store.specs}, shared_store_count=1,
        controls=controls, registry_counts=registry['counts'], files=files,
        phases_seconds=phases, wall_seconds=wall, phase_coverage=sum(phases.values()) / wall,
        all_native_windows_preserved=True, historical_split_membership_unchanged=True,
        live_kernel_connected=False, learned_capability=False, scientific_admission=False, serving=False,
        producer_sha256=file_sha(Path(__file__)))
    need(result['phase_coverage'] >= .9, 'DELTA_FSV_TIMING')
    pin = write_committed(a.output / 'report.json', result)
    print(json.dumps(dict(report_sha256=pin, populations=len(native),
        fields=sum(len(p.fields) for p in store.populations.values()),
        tensor_cases=sum(map(len, tensor_cases.values())), wall_seconds=wall, controls=controls)))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'native', 'old-config', 'delta-config', 'historical-splits', 'registry', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    build(p.parse_args())
