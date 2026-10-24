"""Source field -> native window -> actual typed input/target tensor readback."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'lens'))
from claims.capability_population import need
from claims.episodic_memory import digest
from claims.process_conditions import write_committed
from model.capability_input import CapabilityNative
from model.contracts import ModelConfig
from text.evaluation_multiscale import file_sha


def run(args):
    start = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=False)
    parent = CapabilityNative(args.population, (args.population / 'COMMITTED').read_text(),
        args.plan, file_sha(args.plan / 'report.json'), args.native, file_sha(args.native / 'report.json'))
    pop, config = parent.population, ModelConfig()
    fields, source_arrays = {}, {}
    for key in pop.fields:
        fields[key] = {}
        for spec in parent.specs:
            values, desc = parent.field(key, spec.name)
            # Open original saved arrays separately, rather than accepting the provider's cache.
            pieces = []
            for part in desc['parts']:
                root = args.native if part['origin'] == 'fresh' else Path(part['origin'])
                source_key = (str(root), spec.name)
                if source_key not in source_arrays:
                    with np.load(root / spec.name / 'slots.npz', allow_pickle=False) as z:
                        source_arrays[source_key] = (z['vectors'].copy(), z['offsets'].copy())
                raw, offsets = source_arrays[source_key]
                row = part['row']
                lo, hi = map(int, offsets[row:row + 2])
                need([lo, hi] == part['source_window_span'], 'NATIVE_FSV_WINDOW_ADDRESS')
                pieces.append(raw[lo:hi])
            expected = np.concatenate(pieces)
            need(np.array_equal(values, expected), 'NATIVE_FSV_VALUES')
            fields[key][spec.name] = dict(windows=len(values), dimensions=values.shape[1],
                sha256=hashlib.sha256(values.astype('<f4').tobytes()).hexdigest(), description=desc)
    write_committed(args.output / 'field-receipts.json', fields)

    def width(key, side):
        return max(sum(fields[f][s.name]['windows'] for f in pop.tasks[key][side]) for s in parent.specs)

    # All source fields above; tensor assembly at the first and largest real boundaries below.
    keys = list(dict.fromkeys([next(iter(pop.tasks)),
        max(pop.tasks, key=lambda k: width(k, 'context')),
        max(pop.tasks, key=lambda k: width(k, 'targets'))]))
    cases = {}
    for index, key in enumerate(keys):
        cases[key], arrays = {}, {}
        for side in ('context', 'targets'):
            slots, desc = parent.batch(key, side=side, config=config, device='cpu')
            sides = {}
            for spec in parent.specs:
                actual = slots[spec.name].values.numpy()[0]
                expected = np.concatenate([parent.field(f, spec.name)[0] for f in pop.tasks[key][side]])
                need(np.array_equal(actual, expected), 'NATIVE_FSV_TENSOR_CONSERVATION')
                arrays[side + '__' + spec.name] = actual
                sides[spec.name] = dict(shape=list(actual.shape), instrument=spec.identity,
                                       sha256=hashlib.sha256(actual.astype('<f4').tobytes()).hexdigest())
            cases[key][side] = dict(description=desc, slots=sides)
        np.savez(args.output / f'tensors-{index}.npz', **arrays)
        with np.load(args.output / f'tensors-{index}.npz', allow_pickle=False) as saved:
            need(set(saved.files) == set(arrays) and all(np.array_equal(saved[n], a) for n, a in arrays.items()),
                 'NATIVE_FSV_TENSOR_PERSISTENCE')
    parent.fresh()
    write_committed(args.output / 'tensor-cases.json', cases)
    write_committed(args.output / 'binding.json', parent.binding)
    lengths = [sum(len(pop.fields[k]['text'].encode()) for k in task['targets']) for task in pop.tasks.values()]
    report = dict(schema='capability-native-source-join-v1', complete=True, family=pop.family,
        population=pop.pin, native_binding=parent.pin, fields=len(fields), tasks=len(pop.tasks),
        native_field_windows={s.name: sum(r[s.name]['windows'] for r in fields.values()) for s in parent.specs},
        instruments={s.name: s.identity for s in parent.specs}, tensor_cases=keys,
        max_context_windows=max(width(k, 'context') for k in pop.tasks),
        max_target_windows=max(width(k, 'targets') for k in pop.tasks),
        target_surface_max_bytes=max(lengths), target_surfaces_over_512_bytes=sum(n > 512 for n in lengths),
        all_fields_source_bound=True, all_native_windows_preserved=True,
        world_prediction_forward_measured=False, decoder_forward_measured=False, kernel_connected=False,
        author_checked=True, scientific_admission=False, learned_capability=False, serving=False,
        files={p.name: dict(bytes=p.stat().st_size, sha256=file_sha(p)) for p in args.output.iterdir() if p.is_file()},
        wall_seconds=time.perf_counter() - start, code_sha256=file_sha(Path(__file__)))
    pin = write_committed(args.output / 'report.json', report)
    print(json.dumps({**{k: report[k] for k in ('family', 'fields', 'tasks', 'max_context_windows',
        'max_target_windows', 'target_surface_max_bytes', 'target_surfaces_over_512_bytes', 'wall_seconds')},
        'report_sha256': pin}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('population', 'plan', 'native', 'output'): p.add_argument('--' + name, type=Path, required=True)
    run(p.parse_args())
