"""Independent packed-byte recount and complete cold optimizer replay comparison."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/'lens')]
import torch
from audit import surface_device_authority as authority
from audit.mixed_surface_readback import check, rejected
from claims.process_conditions import write_committed
from model import surface_device_runtime as runtime
from model.native_population_training import fingerprint
from model.stream_surface import StreamSurfaceDecoder
from text.constellation_corpus import need


def run(a):
    began = time.perf_counter(); a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    first = authority.pinned(a.input, a.input_pin); second = authority.pinned(a.cold, a.cold_pin)
    config, specs, examples, metadata, initial, _, saved, code, fresh = authority.load(a.root)
    need(first['code'] == second['code'] == code and first['environment'] == second['environment'], 'SURFACE_READBACK_BINDING')
    need(first['same_cohort_exact'] and second['same_cohort_exact'] and not first['promoted_to_training']
         and not second['promoted_to_training'], 'SURFACE_READBACK_VERDICT')
    one = json.loads((a.input/'results.json').read_bytes()); two = json.loads((a.cold/'results.json').read_bytes())
    need(one['metadata'] == two['metadata'] == metadata and one['donors'] == two['donors'] == saved['donors'], 'SURFACE_READBACK_METADATA')
    need(one['execution_order'] == [v[0] for v in runtime.ARMS] and two['execution_order'] == [v[0] for v in reversed(runtime.ARMS)],
         'SURFACE_READBACK_ORDER')
    selected = [next(e for e in examples if e.occurrence == i) for i in saved['selected']]
    counts = {}; states = {}
    baseline = torch.load(a.input/'reference.pt', map_location='cpu', weights_only=True)
    for label, mode, rows in runtime.ARMS:
        pack = torch.load(a.input/f'packed-{label}.pt', map_location='cpu', weights_only=True)
        coldpack = torch.load(a.cold/f'packed-{label}.pt', map_location='cpu', weights_only=True)
        need(fingerprint(pack) == fingerprint(coldpack) == one['packing'][label]['state'] == two['packing'][label]['state'],
             'SURFACE_READBACK_PACK_EXACT')
        counts[label] = dict(selected=check(pack, selected, config, specs))
        manifest = json.loads((a.input/f'schedule-{label}.json').read_bytes())
        need(manifest == json.loads((a.cold/f'schedule-{label}.json').read_bytes()), 'SURFACE_READBACK_MANIFEST_EXACT')
        observed = []
        def stream():
            for b in runtime.batches(examples, config, specs, mode, rows):
                state = runtime.packed(b); observed.append(fingerprint(state)); yield state
        counts[label]['all_atoms'] = check(stream(), examples, config, specs)
        need(observed == manifest and len(observed) == one['packing'][label]['calls'], 'SURFACE_READBACK_FULL_SCHEDULE')
        value = torch.load(a.input/f'{label}.pt', map_location='cpu', weights_only=True)
        cold = torch.load(a.cold/f'{label}.pt', map_location='cpu', weights_only=True)
        states[label] = fingerprint(value)
        need(states[label] == fingerprint(cold) == one['results'][label]['state'] == two['results'][label]['state'],
             'SURFACE_READBACK_COLD_STATE')
        need(runtime.differences(value, baseline) == one['results'][label]['difference_from_reference']
             == two['results'][label]['difference_from_reference'], 'SURFACE_READBACK_DIFFERENCES')
        for field in ['training', 'evaluation', 'shuffled']:
            source = [e for e in examples if e.side == 'train'] if field == 'training' else examples
            need(len(value[field]) == len(source), 'SURFACE_READBACK_SCORE_COUNT')
            for e, v in zip(source, value[field]):
                need(v['losses'].shape == v['argmax'].shape == (len(e.raw)+1,) and bool(v['losses'].isfinite().all())
                     and bool(((v['argmax'] >= 0) & (v['argmax'] < 258)).all()), 'SURFACE_READBACK_SCORED_BYTES')
        need(value['gradients'].keys() == value['after'].keys() == initial.keys(), 'SURFACE_READBACK_PARAMETER_COVERAGE')
        need(all(bool(v.isfinite().all()) for field in ['gradients', 'after'] for v in value[field].values()), 'SURFACE_READBACK_FINITE')
        need(all(any(bool(v.abs().any()) for n, v in value['gradients'].items() if n.startswith(prefix)) for s in specs
            for prefix in ['projectors.'+s.name+'.', 'blocks.0.branches.'+s.name+'.']), 'SURFACE_READBACK_BRANCH_GRADIENTS')
        moments = value['optimizer']['state']; need(len(moments) == len(initial), 'SURFACE_READBACK_OPTIMIZER_COVERAGE')
        for v in moments.values():
            need(set(v) == {'step', 'exp_avg', 'exp_avg_sq'} and float(v['step']) == 1
                 and all(bool(t.isfinite().all()) for t in v.values()), 'SURFACE_READBACK_OPTIMIZER_STEP')
        delta = [float((x['losses']-y['losses']).mean()) for x, y in zip(value['shuffled'], value['evaluation'])]
        need(delta == one['results'][label]['conditioning_delta'] == two['results'][label]['conditioning_delta'], 'SURFACE_READBACK_DELTA')
        for report in [one, two]:
            memory = report['results'][label]['memory']
            need(memory['allocated_before'] == memory['allocated_after'] and memory['peak_reserved'] <= 30000*1024**2,
                 'SURFACE_READBACK_MEMORY')
    need(all(states[f'mixed-{n}'] == states[f'gathered-{n}'] for n in runtime.COHORTS), 'SURFACE_READBACK_GATHER_DIFFERENCE')
    controls = {}; good = torch.load(a.input/'packed-gathered-32.pt', map_location='cpu', weights_only=True)
    for mode in ['native', 'identity', 'target', 'denominator', 'tail']:
        bad = deepcopy(good); s = bad[0]; name = specs[0].name
        if mode == 'native': s['native'][name]['values'][0, 0, 0] += 1
        elif mode == 'identity': s['native'][name]['identity'] = '0'*64
        elif mode == 'target': s['labels'][0, 0] = (int(s['labels'][0, 0])+1)%256
        elif mode == 'denominator': s['denominators'][0] += 1
        else: bad.pop()
        controls[mode] = rejected(lambda: check(bad, selected, config, specs))
    controls['held_gradient'] = rejected(lambda: runtime.evaluate(StreamSurfaceDecoder(config, specs),
        [next(e for e in examples if e.side == 'query')], 'gathered', 32, backward=True))
    controls['wrong_pin'] = rejected(lambda: authority.pinned(a.input, '0'*64))
    fresh(); need(not torch.cuda.is_initialized(), 'SURFACE_READBACK_GPU_INITIALIZED')
    pin = write_committed(a.output/'report.json', dict(complete=True, input_pin=a.input_pin, cold_pin=a.cold_pin,
        code=code, environment=first['environment'], states=states, counts=counts, controls=controls,
        all_cold_arm_states_exact=True, same_cohort_exact=True, original_per_atom_exact=len(set(states.values())) == 1,
        seconds=time.perf_counter()-began, promoted_to_training=False, scientific_admission=False, capability_demonstrated=False,
        number_status='measured', acceptance='author-checked'))
    print(json.dumps(dict(complete=True, pin=pin)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=authority.ROOT)
    for n in ['input', 'cold', 'output']: p.add_argument('--'+n, type=Path, required=True)
    for n in ['input-pin', 'cold-pin']: p.add_argument('--'+n, required=True)
    a = p.parse_args()
    try: run(a)
    except BaseException as e:
        if a.output.exists(): write_committed(a.output/'failure.json', dict(error=repr(e)))
        raise
