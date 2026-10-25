"""Independently recount saved full-answer states and energy operator traces."""
import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit import captured_answer_readback as rd

NEGATIVES = {'policy', 'memory_type', 'empty', 'batch', 'missing_lane', 'one_sample', 'shape', 'dtype',
             'target_grad', 'sample_nan', 'target_inf', 'variance_nan', 'variance_zero', 'memory', 'combined', 'program_layout'}


def program(value):
    roster = value['roster']; b = value['batch']; s = value['count']; inputs = value['layouts']
    rd.require(len(roster) == len(set(roster)) == len(value['scales']) == 6
        and all(math.isfinite(v) and v > 0 for v in value['scales'])
        and 1 <= b <= 64 and 2 <= s <= 8 and type(value['empirical']) is bool
        and len(inputs) == 2*b*len(roster), 'ENERGY_READ_PROGRAM')
    elements = 0
    for i in range(0, len(inputs), 2):
        x, y = inputs[i:i+2]
        rd.require(len(y['shape']) == 2 and min(y['shape']) > 0 and x['shape'] == [s, *y['shape']]
            and x['dtype'] in ('torch.float32', 'torch.float64') and y['dtype'] in ('torch.float32', 'torch.float64')
            and x['device'] == y['device'] == 'cpu', 'ENERGY_READ_LAYOUT')
        elements += math.prod(y['shape'])
    return elements


def traces(bundle, expected):
    rows = {}
    for label in ('unbiased', 'empirical'):
        trace = bundle['traces'][label]; pure = trace['program']; original = trace['reference']
        rd.require(pure and original[-len(pure):] == pure and len(original) > len(pure)
            and all('_local_scalar_dense' not in r['op'] for r in pure)
            and trace['description']['empirical'] == (label == 'empirical'), 'ENERGY_READ_OPERATORS')
        description = json.loads(json.dumps(trace['description'])); elements = program(description)
        rd.require(expected[label]['operations'] == len(pure) and expected[label]['checked_operations'] == len(original)
                   and expected[label]['no_host_scalar_reads'], 'ENERGY_READ_TRACE_COUNT')
        rd.require(tuple(bundle['variances']) == tuple(description['roster'])
                   and len(bundle['samples']) == len(bundle['targets']) == description['batch'], 'ENERGY_READ_TRACE_INPUTS')
        for index, (sample, target) in enumerate(zip(bundle['samples'], bundle['targets'])):
            for lane, name in enumerate(description['roster']):
                for offset, t in enumerate((sample[name], target[name])):
                    layout = description['layouts'][2*(index*6+lane)+offset]
                    rd.require(list(t.shape) == layout['shape'] and str(t.dtype) == layout['dtype']
                        and str(t.device) == layout['device'] and bool(t.isfinite().all()), 'ENERGY_READ_REAL_INPUT_LAYOUT')
        rows[label] = dict(operations=len(pure), checked_operations=len(original), native_elements=elements)
    return rows


def run(a):
    began = time.perf_counter(); rd.torch.set_num_threads(1); a.output.mkdir(exist_ok=False)
    roots = (a.producer, a.cold)
    wrappers = [rd.read(p/'report.json', h) for p, h in zip(roots, (a.producer_pin, a.cold_pin))]
    first, cold = wrappers
    rd.require(first['complete'] and cold['complete'] and cold['cold_reproduced'] and cold['previous'] == a.producer_pin
        and first['code'] == cold['code'] and first['controls'] == cold['controls']
        and first['scope_controls'] == cold['scope_controls'], 'ENERGY_READ_BINDING')
    rd.require(set(first['scope_controls']) == {'nested', 'exception', 'ownership', 'closed'}, 'ENERGY_READ_SCOPE_CONTROLS')
    for name, pin in first['code'].items():
        rd.require(rd.sha(rd.BASE/name) == pin and all(rd.sha(p/'source'/name) == pin for p in roots), 'ENERGY_READ_SOURCE')
    reports = [rd.read(p/'numerical/report.json', w['numerical_report']) for p, w in zip(roots, wrappers)]
    rd.require(reports[0]['binding'] == reports[1]['binding']
               and reports[1]['previous_pin'] == first['numerical_report'], 'ENERGY_READ_NUMERICAL_BINDING')
    reference_root = a.root/rd.REFERENCE[0]; reference = rd.read(reference_root/'report.json', rd.REFERENCE[1])
    rd.manifest(reference_root, reference)
    totals = dict(tensors=0, bytes=0); updates = decisions = 0; counts = {}; trace_counts = {}; last_bundle = None
    for root, wrapper, numerical in zip(roots, wrappers, reports):
        rd.manifest(root/'numerical', numerical)
        rd.require(wrapper['captured_backward_in_both_arms'] and not wrapper['gpu_used'] and not wrapper['promoted']
            and not wrapper['cuda_graph_created'] and not wrapper['performance_claim']
            and numerical['all_updates_exact'] and numerical['restored_continuations_exact'], 'ENERGY_READ_SCOPE')
        observed = {(r['architecture'], r['document'], r['variant']): r for r in wrapper['observations']}
        rd.require(len(observed) == len(wrapper['observations']) == 30, 'ENERGY_READ_OBSERVATIONS')
        counts[root.name] = {}; trace_counts[root.name] = {}
        for arch in rd.ARCHS:
            control = wrapper['controls'][arch]
            rd.require(set(control['negative']) == NEGATIVES and control['negative']['combined'] == control['negative']['sample_nan'],
                       'ENERGY_READ_CONTROLS')
            path = root/(arch+'-controls.pt'); rd.require(rd.sha(path) == wrapper['control_files'][path.name], 'ENERGY_READ_CONTROL_PIN')
            bundle = rd.torch.load(path, map_location='cpu', weights_only=True)
            trace_counts[root.name][arch] = traces(bundle, control['positive']); last_bundle = bundle
            counter = dict(calls=0, completed=0, checkpoint_recomputations=0, native_elements=0, decisions=0)
            for index, doc in enumerate(rd.DOCS):
                step = index+6; expected = rd.load(reference_root, f'{arch}/update-{step}.pt')
                row = numerical['results'][arch]['updates'][index]
                other = reports[1 if numerical is reports[0] else 0]['results'][arch]['updates'][index]
                rd.require(row['order'] == other['order'][::-1], 'ENERGY_READ_ARM_ORDER')
                for key in ('document', 'step', 'numerical', 'chain', 'counts', 'restored_state'):
                    rd.require(row.get(key) == other.get(key), 'ENERGY_READ_COLD_RECORD')
                for mode, label in (('original', 'original'), ('captured', 'program')):
                    value = rd.load(root/'numerical', f'{arch}/{mode}/update-{step}.pt')
                    c = rd.exact(rd.numeric(value), rd.numeric(expected))
                    for key in totals: totals[key] += c[key]
                    obs = observed[arch, doc, label]; n = value['record']['decisions']
                    rd.require(obs['step'] == step-1 and obs['decisions'] == n and value['record']['document'] == doc
                        and sum(r['end'] for r in value['record']['records']) == 1, 'ENERGY_READ_DECISIONS')
                    if mode == 'original': rd.require(not obs['calls'], 'ENERGY_READ_ORIGINAL')
                    else:
                        proposals = 6 if arch == 'whole_proposals' else 1
                        completed = sum(r['completed'] for r in obs['calls'])
                        rd.require(len(obs['calls']) == 2*(n-1)*proposals and completed == (2*proposals-1)*(n-1),
                                   'ENERGY_READ_CALL_COVERAGE')
                        counter['calls'] += len(obs['calls']); counter['completed'] += completed
                        counter['checkpoint_recomputations'] += len(obs['calls'])-completed
                        counter['native_elements'] += sum(program(r['program']) for r in obs['calls'])
                        counter['decisions'] += n
                    rd.require(value['record']['storage']['remaining_saved_storage_bytes'] == 0
                        and value['record']['storage']['remaining_storages'] == 0
                        and all(float(s['step']) == step for s in value['optimizer']['state'].values()), 'ENERGY_READ_STORAGE_ADAM')
                    for branch in value['early'].values():
                        rd.require(branch['requires_grad'] and bool(branch['gradient'].abs().any())
                            and bool(branch['end_gradient'].abs().any()), 'ENERGY_READ_FULL_HISTORY')
                    if index in (1, 4):
                        cp = rd.load(root/'numerical', f'{arch}/{mode}/checkpoint-{index+1}.pt')
                        rd.exact(cp['model'], value['model']); rd.exact(cp['optimizer'], value['optimizer'])
                        rd.require(cp['cursor'] == index+1 and cp['step'] == step and cp['chain'] == row['chain']
                                   and cp['counts'] == row['counts'], 'ENERGY_READ_RESTART')
                    updates += 1; decisions += n
            counts[root.name][arch] = counter
    rd.exact(*counts.values()); rd.exact(*trace_counts.values())
    negative = rd.controls()
    for label in ('host_read', 'changed_operator', 'missing_layout'):
        bundle = deepcopy(last_bundle); trace = bundle['traces']['unbiased']
        if label == 'host_read': trace['program'][0]['op'] = 'aten._local_scalar_dense.default'
        elif label == 'changed_operator': trace['program'].pop()
        else: trace['description']['layouts'] = trace['description']['layouts'][:-1]
        try: traces(bundle, first['controls']['whole_proposals']['positive'])
        except ValueError as exc: negative[label] = str(exc)
        else: raise ValueError('ENERGY_READ_CONTROL_ACCEPTED')
    rd.require(updates == 60 and decisions == 792 and not rd.torch.cuda.is_initialized(), 'ENERGY_READ_TOTAL')
    result = dict(complete=True, producer=a.producer_pin, cold=a.cold_pin, reference=rd.REFERENCE[1],
        updates_compared=updates, decisions=decisions, comparison=totals, counters=counts, traces=trace_counts,
        all_numerical_bytes_exact=True, checkpoints_exact=True, controls=negative,
        reader_source=rd.sha(__file__), helper_source=rd.sha(rd.BASE/'audit/captured_answer_readback.py'),
        gpu_used=False, promoted=False, cuda_graph_created=False, performance_claim=False,
        seconds=time.perf_counter()-began, acceptance='author-checked')
    (a.output/'source.py').write_bytes(Path(__file__).read_bytes())
    with (a.output/'report.json').open('x') as f:
        json.dump(result, f, sort_keys=True, indent=2); f.flush(); os.fsync(f.fileno())
    print(json.dumps(dict(complete=True, pin=rd.sha(a.output/'report.json'), counters=counts, traces=trace_counts)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'producer', 'cold', 'output'): p.add_argument('--'+name, type=Path, required=True)
    for name in ('producer-pin', 'cold-pin'): p.add_argument('--'+name, required=True)
    a = p.parse_args()
    try: run(a)
    except BaseException as exc:
        if a.output.exists(): (a.output/'failure.json').write_text(json.dumps(dict(error=type(exc).__name__, detail=str(exc))))
        raise
