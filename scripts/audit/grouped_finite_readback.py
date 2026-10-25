"""Recount physical batched-validation updates against their prior CPU states and profile."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import pstats
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path[:0] = [str(Path(__file__).resolve().parents[1]),
               str(Path(__file__).resolve().parents[1]/'lens')]
import torch
from audit import captured_answer_readback as exact
from audit.grouped_profile_readback import checkpoint, reject
from claims.process_conditions import write_committed
from model.native_population_training import fingerprint
from text.constellation_corpus import file_sha, need

PROFILE = ('grouped-update-profile-v1-20261024',
           'a16602a69232095c54d91b2f3649e9622cb6e200a5eae13c429f0b3a54624b4b')
CONTROL_NAMES = {'finite', 'empty', 'empty_tensor', 'noncontiguous', 'mixed_dtype', 'small_chunks',
    'nan', 'positive_inf', 'negative_inf', 'nan_strided', 'positive_inf_strided', 'negative_inf_strided',
    'zero_allowance', 'large_allowance', 'boolean_allowance', 'nested', 'exception', 'ownership', 'closed'}


def observed(report, architecture, expected_calls, expected_tensors):
    rows = [x for x in report['observations'] if x['architecture'] == architecture]
    need([x['step'] for x in rows] == list(range(1, 6))
         and [x['document'] for x in rows] == list(exact.DOCS)
         and sum(x['targets'] for x in rows) == 66, 'FINITE_READ_CASES')
    counts = {k: sum(x['checks'][k] for x in rows)
              for k in ('calls', 'tensors', 'elements', 'batches', 'direct_checks')}
    need(counts['calls'] == expected_calls and counts['tensors'] == expected_tensors
         and counts['elements'] > 0 and counts['direct_checks'] == 0
         and 0 < counts['batches'] < counts['tensors'], 'FINITE_READ_CHECK_COVERAGE')
    need(all(x['checks']['closed'] and x['checks']['join_bytes'] == 1 << 20
             and 0 <= x['checks']['peak_joined_bytes'] <= x['checks']['peak_batch_bytes'] <= 1 << 20
             for x in rows), 'FINITE_READ_MEMORY')
    return counts


def run(a):
    torch.set_num_threads(1)
    a.output.mkdir(exist_ok=False)
    first = exact.read(a.producer/'report.json', a.producer_pin)
    cold = exact.read(a.cold/'report.json', a.cold_pin)
    need(first['complete'] and cold['complete'] and cold['previous_pin'] == a.producer_pin
         and not first['reversed'] and cold['reversed'] and first['code'] == cold['code']
         and first['controls'] == cold['controls'], 'FINITE_READ_BINDING')
    need(first['roles'] == dict(original='existing', efficient='batched')
         and cold['roles'] == dict(original='batched', efficient='existing'), 'FINITE_READ_REVERSED')
    need(set(first['controls']) == CONTROL_NAMES, 'FINITE_READ_CONTROL_SET')
    for label in ('noncontiguous', 'nan_strided', 'positive_inf_strided', 'negative_inf_strided'):
        need(first['controls'][label]['direct_checks'] == 1, 'FINITE_READ_STRIDED_CONTROL')
    profile = exact.read(a.root/PROFILE[0]/'report.json', PROFILE[1])
    reference = exact.read(a.root/PROFILE[0]/'numerical/report.json', profile['numerical_report'])
    exact.manifest(a.root/PROFILE[0]/'numerical', reference)
    authority = exact.read(a.root/first['source'][0]/'report.json', first['source'][1])
    need(file_sha(a.root/first['source'][0]/'bundle.pt') == authority['bundle_sha256'], 'FINITE_READ_BUNDLE')
    bundle = torch.load(a.root/first['source'][0]/'bundle.pt', map_location='cpu', weights_only=True)
    need(fingerprint(bundle) == authority['bundle_state'], 'FINITE_READ_BUNDLE_STATE')
    totals = dict(tensors=0, bytes=0, checkpoints=0, targets=0)
    performance = {}
    for root, report, label in ((a.producer, first, 'producer'), (a.cold, cold, 'cold')):
        need(report['paired_updates'] == 30 and report['paired_targets'] == 396
             and not report['gpu_used'] and not report['promoted'] and not report['capability_demonstrated'],
             'FINITE_READ_SCOPE')
        for name, pin in report['code'].items():
            need(file_sha(root/'source'/name) == pin and file_sha(exact.BASE/name) == pin, 'FINITE_READ_SOURCE')
        numerical = exact.read(root/'numerical/report.json', report['numerical_pin'])
        exact.manifest(root/'numerical', numerical)
        result = exact.read(root/'numerical/results.json', numerical['results_pin'])
        need(numerical['complete'] and numerical['all_updates_exact'] and numerical['cpu_reference_exact'],
             'FINITE_READ_NUMERICAL')
        need(report['timings'] == numerical['timings'], 'FINITE_READ_TIMINGS')
        performance[label] = {}
        for architecture in exact.ARCHS:
            stats = pstats.Stats()
            for row in profile['observations']:
                if row['architecture'] != architecture:
                    continue
                path = a.root/PROFILE[0]/row['profile']
                need(file_sha(path) == row['pin'], 'FINITE_READ_PROFILE_BYTES')
                stats.add(str(path))
            helpers = [key for key in stats.stats if key[0].endswith('/native_device_transport.py')
                       and key[2] == 'finite_tensors']
            need(len(helpers) == 1, 'FINITE_READ_PROFILE_HELPER')
            helper = helpers[0]
            predicates = [v for k, v in stats.stats.items() if k[2] == '<built-in method torch.isfinite>']
            need(len(predicates) == 1, 'FINITE_READ_PROFILE_PREDICATE')
            calls = predicates[0][4][helper]
            need(calls[0] == calls[1], 'FINITE_READ_PROFILE_CALLS')
            counts = observed(report, architecture, stats.stats[helper][1], calls[0])
            if label == 'cold':
                need(counts == observed(first, architecture, stats.stats[helper][1], calls[0]),
                     'FINITE_READ_COLD_CHECK_COUNTS')
            for i, row in enumerate(result[architecture]['steps']):
                relative = f'{architecture}/checkpoint-{i+1}.pt'
                value = exact.load(root/'numerical', relative)
                expected = exact.load(a.root/PROFILE[0]/'numerical', relative)
                n = exact.exact(value, expected)
                for k in n:
                    totals[k] += n[k]
                checkpoint(value, bundle['cpu_reference'][architecture]['updates'][i]['record'],
                           i+1, bundle['policy']['noise_seed'])
                need(row == value['record'] and len(result[architecture]['controls']) == 9, 'FINITE_READ_RECORD')
                totals['checkpoints'] += 1
                totals['targets'] += row['targets']
            times = {role: sum(x['seconds'] for x in report['timings']
                              if x['architecture'] == architecture and report['roles'][x['arm']] == role)
                     for role in ('existing', 'batched')}
            performance[label][architecture] = dict(times=times, counts=counts,
                measured_fraction_saved=1-times['batched']/times['existing'])
    need(totals['checkpoints'] == 30 and totals['targets'] == 396, 'FINITE_READ_DENOMINATOR')
    changed = deepcopy(value)
    next(iter(changed['gradients'].values())).view(-1)[0] += 1
    missing = deepcopy(value)
    missing['optimizer']['state'].pop(next(iter(missing['optimizer']['state'])))
    fewer = deepcopy(first)
    need(fewer['observations'][0]['architecture'] == 'shared', 'FINITE_READ_CONTROL_CASE')
    fewer['observations'][0]['checks']['tensors'] -= 1
    shared_counts = performance['producer']['shared']['counts']
    controls = dict(changed_gradient=reject(lambda: exact.exact(changed, expected)),
        missing_moment=reject(lambda: exact.exact(missing, expected)),
        changed_check_count=reject(lambda: observed(fewer, 'shared', shared_counts['calls'], shared_counts['tensors'])),
        strict_bytes=exact.controls())
    need(not torch.cuda.is_initialized(), 'FINITE_READ_GPU')
    result = dict(complete=True, producer=a.producer_pin, cold=a.cold_pin, reference=PROFILE,
        counts=totals, paired_updates=60, paired_targets=792, all_numerical_bytes_exact=True,
        performance=performance, controls=controls, predicate_controls=first['controls'],
        reader_source=file_sha(__file__), acceptance='author-checked', gpu_used=False, promoted=False,
        gpu_speedup_proven=False, long_answer_speedup_proven=False)
    print(json.dumps(dict(complete=True, report=write_committed(a.output/'report.json', result))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/zfs/archive/mnlm/evidence/fsv/29'))
    for name in ('producer', 'cold'):
        parser.add_argument('--'+name, type=Path, required=True)
        parser.add_argument('--'+name+'-pin', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args)
    except BaseException as exc:
        if args.output.exists():
            write_committed(args.output/'failure.json', dict(error=type(exc).__name__, detail=str(exc)))
        raise
