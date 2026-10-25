"""Read persisted grouped-update profiles and numerical states through a separate path."""
import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import pstats
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path[:0] = [str(Path(__file__).resolve().parents[1]),
               str(Path(__file__).resolve().parents[1]/'lens')]
import torch
from claims.process_conditions import write_committed
from model.native_population_training import fingerprint
from text.constellation_corpus import file_sha, need


def read(path, pin):
    need(file_sha(path) == pin, 'GROUPED_PROFILE_READ_PIN')
    return json.loads(path.read_bytes())


def checkpoint(value, expected, cursor, seed):
    need(value['record'] == expected and value['step'] == cursor
         and value['noise_cursor'] == seed+cursor, 'GROUPED_PROFILE_READ_RECORD')
    for key, record_key in (('model', 'after'), ('optimizer', 'optimizer'),
                            ('gradients', 'gradient'), ('noise', 'noise')):
        need(fingerprint(value[key]) == expected[record_key], 'GROUPED_PROFILE_READ_STATE')


def reject(fn):
    try:
        fn()
    except (ValueError, RuntimeError, KeyError) as exc:
        return str(exc)
    raise AssertionError('GROUPED_PROFILE_READ_NEGATIVE_ACCEPTED')


def run(a):
    a.output.mkdir(exist_ok=False)
    report = read(a.profile/'report.json', a.profile_pin)
    numerical = read(a.profile/'numerical/report.json', report['numerical_report'])
    results = read(a.profile/'numerical/results.json', numerical['results_pin'])
    need(report['complete'] and numerical['complete'] and numerical['all_updates_exact']
         and numerical['cpu_reference_exact'] and not numerical['gpu_qualified'],
         'GROUPED_PROFILE_READ_SCOPE')
    need(file_sha(a.profile/'source.py') == report['code'], 'GROUPED_PROFILE_READ_SOURCE')
    source = a.root/report['source'][0]
    authority = read(source/'report.json', report['source'][1])
    need(file_sha(source/'bundle.pt') == authority['bundle_sha256'], 'GROUPED_PROFILE_READ_BUNDLE')
    bundle = torch.load(source/'bundle.pt', map_location='cpu', weights_only=True)
    need(fingerprint(bundle) == authority['bundle_state'], 'GROUPED_PROFILE_READ_BUNDLE_STATE')
    for name, pin in numerical['files'].items():
        need(file_sha(a.profile/'numerical'/name) == pin, 'GROUPED_PROFILE_READ_FILE')
    seed = bundle['policy']['noise_seed']
    total = 0
    stats_by_arch = {}
    for architecture, result in results.items():
        need(len(result['steps']) == 5 and len(result['checkpoint_states']) == 5
             and len(result['controls']) == 9, 'GROUPED_PROFILE_READ_DENOMINATOR')
        for i, record in enumerate(result['steps']):
            value = torch.load(a.profile/'numerical'/architecture/f'checkpoint-{i+1}.pt',
                               map_location='cpu', weights_only=True)
            expected = bundle['cpu_reference'][architecture]['updates'][i]['record']
            need(record == expected and fingerprint(value) == result['checkpoint_states'][i],
                 'GROUPED_PROFILE_READ_REFERENCE')
            checkpoint(value, expected, i+1, seed)
            total += record['targets']
        selected = [x for x in report['observations'] if x['architecture'] == architecture]
        need([x['step'] for x in selected] == list(range(1, 6)), 'GROUPED_PROFILE_READ_STEPS')
        for observation, record in zip(selected, result['steps']):
            need(observation['document'] == record['document']
                 and observation['targets'] == record['targets'], 'GROUPED_PROFILE_READ_CASE')
            path = a.profile/observation['profile']
            need(file_sha(path) == observation['pin'], 'GROUPED_PROFILE_READ_STATS_PIN')
            stats = pstats.Stats(str(path))
            need(stats.total_calls == observation['total_calls']
                 and stats.prim_calls == observation['primitive_calls']
                 and stats.total_tt == observation['profiler_seconds']
                 and .9 <= stats.total_tt/observation['seconds'] <= 1.000001,
                 'GROUPED_PROFILE_READ_TIMING')
        stats = pstats.Stats(*(str(a.profile/x['profile']) for x in selected))
        group = report['groups'][architecture]
        actual = {(r['file'], r['line'], r['function']):
                  (r['primitive_calls'], r['calls'], r['self_seconds'], r['cumulative_seconds'])
                  for r in group['functions']}
        need(len(actual) == len(group['functions']) == len(stats.stats)
             and actual == {k: v[:4] for k, v in stats.stats.items()}
             and math.isclose(sum(v[2] for v in actual.values()), stats.total_tt,
                              rel_tol=1e-12, abs_tol=1e-12), 'GROUPED_PROFILE_READ_EXCLUSIVE_TIME')
        stats_by_arch[architecture] = dict(seconds=stats.total_tt, calls=stats.total_calls,
            profiled_updates=5, targets=sum(x['targets'] for x in selected))
    need(total == 198 and len(report['observations']) == 15, 'GROUPED_PROFILE_READ_TOTAL')
    changed = deepcopy(value)
    next(iter(changed['gradients'].values())).view(-1)[0] += 1
    missing = deepcopy(value)
    missing['optimizer']['state'].pop(next(iter(missing['optimizer']['state'])))
    controls = dict(changed_gradient=reject(lambda: checkpoint(changed, expected, 5, seed)),
        missing_moment=reject(lambda: checkpoint(missing, expected, 5, seed)),
        wrong_cursor=reject(lambda: checkpoint(value, expected, 6, seed)),
        wrong_pin=reject(lambda: read(a.profile/'report.json', '0'*64)))
    need(not torch.cuda.is_initialized(), 'GROUPED_PROFILE_READ_GPU')
    result = dict(complete=True, profile=a.profile_pin, source=report['source'],
        numerical=report['numerical_report'], profiled_updates=15, profiled_targets=198,
        paired_updates=30, paired_targets=396, checkpoint_states=15,
        groups=stats_by_arch, controls=controls,
        reader_source=file_sha(__file__), acceptance='author-checked', gpu_used=False,
        speedup_claim=False, promoted=False)
    # Keep existing control outcomes without duplicating update records or tensor states.
    result['original_controls'] = {k: v['controls'] for k, v in results.items()}
    print(json.dumps(dict(complete=True, report=write_committed(a.output/'report.json', result))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/zfs/archive/mnlm/evidence/fsv/29'))
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--profile-pin', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args)
    except BaseException as exc:
        if args.output.exists():
            write_committed(args.output/'failure.json', dict(error=type(exc).__name__, detail=str(exc)))
        raise
