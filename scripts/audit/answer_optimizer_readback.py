"""Reopen complete-answer updates, moments and profiles independently of the candidate."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import pstats
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit import captured_answer_readback as rd

PROFILE = ('answer-update-profile-v1-20261024',
           'b54ada053d616cd31fd6c77fd7778bb5c38caebbff452b1217cc19d9681dffb0')
CONTROLS = {'missing_state', 'weights_only', 'missing_moment', 'extra_moment', 'cursor', 'step_nan',
    'negative_step', 'boolean_step', 'recipe', 'order', 'shape', 'dtype', 'device', 'nan',
    'positive_inf', 'negative_inf', 'combined', 'nested', 'exception', 'ownership', 'closed',
    'source_unchanged', 'scope_restored'}


def recount(value, observed):
    states = value['optimizer']['state']; step = observed['step']
    rd.require(states and all(set(s) == {'step', 'exp_avg', 'exp_avg_sq'} for s in states.values()),
               'ANSWER_OPT_READ_MOMENTS')
    tensors = [t for s in states.values() for t in s.values()]
    rows = observed['validations']
    rd.require(len(rows) == 2 and [v['step'] for v in rows] == [step, step+1], 'ANSWER_OPT_READ_CURSORS')
    counts = dict(calls=0, tensors=0, elements=0, batches=0, direct_checks=0, states=0)
    for row in rows:
        c = row['checks']
        rd.require(row['states'] == len(states) and c['closed'] and c['calls'] == 1
                   and c['tensors'] == len(tensors) and c['elements'] == sum(t.numel() for t in tensors)
                   and c['direct_checks'] == 0 and 0 < c['batches'] < len(tensors), 'ANSWER_OPT_READ_COVERAGE')
        rd.require(c['join_bytes'] == 1 << 20 and 0 <= c['peak_joined_bytes'] <= c['peak_batch_bytes'] <= 1 << 20,
                   'ANSWER_OPT_READ_MEMORY')
        counts['states'] += row['states']
        for name in counts:
            if name != 'states': counts[name] += c[name]
    return counts


def profile_counts(root, profile, arch):
    stats = pstats.Stats(); selected = []
    for row in profile['observations']:
        if row['architecture'] != arch or row['variant'] != 'captured': continue
        path = root/PROFILE[0]/row['profile']
        rd.require(rd.sha(path) == row['pin'], 'ANSWER_OPT_READ_PROFILE_PIN')
        stats.add(str(path)); selected.append(row['document'])
    rd.require(selected == list(rd.DOCS), 'ANSWER_OPT_READ_PROFILE_CASES')
    helpers = [k for k in stats.stats if k[0].endswith('/loop_training_checkpoint.py') and k[2] == 'finite']
    validators = [v for k, v in stats.stats.items() if k[0].endswith('/native_placement_checked.py')
                  and k[2] == 'validate_optimizer']
    predicates = [v for k, v in stats.stats.items() if k[2] == '<built-in method torch.isfinite>']
    rd.require(len(helpers) == len(validators) == len(predicates) == 1, 'ANSWER_OPT_READ_PROFILE_HELPERS')
    calls = predicates[0][4][helpers[0]]
    rd.require(calls[0] == calls[1], 'ANSWER_OPT_READ_PROFILE_CALLS')
    return dict(calls=validators[0][1], tensors=calls[0])


def run(a):
    began = time.perf_counter(); rd.torch.set_num_threads(1); a.output.mkdir(exist_ok=False)
    roots = (a.producer, a.cold)
    wrappers = [rd.read(p/'report.json', h) for p, h in zip(roots, (a.producer_pin, a.cold_pin))]
    first, cold = wrappers
    rd.require(first['complete'] and cold['complete'] and cold['previous'] == a.producer_pin
        and cold['cold_reproduced'] and first['code'] == cold['code'] and first['controls'] == cold['controls'],
        'ANSWER_OPT_READ_BINDING')
    for name, pin in first['code'].items():
        rd.require(rd.sha(rd.BASE/name) == pin and all(rd.sha(p/'source'/name) == pin for p in roots),
                   'ANSWER_OPT_READ_SOURCE')
    reports = [rd.read(p/'numerical/report.json', v['numerical_report']) for p, v in zip(roots, wrappers)]
    rd.require(reports[1]['previous_pin'] == first['numerical_report']
               and reports[0]['binding'] == reports[1]['binding'], 'ANSWER_OPT_READ_NUMERICAL_BINDING')
    refroot = a.root/rd.REFERENCE[0]
    reference = rd.read(refroot/'report.json', rd.REFERENCE[1]); rd.manifest(refroot, reference)
    profile = rd.read(a.root/PROFILE[0]/'report.json', PROFILE[1])
    totals = dict(tensors=0, bytes=0); updates = decisions = 0; performance = {}; counters = {}
    last = None
    for root, wrapper, report in zip(roots, wrappers, reports):
        rd.manifest(root/'numerical', report)
        rd.require(wrapper['captured_backward_in_both_arms'] and not wrapper['gpu_used'] and not wrapper['promoted']
            and report['all_updates_exact'] and report['restored_continuations_exact']
            and report['optimizer_updates'] == 30 and report['decisions'] == 396, 'ANSWER_OPT_READ_SCOPE')
        obs = {(v['architecture'], v['document'], v['variant']): v for v in wrapper['observations']}
        rd.require(len(obs) == len(wrapper['observations']) == 30, 'ANSWER_OPT_READ_OBSERVATIONS')
        performance[root.name] = {}; counters[root.name] = {}
        for arch in rd.ARCHS:
            controls = wrapper['controls'][arch]
            rd.require(set(controls) == CONTROLS and controls['scope_restored'] and controls['source_unchanged']
                and controls['combined'] == controls['nan'] and controls['combined'] != controls['cursor'],
                'ANSWER_OPT_READ_CONTROLS')
            sums = dict(original=0., batched=0.); counts = dict(calls=0, tensors=0, elements=0, batches=0,
                                                             direct_checks=0, states=0)
            for i, doc in enumerate(rd.DOCS):
                step = i+6; expected = rd.load(refroot, f'{arch}/update-{step}.pt')
                row = report['results'][arch]['updates'][i]
                peer = reports[1 if report is reports[0] else 0]['results'][arch]['updates'][i]
                rd.require(row['order'] == peer['order'][::-1], 'ANSWER_OPT_READ_REVERSED_ORDER')
                for key in ('document', 'step', 'numerical', 'chain', 'counts', 'restored_state'):
                    rd.require(row.get(key) == peer.get(key), 'ANSWER_OPT_READ_COLD_RECORD')
                for mode, label in (('original', 'original'), ('captured', 'batched')):
                    value = rd.load(root/'numerical', f'{arch}/{mode}/update-{step}.pt')
                    compared = rd.exact(rd.numeric(value), rd.numeric(expected))
                    for key in totals: totals[key] += compared[key]
                    observed = obs[arch, doc, label]; sums[label] += observed['seconds']
                    rd.require(observed['step'] == step-1 and observed['decisions'] == value['record']['decisions']
                        and observed['seconds'] > 0 and value['record']['document'] == doc
                        and sum(v['end'] for v in value['record']['records']) == 1, 'ANSWER_OPT_READ_DECISIONS')
                    if mode == 'captured':
                        r = recount(value, observed)
                        for key in counts: counts[key] += r[key]
                        last = value, observed
                    else: rd.require(not observed['validations'], 'ANSWER_OPT_READ_BASELINE')
                    rd.require(value['record']['storage']['remaining_saved_storage_bytes'] == 0
                        and value['record']['storage']['remaining_storages'] == 0
                        and all(float(s['step']) == step for s in value['optimizer']['state'].values()),
                        'ANSWER_OPT_READ_STORAGE_CURSOR')
                    for credit in value['early'].values():
                        rd.require(credit['requires_grad'] and bool(credit['gradient'].abs().any())
                            and bool(credit['end_gradient'].abs().any()), 'ANSWER_OPT_READ_FULL_HISTORY')
                    if i in (1, 4):
                        cp = rd.load(root/'numerical', f'{arch}/{mode}/checkpoint-{i+1}.pt')
                        rd.exact(cp['model'], value['model']); rd.exact(cp['optimizer'], value['optimizer'])
                        rd.require(cp['cursor'] == i+1 and cp['step'] == step and cp['chain'] == row['chain']
                            and cp['counts'] == row['counts'], 'ANSWER_OPT_READ_RESTART')
                    updates += 1; decisions += observed['decisions']
            rd.exact(sums, wrapper['groups'][arch])
            profiled = profile_counts(a.root, profile, arch)
            rd.require(all(counts[k] == v for k, v in profiled.items()), 'ANSWER_OPT_READ_PROFILE_DENOMINATOR')
            counters[root.name][arch] = counts
            performance[root.name][arch] = dict(**sums, reduction_fraction=1-sums['batched']/sums['original'])
    rd.exact(*counters.values())
    negative = rd.controls(); value, observed = last
    for label in ('missing_moment', 'missing_check', 'changed_elements'):
        bad = deepcopy(value); record = deepcopy(observed)
        if label == 'missing_moment': del next(iter(bad['optimizer']['state'].values()))['exp_avg']
        elif label == 'missing_check': record['validations'].pop()
        else: record['validations'][0]['checks']['elements'] -= 1
        try: recount(bad, record)
        except ValueError as exc: negative[label] = str(exc)
        else: raise ValueError('ANSWER_OPT_READ_CONTROL_ACCEPTED')
    rd.require(updates == 60 and decisions == 792 and not rd.torch.cuda.is_initialized(), 'ANSWER_OPT_READ_TOTAL')
    result = dict(complete=True, producer=a.producer_pin, cold=a.cold_pin, reference=rd.REFERENCE[1],
        profile=PROFILE, updates_compared=updates, decisions=decisions, comparison=totals,
        performance=performance, counters=counters, all_numerical_bytes_exact=True, physical_checkpoints_exact=True,
        controls=negative, candidate_controls=first['controls'], gpu_used=False, promoted=False,
        seconds=time.perf_counter()-began, acceptance='author-checked', reader_source=rd.sha(__file__),
        source_code=first['code'])
    (a.output/'source.py').write_bytes(Path(__file__).read_bytes())
    with (a.output/'report.json').open('x') as f: json.dump(result, f, sort_keys=True, indent=2); f.flush(); os.fsync(f.fileno())
    print(json.dumps(dict(complete=True, pin=rd.sha(a.output/'report.json'), performance=performance,
                          counters=counters)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'producer', 'cold', 'output'): p.add_argument('--'+name, type=Path, required=True)
    for name in ('producer-pin', 'cold-pin'): p.add_argument('--'+name, required=True)
    a = p.parse_args()
    try: run(a)
    except BaseException as exc:
        if a.output.exists(): (a.output/'failure.json').write_text(json.dumps(dict(error=type(exc).__name__, detail=str(exc))))
        raise
