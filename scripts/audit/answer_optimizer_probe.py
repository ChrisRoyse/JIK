"""Matched complete-answer updates with original or batched optimizer validation."""
import argparse
from copy import deepcopy
import gc
import json
import os
from pathlib import Path
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/'lens')]
import torch
from audit import captured_answer_backward as proof
from claims.process_conditions import write_committed
from model import answer_optimizer_scope as candidate
from model.loop_training_checkpoint import fingerprint
from text.constellation_corpus import file_sha, need

NEW = ('model/answer_optimizer_scope.py', 'model/batched_finite_scope.py', 'audit/answer_optimizer_probe.py')


def controls(source, optimizer, policy, step):
    before = fingerprint(source, optimizer)
    result = {}

    def error(call):
        try:
            call()
        except Exception as exc:
            return dict(type=type(exc).__name__, detail=str(exc))
        raise ValueError('ANSWER_OPTIMIZER_CONTROL_ACCEPTED')

    def defect(label):
        model, opt = deepcopy((source, optimizer))
        states = list(opt.state.values()); first = states[0]; cursor = step
        if label == 'missing_state': del opt.state[next(iter(opt.state))]
        elif label == 'weights_only': opt.state.clear()
        elif label == 'missing_moment': del first['exp_avg_sq']
        elif label == 'extra_moment': first['extra'] = first['exp_avg'].clone()
        elif label == 'cursor': first['step'].add_(1)
        elif label == 'step_nan': first['step'].fill_(float('nan'))
        elif label == 'negative_step': cursor = -1
        elif label == 'boolean_step': cursor = True
        elif label == 'recipe': opt.param_groups[0]['eps'] = 1e-7
        elif label == 'order': opt.param_groups[0]['params'].reverse()
        elif label == 'shape': first['exp_avg'] = first['exp_avg'].reshape(-1)[:1]
        elif label == 'dtype': first['exp_avg'] = first['exp_avg'].double()
        elif label == 'device': first['exp_avg'] = first['exp_avg'].to('meta')
        elif label in ('nan', 'positive_inf', 'negative_inf', 'combined'):
            first['exp_avg'].reshape(-1)[-1] = {'nan': float('nan'), 'positive_inf': float('inf'),
                'negative_inf': -float('inf'), 'combined': float('nan')}[label]
            if label == 'combined': states[-1]['step'].add_(1)
        else: raise ValueError('ANSWER_OPTIMIZER_CONTROL_UNKNOWN')
        expected = error(lambda: candidate.ORIGINAL(model, opt, policy, cursor))
        with candidate.batched_optimizer() as checker:
            actual = error(lambda: proof.learner.validate_optimizer(model, opt, policy, cursor))
        need(expected == actual and not checker.records, 'ANSWER_OPTIMIZER_CONTROL_ERROR_ORDER:'+label)
        return actual

    for label in ('missing_state', 'weights_only', 'missing_moment', 'extra_moment', 'cursor',
                  'step_nan', 'negative_step', 'boolean_step', 'recipe', 'order', 'shape', 'dtype',
                  'device', 'nan', 'positive_inf', 'negative_inf', 'combined'):
        result[label] = defect(label)

    def nested():
        with candidate.batched_optimizer():
            with candidate.batched_optimizer(): pass
    def interrupted():
        with candidate.batched_optimizer(): raise ValueError('ANSWER_OPTIMIZER_INTERRUPTED')
    def replaced():
        with candidate.batched_optimizer(): proof.learner.validate_optimizer = candidate.ORIGINAL
    for label, code, call in (('nested', 'ANSWER_OPTIMIZER_ALREADY_INSTALLED', nested),
                              ('exception', 'ANSWER_OPTIMIZER_INTERRUPTED', interrupted),
                              ('ownership', 'ANSWER_OPTIMIZER_OWNERSHIP_CHANGED', replaced)):
        result[label] = proof.reject(call, code)
    with candidate.batched_optimizer() as checker:
        proof.learner.validate_optimizer(source, optimizer, policy, step)
    result['closed'] = proof.reject(lambda: checker(source, optimizer, policy, step), 'ANSWER_OPTIMIZER_CLOSED')
    need(before == fingerprint(source, optimizer) and checker.closed and len(checker.records) == 1
         and proof.learner.validate_optimizer is candidate.ORIGINAL and not candidate.ACTIVE,
         'ANSWER_OPTIMIZER_CONTROL_NOT_RESTORED')
    result.update(source_unchanged=True, scope_restored=True)
    gc.collect()
    return result


def run(a):
    a.output.mkdir(exist_ok=False); original = proof.learner.update; negative = {}; observations = []
    code = {n: file_sha(proof.source.BASE/n) for n in NEW}; began = time.perf_counter()
    for name in code:
        path = a.output/'source'/name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((proof.source.BASE/name).read_bytes())

    def observe(*args, **kwargs):
        model, opt, item, step, policy = args; arch = model.forecaster.architecture
        if arch not in negative: negative[arch] = controls(model, opt, policy, step)
        batched = proof.candidate.ACTIVE; tick = time.perf_counter(); records = []
        with candidate.batched_optimizer() if batched else proof.candidate.captured_backward() as context:
            value = original(*args, **kwargs)
        seconds = time.perf_counter()-tick
        if batched:
            need(context.closed, 'ANSWER_OPTIMIZER_PROBE_OPEN'); records = context.records
        observations.append(dict(architecture=arch, document=item['document'], step=step,
            variant='batched' if batched else 'original', seconds=seconds, decisions=value[0]['decisions'],
            validations=records))
        return value

    proof.learner.update = observe
    try:
        prior = proof.source.read(a.previous/'report.json', a.previous_pin) if a.previous else None
        if prior: need(prior['complete'] and prior['code'] == code, 'ANSWER_OPTIMIZER_PROBE_COLD_BINDING')
        proof.run(argparse.Namespace(root=a.root, output=a.output/'numerical', cpu_proof=True,
            previous=a.previous/'numerical' if prior else None,
            previous_pin=prior['numerical_report'] if prior else None))
    finally:
        intact = proof.learner.update is observe; proof.learner.update = original
        need(intact, 'ANSWER_OPTIMIZER_PROBE_UPDATE_OWNERSHIP')
    report = json.loads((a.output/'numerical/report.json').read_bytes())
    need(report['complete'] and report['all_updates_exact'] and len(observations) == 30
         and sum(x['decisions'] for x in observations) == 396, 'ANSWER_OPTIMIZER_PROBE_COVERAGE')
    groups = {arch: {v: sum(x['seconds'] for x in observations if x['architecture'] == arch and x['variant'] == v)
                    for v in ('original', 'batched')} for arch in proof.source.ARCHITECTURES}
    if prior: need(negative == prior['controls'], 'ANSWER_OPTIMIZER_PROBE_COLD_CONTROLS')
    need(all(file_sha(proof.source.BASE/n) == h for n, h in code.items()), 'ANSWER_OPTIMIZER_PROBE_CODE_CHANGED')
    result = dict(complete=True, code=code, controls=negative, observations=observations, groups=groups,
        numerical_report=file_sha(a.output/'numerical/report.json'), seconds=time.perf_counter()-began,
        captured_backward_in_both_arms=True, cold_reproduced=prior is not None, previous=a.previous_pin,
        gpu_used=False, promoted=False, acceptance='author-checked')
    print(json.dumps(dict(complete=True, pin=write_committed(a.output/'report.json', result), groups=groups)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=proof.source.ROOT); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--previous', type=Path); p.add_argument('--previous-pin'); a = p.parse_args()
    try: run(a)
    except BaseException as exc:
        if a.output.exists(): write_committed(a.output/'failure.json', dict(error=type(exc).__name__, detail=str(exc)))
        raise
