"""Real-input VJPs and full-answer update proof for transient energy reconstruction."""
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
from audit import captured_answer_backward as proof
from claims.process_conditions import write_committed
from model import transient_energy_score as candidate
from model.native_population_training import cpu, fingerprint
from model.saved_gradient_storage import SavedStorage
from text.constellation_corpus import file_sha, need

NEW = ('model/energy_score_program.py', 'model/transient_energy_score.py', 'audit/transient_energy_probe.py')
REAL = ('energy-program-probe-v3-20261024', '83144eab2d112c5b017a74f3a11d7f9865039dc367fc9da3ef4b62a303b0b760')


def leaf_inputs(bundle, case):
    samples = tuple({n:x.detach().clone().requires_grad_(True) for n,x in row.items()} for row in bundle['samples'])
    targets = cpu(bundle['targets']); names = tuple(bundle['variances'])
    if case == 'noncontiguous':
        samples = tuple({n:x.transpose(0,2).contiguous().transpose(0,2).detach().requires_grad_(True)
                         for n,x in row.items()} for row in samples)
    elif case == 'duplicate_samples':
        samples = tuple({n:x[:1].expand_as(x).clone().detach().requires_grad_(True) for n,x in row.items()} for row in samples)
    elif case == 'zero_distance':
        samples = tuple({n:t[n][None].expand_as(x).clone().requires_grad_(True) for n,x in row.items()}
                        for row,t in zip(samples, targets))
    elif case == 'partial_inputs':
        for row in samples:
            for n,x in row.items(): x.requires_grad_(n == names[0])
    return samples, targets


def differentiated(fn, bundle, case):
    samples, targets = leaf_inputs(bundle, case); roster = tuple(bundle['variances'])
    inputs = tuple(x for row in samples for x in row.values() if x.requires_grad)
    value = fn(samples, targets, bundle['variances'], empirical=case == 'empirical')
    if case.startswith('lane:'): selected = (value['lanes'][case[5:]],)
    elif case == 'joint': selected = (value['joint'],)
    elif case == 'all_outputs': selected = candidate.outputs(value, roster)
    else: selected = (value['loss'],)
    gradients = torch.autograd.grad(selected, inputs, tuple(torch.ones_like(x) for x in selected), allow_unused=True)
    return cpu(dict(inputs=samples, targets=targets, value=value, gradients=gradients)), tuple(gradients)


def direct_controls(a):
    root = a.root/REAL[0]; prior = proof.source.read(root/'report.json', REAL[1]); results = {}
    for arch in proof.source.ARCHITECTURES:
        path = root/(arch+'-controls.pt'); need(file_sha(path) == prior['control_files'][path.name], 'TRANSIENT_ENERGY_REAL_PIN')
        bundle = torch.load(path, map_location='cpu', weights_only=True); rows = []
        cases = ('loss', 'joint', 'all_outputs', 'empirical', 'noncontiguous', 'duplicate_samples', 'zero_distance', 'partial_inputs',
                 *('lane:'+n for n in bundle['variances']))
        for case in cases:
            expected, _ = differentiated(candidate.ORIGINAL, bundle, case)
            storage = SavedStorage(512<<20, 'cpu')
            with storage.hooks(), candidate.checked_energy() as scope:
                actual, derivatives = differentiated(scope, bundle, case)
                need(fingerprint(actual) == fingerprint(expected), 'TRANSIENT_ENERGY_VJP:'+case)
            del derivatives
            need(storage.current == 0, 'TRANSIENT_ENERGY_DIRECT_RETAINED')
            rows.append(dict(case=case, expected=expected, actual=actual, calls=deepcopy(scope.records), storage=storage.report()))
        negative = controls(bundle)
        ref = proof.persist(a.output/(arch+'-vjps.pt'), dict(rows=rows, negative=negative, source=file_sha(path)))
        results[arch] = dict(cases=list(cases), pin=ref['pin'], negative=negative)
    return results


def controls(bundle):
    def error(fn):
        try: fn()
        except (ValueError, RuntimeError) as exc: return dict(type=type(exc).__name__, detail=str(exc))
        raise ValueError('TRANSIENT_ENERGY_BAD_INPUT_ACCEPTED')
    malformed = {}
    for label in ('empty', 'roster', 'dtype', 'shape', 'sample_nan', 'target_inf', 'variance', 'memory', 'target_grad'):
        samples, targets = leaf_inputs(bundle, 'loss'); variances = deepcopy(bundle['variances']); kwargs = {}
        name = next(iter(variances))
        if label == 'empty': samples = ()
        elif label == 'roster': del targets[0][name]
        elif label == 'dtype': samples[0][name] = samples[0][name].half()
        elif label == 'shape': samples[0][name] = samples[0][name][..., :-1]
        elif label == 'sample_nan':
            with torch.no_grad(): samples[0][name].view(-1)[-1] = float('nan')
        elif label == 'target_inf': targets[0][name].view(-1)[-1] = float('inf')
        elif label == 'variance': variances[name] = 0.
        elif label == 'memory': kwargs['working_bytes'] = 1
        elif label == 'target_grad': targets[0][name].requires_grad_(True)
        expected = error(lambda:candidate.ORIGINAL(samples,targets,variances,**kwargs))
        with candidate.checked_energy() as scope:
            actual = error(lambda:scope(samples,targets,variances,**kwargs))
        need(actual == expected, 'TRANSIENT_ENERGY_REFUSAL:'+label); malformed[label] = actual
    def higher_order():
        samples, targets = leaf_inputs(bundle, 'loss')
        with candidate.checked_energy() as scope:
            value = scope(samples, targets, bundle['variances'])
            torch.autograd.grad(value['loss'], tuple(samples[0].values()), create_graph=True)
    def mutation():
        samples, targets = leaf_inputs(bundle, 'loss'); storage = SavedStorage(512<<20, 'cpu')
        with storage.hooks(), candidate.checked_energy() as scope:
            value = scope(samples, targets, bundle['variances'])
            with torch.no_grad(): next(iter(samples[0].values())).add_(1)
            value['loss'].backward()
    def nested():
        with candidate.checked_energy():
            with candidate.checked_energy(): pass
    def changed():
        with candidate.checked_energy(): candidate.training.energy_loss = candidate.ORIGINAL
    def interrupted():
        with candidate.checked_energy(): raise ValueError('TRANSIENT_ENERGY_INTENTIONAL')
    result = {n:proof.reject(fn, code) for n,code,fn in (
        ('higher_order','ENERGY_TRANSIENT_HIGHER_ORDER_UNSUPPORTED',higher_order),
        ('mutation','GRADIENT_STORAGE_MUTATED',mutation),
        ('nested','ENERGY_TRANSIENT_ALREADY_INSTALLED',nested),
        ('ownership','ENERGY_TRANSIENT_OWNERSHIP_CHANGED',changed),
        ('exception','TRANSIENT_ENERGY_INTENTIONAL',interrupted))}
    result['malformed'] = malformed
    with candidate.checked_energy() as scope: pass
    result['closed'] = proof.reject(lambda:scope(None,None,None), 'ENERGY_TRANSIENT_CLOSED')
    need(candidate.training.energy_loss is candidate.ORIGINAL and not candidate.ACTIVE, 'TRANSIENT_ENERGY_SCOPE_RESTORED')
    # Keep several real forwards pending, then differentiate in reversed order.
    rows = []
    with candidate.checked_energy() as scope:
        pending = []
        for case in ('loss', 'duplicate_samples', 'zero_distance'):
            samples, targets = leaf_inputs(bundle, case); value = scope(samples, targets, bundle['variances'])
            pending.append((case,samples,value,cpu(value)))
        for case,samples,value,saved in reversed(pending):
            need(fingerprint(value) == fingerprint(saved), 'TRANSIENT_ENERGY_OVERWRITTEN_OUTPUT')
            gradients = torch.autograd.grad(value['loss'], tuple(x for r in samples for x in r.values()))
            expected,_ = differentiated(candidate.ORIGINAL,bundle,case)
            need(fingerprint(gradients) == fingerprint(expected['gradients']), 'TRANSIENT_ENERGY_PENDING_VJP')
            rows.append(dict(case=case, value=fingerprint(saved), gradient=fingerprint(gradients)))
    result['pending'] = rows
    return result


def run(a):
    a.output.mkdir(exist_ok=False); began = time.perf_counter(); original = proof.learner.update
    code = {n:file_sha(proof.source.BASE/n) for n in NEW}; observations = []
    for n in code:
        path = a.output/'source'/n; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes((proof.source.BASE/n).read_bytes())
    torch.set_num_threads(1)
    direct = direct_controls(a)
    def update(*args, **kwargs):
        model, _, item, step, _ = args; active = proof.candidate.ACTIVE
        with candidate.checked_energy() if active else proof.candidate.captured_backward() as scope:
            value = original(*args, **kwargs)
        observations.append(dict(architecture=model.forecaster.architecture, document=item['document'], step=step,
            variant='transient' if active else 'original', decisions=value[0]['decisions'], calls=deepcopy(scope.records) if active else []))
        return value
    proof.learner.update = update
    try:
        prior = proof.source.read(a.previous/'report.json', a.previous_pin) if a.previous else None
        if prior: need(prior['complete'] and prior['code'] == code, 'TRANSIENT_ENERGY_COLD_CODE')
        proof.run(argparse.Namespace(root=a.root, output=a.output/'numerical', cpu_proof=True,
            previous=a.previous/'numerical' if prior else None, previous_pin=prior['numerical_report'] if prior else None))
    finally:
        intact = proof.learner.update is update; proof.learner.update = original
        need(intact, 'TRANSIENT_ENERGY_UPDATE_OWNERSHIP')
    if prior: need(direct == prior['direct'], 'TRANSIENT_ENERGY_COLD_CONTROLS')
    numerical = json.loads((a.output/'numerical/report.json').read_bytes())
    need(numerical['complete'] and len(observations) == 30 and sum(r['decisions'] for r in observations) == 396, 'TRANSIENT_ENERGY_COVERAGE')
    need(all(file_sha(proof.source.BASE/n) == h for n,h in code.items()) and not torch.cuda.is_initialized(), 'TRANSIENT_ENERGY_FINAL')
    report = dict(complete=True, code=code, real_source=REAL, direct=direct, observations=observations,
        numerical_report=file_sha(a.output/'numerical/report.json'), previous=a.previous_pin, cold_reproduced=prior is not None,
        gpu_used=False, cuda_graph_created=False, promoted=False, performance_claim=False,
        captured_backward_in_both_arms=True, seconds=time.perf_counter()-began, acceptance='author-checked')
    print(json.dumps(dict(complete=True, pin=write_committed(a.output/'report.json', report))), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--root', type=Path, default=proof.source.ROOT)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--previous', type=Path); p.add_argument('--previous-pin')
    a = p.parse_args()
    try: run(a)
    except BaseException as exc:
        if a.output.exists(): write_committed(a.output/'failure.json', dict(error=type(exc).__name__, detail=str(exc)))
        raise
