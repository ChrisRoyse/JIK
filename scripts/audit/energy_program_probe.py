"""Complete-answer equivalence and real-input controls for the capture boundary."""
import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/'lens')]
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from audit import captured_answer_backward as proof
from claims.process_conditions import write_committed
from model import energy_score_program as candidate
from model.native_population_training import cpu, fingerprint
from text.constellation_corpus import file_sha, need

NEW = ('model/energy_score_program.py', 'audit/energy_program_probe.py')


def metadata(value):
    if isinstance(value, torch.Tensor): return dict(shape=list(value.shape), dtype=str(value.dtype), device=str(value.device))
    if isinstance(value, (list, tuple)): return [metadata(x) for x in value]
    if isinstance(value, dict): return {str(k): metadata(v) for k, v in value.items()}
    # isfinite itself uses infinity as an operator constant; preserve it as typed metadata.
    if isinstance(value, float) and not math.isfinite(value): return dict(floating_constant=value.hex())
    if value is None or isinstance(value, (bool, int, float, str)): return value
    return str(value)


class Trace(TorchDispatchMode):
    def __init__(self): super().__init__(); self.operations = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        self.operations.append(dict(op=str(func), args=metadata(args), kwargs=metadata(kwargs or {}), output=metadata(result)))
        return result


def controls(samples, targets, variances):
    samples = tuple({n: x.detach().clone().requires_grad_(x.requires_grad) for n, x in row.items()} for row in samples)
    targets = cpu(targets); variances = cpu(variances); roster = tuple(variances); first = roster[0]
    need(not torch.cuda.is_initialized(), 'ENERGY_PROOF_DEVICE')
    negative = {}; positive = {}

    def prepared(s, t, v, **kw):
        program, inputs = candidate.prepare(s, t, v, **kw); return program(*inputs)

    def error(fn):
        try: fn()
        except Exception as exc: return dict(type=type(exc).__name__, detail=str(exc))
        raise ValueError('ENERGY_PROOF_CONTROL_ACCEPTED')

    for label in ('policy', 'memory_type', 'empty', 'batch', 'missing_lane', 'one_sample', 'shape', 'dtype',
                  'target_grad', 'sample_nan', 'target_inf', 'variance_nan', 'variance_zero', 'memory', 'combined'):
        s = tuple({n: x.detach().clone() for n, x in row.items()} for row in samples)
        t = tuple({n: x.clone() for n, x in row.items()} for row in targets); v = deepcopy(variances); kw = {}
        if label == 'policy': kw['empirical'] = 1
        elif label == 'memory_type': kw['working_bytes'] = True
        elif label == 'empty': s = ()
        elif label == 'batch': t = ()
        elif label == 'missing_lane': del t[0][first]
        elif label == 'one_sample': s[0][first] = s[0][first][:1]
        elif label == 'shape': s[0][first] = s[0][first][..., :-1]
        elif label == 'dtype': s[0][first] = s[0][first].half()
        elif label == 'target_grad': t[0][first].requires_grad_(True)
        elif label in ('sample_nan', 'combined'): s[0][first].view(-1)[-1] = float('nan')
        elif label == 'target_inf': t[0][first].view(-1)[-1] = float('inf')
        elif label == 'variance_nan': v[first] = float('nan')
        elif label == 'variance_zero': v[first] = 0.
        elif label == 'memory':
            count = len(s[0][first]); elements = sum(x.numel() for row in t for x in row.values())
            kw['working_bytes'] = 4*8*elements*(count+count*(count-1)//2)-1
        if label == 'combined': kw['working_bytes'] = 1
        expected = error(lambda: candidate.ORIGINAL(s, t, v, **kw))
        actual = error(lambda: prepared(s, t, v, **kw))
        need(expected == actual, 'ENERGY_PROOF_REFUSAL_ORDER:'+label); negative[label] = actual
    traces = {}
    for empirical in (False, True):
        program, inputs = candidate.prepare(samples, targets, variances, empirical=empirical)
        with torch.no_grad(), Trace() as baseline:
            expected = candidate.ORIGINAL(samples, targets, variances, empirical=empirical)
        with torch.no_grad(), Trace() as pure: actual = program(*inputs)
        need(fingerprint(expected) == fingerprint(actual), 'ENERGY_PROOF_PRIMAL')
        need(pure.operations and baseline.operations[-len(pure.operations):] == pure.operations
             and all('_local_scalar_dense' not in x['op'] for x in pure.operations), 'ENERGY_PROOF_OPERATORS')
        key = 'empirical' if empirical else 'unbiased'
        positive[key] = dict(output=fingerprint(actual), operations=len(pure.operations),
                             checked_operations=len(baseline.operations), no_host_scalar_reads=True)
        traces[key] = dict(reference=baseline.operations, program=pure.operations, description=program.description())
    program, inputs = candidate.prepare(samples, targets, variances)
    negative['program_layout'] = error(lambda: program(*inputs[:-1]))
    return dict(negative=negative, positive=positive, scope_checked_separately=True), dict(
        samples=cpu(samples), targets=targets, variances=variances, traces=traces)


def scope_controls():
    def nested():
        with candidate.checked_energy():
            with candidate.checked_energy(): pass
    def interrupted():
        with candidate.checked_energy(): raise ValueError('ENERGY_PROOF_INTERRUPTED')
    def changed():
        with candidate.checked_energy(): candidate.training.energy_loss = candidate.ORIGINAL
    rows = {label: proof.reject(fn, code) for label, code, fn in (
        ('nested', 'ENERGY_PROGRAM_ALREADY_INSTALLED', nested),
        ('exception', 'ENERGY_PROOF_INTERRUPTED', interrupted),
        ('ownership', 'ENERGY_PROGRAM_OWNERSHIP_CHANGED', changed))}
    with candidate.checked_energy() as value: pass
    rows['closed'] = proof.reject(lambda: value(None, None, None), 'ENERGY_PROGRAM_CLOSED')
    need(candidate.training.energy_loss is candidate.ORIGINAL and not candidate.ACTIVE, 'ENERGY_PROOF_NOT_RESTORED')
    return rows


def run(a):
    a.output.mkdir(exist_ok=False); original = proof.learner.update; negative = {}; observations = []
    code = {n: file_sha(proof.source.BASE/n) for n in NEW}; began = time.perf_counter()
    for name in code:
        path = a.output/'source'/name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((proof.source.BASE/name).read_bytes())
    scoped = scope_controls()

    def observe(*args, **kwargs):
        model, _, item, step, _ = args; arch = model.forecaster.architecture
        active = proof.candidate.ACTIVE; tick = time.perf_counter()
        def first(program, inputs, samples, targets, variances, policy):
            if arch in negative: return
            # Diagnostic finite predicates on grad-enabled samples must not enter
            # the learner's checkpoint saved-tensor sequence.
            with torch.no_grad():
                result, state = controls(samples, targets, variances); negative[arch] = result
                proof.persist(a.output/(arch+'-controls.pt'), state)
        with candidate.checked_energy(first) if active else proof.candidate.captured_backward() as context:
            value = original(*args, **kwargs)
        records = context.records if active else []
        need(not active or context.closed, 'ENERGY_PROOF_OPEN_SCOPE')
        observations.append(dict(architecture=arch, document=item['document'], step=step,
            variant='program' if active else 'original', seconds=time.perf_counter()-tick,
            decisions=value[0]['decisions'], calls=records))
        return value

    proof.learner.update = observe
    try:
        prior = proof.source.read(a.previous/'report.json', a.previous_pin) if a.previous else None
        if prior: need(prior['complete'] and prior['code'] == code, 'ENERGY_PROOF_COLD_BINDING')
        proof.run(argparse.Namespace(root=a.root, output=a.output/'numerical', cpu_proof=True,
            previous=a.previous/'numerical' if prior else None,
            previous_pin=prior['numerical_report'] if prior else None))
    finally:
        intact = proof.learner.update is observe; proof.learner.update = original
        need(intact, 'ENERGY_PROOF_UPDATE_OWNERSHIP')
    report = json.loads((a.output/'numerical/report.json').read_bytes())
    need(report['complete'] and report['all_updates_exact'] and len(observations) == 30
         and sum(x['decisions'] for x in observations) == 396, 'ENERGY_PROOF_COVERAGE')
    if prior: need(negative == prior['controls'] and scoped == prior['scope_controls'], 'ENERGY_PROOF_COLD_CONTROLS')
    need(all(file_sha(proof.source.BASE/n) == h for n, h in code.items()), 'ENERGY_PROOF_CODE_CHANGED')
    result = dict(complete=True, code=code, controls=negative, scope_controls=scoped, observations=observations,
        control_files={arch+'-controls.pt': file_sha(a.output/(arch+'-controls.pt')) for arch in negative},
        numerical_report=file_sha(a.output/'numerical/report.json'), seconds=time.perf_counter()-began,
        captured_backward_in_both_arms=True, cold_reproduced=prior is not None, previous=a.previous_pin,
        gpu_used=False, promoted=False, cuda_graph_created=False, performance_claim=False, acceptance='author-checked')
    print(json.dumps(dict(complete=True, pin=write_committed(a.output/'report.json', result))), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=proof.source.ROOT); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--previous', type=Path); p.add_argument('--previous-pin'); a = p.parse_args()
    try: run(a)
    except BaseException as exc:
        if a.output.exists(): write_committed(a.output/'failure.json', dict(error=type(exc).__name__, detail=str(exc)))
        raise
