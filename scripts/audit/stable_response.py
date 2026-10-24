"""Compare a reproduced real trial under folded and separated residual computation."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit.fresh_learning_open import prepare, open_fresh
from audit.fsv_identity_loop import configure
from audit.proposal_recognition import authenticate, evidence, read
from claims.codata_source import check, sha
from claims.episodic_memory import digest
from claims.process_conditions import write_committed
from lens.lenscommon import run_cli, take_lease
from model.fresh_learning import FreshData
from model.loop_training_checkpoint import fingerprint
from model import field_checkpoint, rehearsal_retention, response_basis, response_update, stable_response


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--proof',type=Path,required=True)
    parser.add_argument('--proof-pin',required=True)
    a = parser.parse_args()
    started = time.perf_counter()
    config = read(a.config)
    seal = authenticate(a.proof,a.proof_pin)
    check(evidence(a.config) == seal['files']['session/config.json'] and
          evidence(a.config.parent/'learning.json') == seal['files']['session/learning.json'],
          'STABLE_AUDIT_SOURCE')
    resolved = prepare(config,a.output)
    write_committed(a.output/'config.json',config)
    gpu = subprocess.check_output(['nvidia-smi'],text=True)
    take_lease('28','stable-response-audit')
    (a.output/'gpu.txt').open('x').write(gpu)
    configure()
    session,base,sessions = open_fresh(config,a.output,sources_value=resolved)
    data = FreshData(session,base,config['spec'])
    before_snapshot = session.snapshot()
    model,optimizer,_ = field_checkpoint.start(data)
    before_fingerprint = fingerprint(model,optimizer)
    baseline = data.evaluate(model)
    replayed = time.perf_counter()
    basis = data.response_basis
    original_measure = basis.measure
    captured = []

    def measure(data, model):
        result = original_measure(data,model)
        if len(captured) < 2:
            captured.append({n:p.detach().clone() for n,p in model.named_parameters() if n in basis.names})
        return result

    basis.measure = measure
    original_step = response_update.step(data,model,optimizer)
    basis.measure = original_measure
    previous = read(a.config.parent/'learning.json')
    write_committed(a.output/'original-step.json',original_step)
    check(digest(original_step) == digest(previous['losses'][0]), 'STABLE_AUDIT_ORIGINAL_STEP')
    check(len(captured) == 2 and any(not torch.equal(captured[0][n],captured[1][n]) for n in basis.names),
          'STABLE_AUDIT_NO_TRIAL')
    flat,flat_optimizer,_ = field_checkpoint.start(data)
    flat_before = fingerprint(flat,flat_optimizer)
    check(flat_before == before_fingerprint,'STABLE_AUDIT_PARENT')
    with torch.no_grad():
        for n,p in flat.named_parameters():
            if n in basis.names:
                p.copy_(captured[1][n])
    flat_scores,flat_responses = response_basis.capture_scores(data,flat)
    flat_evaluation = data.evaluate(flat)
    separated,separated_optimizer,_ = field_checkpoint.start(data)
    static = stable_response.install(separated,basis)
    check(data.evaluate(separated) == baseline,'STABLE_AUDIT_INITIAL_BEHAVIOUR')
    with torch.no_grad():
        for n,p in separated.named_parameters():
            if n in basis.names:
                p.copy_(captured[1][n])
    scores,responses = response_basis.capture_scores(data,separated)
    evaluation = data.evaluate(separated)
    objective = float(data.loss(separated)[0].detach())
    check(stable_response.buffers(separated) == static,'STABLE_AUDIT_BUFFER_DRIFT')
    arrays,branches = {},[]
    for i,(name,row) in enumerate(responses.items()):
        parameter = name+'.weight'
        check(np.array_equal(row['h'],basis.arrays[f'h_{i}'])
              and np.array_equal(flat_responses[name]['h'],row['h']), 'STABLE_AUDIT_FEATURE_DRIFT')
        arrays.update({f'h_{i}':row['h'],f'parent_y_{i}':basis.arrays[f'y_{i}'],
            f'flat_y_{i}':flat_responses[name]['y'],f'separated_y_{i}':row['y'],
            f'parent_w_{i}':captured[0][parameter].cpu().numpy(),
            f'trial_w_{i}':captured[1][parameter].cpu().numpy(),f'p_{i}':basis.arrays[f'p_{i}']})
        branches.append(dict(name=name,rows=len(row['h']),
            flat_equal=bool(np.array_equal(flat_responses[name]['y'],basis.arrays[f'y_{i}'])),
            separated_equal=bool(np.array_equal(row['y'],basis.arrays[f'y_{i}']))))
    np.savez(a.output/'arrays.npz',**arrays)
    controls = []
    module = next(iter(response_basis.modules(separated).values()))
    with torch.no_grad():
        module.anchor.add_(.01)
    try:
        data.forward(separated,data.retained_keys[0])
    except ValueError as error:
        check('STABLE_RESPONSE_FROZEN_BUFFER' in str(error),'STABLE_AUDIT_CONTROL_CODE')
        controls.append('changed_parent_buffer')
    else:
        raise AssertionError('STABLE_AUDIT_CONTROL_ACCEPTED')
    check(session.snapshot() == before_snapshot,'STABLE_AUDIT_SESSION_CHANGED')
    for s in sessions:
        s.close()
    finished = time.perf_counter()
    result = dict(schema='stable-response-audit-v1',acceptance='author-checked',
        source=str(a.config.parent),source_learning=sha((a.config.parent/'learning.json').read_bytes()),
        proof=dict(root=str(a.proof),pin=a.proof_pin),
        binding=digest(data.binding),basis=config['spec']['basis'],parent=config['spec']['checkpoint'],
        original_step=original_step,branches=branches,flat_scores=flat_scores,separated_scores=scores,
        flat_evaluation=flat_evaluation,separated_evaluation=evaluation,
        flat_checks=rehearsal_retention.comparisons(data,flat_scores),
        separated_checks=rehearsal_retention.comparisons(data,scores),
        objective_before=original_step['update']['objective_before'],objective_separated=objective,
        initial_behaviour_equal=True,controls=controls,static_buffers=static,
        arrays=dict(bytes=(a.output/'arrays.npz').stat().st_size,sha256=sha((a.output/'arrays.npz').read_bytes())),
        code={n:sha((Path(__file__).resolve().parents[1]/n).read_bytes()) for n in
              ('audit/stable_response.py','audit/fresh_learning_open.py','model/stable_response.py')},
        phases=dict(authentication_replay=replayed-started,comparison_controls_close=finished-replayed),
        wall_seconds=finished-started,serving=False,scientific_admission=False)
    write_committed(a.output/'report.json',result)
    print(json.dumps(dict(pin=sha((a.output/'report.json').read_bytes()),
        flat_retained=sum(r['retained'] for r in result['flat_checks']),
        separated_retained=sum(r['retained'] for r in result['separated_checks']),
        branches=branches,objective_before=result['objective_before'],objective_separated=objective)),flush=True)


if __name__ == '__main__':
    run_cli(main)
