"""Trace a persisted proposal failure without fitting or changing its source state."""
import argparse
from functools import partial
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.corrective_open import open_session
from audit.fsv_identity_loop import configure
from claims.codata_source import check, sha
from claims.episodic_memory import digest
from claims.process_conditions import write_committed
from lens.lenscommon import run_cli, take_lease
from model.constellation_learning import ordered_loss, shared_region_loss, trainable
from model.constellation_world import ConstellationWorld
from model.corrective_learning import CorrectiveData
from model.corrective_projection import project
from model.loop_training_checkpoint import fingerprint, tensor_fingerprint
from model.native_world import NativeWorld
from model.proposal_learning import ProposalData
from model.proposal_learning_session import ProposalLearningLoop
from model import field_checkpoint, process_anchor, rehearsal_retention


def read(path):
    return json.loads(path.read_bytes())


def evidence(path):
    raw = path.read_bytes()
    return dict(bytes=len(raw), sha256=sha(raw))


def authenticate(root, pin):
    raw = (root/'seal.json').read_bytes()
    check(sha(raw) == pin, 'RECOGNITION_SOURCE_SEAL')
    seal = json.loads(raw)
    paths = {str(root/n): v for n, v in seal['files'].items()}
    paths.update(seal['external'])
    for name, item in paths.items():
        check(evidence(Path(name)) == item, 'RECOGNITION_SOURCE_BYTES:'+name)
    return seal


def gradient(term, params):
    values = torch.autograd.grad(term, tuple(params.values()), allow_unused=True)
    return torch.cat([(g if g is not None else torch.zeros_like(p)).detach().reshape(-1).double()
                      for g, p in zip(values, params.values())])


def example_loss(data, model, key):
    output = data.forward(model, key)
    e = data.examples[key]
    label = torch.tensor([e['region']], device=model.core.initial.device)
    shared = shared_region_loss(output, label, e['allowed'], model.specs)
    risks = data.recognition_risk(model, key, output)
    placed = data.prediction_model(model, key).supervised_placement(
        output, label, allowed=e['allowed'], event_ids=[output['states'][0].last_event])
    placements = []
    for spec in model.specs:
        slot, target = placed.slots[spec.name], e['target'][spec.name]
        placements.append(ordered_loss(slot.values[0, ~slot.padding[0]],
                                       target.values[0, ~target.padding[0]]))
    risk = torch.stack([risks[s.name] for s in model.specs]).mean()
    full = shared+torch.stack([p+risks[s.name] for p, s in zip(placements, model.specs)]).mean()
    return full, dict(shared=float(shared.detach()), placement=float(torch.stack(placements).mean().detach()),
                      risk=float(risk.detach()), total=float(full.detach()))


def trace(data, model, prior, arrays, stage):
    result = {}
    with torch.no_grad():
        for key in data.plan['train']:
            e = data.examples[key]
            out = data.forward(model, key)
            view = data.prediction_model(model, key)
            p = process_anchor.predict(out, view.execution_anchor_bank, e['allowed'])
            check(p == prior[key]['termination']['prediction'], 'RECOGNITION_PRIOR_PREDICTION')
            check(out['predicted_constellation'].pack(model.specs, model.config) == prior[key]['predictions'],
                  'RECOGNITION_PRIOR_NATIVE')
            visible = e['allowed'].nonzero().flatten()
            check(bool(e['allowed'].all()), 'RECOGNITION_FULL_TRAIN_BANK')
            rows = next(iter(view.reference_rows.values()))
            selected = rows.index(prior[key]['region'])
            prefix = stage+'/'+key+'/'
            arrays[prefix+'common'] = out['constellation_regions'][0].cpu().numpy()
            counts, lanes = {}, {}
            independent = None
            if key in data.new:
                args = dict(allowed=e['allowed'], phase=2, event_ids=[digest(e['request'])],
                            evidence_origin=['generated_proposal'], history=e['history'])
                state = (view.bind_state(e['state']),)
                base = ConstellationWorld.forward(view, e['native'], state, **args)
                native = NativeWorld.forward(view, e['native'], state, **args)
                independent = base, native
            for i, spec in enumerate(model.specs):
                name = spec.name
                full = out['regions'][name][0, visible].double()
                nr = out['native_residuals'][name][0].double()
                fr = out['field_residuals'][name][0].double()
                parts = dict(full=full, native=nr, fields=fr, base=full-nr-fr)
                if independent is not None:
                    b, n = independent
                    check(torch.equal(n['regions'][name], b['regions'][name]+out['native_residuals'][name])
                          and torch.equal(out['regions'][name], n['regions'][name]+out['field_residuals'][name]),
                          'RECOGNITION_COMPONENT_PATH')
                    parts['independent_base'] = b['regions'][name][0].double()
                for part, values in parts.items():
                    arrays[prefix+name+'/'+part] = values.cpu().numpy()
                padding = getattr(view, f'world_padding_{i}')
                counts[name] = dict(input=int((~e['native'][name].padding).sum()),
                                   reference=int((~padding).sum()), dims=spec.dims)
                lanes[name] = {part: float(v[e['region']]-v[selected]) for part, v in parts.items()}
            result[key] = dict(target=e['region'], selected=selected, rows=list(rows),
                outcome=data.outcomes[key], new=key in data.new, lanes=lanes, counts=counts,
                probabilities=p, input=digest(tensor_fingerprint({n:s.values for n,s in e['native'].items()})),
                prestate=digest(tensor_fingerprint(e['state'].hidden)))
    return result


def gradients(data, model, displacement, arrays):
    params = dict(trainable(model))
    examples, vectors = {}, []
    for key in data.plan['train']:
        term, details = example_loss(data, model, key)
        vectors.append(gradient(term, params))
        examples[key] = details
    stacked = torch.stack(vectors)
    loss, _ = data.loss(model)
    objective = gradient(loss, params)
    relative_error = float(torch.linalg.vector_norm(stacked.mean(0)-objective)
                           / torch.linalg.vector_norm(objective))
    check(relative_error < 64*torch.finfo(torch.float32).eps, 'RECOGNITION_OBJECTIVE_GRADIENT')
    check(abs(sum(v['total'] for v in examples.values())/len(examples)-float(loss.detach())) < 1e-7,
          'RECOGNITION_OBJECTIVE_VALUE')
    retained = rehearsal_retention.prepare(data, model)
    g = torch.cat((retained['gradients'], objective[None, :]))
    norms = torch.linalg.vector_norm(g, dim=1)
    direction, projection = project(g, -F.normalize(objective, dim=0))
    check(direction is not None, 'RECOGNITION_COMMON_DESCENT_UNAVAILABLE')
    sample_norms = torch.linalg.vector_norm(stacked, dim=1)
    arrays['gradient/retention_example_cosines'] = (
        F.normalize(retained['gradients'], dim=1)@F.normalize(stacked, dim=1).T).cpu().numpy()
    arrays['gradient/example_gram'] = (F.normalize(stacked, dim=1)@F.normalize(stacked, dim=1).T).cpu().numpy()
    for index, key in enumerate(data.plan['train']):
        examples[key].update(norm=float(sample_norms[index]),
            objective_descent_cosine=float(-vectors[index]@objective/(sample_norms[index]*norms[-1])),
            common_descent_derivative=float(vectors[index]@direction),
            common_descent_cosine=float(vectors[index]@direction/sample_norms[index]),
            actual_update_linear=float(vectors[index]@displacement))
    comparisons = rehearsal_retention.comparisons(data, retained['scores'])
    check(all(r['retained'] for r in comparisons), 'RECOGNITION_RETENTION')
    for row in comparisons:
        row['slack'] = row['ceiling']-row['value']
    return dict(examples=examples, objective=float(loss.detach()), gradient_reconstruction=relative_error,
        objective_norm=float(norms[-1]), common_descent_objective_derivative=float(objective@direction),
        common_descent_objective_cosine=float(objective@direction/norms[-1]),
        retained=comparisons, order=retained['order'], projection=projection,
        actual_update_norm=float(torch.linalg.vector_norm(displacement)),
        scope='local_linear_diagnostic_no_parameter_update')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--learning', type=Path, required=True)
    parser.add_argument('--seal', type=Path, required=True)
    parser.add_argument('--pin', required=True)
    parser.add_argument('--root', type=Path, required=True)
    a = parser.parse_args()
    started = time.perf_counter()
    seal = authenticate(a.seal, a.pin)
    config, prior = read(a.learning/'config.json'), read(a.learning/'learning.json')
    for name in ('config.json', 'learning.json', 'training-binding.json'):
        check(evidence(a.learning/name) == seal['files']['session/'+name], 'RECOGNITION_RUN_SOURCE')
    source = Path(config['source'])
    echo_seal = read(Path(config['proof'])/'seal.json')
    for name in ('config.json', 'experience.sqlite3', 'snapshot-final.json'):
        check(evidence(source/name) == echo_seal['files']['session/'+name], 'RECOGNITION_PREFIX')
    prospective = read(Path(config['prospective'])/'config.json')
    ancestor = Path(prospective['source'])
    check(not a.root.exists(), 'RECOGNITION_ROOT_EXISTS')
    a.root.mkdir()
    (a.root/'base').mkdir()
    (a.root/'candidates').mkdir()
    shutil.copyfile(ancestor/'experience.sqlite3', a.root/'base/experience.sqlite3')
    shutil.copyfile(source/'experience.sqlite3', a.root/'experience.sqlite3')
    journal_before = {n:evidence(a.root/n) for n in ('experience.sqlite3', 'base/experience.sqlite3')}
    (a.root/'gpu.txt').open('x').write(subprocess.check_output(['nvidia-smi'], text=True))
    take_lease('28', 'proposal-recognition')
    configure()
    authenticated = time.perf_counter()
    original, _ = open_session(ancestor, prospective['source_pin'], a.root/'base', read(ancestor/'snapshot-final.json'))
    base = CorrectiveData(original, prospective['experiment']['correction'])
    session, _ = open_session(ancestor, prospective['source_pin'], a.root, read(source/'snapshot-final.json'),
                             loop_type=partial(ProposalLearningLoop, learning_base=base))
    data = ProposalData(session, base, config['spec'])
    check(data.binding == read(a.learning/'training-binding.json'), 'RECOGNITION_BINDING')
    replayed = time.perf_counter()
    arrays, stages, snapshots = {}, {}, {}
    for stage, name, pin in (('before', 'proposal-1-start', prior['start_pin']),
                             ('after', 'proposal-1', prior['candidate_pin'])):
        model, opt, manifest = field_checkpoint.load(a.learning/'candidates'/name, data, expected=pin)
        before = fingerprint(model, opt)
        stages[stage] = trace(data, model, prior[stage], arrays, stage)
        snapshots[stage] = dict(pin=pin, fingerprint=before, step=manifest['step'])
        values = torch.cat([p.detach().reshape(-1).double() for _, p in trainable(model)])
        if stage == 'before':
            start_parameters = values
        check(fingerprint(model, opt) == before, 'RECOGNITION_FORWARD_MUTATION')
    traced = time.perf_counter()
    diagnosis = gradients(data, model, values-start_parameters, arrays)
    differentiated = time.perf_counter()
    controls = []
    e = data.examples[next(iter(data.new))]
    for name, obj, code in (('input', next(iter(e['native'].values())), 'PROPOSAL_LEARNING_INPUT'),
                            ('target', next(iter(e['target'].values())), 'PROPOSAL_LEARNING_TARGET_CHANGED')):
        old = obj.values.clone()
        try:
            obj.values[0, 0, 0] += .01
            try:
                data.forward(model, next(iter(data.new)))
            except ValueError as error:
                check(code in str(error), 'RECOGNITION_CONTROL_CODE')
                controls.append(dict(name=name, code=code))
            else:
                raise AssertionError('RECOGNITION_CONTROL_ACCEPTED')
        finally:
            obj.values.copy_(old)
    check(fingerprint(model, opt) == snapshots['after']['fingerprint'], 'RECOGNITION_DIAGNOSTIC_MUTATION')
    data.fresh()
    final = session.snapshot()
    check(final == read(source/'snapshot-final.json'), 'RECOGNITION_JOURNAL_CHANGED')
    session.close()
    original.close()
    check({n:evidence(a.root/n) for n in journal_before} == journal_before, 'RECOGNITION_DB_CHANGED')
    np.savez(a.root/'arrays.npz', **arrays)
    finished = time.perf_counter()
    phases = dict(authenticate=authenticated-started, replay=replayed-authenticated,
                  checkpoints_and_scores=traced-replayed, gradients=differentiated-traced,
                  controls_and_readback=finished-differentiated)
    report = dict(schema='proposal-recognition-diagnostic-v1', source=str(a.learning), seal=str(a.seal), pin=a.pin,
        stages=stages, checkpoints=snapshots, gradient=diagnosis, controls=controls,
        new=data.new, train=data.plan['train'], binding=digest(data.binding), events=final['events'],
        head=final['head'], snapshot=digest(final), journals=journal_before, arrays=evidence(a.root/'arrays.npz'),
        phases=phases, wall_seconds=finished-started, phase_coverage=sum(phases.values())/(finished-started),
        code={Path(__file__).name:evidence(Path(__file__))},
        weights_optimizer_unchanged=True, serving=False, scientific_admission=False)
    write_committed(a.root/'diagnostic.json', report)
    print(json.dumps(dict(events=report['events'], report=sha((a.root/'diagnostic.json').read_bytes()),
                         wall_seconds=report['wall_seconds'], phases=phases)), flush=True)


if __name__ == '__main__':
    run_cli(main)
