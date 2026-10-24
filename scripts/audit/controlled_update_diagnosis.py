"""Replay a sealed failed update and attribute objective movement to its displacement."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.fsv_identity_loop import configure
import numpy as np
import torch
from audit.controlled_open import open_session
from audit.fsv_field_world import authenticate
from audit.fsv_process_feedback import write
from claims.codata_source import check, sha
from claims.episodic_memory import digest
from lens.lenscommon import run_cli, take_lease
from model.constellation_learning import ordered_loss, shared_region_loss, trainable
from model.controlled_learning import ControlledData
from model.loop_training_checkpoint import fingerprint
from model import field_checkpoint, field_retention, rehearsal_retention


def objective(data, model, key):
    e = data.examples[key]
    output = data.forward(model, key)
    label = torch.tensor([e['region']], dtype=torch.long, device=model.core.initial.device)
    shared = shared_region_loss(output, label, e['allowed'], model.specs)
    risks = data.recognition_risk(model, key, output)
    placed = data.prediction_model(model, key).supervised_placement(
        output, label, allowed=e['allowed'], event_ids=[output['states'][0].last_event])
    terms = []
    for spec in model.specs:
        slot, target = placed.slots[spec.name], e['target'][spec.name]
        terms.append(ordered_loss(slot.values[0, ~slot.padding[0]],
                                  target.values[0, ~target.padding[0]])+risks[spec.name])
    loss = shared+torch.stack(terms).mean()
    return loss, dict(loss=float(loss.detach()), shared_region=float(shared.detach()),
                     region=output['predicted_constellation'].regions[0])


def read_training(data, model):
    # These are supervised training losses, never inference-time predictions.
    return {k:objective(data, model, k)[1] for k in data.plan['train']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--pin', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    started = time.perf_counter()
    root = a.output
    root.mkdir(exist_ok=False)
    proof = authenticate(a.source, a.pin)
    source = json.loads((a.source/'source.json').read_bytes())
    echo = Path(source['root'])
    saved = json.loads((echo/'snapshot-final.json').read_bytes())
    recorded = json.loads((a.source/'learning.json').read_bytes())
    shutil.copyfile(echo/'experience.sqlite3', root/'experience.sqlite3')
    gpu = subprocess.check_output(['nvidia-smi']).decode()
    take_lease('28', 'controlled-update-diagnosis')
    (root/'gpu-before.txt').write_text(gpu)
    configure()
    session, echo_proof = open_session(echo, source['seal'], root, saved)
    data = ControlledData(session, recorded['decision']['spec'])
    model, optimizer, initial = field_checkpoint.load(a.source/'candidates/controlled-1-start', data,
                                                     expected=recorded['start_pin'])
    check(data.evaluate(model) == recorded['before'], 'DIAGNOSIS_BASELINE')
    phases = dict(authentication_and_baseline=time.perf_counter()-started)
    names, params = zip(*trainable(model))
    order = list(data.plan['train'])
    rows, vector_file = [], None
    tick = time.perf_counter()
    for cursor, expected in enumerate(recorded['losses'], 1):
        optimizer.zero_grad(set_to_none=True)
        protected = rehearsal_retention.prepare(data, model)
        gradients, before = [], {}
        for key in order:
            loss, before[key] = objective(data, model, key)
            values = torch.autograd.grad(loss, params, allow_unused=True)
            gradients.append(torch.cat([(v if v is not None else torch.zeros_like(q)).detach().reshape(-1).double()
                                        for v,q in zip(values, params)]))
        g = torch.stack(gradients)
        loss, details = data.loss(model)
        check(abs(float(loss.detach())-sum(v['loss'] for v in before.values())/len(order)) < 1e-12,
              'DIAGNOSIS_OBJECTIVE_EQUALITY')
        loss.backward()
        total = torch.cat([(q.grad if q.grad is not None else torch.zeros_like(q)).detach().reshape(-1).double()
                           for q in params])
        gradient_error = float((total-g.mean(0)).abs().max())
        check(torch.allclose(total, g.mean(0), rtol=1e-5, atol=1e-7), 'DIAGNOSIS_GRADIENT_EQUALITY')
        active = [n for n,q in zip(names,params) if q.grad is not None and bool((q.grad != 0).any())]
        torch.nn.utils.clip_grad_norm_(params, 1.)
        old = {n:q.detach().clone() for n,q in zip(names,params)}
        optimizer.step()
        raw = torch.cat([(q.detach()-old[n]).reshape(-1).double() for n,q in zip(names,params)])
        raw_scores = read_training(data, model)
        update = field_retention.apply(data, model, old, protected)
        actual = torch.cat([(q.detach()-old[n]).reshape(-1).double() for n,q in zip(names,params)])
        result = dict(loss=float(loss.detach()), details=details, gradients=active, update=update)
        # JSON stores the original tuple-based constraint order as arrays.
        if digest(result) != digest(expected):
            write(root/('replay-mismatch-'+str(cursor)+'.json'), dict(expected=expected, actual=result))
        check(digest(result) == digest(expected), 'DIAGNOSIS_EXACT_UPDATE_REPLAY')
        after = read_training(data, model)
        raw_dot, actual_dot = (g@raw).tolist(), (g@actual).tolist()
        row = dict(step=cursor, before=before, raw=raw_scores, after=after,
            directional={k:dict(raw_adam=raw_dot[i], retained=actual_dot[i]) for i,k in enumerate(order)},
            total_directional=dict(raw_adam=float(total@raw), retained=float(total@actual)),
            gradient_max_error=gradient_error, raw_norm=float(torch.linalg.vector_norm(raw)),
            retained_norm=float(torch.linalg.vector_norm(actual)), scale=update['scale'],
            projection=update['projections']['status'], exact_update_replay=True)
        switched = [k for k in data.controlled if before[k]['region'] != after[k]['region']]
        row['controlled_region_switches'] = switched
        if switched and vector_file is None:
            vector_file = dict(step=cursor, path='boundary-vectors.npz', examples=order,
                               parameters=list(names), coordinates=len(raw))
            np.savez(root/vector_file['path'], gradients=g.cpu().numpy(),
                     raw=raw.cpu().numpy(), retained=actual.cpu().numpy(), total=total.cpu().numpy())
            vector_file.update(sha256=sha((root/vector_file['path']).read_bytes()),
                               bytes=(root/vector_file['path']).stat().st_size)
        rows.append(row)
    phases['exact_updates_and_directional_audit'] = time.perf_counter()-tick
    tick = time.perf_counter()
    final = json.loads((a.source/'candidates/controlled-1/manifest.json').read_bytes())
    check(fingerprint(model, optimizer) == final['fingerprint'], 'DIAGNOSIS_EXACT_FINAL')
    check(data.evaluate(model) == recorded['after'], 'DIAGNOSIS_EXACT_BEHAVIOUR')
    check(session.snapshot() == saved, 'DIAGNOSIS_JOURNAL_MUTATION')
    session.close()
    check(sha((echo/'experience.sqlite3').read_bytes()) == source['journal'] ==
          sha((root/'experience.sqlite3').read_bytes()), 'DIAGNOSIS_JOURNAL_BYTES')
    check(vector_file is not None, 'DIAGNOSIS_MISSING_BOUNDARY')
    phases['final_readback'] = time.perf_counter()-tick
    wall = time.perf_counter()-started
    report = dict(complete=True, acceptance='author-checked', source=dict(root=str(a.source), seal=a.pin),
        code=sha(Path(__file__).read_bytes()), source_proof=proof, echo_proof=echo_proof,
        start_fingerprint=initial['fingerprint'], final_fingerprint=final['fingerprint'],
        original_journal_unchanged=True, active_model_unchanged=True,
        steps=rows, vectors=vector_file, exact_replay=True, phases=phases, wall_seconds=wall,
        phase_coverage=sum(phases.values())/wall,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
        serving=False, scientific_admission=False)
    write(root/'diagnosis.json', report)
    print(json.dumps(dict(complete=True, boundary=vector_file,
        directional=[dict(step=r['step'], total=r['total_directional'], switches=r['controlled_region_switches']) for r in rows],
        wall_seconds=wall)), flush=True)


if __name__ == '__main__':
    run_cli(main)
