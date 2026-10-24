"""Read actual region selection and counterfactual placements from frozen checkpoints."""
import argparse
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.fresh_learning_open import prepare, open_fresh
from audit.fsv_identity_loop import configure
from audit.proposal_recognition import authenticate, evidence, read
from claims.codata_source import check, sha
from claims.episodic_memory import digest
from claims.process_conditions import write_committed
from lens.lenscommon import run_cli, take_lease
from model.constellation_learning import ordered_loss, shared_region_loss
from model.loop_training_checkpoint import fingerprint
from model.stable_learning import StableData
from model import constellation_risk, process_anchor, stable_checkpoint


def capture(data, model, key, prefix, arrays):
    e = data.examples[key]
    output = data.forward(model, key)
    view = data.prediction_model(model, key)
    rows = next(iter(view.reference_rows.values()))
    label = torch.tensor([e['region']], device=model.core.initial.device)
    shared = shared_region_loss(output, label, e['allowed'], model.specs)
    expected = constellation_risk.expected(output, e['region_costs'], e['allowed'], model.specs)
    masses = process_anchor.tensors(output, data.labels[key], e['allowed'])
    truth = masses['common'][1].new_tensor([float(i == data.outcomes[key]) for i in range(3)])
    briers = {n:float((v[1]-truth).square().sum()) for n,v in masses.items()}
    selected = int(output['constellation_regions'].argmax(-1)[0])
    report = dict(target=e['region'], selected=selected, rows=list(rows), outcome=data.outcomes[key],
        selected_id=output['predicted_constellation'].regions[0], common_brier=briers['common'],
        shared=float(shared), instruments={}, bank=view.execution_anchor_bank,
        input=data.new[key], world=view.world_id)
    arrays[prefix+'_common'] = output['constellation_regions'][0].cpu().numpy()
    arrays[prefix+'_labels'] = data.labels[key].cpu().numpy()
    arrays[prefix+'_allowed'] = e['allowed'].cpu().numpy()
    all_placements = []
    for j in range(len(rows)):
        choice = torch.tensor([j], device=model.core.initial.device)
        # Diagnostic only: never fed back to forward, prediction, or model state.
        all_placements.append(view._placement(output['placement_hidden'], choice,
                                             [output['states'][0].last_event]))
    for i,spec in enumerate(model.specs):
        tag = prefix+'_'+str(i)
        target = e['target'][spec.name]
        target = target.values[0, ~target.padding[0]]
        arrays[tag+'_target'] = target.cpu().numpy()
        arrays[tag+'_logits'] = output['regions'][spec.name][0].cpu().numpy()
        arrays[tag+'_costs'] = e['region_costs'][spec.name][1].cpu().numpy()
        arrays[tag+'_native_residual'] = output['native_residuals'][spec.name][0].cpu().numpy()
        arrays[tag+'_field_residual'] = output['field_residuals'][spec.name][0].cpu().numpy()
        references = view.references()[spec.name]
        for j,placed in enumerate(all_placements):
            slot = placed.slots[spec.name]
            arrays[tag+'_placed_'+str(j)] = slot.values[0, ~slot.padding[0]].cpu().numpy()
            arrays[tag+'_reference_'+str(j)] = references.values[j, ~references.padding[j]].cpu().numpy()
        teacher = all_placements[e['region']].slots[spec.name]
        placement = ordered_loss(teacher.values[0, ~teacher.padding[0]], target)
        lane = dict(identity=spec.identity, placement=float(placement), expected_native=float(expected[spec.name]),
                    brier=briers[spec.name], target_windows=len(target))
        report['instruments'][spec.name] = lane
    lanes = list(report['instruments'].values())
    report['objective'] = report['shared']+sum(v['placement']+v['expected_native']+v['brier']+
        report['common_brier']/len(lanes) for v in lanes)/len(lanes)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--proof', type=Path, required=True)
    parser.add_argument('--proof-pin', required=True)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    started = time.perf_counter()
    seal = authenticate(a.proof, a.proof_pin)
    for name in ('learning.json', 'config.json'):
        check(evidence(a.source/name) == seal['files']['session/'+name], 'REGION_AUDIT_SOURCE')
    config, learning = read(a.source/'config.json'), read(a.source/'learning.json')
    resolved = prepare(config, a.output)
    write_committed(a.output/'config.json', config)
    gpu = subprocess.check_output(['nvidia-smi'], text=True)
    take_lease('28', 'region-selection-audit')
    (a.output/'gpu.txt').open('x').write(gpu)
    configure()
    session,base,sessions = open_fresh(config, a.output, sources_value=resolved)
    data = StableData(session, base, config['spec'])
    check(data.binding == learning['binding'], 'REGION_AUDIT_BINDING')
    snapshot = session.snapshot()
    replayed = time.perf_counter()
    arrays, states, phases = {}, {}, dict(authentication_replay=replayed-started)
    for stage, suffix, pin_name in (('start','-start','start_pin'), ('mid','-mid','mid_pin'),
                                   ('final','','candidate_pin')):
        tick = time.perf_counter()
        root = a.source/'candidates'/('stable-1'+suffix)
        model,optimizer,manifest = stable_checkpoint.load(root,data,expected=learning[pin_name])
        prior = fingerprint(model,optimizer)
        evaluation = data.evaluate(model)
        if stage != 'mid':
            check(evaluation == learning['before' if stage == 'start' else 'after'], 'REGION_AUDIT_REPRODUCTION')
        with torch.no_grad():
            details = {key:capture(data,model,key,stage+'_'+str(i),arrays)
                       for i,key in enumerate(data.new)}
        check(fingerprint(model,optimizer) == prior, 'REGION_AUDIT_CHECKPOINT_CHANGED')
        states[stage] = dict(checkpoint=str(root), pin=learning[pin_name], fingerprint=manifest['fingerprint'],
                             examples=details, evaluation=evaluation)
        phases[stage] = time.perf_counter()-tick
        print(stage+' captured', flush=True)
    tick = time.perf_counter()
    check(session.snapshot() == snapshot, 'REGION_AUDIT_HISTORY_CHANGED')
    for item in sessions:
        item.close()
    np.savez(a.output/'arrays.npz', **arrays)
    phases['persist_close'] = time.perf_counter()-tick
    elapsed = time.perf_counter()-started
    report = dict(schema='region-selection-audit-v1', source=str(a.source), source_learning=evidence(a.source/'learning.json'),
        proof=dict(root=str(a.proof),pin=a.proof_pin), binding=digest(data.binding), snapshot=digest(snapshot),
        examples=list(data.new), instruments=[s.name for s in data.model.specs], states=states,
        arrays=evidence(a.output/'arrays.npz'), phases=phases, wall_seconds=elapsed,
        phase_coverage=sum(phases.values())/elapsed, checkpoint_unchanged=True, history_unchanged=True,
        code={n:sha((Path(__file__).parent/n).read_bytes()) for n in ('region_selection.py','region_selection_readback.py')},
        acceptance='author-checked', scope='exposed_feedback_diagnostic_not_transfer', serving=False, scientific_admission=False)
    write_committed(a.output/'report.json', report)
    print('report '+sha((a.output/'report.json').read_bytes()), flush=True)


if __name__ == '__main__':
    run_cli(main)
