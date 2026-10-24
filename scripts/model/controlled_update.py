"""Fixed retained update recipe on captured controlled outcomes, with exact recovery."""
import time

import torch

from claims.codata_source import check
from model.constellation_learning import trainable
from model.controlled_learning import ControlledData
from model.controlled_session import candidate_path
from model.loop_training_checkpoint import fingerprint, tensor_fingerprint
from model import field_checkpoint, field_retention, rehearsal_retention


def increment(session, root, *, name, spec):
    started = time.perf_counter()
    data = ControlledData(session, spec)
    model, optimizer, migration = field_checkpoint.start(data)
    before = data.evaluate(model)
    previous = (next(reversed(session.controlled_candidates.values())) if session.controlled_candidates
                else session.decisions[-1])
    previous_scores = session.journal.payload(previous['payload'])['candidate']
    check(all(before[k] == previous_scores[k] for k in previous_scores), 'CONTROLLED_CONTINUITY')
    # A changed event identifier does not change the sealed numerical forecast.
    if not session.controlled_candidates:
        for key, pair in data.controlled.items():
            entry = session.echo_contexts[pair['context']]
            forecast = session.experiments[entry['experiment']]['forecast']
            sealed = session.journal.payload(forecast['payload'])['predictions'][entry['condition']]
            check(before[key]['predictions'] == sealed['constellation'] and
                  before[key]['termination']['prediction'] == sealed['candidate'], 'CONTROLLED_SEALED_CONTINUITY')
    parameters = tensor_fingerprint(dict(model.named_parameters()))
    initial = field_checkpoint.save(root/(name+'-start'), model, optimizer, data, step=0, parent=data.parent_pin)
    phases = dict(binding_and_baseline=time.perf_counter()-started)

    def step(model, optimizer):
        optimizer.zero_grad(set_to_none=True)
        protected = rehearsal_retention.prepare(data, model)
        loss, details = data.loss(model)
        check(bool(torch.isfinite(loss)), 'CONTROLLED_NONFINITE_LOSS')
        loss.backward()
        gradients = [n for n,p in trainable(model) if p.grad is not None and bool((p.grad != 0).any())]
        check(gradients and all(n.startswith(('query.','place.')) for n in gradients), 'CONTROLLED_GRADIENT_SCOPE')
        torch.nn.utils.clip_grad_norm_([p for _,p in trainable(model)], 1.)
        old = {n:p.detach().clone() for n,p in trainable(model)}
        optimizer.step()
        update = field_retention.apply(data, model, old, protected)
        return dict(loss=float(loss.detach()), details=details, gradients=gradients, update=update)

    tick = time.perf_counter()
    losses = []
    for cursor in range(1, data.plan['steps']+1):
        losses.append(step(model, optimizer))
        if cursor == 4:
            mid = field_checkpoint.save(root/(name+'-mid'), model, optimizer, data, step=cursor, parent=data.parent_pin)
    phases['fixed_updates'] = time.perf_counter()-tick
    tick = time.perf_counter()
    restored, restored_optimizer, _ = field_checkpoint.load(root/(name+'-mid'), data, expected=mid)
    for cursor in range(4, data.plan['steps']):
        check(step(restored, restored_optimizer) == losses[cursor], 'CONTROLLED_UPDATE_REPLAY')
    check(fingerprint(restored, restored_optimizer) == fingerprint(model, optimizer), 'CONTROLLED_OPTIMIZER_RECOVERY')
    phases['optimizer_recovery'] = time.perf_counter()-tick
    tick = time.perf_counter()
    after = data.evaluate(model)
    now = tensor_fingerprint(dict(model.named_parameters()))
    changed = [n for n in parameters if parameters[n] != now[n]]
    check(all(n.startswith(('query.','place.')) for n in changed), 'CONTROLLED_FROZEN_DRIFT')
    pin = field_checkpoint.save(candidate_path(session, name), model, optimizer, data,
                                step=data.plan['steps'], parent=data.parent_pin)
    # Roll back to the saved parent state, including its optimizer, and read behaviour again.
    rollback, rollback_optimizer, _ = field_checkpoint.load(root/(name+'-start'), data, expected=initial)
    check(data.evaluate(rollback) == before, 'CONTROLLED_ROLLBACK_BEHAVIOUR')
    selected = session.consider_controlled(name, pin, spec)
    phases['readback_rollback_and_decision'] = time.perf_counter()-tick
    wall = time.perf_counter()-started
    return dict(complete=True, binding=data.binding, before=before, after=after, losses=losses,
        migration=migration, changed=changed, decision=selected, exact_optimizer_recovery=True,
        rollback_fingerprint=fingerprint(rollback, rollback_optimizer), rollback_behaviour_equal=True,
        start_pin=initial, mid_pin=mid, candidate_pin=pin,
        parameter_count=sum(p.numel() for p in model.parameters()),
        trainable_count=sum(p.numel() for _,p in trainable(model)),
        phases=phases, wall_seconds=wall, phase_coverage=sum(phases.values())/wall)
