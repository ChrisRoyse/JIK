"""One bounded checkpoint/recovery driver for versioned candidate update recipes."""
import json
import time

from claims.codata_source import check
from model.loop_training_checkpoint import fingerprint, tensor_fingerprint
from model import field_checkpoint, rehearsal_retention


def increment(session, root, *, name, spec, data_factory, update, review, eligible):
    started = time.perf_counter()
    data = data_factory(session,session.learning_base,spec)
    model, optimizer, migration = field_checkpoint.start(data)
    before = data.evaluate(model)
    check(all(r['retained'] for r in rehearsal_retention.comparisons(data,
        rehearsal_retention.scores(data,model))), 'BOUNDED_INITIAL_RETENTION')
    parameters = tensor_fingerprint(dict(model.named_parameters()))
    allowed = set(eligible(data,model))
    initial = field_checkpoint.save(root/(name+'-start'),model,optimizer,data,step=0,parent=data.parent_pin)
    phases = dict(binding_and_baseline=time.perf_counter()-started)
    check(data.plan['steps'] > 0 and data.plan['steps'] % 2 == 0, 'BOUNDED_STEP_PLAN')
    middle = data.plan['steps']//2
    tick, losses = time.perf_counter(), []
    for cursor in range(1,data.plan['steps']+1):
        row = update(data,model,optimizer)
        losses.append(row)
        print(json.dumps(dict(step=cursor,accepted=row['update']['accepted'],
            objective=row['update']['objective_after'],scale=row['update']['scale'])),flush=True)
        if cursor == middle:
            mid = field_checkpoint.save(root/(name+'-mid'),model,optimizer,data,step=cursor,parent=data.parent_pin)
    phases['updates'] = time.perf_counter()-tick
    tick = time.perf_counter()
    recovered, recovered_optimizer, _ = field_checkpoint.load(root/(name+'-mid'),data,expected=mid)
    for cursor in range(middle,data.plan['steps']):
        check(update(data,recovered,recovered_optimizer) == losses[cursor], 'BOUNDED_UPDATE_REPLAY')
    check(fingerprint(recovered,recovered_optimizer) == fingerprint(model,optimizer), 'BOUNDED_OPTIMIZER_RECOVERY')
    phases['optimizer_recovery'] = time.perf_counter()-tick
    tick = time.perf_counter()
    after = data.evaluate(model)
    now = tensor_fingerprint(dict(model.named_parameters()))
    changed = [n for n in parameters if parameters[n] != now[n]]
    check(set(changed) <= allowed, 'BOUNDED_PARAMETER_SCOPE')
    pin = field_checkpoint.save(root/name,model,optimizer,data,step=data.plan['steps'],parent=data.parent_pin)
    rollback, rollback_optimizer, _ = field_checkpoint.load(root/(name+'-start'),data,expected=initial)
    check(data.evaluate(rollback) == before, 'BOUNDED_ROLLBACK_BEHAVIOUR')
    result = review(session,name,pin,spec)
    phases['readback_rollback_decision'] = time.perf_counter()-tick
    wall = time.perf_counter()-started
    return dict(binding=data.binding,before=before,after=after,losses=losses,migration=migration,
        changed=changed,eligible=sorted(allowed),decision=result,exact_optimizer_recovery=True,
        rollback_behaviour_equal=True,rollback_fingerprint=fingerprint(rollback,rollback_optimizer),
        start_pin=initial,mid_pin=mid,candidate_pin=pin,parent=data.parent_pin,phases=phases,
        wall_seconds=wall,phase_coverage=sum(phases.values())/wall,serving=False,scientific_admission=False)
