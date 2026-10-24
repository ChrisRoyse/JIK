"""Versioned proposal correction using unchanged retention ceilings and real slack."""
import copy
import json
from pathlib import Path
import time

import torch

from claims.codata_source import check, sha
from claims.episodic_memory import digest
from model.constellation_learning import decision, trainable
from model.loop_training_checkpoint import fingerprint, tensor_fingerprint
from model.proposal_learning import ProposalData
from model.proposal_learning_session import ProposalLearningLoop
from model.tool_rehearsal import termination_gate
from model import field_checkpoint, rehearsal_retention, slack_update, tool_retention


def code_pins():
    return {n:sha((Path(__file__).parent/n).read_bytes()) for n in ('slack_learning.py', 'slack_update.py')}


class SlackData(ProposalData):
    def __init__(self, session, base, spec):
        super().__init__(session, base, spec['proposal'])
        root = Path(spec['continuation']['root'])
        raw = (root/'manifest.json').read_bytes()
        check(sha(raw) == spec['continuation']['pin'], 'SLACK_PARENT_PIN')
        manifest = json.loads(raw)
        check(manifest['binding'] == self.binding and manifest['step'] == self.plan['steps'],
              'SLACK_PARENT_BINDING')
        report_raw = Path(spec['evidence']['path']).read_bytes()
        check(sha(report_raw) == spec['evidence']['pin'], 'SLACK_PARENT_EVIDENCE')
        report = json.loads(report_raw)
        check(report['candidate_pin'] == spec['continuation']['pin'] and report['binding'] == self.binding,
              'SLACK_PARENT_DECISION')
        self.field_predecessor = root, spec['continuation']['pin'], manifest
        self.parent_pin = spec['continuation']['pin']
        self.spec = copy.deepcopy(spec)
        self.binding['native_world']['offsets'] = {n:v+manifest['step'] for n,v in manifest['offsets'].items()}
        ceilings = digest(self.binding['grounded_retention']['ceilings'])
        self.slack_code = code_pins()
        self.binding['slack_learning'] = dict(rule=slack_update.RULE, code=self.slack_code, spec=spec,
            predecessor_binding=digest(manifest['binding']), ceilings=ceilings,
            parent_decision=digest(report['decision']), scope='exposed_feedback_continuation_not_transfer')
        self.binding_pin = digest(self.binding)
        self.fresh()

    def fresh(self):
        super().fresh()
        if 'slack_learning' in self.binding:
            saved = self.binding['slack_learning']
            check(saved['code'] == self.slack_code and saved['rule'] == slack_update.RULE,
                  'SLACK_LEARNING_CODE')
            check(saved['ceilings'] == digest(self.binding['grounded_retention']['ceilings']),
                  'SLACK_CEILINGS_CHANGED')


class SlackLearningLoop(ProposalLearningLoop):
    def __init__(self, *args, **kwargs):
        self.slack_candidates = {}
        super().__init__(*args, **kwargs)

    def consider_slack_learning(self, name, pin, spec, expected=None):
        self._fresh()
        check(not self.slack_candidates, 'SLACK_DUPLICATE_DECISION')
        check(isinstance(name, str) and name not in ('', '.', '..') and '/' not in name and '\\' not in name,
              'SLACK_CANDIDATE_PATH')
        root = self.controlled_root/name
        check(root.resolve().parent == self.controlled_root.resolve(), 'SLACK_CANDIDATE_PATH')
        data = SlackData(self, self.learning_base, spec)
        candidate, _, manifest = field_checkpoint.load(root, data, expected=pin)
        check(manifest['step'] == data.plan['steps'] and manifest['parent'] == data.parent_pin, 'SLACK_INCOMPLETE')
        parent, _, _ = field_checkpoint.start(data)
        before, after = data.evaluate(parent), data.evaluate(candidate)
        result = decision(data, before, after, candidate)
        result = tool_retention.assess(data, result, after)
        result = termination_gate(data, before, after, result)
        result = rehearsal_retention.assess(data, after, result)
        payload = dict(before=before, after=after, result=result, binding=digest(data.binding),
            source_prediction_unchanged=True, weight_learning_comparison='same_expanded_reference_bank',
            serving=False, scientific_admission=False)
        body = dict(kind='slack_learning_candidate', candidate=name, pin=pin, parent=data.parent_pin,
            spec=spec, prior_head=self.head, payload=digest(payload), code=code_pins(), result=result)
        if expected is None:
            self.journal.append(body, payload)
        else:
            check(body == expected and payload == self.journal.payload(expected['payload']), 'SLACK_DECISION_REPLAY')
        self.slack_candidates[name] = body
        return body

    def _replay(self, body):
        if body.get('kind') == 'slack_learning_candidate':
            check(body['code'] == code_pins(), 'SLACK_REPLAY_CODE')
            self.consider_slack_learning(body['candidate'], body['pin'], body['spec'], expected=body)
        else:
            super()._replay(body)

    def snapshot(self):
        result = super().snapshot()
        if self.slack_candidates:
            result['slack_learning_candidates'] = self.slack_candidates
        return result


def step(data, model, optimizer):
    optimizer.zero_grad(set_to_none=True)
    protected = rehearsal_retention.prepare(data, model)
    loss, details = data.loss(model)
    check(bool(torch.isfinite(loss)), 'SLACK_NONFINITE_LOSS')
    loss.backward()
    named = trainable(model)
    gradient = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).detach().reshape(-1).double()
                          for _,p in named])
    torch.nn.utils.clip_grad_norm_([p for _,p in named], 1.)
    before = {n:p.detach().clone() for n,p in named}
    optimizer.step()
    result = slack_update.apply(data, model, before, protected, float(loss.detach()), gradient)
    return dict(loss=float(loss.detach()), details=details, update=result)


def increment(session, root, *, name, spec):
    started = time.perf_counter()
    data = SlackData(session, session.learning_base, spec)
    model, optimizer, migration = field_checkpoint.start(data)
    before = data.evaluate(model)
    check(all(r['retained'] for r in rehearsal_retention.comparisons(data,
        rehearsal_retention.scores(data, model))), 'SLACK_INITIAL_RETENTION')
    parameters = tensor_fingerprint(dict(model.named_parameters()))
    initial = field_checkpoint.save(root/(name+'-start'), model, optimizer, data, step=0, parent=data.parent_pin)
    phases = dict(binding_and_baseline=time.perf_counter()-started)
    tick, losses = time.perf_counter(), []
    for cursor in range(1, data.plan['steps']+1):
        row = step(data, model, optimizer)
        losses.append(row)
        print(json.dumps(dict(step=cursor, accepted=row['update']['accepted'],
            projection=row['update']['projection']['status'], objective=row['update']['objective_after'])), flush=True)
        if cursor == 4:
            mid = field_checkpoint.save(root/(name+'-mid'), model, optimizer, data, step=cursor, parent=data.parent_pin)
    phases['updates'] = time.perf_counter()-tick
    tick = time.perf_counter()
    recovered, recovered_optimizer, _ = field_checkpoint.load(root/(name+'-mid'), data, expected=mid)
    for cursor in range(4, data.plan['steps']):
        check(step(data, recovered, recovered_optimizer) == losses[cursor], 'SLACK_UPDATE_REPLAY')
    check(fingerprint(recovered, recovered_optimizer) == fingerprint(model, optimizer), 'SLACK_OPTIMIZER_RECOVERY')
    phases['optimizer_recovery'] = time.perf_counter()-tick
    tick = time.perf_counter()
    after = data.evaluate(model)
    now = tensor_fingerprint(dict(model.named_parameters()))
    changed = [n for n in parameters if parameters[n] != now[n]]
    check(all(n.startswith(('query.', 'place.')) for n in changed), 'SLACK_FROZEN_DRIFT')
    pin = field_checkpoint.save(root/name, model, optimizer, data, step=data.plan['steps'], parent=data.parent_pin)
    rollback, rollback_optimizer, _ = field_checkpoint.load(root/(name+'-start'), data, expected=initial)
    check(data.evaluate(rollback) == before, 'SLACK_ROLLBACK_BEHAVIOUR')
    result = session.consider_slack_learning(name, pin, spec)
    phases['readback_rollback_decision'] = time.perf_counter()-tick
    wall = time.perf_counter()-started
    return dict(binding=data.binding, before=before, after=after, losses=losses, migration=migration,
        changed=changed, decision=result, exact_optimizer_recovery=True, rollback_behaviour_equal=True,
        rollback_fingerprint=fingerprint(rollback, rollback_optimizer), start_pin=initial, mid_pin=mid,
        candidate_pin=pin, parent=data.parent_pin, phases=phases, wall_seconds=wall,
        phase_coverage=sum(phases.values())/wall, serving=False, scientific_admission=False)
