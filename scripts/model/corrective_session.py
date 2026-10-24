"""Journal corrective learners separately from the preserved failed branch."""
from claims.codata_source import check
from claims.episodic_memory import digest
from model.constellation_learning import decision
from model.controlled_session import ControlledLoop, candidate_path
from model.corrective_learning import CorrectiveData, code_pins
from model.tool_rehearsal import termination_gate
from model import field_checkpoint, rehearsal_retention, tool_retention


class CorrectiveLoop(ControlledLoop):
    def __init__(self, *args, **kwargs):
        self.corrective_candidates = {}
        super().__init__(*args, **kwargs)

    def consider_corrective(self, name, pin, spec, expected=None):
        self._fresh()
        check(name not in self.corrective_candidates and name not in self.controlled_candidates,
              'CORRECTION_DUPLICATE_CANDIDATE')
        data = CorrectiveData(self, spec)
        model, _, manifest = field_checkpoint.load(candidate_path(self,name), data, expected=pin)
        check(manifest['step'] == data.plan['steps'] and manifest['parent'] == data.parent_pin,
              'CORRECTION_CANDIDATE_INCOMPLETE')
        original, _, _ = field_checkpoint.start(data)
        baseline, candidate = data.evaluate(original), data.evaluate(model)
        check(baseline == data.baseline_scores, 'CORRECTION_BASELINE_CONTINUITY')
        result = decision(data,baseline,candidate,model)
        result = tool_retention.assess(data,result,candidate)
        result = termination_gate(data,baseline,candidate,result)
        result = rehearsal_retention.assess(data,candidate,result)
        old_loss, new_loss = (float(data.loss(m)[0].detach()) for m in (original,model))
        gain = new_loss < old_loss
        result = dict(result, current_objective=dict(before=old_loss,after=new_loss,improved=gain),
            metric_gate_passed=result['metric_gate_passed'] and gain,
            reasons=result['reasons']+([] if gain else ['CORRECTION_NO_OBJECTIVE_GAIN']))
        payload = dict(baseline=baseline,candidate=candidate,result=result,controlled_pairs=data.controlled)
        body = dict(kind='corrective_feedback_candidate',candidate=name,pin=pin,parent=data.parent_pin,
            spec=spec,binding=digest(data.binding),result=result,payload=digest(payload),prior_head=self.head,code=code_pins())
        if expected is None:
            self.journal.append(body,payload)
        else:
            check(body == expected and payload == self.journal.payload(expected['payload']), 'CORRECTION_REPLAY')
        self.corrective_candidates[name] = body
        return body

    def require_selection(self, name):
        if name in self.corrective_candidates:
            check(self.corrective_candidates[name]['result']['selected'], 'TOOL_CANDIDATE_NOT_SELECTED')
            check(False,'CORRECTION_ACTIVATION_NOT_ADMITTED')
        return super().require_selection(name)

    def _replay(self, body):
        if body.get('kind') == 'corrective_feedback_candidate':
            check(body['code'] == code_pins(), 'CORRECTION_CODE_PIN')
            self.consider_corrective(body['candidate'],body['pin'],body['spec'],expected=body)
        else:
            super()._replay(body)

    def snapshot(self):
        result = super().snapshot()
        if self.corrective_candidates:
            result['corrective_feedback_candidates'] = dict(self.corrective_candidates)
        return result
