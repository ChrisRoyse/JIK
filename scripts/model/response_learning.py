"""Versioned learning from proposal feedback under retained-response constraints."""
import copy
import json
from pathlib import Path

from claims.codata_source import check, sha
from claims.episodic_memory import digest
from model.constellation_learning import decision
from model.slack_learning import SlackData, SlackLearningLoop
from model.tool_rehearsal import termination_gate
from model import bounded_training, field_checkpoint, rehearsal_retention, response_basis, response_update, tool_retention


def code_pins():
    return {n:sha((Path(__file__).parent/n).read_bytes()) for n in
            ('response_learning.py','response_basis.py','response_update.py','bounded_training.py')}


class ResponseData(SlackData):
    def __init__(self, session, base, spec):
        super().__init__(session,base,spec['previous'])
        root = Path(spec['continuation']['root'])
        raw = (root/'manifest.json').read_bytes()
        check(sha(raw) == spec['continuation']['pin'], 'RESPONSE_PARENT_PIN')
        manifest = json.loads(raw)
        check(manifest['binding'] == self.binding and manifest['step'] == self.plan['steps'], 'RESPONSE_PARENT_BINDING')
        raw = Path(spec['evidence']['path']).read_bytes()
        check(sha(raw) == spec['evidence']['pin'], 'RESPONSE_PARENT_EVIDENCE')
        report = json.loads(raw)
        check(report['candidate_pin'] == spec['continuation']['pin'] and report['binding'] == self.binding,
              'RESPONSE_PARENT_DECISION')
        self.response_basis = response_basis.Basis(spec['basis'],spec['continuation']['pin'],digest(self.binding))
        check(self.response_basis.manifest['retained_keys'] == self.retained_keys, 'RESPONSE_RETAINED_SPLIT')
        self.field_predecessor = root,spec['continuation']['pin'],manifest
        self.parent_pin, self.spec = spec['continuation']['pin'],copy.deepcopy(spec)
        self.binding['native_world']['offsets'] = {n:v+manifest['step'] for n,v in manifest['offsets'].items()}
        self.response_code = code_pins()
        self.binding['response_learning'] = dict(rule=response_basis.RULE,code=self.response_code,spec=spec,
            predecessor_binding=digest(manifest['binding']),parent_decision=digest(report['decision']),
            ceilings=digest(self.binding['grounded_retention']['ceilings']),scope='exposed_feedback_not_transfer')
        self.binding_pin = digest(self.binding)
        self.fresh()

    def fresh(self):
        super().fresh()
        if 'response_learning' in self.binding:
            saved = self.binding['response_learning']
            check(saved['rule'] == response_basis.RULE and saved['code'] == self.response_code, 'RESPONSE_LEARNING_CODE')
            check(saved['ceilings'] == digest(self.binding['grounded_retention']['ceilings']), 'RESPONSE_CEILINGS_CHANGED')


class ResponseLearningLoop(SlackLearningLoop):
    def __init__(self, *args, **kwargs):
        self.response_candidates = {}
        super().__init__(*args,**kwargs)

    def consider_response_learning(self, name, pin, spec, expected=None):
        self._fresh()
        check(not self.response_candidates, 'RESPONSE_DUPLICATE_DECISION')
        check(isinstance(name,str) and name not in ('','.','..') and '/' not in name and '\\' not in name,
              'RESPONSE_CANDIDATE_PATH')
        root = self.controlled_root/name
        check(root.resolve().parent == self.controlled_root.resolve(), 'RESPONSE_CANDIDATE_PATH')
        data = ResponseData(self,self.learning_base,spec)
        candidate, _, manifest = field_checkpoint.load(root,data,expected=pin)
        check(manifest['step'] == data.plan['steps'] and manifest['parent'] == data.parent_pin, 'RESPONSE_INCOMPLETE')
        parent, _, _ = field_checkpoint.start(data)
        before, after = data.evaluate(parent),data.evaluate(candidate)
        data.response_basis.measure(data,candidate)
        result = decision(data,before,after,candidate)
        result = tool_retention.assess(data,result,after)
        result = termination_gate(data,before,after,result)
        result = rehearsal_retention.assess(data,after,result)
        payload = dict(before=before,after=after,result=result,binding=digest(data.binding),
            source_prediction_unchanged=True,weight_learning_comparison='same_expanded_reference_bank',
            serving=False,scientific_admission=False)
        body = dict(kind='response_learning_candidate',candidate=name,pin=pin,parent=data.parent_pin,
            spec=spec,prior_head=self.head,payload=digest(payload),code=code_pins(),result=result)
        if expected is None:
            self.journal.append(body,payload)
        else:
            check(body == expected and payload == self.journal.payload(expected['payload']), 'RESPONSE_DECISION_REPLAY')
        self.response_candidates[name] = body
        return body

    def _replay(self, body):
        if body.get('kind') == 'response_learning_candidate':
            check(body['code'] == code_pins(), 'RESPONSE_REPLAY_CODE')
            self.consider_response_learning(body['candidate'],body['pin'],body['spec'],expected=body)
        else:
            super()._replay(body)

    def snapshot(self):
        result = super().snapshot()
        if self.response_candidates:
            result['response_learning_candidates'] = self.response_candidates
        return result


def increment(session, root, *, name, spec):
    return bounded_training.increment(session,root,name=name,spec=spec,data_factory=ResponseData,
        update=response_update.step,review=ResponseLearningLoop.consider_response_learning,
        eligible=lambda data,_model:data.response_basis.names)
