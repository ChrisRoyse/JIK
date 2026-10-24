"""Versioned fresh-feedback learning with an authenticated separated computation."""
import copy
import json
from pathlib import Path

from claims.codata_source import check, sha
from claims.episodic_memory import digest
from model.fresh_learning import FreshData
from model.fresh_learning_session import FreshLearningLoop
from model.constellation_learning import decision
from model.tool_rehearsal import termination_gate
from model import rehearsal_retention, stable_checkpoint, stable_response, tool_retention

KIND = 'stable_response_candidate'


def code_pins():
    return {n:sha((Path(__file__).parent/n).read_bytes()) for n in
            ('stable_learning.py','stable_response.py','stable_checkpoint.py','stable_training.py','response_update.py')}


class StableData(FreshData):
    def __init__(self, session, base, spec):
        super().__init__(session,base,{k:v for k,v in spec.items() if k != 'stable'})
        path = Path(spec['stable']['root'])/'report.json'
        raw = path.read_bytes()
        check(sha(raw) == spec['stable']['pin'],'STABLE_LEARNING_DIAGNOSTIC_PIN')
        report = json.loads(raw)
        numeric_raw = (path.parent/'numeric-readback.json').read_bytes()
        check(sha(numeric_raw) == spec['stable']['readback_pin'],'STABLE_LEARNING_NUMERIC_PIN')
        numeric = json.loads(numeric_raw)
        check(numeric['report'] == spec['stable']['pin'] and
              all(v['retained'] for v in numeric['comparisons']['separated']), 'STABLE_LEARNING_NUMERIC_BINDING')
        check(report['parent'] == spec['checkpoint'] and report['basis'] == spec['basis']
              and report['binding'] == digest(self.binding) and report['initial_behaviour_equal']
              and all(v['retained'] for v in report['separated_checks'])
              and report['code']['model/stable_response.py'] == stable_response.code_pin(),
              'STABLE_LEARNING_DIAGNOSTIC_BINDING')
        self.spec = copy.deepcopy(spec)
        self.binding['stable_response'] = dict(rule=stable_response.RULE,code=code_pins(),
            diagnostic=spec['stable'],static_buffers=report['static_buffers'],
            parent=spec['checkpoint'],basis=spec['basis'],scope='exposed_feedback_not_transfer')
        self.binding_pin = digest(self.binding)
        self.fresh()


class StableLoop(FreshLearningLoop):
    def __init__(self, *args, **kwargs):
        self.stable_candidates = {}
        super().__init__(*args,**kwargs)

    def consider_stable(self, name, pin, spec, expected=None):
        self._fresh()
        check(not self.stable_candidates,'STABLE_LEARNING_DUPLICATE')
        check(isinstance(name,str) and name not in ('','.','..') and '/' not in name and '\\' not in name,
              'STABLE_LEARNING_PATH')
        root = self.controlled_root/name
        check(root.resolve().parent == self.controlled_root.resolve(),'STABLE_LEARNING_PATH')
        data = StableData(self,self.learning_base,spec)
        candidate,_,manifest = stable_checkpoint.load(root,data,expected=pin)
        check(manifest['step'] == data.plan['steps'] and manifest['parent'] == data.parent_pin,
              'STABLE_LEARNING_INCOMPLETE')
        parent,_,_ = stable_checkpoint.start(data)
        before,after = data.evaluate(parent),data.evaluate(candidate)
        _,drift = data.response_basis.measure(data,candidate)
        result = decision(data,before,after,candidate)
        result = tool_retention.assess(data,result,after)
        result = termination_gate(data,before,after,result)
        result = rehearsal_retention.assess(data,after,result)
        payload = dict(before=before,after=after,result=result,binding=digest(data.binding),
            response_drift=drift,source_prediction_unchanged=True,
            weight_learning_comparison='same_expanded_reference_bank',serving=False,scientific_admission=False)
        body = dict(kind=KIND,candidate=name,pin=pin,parent=data.parent_pin,spec=spec,
            prior_head=self.head,payload=digest(payload),code=code_pins(),result=result)
        if expected is None:
            self.journal.append(body,payload)
        else:
            check(body == expected and payload == self.journal.payload(expected['payload']),'STABLE_LEARNING_REPLAY')
        self.stable_candidates[name] = body
        return body

    def _replay(self, body):
        if body.get('kind') == KIND:
            check(body['code'] == code_pins(),'STABLE_LEARNING_REPLAY_CODE')
            self.consider_stable(body['candidate'],body['pin'],body['spec'],expected=body)
        else:
            super()._replay(body)

    def snapshot(self):
        result = super().snapshot()
        if self.stable_candidates:
            result['stable_response_candidates'] = self.stable_candidates
        return result
