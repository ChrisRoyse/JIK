"""Bind a region correction to its diagnostic, checkpoint and durable withheld decision."""
import copy
import json
from pathlib import Path

from claims.codata_source import check, sha
from claims.episodic_memory import digest
from model.constellation_learning import decision
from model.stable_learning import StableData, StableLoop
from model.tool_rehearsal import termination_gate
from model import rehearsal_retention, region_correction, stable_checkpoint, tool_retention

KIND = 'native_region_correction_candidate'


def code_pins():
    return {n:sha((Path(__file__).parent/n).read_bytes()) for n in
            ('region_correction_session.py','region_correction.py','region_correction_solve.py')}


class RegionData(StableData):
    def __init__(self, session, base, spec):
        super().__init__(session,base,{k:v for k,v in spec.items() if k != 'correction'})
        root = Path(spec['correction']['root'])
        raw,numeric = (root/'report.json').read_bytes(),(root/'readback.json').read_bytes()
        check(sha(raw) == spec['correction']['pin'] and sha(numeric) == spec['correction']['readback_pin'],
              'REGION_CORRECTION_DIAGNOSTIC_PIN')
        report,reader = json.loads(raw),json.loads(numeric)
        check(report['binding'] == digest(self.binding) and reader['report'] == sha(raw)
              and set(report['examples']) == set(self.new),'REGION_CORRECTION_DIAGNOSTIC_BINDING')
        self.spec = copy.deepcopy(spec)
        self.binding['region_correction'] = dict(rule=region_correction.RULE,code=code_pins(),
            diagnostic=spec['correction'],scope='exposed_feedback_not_transfer',
            acceptance='strict_prior_retention_and_nonincreasing_new_native_errors',
            objective='sum_positive_native_cost_margin_violations',forward_unchanged=True)
        self.binding_pin = digest(self.binding)
        self.fresh()


class RegionLoop(StableLoop):
    def __init__(self,*args,**kwargs):
        self.region_candidates = {}
        super().__init__(*args,**kwargs)

    def consider_region(self,name,pin,spec,expected=None):
        self._fresh()
        check(not self.region_candidates,'REGION_CORRECTION_DUPLICATE')
        check(isinstance(name,str) and name not in ('','.','..') and '/' not in name and '\\' not in name,
              'REGION_CORRECTION_PATH')
        root = self.controlled_root/name
        check(root.resolve().parent == self.controlled_root.resolve(),'REGION_CORRECTION_PATH')
        data = RegionData(self,self.learning_base,spec)
        candidate,_,manifest = stable_checkpoint.load(root,data,expected=pin)
        check(manifest['step'] == data.plan['steps'] and manifest['parent'] == data.parent_pin,
              'REGION_CORRECTION_INCOMPLETE')
        parent,_,_ = stable_checkpoint.start(data)
        before,after = data.evaluate(parent),data.evaluate(candidate)
        _,drift = data.response_basis.measure(data,candidate)
        result = decision(data,before,after,candidate)
        result = tool_retention.assess(data,result,after)
        result = termination_gate(data,before,after,result)
        result = rehearsal_retention.assess(data,after,result)
        payload = dict(before=before,after=after,result=result,binding=digest(data.binding),response_drift=drift,
            source_prediction_unchanged=True,weight_learning_comparison='same_expanded_reference_bank',
            serving=False,scientific_admission=False)
        body = dict(kind=KIND,candidate=name,pin=pin,parent=data.parent_pin,spec=spec,prior_head=self.head,
            payload=digest(payload),code=code_pins(),result=result)
        if expected is None:
            self.journal.append(body,payload)
        else:
            check(body == expected and payload == self.journal.payload(expected['payload']),'REGION_CORRECTION_REPLAY')
        self.region_candidates[name] = body
        return body

    def _replay(self,body):
        if body.get('kind') == KIND:
            check(body['code'] == code_pins(),'REGION_CORRECTION_REPLAY_CODE')
            self.consider_region(body['candidate'],body['pin'],body['spec'],expected=body)
        else:
            super()._replay(body)

    def snapshot(self):
        result = super().snapshot()
        if self.region_candidates:
            result['region_correction_candidates'] = self.region_candidates
        return result
