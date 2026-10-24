"""Authenticated per-instrument correction and durable, fail-closed candidate review."""
import copy
from pathlib import Path

from claims.codata_source import check, sha
from claims.episodic_memory import digest
from model.constellation_learning import decision
from model.region_correction_session import RegionData, RegionLoop
from model.tool_rehearsal import termination_gate
from model import rehearsal_retention, stable_checkpoint, tool_retention, typed_correction_targets

KIND = 'typed_region_correction_candidate'


def code_pins():
    return {n:sha((Path(__file__).parent/n).read_bytes()) for n in
            ('typed_correction_session.py','typed_correction.py','typed_correction_targets.py')}


class TypedData(RegionData):
    def __init__(self, session, base, spec):
        super().__init__(session,base,{k:v for k,v in spec.items() if k != 'typed'})
        bound = typed_correction_targets.bind(self,spec['typed'])
        self.spec = copy.deepcopy(spec)
        self.binding['typed_correction'] = dict(bound,code=code_pins())
        self.binding_pin = digest(self.binding)
        self.fresh()


class TypedLoop(RegionLoop):
    def __init__(self,*args,**kwargs):
        self.typed_candidates = {}
        super().__init__(*args,**kwargs)

    def consider_typed(self,name,pin,spec,expected=None):
        self._fresh()
        check(not self.typed_candidates,'TYPED_CORRECTION_DUPLICATE')
        check(isinstance(name,str) and name not in ('','.','..') and '/' not in name and '\\' not in name,
              'TYPED_CORRECTION_PATH')
        root = self.controlled_root/name
        check(root.resolve().parent == self.controlled_root.resolve(),'TYPED_CORRECTION_PATH')
        data = TypedData(self,self.learning_base,spec)
        candidate,_,manifest = stable_checkpoint.load(root,data,expected=pin)
        check(manifest['step'] == data.plan['steps'] and manifest['parent'] == data.parent_pin,
              'TYPED_CORRECTION_INCOMPLETE')
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
            check(body == expected and payload == self.journal.payload(expected['payload']),
                  'TYPED_CORRECTION_REPLAY')
        self.typed_candidates[name] = body
        return body

    def _replay(self,body):
        if body.get('kind') == KIND:
            check(body['code'] == code_pins(),'TYPED_CORRECTION_REPLAY_CODE')
            self.consider_typed(body['candidate'],body['pin'],body['spec'],expected=body)
        else:
            super()._replay(body)

    def snapshot(self):
        result = super().snapshot()
        if self.typed_candidates:
            result['typed_correction_candidates'] = self.typed_candidates
        return result
