"""Bind the measured numerical repair to a withheld typed learning candidate."""
import copy
import json
from pathlib import Path

from claims.codata_source import check, sha
from claims.episodic_memory import digest
from model.constellation_learning import decision
from model.precise_correction_session import PreciseData, PreciseLoop
from model.tool_rehearsal import termination_gate
from model import brier_region_targets, rehearsal_retention, stable_checkpoint, tool_retention

KIND = 'brier_region_candidate'


def code_pins():
    return {n:sha((Path(__file__).parent/n).read_bytes()) for n in
            ('brier_region_session_v2.py','brier_region_correction_v2.py','brier_region_targets.py')}


class BrierData(PreciseData):
    def __init__(self, session, base, spec):
        super().__init__(session,base,{k:v for k,v in spec.items() if k != 'brier'})
        bound = brier_region_targets.bind(self,spec['brier'])
        self.spec = copy.deepcopy(spec)
        self.binding['brier_region'] = dict(bound,code=code_pins(),
            direction_cache='one_entry_full_model_state_binding_and_rows_eval_mode')
        self.binding_pin = digest(self.binding)
        self.direction_cache = None
        self.cache_trace = []
        self.fresh()

class BrierLoop(PreciseLoop):
    def __init__(self,*args,**kwargs):
        self.brier_candidates = {}
        super().__init__(*args,**kwargs)

    def consider_brier(self,name,pin,spec,expected=None):
        self._fresh()
        check(not self.brier_candidates,'BRIER_REGION_DUPLICATE')
        check(isinstance(name,str) and name not in ('','.','..') and '/' not in name and '\\' not in name,
              'BRIER_REGION_PATH')
        root = self.controlled_root/name
        check(root.resolve().parent == self.controlled_root.resolve(),'BRIER_REGION_PATH')
        data = BrierData(self,self.learning_base,spec)
        candidate,_,manifest = stable_checkpoint.load(root,data,expected=pin)
        check(manifest['step'] == data.plan['steps'] and manifest['parent'] == data.parent_pin,
              'BRIER_REGION_INCOMPLETE')
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
                  'BRIER_REGION_REPLAY')
        self.brier_candidates[name] = body
        return body

    def _replay(self,body):
        if body.get('kind') == KIND:
            check(body['code'] == code_pins(),'BRIER_REGION_REPLAY_CODE')
            self.consider_brier(body['candidate'],body['pin'],body['spec'],expected=body)
        else:
            super()._replay(body)

    def snapshot(self):
        result = super().snapshot()
        if self.brier_candidates:
            result['brier_region_candidates'] = self.brier_candidates
        return result
