"""Ledger a retained successor without promoting a development candidate."""
from pathlib import Path

from claims.codata_source import check,sha
from claims.episodic_memory import digest
from model.constellation_learning import decision
from model.next_echo import NextEchoLoop
from model.successor_learning import SuccessorData
from model.tool_rehearsal import termination_gate
from model import successor_checkpoint, rehearsal_retention, tool_retention


def code_pins():
    return {n:sha((Path(__file__).parent/n).read_bytes()) for n in (
        'successor_session.py','successor_learning.py','successor_targets.py','successor_checkpoint.py',
        'successor_training.py','brier_region_correction_v3.py','coordinate_region_solve.py')}


class SuccessorLoop(NextEchoLoop):
    def __init__(self,*args,**kwargs):
        self.successor_candidates = {}
        super().__init__(*args,**kwargs)

    def consider_successor(self,name,pin,spec,expected=None):
        self._fresh()
        check(not self.successor_candidates,'SUCCESSOR_DUPLICATE')
        check(isinstance(name,str) and name not in ('','.','..') and '/' not in name and '\\' not in name,
              'SUCCESSOR_PATH')
        root = self.controlled_root/name
        check(root.resolve().parent == self.controlled_root.resolve(),'SUCCESSOR_PATH')
        data = SuccessorData(self,self.transfer_data,spec)
        candidate,_,manifest = successor_checkpoint.load(root,data,expected=pin)
        check(manifest['step'] == data.plan['steps'] and manifest['parent'] == data.parent_pin,'SUCCESSOR_INCOMPLETE')
        parent,_,_ = successor_checkpoint.start(data)
        before,after = data.evaluate(parent),data.evaluate(candidate)
        _,drift = data.response_basis.measure(data,candidate)
        result = decision(data,before,after,candidate)
        result = tool_retention.assess(data,result,after)
        result = termination_gate(data,before,after,result)
        result = rehearsal_retention.assess(data,after,result)
        payload = dict(before=before,after=after,result=result,binding=digest(data.binding),response_drift=drift,
            source_prediction_unchanged=True,weight_learning_comparison='same_expanded_reference_bank',
            serving=False,scientific_admission=False)
        body = dict(kind='successor_candidate',candidate=name,pin=pin,parent=data.parent_pin,spec=spec,
            prior_head=self.head,payload=digest(payload),code=code_pins(),result=result)
        if expected is None:
            self.journal.append(body,payload)
        else:
            check(body == expected and payload == self.journal.payload(expected['payload']),'SUCCESSOR_REPLAY')
        self.successor_candidates[name] = body
        return body

    def _replay(self,body):
        if body.get('kind') == 'successor_candidate':
            check(body['code'] == code_pins(),'SUCCESSOR_REPLAY_CODE')
            self.consider_successor(body['candidate'],body['pin'],body['spec'],expected=body)
        else:
            super()._replay(body)

    def snapshot(self):
        value = super().snapshot()
        if self.successor_candidates:
            value['successor_candidates'] = self.successor_candidates
        return value
