"""Bind the measured numerical repair to a withheld typed learning candidate."""
import copy
import json
from pathlib import Path

from claims.codata_source import check, sha
from claims.episodic_memory import digest
from model.constellation_learning import decision
from model.typed_correction_session import TypedData, TypedLoop
from model.tool_rehearsal import termination_gate
from model import precise_region_solve, rehearsal_retention, stable_checkpoint, tool_retention

KIND = 'precise_typed_region_candidate'


def code_pins():
    return {n:sha((Path(__file__).parent/n).read_bytes()) for n in
            ('precise_correction_session.py','precise_correction.py','precise_region_solve.py')}


class PreciseData(TypedData):
    def __init__(self, session, base, spec):
        super().__init__(session,base,{k:v for k,v in spec.items() if k != 'precise'})
        root = Path(spec['precise']['root'])
        report_raw,reader_raw = (root/'report.json').read_bytes(),(root/'readback.json').read_bytes()
        check(sha(report_raw) == spec['precise']['pin'] and sha(reader_raw) == spec['precise']['readback_pin'],
              'PRECISE_CORRECTION_DIAGNOSTIC_PIN')
        report,reader = json.loads(report_raw),json.loads(reader_raw)
        raw = (Path(report['source'])/'learning.json').read_bytes()
        check(dict(bytes=len(raw),sha256=sha(raw)) == report['source_learning'],
              'PRECISE_CORRECTION_SOURCE_PIN')
        source = json.loads(raw)
        check(source['binding'] == self.binding and source['parent'] == self.parent_pin
              and reader['report']['sha256'] == sha(report_raw)
              and set(reader['checked']) == set(self.typed_rows)
              and all(v['methods']['extended_gram']['actual']['passed'] for v in report['results'].values()),
              'PRECISE_CORRECTION_DIAGNOSTIC_BINDING')
        self.spec = copy.deepcopy(spec)
        self.binding['precise_correction'] = dict(rule=precise_region_solve.RULE,code=code_pins(),
            diagnostic=spec['precise'],original_tolerances_unchanged=True,
            original_iteration_cap_unchanged=True,actual_retention_unchanged=True)
        self.binding_pin = digest(self.binding)
        self.fresh()


class PreciseLoop(TypedLoop):
    def __init__(self,*args,**kwargs):
        self.precise_candidates = {}
        super().__init__(*args,**kwargs)

    def consider_precise(self,name,pin,spec,expected=None):
        self._fresh()
        check(not self.precise_candidates,'PRECISE_CORRECTION_DUPLICATE')
        check(isinstance(name,str) and name not in ('','.','..') and '/' not in name and '\\' not in name,
              'PRECISE_CORRECTION_PATH')
        root = self.controlled_root/name
        check(root.resolve().parent == self.controlled_root.resolve(),'PRECISE_CORRECTION_PATH')
        data = PreciseData(self,self.learning_base,spec)
        candidate,_,manifest = stable_checkpoint.load(root,data,expected=pin)
        check(manifest['step'] == data.plan['steps'] and manifest['parent'] == data.parent_pin,
              'PRECISE_CORRECTION_INCOMPLETE')
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
                  'PRECISE_CORRECTION_REPLAY')
        self.precise_candidates[name] = body
        return body

    def _replay(self,body):
        if body.get('kind') == KIND:
            check(body['code'] == code_pins(),'PRECISE_CORRECTION_REPLAY_CODE')
            self.consider_precise(body['candidate'],body['pin'],body['spec'],expected=body)
        else:
            super()._replay(body)

    def snapshot(self):
        result = super().snapshot()
        if self.precise_candidates:
            result['precise_correction_candidates'] = self.precise_candidates
        return result
