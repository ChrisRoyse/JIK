"""Explicit checkpoint lineage for further bounded fits to immutable feedback."""
import copy
import json
from pathlib import Path

from claims.codata_source import check, sha
from claims.episodic_memory import digest
from model.response_learning import ResponseData, ResponseLearningLoop
from model import candidate_review

RULE = 'retained_response_continuation_v1'
KIND = 'response_cycle_candidate'


def code_pins():
    return {n:sha((Path(__file__).parent/n).read_bytes()) for n in
            ('response_cycles.py','candidate_review.py','bounded_training.py','response_update.py')}


class CycleData(ResponseData):
    def __init__(self, session, base, spec):
        super().__init__(session,base,spec['initial'])
        source_binding = digest(self.binding)
        check(isinstance(spec['continuations'],list) and spec['continuations'], 'RESPONSE_CYCLE_LINEAGE')
        self.cycle_code = code_pins()
        seen = set()
        for index,link in enumerate(spec['continuations']):
            root = Path(link['checkpoint']['root'])
            pin = link['checkpoint']['pin']
            check(pin not in seen, 'RESPONSE_CYCLE_DUPLICATE_PARENT')
            seen.add(pin)
            raw = (root/'manifest.json').read_bytes()
            check(sha(raw) == pin, 'RESPONSE_CYCLE_PARENT_PIN')
            manifest = json.loads(raw)
            check(manifest['binding'] == self.binding and manifest['step'] == self.plan['steps'], 'RESPONSE_CYCLE_PARENT_BINDING')
            raw = Path(link['evidence']['path']).read_bytes()
            check(sha(raw) == link['evidence']['pin'], 'RESPONSE_CYCLE_REPORT_PIN')
            report = json.loads(raw)
            check(report['binding'] == self.binding and report['candidate_pin'] == pin
                  and report['decision']['pin'] == pin, 'RESPONSE_CYCLE_PARENT_DECISION')
            self.field_predecessor, self.parent_pin = (root,pin,manifest),pin
            self.binding['native_world']['offsets'] = {n:v+manifest['step'] for n,v in manifest['offsets'].items()}
            self.binding['response_cycles'] = dict(rule=RULE,code=self.cycle_code,initial_binding=source_binding,
                continuations=copy.deepcopy(spec['continuations'][:index+1]),
                parent_decision=digest(report['decision']),scope='repeated_exposed_feedback_not_new_experience')
        self.spec = copy.deepcopy(spec)
        self.binding_pin = digest(self.binding)
        self.fresh()

    def fresh(self):
        super().fresh()
        if 'response_cycles' in self.binding:
            saved = self.binding['response_cycles']
            check(saved['rule'] == RULE and saved['code'] == self.cycle_code, 'RESPONSE_CYCLE_CODE')


class CycleLoop(ResponseLearningLoop):
    def __init__(self, *args, **kwargs):
        self.cycle_candidates = {}
        super().__init__(*args,**kwargs)

    def consider_cycle(self, name, pin, spec, expected=None):
        return candidate_review.review(self,name,pin,spec,factory=CycleData,code=code_pins(),kind=KIND,
            registry=self.cycle_candidates,expected=expected)

    def _replay(self, body):
        if body.get('kind') == KIND:
            check(body['code'] == code_pins(), 'RESPONSE_CYCLE_REPLAY_CODE')
            self.consider_cycle(body['candidate'],body['pin'],body['spec'],expected=body)
        else:
            super()._replay(body)

    def snapshot(self):
        result = super().snapshot()
        if self.cycle_candidates:
            result['response_cycle_candidates'] = self.cycle_candidates
        return result
