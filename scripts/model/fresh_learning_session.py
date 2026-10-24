"""Persist a learning decision from fresh, completed native feedback."""
from claims.codata_source import check
from model.fresh_echo import FreshEchoLoop
from model.fresh_learning import FreshData, code_pins
from model import candidate_review

KIND = 'fresh_learning_candidate'


class FreshLearningLoop(FreshEchoLoop):
    def __init__(self, *args, increment_base, **kwargs):
        self.increment_base = increment_base
        self.fresh_candidates = {}
        super().__init__(*args,**kwargs)

    def consider_fresh_learning(self, name, pin, spec, expected=None):
        return candidate_review.review(self,name,pin,spec,factory=FreshData,code=code_pins(),kind=KIND,
            registry=self.fresh_candidates,expected=expected)

    def _replay(self, body):
        if body.get('kind') == KIND:
            check(body['code'] == code_pins(), 'FRESH_LEARNING_REPLAY_CODE')
            self.consider_fresh_learning(body['candidate'],body['pin'],body['spec'],expected=body)
        else:
            super()._replay(body)

    def snapshot(self):
        result = super().snapshot()
        if self.fresh_candidates:
            result['fresh_learning_candidates'] = self.fresh_candidates
        return result
