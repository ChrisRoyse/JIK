"""Journal candidate decisions for learning from captured proposal feedback."""
from pathlib import Path

from claims.codata_source import check
from claims.episodic_memory import digest
from model.constellation_learning import decision
from model.proposal_echo import ProposalEchoLoop
from model.proposal_learning import ProposalData, code_pins
from model import field_checkpoint, rehearsal_retention, tool_retention
from model.tool_rehearsal import termination_gate


class ProposalLearningLoop(ProposalEchoLoop):
    def __init__(self, *args, learning_base, **kwargs):
        self.learning_base, self.proposal_candidates = learning_base,{}
        super().__init__(*args,**kwargs)

    def consider_proposal_learning(self, name, pin, spec, expected=None):
        self._fresh()
        check(not self.proposal_candidates,'PROPOSAL_LEARNING_DUPLICATE_DECISION')
        check(isinstance(name,str) and name not in ('','.','..') and '/' not in name and '\\' not in name,
              'PROPOSAL_LEARNING_CANDIDATE_PATH')
        root = self.controlled_root/name
        check(root.resolve().parent == self.controlled_root.resolve(),'PROPOSAL_LEARNING_CANDIDATE_PATH')
        data = ProposalData(self,self.learning_base,spec)
        candidate,_,manifest = field_checkpoint.load(root,data,expected=pin)
        check(manifest['step'] == data.plan['steps'] and manifest['parent'] == data.parent_pin,
              'PROPOSAL_LEARNING_INCOMPLETE')
        parent,_,_ = field_checkpoint.start(data)
        before,after = data.evaluate(parent),data.evaluate(candidate)
        result = decision(data,before,after,candidate)
        result = tool_retention.assess(data,result,after)
        result = termination_gate(data,before,after,result)
        result = rehearsal_retention.assess(data,after,result)
        new_changes = {k:{kind:{n:dict(before=v,after=rehearsal_retention.measurements(after[k])[kind][n])
            for n,v in row.items()} for kind,row in rehearsal_retention.measurements(before[k]).items()}
            for k in data.new}
        payload = dict(before=before,after=after,result=result,new_changes=new_changes,
            binding=digest(data.binding),weight_learning_comparison='same_expanded_reference_bank',
            source_prediction_unchanged=True,serving=False,scientific_admission=False)
        body = dict(kind='proposal_learning_candidate',candidate=name,pin=pin,parent=data.parent_pin,
            spec=spec,prior_head=self.head,payload=digest(payload),code=code_pins(),result=result)
        if expected is None:
            self.journal.append(body,payload)
        else:
            check(body == expected and payload == self.journal.payload(expected['payload']),
                  'PROPOSAL_LEARNING_DECISION_REPLAY')
        self.proposal_candidates[name] = body
        return body

    def _replay(self, body):
        if body.get('kind') == 'proposal_learning_candidate':
            check(body['code'] == code_pins(),'PROPOSAL_LEARNING_CODE')
            self.consider_proposal_learning(body['candidate'],body['pin'],body['spec'],expected=body)
        else:
            super()._replay(body)

    def snapshot(self):
        result = super().snapshot()
        if self.proposal_candidates:
            result['proposal_learning_candidates'] = self.proposal_candidates
        return result
