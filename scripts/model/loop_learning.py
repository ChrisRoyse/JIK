"""Checked-memory training inputs and finite development gates for intrinsic loops.

The profile's response relation is supervision, not a universal conduct oracle.
Only echo readings enter the predictor; response/unchosen readings are targets.
"""
import hashlib
import math

import torch
from torch.nn import functional as F

from senses.contract import digest, require


def check(ok,code):require(ok,'MNLM_LOOP_LEARNING_'+code,'restore the checked input version or retained training state')


class LearningInputs:
    def __init__(self, session, model, plan):
        self.session,self.model,self.plan=session,model,plan
        self.head=session.memory.head;self.contract=session.contract_id
        check(model.source_id==session.model.source_id and model.standard_id==session.model.standard_id,'STANDARD')
        ids=sorted((r['id'] for r in session.checker.records.values() if r['type']=='correction'),
                   key=lambda x:hashlib.sha256(('loop-learning-v1:'+x).encode()).hexdigest())
        expected={c:role for role,group in [('old_train',ids[:6]),('new_train',ids[6:12]),('retention',ids[12:17]),('acquisition',ids[17:])]
                  for c in group}
        check(len(ids)==22 and plan['roles']==expected and plan['base_steps']==plan['extension_steps']==64
              and plan['learning_rate']==.001 and plan['weight_decay']==.01 and plan['production_admission'] is False,'PLAN')
        self.records={};self.receipts={};self.cache={}
        for c in ids:
            units=[session.inputs.lookup[(c,f)] for f in ('echo','response','not_chosen')]
            check(all(session.inputs.refs[u]['record']==c and session.inputs.refs[u]['field']==f
                      for u,f in zip(units,('echo','response','not_chosen'))),'SOURCE_FIELD_JOIN')
            for u in units:
                ep=session.checker.episode(u);key=ep['id'];state=session.memory.state.get(key)
                check(state is not None and state['active'] and state['review']=='source_checked'
                      and state['grounded_scope']=='pinned_artifact_contents_only'
                      and state['receipt']==session.checker.assess(ep),'UNCHECKED_TARGET')
                self.receipts[str(u)]=state['receipt']['id']
            self.records[c]=units
        self.binding=digest(dict(source_id=model.source_id,standard=model.standard_id,contract=self.contract,
                                 memory_head=self.head,plan=plan,receipts=self.receipts))
        for role in ('old_train','retention','acquisition','continuation'):
            names=[c for c in ids if plan['roles'][c] in (('old_train','new_train') if role=='continuation' else (role,))]
            native=[session.inputs.batch([self.records[c][j] for c in names],model.config,session.device) for j in range(3)]
            codes={}
            for i,spec in enumerate(model.specs):
                book=getattr(model,f'ref_values_{i}')[~getattr(model,f'ref_padding_{i}')]
                x=native[1][spec.name]
                scores=(F.normalize(x.values,dim=-1)@book.T).masked_fill(x.padding[:,:,None],-torch.inf)
                codes[spec.name]=scores.amax(1).argmax(1).detach()
            self.cache[role]=dict(names=names,native=native,codes=codes)

    def fresh(self, *, disk=False):
        self.session._live()
        check(self.session.memory.head==self.head,'SOURCE_HEAD_CHANGED')
        if disk:
            row=self.session.db.execute('SELECT hash FROM events ORDER BY seq DESC LIMIT 1').fetchone()
            check(row is not None and row[0]==self.session.head,'SOURCE_HEAD_CHANGED')

    def predict(self, model, role):
        self.fresh();b=self.cache[role]
        states=tuple(model.start('profile:'+c) for c in b['names'])
        for phase in (0,1):
            events=[digest(dict(source_head=self.head,record=c,phase=phase,purpose='profile_response_supervision')) for c in b['names']]
            out=model(b['native'][0],states,phase=phase,event_ids=events,evidence_origin=['source_observation']*len(states))
            states=out['states']
        return out

    def losses(self, model, role):
        out=self.predict(model,role);b=self.cache[role];terms=[];margins=[]
        for spec in model.specs:
            name=spec.name;slot=out['slots'][name]
            scores=[]
            for native in b['native'][1:]:
                x=native[name]
                scores.append(torch.einsum('bd,bwd->bw',slot,F.normalize(x.values,dim=-1)).masked_fill(x.padding,-torch.inf).amax(1))
            positive,negative=scores;ce=F.cross_entropy(out['regions'][name],b['codes'][name],reduction='none')
            terms.append(ce+1-positive+F.softplus(negative-positive));margins.append(positive-negative)
        return torch.stack(terms,1),torch.stack(margins,1)

    def step(self, model, optimizer, role):
        check(role in ('old_train','continuation'),'EVALUATION_GRADIENT')
        self.fresh();model.train();optimizer.zero_grad(set_to_none=True)
        loss,_=self.losses(model,role);value=loss.mean()
        check(bool(torch.isfinite(value)),'NONFINITE')
        value.backward()
        check(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()),'NONFINITE_GRADIENT')
        optimizer.step()
        return float(value.detach())

    @torch.no_grad()
    def evaluate(self, model):
        self.fresh();model.eval();result=[]
        for role in ('retention','acquisition'):
            loss,margin=self.losses(model,role)
            for i,c in enumerate(self.cache[role]['names']):
                result.append(dict(record=c,role=role,loss=loss[i].cpu().tolist(),margin=margin[i].cpu().tolist()))
        return result


def decide(plan, baseline, candidate, instrument_count):
    expected={c:r for c,r in plan['roles'].items() if r in ('retention','acquisition')}
    maps=[]
    for rows in (baseline,candidate):
        check(len(rows)==len(expected) and {r['record'] for r in rows}==set(expected),'EVAL_COVERAGE')
        check(all(r['role']==expected[r['record']] and set(r)=={'record','role','loss','margin'}
                  and all(len(r[k])==instrument_count and all(math.isfinite(v) for v in r[k]) for k in ('loss','margin'))
                  for r in rows),'EVAL_VALUES')
        maps.append({r['record']:r for r in rows})
    changes=[];regressions=[]
    for c,role in expected.items():
        a,b=maps[0][c],maps[1][c]
        deltas=[y-x for x,y in zip(a['loss'],b['loss'])]
        lost=[k for k,(x,y) in enumerate(zip(a['margin'],b['margin'])) if x>0 and y<=0]
        if (role=='retention' and any(d>0 for d in deltas)) or lost:regressions.append(c)
        changes.append(dict(record=c,role=role,loss_delta=deltas,lost_correct_instruments=lost))
    gains=[d for row in changes if row['role']=='acquisition' for d in row['loss_delta']]
    gain=math.fsum(gains)/len(gains)
    reasons=(['RETENTION_OR_RANKING_REGRESSION'] if regressions else [])+(['NO_ACQUISITION_GAIN'] if gain>=0 else [])
    return dict(eligible=not reasons,reasons=reasons,regressions=regressions,acquisition_mean_loss_delta=gain,
                changes=changes,scope='exposed_development_profile_relationship_only',production_admission=False,
                behavioural_compliance_established=False,world_truth_established=False)
