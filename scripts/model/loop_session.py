"""Durable evidence-bound intrinsic loops. No serving or world-truth authority.

The immutable model is stored once. SQLite appends only changed context state
and evidence events. A retained external head detects truncation/rollback.
Computation precedes the short optimistic transaction; ambiguous writes poison
the handle. An explicit restart/review is required, never fallback success.
"""
import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np
import torch

from claims.episodic_memory import Memory, ZERO
from claims.identity_source import IdentitySourceChecker
from model.identity_loop import ContextState
from model.loop_checkpoint import load as load_model
from senses.contract import canonical, digest, require


def check(ok, code):
    require(ok, 'MNLM_LOOP_SESSION_'+code, 'restore the retained version or correct the named evidence/context')


def sha(raw):return hashlib.sha256(raw).hexdigest()


def output_digest(out):
    values={}
    for kind in ('regions','slots','predicted_echo','predicted_verdict'):
        for name,t in out[kind].items():
            a=t.detach().cpu().contiguous().numpy()
            values[kind+':'+name]=dict(shape=list(a.shape),dtype=a.dtype.str,sha256=sha(a.tobytes()))
    return digest(values)


def code_pins():
    root=Path(__file__).resolve().parents[1]
    return {n:sha((root/n).read_bytes()) for n in (
        'model/loop_session.py','claims/identity_source.py','claims/episodic_memory.py','model/loop_input.py','model/loop_learning.py')}


class LoopSession:
    def __init__(self, path, base, inputs, source, *, checkpoint, standard, device,
                 create=False, expected_contract=None, expected_head=None, neural_replay=False):
        self.path=Path(path);self.inputs=inputs;self.device=device;self.broken=False
        check(create or (expected_contract is not None and expected_head is not None), 'RETAINED_PINS')
        check(not self.path.exists() if create else self.path.is_file(), 'PATH')
        self.model,states,_=load_model(base,expected_checkpoint=checkpoint,expected_standard=standard,device=device)
        check(self.model.source_id==inputs.source_id and tuple(self.model.specs)==inputs.specs, 'INPUT_JOIN')
        check(all(s.sequence==0 for s in states), 'BASE_NOT_INITIAL')
        self.model.eval()
        self.states={s.context:s for s in states};self.dirty=set()
        self.support={s.context:set() for s in states};self.consumers={}
        self.checker=IdentitySourceChecker(inputs,source);self.memory=Memory(self.checker)
        self.head=ZERO;self.count=0;self.neural_replayed=0
        self.selected_learning=None
        contract=dict(schema='evidence-bound-loop-session-v1',checkpoint=checkpoint,standard=standard,
                      source_id=inputs.source_id,contexts=sorted(self.states),code_sha256=code_pins(),
                      serving=False,world_truth_established=False,behavioural_compliance_established=False,
                      historical_reconstruction=False,optimizer_resume_supported=False)
        self.contract_id=digest(contract)
        if create:
            self.path.parent.mkdir(parents=True,exist_ok=True)
            with self.path.open('xb'):pass
        self.db=sqlite3.connect(str(self.path),isolation_level=None)
        self.db.execute('PRAGMA synchronous=FULL')
        try:
            if create:
                self.db.execute('BEGIN IMMEDIATE')
                self.db.execute('CREATE TABLE metadata (key TEXT PRIMARY KEY, value BLOB NOT NULL)')
                self.db.execute('CREATE TABLE events (seq INTEGER PRIMARY KEY, previous TEXT NOT NULL, body BLOB NOT NULL, hidden BLOB, hash TEXT UNIQUE NOT NULL)')
                self.db.execute('INSERT INTO metadata VALUES (?,?)',('contract',canonical(contract)))
                self.db.execute('COMMIT')
            else:
                self.db.execute('BEGIN')
                actual=self.db.execute('SELECT value FROM metadata WHERE key=?',('contract',)).fetchone()
                check(actual is not None and actual[0]==canonical(contract) and self.contract_id==expected_contract, 'CONTRACT')
                for seq,previous,raw,hidden,pin in self.db.execute('SELECT seq,previous,body,hidden,hash FROM events ORDER BY seq'):
                    check(len(raw)+(len(hidden) if hidden else 0)<=self.model.config.checkpoint_bytes, 'EVENT_BUDGET')
                    body=json.loads(raw)
                    check(raw==canonical(body) and seq==self.count and previous==self.head, 'CHAIN_ORDER')
                    check(pin==digest(dict(seq=seq,previous=previous,body=body)), 'CHAIN_HASH')
                    self._replay(body,hidden,neural_replay)
                    self.head=pin;self.count+=1
                check(self.head==expected_head, 'HEAD_PIN')
                self.db.execute('COMMIT')
        except BaseException:
            self.db.close();self.broken=True;raise

    def close(self):self.db.close();self.broken=True

    def _live(self):check(not self.broken, 'RELOAD_REQUIRED')

    def _append(self, body, hidden=None):
        self._live()
        raw=canonical(body)
        check(len(raw)+(len(hidden) if hidden else 0)<=self.model.config.checkpoint_bytes, 'EVENT_BUDGET')
        pin=digest(dict(seq=self.count,previous=self.head,body=body))
        try:
            self.db.execute('BEGIN IMMEDIATE')
            row=self.db.execute('SELECT hash FROM events ORDER BY seq DESC LIMIT 1').fetchone()
            check((row[0] if row else ZERO)==self.head, 'WRITER_CONFLICT')
            self.db.execute('INSERT INTO events VALUES (?,?,?,?,?)',(self.count,self.head,raw,hidden,pin))
            self.db.execute('COMMIT')
        except BaseException:
            self.broken=True
            if self.db.in_transaction:self.db.execute('ROLLBACK')
            raise
        self.head=pin;self.count+=1
        return pin

    def _affected(self, event):
        affected=set()
        for ep in event['event'].get('affected',[]):affected.update(self.consumers.get(ep,()))
        return sorted(affected)

    def apply_memory(self, command):
        self._live()
        event=self.memory.apply(command)
        affected=self._affected(event)
        # A failed disk commit requires reconstruction; mutated RAM cannot escape.
        try:self._append(dict(kind='memory',record=event,dirty_contexts=affected))
        except BaseException:
            self.broken=True
            raise
        self.dirty.update(affected)
        return event

    def _request(self, context, unit, receipt):
        self._live();check(context in self.states, 'CONTEXT')
        check(context not in self.dirty, 'CONTEXT_NEEDS_REVIEW')
        ep=self.checker.episode(unit);key=ep['id']
        check(key in self.memory.state, 'MISSING_EPISODE')
        state=self.memory.state[key]
        check(state['active'] and state['grounded_scope']=='pinned_artifact_contents_only'
              and state['review']=='source_checked' and state['receipt'] is not None, 'UNCHECKED_SOURCE')
        old=self.states[context];phase=old.sequence%5
        if phase==3:
            check(receipt==state['receipt']['id'] and state['receipt']==self.checker.assess(ep), 'ECHO_RECEIPT')
        else:check(receipt is None, 'UNEXPECTED_RECEIPT')
        return dict(context=context,unit=unit,episode=key,phase=phase,prior_sequence=old.sequence,
                    receipt=state['receipt'] if phase==3 else None,ledger_head=self.head,
                    origin='execution_receipt' if phase==3 else 'source_observation')

    def _forward(self, request):
        native=self.inputs.batch([request['unit']],self.model.config,self.device)
        with torch.no_grad():
            return self.model(native,(self.states[request['context']],),phase=request['phase'],
                              event_ids=[digest(request)],evidence_origin=[request['origin']])

    def advance(self, context, unit, *, receipt=None):
        request=self._request(context,unit,receipt);out=self._forward(request)
        s=out['states'][0];hidden=s.hidden.detach().cpu().contiguous().numpy().tobytes()
        body=dict(kind='advance',request=request,hidden_sha256=sha(hidden),output_sha256=output_digest(out))
        self._append(body,hidden)
        self.states[context]=s;self._support(context,request['episode'])
        return out

    def _support(self, context, key):
        self.support[context].add(key);self.consumers.setdefault(key,set()).add(context)

    def reset(self, context, *, reason):
        self._live();check(context in self.states and context in self.dirty, 'RESET_REQUIRES_REVIEW')
        check(type(reason) is str and bool(reason.strip()), 'RESET_REASON')
        body=dict(kind='reset',context=context,reason=reason,prior_sequence=self.states[context].sequence)
        self._append(body);self._reset(context)

    def _reset(self, context):
        for key in self.support[context]:self.consumers[key].discard(context)
        self.support[context]=set();self.states[context]=self.model.start(context);self.dirty.remove(context)

    def assess_learning(self, data, baseline_model, candidate_model, *, baseline_pin, candidate_pin,
                        baseline_manifest, candidate_manifest):
        """Execute the finite gate and append its selection to this same ledger.

        The selection is for a learning successor. It never switches live model
        weights under existing context tensors or grants production admission.
        """
        from model.loop_learning import decide
        from model.contracts import HEX
        from model.loop_training_checkpoint import tensor_fingerprint
        data.fresh(disk=True)
        check(data.session is self and all(isinstance(p,str) and HEX.fullmatch(p) for p in (baseline_pin,candidate_pin)), 'LEARNING_BINDING')
        for m,manifest,pin in ((baseline_model,baseline_manifest,baseline_pin),(candidate_model,candidate_manifest,candidate_pin)):
            check(digest(manifest)==pin and manifest['binding']==data.binding
                  and manifest['memory_head']==self.memory.head and manifest['standard']==self.model.standard_id
                  and manifest['fingerprint']['model']==digest(tensor_fingerprint(m.state_dict())), 'LEARNING_CHECKPOINT')
            m.check_reference()
            check(m.standard_id==self.model.standard_id and m.source_id==self.model.source_id and m.specs==self.model.specs,'LEARNING_STANDARD')
        check(candidate_manifest['parent']==baseline_pin, 'LEARNING_PARENT')
        baseline=data.evaluate(baseline_model);candidate=data.evaluate(candidate_model)
        decision=decide(data.plan,baseline,candidate,len(self.model.specs))
        body=dict(kind='learning',memory_head=self.memory.head,binding=data.binding,plan=data.plan,
                  baseline_pin=baseline_pin,candidate_pin=candidate_pin,baseline=baseline,candidate=candidate,
                  decision=decision,selected=candidate_pin if decision['eligible'] else baseline_pin,
                  prior_selected=self.selected_learning,serving_version_changed=False)
        self._append(body);self.selected_learning=body['selected']
        return body

    def _replay(self, body, hidden, neural):
        kind=body.get('kind')
        if kind=='memory':
            check(set(body)=={'kind','record','dirty_contexts'} and hidden is None, 'MEMORY_EVENT')
            event=body['record']['event'];command={k:v for k,v in event.items() if k not in ('result','affected')}
            actual=self.memory.apply(command)
            check(actual==body['record'] and self._affected(actual)==body['dirty_contexts'], 'MEMORY_REPLAY')
            self.dirty.update(body['dirty_contexts'])
        elif kind=='advance':
            check(set(body)=={'kind','request','hidden_sha256','output_sha256'} and hidden is not None, 'ADVANCE_EVENT')
            r=body['request'];receipt=r['receipt']['id'] if r['receipt'] else None
            check(r==self._request(r['context'],r['unit'],receipt), 'SOURCE_JOIN')
            check(sha(hidden)==body['hidden_sha256'], 'CONTEXT_BYTES')
            a=np.frombuffer(hidden,dtype=np.float32)
            shape=(len(self.model.specs),self.model.config.width)
            check(a.size==shape[0]*shape[1] and np.isfinite(a).all(), 'CONTEXT_SHAPE')
            if neural:
                out=self._forward(r)
                check(output_digest(out)==body['output_sha256'] and
                      out['states'][0].hidden.detach().cpu().contiguous().numpy().tobytes()==hidden, 'NEURAL_REPLAY')
                self.neural_replayed+=1
            context=r['context']
            self.states[context]=ContextState(context,self.model.standard_id,r['prior_sequence']+1,
                                             torch.from_numpy(a.copy().reshape(shape)).to(self.device),digest(r))
            self._support(context,r['episode'])
        elif kind=='learning':
            from model.loop_learning import decide
            from model.contracts import HEX
            check(set(body)=={'kind','memory_head','binding','plan','baseline_pin','candidate_pin','baseline','candidate','decision','selected','prior_selected','serving_version_changed'} and hidden is None, 'LEARNING_EVENT')
            expected=decide(body['plan'],body['baseline'],body['candidate'],len(self.model.specs))
            check(body['memory_head']==self.memory.head and body['decision']==expected
                  and body['prior_selected']==self.selected_learning and body['serving_version_changed'] is False
                  and all(isinstance(body[p],str) and HEX.fullmatch(body[p]) for p in ('baseline_pin','candidate_pin','binding'))
                  and body['selected']==body['candidate_pin' if expected['eligible'] else 'baseline_pin'], 'LEARNING_DECISION')
            self.selected_learning=body['selected']
        elif kind=='reset':
            check(set(body)=={'kind','context','reason','prior_sequence'} and hidden is None, 'RESET_EVENT')
            context=body['context']
            check(context in self.dirty and body['prior_sequence']==self.states[context].sequence
                  and type(body['reason']) is str and bool(body['reason'].strip()), 'RESET_EVENT')
            self._reset(context)
        else:check(False,'EVENT_KIND')

    def snapshot(self):
        self._live()
        return dict(contract=self.contract_id,head=self.head,events=self.count,memory=self.memory.snapshot(),
                    contexts={c:dict(sequence=s.sequence,last_event=s.last_event,needs_review=c in self.dirty,
                        support=sorted(self.support[c]),hidden_sha256=sha(s.hidden.detach().cpu().contiguous().numpy().tobytes()))
                        for c,s in sorted(self.states.items())},
                    selected_learning=self.selected_learning,
                    world_truth_established=False,behavioural_compliance_established=False,served=False)
