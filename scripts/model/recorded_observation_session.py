"""Persistent scene observation without inventing actions or outcome labels.

Every native reading remains source-bound and separately typed. Observation
updates recurrent scene state; it does not advance the identity action cycle,
update weights, admit an instrument, or assert a world interpretation as fact.
"""
from fractions import Fraction
from pathlib import Path

import torch

from claims.codata_source import check,sha
from claims.event_journal import EventJournal
from model.loop_session import output_digest
from model.native_episode_session import tensor
from model.native_world_checkpoint import load
from senses.contract import canonical,digest

FLAGS = dict(source_measurements=True,parameter_update=False,
             scientific_admission=False,serving=False,learned_capability=False)


class RecordedObservationSession:
    def __init__(self,path,model,inputs,*,checkpoint,checkpoint_pin,contexts,
                 create,retained=None):
        check(type(create) is bool and (retained is None if create else
              isinstance(retained,dict) and set(retained)=={'head','contract'}),
              'OBSERVATION_RETAINED')
        check(type(contexts) is tuple and contexts and len(set(contexts))==len(contexts)
              and all(isinstance(c,str) and c for c in contexts),'OBSERVATION_CONTEXTS')
        inputs.fresh()
        initial=load(checkpoint,model,expected=checkpoint_pin,binding=inputs.binding)
        check(all(s.sequence==0 and s.last_event is None for s in initial),'OBSERVATION_INITIAL')
        check(tuple(model.specs)==inputs.specs and
              model.core.temporal_id==digest(inputs.temporal_catalogue),'OBSERVATION_BINDING')
        self.model,self.inputs=model,inputs
        self.model.eval()
        self.states={c:model.start(c) for c in contexts}
        self.histories={c:[] for c in contexts}
        self.broken=False
        root=Path(__file__).resolve().parents[1]
        contract=dict(schema='recorded-observation-session-v1',checkpoint=checkpoint_pin,
            input_binding=inputs.binding,memory=inputs.memory.retained(),
            standard=model.standard_id,world=model.world_id,contexts=list(contexts),
            instruments=[s.identity for s in model.specs],flags=FLAGS,
            scope='retrospective_recorded_scene_observation_not_action_feedback',
            code={n:sha((root/n).read_bytes()) for n in
                  ('model/recorded_observation_session.py','model/native_episode_session.py',
                   'model/loop_session.py','claims/event_journal.py')})
        self.journal=EventJournal(path,contract,create=create,payload_bound=model.config.checkpoint_bytes)
        if not create:
            try:self.journal.restore(self._replay,**retained)
            except BaseException:
                self.close()
                raise

    def close(self):
        self.journal.close()
        self.broken=True

    def fresh(self):
        check(not self.broken,'OBSERVATION_RELOAD')
        self.journal.fresh()
        self.inputs.fresh()

    def retained(self):
        self.fresh()
        return dict(head=self.journal.head,contract=self.journal.pin)

    def _request(self,context,index,expected_head,purpose):
        check(purpose=='recorded_observation','OBSERVATION_PURPOSE')
        check(not self.broken,'OBSERVATION_RELOAD')
        self.inputs.fresh()
        check(expected_head==self.journal.head,'OBSERVATION_WRITER_CONFLICT')
        check(context in self.states,'OBSERVATION_CONTEXT')
        check(type(index) is int and 0<=index<len(self.inputs.moments),'OBSERVATION_INDEX')
        moment=self.inputs.moments[index];history=self.histories[context]
        check(moment['id'] not in {r['episode'] for r in history},'OBSERVATION_DUPLICATE')
        if history:
            previous=history[-1]
            check(all(previous['source'][k]==moment['source'][k] for k in
                      ('bundle','clock_id','split_group')),'OBSERVATION_SCENE')
            check(Fraction(*moment['event_time']['at'])>=Fraction(*previous['event_time']['at']),
                  'OBSERVATION_ORDER')
        return dict(context=context,index=index,episode=moment['id'],
            observation=len(history),identity_sequence=self.states[context].sequence,
            previous_event=self.states[context].last_event,prior_head=expected_head,
            input_binding=self.inputs.binding,source=moment['source'],event_time=moment['event_time'],
            situation=moment['content'],purpose=purpose,origin='source_observation')

    def _execute(self,request):
        # The upstream reader authenticates measurements without granting meaning admission.
        native=self.inputs.batch(request['index'],purpose='engineering_boundary')
        rows=next(iter(self.model.reference_rows.values()))
        allowed=torch.ones(len(rows),dtype=torch.bool,device=self.model.core.initial.device)
        before=self.states[request['context']]
        traces=[];old_trace=self.inputs.trace
        self.inputs.trace=traces.append
        try:
            with torch.no_grad():
                out=self.model(native,(before,),allowed=allowed,phase=before.sequence%5,
                    event_ids=[digest(request)],evidence_origin=['source_observation'],advance_phase=False)
        finally:self.inputs.trace=old_trace
        state=out['states'][0]
        check(out['phase']=='Observation' and state.sequence==before.sequence,'OBSERVATION_PHASE')
        expected=self.inputs.lanes[request['index']]
        check(len(traces)==2*len(expected),'OBSERVATION_TRAVERSAL_ROSTER')
        for name,lane in expected.items():
            found=[t for t in traces if t['lane']==name]
            check(len(found)==2 and found[0]==found[1] and found[0]['row']==request['episode']
                  and found[0]['count']==lane.count and found[0]['bytes']==lane.count*lane.dims*4
                  and found[0]['readings']==len(lane.pieces),'OBSERVATION_COMPLETE_NATIVE')
        self.inputs.fresh()
        payload=dict(world=tensor(state.hidden),identity=tensor(state.identity_hidden),
            identity_sequence=state.sequence,last_event=state.last_event,traversals=traces,
            predicted_regions=list(out['predicted_constellation'].regions))
        check(len(canonical(payload))<=self.model.config.checkpoint_bytes,'OBSERVATION_PAYLOAD_BUDGET')
        body=dict(kind='recorded_observation',request=request,payload=digest(payload),
                  output=output_digest(out),flags=FLAGS)
        return body,payload,state,out

    def observe(self,context,index,*,expected_head,purpose='recorded_observation'):
        self.fresh()
        request=self._request(context,index,expected_head,purpose)
        body,payload,state,out=self._execute(request)
        try:self.journal.append(body,payload)
        except BaseException:
            self.broken=True
            raise
        self.journal.cache.clear()
        self.states[context]=state
        self.histories[context].append(request)
        return out

    def _replay(self,body):
        check(isinstance(body,dict) and set(body)=={'kind','request','payload','output','flags'}
              and body['kind']=='recorded_observation','OBSERVATION_EVENT')
        r=body['request']
        request=self._request(r['context'],r['index'],r['prior_head'],r['purpose'])
        check(r==request,'OBSERVATION_REQUEST_REPLAY')
        stored=self.journal.payload(body['payload'])
        expected,payload,state,_=self._execute(request)
        check(body==expected and stored==payload,'OBSERVATION_NEURAL_REPLAY')
        self.journal.cache.clear()
        self.states[r['context']]=state
        self.histories[r['context']].append(request)

    def snapshot(self):
        return dict(retained=self.retained(),flags=dict(FLAGS),
            contexts={c:dict(observations=len(self.histories[c]),
                episodes=[r['episode'] for r in self.histories[c]],identity_sequence=s.sequence,
                last_event=s.last_event,world=tensor(s.hidden)['sha256'],
                identity=tensor(s.identity_hidden)['sha256']) for c,s in sorted(self.states.items())})
