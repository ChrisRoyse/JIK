"""Carry corrected native state into a sealed later action and its real consequence."""
import copy
from dataclasses import replace
from pathlib import Path

import torch

from claims.codata_source import check
from claims.episodic_memory import digest
from claims import state_cycle_forecast
from claims.state_cycle_observation import StateCycleObservations
from model.attempt_echo import AttemptEchoLoop, owned_view
from model.attempt_experiment import baselines
from model.attempt_input import AttemptNative
from model.field_echo import state_record
from model.field_experiment import ExperimentEvidence
from model.loop_training_checkpoint import fingerprint, tensor_fingerprint
from model import process_anchor, state_cycle_steps as steps


class StateCycleLoop(AttemptEchoLoop):
    def __init__(self,*args,**kwargs):
        self.state_cycles,self.cycle_contexts,self.cycle_runtime = {},{},{}
        super().__init__(*args,**kwargs)

    def forecast_cycle(self,spec,expected=None):
        self._fresh()
        check(not self.state_cycles,'STATE_CYCLE_ALREADY_STARTED')
        check(self.head == spec['source_head'],'STATE_CYCLE_SOURCE_HEAD')
        if expected is None:
            check(not Path(spec['execution']).exists(),'STATE_CYCLE_ALREADY_EXECUTED')
        experiment = spec['source_experiment']
        contexts = sorted(k for k,v in self.echo_contexts.items() if v['experiment'] == experiment)
        check(contexts == spec['contexts'] and len(contexts) == 4,'STATE_CYCLE_CONTEXT_ROSTER')
        prior = self.attempt_proposals[experiment]
        check(prior['status'] == 'observed','STATE_CYCLE_PRIOR_UNOBSERVED')
        native = AttemptNative(**prior['forecast']['spec']['native'])
        prepared = native.population.prepared
        old = digest(self.snapshot())
        entries,runtimes,predictions = {},{},{}
        for source_context in contexts:
            source,runtime = steps.checked(self,source_context)
            condition,variant = source['condition'],source['variant']
            model,optimizer = runtime['model'],runtime['optimizer']
            batch = native.batch(condition,model.config,self.device,model.specs)
            condition_record = predictions.setdefault(condition,dict(fields=batch.description,
                values=tensor_fingerprint({n:s.values for n,s in batch.items()}),models={}))
            for mode in ('carried','reset'):
                name = variant+'/'+mode
                context = 'cycle:'+digest([self.head,source_context,mode])
                view = owned_view(runtime['view'],model)
                state = (replace(runtime['state'],context=context,hidden=runtime['state'].hidden.clone())
                         if mode == 'carried' else view.start(context))
                history = steps.observed_history(runtime,source['observation'],context)
                history.validate(view,(state,))
                start = state_record(state)
                trace = []
                for phase in (0,1,2):
                    before = state_record(state)
                    request = dict(context=context,source_context=source_context,condition=condition,
                        source_head=self.head,phase=phase,origin='generated_proposal',mode=mode,
                        before=digest(before),history=digest(steps.history_record(history)),
                        source=prepared.source.reference(),native=native.pins)
                    with torch.no_grad():
                        output = view(batch,(state,),allowed=runtime['allowed'],phase=phase,
                            event_ids=[digest(request)],evidence_origin=['generated_proposal'],history=history)
                    state = output['states'][0]
                    check(state.sequence == before['sequence']+1,'STATE_CYCLE_PHASE')
                    trace.append(dict(request=request,before=before,after=state_record(state)))
                prediction = dict(termination=process_anchor.predict(output,view.execution_anchor_bank,runtime['allowed']),
                    constellation=output['predicted_constellation'].pack(model.specs,model.config))
                bank = prediction['termination']['bank']
                check(len(bank['rows']) == 19,'STATE_CYCLE_BANK')
                condition_record['baselines'] = baselines(bank)
                condition_record['models'][name] = prediction
                entries[context] = dict(source_context=source_context,source_observation=source['observation'],
                    source_state=source['state'],checkpoint=source['fork']['candidate'],variant=variant,mode=mode,
                    condition=condition,start=start,state=state_record(state),trace=trace,
                    history=steps.history_record(history),status='awaiting_echo',prediction=name)
                runtimes[context] = dict(model=model,optimizer=optimizer,frozen=runtime['frozen'],view=view,
                    state=state,history=history,allowed=runtime['allowed'],inputs=runtime['inputs'],
                    source_observation=source['observation'],prediction=prediction['constellation'])
                check(fingerprint(model,optimizer) == runtime['frozen'],'STATE_CYCLE_WEIGHTS_CHANGED')
            steps.checked(self,source_context)
        check(digest(self.snapshot()) == old,'STATE_CYCLE_SOURCE_MUTATED')
        check(len(entries) == 8 and len(predictions) == 2 and
              all(set(v['models']) == {'before/carried','before/reset','corrected/carried','corrected/reset'}
                  for v in predictions.values()),'STATE_CYCLE_BRANCH_ROSTER')
        payload = dict(contexts=entries,predictions=predictions,source=prepared.source.reference(),
            conditions=dict(root=str(prepared.root),pin=prepared.conditions.pin),native=native.pins,
            source_experiment=experiment,scientific_admission=False,serving=False,new_weight_update=False,
            correction_effectiveness_established=False,scope='known_action_next_occurrence_with_state_reset_control')
        body = dict(kind='state_cycle_forecast',spec=spec,prior_head=self.head,payload=digest(payload),
                    code=state_cycle_forecast.code_pins())
        self._echo_publish(body,payload,expected)
        self.state_cycles[body['payload']] = dict(status='pending',forecast=body,feedback=None)
        self.cycle_contexts.update(entries)
        self.cycle_runtime.update(runtimes)
        return body

    def observe_cycle(self,root,pin,forecast_root,forecast_pin,expected=None):
        self._fresh()
        seal,forecasts = state_cycle_forecast.read(forecast_root,forecast_pin)
        key = seal['forecast']['payload']
        check(key in self.state_cycles and self.state_cycles[key]['status'] == 'pending'
              and self.state_cycles[key]['forecast'] == seal['forecast'],'STATE_CYCLE_NOT_PENDING')
        observations = StateCycleObservations(root,pin,forecast_root,forecast_pin)
        by_condition = {e['condition']:ob for ob,e in observations.entries.items()}
        check(set(by_condition) == set(forecasts['predictions']),'STATE_CYCLE_OBSERVATION_ROSTER')
        scores,cached = {},{}
        for condition,observation in sorted(by_condition.items()):
            actual = observations.read(observation)
            saved = forecasts['predictions'][condition]
            outcome = int(actual['result']['returncode'] != 0)
            scores[condition] = dict(observation=observation,
                seconds_before_execution=state_cycle_forecast.match(seal,forecasts,condition,actual),
                models={n:process_anchor.score(p['termination'],actual,observation) for n,p in saved['models'].items()},
                baselines={n:dict(probabilities=p,brier=sum((v-float(k==outcome))**2 for k,v in enumerate(p)))
                           for n,p in saved['baselines'].items()})
            for context,entry in self.cycle_contexts.items():
                if entry['condition'] != condition:
                    continue
                runtime = self.cycle_runtime[context]
                previous = runtime['inputs'].observations.read(entry['source_observation'])
                check(actual['container'] != previous['container'] and actual['started'] > previous['finished'],
                      'STATE_CYCLE_OCCURRENCE_REUSED')
                cached[context] = steps.cached_observation(runtime,previous,actual,observation)
        old,staged = self.memory,copy.copy(self.memory)
        for name in ('episodes','state','children','events'):
            setattr(staged,name,copy.deepcopy(getattr(old,name)))
        staged.checker = ExperimentEvidence(old.checker,observations)
        self.memory = staged
        entries,states,transitions,reuse = copy.deepcopy(self.cycle_contexts),{},{},{}
        try:
            records,episodes,episode_map = [],[],{}
            for condition,observation in sorted(by_condition.items()):
                pre = observations.condition_inputs().episode(condition)
                records.extend(self._intake(pre))
                episode = observations.episode(observation,evidence=(pre['id'],))
                records.extend(self._intake(episode));episodes.append(episode['id'])
                episode_map[condition] = episode['id']
            for context,entry in sorted(entries.items()):
                runtime = self.cycle_runtime[context]
                check(state_record(runtime['state']) == entry['state'],'STATE_CYCLE_STATE_CHANGED')
                observation = by_condition[entry['condition']]
                episode = episode_map[entry['condition']]
                receipt = self._support_checked(episode)['id']
                state,trace = steps.feedback(runtime,cached[context],observation,episode,receipt,context)
                entry.update(state=state_record(state),status='corrected_state',observation=observation,
                             episode=episode,receipt=receipt)
                states[context],transitions[context] = state,trace
                reuse[context] = dict(previous=entry['source_observation'],current=observation,
                    native=runtime['inputs'].identity,exact_surface_match=True,
                    values=tensor_fingerprint({n:s.values for n,s in cached[context].slots.items()}))
            payload = dict(scores=scores,memory_records=records,episodes=episodes,contexts=entries,
                transitions=transitions,cached_readings=reuse,why_status='unresolved',
                task_correctness_established=False,new_weight_update=False,scientific_admission=False,
                serving=False,correction_effectiveness_established=False)
            body = dict(kind='state_cycle_feedback',forecast=key,root=str(root),pin=pin,
                forecast_root=str(forecast_root),forecast_pin=forecast_pin,prior_head=self.head,
                payload=digest(payload),code=state_cycle_forecast.code_pins())
            self._echo_publish(body,payload,expected)
        except BaseException:
            self.memory = old
            raise
        self.cycle_contexts = entries
        for context,state in states.items():
            self.cycle_runtime[context]['state'] = state
        self.state_cycles[key].update(status='observed',feedback=body)
        return body

    def _replay(self,body):
        kind = body.get('kind')
        if kind in ('state_cycle_forecast','state_cycle_feedback'):
            check(body['code'] == state_cycle_forecast.code_pins(),'STATE_CYCLE_CODE')
        if kind == 'state_cycle_forecast':
            self.forecast_cycle(body['spec'],expected=body)
        elif kind == 'state_cycle_feedback':
            self.observe_cycle(body['root'],body['pin'],body['forecast_root'],body['forecast_pin'],expected=body)
        else:
            super()._replay(body)

    def snapshot(self):
        result = super().snapshot()
        if self.state_cycles:
            result['state_cycles'] = copy.deepcopy(self.state_cycles)
            result['state_cycle_contexts'] = copy.deepcopy(self.cycle_contexts)
        return result
