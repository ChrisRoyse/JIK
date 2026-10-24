"""Forecast a distinct proposal with two checkpoints over the same prior world."""
import copy
from pathlib import Path

import torch

from claims.codata_source import check
from claims.episodic_memory import digest
from claims import fresh_forecast
from claims.fresh_observation import FreshObservations
from model.field_experiment import ExperimentEvidence
from model.loop_training_checkpoint import fingerprint, tensor_fingerprint
from model.fresh_input import FreshNative
from model.response_cycles import CycleData, CycleLoop
from model import field_checkpoint, process_anchor


def predict(data, model, key, batch, request):
    example = data.examples[key]
    view = data.prediction_model(model,key)
    view.check_reference()
    with torch.no_grad():
        output = view(batch,(view.bind_state(example['state']),),allowed=example['allowed'],phase=2,
            event_ids=[digest(request)],evidence_origin=['generated_proposal'],history=example['history'])
    return dict(termination=process_anchor.predict(output,view.execution_anchor_bank,example['allowed']),
        constellation=output['predicted_constellation'].pack(model.specs,model.config))


class FreshLoop(CycleLoop):
    def __init__(self, *args, **kwargs):
        self.fresh_proposals = {}
        super().__init__(*args,**kwargs)

    def forecast_fresh(self, spec, expected=None):
        self._fresh()
        check(not self.fresh_proposals, 'FRESH_FORECAST_PREFIX')
        if expected is None:
            check(not Path(spec['execution']).exists(), 'FRESH_ALREADY_EXECUTED')
        native = FreshNative(**spec['native'])
        prepared = native.population.prepared
        data = CycleData(self,self.learning_base,spec['learning'])
        key = spec['context']
        check(key in data.new and key in data.plan['train'], 'FRESH_FORECAST_CONTEXT')
        before,before_opt,bm = field_checkpoint.load(Path(spec['before']['root']),self.learning_base,
            expected=spec['before']['pin'])
        corrected,corrected_opt,cm = field_checkpoint.load(Path(spec['corrected']['root']),data,
            expected=spec['corrected']['pin'])
        check(bm['step'] == self.learning_base.plan['steps'] and cm['step'] == data.plan['steps'],
              'FRESH_CHECKPOINT_INCOMPLETE')
        models = dict(before=(before,before_opt,spec['before']['pin']),
                      corrected=(corrected,corrected_opt,spec['corrected']['pin']))
        fingerprints = {n:fingerprint(m,o) for n,(m,o,_) in models.items()}
        example = data.examples[key]
        history = dict(context=example['history'].context,rows=list(example['history'].rows),
            frames=tensor_fingerprint([{n:dict(values=s.values,padding=s.padding,absent=s.absent,rows=s.rows)
                for n,s in frame.slots.items()} for frame in example['history'].frames]))
        state = digest(tensor_fingerprint(example['state'].hidden))
        predictions = {}
        for condition in sorted(prepared.conditions.inputs):
            batch = native.batch(condition,before.config,self.device,before.specs)
            request = dict(origin='generated_proposal',source=prepared.source.reference(),
                condition=condition,commitment=prepared.conditions.pin,
                fixed_context_request=digest(example['request']))
            # attach mutates the view's weight references: each forward must finish
            # before attaching the other checkpoint. State and history stay fixed.
            values = {n:predict(data,m,key,batch,request) for n,(m,_,_) in models.items()}
            check(values['before']['termination']['bank'] == values['corrected']['termination']['bank'],
                  'FRESH_REFERENCE_BANK_CHANGED')
            predictions[condition] = dict(request=request,fields=batch.description,
                values=digest(tensor_fingerprint({n:s.values for n,s in batch.items()})),models=values)
        check(all(fingerprint(m,o) == fingerprints[n] for n,(m,o,_) in models.items()),
              'FRESH_MODEL_MUTATED')
        check(digest(tensor_fingerprint(example['state'].hidden)) == state, 'FRESH_STATE_MUTATED')
        data.fresh()
        payload = dict(conditions=dict(root=str(prepared.root),pin=prepared.conditions.pin),
            native=native.pins,source=prepared.source.reference(),base_forecast=key,state=state,
            history=history,checkpoints={n:dict(pin=p,fingerprint=fingerprints[n]) for n,(_,_,p) in models.items()},
            predictions=predictions,exposure_closure_complete=False,fixed_historical_context=True,
            same_expanded_reference_bank=True,scientific_admission=False,serving=False,new_weight_update=False,
            scope='cumulative_weight_transfer_on_fresh_execution_with_fixed_prior_state')
        body = dict(kind='fresh_forecast',spec=spec,prior_head=self.head,payload=digest(payload),
                    code=fresh_forecast.code_pins())
        if expected is None:
            self.journal.append(body,payload)
        else:
            check(body == expected and payload == self.journal.payload(expected['payload']),
                  'FRESH_FORECAST_REPLAY')
        self.fresh_proposals[body['payload']] = dict(status='pending',forecast=body,feedback=None)
        return body

    def observe_fresh(self, root, pin, forecast_root, forecast_pin, expected=None):
        self._fresh()
        seal, forecasts = fresh_forecast.read(forecast_root, forecast_pin)
        key = seal['forecast']['payload']
        check(key in self.fresh_proposals and self.fresh_proposals[key]['status'] == 'pending'
              and self.fresh_proposals[key]['forecast'] == seal['forecast'], 'PROPOSAL_NOT_PENDING')
        observations = FreshObservations(root, pin, forecast_root, forecast_pin)
        by_condition = {e['condition']:ob for ob,e in observations.entries.items()}
        check(len(by_condition) == len(observations.entries) == len(forecasts['predictions'])
              and set(by_condition) == set(forecasts['predictions']), 'PROPOSAL_OBSERVATION_ROSTER')
        scores = {}
        for condition, ob in by_condition.items():
            actual = observations.read(ob)
            scores[condition] = dict(observation=ob,
                seconds_before_execution=fresh_forecast.match(seal,forecasts,condition,actual),
                models={n:process_anchor.score(v['termination'], actual, ob)
                        for n,v in forecasts['predictions'][condition]['models'].items()})
        old, staged = self.memory, copy.copy(self.memory)
        for name in ('episodes','state','children','events'):
            setattr(staged, name, copy.deepcopy(getattr(old,name)))
        staged.checker = ExperimentEvidence(old.checker, observations)
        self.memory = staged
        try:
            records, episodes = [], []
            for condition, ob in sorted(by_condition.items()):
                pre = staged.checker.conditions.episode(condition)
                records.extend(self._intake(pre))
                episode = observations.episode(ob, evidence=(pre['id'],))
                records.extend(self._intake(episode))
                episodes.append(episode['id'])
            payload = dict(scores=scores, memory_records=records, episodes=episodes,
                why_status='unresolved', task_correctness_established=False, new_weight_update=False)
            body = dict(kind='fresh_feedback', forecast=key, root=str(root), pin=pin,
                forecast_root=str(forecast_root), forecast_pin=forecast_pin, prior_head=self.head,
                payload=digest(payload), code=fresh_forecast.code_pins())
            if expected is None:
                self.journal.append(body,payload)
            else:
                check(body == expected and payload == self.journal.payload(expected['payload']),
                      'PROPOSAL_FEEDBACK_REPLAY')
        except BaseException:
            self.memory = old
            raise
        self.fresh_proposals[key].update(status='observed',feedback=body)
        return body

    def _replay(self, body):
        kind = body.get('kind')
        if kind in ('fresh_forecast','fresh_feedback'):
            check(body['code'] == fresh_forecast.code_pins(), 'PROPOSAL_EXPERIMENT_CODE')
        if kind == 'fresh_forecast':
            self.forecast_fresh(body['spec'],expected=body)
        elif kind == 'fresh_feedback':
            self.observe_fresh(body['root'],body['pin'],body['forecast_root'],body['forecast_pin'],expected=body)
        else:
            super()._replay(body)

    def snapshot(self):
        value = super().snapshot()
        if self.fresh_proposals:
            value['fresh_experiments'] = copy.deepcopy(self.fresh_proposals)
        return value
