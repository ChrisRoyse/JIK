"""Compare fixed-context predictions before fresh execution, preserving the parent."""
import copy
from pathlib import Path

from claims.codata_source import check
from claims.episodic_memory import digest
from claims import attempt_forecast
from claims.attempt_observation import AttemptObservations
from model.attempt_input import AttemptNative
from model.field_experiment import ExperimentEvidence
from model.loop_training_checkpoint import fingerprint, tensor_fingerprint
from model.next_experiment import predict
from model.successor_residual_session import ResidualLoop
from model import successor_checkpoint, successor_residual_checkpoint, process_anchor


def baselines(bank):
    outcomes = [r['outcome'] for r in bank['rows'] if r['observation'] is not None]
    check(outcomes and all(v in (0,1) for v in outcomes), 'ATTEMPT_BASELINE_ROSTER')
    return dict(always_success=[1.,0.,0.], always_failure=[0.,1.,0.],
                binary_uniform=[.5,.5,0.],
                observed_frequency=[outcomes.count(k)/len(outcomes) for k in range(3)])


class AttemptLoop(ResidualLoop):
    def __init__(self, *args, attempt_data, attempt_spec, **kwargs):
        self.attempt_data, self.attempt_spec = attempt_data, copy.deepcopy(attempt_spec)
        self.attempt_proposals = {}
        super().__init__(*args, **kwargs)

    def forecast_attempt(self, spec, expected=None):
        self._fresh()
        check(not self.attempt_proposals, 'ATTEMPT_FORECAST_PREFIX')
        if expected is None:
            check(not Path(spec['execution']).exists(), 'ATTEMPT_ALREADY_EXECUTED')
        data = self.attempt_data
        data.fresh()
        candidate = self.successor_candidates['successor-1']
        check(spec['learning'] == self.attempt_spec == candidate['spec']
              and spec['before'] == self.attempt_spec['checkpoint']
              and candidate['pin'] == spec['corrected']['pin']
              and candidate['parent'] == data.parent_pin, 'ATTEMPT_LEARNING_BINDING')
        key = spec['context']
        check(key == sorted(data.new)[0] and key in data.plan['train'], 'ATTEMPT_CONTEXT')
        native = AttemptNative(**spec['native'])
        prepared = native.population.prepared
        parent, parent_opt, _ = successor_checkpoint.start(data)
        corrected, corrected_opt, manifest = successor_residual_checkpoint.load(
            Path(spec['corrected']['root']), data, expected=spec['corrected']['pin'])
        check(manifest['step'] == data.plan['steps'] and manifest['parent'] == data.parent_pin,
              'ATTEMPT_CHECKPOINT_INCOMPLETE')
        models = dict(before=(parent,parent_opt,spec['before']['pin']),
                      corrected=(corrected,corrected_opt,spec['corrected']['pin']))
        for model, _, _ in models.values():
            model.eval()
        fingerprints = {n:fingerprint(m,o) for n,(m,o,_) in models.items()}
        example = data.examples[key]
        check(len(example['allowed']) == 19 and bool(example['allowed'].all())
              and example['state'].sequence % 5 == 2, 'ATTEMPT_REFERENCE_STATE')
        state = digest(tensor_fingerprint(example['state'].hidden))
        history = dict(context=example['history'].context, rows=list(example['history'].rows),
            frames=tensor_fingerprint([{n:dict(values=s.values,padding=s.padding,absent=s.absent,rows=s.rows)
                for n,s in frame.slots.items()} for frame in example['history'].frames]))
        predictions = {}
        for condition in sorted(prepared.conditions.inputs):
            batch = native.batch(condition,parent.config,self.device,parent.specs)
            request = dict(origin='generated_proposal',source=prepared.source.reference(),
                condition=condition,commitment=prepared.conditions.pin,
                fixed_context_request=digest(example['request']))
            # Finish each call before the shared immutable view attaches the other model.
            values = {n:predict(data,m,key,batch,request) for n,(m,_,_) in models.items()}
            bank = values['before']['termination']['bank']
            check(bank == values['corrected']['termination']['bank'], 'ATTEMPT_BANK_CHANGED')
            predictions[condition] = dict(request=request,fields=batch.description,
                values=digest(tensor_fingerprint({n:s.values for n,s in batch.items()})),
                models=values,baselines=baselines(bank))
        check(all(fingerprint(m,o) == fingerprints[n] for n,(m,o,_) in models.items()),
              'ATTEMPT_MODEL_MUTATED')
        check(digest(tensor_fingerprint(example['state'].hidden)) == state, 'ATTEMPT_STATE_MUTATED')
        data.fresh()
        payload = dict(conditions=dict(root=str(prepared.root),pin=prepared.conditions.pin),
            native=native.pins,source=prepared.source.reference(),base_forecast=key,state=state,
            history=history,checkpoints={n:dict(pin=p,fingerprint=fingerprints[n]) for n,(_,_,p) in models.items()},
            predictions=predictions,exposure_closure_complete=False,fixed_historical_context=True,
            correction_state_used_as_input=False,same_expanded_reference_bank=True,
            learning_binding=data.binding_pin,scientific_admission=False,serving=False,new_weight_update=False,
            scope='successor_weight_transfer_on_fresh_execution_with_fixed_prior_state')
        body = dict(kind='attempt_forecast',spec=spec,prior_head=self.head,payload=digest(payload),
                    code=attempt_forecast.code_pins())
        if expected is None:
            self.journal.append(body,payload)
        else:
            check(body == expected and payload == self.journal.payload(expected['payload']),
                  'ATTEMPT_FORECAST_REPLAY')
        self.attempt_proposals[body['payload']] = dict(status='pending',forecast=body,feedback=None)
        return body

    def observe_attempt(self, root, pin, forecast_root, forecast_pin, expected=None):
        self._fresh()
        seal, forecasts = attempt_forecast.read(forecast_root,forecast_pin)
        key = seal['forecast']['payload']
        check(key in self.attempt_proposals and self.attempt_proposals[key]['status'] == 'pending'
              and self.attempt_proposals[key]['forecast'] == seal['forecast'], 'ATTEMPT_NOT_PENDING')
        observations = AttemptObservations(root,pin,forecast_root,forecast_pin)
        by_condition = {e['condition']:ob for ob,e in observations.entries.items()}
        check(len(by_condition) == len(observations.entries) == len(forecasts['predictions'])
              and set(by_condition) == set(forecasts['predictions']), 'ATTEMPT_OBSERVATION_ROSTER')
        scores = {}
        for condition, ob in by_condition.items():
            actual = observations.read(ob)
            saved = forecasts['predictions'][condition]
            bank = saved['models']['before']['termination']['bank']
            check(saved['baselines'] == baselines(bank), 'ATTEMPT_BASELINE_CHANGED')
            outcome = int(actual['result']['returncode'] != 0)
            scores[condition] = dict(observation=ob,
                seconds_before_execution=attempt_forecast.match(seal,forecasts,condition,actual),
                models={n:process_anchor.score(v['termination'],actual,ob) for n,v in saved['models'].items()},
                baselines={n:dict(probabilities=p,brier=sum((v-float(k==outcome))**2 for k,v in enumerate(p)))
                           for n,p in saved['baselines'].items()})
        old, staged = self.memory, copy.copy(self.memory)
        for name in ('episodes','state','children','events'):
            setattr(staged,name,copy.deepcopy(getattr(old,name)))
        staged.checker = ExperimentEvidence(old.checker,observations)
        self.memory = staged
        try:
            records,episodes = [],[]
            for condition,ob in sorted(by_condition.items()):
                pre = staged.checker.conditions.episode(condition)
                records.extend(self._intake(pre))
                episode = observations.episode(ob,evidence=(pre['id'],))
                records.extend(self._intake(episode))
                episodes.append(episode['id'])
            payload = dict(scores=scores,memory_records=records,episodes=episodes,why_status='unresolved',
                task_correctness_established=False,new_weight_update=False)
            body = dict(kind='attempt_feedback',forecast=key,root=str(root),pin=pin,
                forecast_root=str(forecast_root),forecast_pin=forecast_pin,prior_head=self.head,
                payload=digest(payload),code=attempt_forecast.code_pins())
            if expected is None:
                self.journal.append(body,payload)
            else:
                check(body == expected and payload == self.journal.payload(expected['payload']),
                      'ATTEMPT_FEEDBACK_REPLAY')
        except BaseException:
            self.memory = old
            raise
        self.attempt_proposals[key].update(status='observed',feedback=body)
        return body

    def _replay(self, body):
        kind = body.get('kind')
        if kind in ('attempt_forecast','attempt_feedback'):
            check(body['code'] == attempt_forecast.code_pins(), 'ATTEMPT_EXPERIMENT_CODE')
        if kind == 'attempt_forecast':
            self.forecast_attempt(body['spec'],expected=body)
        elif kind == 'attempt_feedback':
            self.observe_attempt(body['root'],body['pin'],body['forecast_root'],body['forecast_pin'],expected=body)
        else:
            super()._replay(body)

    def snapshot(self):
        value = super().snapshot()
        if self.attempt_proposals:
            value['attempt_experiments'] = copy.deepcopy(self.attempt_proposals)
        return value
