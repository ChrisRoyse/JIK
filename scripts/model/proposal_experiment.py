"""Compare frozen before/corrected models on a new action, then journal its occurrence."""
import copy
from pathlib import Path

import torch

from claims.codata_source import check
from claims.episodic_memory import digest
from claims import proposal_forecast
from claims.proposal_observation import ProposalObservations
from model.corrective_session import CorrectiveLoop
from model.corrective_learning import CorrectiveData
from model.field_experiment import ExperimentEvidence
from model.loop_training_checkpoint import fingerprint, tensor_fingerprint
from model.proposal_input import ProposalNative
from model import field_checkpoint, process_anchor


def predict(data, model, base_key, batch, request):
    example = data.examples[base_key]
    view = data.prediction_model(model, base_key)
    with torch.no_grad():
        output = view(batch, (view.bind_state(example['state']),), allowed=example['allowed'], phase=2,
            event_ids=[digest(request)], evidence_origin=['generated_proposal'], history=view.context_history)
    return dict(termination=process_anchor.predict(output, view.execution_anchor_bank, example['allowed']),
        constellation=output['predicted_constellation'].pack(model.specs, model.config))


class ProposalLoop(CorrectiveLoop):
    def __init__(self, *args, **kwargs):
        self.proposals = {}
        super().__init__(*args, **kwargs)

    def forecast_proposal(self, spec, expected=None):
        self._fresh()
        check(not self.proposals and not self.corrective_candidates, 'PROPOSAL_FORECAST_PREFIX')
        if expected is None:
            check(not Path(spec['execution']).exists(), 'PROPOSAL_ALREADY_EXECUTED')
        native = ProposalNative(**spec['native'])
        prepared = native.population.prepared
        data = CorrectiveData(self, spec['correction'])
        base = spec['base_forecast']
        check(base in data.plan['train'], 'PROPOSAL_FORECAST_CONTEXT')
        models = {}
        before, opt, _ = field_checkpoint.start(data)
        models['before'] = (before, opt, data.parent_pin)
        corrected, corrected_opt, manifest = field_checkpoint.load(
            Path(spec['corrected']['root']), data, expected=spec['corrected']['pin'])
        check(manifest['step'] == data.plan['steps'], 'PROPOSAL_CHECKPOINT_INCOMPLETE')
        models['corrected'] = (corrected, corrected_opt, spec['corrected']['pin'])
        fingerprints = {n:fingerprint(m,o) for n,(m,o,_) in models.items()}
        predictions = {}
        for condition in sorted(prepared.conditions.inputs):
            batch = native.batch(condition, before.config, self.device, before.specs)
            request = dict(origin='generated_proposal', source=prepared.source.reference(),
                condition=condition, commitment=prepared.conditions.pin,
                fixed_context_request=digest(data.examples[base]['request']))
            predictions[condition] = dict(request=request, fields=batch.description,
                values=digest(tensor_fingerprint({n:s.values for n,s in batch.items()})),
                models={n:predict(data,m,base,batch,request) for n,(m,_,_) in models.items()})
        check(all(fingerprint(m,o) == fingerprints[n] for n,(m,o,_) in models.items()),
              'PROPOSAL_MODEL_MUTATED')
        payload = dict(conditions=dict(root=str(prepared.root), pin=prepared.conditions.pin),
            native=native.pins, source=prepared.source.reference(), base_forecast=base,
            state=digest(tensor_fingerprint(data.examples[base]['state'].hidden)),
            history=data.binding['training_views'][base]['definition'],
            checkpoints={n:dict(pin=p,fingerprint=fingerprints[n]) for n,(_,_,p) in models.items()},
            predictions=predictions, exposure_closure_complete=False, fixed_historical_context=True,
            scientific_admission=False, serving=False, new_weight_update=False)
        body = dict(kind='proposal_forecast', spec=spec, prior_head=self.head,
                    payload=digest(payload), code=proposal_forecast.code_pins())
        if expected is None:
            self.journal.append(body, payload)
        else:
            check(body == expected and payload == self.journal.payload(expected['payload']),
                  'PROPOSAL_FORECAST_REPLAY')
        self.proposals[body['payload']] = dict(status='pending', forecast=body, feedback=None)
        return body

    def observe_proposal(self, root, pin, forecast_root, forecast_pin, expected=None):
        self._fresh()
        seal, forecasts = proposal_forecast.read(forecast_root, forecast_pin)
        key = seal['forecast']['payload']
        check(key in self.proposals and self.proposals[key]['status'] == 'pending'
              and self.proposals[key]['forecast'] == seal['forecast'], 'PROPOSAL_NOT_PENDING')
        observations = ProposalObservations(root, pin, forecast_root, forecast_pin)
        by_condition = {e['condition']:ob for ob,e in observations.entries.items()}
        check(len(by_condition) == len(observations.entries) == len(forecasts['predictions'])
              and set(by_condition) == set(forecasts['predictions']), 'PROPOSAL_OBSERVATION_ROSTER')
        scores = {}
        for condition, ob in by_condition.items():
            actual = observations.read(ob)
            scores[condition] = dict(observation=ob,
                seconds_before_execution=proposal_forecast.match(seal,forecasts,condition,actual),
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
            body = dict(kind='proposal_feedback', forecast=key, root=str(root), pin=pin,
                forecast_root=str(forecast_root), forecast_pin=forecast_pin, prior_head=self.head,
                payload=digest(payload), code=proposal_forecast.code_pins())
            if expected is None:
                self.journal.append(body,payload)
            else:
                check(body == expected and payload == self.journal.payload(expected['payload']),
                      'PROPOSAL_FEEDBACK_REPLAY')
        except BaseException:
            self.memory = old
            raise
        self.proposals[key].update(status='observed',feedback=body)
        return body

    def _replay(self, body):
        kind = body.get('kind')
        if kind in ('proposal_forecast','proposal_feedback'):
            check(body['code'] == proposal_forecast.code_pins(), 'PROPOSAL_EXPERIMENT_CODE')
        if kind == 'proposal_forecast':
            self.forecast_proposal(body['spec'],expected=body)
        elif kind == 'proposal_feedback':
            self.observe_proposal(body['root'],body['pin'],body['forecast_root'],body['forecast_pin'],expected=body)
        else:
            super()._replay(body)

    def snapshot(self):
        value = super().snapshot()
        if self.proposals:
            value['proposal_experiments'] = copy.deepcopy(self.proposals)
        return value
