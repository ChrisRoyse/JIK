"""Prospective resource interventions connected to the retained field learner journal."""
import copy
from pathlib import Path

import torch

from claims.codata_source import check
from claims.episodic_memory import digest
from claims.process_conditions import ConditionInputs
from claims.process_observation import ProcessObservations
from claims import field_forecast
from model.field_learning import FieldData, FieldToolLoop
from model.field_native import FieldBatch
from model.native import NativeInput
from model.loop_training_checkpoint import fingerprint, tensor_fingerprint
from model import field_checkpoint, process_anchor


def intervention(data, model, base_key, condition, conditions):
    """Reuse exact measured fields, changing only the resource window sequence."""
    e = data.examples[base_key]
    original = e['native']
    provider = data.session.condition_inputs.for_condition(condition)
    binding = provider.checked(condition)
    public = provider.conditions.read(condition)
    check(public == conditions.read(condition), 'FIELD_EXPERIMENT_PUBLIC_INPUT')
    old = original.description['condition']
    old_public = data.session.condition_inputs.conditions.for_condition(old['condition']).read(old['condition'])
    check({k:v for k,v in old_public.items() if k != 'wheels'} ==
          {k:v for k,v in public.items() if k != 'wheels'}, 'FIELD_EXPERIMENT_PROGRAM_CHANGED')
    extra = provider.batch([binding['unit']], model.config, data.session.device)
    slots, fields = {}, {}
    for spec in model.specs:
        name = spec.name
        pieces, rows, cursor = [], [], 0
        for role in ('action', 'program'):
            value = original.role(name, role)
            source, = [r for r in original.description['fields'][name] if r['role'] == role]
            pieces.append(value)
            rows.append(dict(source, start=cursor, end=cursor+value.shape[1]))
            cursor += value.shape[1]
        program, = [r for r in binding['fields'][name] if r['field'] == 'program']
        check(torch.equal(original.role(name, 'program'),
              extra[name].values[:, program['window_start']:program['window_end']]),
              'FIELD_EXPERIMENT_PROGRAM_WINDOWS')
        for r in binding['fields'][name]:
            if r['field'] == 'wheel_filename':
                value = extra[name].values[:, r['window_start']:r['window_end']]
                pieces.append(value)
                rows.append(dict(role='wheel_filename', start=cursor, end=cursor+value.shape[1],
                                 text_sha256=r['text_sha256']))
                cursor += value.shape[1]
        check(cursor <= model.config.max_native_tokens and cursor*spec.dims <= model.config.max_native_elements,
              'FIELD_EXPERIMENT_BUDGET')
        values = torch.cat(pieces, dim=1)
        slot = original[name]
        slots[name] = NativeInput(slot.identity, slot.rows, values,
            torch.zeros(values.shape[:2], dtype=torch.bool, device=values.device), slot.absent)
        fields[name] = rows
    description = dict(schema='controlled-native-fields-v1', historical_request=digest(e['request']),
        fresh_conditions=dict(pin=conditions.pin, condition=condition), measured_fields=binding,
        fields=fields, numerical_identifiers=False, scope='planned_resource_intervention')
    batch = FieldBatch(slots, description)
    batch.checked(model.specs)
    return batch


def predict(data, model, base_key, batch):
    e = data.examples[base_key]
    view = data.prediction_model(model, base_key)
    request = e['request']
    with torch.no_grad():
        output = view(batch, (view.bind_state(e['state']),), allowed=e['allowed'], phase=2,
            event_ids=[digest(request)], evidence_origin=[request['origin']], history=view.context_history)
    return dict(termination=process_anchor.predict(output, view.execution_anchor_bank, e['allowed']),
        constellation=output['predicted_constellation'].pack(model.specs, model.config))


class ExperimentEvidence:
    def __init__(self, base, observations):
        self.base, self.observations = base, observations
        self.conditions = observations.condition_inputs()

    def __getattr__(self, name):
        return getattr(self.base, name)

    def checker(self, episode):
        source = episode.get('source', {})
        if episode.get('claim_kind') == 'event_occurrence' and source.get('observation_set') == self.observations.pin:
            return self.observations
        if episode.get('claim_kind') == 'execution_precondition' and source.get('commitment') == self.conditions.pin:
            return self.conditions
        return self.base

    def validate_episode(self, episode):
        self.checker(episode).validate_episode(episode)

    def assess(self, episode):
        return self.checker(episode).assess(episode)


class ExperimentLoop(FieldToolLoop):
    def __init__(self, *args, **kwargs):
        self.experiments = {}
        super().__init__(*args, **kwargs)

    def forecast_fields(self, spec, expected=None):
        self._fresh()
        check(not self.experiments, 'FIELD_EXPERIMENT_DUPLICATE')
        conditions = ConditionInputs(Path(spec['conditions']), spec['condition_pin'])
        if expected is None:
            check(not any((conditions.root/n/'execution').exists() for n in conditions.commitment['routing']),
                  'FIELD_EXPERIMENT_ALREADY_EXECUTED')
        data = FieldData(self, train=spec['train'], evaluate=spec['evaluate'], continuation=spec['candidate'])
        check(spec['base_forecast'] in spec['train'], 'FIELD_EXPERIMENT_BASE')
        model, optimizer, _ = field_checkpoint.start(data)
        before = fingerprint(model, optimizer)
        e = data.examples[spec['base_forecast']]
        batches = {key: intervention(data, model, spec['base_forecast'], key, conditions)
                   for key in sorted(conditions.inputs)}
        predictions = {}
        for key, batch in batches.items():
            value = predict(data, model, spec['base_forecast'], batch)
            predictions[key] = dict(candidate=value['termination'], constellation=value['constellation'],
                fields=batch.description, values=digest(tensor_fingerprint({n:s.values for n,s in batch.items()})))
        up = {s.name:model.query['_fields'][s.name].up.weight.detach().clone() for s in model.specs}
        try:
            with torch.no_grad():
                for s in model.specs:
                    model.query['_fields'][s.name].up.weight.zero_()
            for key, batch in batches.items():
                predictions[key]['field_disabled'] = predict(data, model, spec['base_forecast'], batch)['termination']
        finally:
            with torch.no_grad():
                for name, value in up.items():
                    model.query['_fields'][name].up.weight.copy_(value)
        check(fingerprint(model, optimizer) == before, 'FIELD_EXPERIMENT_MODEL_MUTATED')
        payload = dict(conditions=dict(root=spec['conditions'], pin=conditions.pin),
            base_forecast=spec['base_forecast'], historical_request=e['request'],
            state=digest(tensor_fingerprint(e['state'].hidden)),
            history=data.binding['training_views'][spec['base_forecast']]['definition'],
            candidate=dict(name=spec['candidate'], pin=self.decisions[-1]['pin'], fingerprint=before),
            predictions=predictions, field_disabled_scope='diagnostic_ablation_not_a_separately_trained_baseline',
            known_program=True, native_media_capability=False, scientific_admission=False, serving=False)
        body = dict(kind='field_experiment_forecast', spec=spec, prior_head=self.head,
                    payload=digest(payload), code=field_forecast.code_pins())
        if expected is None:
            self.journal.append(body, payload)
        else:
            check(body == expected and payload == self.journal.payload(expected['payload']),
                  'FIELD_EXPERIMENT_REPLAY')
        self.experiments[body['payload']] = dict(status='pending', forecast=body, feedback=None)
        return body

    def observe_fields(self, root, pin, forecast_root, forecast_pin, expected=None):
        self._fresh()
        seal, predictions = field_forecast.read(Path(forecast_root), forecast_pin)
        key = seal['forecast']['payload']
        check(key in self.experiments and self.experiments[key]['status'] == 'pending',
              'FIELD_EXPERIMENT_NOT_PENDING')
        check(self.experiments[key]['forecast'] == seal['forecast'], 'FIELD_EXPERIMENT_FORECAST_JOIN')
        observations = ProcessObservations(Path(root), pin, self.source)
        check(observations.manifest['controlled_forecast'] == forecast_pin, 'FIELD_EXPERIMENT_CAPTURE_JOIN')
        by_condition = {entry['condition']:ob for ob,entry in observations.entries.items()}
        check(len(by_condition) == len(observations.entries) == len(predictions['predictions'])
              and set(by_condition) == set(predictions['predictions']), 'FIELD_EXPERIMENT_OBSERVATIONS')
        scores = {}
        for condition, ob in by_condition.items():
            entry = observations.entries[ob]
            actual = observations.read(ob)
            check(entry['experiment'] == forecast_pin and entry['occurrence'] == actual['container'],
                  'FIELD_EXPERIMENT_OCCURRENCE')
            lead = field_forecast.match(seal, predictions, condition, actual)
            scores[condition] = dict(observation=ob, seconds_before_execution=lead,
                **{name:process_anchor.score(predictions['predictions'][condition][name], actual, ob)
                   for name in ('candidate', 'field_disabled')})
        old = self.memory
        staged = copy.copy(old)
        for name in ('episodes', 'state', 'children', 'events'):
            setattr(staged, name, copy.deepcopy(getattr(old, name)))
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
            body = dict(kind='field_experiment_feedback', forecast=key, root=str(root), pin=pin,
                forecast_root=str(forecast_root), forecast_pin=forecast_pin,
                prior_head=self.head, payload=digest(payload), code=field_forecast.code_pins())
            if expected is None:
                self.journal.append(body, payload)
            else:
                check(body == expected and payload == self.journal.payload(expected['payload']),
                      'FIELD_EXPERIMENT_FEEDBACK_REPLAY')
        except BaseException:
            self.memory = old
            raise
        self.experiments[key].update(status='observed', feedback=body)
        return body

    def _replay(self, body):
        kind = body.get('kind')
        if kind in ('field_experiment_forecast', 'field_experiment_feedback'):
            check(body['code'] == field_forecast.code_pins(), 'FIELD_EXPERIMENT_CODE')
        if kind == 'field_experiment_forecast':
            self.forecast_fields(body['spec'], expected=body)
        elif kind == 'field_experiment_feedback':
            self.observe_fields(body['root'], body['pin'], body['forecast_root'], body['forecast_pin'], expected=body)
        else:
            super()._replay(body)

    def snapshot(self):
        value = super().snapshot()
        if self.experiments:
            value['controlled_experiments'] = copy.deepcopy(self.experiments)
        return value
