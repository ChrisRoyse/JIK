"""Captured controlled-process consequences enter intrinsic Echo/Correction state.

The field branch describes planned actions. Output surfaces have their own explicit
NativeWorld path; they never acquire invented action/program/resource fields.
"""
import base64
import copy
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from claims.codata_source import check, sha
from claims.episodic_memory import digest
from claims.process_observation import ProcessObservations
from claims import field_forecast
from model.constellation import ordered_distance, unpack
from model.field_experiment import ExperimentLoop, intervention
from model.field_learning import FieldData
from model.loop_training_checkpoint import fingerprint, tensor_fingerprint
from model.native_world import NativeWorld
from model.process_input import ProcessInputs
from model import field_checkpoint, process_anchor


def code_pins():
    return dict(field_forecast.code_pins(), **{
        'model/field_echo.py': sha(Path(__file__).read_bytes())})


def state_record(state):
    a = state.hidden.detach().cpu().contiguous().numpy().astype('<f4')
    raw = a.tobytes()
    return dict(context=state.context, standard=state.standard_id, world=state.world_id,
        sequence=state.sequence, last_event=state.last_event,
        hidden=dict(shape=list(a.shape), dtype='float32-le', sha256=sha(raw),
                    base64=base64.b64encode(raw).decode()))


def feedback_forward(view, native, state, *, allowed, phase, event, history, origin):
    check(phase in (3, 4), 'FIELD_ECHO_PHASE')
    check(native.origin == 'observed', 'FIELD_ECHO_OBSERVED_REQUIRED')
    check(origin == 'execution_receipt', 'FIELD_ECHO_CAPTURED_REQUIRED')
    native.validate(view.specs, view.config)
    # Explicit typed entrypoint. Action-field validation remains unchanged.
    return NativeWorld.forward(view, native.slots, (state,), allowed=allowed,
        phase=phase, event_ids=[event], evidence_origin=[origin], history=history)


def discrepancy(prediction, observed, model):
    result = {}
    for spec in model.specs:
        predicted = unpack(prediction[spec.name], spec, max_windows=model.config.max_native_tokens)
        slot = observed.slots[spec.name]
        actual = slot.values[0, ~slot.padding[0]].detach().cpu().numpy().astype(np.float64)
        actual /= np.linalg.norm(actual, axis=1, keepdims=True)
        check(len(actual)*len(predicted) <= model.config.max_attention_elements,
              'FIELD_ECHO_SCORE_BUDGET')
        cosines = np.clip(actual@predicted.T, -1, 1)
        result[spec.name] = dict(instrument=spec.identity,
            prediction_windows=len(predicted), observed_windows=len(actual),
            cosines=cosines.tolist(), ordered_distance=ordered_distance(cosines))
    return result


class EchoExperimentLoop(ExperimentLoop):
    def __init__(self, *args, **kwargs):
        self.echo_contexts, self.echo_runtime, self.echo_enrollments = {}, {}, {}
        super().__init__(*args, **kwargs)

    def enroll_echo(self, spec, expected=None):
        self._fresh()
        experiment = spec['experiment']
        check(experiment not in self.echo_enrollments, 'FIELD_ECHO_DUPLICATE_ENROLLMENT')
        check(experiment in self.experiments and self.experiments[experiment]['status'] == 'observed',
              'FIELD_ECHO_UNOBSERVED_EXPERIMENT')
        prior = self.experiments[experiment]
        feedback = prior['feedback']
        seal, prediction = field_forecast.read(Path(feedback['forecast_root']), feedback['forecast_pin'])
        check(seal['forecast'] == prior['forecast'], 'FIELD_ECHO_FORECAST_JOIN')
        observations = ProcessObservations(Path(feedback['root']), feedback['pin'], self.source)
        base = self.execution_inputs.base
        inputs = ProcessInputs(observations, base, Path(spec['population']), Path(spec['native']),
            population_pin=spec['population_pin'], native_pin=spec['native_pin'])
        forecast_spec = prior['forecast']['spec']
        data = FieldData(self, train=forecast_spec['train'], evaluate=forecast_spec['evaluate'],
                         continuation=forecast_spec['candidate'])
        model, optimizer, _ = field_checkpoint.start(data)
        frozen = fingerprint(model, optimizer)
        check(frozen == prediction['candidate']['fingerprint'], 'FIELD_ECHO_CANDIDATE_DRIFT')
        e = data.examples[forecast_spec['base_forecast']]
        check(digest(tensor_fingerprint(e['state'].hidden)) == prediction['state'],
              'FIELD_ECHO_PRESTATE_DRIFT')
        feedback_payload = self.journal.payload(feedback['payload'])
        entries, runtimes = {}, {}
        for observation, entry in sorted(observations.entries.items()):
            condition = entry['condition']
            saved = prediction['predictions'][condition]
            check(feedback_payload['scores'][condition]['observation'] == observation,
                  'FIELD_ECHO_OBSERVATION_JOIN')
            episode, = [key for key in feedback_payload['episodes']
                        if self.memory.episodes[key]['source'].get('observation') == observation]
            receipt = self._support_checked(episode)
            batch = intervention(data, model, forecast_spec['base_forecast'], condition,
                                 observations.condition_inputs())
            check(batch.description == saved['fields'] and
                  digest(tensor_fingerprint({n:s.values for n,s in batch.items()})) == saved['values'],
                  'FIELD_ECHO_INPUT_DRIFT')
            view = data.prediction_model(model, forecast_spec['base_forecast'])
            with torch.no_grad():
                act = view(batch, (view.bind_state(e['state']),), allowed=e['allowed'], phase=2,
                    event_ids=[digest(e['request'])], evidence_origin=[e['request']['origin']],
                    history=view.context_history)
            check(act['predicted_constellation'].pack(model.specs, model.config) == saved['constellation']
                  and process_anchor.predict(act, view.execution_anchor_bank, e['allowed']) == saved['candidate'],
                  'FIELD_ECHO_SEALED_PREDICTION_DRIFT')
            context = 'controlled:'+digest([experiment, condition])
            state = replace(act['states'][0], context=context, hidden=act['states'][0].hidden.clone())
            history = replace(view.context_history, context=context)
            inputs.checked(observation)
            observed = inputs.observed(observation, model.config, self.device)
            entries[context] = dict(experiment=experiment, condition=condition, observation=observation,
                episode=episode, receipt=receipt['id'], native=inputs.checked(observation),
                fork=dict(kind='sealed_action_state_reconstruction', historical_context=e['state'].context,
                          historical_event=digest(e['request']), prestate=prediction['state'],
                          forecast=feedback['forecast_pin'], candidate=prediction['candidate']),
                status='awaiting_echo', state=state_record(state), last_transition=None)
            runtimes[context] = dict(model=model, optimizer=optimizer, frozen=frozen, view=view,
                state=state, history=history, allowed=e['allowed'], inputs=inputs,
                prediction=saved['constellation'], observed=observed)
        check(fingerprint(model, optimizer) == frozen, 'FIELD_ECHO_WEIGHTS_CHANGED')
        payload = dict(contexts=entries, native=inputs.identity,
            instrument_count=len(model.specs), learned_understanding=False, new_weight_update=False,
            scientific_admission=False, serving=False)
        body = dict(kind='field_echo_enrollment', spec=spec, prior_head=self.head,
                    payload=digest(payload), code=code_pins())
        self._echo_publish(body, payload, expected)
        self.echo_contexts.update(entries)
        self.echo_runtime.update(runtimes)
        self.echo_enrollments[experiment] = body
        return body

    def advance_echo(self, context, observation, phase, *, origin='execution_receipt', expected=None):
        self._fresh()
        check(context in self.echo_contexts, 'FIELD_ECHO_CONTEXT')
        entry, runtime = self.echo_contexts[context], self.echo_runtime[context]
        check(observation == entry['observation'], 'FIELD_ECHO_OBSERVATION_JOIN')
        check((phase == 3 and entry['status'] == 'awaiting_echo') or
              (phase == 4 and entry['status'] == 'awaiting_correction'), 'FIELD_ECHO_PHASE_ORDER')
        check(origin == 'execution_receipt', 'FIELD_ECHO_CAPTURED_REQUIRED')
        check(self._support_checked(entry['episode'])['id'] == entry['receipt'], 'FIELD_ECHO_RECEIPT')
        check(runtime['inputs'].checked(observation) == entry['native'], 'FIELD_ECHO_NATIVE_DRIFT')
        model, view = runtime['model'], runtime['view']
        check(fingerprint(model, runtime['optimizer']) == runtime['frozen'], 'FIELD_ECHO_WEIGHTS_CHANGED')
        before = state_record(runtime['state'])
        check(before == entry['state'], 'FIELD_ECHO_STATE_DRIFT')
        request = dict(context=context, observation=observation, condition=entry['condition'],
            native=entry['native'], episode=entry['episode'], receipt=entry['receipt'], phase=phase,
            origin=origin, before=digest(before), prior_head=self.head,
            same_occurrence_reflection=phase == 4)
        with torch.no_grad():
            output = feedback_forward(view, runtime['observed'], runtime['state'],
                allowed=runtime['allowed'], phase=phase, event=digest(request),
                history=runtime['history'], origin=origin)
        state = output['states'][0]
        after = state_record(state)
        check(before['hidden'] != after['hidden'] and state.sequence == runtime['state'].sequence+1,
              'FIELD_ECHO_NO_TRANSITION')
        check(state.standard_id == runtime['state'].standard_id and
              fingerprint(model, runtime['optimizer']) == runtime['frozen'], 'FIELD_ECHO_WEIGHTS_CHANGED')
        payload = dict(before=before, after=after,
            observed={n:tensor_fingerprint(s.values[0, ~s.padding[0]])
                      for n,s in runtime['observed'].slots.items()},
            prediction_error=discrepancy(runtime['prediction'], runtime['observed'], model),
            output=output['predicted_constellation'].pack(model.specs, model.config),
            recognition=output['recognition'], frozen=runtime['frozen'],
            state_delta_by_instrument={s.name:float(torch.linalg.vector_norm(
                state.hidden[i]-runtime['state'].hidden[i])) for i,s in enumerate(model.specs)},
            why_status='unresolved', learned_understanding=False, new_weight_update=False,
            correction_effectiveness_established=False, scientific_admission=False, serving=False)
        body = dict(kind='field_echo_transition', request=request, payload=digest(payload), code=code_pins())
        self._echo_publish(body, payload, expected)
        runtime['state'] = state
        entry.update(state=after, status='awaiting_correction' if phase == 3 else 'corrected_state',
                     last_transition=body['payload'])
        return body

    def _echo_publish(self, body, payload, expected):
        if expected is None:
            self.journal.append(body, payload)
        else:
            check(body == expected and payload == self.journal.payload(expected['payload']),
                  'FIELD_ECHO_REPLAY')

    def _replay(self, body):
        kind = body.get('kind')
        if kind in ('field_echo_enrollment', 'field_echo_transition'):
            check(body['code'] == code_pins(), 'FIELD_ECHO_CODE')
        if kind == 'field_echo_enrollment':
            self.enroll_echo(body['spec'], expected=body)
        elif kind == 'field_echo_transition':
            r = body['request']
            self.advance_echo(r['context'], r['observation'], r['phase'], origin=r['origin'], expected=body)
        else:
            super()._replay(body)

    def snapshot(self):
        value = super().snapshot()
        if self.echo_contexts:
            value['native_feedback_contexts'] = copy.deepcopy(self.echo_contexts)
            value['native_feedback_enrollments'] = copy.deepcopy(self.echo_enrollments)
        return value
