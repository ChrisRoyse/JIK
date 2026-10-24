"""Incremental learning pairs from completed, captured controlled feedback.

Only pre-execution state and condition readings enter the forecast. Observed
outputs label supervision after Echo/Correction; they never enter that input.
"""
import copy
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from claims.codata_source import check, sha
from claims.episodic_memory import digest
from model.constellation import ordered_distance
from model.constellation_learning import LearningData
from model.field_echo import code_pins as echo_code_pins
from model.field_experiment import intervention
from model.field_learning import FieldData
from model.loop_training_checkpoint import tensor_fingerprint
from model.tool_rehearsal import RehearsalData, OBJECTIVE
from model import constellation_risk, process_anchor, tool_context_regions


def code_pins():
    return dict(echo_code_pins(), **{
        'model/'+n:sha((Path(__file__).parent/n).read_bytes()) for n in
        ('controlled_learning.py', 'controlled_update.py', 'controlled_session.py')})


class ControlledData:
    def __init__(self, session, spec):
        self.base = FieldData(session, train=spec['train'], evaluate=spec['evaluate'],
                              continuation=spec['foundation'])
        self.session, self.spec = session, copy.deepcopy(spec)
        self.model, self.frozen = self.base.model, self.base.frozen
        self.plan = copy.deepcopy(self.base.plan)
        self.binding = copy.deepcopy(self.base.binding)
        self.binding['plan'] = self.plan
        self.examples, self.labels = dict(self.base.examples), dict(self.base.labels)
        self.actual, self.outcomes = dict(self.base.actual), dict(self.base.outcomes)
        self.entry_pins, self.controlled = {}, {}
        contexts = spec['contexts']
        check(contexts and len(set(contexts)) == len(contexts), 'CONTROLLED_CONTEXT_ROSTER')
        for context in contexts:
            check(context in session.echo_contexts, 'CONTROLLED_CONTEXT_UNKNOWN')
            entry = session.echo_contexts[context]
            check(entry['status'] == 'corrected_state' and entry['state']['sequence'] % 5 == 0,
                  'CONTROLLED_FEEDBACK_INCOMPLETE')
            experiment = session.experiments[entry['experiment']]
            f_spec = experiment['forecast']['spec']
            check(f_spec['train'] == spec['train'] and f_spec['evaluate'] == spec['evaluate'] and
                  f_spec['candidate'] == spec['foundation'], 'CONTROLLED_FOUNDATION')
            source_key = f_spec['base_forecast']
            source = self.base.examples[source_key]
            check(source_key in spec['train'] and
                  session.source.by_task[source['request']['task']]['partition'] == 'train',
                  'CONTROLLED_TRAIN_SPLIT')
            runtime = session.echo_runtime[context]
            provider = runtime['inputs']
            observation = entry['observation']
            check(provider.checked(observation) == entry['native'], 'CONTROLLED_NATIVE_BINDING')
            actual = provider.observations.read(observation)
            check(provider.observations.entries[observation]['condition'] == entry['condition'],
                  'CONTROLLED_CONDITION_JOIN')
            receipt = session._support_checked(entry['episode'])
            check(receipt['id'] == entry['receipt'], 'CONTROLLED_RECEIPT')
            # Reuse the canonical region only when complete captured surfaces match.
            matches = [k for k in spec['train'] if self.base.actual[k][0]['captured'] == actual['captured']]
            check(len(matches) == 1, 'CONTROLLED_CANONICAL_OUTPUT_REGION')
            matched, = matches
            target_region = self.base.binding['training_views'][matched]['target_region']
            check(target_region in source['units'], 'CONTROLLED_REGION_UNAVAILABLE')
            region = source['units'].index(target_region)
            check(source['view'].execution_anchor_bank['rows'][region]['returncode'] == actual['result']['returncode'],
                  'CONTROLLED_OUTPUT_ANCHOR')
            batch = intervention(self.base, self.model, source_key, entry['condition'],
                                 provider.observations.condition_inputs())
            prediction = session.journal.payload(experiment['forecast']['payload'])['predictions'][entry['condition']]
            check(batch.description == prediction['fields'] and
                  digest(tensor_fingerprint({n:s.values for n,s in batch.items()})) == prediction['values'],
                  'CONTROLLED_SEALED_INPUT')
            target = provider.observed(observation, self.model.config, session.device)
            costs, counts = {}, {}
            references = source['view'].references()
            for lane in self.model.specs:
                slot = target.slots[lane.name]
                values = slot.values[0, ~slot.padding[0]].cpu().numpy().astype(np.float64)
                values /= np.linalg.norm(values, axis=1, keepdims=True)
                reference = references[lane.name]
                distances = []
                for i in range(len(source['units'])):
                    old = reference.values[i, ~reference.padding[i]].cpu().numpy().astype(np.float64)
                    check(len(values)*len(old) <= self.model.config.max_attention_elements,
                          'CONTROLLED_DISTANCE_BUDGET')
                    old /= np.linalg.norm(old, axis=1, keepdims=True)
                    distances.append(ordered_distance(np.clip(values@old.T, -1, 1)))
                costs[lane.name] = (lane.identity, torch.tensor(distances, dtype=torch.float32, device=session.device))
                counts[lane.name] = dict(target_windows=len(values),
                    reference_windows=int((~reference.padding).sum()), regions=len(distances))
            key = digest(dict(context=context, forecast=entry['fork']['forecast'], observation=observation))
            check(key not in self.examples, 'CONTROLLED_EXAMPLE_COLLISION')
            state = replace(source['state'], context=context, hidden=source['state'].hidden.clone())
            request = dict(schema='controlled-learning-action-v1', context=context, origin='generated_proposal',
                historical_request=source['request'], condition=entry['condition'],
                forecast=entry['fork']['forecast'], inputs=prediction['values'])
            check(state.sequence % 5 == 2 and digest(tensor_fingerprint(state.hidden)) == entry['fork']['prestate'],
                  'CONTROLLED_PREACT_STATE')
            self.examples[key] = dict(source, state=state, request=request, native=batch, region=region,
                target=target.slots, target_inputs=provider, target_unit=provider.lookup[observation],
                region_costs=costs, history=replace(source['view'].context_history, context=context))
            self.labels[key] = self.base.labels[source_key]
            self.actual[key], self.outcomes[key] = (actual, observation), int(actual['result']['returncode'] != 0)
            self.plan['train'].append(key)
            self.entry_pins[context] = digest(entry)
            self.controlled[key] = dict(context=context, entry=digest(entry), episode=entry['episode'],
                receipt=receipt['id'], observation=observation, native=entry['native'],
                source_forecast=source_key, sealed_forecast=entry['fork']['forecast'],
                input=prediction['values'], fields=batch.description,
                prestate=entry['fork']['prestate'], correction_state_used_as_input=False,
                canonical_output_example=matched, canonical_region=target_region,
                region=region, costs={n:tensor_fingerprint(v) for n,(_,v) in costs.items()},
                measured_costs=counts, known_program=True, heldout=False)
        self.keys = tuple(self.plan['train'])+tuple(self.plan['evaluate'])
        self.binding['controlled_feedback'] = dict(schema='controlled-feedback-learning-v1',
            code=code_pins(), spec=spec, pairs=self.controlled,
            scope='post_observation_supervision_of_pre_execution_inputs', scientific_admission=False)
        self.field_predecessor = self.base.field_predecessor
        self.parent_pin = session.decisions[-1]['pin']
        if session.controlled_candidates:
            prior = next(reversed(session.controlled_candidates.values()))
            path = session.controlled_root/prior['candidate']
            raw = (path/'manifest.json').read_bytes()
            manifest = json.loads(raw)
            check(sha(raw) == prior['pin'] and manifest['binding']['controlled_feedback']['code'] == code_pins(),
                  'CONTROLLED_PREDECESSOR_PIN')
            check(set(prior['spec']['contexts']) <= set(contexts), 'CONTROLLED_RETAINED_CONTEXTS')
            self.binding['native_world']['offsets'] = {n:v+manifest['step'] for n,v in manifest['offsets'].items()}
            self.field_predecessor = (path, prior['pin'], manifest)
            self.parent_pin = prior['pin']
            self.binding['controlled_predecessor'] = dict(candidate=prior['candidate'], pin=prior['pin'],
                                                         payload=prior['payload'])
        self.fresh()

    def __getattr__(self, name):
        return getattr(self.base, name)

    def fresh(self):
        self.base.fresh()
        for context, pin in self.entry_pins.items():
            entry = self.session.echo_contexts[context]
            check(digest(entry) == pin and self.session._support_checked(entry['episode'])['id'] == entry['receipt'],
                  'CONTROLLED_EVIDENCE_STALE')

    def prediction_model(self, model, key):
        return tool_context_regions.attach(self.examples[key]['view'], model)

    def forward(self, model, key):
        if key not in self.controlled:
            return self.base.forward(model, key)
        e = self.examples[key]
        check(torch.equal(e['allowed'], e['cutoff']) and e['native'].description == self.controlled[key]['fields'],
              'CONTROLLED_INPUT_CHANGED')
        e['native'].checked(model.specs)
        view = self.prediction_model(model, key)
        return view(e['native'], (view.bind_state(e['state']),), allowed=e['allowed'], phase=2,
            event_ids=[digest(e['request'])], evidence_origin=[e['request']['origin']], history=e['history'])

    def recognition_risk(self, model, key, output):
        check(self.plan['outcome_objective'] == OBJECTIVE, 'CONTROLLED_OBJECTIVE')
        e = self.examples[key]
        native = constellation_risk.expected(output, e['region_costs'], e['allowed'], model.specs)
        masses = process_anchor.tensors(output, self.labels[key], e['allowed'])
        truth = masses['common'][1].new_tensor([float(i == self.outcomes[key]) for i in range(3)])
        common = (masses['common'][1]-truth).square().sum()/len(model.specs)
        return {s.name:native[s.name]+(masses[s.name][1]-truth).square().sum()+common for s in model.specs}

    loss = LearningData.loss
    evaluate = RehearsalData.evaluate
