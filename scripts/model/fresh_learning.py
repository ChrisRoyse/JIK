"""Learn fresh completed consequences while retaining every earlier training example."""
import copy
from dataclasses import replace
import json
from pathlib import Path

import torch

from claims.codata_source import check, sha
from claims.episodic_memory import digest
from model.constellation_learning import LearningData
from model.controlled_learning import ControlledData
from model.loop_training_checkpoint import tensor_fingerprint
from model.tool_rehearsal import RehearsalData
from model import proposal_regions, rehearsal_retention, response_basis, tool_context_regions


def code_pins():
    root = Path(__file__).resolve().parents[1]
    names = ('model/fresh_learning.py','model/fresh_learning_session.py',
             'model/proposal_regions.py','model/response_basis.py','model/response_update.py')
    return {n:sha((root/n).read_bytes()) for n in names}

class FreshData:
    def __init__(self, session, base, spec):
        check(base is session.learning_base, 'FRESH_LEARNING_BASE')
        base = session.increment_base
        self.base, self.session, self.spec = base,session,copy.deepcopy(spec)
        base.fresh()
        self.head, self.memory_head = session.head,session.memory.head
        self.model, self.frozen = base.model,base.frozen
        self.plan, self.binding = copy.deepcopy(base.plan),copy.deepcopy(base.binding)
        self.examples, self.labels = dict(base.examples),dict(base.labels)
        self.actual, self.outcomes = dict(base.actual),dict(base.outcomes)
        self.new, self.entry_pins = {},{}
        checkpoint = Path(spec['checkpoint']['root'])
        raw = (checkpoint/'manifest.json').read_bytes()
        check(sha(raw) == spec['checkpoint']['pin'],'PROPOSAL_LEARNING_PARENT_PIN')
        manifest = json.loads(raw)
        check(manifest['binding'] == base.binding and manifest['step'] == base.plan['steps'],
              'PROPOSAL_LEARNING_PARENT_BINDING')
        self.field_predecessor = checkpoint,spec['checkpoint']['pin'],manifest
        self.parent_pin = spec['checkpoint']['pin']
        self.binding['native_world']['offsets'] = {n:v+manifest['step'] for n,v in manifest['offsets'].items()}
        experiment = spec['experiment']
        forecast = session.fresh_proposals[experiment]['forecast']
        check(forecast['spec']['corrected'] == spec['checkpoint'],'PROPOSAL_LEARNING_FORECAST_PARENT')
        payload = session.journal.payload(experiment)
        source_key = forecast['spec']['context']
        source = base.examples[source_key]
        check(source_key in base.plan['train'],'PROPOSAL_LEARNING_BASE_SPLIT')
        contexts = spec['contexts']
        expected = sorted(k for k,v in session.echo_contexts.items()
                          if k.startswith('fresh:') and v['experiment'] == experiment and v['variant'] == 'corrected')
        check(contexts == expected and contexts,'PROPOSAL_LEARNING_CONTEXT_ROSTER')
        observations = [session.echo_contexts[k]['observation'] for k in contexts]
        check(len(set(observations)) == len(observations),'PROPOSAL_LEARNING_DUPLICATE_OBSERVATION')
        provider = session.echo_runtime[contexts[0]]['inputs']
        view, regions = proposal_regions.extend(source['view'],self.model,provider,observations,
            dict(head=self.head,experiment=experiment,contexts=contexts))
        for index,context in enumerate(contexts):
            entry = session.echo_contexts[context]
            check(entry['status'] == 'corrected_state' and entry['state']['sequence'] % 5 == 0,
                  'PROPOSAL_LEARNING_FEEDBACK_INCOMPLETE')
            observation, condition = entry['observation'],entry['condition']
            runtime = session.fresh_actions[experiment][condition]['corrected']
            saved = payload['predictions'][condition]
            check(entry['fork']['candidate']['pin'] == self.parent_pin
                  and runtime['checkpoint'] == entry['fork']['candidate'],'PROPOSAL_LEARNING_CHECKPOINT')
            check(provider.checked(observation) == entry['native']
                  and provider.observations.entries[observation]['condition'] == condition,
                  'PROPOSAL_LEARNING_OCCURRENCE')
            check(session._support_checked(entry['episode'])['id'] == entry['receipt'],'PROPOSAL_LEARNING_RECEIPT')
            state = replace(runtime['prestate'],context=context,hidden=runtime['prestate'].hidden.clone())
            check(state.sequence % 5 == 2 and digest(tensor_fingerprint(state.hidden)) == payload['state'],
                  'PROPOSAL_LEARNING_PRESTATE')
            batch = runtime['batch']
            check(batch.description == saved['fields'] and digest(tensor_fingerprint(
                {n:s.values for n,s in batch.items()})) == saved['values'],'PROPOSAL_LEARNING_INPUT')
            target = provider.observed(observation,self.model.config,session.device)
            region = regions['first']+index
            key = digest(dict(experiment=experiment,observation=observation,training='fresh-feedback-v1'))
            check(key not in self.examples,'PROPOSAL_LEARNING_KEY')
            allowed = torch.ones(regions['regions'],dtype=torch.bool,device=session.device)
            self.examples[key] = dict(source,state=state,request=saved['request'],native=batch,
                view=view,allowed=allowed,cutoff=allowed.clone(),region=region,target=target.slots,
                target_inputs=provider,target_unit=provider.lookup[observation],
                region_costs=proposal_regions.costs(view,target.slots),history=replace(runtime['history'],context=context))
            self.labels[key] = torch.tensor([r['outcome'] for r in regions['bank']['rows']],
                                            dtype=torch.long,device=session.device)
            actual = provider.observations.read(observation)
            self.actual[key], self.outcomes[key] = (actual,observation),int(actual['result']['returncode'] != 0)
            self.plan['train'].append(key)
            self.entry_pins[context] = digest(entry)
            self.new[key] = dict(context=context,observation=observation,condition=condition,region=region,
                native=entry['native'],receipt=entry['receipt'],input=saved['values'],fields=saved['fields'],
                prestate=payload['state'],request=digest(saved['request']),
                target=digest(tensor_fingerprint({n:(s.values,s.padding) for n,s in target.slots.items()})),
                correction_state_used_as_input=False,heldout=False)
        self.keys = tuple(self.plan['train'])+tuple(self.plan['evaluate'])
        self.binding.update(head=self.head,memory_head=self.memory_head,plan=self.plan)
        # Tighten to already achieved values; never weaken the inherited ceilings.
        previous_raw = Path(spec['retained_report']['path']).read_bytes()
        check(sha(previous_raw) == spec['retained_report']['pin'],'PROPOSAL_LEARNING_RETAINED_BYTES')
        previous = json.loads(previous_raw)
        check(previous['candidate_pin'] == self.parent_pin,'PROPOSAL_LEARNING_RETAINED_PARENT')
        check(previous['binding'] == base.binding, 'FRESH_LEARNING_PARENT_REPORT')
        ceilings = copy.deepcopy(base.binding['grounded_retention']['ceilings'])
        self.retained_keys = list(base.plan['train'])
        check(len(self.retained_keys) == len(set(self.retained_keys)) and
              not set(self.retained_keys) & set(self.plan['evaluate']), 'FRESH_LEARNING_RETAINED_SPLIT')
        for key in self.retained_keys:
            achieved = rehearsal_retention.measurements(previous['after'][key])
            if key not in ceilings:
                ceilings[key] = achieved
            else:
                for kind,names in ceilings[key].items():
                    for name,value in names.items():
                        names[name] = min(value,achieved[kind][name])
        self.retained_ceilings = {k:v['native'] for k,v in ceilings.items()}
        self.binding['grounded_retention'].update(keys=self.retained_keys,ceilings=ceilings)
        self.binding['training_retention'].update(keys=self.retained_keys,ceilings=self.retained_ceilings)
        self.grounded_retention_pin = digest(self.binding['grounded_retention'])
        recipe = {k:v for k,v in spec.items() if k != 'basis'}
        self.binding['fresh_feedback_learning'] = dict(code=code_pins(),spec=recipe,pairs=self.new,regions=regions,
            historical_binding=digest(base.binding),previous_scores=digest(previous['after']),
            scope='exposed_observation_rehearsal_not_transfer',scientific_admission=False)
        if 'basis' in spec:
            self.response_basis = response_basis.Basis(spec['basis'],self.parent_pin,digest(self.binding))
            check(self.response_basis.manifest['retained_keys'] == self.retained_keys,
                  'FRESH_LEARNING_BASIS_SPLIT')
            self.binding['fresh_feedback_learning']['response_basis'] = spec['basis']
        self.binding_pin = digest(self.binding)
        self.fresh()

    def __getattr__(self, name):
        return getattr(self.base,name)

    def fresh(self):
        self.base.fresh()
        self.session._fresh()
        check(self.session.head == self.head and self.session.memory.head == self.memory_head,
              'PROPOSAL_LEARNING_STALE')
        for context,pin in self.entry_pins.items():
            entry = self.session.echo_contexts[context]
            check(digest(entry) == pin and self.session._support_checked(entry['episode'])['id'] == entry['receipt'],
                  'PROPOSAL_LEARNING_EVIDENCE_CHANGED')
        check(digest(self.binding) == self.binding_pin,'PROPOSAL_LEARNING_BINDING_CHANGED')
        rehearsal_retention.validate(self)

    def prediction_model(self, model, key):
        return tool_context_regions.attach(self.examples[key]['view'],model)

    def forward(self, model, key):
        if key not in self.new:
            return self.base.forward(model,key)
        e, saved = self.examples[key],self.new[key]
        check(torch.equal(e['allowed'],e['cutoff']) and bool(e['allowed'].all()),'PROPOSAL_LEARNING_CUTOFF')
        check(e['native'].description == saved['fields'] and digest(tensor_fingerprint(
            {n:s.values for n,s in e['native'].items()})) == saved['input'],'PROPOSAL_LEARNING_INPUT')
        check(e['state'].sequence % 5 == 2 and digest(tensor_fingerprint(e['state'].hidden)) == saved['prestate'],
              'PROPOSAL_LEARNING_PRESTATE')
        check(digest(e['request']) == saved['request'],'PROPOSAL_LEARNING_REQUEST')
        check(digest(tensor_fingerprint({n:(s.values,s.padding) for n,s in e['target'].items()})) == saved['target'],
              'PROPOSAL_LEARNING_TARGET_CHANGED')
        provider = e['target_inputs']
        check(provider.memory_hash() == provider.memory_pin,'PROPOSAL_LEARNING_NATIVE_CHANGED')
        view = self.prediction_model(model,key)
        view.check_reference()
        return view(e['native'],(view.bind_state(e['state']),),allowed=e['allowed'],phase=2,
            event_ids=[digest(e['request'])],evidence_origin=['generated_proposal'],history=e['history'])

    recognition_risk = ControlledData.recognition_risk
    loss = LearningData.loss
    evaluate = RehearsalData.evaluate
