"""A second observed increment, with the prior examples and their banks retained."""
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
    names = ('successor_learning.py','successor_checkpoint.py','successor_targets.py','successor_session.py')
    return {n:sha((Path(__file__).parent/n).read_bytes()) for n in names}


class SuccessorData:
    def __init__(self, session, base, spec):
        check(base is session.transfer_data,'SUCCESSOR_BASE')
        base.fresh()
        self.session,self.base,self.spec = session,base,copy.deepcopy(spec)
        self.head,self.memory_head = session.head,session.memory.head
        self.model,self.frozen = base.model,base.frozen
        self.plan,self.binding = copy.deepcopy(base.plan),copy.deepcopy(base.binding)
        self.examples,self.labels = dict(base.examples),dict(base.labels)
        self.actual,self.outcomes = dict(base.actual),dict(base.outcomes)
        self.new,self.entry_pins = {},{}
        self.parent_pin = spec['checkpoint']['pin']
        raw = (Path(spec['checkpoint']['root'])/'manifest.json').read_bytes()
        check(sha(raw) == self.parent_pin,'SUCCESSOR_PARENT_PIN')
        manifest = json.loads(raw)
        check(manifest['binding'] == base.binding and manifest['step'] == base.plan['steps'],
              'SUCCESSOR_PARENT_BINDING')
        self.binding['native_world']['offsets'] = {n:v+manifest['step'] for n,v in manifest['offsets'].items()}
        experiment = spec['experiment']
        forecast = session.next_proposals[experiment]['forecast']
        check(forecast['spec']['corrected'] == spec['checkpoint'],'SUCCESSOR_FORECAST_PARENT')
        payload = session.journal.payload(experiment)
        source = base.examples[forecast['spec']['context']]
        check(forecast['spec']['context'] in base.plan['train'],'SUCCESSOR_CONTEXT_SPLIT')
        contexts = spec['contexts']
        expected = sorted(k for k,v in session.echo_contexts.items() if k.startswith('next:') and
                          v['experiment'] == experiment and v['variant'] == 'corrected')
        check(contexts == expected and len(contexts) == 2,'SUCCESSOR_CONTEXT_ROSTER')
        observations = [session.echo_contexts[k]['observation'] for k in contexts]
        check(len(set(observations)) == len(observations),'SUCCESSOR_OBSERVATION_ROSTER')
        provider = session.echo_runtime[contexts[0]]['inputs']
        view,regions = proposal_regions.extend(source['view'],self.model,provider,observations,
            dict(head=self.head,experiment=experiment,contexts=contexts))
        check(regions['first'] == 17 and regions['regions'] == 19,'SUCCESSOR_REGION_ROSTER')
        for index,context in enumerate(contexts):
            entry = session.echo_contexts[context]
            check(entry['status'] == 'corrected_state' and entry['state']['sequence'] % 5 == 0,
                  'SUCCESSOR_FEEDBACK_INCOMPLETE')
            observation,condition = entry['observation'],entry['condition']
            runtime = session.next_actions[experiment][condition]['corrected']
            saved = payload['predictions'][condition]
            check(entry['fork']['candidate']['pin'] == self.parent_pin and
                  runtime['checkpoint'] == entry['fork']['candidate'],'SUCCESSOR_CHECKPOINT')
            check(provider.checked(observation) == entry['native'] and
                  provider.observations.entries[observation]['condition'] == condition,'SUCCESSOR_OCCURRENCE')
            check(session._support_checked(entry['episode'])['id'] == entry['receipt'],'SUCCESSOR_RECEIPT')
            state = replace(runtime['prestate'],context=context,hidden=runtime['prestate'].hidden.clone())
            check(state.sequence % 5 == 2 and digest(tensor_fingerprint(state.hidden)) == payload['state'],
                  'SUCCESSOR_PRESTATE')
            batch = runtime['batch']
            check(batch.description == saved['fields'] and digest(tensor_fingerprint(
                {n:s.values for n,s in batch.items()})) == saved['values'],'SUCCESSOR_INPUT')
            target = provider.observed(observation,self.model.config,session.device)
            region = regions['first']+index
            key = digest(dict(experiment=experiment,observation=observation,training='successor-feedback-v1'))
            check(key not in self.examples,'SUCCESSOR_KEY')
            allowed = torch.ones(regions['regions'],dtype=torch.bool,device=session.device)
            self.examples[key] = dict(source,state=state,request=saved['request'],native=batch,
                view=view,allowed=allowed,cutoff=allowed.clone(),region=region,target=target.slots,
                target_inputs=provider,target_unit=provider.lookup[observation],
                region_costs=proposal_regions.costs(view,target.slots),history=replace(runtime['history'],context=context))
            self.labels[key] = torch.tensor([r['outcome'] for r in regions['bank']['rows']],
                                           dtype=torch.long,device=session.device)
            actual = provider.observations.read(observation)
            self.actual[key],self.outcomes[key] = (actual,observation),int(actual['result']['returncode'] != 0)
            self.plan['train'].append(key)
            self.entry_pins[context] = digest(entry)
            self.new[key] = dict(context=context,observation=observation,condition=condition,region=region,
                native=entry['native'],receipt=entry['receipt'],input=saved['values'],fields=saved['fields'],
                prestate=payload['state'],request=digest(saved['request']),
                target=digest(tensor_fingerprint({n:(s.values,s.padding) for n,s in target.slots.items()})),
                correction_state_used_as_input=False,heldout=False)
        self.keys = tuple(self.plan['train'])+tuple(self.plan['evaluate'])
        self.binding.update(head=self.head,memory_head=self.memory_head,plan=self.plan)
        raw = Path(spec['retained_report']['path']).read_bytes()
        check(sha(raw) == spec['retained_report']['pin'],'SUCCESSOR_RETAINED_PIN')
        previous = json.loads(raw)
        check(previous['candidate_pin'] == self.parent_pin and previous['binding'] == base.binding,
              'SUCCESSOR_RETAINED_BINDING')
        self.retained_keys = list(base.plan['train'])
        ceilings = copy.deepcopy(base.binding['grounded_retention']['ceilings'])
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
        recipe = {k:v for k,v in spec.items() if k not in ('basis','baseline')}
        self.binding['successor_feedback_learning'] = dict(code=code_pins(),spec=recipe,pairs=self.new,
            regions=regions,historical_binding=digest(base.binding),previous_scores=digest(previous['after']),
            scope='exposed_observation_rehearsal_not_transfer',scientific_admission=False)
        if 'basis' in spec:
            self.response_basis = response_basis.Basis(spec['basis'],self.parent_pin,digest(self.binding))
            check(self.response_basis.manifest['retained_keys'] == self.retained_keys,'SUCCESSOR_BASIS_SPLIT')
            self.binding['successor_feedback_learning']['basis'] = spec['basis']
        if 'baseline' in spec:
            from model.successor_targets import bind
            self.binding['successor_targets'] = bind(self,spec['baseline'])
        self.binding_pin = digest(self.binding)
        self.direction_cache,self.cache_trace = None,[]
        self.fresh()

    def __getattr__(self,name):
        return getattr(self.base,name)

    def fresh(self):
        self.base.fresh()
        self.session._fresh()
        check(self.session.head == self.head and self.session.memory.head == self.memory_head,
              'SUCCESSOR_STALE')
        for context,pin in self.entry_pins.items():
            entry = self.session.echo_contexts[context]
            check(digest(entry) == pin and self.session._support_checked(entry['episode'])['id'] == entry['receipt'],
                  'SUCCESSOR_EVIDENCE_CHANGED')
        check(digest(self.binding) == self.binding_pin,'SUCCESSOR_BINDING_CHANGED')
        rehearsal_retention.validate(self)

    def prediction_model(self,model,key):
        return tool_context_regions.attach(self.examples[key]['view'],model)

    def forward(self,model,key):
        if key not in self.new:
            return self.base.forward(model,key)
        e,saved = self.examples[key],self.new[key]
        check(torch.equal(e['allowed'],e['cutoff']) and bool(e['allowed'].all()),'SUCCESSOR_CUTOFF')
        check(e['native'].description == saved['fields'] and digest(tensor_fingerprint(
            {n:s.values for n,s in e['native'].items()})) == saved['input'],'SUCCESSOR_INPUT')
        check(e['state'].sequence % 5 == 2 and digest(tensor_fingerprint(e['state'].hidden)) == saved['prestate'],
              'SUCCESSOR_PRESTATE')
        check(digest(e['request']) == saved['request'] and digest(tensor_fingerprint(
            {n:(s.values,s.padding) for n,s in e['target'].items()})) == saved['target'],'SUCCESSOR_TARGET')
        provider = e['target_inputs']
        check(provider.memory_hash() == provider.memory_pin,'SUCCESSOR_NATIVE_CHANGED')
        view = self.prediction_model(model,key)
        view.check_reference()
        return view(e['native'],(view.bind_state(e['state']),),allowed=e['allowed'],phase=2,
            event_ids=[digest(e['request'])],evidence_origin=['generated_proposal'],history=e['history'])

    recognition_risk = ControlledData.recognition_risk
    loss = LearningData.loss
    evaluate = RehearsalData.evaluate
