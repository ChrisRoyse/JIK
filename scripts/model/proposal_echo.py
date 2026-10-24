"""Feed actual proposal consequences into separate frozen intrinsic-loop contexts."""
from dataclasses import replace
from pathlib import Path

import torch

from claims.codata_source import check, sha
from claims.episodic_memory import digest
from claims import proposal_forecast
from model.corrective_learning import CorrectiveData
from model.field_echo import state_record
from model.field_world import FieldWorld
from model.loop_training_checkpoint import fingerprint, tensor_fingerprint
from model.proposal_experiment import ProposalLoop
from model.proposal_input import ProposalNative
from model.proposal_echo_input import ProposalEchoInputs
from model import field_checkpoint, process_anchor, tool_context_regions


def code_pins():
    root = Path(__file__).resolve().parents[1]
    return {n:sha((root/n).read_bytes()) for n in (
        'model/proposal_echo.py','model/proposal_echo_input.py','claims/proposal_echo_population.py',
        'model/field_echo.py','model/proposal_experiment.py','claims/proposal_forecast.py')}


class ProposalEchoLoop(ProposalLoop):
    def __init__(self, *args, **kwargs):
        self.proposal_actions = {}
        super().__init__(*args, **kwargs)

    def forecast_proposal(self, spec, expected=None):
        # Capture the exact prefix-bound views before the parent appends its forecast.
        # The historical correction binding must not be rebuilt against later outcomes.
        data = CorrectiveData(self,spec['correction'])
        native = ProposalNative(**spec['native'])
        base = spec['base_forecast']
        e = data.examples[base]
        model, optimizer, _ = field_checkpoint.start(data)
        corrected, corrected_opt, _ = field_checkpoint.load(Path(spec['corrected']['root']),data,
                                                            expected=spec['corrected']['pin'])
        models = dict(before=(model,optimizer),corrected=(corrected,corrected_opt))
        rows = {}
        for condition in sorted(native.population.prepared.conditions.inputs):
            batch = native.batch(condition,model.config,self.device,model.specs)
            rows[condition] = {}
            for name,(m,o) in models.items():
                # attach() mutates the reusable training view. Concurrent branches
                # need their own view wrappers before binding candidate parameters.
                view = FieldWorld.from_world(data.prediction_model(m,base))
                tool_context_regions.attach(view,m)
                rows[condition][name] = dict(model=m,optimizer=o,frozen=fingerprint(m,o),
                    view=view,batch=batch,prestate=e['state'],allowed=e['allowed'])
        body = super().forecast_proposal(spec,expected=expected)
        payload = self.journal.payload(body['payload'])
        for condition, variants in rows.items():
            saved = payload['predictions'][condition]
            for name, runtime in variants.items():
                view, m = runtime['view'], runtime['model']
                check(runtime['frozen'] == payload['checkpoints'][name]['fingerprint']
                      and digest(tensor_fingerprint(e['state'].hidden)) == payload['state'],
                      'PROPOSAL_ECHO_PRESTATE')
                check(runtime['batch'].description == saved['fields']
                      and digest(tensor_fingerprint({n:s.values for n,s in runtime['batch'].items()})) == saved['values'],
                      'PROPOSAL_ECHO_ACTION_INPUT')
                with torch.no_grad():
                    act = view(runtime['batch'],(view.bind_state(e['state']),),allowed=e['allowed'],phase=2,
                        event_ids=[digest(saved['request'])],evidence_origin=['generated_proposal'],
                        history=view.context_history)
                check(act['predicted_constellation'].pack(m.specs,m.config) == saved['models'][name]['constellation']
                      and process_anchor.predict(act,view.execution_anchor_bank,e['allowed']) ==
                          saved['models'][name]['termination'], 'PROPOSAL_ECHO_SEALED_ACTION')
                check(fingerprint(m,runtime['optimizer']) == runtime['frozen'],'PROPOSAL_ECHO_WEIGHTS_CHANGED')
                runtime.update(act=act,prediction=saved['models'][name]['constellation'],
                               history=view.context_history,checkpoint=payload['checkpoints'][name])
        self.proposal_actions[body['payload']] = rows
        return body

    def enroll_proposal_echo(self, spec, expected=None):
        self._fresh()
        experiment = spec['experiment']
        check(experiment in self.proposals and self.proposals[experiment]['status'] == 'observed'
              and experiment not in self.echo_enrollments,'PROPOSAL_ECHO_ENROLLMENT_STATE')
        prior = self.proposals[experiment]
        feedback = prior['feedback']
        seal, forecasts = proposal_forecast.read(feedback['forecast_root'],feedback['forecast_pin'])
        check(seal['forecast'] == prior['forecast'],'PROPOSAL_ECHO_FORECAST_JOIN')
        inputs = ProposalEchoInputs(**spec['inputs'])
        check(inputs.observations.pin == feedback['pin'],'PROPOSAL_ECHO_OBSERVATION_SET')
        feedback_payload = self.journal.payload(feedback['payload'])
        entries, runtimes = {}, {}
        for observation, entry in sorted(inputs.observations.entries.items()):
            condition = entry['condition']
            check(feedback_payload['scores'][condition]['observation'] == observation,'PROPOSAL_ECHO_OBSERVATION_JOIN')
            episode, = [key for key in feedback_payload['episodes']
                        if self.memory.episodes[key]['source'].get('observation') == observation]
            receipt = self._support_checked(episode)
            for name, prior_runtime in self.proposal_actions[experiment][condition].items():
                model = prior_runtime['model']
                check(tuple(model.specs) == inputs.specs,'PROPOSAL_ECHO_INSTRUMENT_CHANGED')
                context = 'proposal:'+digest([experiment,condition,name])
                state = replace(prior_runtime['act']['states'][0],context=context,
                                hidden=prior_runtime['act']['states'][0].hidden.clone())
                history = replace(prior_runtime['history'],context=context)
                native = inputs.checked(observation)
                observed = inputs.observed(observation,model.config,self.device)
                entries[context] = dict(experiment=experiment,condition=condition,observation=observation,
                    episode=episode,receipt=receipt['id'],native=native,variant=name,
                    fork=dict(kind='sealed_proposal_action_state_reconstruction',forecast=feedback['forecast_pin'],
                              candidate=prior_runtime['checkpoint'],prestate=forecasts['state']),
                    status='awaiting_echo',state=state_record(state),last_transition=None)
                runtimes[context] = dict(model=model,optimizer=prior_runtime['optimizer'],
                    frozen=prior_runtime['frozen'],view=prior_runtime['view'],state=state,history=history,
                    allowed=prior_runtime['allowed'],inputs=inputs,prediction=prior_runtime['prediction'],observed=observed)
                check(fingerprint(model,prior_runtime['optimizer']) == prior_runtime['frozen'],
                      'PROPOSAL_ECHO_WEIGHTS_CHANGED')
        check(len(entries) == 4,'PROPOSAL_ECHO_CONTEXT_ROSTER')
        payload = dict(contexts=entries,native=inputs.identity,instrument_count=len(inputs.specs),
            learned_understanding=False,new_weight_update=False,scientific_admission=False,serving=False)
        body = dict(kind='proposal_echo_enrollment',spec=spec,prior_head=self.head,payload=digest(payload),code=code_pins())
        self._echo_publish(body,payload,expected)
        self.echo_contexts.update(entries)
        self.echo_runtime.update(runtimes)
        self.echo_enrollments[experiment] = body
        return body

    def advance_echo(self, context, observation, phase, *, origin='execution_receipt', expected=None):
        if context.startswith('proposal:') and context in self.echo_runtime:
            runtime = self.echo_runtime[context]
            check(all(getattr(runtime['view'],n) is getattr(runtime['model'],n)
                      for n in ('core','query','place')),'PROPOSAL_ECHO_MODEL_OWNER')
            check(observation == self.echo_contexts[context]['observation'],'PROPOSAL_ECHO_OCCURRENCE')
            original = runtime['inputs'].observed(observation,runtime['model'].config,self.device)
            check(runtime['observed'].origin == 'observed','PROPOSAL_ECHO_OBSERVED_REQUIRED')
            for spec in runtime['model'].specs:
                actual, source = runtime['observed'].slots[spec.name],original.slots[spec.name]
                check(actual.identity == source.identity and actual.rows == source.rows
                      and torch.equal(actual.values,source.values) and torch.equal(actual.padding,source.padding)
                      and torch.equal(actual.absent,source.absent),'PROPOSAL_ECHO_NATIVE_CHANGED')
        return super().advance_echo(context,observation,phase,origin=origin,expected=expected)

    def _replay(self, body):
        if body.get('kind') == 'proposal_echo_enrollment':
            check(body['code'] == code_pins(),'PROPOSAL_ECHO_CODE')
            self.enroll_proposal_echo(body['spec'],expected=body)
        else:
            super()._replay(body)
