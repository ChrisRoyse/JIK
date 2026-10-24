"""Replay a sealed prospective action into independently owned native feedback loops."""
import copy
from dataclasses import replace
from pathlib import Path

import torch

from claims.codata_source import check, sha
from claims.episodic_memory import digest
from claims import next_forecast
from model.field_echo import state_record
from model.loop_training_checkpoint import fingerprint, tensor_fingerprint
from model.next_experiment import NextLoop
from model.next_input import NextNative
from model.next_echo_input import NextEchoInputs
from model import stable_checkpoint, stable_response, process_anchor, tool_context_regions


def code_pins():
    root = Path(__file__).resolve().parents[1]
    names = ('model/next_echo.py','model/next_echo_input.py','claims/next_echo_population.py',
             'model/field_echo.py','model/next_experiment.py','claims/next_forecast.py',
             'model/proposal_echo_input.py','model/stable_checkpoint.py','model/stable_response.py')
    return {n:sha((root/n).read_bytes()) for n in names}


def owned_view(source, model):
    # Only immutable references are shared. Module registration must be private:
    # attaching the second checkpoint cannot replace the first one's parameters.
    view = copy.copy(source)
    view._modules = source._modules.copy()
    tool_context_regions.attach(view,model)
    view.check_reference()
    return view


class NextEchoLoop(NextLoop):
    def __init__(self, *args, **kwargs):
        self.next_actions = {}
        super().__init__(*args,**kwargs)

    def forecast_next(self, spec, expected=None):
        body = super().forecast_next(spec,expected=expected)
        data = self.transfer_data
        data.fresh()
        native = NextNative(**spec['native'])
        example = data.examples[spec['context']]
        before,bo,_ = stable_checkpoint.start(data)
        corrected,co,_ = stable_checkpoint.load(Path(spec['corrected']['root']),data,
                                               expected=spec['corrected']['pin'])
        models = dict(before=(before,bo),corrected=(corrected,co))
        payload = self.journal.payload(body['payload'])
        history = dict(context=example['history'].context,rows=list(example['history'].rows),
            frames=tensor_fingerprint([{n:dict(values=s.values,padding=s.padding,absent=s.absent,rows=s.rows)
                for n,s in frame.slots.items()} for frame in example['history'].frames]))
        check(history == payload['history'] and digest(tensor_fingerprint(example['state'].hidden)) ==
              payload['state'],'NEXT_ECHO_PRESTATE')
        rows = {}
        for condition,saved in sorted(payload['predictions'].items()):
            batch = native.batch(condition,before.config,self.device,before.specs)
            check(batch.description == saved['fields'] and digest(tensor_fingerprint(
                {n:s.values for n,s in batch.items()})) == saved['values'],'NEXT_ECHO_ACTION_INPUT')
            rows[condition] = {}
            for name,(model,optimizer) in models.items():
                model.eval()
                frozen = fingerprint(model,optimizer)
                check(frozen == payload['checkpoints'][name]['fingerprint'],'NEXT_ECHO_CHECKPOINT')
                view = owned_view(example['view'],model)
                check(stable_response.buffers(view) == data.binding['stable_response']['static_buffers'],
                      'NEXT_ECHO_STABLE_BUFFERS')
                with torch.no_grad():
                    act = view(batch,(view.bind_state(example['state']),),allowed=example['allowed'],phase=2,
                        event_ids=[digest(saved['request'])],evidence_origin=['generated_proposal'],
                        history=example['history'])
                check(act['predicted_constellation'].pack(model.specs,model.config) ==
                      saved['models'][name]['constellation'] and process_anchor.predict(
                          act,view.execution_anchor_bank,example['allowed']) == saved['models'][name]['termination'],
                      'NEXT_ECHO_SEALED_ACTION')
                check(fingerprint(model,optimizer) == frozen,'NEXT_ECHO_WEIGHTS_CHANGED')
                rows[condition][name] = dict(model=model,optimizer=optimizer,frozen=frozen,view=view,
                    batch=batch,prestate=example['state'],allowed=example['allowed'],history=example['history'],
                    act=act,prediction=saved['models'][name]['constellation'],checkpoint=payload['checkpoints'][name])
        self.next_actions[body['payload']] = rows
        return body

    def enroll_next_echo(self, spec, expected=None):
        self._fresh()
        experiment = spec['experiment']
        check(experiment in self.next_proposals and self.next_proposals[experiment]['status'] == 'observed'
              and experiment not in self.echo_enrollments,'PROPOSAL_ECHO_ENROLLMENT_STATE')
        prior = self.next_proposals[experiment]
        feedback = prior['feedback']
        seal, forecasts = next_forecast.read(feedback['forecast_root'],feedback['forecast_pin'])
        check(seal['forecast'] == prior['forecast'],'PROPOSAL_ECHO_FORECAST_JOIN')
        inputs = NextEchoInputs(**spec['inputs'])
        check(inputs.observations.pin == feedback['pin'],'PROPOSAL_ECHO_OBSERVATION_SET')
        feedback_payload = self.journal.payload(feedback['payload'])
        entries, runtimes = {}, {}
        for observation,entry in sorted(inputs.observations.entries.items()):
            condition = entry['condition']
            check(feedback_payload['scores'][condition]['observation'] == observation,'PROPOSAL_ECHO_OBSERVATION_JOIN')
            episode, = [key for key in feedback_payload['episodes']
                        if self.memory.episodes[key]['source'].get('observation') == observation]
            receipt = self._support_checked(episode)
            for name,prior_runtime in self.next_actions[experiment][condition].items():
                model = prior_runtime['model']
                check(tuple(model.specs) == inputs.specs,'PROPOSAL_ECHO_INSTRUMENT_CHANGED')
                context = 'next:'+digest([experiment,condition,name])
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
        body = dict(kind='next_echo_enrollment',spec=spec,prior_head=self.head,payload=digest(payload),code=code_pins())
        self._echo_publish(body,payload,expected)
        self.echo_contexts.update(entries)
        self.echo_runtime.update(runtimes)
        self.echo_enrollments[experiment] = body
        return body

    def advance_echo(self, context, observation, phase, *, origin='execution_receipt', expected=None):
        if context.startswith('next:') and context in self.echo_runtime:
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
        if body.get('kind') == 'next_echo_enrollment':
            check(body['code'] == code_pins(),'PROPOSAL_ECHO_CODE')
            self.enroll_next_echo(body['spec'],expected=body)
        else:
            super()._replay(body)
