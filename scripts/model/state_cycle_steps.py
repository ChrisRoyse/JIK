"""Typed observed history, owned state forks and exact cached-output reuse."""
from dataclasses import replace

import torch

from claims.codata_source import check
from claims.episodic_memory import digest
from model.constellation import Constellation
from model.constellation_history import CausalHistory
from model.field_echo import discrepancy, feedback_forward, state_record
from model.loop_training_checkpoint import fingerprint, tensor_fingerprint


def checked(session, context):
    check(context in session.echo_contexts and context in session.echo_runtime,
          'STATE_CYCLE_CONTEXT')
    entry,runtime = session.echo_contexts[context],session.echo_runtime[context]
    check(entry['status'] == 'corrected_state' and runtime['state'].sequence % 5 == 0,
          'STATE_CYCLE_PENDING_REPAIR')
    check(state_record(runtime['state']) == entry['state'],'STATE_CYCLE_STATE_CHANGED')
    check(runtime['history'].context == runtime['state'].context,'STATE_CYCLE_HISTORY_CONTEXT')
    check(all(getattr(runtime['view'],n) is getattr(runtime['model'],n)
              for n in ('core','query','place')),'STATE_CYCLE_MODEL_OWNER')
    check(fingerprint(runtime['model'],runtime['optimizer']) == runtime['frozen'],
          'STATE_CYCLE_WEIGHTS_CHANGED')
    check(runtime['inputs'].checked(entry['observation']) == entry['native'],
          'STATE_CYCLE_NATIVE_CHANGED')
    return entry,runtime


def observed_history(runtime, observation, context):
    model,inputs = runtime['model'],runtime['inputs']
    observed = inputs.observed(observation,model.config,model.core.initial.device)
    # Address the occurrence, not the local array index; all native values stay intact.
    frame = Constellation({n:replace(s,rows=(observation,)) for n,s in observed.slots.items()},'observed')
    old = runtime['history']
    check(observation not in old.rows,'STATE_CYCLE_DUPLICATE_HISTORY')
    history = CausalHistory(context,old.rows+(observation,),old.frames+(frame,))
    check(len(history.frames) == 10 and all(sum((~f.slots[s.name].padding).sum().item()
          for f in history.frames) == 12 for s in model.specs),'STATE_CYCLE_HISTORY_CENSUS')
    return history


def history_record(history):
    return dict(context=history.context,rows=list(history.rows),frames=tensor_fingerprint([
        {n:dict(values=s.values,padding=s.padding,absent=s.absent,rows=s.rows)
         for n,s in f.slots.items()} for f in history.frames]))


def cached_observation(runtime, previous, actual, observation):
    check(actual['captured'] == previous['captured'],'STATE_CYCLE_NEW_OUTPUT_READING_REQUIRED')
    inputs,model = runtime['inputs'],runtime['model']
    check(tuple(model.specs) == inputs.specs,'STATE_CYCLE_INSTRUMENT_CHANGED')
    source = inputs.observed(runtime['source_observation'],model.config,model.core.initial.device)
    result = Constellation({n:replace(s,rows=(observation,)) for n,s in source.slots.items()},'observed')
    result.validate(model.specs,model.config)
    return result


def feedback(runtime, observed, observation, episode, receipt, context):
    model,view = runtime['model'],runtime['view']
    check(observed.origin == 'observed','STATE_CYCLE_OBSERVED_REQUIRED')
    state,records = runtime['state'],[]
    for phase in (3,4):
        before = state_record(state)
        request = dict(context=context,observation=observation,episode=episode,receipt=receipt,
            phase=phase,origin='execution_receipt',before=digest(before),same_occurrence_reflection=phase==4)
        with torch.no_grad():
            result = feedback_forward(view,observed,state,allowed=runtime['allowed'],phase=phase,
                event=digest(request),history=runtime['history'],origin='execution_receipt')
        state = result['states'][0]
        check(state.sequence == before['sequence']+1 and state.standard_id == before['standard'],
              'STATE_CYCLE_FEEDBACK_PHASE')
        records.append(dict(request=request,before=before,after=state_record(state),
            observed=tensor_fingerprint({n:s.values for n,s in observed.slots.items()}),
            error=discrepancy(runtime['prediction'],observed,model)))
    check(fingerprint(model,runtime['optimizer']) == runtime['frozen'],'STATE_CYCLE_WEIGHTS_CHANGED')
    return state,records
