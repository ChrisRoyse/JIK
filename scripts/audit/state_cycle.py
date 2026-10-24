"""Continue corrected state, seal predictions, execute, correct and cold-replay."""
import argparse
import copy
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit.fsv_identity_loop import configure
from audit.state_cycle_open import authenticate,prepare,open_echo
from audit.proposal_recognition import read
from claims.codata_source import check,sha
from claims.episodic_memory import digest
from claims.process_conditions import write_committed
from claims import state_cycle_forecast
from lens.lenscommon import run_cli,take_lease
from model.attempt_input import AttemptNative
from model import state_cycle_steps as steps
from oracle.state_cycle import run as execute
from senses.contract import SenseError


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--stage',choices=('forecast','restart','execute','intake','cold'),required=True)
    a=parser.parse_args()
    started=time.perf_counter()
    config=read(a.config)
    root,source=Path(config['root']),Path(config['source'])
    _,recipes=authenticate(config)
    source_snapshot=read(source/'snapshot-final.json')
    if a.stage=='forecast':
        prepare(config,recipes,root)
        write_committed(root/'config.json',config)
    else:
        check(read(root/'config.json')==config,'STATE_CYCLE_CONFIG_CHANGED')
    if a.stage=='execute':
        pin=(root/'forecast/COMMITTED').read_text()
        observed=execute(root/'execution',root/'forecast',pin)
        result=dict(observation_pin=observed,wall_seconds=time.perf_counter()-started)
        write_committed(root/'execute-stage.json',result)
        print(json.dumps(result),flush=True)
        return
    if a.stage in ('forecast','restart'):
        check(not (root/'execution').exists(),'STATE_CYCLE_FORECAST_LATE')
    gpu=subprocess.check_output(['nvidia-smi'],text=True)
    take_lease('28','state-cycle-'+a.stage)
    (root/('gpu-'+a.stage+'.txt')).open('x').write(gpu)
    configure()
    saved=(source_snapshot if a.stage=='forecast' else
           read(root/('snapshot-final.json' if a.stage=='cold' else 'snapshot-forecast.json')))
    session,sessions=open_echo(config,recipes,root,saved)
    replayed=time.perf_counter()
    print(json.dumps(dict(stage=a.stage,replayed_events=session.journal.count)),flush=True)
    controls=[]
    def refuse(name,code,fn):
        before=digest(session.snapshot())
        try:
            fn()
        except (ValueError,SenseError) as error:
            check(code in str(error),'STATE_CYCLE_CONTROL_CODE:'+name+':'+str(error))
        else:
            raise AssertionError('STATE_CYCLE_CONTROL_ACCEPTED:'+name)
        check(digest(session.snapshot())==before,'STATE_CYCLE_CONTROL_MUTATED')
        controls.append(dict(name=name,code=code))
    spec=dict(config['experiment'],execution=str(root/'execution'))
    if a.stage=='forecast':
        bad=dict(spec,contexts=list(reversed(spec['contexts'])))
        refuse('wrong_context_roster','STATE_CYCLE_CONTEXT_ROSTER',lambda:session.forecast_cycle(bad))
        first=spec['contexts'][0]
        entry,runtime=steps.checked(session,first)
        state,history=runtime['state'],runtime['history']
        try:
            runtime['state']=replace(state,hidden=state.hidden+.001)
            refuse('changed_source_state','STATE_CYCLE_STATE_CHANGED',lambda:session.forecast_cycle(spec))
            runtime['state']=state
            runtime['history']=replace(history,context='wrong-context')
            refuse('wrong_history_context','STATE_CYCLE_HISTORY_CONTEXT',lambda:session.forecast_cycle(spec))
        finally:
            runtime['state'],runtime['history']=state,history
        old_forecast=session.attempt_proposals[spec['source_experiment']]['forecast']
        native=AttemptNative(**old_forecast['spec']['native'])
        batch=native.batch(entry['condition'],runtime['model'].config,session.device,runtime['model'].specs)
        refuse('skipped_standard_and_claim','MNLM_LOOP_PHASE_ORDER',lambda:runtime['view'](
            batch,(state,),allowed=runtime['allowed'],phase=2,event_ids=['0'*64],
            evidence_origin=['generated_proposal'],history=history))
        actual=runtime['inputs'].observations.read(entry['observation'])
        changed=copy.deepcopy(actual)
        changed['captured']['stderr']+='changed'
        rr=dict(runtime,source_observation=entry['observation'])
        refuse('changed_output_requires_measurement','STATE_CYCLE_NEW_OUTPUT_READING_REQUIRED',
               lambda:steps.cached_observation(rr,actual,changed,'0'*64))
        observed=runtime['inputs'].observed(entry['observation'],runtime['model'].config,session.device)
        refuse('predicted_feedback','STATE_CYCLE_OBSERVED_REQUIRED',lambda:steps.feedback(
            runtime,replace(observed,origin='predicted'),'0'*64,entry['episode'],entry['receipt'],first))
        body=session.forecast_cycle(spec)
        refuse('duplicate_forecast','STATE_CYCLE_ALREADY_STARTED',lambda:session.forecast_cycle(spec))
        pin=state_cycle_forecast.publish(session,root/'forecast',body)
        write_committed(root/'snapshot-forecast.json',session.snapshot())
        result=dict(forecast=pin,body=body)
    elif a.stage=='intake':
        pin=(root/'forecast/COMMITTED').read_text()
        before=state_cycle_forecast.read(root/'forecast',pin)
        observation_pin=(root/'execution/COMMITTED').read_text()
        body=session.observe_cycle(root/'execution',observation_pin,root/'forecast',pin)
        refuse('duplicate_feedback','STATE_CYCLE_NOT_PENDING',
               lambda:session.observe_cycle(root/'execution',observation_pin,root/'forecast',pin))
        check(state_cycle_forecast.read(root/'forecast',pin)==before,'STATE_CYCLE_REWROTE_FORECAST')
        write_committed(root/'snapshot-final.json',session.snapshot())
        write_committed(root/'feedback.json',session.journal.payload(body['payload']))
        result=dict(forecast=pin,body=body,forecast_unchanged=True)
    else:
        check(session.snapshot()==saved,'STATE_CYCLE_COLD_REPLAY')
        pin=(root/'forecast/COMMITTED').read_text()
        state_cycle_forecast.read(root/'forecast',pin)
        result=dict(forecast=pin,snapshot=digest(saved),cold_replay=True,executed=a.stage!='restart')
    final=session.snapshot()
    check(final['native_feedback_contexts']==source_snapshot['native_feedback_contexts'] and
          final['native_feedback_enrollments']==source_snapshot['native_feedback_enrollments'],
          'STATE_CYCLE_OLD_CONTEXT_CHANGED')
    for context in spec['contexts']:
        steps.checked(session,context)
    result.update(stage=a.stage,head=session.head,events=session.journal.count,controls=controls,
                  old_contexts_unchanged=True)
    for opened in sessions:
        opened.close()
    finished=time.perf_counter()
    result.update(phases=dict(authentication_replay=replayed-started,action_close=finished-replayed),
        wall_seconds=finished-started,journal=sha((root/'experience.sqlite3').read_bytes()),
        source_journal=sha((source/'experience.sqlite3').read_bytes()),
        code=sha(Path(__file__).read_bytes()),serving=False,scientific_admission=False,acceptance='author-checked')
    write_committed(root/('forecast-restart.json' if a.stage=='restart' else a.stage+'-stage.json'),result)
    print(json.dumps({k:v for k,v in result.items() if k!='body'}),flush=True)


if __name__=='__main__':
    run_cli(main)
