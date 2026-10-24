"""Real captured-output Echo/Correction, pending-state restart and cold replay."""
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
from claims.codata_source import check, sha
from claims.episodic_memory import digest
from claims.process_conditions import write_committed
from lens.lenscommon import run_cli, take_lease
from audit.next_echo_open import authenticate, prepare, open_echo
from audit.proposal_recognition import read
from senses.contract import SenseError


def original(value):
    return {k:v for k,v in value.items() if k not in
            ('head','events','native_feedback_contexts','native_feedback_enrollments')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--stage',choices=('begin','finish','cold'),required=True)
    a = parser.parse_args()
    started = time.perf_counter()
    config = json.loads(a.config.read_bytes())
    root, source = Path(config['root']),Path(config['source'])
    prior_config,previous = authenticate(config)
    before = read(source/'snapshot-final.json')
    if a.stage == 'begin':
        prepare(config,previous,root)
        write_committed(root/'config.json',config)
    else:
        check(json.loads((root/'config.json').read_bytes()) == config,'PROPOSAL_ECHO_CONFIG')
    gpu = subprocess.check_output(['nvidia-smi']).decode()
    take_lease('28','next-echo-'+a.stage)
    (root/('gpu-'+a.stage+'.txt')).open('x').write(gpu)
    configure()
    saved = before if a.stage == 'begin' else json.loads((root/(
        'snapshot-begin.json' if a.stage == 'finish' else 'snapshot-final.json')).read_bytes())
    session,sessions = open_echo(config,previous,root,saved)
    replayed = time.perf_counter()
    print(json.dumps(dict(stage=a.stage,replayed_events=session.journal.count)),flush=True)
    controls, transitions, isolation = [], [], []

    def refuse(name, code, fn):
        prior = digest(session.snapshot())
        try:
            fn()
        except (ValueError,SenseError) as error:
            check(code in str(error),'PROPOSAL_ECHO_CONTROL_CODE:'+name+':'+str(error))
        else:
            raise AssertionError('PROPOSAL_ECHO_CONTROL_ACCEPTED:'+name)
        check(digest(session.snapshot()) == prior,'PROPOSAL_ECHO_CONTROL_MUTATED')
        controls.append(dict(name=name,code=code))

    def advance(context, phase):
        old = copy.deepcopy(session.echo_contexts)
        event = session.advance_echo(context,old[context]['observation'],phase)
        check(all(v == session.echo_contexts[k] for k,v in old.items() if k != context),
              'PROPOSAL_ECHO_CONTEXT_INTERFERENCE')
        isolation.append(dict(changed=context,untouched={k:digest(v) for k,v in old.items() if k != context}))
        transitions.append(event)

    if a.stage == 'begin':
        experiment, = session.next_proposals
        spec = dict(experiment=experiment,inputs=config['inputs'])
        event = session.enroll_next_echo(spec)
        write_committed(root/'enrollment.json',dict(body=event,payload=session.journal.payload(event['payload'])))
        refuse('duplicate_enrollment','PROPOSAL_ECHO_ENROLLMENT_STATE',lambda:session.enroll_next_echo(spec))
        contexts = sorted(k for k in session.echo_contexts if k.startswith('next:'))
        first = contexts[0]
        observation = session.echo_contexts[first]['observation']
        other = next(v['observation'] for k,v in session.echo_contexts.items()
                     if k in contexts and v['observation'] != observation)
        refuse('swapped_occurrence','PROPOSAL_ECHO_OCCURRENCE',lambda:session.advance_echo(first,other,3))
        refuse('correction_before_echo','FIELD_ECHO_PHASE_ORDER',lambda:session.advance_echo(first,observation,4))
        refuse('generated_origin','FIELD_ECHO_CAPTURED_REQUIRED',
               lambda:session.advance_echo(first,observation,3,origin='generated_proposal'))
        runtime = session.echo_runtime[first]
        own_view = runtime['view']
        try:
            other_context = next(k for k in contexts
                                 if session.echo_contexts[k]['variant'] != session.echo_contexts[first]['variant'])
            runtime['view'] = session.echo_runtime[other_context]['view']
            refuse('other_checkpoint_view','PROPOSAL_ECHO_MODEL_OWNER',
                   lambda:session.advance_echo(first,observation,3))
        finally:
            runtime['view'] = own_view
        real = runtime['observed']
        try:
            runtime['observed'] = replace(real,origin='predicted')
            refuse('predicted_as_observed','PROPOSAL_ECHO_OBSERVED_REQUIRED',
                   lambda:session.advance_echo(first,observation,3))
            slots = dict(real.slots)
            name = next(iter(slots))
            changed = slots[name].values.clone()
            changed[0,0,0] += .001
            slots[name] = replace(slots[name],values=changed)
            runtime['observed'] = replace(real,slots=slots)
            refuse('changed_feedback_window','PROPOSAL_ECHO_NATIVE_CHANGED',
                   lambda:session.advance_echo(first,observation,3))
        finally:
            runtime['observed'] = real
        advance(first,3)
        refuse('repeated_echo','FIELD_ECHO_PHASE_ORDER',lambda:session.advance_echo(first,observation,3))
        write_committed(root/'snapshot-begin.json',session.snapshot())
    elif a.stage == 'finish':
        contexts = sorted(k for k in session.echo_contexts if k.startswith('next:'))
        check(len(contexts) == 4 and session.echo_contexts[contexts[0]]['status'] == 'awaiting_correction'
              and all(session.echo_contexts[k]['status'] == 'awaiting_echo' for k in contexts[1:]),
              'PROPOSAL_ECHO_PENDING_RESTART')
        for context in contexts[1:]:
            advance(context,3)
        for context in contexts:
            advance(context,4)
        refuse('repeated_correction','FIELD_ECHO_PHASE_ORDER',
               lambda:session.advance_echo(context,session.echo_contexts[context]['observation'],4))
        write_committed(root/'snapshot-final.json',session.snapshot())
    else:
        check(all(v['status'] == 'corrected_state' for v in session.echo_contexts.values()),
              'PROPOSAL_ECHO_FINAL_REPLAY')
    final = session.snapshot()
    check(original(final) == original(before),'PROPOSAL_ECHO_EXISTING_STATE_CHANGED')
    check(all(final['native_feedback_contexts'][k] == v for k,v in before['native_feedback_contexts'].items()),
          'PROPOSAL_ECHO_OLD_CONTEXT_CHANGED')
    for opened in sessions:
        opened.close()
    finished = time.perf_counter()
    report = dict(stage=a.stage,head=final['head'],events=final['events'],snapshot=digest(final),
        controls=controls,transitions=transitions,isolation=isolation,
        phases=dict(authentication_replay=replayed-started,feedback_and_close=finished-replayed),
        wall_seconds=finished-started,source_journal=sha((source/'experience.sqlite3').read_bytes()),
        journal=sha((root/'experience.sqlite3').read_bytes()),
        scientific_admission=False,serving=False,learned_understanding=False,new_weight_update=False)
    write_committed(root/(a.stage+'-report.json'),report)
    print(json.dumps({k:report[k] for k in ('stage','head','events','snapshot','phases','wall_seconds')}),flush=True)


if __name__ == '__main__':
    run_cli(main)
