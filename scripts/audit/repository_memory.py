"""Persist task-oracle occurrences in the live lineage and reproduce them after restart."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit.fsv_identity_loop import configure
from audit.proposal_recognition import read
from audit.repository_memory_open import authenticate,prepare,open_memory
from claims.codata_source import check,sha
from claims.episodic_memory import digest
from claims.process_conditions import write_committed
from claims.repository_observation import RepositoryObservations
from lens.lenscommon import run_cli,take_lease
from senses.contract import SenseError


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--stage',choices=('intake','cold'),required=True)
    a = p.parse_args()
    started = time.perf_counter()
    config = read(a.config)
    root,parent = Path(config['root']),Path(config['source'])
    recipes = authenticate(config)
    old = read(parent/'snapshot-final.json')
    if a.stage == 'intake':
        prepare(config,recipes,root)
        write_committed(root/'config.json',config)
    else:
        check(read(root/'config.json') == config,'REPOSITORY_MEMORY_CONFIG')
    gpu = subprocess.check_output(['nvidia-smi'],text=True)
    take_lease('28','repository-memory-'+a.stage)
    (root/('gpu-'+a.stage+'.txt')).open('x').write(gpu)
    configure()
    saved = old if a.stage == 'intake' else read(root/'snapshot-final.json')
    authenticated = time.perf_counter()
    session,sessions = open_memory(recipes,root,saved)
    opened = time.perf_counter()
    print(json.dumps(dict(stage=a.stage,replayed_events=session.journal.count)),flush=True)
    controls = []

    def refuse(name,code,action):
        before = digest(session.snapshot())
        try:
            action()
        except (SenseError,ValueError) as exc:
            check(code in str(exc),'REPOSITORY_MEMORY_CONTROL_CODE:'+name)
        else:
            raise AssertionError('REPOSITORY_MEMORY_CONTROL_ACCEPTED:'+name)
        check(digest(session.snapshot()) == before,'REPOSITORY_MEMORY_CONTROL_MUTATED')
        controls.append(dict(name=name,code=code))

    path,pin = Path(config['observations']),config['observation_pin']
    if a.stage == 'intake':
        refuse('wrong_seal','MNLM_UCR_INPUT_DIGEST',lambda:session.observe_repository(path,'0'*64))
        ob = RepositoryObservations(path,pin)
        for label,edit in (
            ('forged_success',lambda e:e['content'].update(outcome=1)),
            ('invented_clock',lambda e:e.update(event_time='2000-01-01T00:00:00Z')),
            ('model_authorship',lambda e:e.update(model_generated_action=True)),
            ('unbounded_truth',lambda e:e.update(scope='general_task_correctness'))):
            bad = copy.deepcopy(ob.episode('base'))
            edit(bad)
            bad['id'] = digest({k:v for k,v in bad.items() if k != 'id'})
            refuse(label,'REPOSITORY_EPISODE',lambda:ob.validate_episode(bad))
        body = session.observe_repository(path,pin)
        refuse('duplicate_occurrences','REPOSITORY_MEMORY_DUPLICATE',
               lambda:session.observe_repository(path,pin))
        write_committed(root/'intake.json',dict(event=body,payload=session.journal.payload(body['payload'])))
        write_committed(root/'snapshot-final.json',session.snapshot())
    final = session.snapshot()
    check(final == read(root/'snapshot-final.json'),'REPOSITORY_MEMORY_COLD_REPLAY')
    unaffected = {k:v for k,v in old.items() if k not in ('head','events','memory')}
    check(all(final[k] == v for k,v in unaffected.items()),'REPOSITORY_MEMORY_OLD_CONTEXT')
    check(all(final['memory']['states'][k] == v for k,v in old['memory']['states'].items()),
          'REPOSITORY_MEMORY_OLD_EPISODE')
    added = sorted(set(final['memory']['grounded_world_claims'])-set(old['memory']['grounded_world_claims']))
    expected = sorted(final['repository_occurrences'][pin]['episodes'].values())
    check(added == expected and len(added) == 3 and final['events'] == old['events']+1,
          'REPOSITORY_MEMORY_COVERAGE')
    result = dict(stage=a.stage,head=session.head,events=session.journal.count,added=added,
        source_head=old['head'],source_events=old['events'],controls=controls,
        old_contexts_and_models_unchanged=True,old_episodes_unchanged=True,
        native_measured=False,learned_skill=False,serving=False,scientific_admission=False,
        snapshot=sha((root/'snapshot-final.json').read_bytes()),
        replay_equal=True,acceptance='author-checked only')
    for opened_session in sessions:
        opened_session.close()
    finished = time.perf_counter()
    result.update(wall_seconds=finished-started,phases=dict(authentication=authenticated-started,
        existing_lineage_replay=opened-authenticated,intake_and_readback=finished-opened))
    write_committed(root/(a.stage+'-stage.json'),result)
    print(json.dumps(result,sort_keys=True),flush=True)


if __name__ == '__main__':
    run_cli(main)
