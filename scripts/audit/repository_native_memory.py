"""Exercise native-to-episode bindings and cold replay in the existing journal."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit.fsv_identity_loop import configure
from audit.proposal_recognition import read
from audit.repository_native_open import authenticate,prepare,open_memory
from claims.codata_source import check,sha
from claims.episodic_memory import digest
from claims.process_conditions import write_committed
from claims.repository_population import RepositoryPopulation
from lens.lenscommon import run_cli,take_lease
from model.repository_input import RepositoryNative
from senses.contract import SenseError


def main():
    p = argparse.ArgumentParser(description=__doc__)
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
        check(read(root/'config.json') == config,'REPOSITORY_NATIVE_MEMORY_CONFIG')
    gpu = subprocess.check_output(['nvidia-smi'],text=True)
    take_lease('28','repository-native-memory-'+a.stage)
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
            check(code in str(exc),'REPOSITORY_NATIVE_CONTROL_CODE:'+name+':'+str(exc))
        else:
            raise AssertionError('REPOSITORY_NATIVE_CONTROL_ACCEPTED:'+name)
        check(digest(session.snapshot()) == before,'REPOSITORY_NATIVE_CONTROL_MUTATED')
        controls.append(dict(name=name,code=code))

    spec = config['native']
    if a.stage == 'intake':
        native = RepositoryNative(**spec)
        fields = native.population.fields
        start = fields['base']['input_available_by']
        refuse('future_echo','REPOSITORY_NATIVE_FUTURE',
            lambda:native.materialize('base','echo',start))
        seconds,fraction = fields['base']['echo_available_after'][:-1].split('.')
        just_before = seconds+'.'+str(int(fraction.ljust(9,'0'))-1).zfill(9)+'Z'
        refuse('echo_one_nanosecond_early','REPOSITORY_NATIVE_FUTURE',
            lambda:native.materialize('base','echo',just_before))
        refuse('swapped_condition','REPOSITORY_NATIVE_CONDITION_FIELDS',
            lambda:native.materialize('base','input',start,fields=fields['reference']['input']))
        refuse('unadmitted_promotion','REPOSITORY_NATIVE_UNADMITTED',
            lambda:native.materialize('base','input',start,scientific_admission=True))
        saved_arrays = native.arrays
        native.arrays = copy.deepcopy(saved_arrays)
        name = next(iter(native.arrays))
        native.arrays[name][0].flags.writeable = True
        native.arrays[name][0][0,0] += 0.125
        refuse('changed_memory_vector','REPOSITORY_NATIVE_MEMORY_CHANGED',native.fresh)
        native.arrays = saved_arrays
        with tempfile.TemporaryDirectory(prefix='repository-native-control-',dir=root.parent) as tmp:
            pop = Path(tmp)/'population'
            shutil.copytree(native.population.root,pop)
            raw = pop/'repository/chunks/atoms.parquet'
            raw.write_bytes(raw.read_bytes()+b'changed')
            refuse('changed_persisted_source','REPOSITORY_POPULATION_BYTES',
                lambda:RepositoryPopulation(pop,native.population.pin))
            copied = Path(tmp)/'native'
            shutil.copytree(native.native,copied)
            raw = copied/name/'slots.npz'
            raw.write_bytes(raw.read_bytes()+b'changed')
            original = native.native
            native.native = copied
            refuse('changed_persisted_vector','REPOSITORY_NATIVE_BYTES_CHANGED',native.fresh)
            native.native = original
        body = session.bind_repository_native(spec)
        refuse('duplicate_binding','REPOSITORY_NATIVE_DUPLICATE',lambda:session.bind_repository_native(spec))
        write_committed(root/'intake.json',dict(event=body,payload=session.journal.payload(body['payload'])))
        write_committed(root/'snapshot-final.json',session.snapshot())
    final = session.snapshot()
    check(final == read(root/'snapshot-final.json'),'REPOSITORY_NATIVE_COLD_REPLAY')
    check(all(final[k] == v for k,v in old.items() if k not in ('head','events')),
          'REPOSITORY_NATIVE_OLD_STATE_CHANGED')
    check(final['events'] == old['events']+1 and
          set(final)-set(old) == {'repository_native_bindings'},'REPOSITORY_NATIVE_DELTA')
    bindings = final['repository_native_bindings'][digest(spec)]['bindings']
    check(len(bindings) == 3,'REPOSITORY_NATIVE_EPISODE_COUNT')
    result = dict(stage=a.stage,head=session.head,events=session.journal.count,
        source_head=old['head'],source_events=old['events'],controls=controls,
        old_state_unchanged=True,replay_equal=True,
        snapshot=sha((root/'snapshot-final.json').read_bytes()),
        native_measured=True,learned_repair=False,scientific_admission=False,serving=False)
    for opened_session in sessions:
        opened_session.close()
    finished = time.perf_counter()
    result.update(wall_seconds=finished-started,phases=dict(authentication=authenticated-started,
        existing_lineage_replay=opened-authenticated,intake_and_readback=finished-opened))
    write_committed(root/(a.stage+'-stage.json'),result)
    print(json.dumps(result,sort_keys=True),flush=True)


if __name__ == '__main__':
    run_cli(main)
