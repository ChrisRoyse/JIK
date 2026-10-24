"""Two real intakes automatically publish kernels, with a durable interrupted transition."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit.fsv_identity_loop import configure
from audit.repository_demonstration_memory import authenticate,retained as retained_memory
from audit.repository_live_open import open_memory,recover_updates
from audit.repository_relation_learning import retained
from audit.state_cycle_open import prepare
from claims.codata_source import check,sha
from claims.episodic_memory import digest
from claims.process_conditions import write_committed
from kernel.repository_live_population import LiveRepositoryPopulation
from lens.lenscommon import run_cli,take_lease
from model.repository_live_memory import code_pins
from senses.contract import SenseError


def read(path):return json.loads(Path(path).read_bytes())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--kernels',type=Path,required=True)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--stage',choices=('intake','resume','cold'),required=True)
    a=parser.parse_args();started=time.perf_counter()
    source_config=read(a.source/'config.json');recipes=authenticate(source_config)
    base=Path(source_config['source']);base_pin=source_config['parent_files']['experience.sqlite3']
    old=read(base/'snapshot-final.json');split=read(a.kernels/'witness.json')['split']
    check(len(source_config['cohorts'])==1,'LIVE_WITNESS_COHORT')
    native=source_config['cohorts'][0]['native']
    config=dict(source=str(a.source),source_config=sha((a.source/'config.json').read_bytes()),
        base=str(base),base_pin=base_pin,split=split,code=code_pins())
    if a.stage=='intake':
        prepare(dict(source=str(base)),recipes,a.root);write_committed(a.root/'config.json',config)
    else:check(read(a.root/'config.json')==config,'LIVE_WITNESS_CONFIG')
    population=LiveRepositoryPopulation(base/'experience.sqlite3',base_pin,a.root/'experience.sqlite3')
    gpu=subprocess.check_output(['nvidia-smi'],text=True)
    take_lease('28','repository-live-'+a.stage);(a.root/('gpu-'+a.stage+'.txt')).write_text(gpu)
    configure();authenticated=time.perf_counter()
    saved=old if a.stage=='intake' else read(a.root/'snapshot-pending.json') if a.stage=='resume' else None
    snapshot_guard={'active':a.stage=='cold'}
    forbidden={(a.root/n).resolve() for n in ('snapshot-final.json','snapshot-pending.json')}
    def deny_latest_snapshot(event,args):
        if snapshot_guard['active'] and event=='open' and isinstance(args[0],str):
            check(Path(args[0]).resolve() not in forbidden,'LIVE_WITNESS_LATEST_SNAPSHOT_READ')
    sys.addaudithook(deny_latest_snapshot)
    # Expose the restored pending boundary to the refusal controls before invoking
    # the same recovery hook the normal opener calls automatically.
    session,sessions=open_memory(recipes,a.root,saved,population,recover=a.stage!='resume')
    snapshot_guard['active']=False
    opened=time.perf_counter();legacy=retained(session,session.model);controls=[]
    print(json.dumps(dict(stage=a.stage,restored_events=session.journal.count,seconds=opened-authenticated)),flush=True)
    query=next(k for k in split['queries'] if population.records[k]['side']=='input')
    def refuse(name,code,fn):
        before=digest(session.snapshot())
        try:fn()
        except (ValueError,SenseError) as error:check(code in str(error),'LIVE_WITNESS_CONTROL_CODE:'+str(error))
        else:raise AssertionError('LIVE_WITNESS_ACCEPTED:'+name)
        check(digest(session.snapshot())==before,'LIVE_WITNESS_CONTROL_MUTATION')
        controls.append(dict(name=name,code=code,before=before,after=before))
    def observe():
        cutoff=max(r['available'] for r in population.records.values())
        session.read_repository('repository:continuous',query,kernel=session.repository_runtime.spec['pin'],cutoff=cutoff)
    if a.stage=='intake':
        session.enable_repository_updates(dict(root=str(a.root/'generations'),
            excluded=split['excluded_tasks'],queries=split['queries']))
        observe();write_committed(a.root/'snapshot-initial.json',session.snapshot())
        session.observe_demonstration(dict(**native,partition='learn_1'),synchronize=False)
        write_committed(a.root/'snapshot-after-intake.json',session.snapshot())
        refuse('new_intake_blocks_stale_read','LIVE_KERNEL_UPDATE_PENDING',observe)
        pending=session.synchronize_repository(stop_after_checkpoint=True)
        check(pending['pending'] and session.live_intent is not None,'LIVE_WITNESS_INTERRUPTION')
        refuse('checkpoint_without_activation_blocks_read','LIVE_KERNEL_UPDATE_PENDING',observe)
        refuse('next_intake_while_deriving','LIVE_DERIVATION_PENDING',lambda:
            session.observe_demonstration(dict(**native,partition='learn_2')))
        write_committed(a.root/'interrupted-checkpoint.json',pending)
        write_committed(a.root/'snapshot-pending.json',session.snapshot())
        write_committed(a.root/'legacy.json',legacy)
    elif a.stage=='resume':
        check(session.snapshot()['repository_updates']['pending'],'LIVE_WITNESS_LOST_PENDING')
        refuse('cold_pending_blocks_read','LIVE_KERNEL_UPDATE_PENDING',observe)
        interrupted=read(a.root/'interrupted-checkpoint.json')
        before=session.journal.count;recover_updates(session)
        check(session.journal.count==before+2 and session.live_checkpoint['pin']==interrupted['pin'],
              'LIVE_WITNESS_DUPLICATED_DERIVATION')
        observe();write_committed(a.root/'snapshot-recovered.json',session.snapshot())
        previous=session.repository_runtime.spec['pin'];before=session.journal.count
        body=session.observe_demonstration(dict(**native,partition='learn_2'))
        check(body['kind']=='repository_demonstration_intake' and session.journal.count==before+4 and
              session.repository_runtime.spec['pin']!=previous and session.live_intent is None,
              'LIVE_WITNESS_NOT_AUTOMATIC')
        observe()
        before=digest(session.snapshot());check(session.synchronize_repository() is False,'LIVE_WITNESS_NOOP')
        check(digest(session.snapshot())==before,'LIVE_WITNESS_NOOP_MUTATION')
        refuse('duplicate_intake','DEMONSTRATION_MEMORY_DUPLICATE',lambda:
            session.observe_demonstration(dict(**native,partition='learn_2')))
        cutoff=max(r['available'] for r in population.records.values())
        refuse('old_kernel','REPOSITORY_RUNTIME_STALE_KERNEL',lambda:
            session.read_repository('repository:continuous',query,kernel=previous,cutoff=cutoff))
        write_committed(a.root/'snapshot-final.json',session.snapshot())
    else:
        check(session.snapshot()==read(a.root/'snapshot-final.json'),'LIVE_WITNESS_COLD')
    final=session.snapshot()
    without_runtime={k:v for k,v in final.items() if k not in ('repository_runtime','repository_updates')}
    retained_memory(old,without_runtime)
    check(retained(session,session.model)==legacy==read(a.root/'legacy.json') and
          retained(session,session.repository_base)==legacy,'LIVE_WITNESS_LEGACY')
    for s in sessions:s.close()
    check(sha((base/'experience.sqlite3').read_bytes())==base_pin,'LIVE_WITNESS_PARENT_MUTATED')
    result=dict(stage=a.stage,events=final['events'],head=final['head'],base_events=old['events'],
        groups=len(population.groups),records=len(population.records),pending=final['repository_updates']['pending'],
        latest_snapshot_required_for_restore=saved is not None,
        latest_snapshot_reads_forbidden_during_restore=a.stage=='cold',
        active_kernel=final['repository_runtime']['active'],checkpoint=final['repository_updates']['checkpoint'],
        journal=sha((a.root/'experience.sqlite3').read_bytes()),controls=controls,legacy_retained=True,
        source_unchanged=True,number_status='measured',acceptance='author-checked',scientific_admission=False,
        learned_capability=False,serving=False,phases=dict(authentication=authenticated-started,
            restore=opened-authenticated,operations=time.perf_counter()-opened),wall_seconds=time.perf_counter()-started)
    pin=write_committed(a.root/(a.stage+'-stage.json'),result)
    print(json.dumps(dict(stage=a.stage,pin=pin,events=final['events'],pending=result['pending'],seconds=result['wall_seconds'])),flush=True)


if __name__=='__main__':
    try:run_cli(main)
    except BaseException:
        if '--root' in sys.argv:
            root=Path(sys.argv[sys.argv.index('--root')+1])
            if root.is_dir():write_committed(root/('failure-'+sys.argv[sys.argv.index('--stage')+1]+'.json'),
                                          dict(traceback=traceback.format_exc()))
        raise
