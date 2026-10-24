"""Activate repository kernels on the retained model and replay their durable state."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit.fsv_identity_loop import configure
from audit.repository_demonstration_memory import authenticate
from audit.repository_kernel_open import open_memory
from audit.repository_relation_learning import retained
from audit.state_cycle_open import prepare
from claims.codata_source import check,sha
from claims.episodic_memory import digest
from claims.process_conditions import write_committed
from kernel.repository_population import RepositoryPopulation
from lens.lenscommon import run_cli,take_lease
from senses.contract import SenseError


def read(path):return json.loads(Path(path).read_bytes())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--kernels',type=Path,required=True)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--stage',choices=('warm','cold'),required=True)
    a=parser.parse_args();started=time.perf_counter()
    old=read(a.source/'snapshot-final.json')
    source_config=read(a.source/'config.json')
    recipes=authenticate(source_config)
    witness=read(a.kernels/'witness.json')
    journal_pin=witness['source_journal']
    check(sha((a.source/'experience.sqlite3').read_bytes())==journal_pin,'REPOSITORY_CONNECTION_SOURCE')
    population=RepositoryPopulation(a.source/'experience.sqlite3',journal_pin)
    specs=[dict(path=s['root'],pin=s['pin'],source=s['source']) for s in witness['stages']]
    config=dict(source=str(a.source),source_pin=journal_pin,kernels=str(a.kernels),
                witness=sha((a.kernels/'witness.json').read_bytes()),specs=specs)
    if a.stage=='warm':
        prepare(dict(source=str(a.source)),recipes,a.root)
        write_committed(a.root/'config.json',config)
    else:check(read(a.root/'config.json')==config,'REPOSITORY_CONNECTION_CONFIG')
    gpu=subprocess.check_output(['nvidia-smi'],text=True)
    take_lease('28','repository-kernel-connection-'+a.stage)
    (a.root/('gpu-'+a.stage+'.txt')).write_text(gpu)
    configure();authenticated=time.perf_counter()
    saved=old if a.stage=='warm' else read(a.root/'snapshot-final.json')
    session,sessions=open_memory(recipes,a.root,saved,population)
    restored=time.perf_counter()
    legacy=retained(session,session.model)
    print(json.dumps(dict(stage=a.stage,events=session.journal.count,restored_seconds=restored-authenticated)),flush=True)
    controls=[]
    def refuse(name,code,fn):
        before=digest(session.snapshot())
        try:fn()
        except (ValueError,SenseError) as error:check(code in str(error),'REPOSITORY_CONNECTION_CONTROL:'+str(error))
        else:raise AssertionError('MNLM_REPOSITORY_CONNECTION_ACCEPTED:'+name)
        check(digest(session.snapshot())==before,'REPOSITORY_CONNECTION_CONTROL_MUTATION')
        controls.append(dict(name=name,code=code,before=before,after=before))
    cutoff=max(r['available'] for r in population.records.values())
    queries=[k for k in witness['split']['queries'] if population.records[k]['side']=='input']
    check(len(queries)>=2,'REPOSITORY_CONNECTION_QUERIES')
    if a.stage=='warm':
        for index,spec in enumerate(specs[:3]):
            session.activate_repository_kernel(spec)
            refuse('duplicate_kernel_'+str(index),'REPOSITORY_RUNTIME_DUPLICATE_KERNEL',lambda:
                session.activate_repository_kernel(spec))
            for context,key in zip(('repository:alpha','repository:beta'),queries[:2],strict=True):
                session.read_repository(context,key,kernel=spec['pin'],cutoff=cutoff)
            write_committed(a.root/('snapshot-prefix-'+str(index)+'.json'),session.snapshot())
            print(json.dumps(dict(kernel=index,events=session.journal.count,regions=len(session.repository_runtime.regions))),flush=True)
        refuse('stale_kernel','REPOSITORY_RUNTIME_STALE_KERNEL',lambda:
            session.read_repository('repository:alpha',queries[0],kernel=specs[0]['pin'],cutoff=cutoff))
        echo=next(k for k,r in population.records.items() if r['side']=='echo' and not r['absence'])
        refuse('echo_as_input','REPOSITORY_RUNTIME_INPUT_SIDE',lambda:
            session.read_repository('repository:alpha',echo,kernel=specs[2]['pin'],cutoff=cutoff))
        refuse('future_reference','REPOSITORY_ROUTE_CUTOFF',lambda:
            session.read_repository('repository:alpha',queries[0],kernel=specs[2]['pin'],cutoff='1970-01-01T00:00:00Z'))
        parent=session.active_checkpoint
        session.active_checkpoint='0'*64
        refuse('changed_parent','REPOSITORY_RUNTIME_PARENT_CHANGED',lambda:
            session.read_repository('repository:alpha',queries[0],kernel=specs[2]['pin'],cutoff=cutoff))
        session.active_checkpoint=parent
        runtime=session.repository_runtime
        original=list(runtime.regions);runtime.regions.reverse()
        refuse('altered_region_order','REPOSITORY_RUNTIME_REGION_CHANGED',lambda:
            session.read_repository('repository:alpha',queries[0],kernel=specs[2]['pin'],cutoff=cutoff))
        runtime.regions[:]=original
        session.activate_repository_kernel(specs[3])
        write_committed(a.root/'snapshot-withdrawn.json',session.snapshot())
        refuse('withdrawn_context','REPOSITORY_RUNTIME_REVIEW_REQUIRED',lambda:
            session.read_repository('repository:alpha',queries[0],kernel=specs[3]['pin'],cutoff=cutoff))
        session.reset_repository_context('repository:alpha',reason='reviewed source withdrawal; restart inference from current support')
        session.read_repository('repository:alpha',queries[0],kernel=specs[3]['pin'],cutoff=cutoff)
        check(session.repository_dirty=={'repository:beta'},'REPOSITORY_CONNECTION_CONTEXT_ISOLATION')
        write_committed(a.root/'snapshot-final.json',session.snapshot())
        write_committed(a.root/'legacy.json',legacy)
        write_committed(a.root/'controls.json',controls)
    final=session.snapshot()
    check(final==read(a.root/'snapshot-final.json'),'REPOSITORY_CONNECTION_COLD')
    check(all(final[k]==v for k,v in old.items() if k not in ('head','events')),'REPOSITORY_CONNECTION_OLD_STATE')
    check(retained(session,session.model)==legacy==read(a.root/'legacy.json'),'REPOSITORY_CONNECTION_LEGACY')
    check(retained(session,session.repository_base)==legacy,'REPOSITORY_CONNECTION_REPOSITORY_PARENT')
    operations=time.perf_counter()
    for s in sessions:s.close()
    check(sha((a.source/'experience.sqlite3').read_bytes())==journal_pin,'REPOSITORY_CONNECTION_SOURCE_CHANGED')
    end=time.perf_counter()
    result=dict(stage=a.stage,events=final['events'],head=final['head'],source_events=old['events'],
        snapshot=sha((a.root/'snapshot-final.json').read_bytes()),journal=sha((a.root/'experience.sqlite3').read_bytes()),
        active_repository_kernel=final['repository_runtime']['active']['pin'],old_tool_kernel=final['kernel'],
        controls=controls,source_unchanged=True,cold_replay_equal=True,legacy_retained=True,
        phases=dict(authentication=authenticated-started,restore=restored-authenticated,
            operations=operations-restored,close_and_hash=end-operations),wall_seconds=end-started,
        measured=True,acceptance='author-checked',learned_capability=False,scientific_admission=False,serving=False)
    pin=write_committed(a.root/(a.stage+'-stage.json'),result)
    print(json.dumps(dict(pin=pin,stage=a.stage,events=final['events'],wall_seconds=end-started)),flush=True)


if __name__=='__main__':
    try:run_cli(main)
    except BaseException:
        if '--root' in sys.argv:
            root=Path(sys.argv[sys.argv.index('--root')+1])
            if root.is_dir():write_committed(root/('failure-'+sys.argv[sys.argv.index('--stage')+1]+'.json'),dict(traceback=traceback.format_exc()))
        raise
