"""Context revision -> one automatic kernel update -> actual prediction loss -> cold recovery."""
import argparse
from functools import partial
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
from audit.capability_delta_live_fsv import checkpoint
from audit.capability_live_fsv import refuse, checkpoint_roster
from audit.capability_prediction_fsv import run as prediction_check
from audit.fsv_identity_loop import configure
from audit.repository_demonstration_memory import authenticate
from audit.repository_live_bootstrap import open_current
from audit.repository_memory_readback import chain
from audit.state_cycle_open import prepare
from claims.capability_population import need,sha
from claims.process_conditions import write_committed
from kernel.repository_live_population import LiveRepositoryPopulation
from lens.lenscommon import run_cli,take_lease
from model.capability_context_memory import ContextCapabilityMemoryLoop,code_pins
from model.tool_growth import parameters


def read(path): return json.loads(Path(path).read_bytes())


def source_spec(config):
    return dict(**config['native_source'],adapter=dict(schema='capability-context-native-v1',
        code=code_pins(),registry=config['registry'],native_proof=config['native_proof']))


def ancestry(parent):
    seen=set(); path=parent
    while True:
        need(str(path) not in seen,'CONTEXT_FSV_ANCESTRY_CYCLE'); seen.add(str(path))
        config=read(path/'config.json')
        if 'base' in config and 'base_pin' in config:
            return path,config,authenticate(read(Path(config['source'])/'config.json'))
        path=Path(config['source'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--stage',choices=('update','cold'),required=True)
    a=p.parse_args(); config=read(a.config); root,parent=Path(config['root']),Path(config['source'])
    need(sha((parent/'experience.sqlite3').read_bytes())==config['source_pin'],'CONTEXT_FSV_PARENT')
    repository,live,recipes=ancestry(parent)
    if a.stage=='update':
        prepare(config,recipes,root); write_committed(root/'config.json',config)
    else: need(read(root/'config.json')==config,'CONTEXT_FSV_CONFIG')
    gpu=subprocess.check_output(['nvidia-smi'],text=True); take_lease('29','capability-context-'+a.stage)
    (root/('gpu-'+a.stage+'.txt')).write_text(gpu); configure(); start=time.perf_counter()
    population=LiveRepositoryPopulation(Path(live['base'])/'experience.sqlite3',live['base_pin'],repository/'experience.sqlite3')
    session,sessions=open_current(recipes,root,population,
        loop_type=partial(ContextCapabilityMemoryLoop,recover_on_open=False))
    restored=time.perf_counter()
    print(json.dumps(dict(stage=a.stage,restored=True,events=session.journal.count,seconds=restored-start)),flush=True)
    try:
        initial_parameters=parameters(session.model); controls=[]; spec=source_spec(config)
        registry=read(config['registry']['path']); split=registry['splits'][spec['population_pin']]
        if a.stage=='update':
            baseline=checkpoint(session); write_committed(root/'baseline.json',baseline)
            need(session.growth_routing and session.task_intent is None,'CONTEXT_FSV_BASELINE')
            session.relocate_task_outputs(str(root/'task-kernels'))
            session.register_task_source(spec,split)
            evidence=session._capability_source(spec); parent_pin=evidence.population.parent.pin
            prior_queries=session._task_population().queries()
            with session.task_batch():
                intake=session.observe_context(spec)
                selection=session.select_context_revision(spec['population_pin'])
                controls.append(refuse(session,'TASK_LIVE_UPDATE_PENDING',lambda:
                    session.read_task(prior_queries[0],kernel=baseline['checkpoint']['kernel']['pin'])))
            generation=checkpoint(session); write_committed(root/'generation.json',generation)
            need(generation['training_tasks']==baseline['training_tasks']
                 and generation['query_tasks']==baseline['query_tasks'],'CONTEXT_FSV_DUPLICATED_EXPERIENCE')
            active=session._task_population()
            need(not any(t['population']==parent_pin for t in (session.capability_tasks[k] for k in active.eligible)),
                 'CONTEXT_FSV_OLD_VIEW_STILL_ACTIVE')
            payload=session.journal.payload(intake['payload'])
            need(len(payload['excluded'])==1 and len(payload['tasks'])==121,'CONTEXT_FSV_WITHDRAWAL')
            need(not set(payload['excluded'])&active.records.keys(),'CONTEXT_FSV_RESURRECTION')
            controls.append(refuse(session,'TASK_LIVE_STALE_KERNEL',lambda:
                session.read_task(split['queries'][0],kernel=baseline['checkpoint']['kernel']['pin'])))
            write_committed(root/'replacement.json',dict(intake=intake,selection=selection,
                original_population=parent_pin,replacement_population=spec['population_pin'],
                excluded=payload['excluded'],admitted_tasks=len(payload['tasks']),
                training_count_unchanged=True,query_count_unchanged=True))
            dest=root/'prediction'; dest.mkdir()
            key=max(split['queries'],key=lambda k:max(len(v) for v in active.values[k].values()))
            result=prediction_check(session,[dict(family='coding',key=key,source='repository-context')],dest)
            write_committed(dest/'report.json',result)
            write_committed(root/'snapshot-final.json',session.snapshot())
        else:
            need(session.snapshot()==read(root/'snapshot-final.json'),'CONTEXT_FSV_COLD_STATE')
            head,count=session.head,session.journal.count; before=checkpoint_roster(root)
            need(session.synchronize_tasks() is False and session.head==head and session.journal.count==count
                 and checkpoint_roster(root)==before,'CONTEXT_FSV_REPEATED_WORK')
            generation=checkpoint(session)
            need(session.read_task(split['queries'][0],kernel=session.task_checkpoint['kernel']['pin'])['targets'],
                 'CONTEXT_FSV_COLD_RETRIEVAL')
        old_contract,prefix=chain(parent/'experience.sqlite3'); contract,rows=chain(root/'experience.sqlite3')
        need(contract==old_contract and rows[:len(prefix)]==prefix
             and sha((parent/'experience.sqlite3').read_bytes())==config['source_pin'],'CONTEXT_FSV_PREFIX')
        need(parameters(session.model)==initial_parameters,'CONTEXT_FSV_PARAMETERS')
        report=dict(schema='capability-context-live-fsv-v1',stage=a.stage,complete=True,
            head=session.head,events=session.journal.count,journal=sha((root/'experience.sqlite3').read_bytes()),
            generation=generation,controls=controls,source_prefix_unchanged=True,parameters_unchanged=True,
            optimizer_steps=0,phases_seconds=dict(restore=restored-start,work=time.perf_counter()-restored),
            generated_surface_guard_complete=False,scientific_admission=False,learned_capability=False,
            acceptance='author-checked',number_status='measured',producer_sha256=sha(Path(__file__).read_bytes()))
        pin=write_committed(root/(a.stage+'-report.json'),report)
        print(json.dumps(dict(report=pin,stage=a.stage,events=report['events'],recall=generation['minimum_recall'])),flush=True)
    except BaseException as exc:
        write_committed(root/(a.stage+'-failure.json'),dict(error=str(exc),trace=traceback.format_exc()))
        raise
    finally:
        for item in sessions: item.close()


if __name__=='__main__': run_cli(main)
