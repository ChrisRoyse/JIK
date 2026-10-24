"""Recover a retained recall failure through a versioned native-window query route."""
import argparse
from functools import partial
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1] / 'lens')]
from audit.capability_delta_live_fsv import checkpoint, sources
from audit.capability_live_fsv import refuse, checkpoint_roster
from audit.capability_prediction_fsv import run as prediction_check
from audit.fsv_identity_loop import configure
from audit.repository_demonstration_memory import authenticate
from audit.repository_live_bootstrap import open_current
from audit.repository_memory_readback import chain
from audit.state_cycle_open import prepare
from claims.capability_population import need, sha
from claims.process_conditions import write_committed
from kernel.repository_live_population import LiveRepositoryPopulation
from lens.lenscommon import run_cli, take_lease
from model.capability_growth_memory import GrowthCapabilityMemoryLoop
from model.tool_growth import parameters


def read(path): return json.loads(Path(path).read_bytes())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--stage', choices=('update', 'cold'), required=True)
    a = p.parse_args(); config = read(a.config)
    root, parent = Path(config['root']), Path(config['source'])
    need(sha((parent / 'experience.sqlite3').read_bytes()) == config['source_pin'], 'GROWTH_FSV_PARENT')
    delta = read(parent / 'config.json')
    original = read(Path(delta['source']) / 'config.json')
    memory = read(Path(original['source']) / 'config.json')
    repository = Path(memory['source']); live = read(repository / 'config.json')
    recipes = authenticate(read(Path(live['source']) / 'config.json'))
    if a.stage == 'update':
        prepare(config, recipes, root); write_committed(root / 'config.json', config)
    else: need(read(root / 'config.json') == config, 'GROWTH_FSV_CONFIG')
    gpu = subprocess.check_output(['nvidia-smi'], text=True)
    take_lease('29', 'capability-growth-live-'+a.stage)
    (root / ('gpu-'+a.stage+'.txt')).write_text(gpu)
    configure(); started = time.perf_counter()
    population = LiveRepositoryPopulation(Path(live['base']) / 'experience.sqlite3', live['base_pin'],
                                          repository / 'experience.sqlite3')
    session, sessions = open_current(recipes, root, population,
        loop_type=partial(GrowthCapabilityMemoryLoop, recover_on_open=False))
    restored = time.perf_counter()
    print(json.dumps(dict(stage=a.stage, restored=True, events=session.journal.count,
                          seconds=restored-started, pending=session.task_intent is not None)), flush=True)
    try:
        before_parameters = parameters(session.model); controls = []
        if a.stage == 'update':
            need(session.task_intent is not None and not session.growth_routing, 'GROWTH_FSV_FAILED_INTENT')
            write_committed(root / 'before.json', dict(head=session.head, events=session.journal.count,
                intent=session.task_intent, checkpoint=session.task_checkpoint))
            previous = session.task_checkpoint['kernel']['pin']
            session.enable_growth_routing(str(root / 'task-kernels'))
            query = session._task_population().queries()[0]
            controls.append(refuse(session, 'TASK_LIVE_UPDATE_PENDING', lambda: session.read_task(query, kernel=previous)))
            session.synchronize_tasks()
            controls.append(refuse(session, 'TASK_LIVE_STALE_KERNEL', lambda: session.read_task(query, kernel=previous)))
            generation = checkpoint(session)
            write_committed(root / 'generation.json', generation)
            print(json.dumps(dict(training=generation['training_tasks'], queries=generation['query_tasks'],
                                  recall=generation['minimum_recall'])), flush=True)
            destination = root / 'prediction'; destination.mkdir()
            registry = read(delta['registry']['path']); cases = []
            for name, spec in sources(delta).items():
                pop = session._capability_source(spec).population
                queries = registry['splits'][pop.pin]['queries']
                unknown = [k for k in queries if any(pop.fields[f]['text'] == "DON'T KNOW" for f in pop.tasks[k]['targets'])]
                key = unknown[0] if unknown else max(queries,
                    key=lambda k: sum(len(pop.fields[f]['text']) for f in pop.tasks[k]['targets']))
                cases.append(dict(family=pop.family, key=key, source=name))
            predictions = prediction_check(session, cases, destination)
            write_committed(destination / 'report.json', predictions)
            write_committed(root / 'snapshot-final.json', session.snapshot())
        else:
            need(session.snapshot() == read(root / 'snapshot-final.json'), 'GROWTH_FSV_COLD_STATE')
            before = checkpoint_roster(root); head, count = session.head, session.journal.count
            need(session.synchronize_tasks() is False and session.head == head and session.journal.count == count
                 and checkpoint_roster(root) == before, 'GROWTH_FSV_REPEATED_WORK')
            generation = checkpoint(session)
            registry = read(delta['registry']['path'])
            for spec in sources(delta).values():
                key = registry['splits'][spec['population_pin']]['queries'][0]
                need(session.read_task(key, kernel=session.task_checkpoint['kernel']['pin'])['targets'],
                     'GROWTH_FSV_COLD_RETRIEVAL')
        contract, rows = chain(root / 'experience.sqlite3'); original_contract, prefix = chain(parent / 'experience.sqlite3')
        need(contract == original_contract and rows[:len(prefix)] == prefix
             and sha((parent / 'experience.sqlite3').read_bytes()) == config['source_pin'], 'GROWTH_FSV_PREFIX')
        need(parameters(session.model) == before_parameters, 'GROWTH_FSV_PARAMETERS')
        report = dict(schema='capability-growth-live-fsv-v1', stage=a.stage, complete=True,
            head=session.head, events=session.journal.count, journal=sha((root / 'experience.sqlite3').read_bytes()),
            generation=generation, controls=controls, source_prefix_unchanged=True,
            native_stores=len(session.delta_stores), parameters_unchanged=True, optimizer_steps=0,
            phases_seconds=dict(restore=restored-started, work=time.perf_counter()-restored),
            generated_surface_guard_complete=False, scientific_admission=False, learned_capability=False,
            acceptance='author-checked', number_status='measured', producer_sha256=sha(Path(__file__).read_bytes()))
        pin = write_committed(root / (a.stage+'-report.json'), report)
        print(json.dumps(dict(report=pin, stage=a.stage, events=report['events'])), flush=True)
    except BaseException as exc:
        write_committed(root / (a.stage+'-failure.json'), dict(error=str(exc), trace=traceback.format_exc()))
        raise
    finally:
        for s in sessions: s.close()


if __name__ == '__main__': run_cli(main)
