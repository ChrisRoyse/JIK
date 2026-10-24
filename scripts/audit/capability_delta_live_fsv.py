"""Successive real source arrivals -> automatic union kernels -> cold recovery and prediction."""
import argparse
from functools import partial
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

sys.path[:0] = [str(Path(__file__).resolve().parents[1]),
               str(Path(__file__).resolve().parents[1] / 'lens')]
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
from model.capability_delta_memory import DeltaCapabilityMemoryLoop, code_pins
from model.tool_growth import parameters


def read(path):
    return json.loads(Path(path).read_bytes())


def sources(config):
    values = read(config['delta_config'])
    adapter = dict(schema='capability-delta-native-v1', code=code_pins(),
                   registry=config['registry'], native_proof=config['native_proof'])
    return {name: dict(value, **config['measurements'], adapter=adapter) for name, value in values.items()}


def checkpoint(session):
    value = read(Path(session.task_checkpoint['kernel']['path']) / 'manifest.json')
    need(value['recall_passes'] and all(c['passes'] and all(min(f['recall'], f['target_recall']) >= .95
        for f in c['families'].values()) for c in value['calibration'].values()), 'DELTA_LIVE_RECALL')
    return dict(head=session.head, events=session.journal.count, checkpoint=session.task_checkpoint,
        training_tasks=len(value['ids']), query_tasks=len(session._task_population().queries()),
        minimum_recall=min(min(f['recall'], f['target_recall']) for c in value['calibration'].values()
                           for f in c['families'].values()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--stage', choices=('update', 'cold'), required=True)
    a = p.parse_args(); config = read(a.config)
    root, parent = Path(config['root']), Path(config['source'])
    need(sha((parent / 'experience.sqlite3').read_bytes()) == config['source_pin'], 'DELTA_LIVE_PARENT')
    original = read(parent / 'config.json')
    memory = read(Path(original['source']) / 'config.json')
    repository = Path(memory['source']); live = read(repository / 'config.json')
    recipes = authenticate(read(Path(live['source']) / 'config.json'))
    if a.stage == 'update':
        prepare(config, recipes, root)
        write_committed(root / 'config.json', config)
    else:
        need(read(root / 'config.json') == config, 'DELTA_LIVE_CONFIG_CHANGED')
    gpu = subprocess.check_output(['nvidia-smi'], text=True)
    take_lease('29', 'capability-delta-live-' + a.stage)
    (root / ('gpu-' + a.stage + '.txt')).write_text(gpu)
    configure(); started = time.perf_counter()
    population = LiveRepositoryPopulation(Path(live['base']) / 'experience.sqlite3',
        live['base_pin'], repository / 'experience.sqlite3')
    session, sessions = open_current(recipes, root, population,
        loop_type=partial(DeltaCapabilityMemoryLoop, recover_on_open=False))
    restored = time.perf_counter()
    print(json.dumps(dict(stage=a.stage, restore_seconds=restored-started,
                          events=session.journal.count, head=session.head)), flush=True)
    try:
        baseline_parameters = parameters(session.model)
        controls, generations = [], []
        specs = sources(config)
        registry = read(config['registry']['path'])
        if a.stage == 'update':
            generations.append(checkpoint(session))
            session.relocate_task_outputs(str(root / 'task-kernels'))
            write_committed(root / 'baseline.json', generations[0])
            wrong = dict(registry['splits'][specs['dialogue']['population_pin']])
            wrong['excluded'] = []
            controls.append(refuse(session, 'DELTA_MEMORY_REGISTER', lambda:
                session.register_task_source(specs['dialogue'], wrong)))
            for name, spec in specs.items():
                session.register_task_source(spec, registry['splits'][spec['population_pin']])
            for names in (('dialogue',), ('condaqa_conversation', 'condaqa_reasoning')):
                previous = session.task_checkpoint['kernel']['pin']
                tick = time.perf_counter()
                with session.task_batch():
                    for name in names:
                        spec = specs[name]
                        session.observe_capability(spec, registry['splits'][spec['population_pin']]['groups'])
                        query = session._task_population().queries()[0]
                        controls.append(refuse(session, 'TASK_LIVE_UPDATE_PENDING', lambda:
                            session.read_task(query, kernel=previous)))
                current = checkpoint(session)
                current['intake_sources'] = list(names)
                current['seconds'] = time.perf_counter()-tick
                generations.append(current)
                controls.append(refuse(session, 'TASK_LIVE_STALE_KERNEL', lambda:
                    session.read_task(session._task_population().queries()[0], kernel=previous)))
                write_committed(root / ('generation-' + str(len(generations)-1) + '.json'), current)
                print(json.dumps(dict(sources=names, training=current['training_tasks'],
                    queries=current['query_tasks'], recall=current['minimum_recall'], seconds=current['seconds'])), flush=True)
            prediction_root = root / 'prediction'; prediction_root.mkdir()
            cases = []
            for name, spec in specs.items():
                evidence = session._capability_source(spec)
                pop = evidence.population
                queries = registry['splits'][pop.pin]['queries']
                unknown = [k for k in queries if any(pop.fields[f]['text'] == "DON'T KNOW" for f in pop.tasks[k]['targets'])]
                key = unknown[0] if unknown else max(queries,
                    key=lambda k: sum(len(pop.fields[f]['text']) for f in pop.tasks[k]['targets']))
                cases.append(dict(family=pop.family, key=key, source=name))
            predictions = prediction_check(session, cases, prediction_root)
            write_committed(prediction_root / 'report.json', predictions)
            write_committed(root / 'snapshot-final.json', session.snapshot())
            need(len(session.delta_stores) == 1, 'DELTA_LIVE_DUPLICATE_NATIVE_STORE')
        else:
            saved = read(root / 'snapshot-final.json')
            need(session.head == saved['head'] and session.snapshot() == saved, 'DELTA_LIVE_COLD_STATE')
            before = checkpoint_roster(root)
            head, events = session.head, session.journal.count
            need(session.synchronize_tasks() is False and session.head == head and session.journal.count == events,
                 'DELTA_LIVE_NO_CHANGE_WROTE')
            need(checkpoint_roster(root) == before, 'DELTA_LIVE_NO_CHANGE_REBUILT')
            generations.append(checkpoint(session))
            for spec in specs.values():
                key = registry['splits'][spec['population_pin']]['queries'][0]
                result = session.read_task(key, kernel=session.task_checkpoint['kernel']['pin'])
                need(result['targets'], 'DELTA_LIVE_COLD_RETRIEVAL')
        need(parameters(session.model) == baseline_parameters, 'DELTA_LIVE_PARAMETER_DRIFT')
        parent_contract, prefix = chain(parent / 'experience.sqlite3')
        contract, rows = chain(root / 'experience.sqlite3')
        need(parent_contract == contract and rows[:len(prefix)] == prefix
             and sha((parent / 'experience.sqlite3').read_bytes()) == config['source_pin'], 'DELTA_LIVE_PREFIX_CHANGED')
        result = dict(schema='capability-delta-live-fsv-v1', stage=a.stage, complete=True,
            head=session.head, events=session.journal.count, journal=sha((root / 'experience.sqlite3').read_bytes()),
            generations=generations, controls=controls, source_prefix_unchanged=True,
            parameters_unchanged=True, optimizer_steps=0, native_stores=len(session.delta_stores),
            phases_seconds=dict(restore=restored-started, work=time.perf_counter()-restored),
            scientific_admission=False, learned_capability=False, serving=False,
            generated_surface_guard_complete=False, author_checked=True,
            code_sha256=sha(Path(__file__).read_bytes()))
        pin = write_committed(root / (a.stage + '-report.json'), result)
        print(json.dumps(dict(report=pin, stage=a.stage, events=session.journal.count)), flush=True)
    except BaseException as exc:
        write_committed(root / (a.stage + '-failure.json'), dict(error=str(exc), trace=traceback.format_exc()))
        raise
    finally:
        for item in sessions:
            item.close()


if __name__ == '__main__':
    run_cli(main)
