"""Physical task kernels: frozen exposed holdouts, growth, closures and cold checkpoints."""
import argparse
import copy
import json
from pathlib import Path
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'lens'))
from audit.capability_journal_reader import JournalReader
from claims.capability_population import need, sha
from claims.episodic_memory import digest
from claims.process_conditions import write_committed
from kernel.capability_kernel import TaskCompanion, TaskKernel
from kernel.capability_population import TaskPopulation, SharedTasks
from kernel.capability_split import freeze
from kernel.capability_update_state import save, restore
from model.capability_input import CapabilityNative


def read(path): return json.loads(Path(path).read_bytes())


def load(config, split=None):
    memory = read(config['memory_config'])
    need(sha(Path(config['journal']).read_bytes()) == config['journal_pin'], 'KERNEL_FSV_JOURNAL_PIN')
    reader = JournalReader(memory, read(config['native_proofs']))
    state = reader.read(config['journal'])
    populations = {}
    for spec in memory['populations']:
        native = CapabilityNative(**spec)
        family = native.population.family
        contract = freeze(native.population) if split is None else split[family]
        populations[family] = TaskPopulation(native, state['tasks'], state['fields'],
                                            state['relationships'], state['states'], contract)
    return populations, state


def negative(population, path, pin, source):
    controls = []
    def refuse(name, error, fn):
        try: fn()
        except ValueError as exc: need(error in str(exc), 'KERNEL_FSV_WRONG_ERROR:' + str(exc))
        else: raise AssertionError('KERNEL_FSV_ACCEPTED:' + name)
        controls.append(dict(name=name, error=error))
    refuse('stale_source', 'TASK_STALE_SOURCE', lambda:
           TaskKernel(population, path, pin, source=dict(source, groups=[])))
    # Re-sign semantic mutations: reject despite valid new content hashes.
    mutations = ('missing_target', 'swapped_role', 'future_answer_as_context', 'lost_relationship')
    original = read(path/'manifest.json')
    for kind in mutations:
        root = path.parent/('negative-'+kind); root.mkdir()
        value = copy.deepcopy(original); key = value['ids'][0]
        link = value['connections'][key]
        if kind == 'missing_target': link['task']['selections']['targets']['fields'] = []
        elif kind == 'swapped_role': link['task']['selections']['targets']['fields'][0]['role'] = 'context'
        elif kind == 'future_answer_as_context':
            value['slots'][key] = copy.deepcopy(value['slots'][value['ids'][-1]])
        else: link['relationships'] = []
        new_pin = write_committed(root/'manifest.json', value)
        write_committed(root/'COMMITTED', dict(manifest=new_pin))
        error = 'TASK_ARTIFACT_COVERAGE' if kind == 'future_answer_as_context' else 'TASK_ARTIFACT_CONNECTIONS'
        refuse(kind, error, lambda: TaskKernel(population, root, new_pin, source=source))
    return controls


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--stage', choices=('freeze', 'build', 'cold'), required=True)
    a = parser.parse_args(); config = read(a.config)
    root = Path(config['root']); start = time.perf_counter()
    if a.stage == 'freeze':
        need(not root.exists(), 'KERNEL_FSV_ROOT_EXISTS')
        pops, state = load(config); root.mkdir()
        write_committed(root/'config.json', config)
        split = {f: p.split for f, p in pops.items()}
        pin = write_committed(root/'splits.json', split)
        inventory = {}
        for family, p in pops.items():
            groups = sorted(set(p.split['groups']) - set(p.split['excluded']))
            before = groups[:max(1, len(groups)//2)]
            inventory[family] = dict(groups=groups, before=before, before_tasks=len(p.ids(before)),
                final_tasks=len(p.ids(groups)), queries=len(p.queries()),
                excluded=p.split['excluded'], duplicate_pairs=p.split['duplicate_pairs'],
                windows={n: dict(train=sum(len(p.values[k][n]) for k in p.ids(groups)),
                    query=sum(len(p.values[k][n]) for k in p.queries())) for n in p.specs})
        write_committed(root/'inventory.json', inventory)
        print(json.dumps(dict(stage='freeze', splits=pin, inventory=inventory)), flush=True)
        return
    need(read(root/'config.json') == config, 'KERNEL_FSV_CONFIG_CHANGED')
    split = read(root/'splits.json'); pops, state = load(config, split)
    loaded = time.perf_counter(); results = {}
    inventory_all = read(root/'inventory.json')
    shared = SharedTasks(pops)
    for family, p in [('shared', shared)]:
        location = root/family; stages = []
        inventory = {s: sorted(g for v in inventory_all.values() for g in v[s]) for s in ('before', 'groups')}
        if a.stage == 'build':
            location.mkdir(); companion = TaskCompanion(p)
            for name, groups in [('initial', inventory['before']), ('expanded', inventory['groups'])]:
                target = location/name; target.mkdir()
                value, pin, timings = companion.sync(groups, target/'kernel')
                write_committed(target/'result.json', dict(pin=pin, source=value['source'], timings=timings,
                    recall={n: [v['recall'], v['target_recall']] for n, v in value['calibration'].items()},
                    passes=value['recall_passes']))
                print(json.dumps(dict(family=family, stage=name, tasks=len(value['ids']), timings=timings,
                    passes=value['recall_passes'], recall={n: v['families'] for n, v in value['calibration'].items()})), flush=True)
                need(value['recall_passes'], 'KERNEL_FSV_RECALL_FAILED')
                state_pin, checks = save(companion, target/'updater')
                stages.append(dict(stage=name, path=str(target/'updater'), pin=state_pin, checks=checks))
            write_committed(location/'stages.json', stages)
            controls = negative(p, location/'expanded/kernel', companion.previous['pin'], companion.last['source'])
        else:
            stages = read(location/'stages.json'); controls = []
            stage = stages[-1]
            companion, accepted = restore(p, stage['path'], stage['pin'])
            # An unchanged source must not rebuild, rewrite, or advance any index.
            before_indices = {n: sha(bytes(u.index.save())) for n, u in companion.updates.items()}
            no_op = location/'unchanged-source-must-not-write'
            value, pin, work = companion.sync(companion.last['source']['groups'], no_op)
            need(value == companion.last and pin == companion.previous['pin'] and work == dict(reused=True)
                 and not no_op.exists()
                 and before_indices == {n: sha(bytes(u.index.save())) for n, u in companion.updates.items()},
                 'KERNEL_FSV_DUPLICATE_WORK')
            controls.append(dict(name='unchanged_source_no_work', persisted_indices_unchanged=True))
            for key in p.queries():
                trace = accepted.route(key, source=companion.last['source'])
                need(all(t == p.target(k) for k, t in trace['targets'].items()), 'KERNEL_FSV_COLD_TARGET')
            print(json.dumps(dict(family=family, stage='cold', queries=len(p.queries()),
                                  kernel=companion.previous)), flush=True)
        results[family] = dict(stages=stages, controls=controls, final_kernel=companion.previous,
                              strict_recall={n: v['families'] for n, v in companion.last['calibration'].items()})
    need(sha(Path(config['journal']).read_bytes()) == config['journal_pin'], 'KERNEL_FSV_JOURNAL_CHANGED')
    report = dict(stage=a.stage, results=results, journal=config['journal_pin'], head=state['head'],
        events=state['events'], splits=sha((root/'splits.json').read_bytes()),
        phases=dict(load=loaded-start, work=time.perf_counter()-loaded), scientific_admission=False,
        learned_capability=False, serving=False, journal_integration=False, number_status='measured',
        acceptance='author-checked')
    pin = write_committed(root/(a.stage+'-report.json'), report)
    print(json.dumps(dict(stage=a.stage, pin=pin, seconds=time.perf_counter()-start)), flush=True)


if __name__ == '__main__':
    try: main()
    except BaseException:
        if '--config' in sys.argv:
            c = read(sys.argv[sys.argv.index('--config')+1]); root = Path(c['root'])
            if root.is_dir():
                write_committed(root/('failure-'+str(time.time_ns())+'.json'), dict(traceback=traceback.format_exc()))
        raise
