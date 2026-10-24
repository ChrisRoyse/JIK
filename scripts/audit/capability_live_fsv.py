"""Real journal-driven capability retraction, review, checkpoint interruption and recovery."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'lens'))
from audit.fsv_identity_loop import configure
from audit.repository_demonstration_memory import authenticate
from audit.capability_live_open import open_live
from audit.repository_memory_readback import chain
from audit.state_cycle_open import prepare
from claims.capability_population import need, sha
from claims.episodic_memory import digest
from claims.process_conditions import write_committed
from kernel.repository_live_population import LiveRepositoryPopulation
from lens.lenscommon import run_cli, take_lease
from model.tool_growth import parameters


def read(path): return json.loads(Path(path).read_bytes())


def chosen_sources(session):
    p = session._task_population()
    queries = {session.capability_tasks[k]['episode'] for k in p.queries()}
    chosen = []
    for family in ('coding', 'conversation', 'reasoning'):
        for key in session.task_runtime.value['ids']:
            task = session.capability_tasks[key]
            if task['family'] != family: continue
            field = task['selections']['context']['fields'][0]['id']
            ep = session.capability_fields[task['population']+':'+field]['episode']
            affected = sorted({ep, *session.memory.descendants(ep)})
            if set(affected) & queries: continue
            chosen.append(dict(family=family, task=key, episode=ep, affected=affected))
            break
        else: raise AssertionError('TASK_LIVE_FSV_SOURCE_SCARCITY:'+family)
    return chosen


def refuse(session, code, operation):
    before = session.head
    try: operation()
    except ValueError as exc:
        error = str(exc)
        need(code in error, 'TASK_LIVE_FSV_ERROR:'+error)
    else: raise AssertionError('TASK_LIVE_FSV_ACCEPTED:'+code)
    need(session.head == before, 'TASK_LIVE_FSV_REFUSAL_MUTATION')
    return dict(expected=code, error=error, before=before, after=session.head)


def checkpoint_roster(root):
    return {str(p.relative_to(root)): dict(pin=sha(p.read_bytes()), modified_ns=p.stat().st_mtime_ns)
            for p in sorted((root/'task-kernels').glob('*/*/updater/COMMITTED'))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--stage', choices=('update', 'cold'), required=True)
    a = parser.parse_args(); config = read(a.config)
    root, parent = Path(config['root']), Path(config['source'])
    need(sha((parent/'experience.sqlite3').read_bytes()) == config['source_pin'], 'TASK_LIVE_FSV_PARENT')
    memory = read(parent/'config.json'); repository = Path(memory['source'])
    live = read(repository/'config.json')
    recipes = authenticate(read(Path(live['source'])/'config.json'))
    if a.stage == 'update':
        prepare(config, recipes, root); write_committed(root/'config.json', config)
    else:
        need(read(root/'config.json') == config, 'TASK_LIVE_FSV_CONFIG')
        before_roster = checkpoint_roster(root)
    gpu = subprocess.check_output(['nvidia-smi'], text=True)
    take_lease('29', 'capability-live-'+a.stage)
    (root/('gpu-'+a.stage+'.txt')).write_text(gpu)
    configure(); start = time.perf_counter()
    population = LiveRepositoryPopulation(Path(live['base'])/'experience.sqlite3',
        live['base_pin'], repository/'experience.sqlite3')
    session, sessions = open_live(recipes, root, population)
    restored = time.perf_counter()
    print(json.dumps(dict(stage=a.stage, events=session.journal.count, restore_seconds=restored-start)), flush=True)
    try:
        original_parameters = parameters(session.model)
        controls = []
        if a.stage == 'update':
            need(session.journal.count == 515, 'TASK_LIVE_FSV_PREFIX_COUNT')
            split = read(Path(config['kernel'])/'splits.json')
            seed = read(Path(config['kernel'])/'shared/stages.json')[-1]
            spec = dict(root=str(root/'task-kernels'),
                splits={e.native.pin: split[e.population.family] for e in session.capability_sources.values()},
                seed={k: seed[k] for k in ('path', 'pin')})
            session.enable_task_updates(spec)
            baseline = session.task_checkpoint
            chosen = chosen_sources(session)
            write_committed(root/'baseline.json', dict(checkpoint=baseline, head=session.head,
                                                       selected=chosen, parameters=original_parameters))
            with session.task_batch():
                for item in chosen:
                    session.invalidate(item['episode'], reason='FSV retained source temporarily withdrawn')
                    query = session._task_population().queries()[0]
                    refuse(session, 'TASK_LIVE_UPDATE_PENDING', lambda:
                           session.read_task(query, kernel=baseline['kernel']['pin']))
            withdrawn = session.task_checkpoint
            affected = {ep for item in chosen for ep in item['affected']}
            removed = {k for k, t in session.capability_tasks.items() if t['episode'] in affected}
            need(not removed & set(session.task_runtime.value['ids']), 'TASK_LIVE_FSV_WITHDRAWAL')
            need(withdrawn['kernel'] != baseline['kernel'], 'TASK_LIVE_FSV_NO_AUTOMATIC_UPDATE')
            write_committed(root/'withdrawn.json', dict(checkpoint=withdrawn, removed=sorted(removed), head=session.head))
            with session.task_batch():
                for item in chosen:
                    for ep in item['affected']:
                        if ep == item['episode']: continue
                        refuse(session, 'TOOL_EVIDENCE_UNAVAILABLE', lambda ep=ep:
                               session.review_capability(ep, reason='FSV premature dependent review'))
                        break
                    session.review_capability(item['episode'], reason='FSV retained bytes independently rechecked')
                for ep in sorted(affected-{item['episode'] for item in chosen}):
                    session.review_capability(ep, reason='FSV restored source dependencies rechecked')
            reviewed = session.task_checkpoint
            need(set(session.task_runtime.value['ids']) == set(read(Path(baseline['kernel']['path'])/'manifest.json')['ids']),
                 'TASK_LIVE_FSV_REVIEW_CLOSURE')
            write_committed(root/'reviewed.json', dict(checkpoint=reviewed, head=session.head))
            query = session._task_population().queries()[0]
            refuse(session, 'TASK_LIVE_STALE_KERNEL', lambda: session.read_task(query, kernel=baseline['kernel']['pin']))
            # Simulated process interruption after the real durable updater commit.
            with session.task_batch(stop_after_checkpoint=True):
                item = chosen[0]
                session.invalidate(item['episode'], reason='FSV crash boundary after checkpoint before publication')
            need(session.task_intent is not None, 'TASK_LIVE_FSV_NO_PENDING')
            refuse(session, 'TASK_LIVE_UPDATE_PENDING', lambda: session.read_task(query, kernel=reviewed['kernel']['pin']))
            pending = session.snapshot()
            write_committed(root/'pending.json', pending)
            write_committed(root/'pending-journal.json', dict(pin=sha((root/'experience.sqlite3').read_bytes()),
                head=session.head, events=session.journal.count))
        else:
            pending = read(root/'pending.json')
            need(checkpoint_roster(root) == before_roster, 'TASK_LIVE_FSV_DUPLICATE_DERIVATION')
            need(session.journal.count == pending['events']+1 and session.task_intent is None,
                 'TASK_LIVE_FSV_RECOVERY_EVENT')
            need(session.task_checkpoint['source'] == pending['task_kernel']['current_source'],
                 'TASK_LIVE_FSV_RECOVERY_SOURCE')
            checkpoint_before = session.task_checkpoint; head_before = session.head
            need(session.synchronize_tasks() is False and session.head == head_before
                 and session.task_checkpoint == checkpoint_before, 'TASK_LIVE_FSV_NOOP')
            query = session._task_population().queries()[0]
            baseline = read(root/'baseline.json')
            controls.append(refuse(session, 'TASK_LIVE_STALE_KERNEL', lambda:
                session.read_task(query, kernel=baseline['checkpoint']['kernel']['pin'])))
            item = baseline['selected'][0]
            child = next(ep for ep in item['affected'] if ep != item['episode'])
            controls.append(refuse(session, 'TOOL_EVIDENCE_UNAVAILABLE', lambda:
                session.review_capability(child, reason='FSV cold premature dependent review')))
            old_intake = next(iter(session.capability_intakes.values()))['body']
            controls.append(refuse(session, 'MEMORY_DUPLICATE_TASK', lambda:
                session.observe_capability(old_intake['spec'], old_intake['groups'])))
            for family in ('coding', 'conversation', 'reasoning'):
                p = session._task_population()
                query = next(k for k in p.queries() if p.records[k]['family'] == family)
                result = session.read_task(query, kernel=session.task_checkpoint['kernel']['pin'])
                need(all(v == p.target(k) for k, v in result['targets'].items()), 'TASK_LIVE_FSV_TARGET_CLOSURE')
            write_committed(root/'snapshot-final.json', session.snapshot())
            write_committed(root/'cold-boundaries.json', dict(controls=controls,
                checkpoint_files_before=before_roster, checkpoint_files_after=checkpoint_roster(root),
                unchanged_source_no_update=True, pending_before=read(root/'pending-journal.json'),
                head_after=session.head, events_after=session.journal.count))
        need(parameters(session.model) == original_parameters, 'TASK_LIVE_FSV_MODEL_DRIFT')
        parent_contract, prefix = chain(parent/'experience.sqlite3')
        contract, rows = chain(root/'experience.sqlite3')
        need(parent_contract == contract and rows[:len(prefix)] == prefix, 'TASK_LIVE_FSV_PREFIX_CHANGED')
        report = dict(stage=a.stage, events=session.journal.count, head=session.head,
            journal=sha((root/'experience.sqlite3').read_bytes()), checkpoint=session.task_checkpoint,
            pending=session.task_intent is not None, parent_unchanged=True, parameters_unchanged=True,
            phases=dict(restore=restored-start, work=time.perf_counter()-restored),
            controls=controls, audit_code=sha(Path(__file__).read_bytes()),
            number_status='measured', acceptance='author-checked', scientific_admission=False,
            learned_capability=False, model_consumer_connected=False)
        pin = write_committed(root/(a.stage+'-report.json'), report)
        print(json.dumps(dict(stage=a.stage, pin=pin, events=session.journal.count)), flush=True)
    finally:
        for item in sessions: item.close()


if __name__ == '__main__': run_cli(main)
