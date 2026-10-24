"""Append real capability tasks to the live lineage and replay from journal bytes."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'lens'))
from audit.fsv_identity_loop import configure
from audit.repository_demonstration_memory import authenticate
from audit.repository_live_bootstrap import open_current
from audit.repository_relation_learning import retained
from audit.state_cycle_open import prepare
from claims.capability_population import need
from claims.codata_source import sha
from claims.episodic_memory import digest
from claims.process_conditions import write_committed
from kernel.repository_live_population import LiveRepositoryPopulation
from lens.lenscommon import run_cli, take_lease
from model.capability_memory import CapabilityMemoryLoop, code_pins
from senses.contract import SenseError


def read(path): return json.loads(Path(path).read_bytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--stage', choices=('intake', 'cold'), required=True)
    parser.add_argument('--surfaces', type=Path)
    a = parser.parse_args()
    start = time.perf_counter()
    config = read(a.config)
    root, parent = Path(config['root']), Path(config['source'])
    need(sha((parent / 'experience.sqlite3').read_bytes()) == config['source_pin'], 'FSV_PARENT_JOURNAL')
    old = read(parent / 'snapshot-final.json')
    live = read(parent / 'config.json')
    need(sha((Path(live['source']) / 'config.json').read_bytes()) == live['source_config'], 'FSV_PARENT_CONFIG')
    recipes = authenticate(read(Path(live['source']) / 'config.json'))
    if a.stage == 'intake':
        prepare(config, recipes, root)
        write_committed(root / 'config.json', config)
    else:
        need(read(root / 'config.json') == config, 'FSV_CONFIG')
    # Repository kernels belong to the immutable parent journal namespace. The
    # capability successor records its own physical history in root/experience.sqlite3.
    base = Path(live['base'])
    population = LiveRepositoryPopulation(base / 'experience.sqlite3', live['base_pin'],
                                          parent / 'experience.sqlite3')
    gpu = subprocess.check_output(['nvidia-smi'], text=True)
    take_lease('29', 'capability-memory-' + a.stage)
    (root / ('gpu-' + a.stage + '.txt')).open('x').write(gpu)
    configure()
    guard = {'active': a.stage == 'cold'}
    forbidden = {(root / n).resolve() for n in ('snapshot-final.json', 'snapshot-intake.json')}

    def deny_snapshot(event, args):
        if guard['active'] and event == 'open' and isinstance(args[0], str):
            need(Path(args[0]).resolve() not in forbidden, 'FSV_LATEST_SNAPSHOT_READ')

    sys.addaudithook(deny_snapshot)
    session, sessions = open_current(recipes, root, population, loop_type=CapabilityMemoryLoop)
    guard['active'] = False
    restored = time.perf_counter()
    print(json.dumps(dict(stage=a.stage, restored_events=session.journal.count,
                          seconds=restored-start)), flush=True)
    controls, boundaries = [], []

    def refuse(name, code, fn):
        before = digest(session.snapshot())
        try: fn()
        except (ValueError, SenseError) as exc:
            need(code in str(exc), 'FSV_WRONG_ERROR:' + str(exc))
        else: raise AssertionError('FSV_ACCEPTED:' + name)
        need(digest(session.snapshot()) == before, 'FSV_CONTROL_MUTATED')
        controls.append(dict(name=name, code=code, before=before, after=before))

    try:
        legacy = retained(session, session.model)
        if a.stage == 'intake':
            need(session.snapshot() == old, 'FSV_RESTORED_PARENT')
            write_committed(root / 'legacy.json', legacy)
            for spec in config['populations']:
                evidence = session._capability_source(spec)
                p = evidence.population
                groups = sorted({t['split_group'] for t in p.tasks.values()})
                split = max(1, len(groups) // 2)
                for selected in (groups[:split], groups[split:]):
                    before = session.journal.count
                    body = session.observe_capability(spec, selected)
                    need(session.journal.count == before + 1, 'FSV_ATOMIC_INTAKE')
                    boundaries.append(dict(family=p.family, groups=selected, before=before,
                        after=session.journal.count, event=digest(body), head=session.head,
                        tasks=len(session.capability_tasks), fields=len(session.capability_fields)))
                    print(json.dumps(boundaries[-1]), flush=True)
                refuse(p.family + ':duplicate', 'MEMORY_DUPLICATE_TASK',
                       lambda: session.observe_capability(spec, groups))
            write_committed(root / 'snapshot-intake.json', session.snapshot())
            for family in ('coding', 'conversation', 'reasoning'):
                key = next(k for k, t in session.capability_tasks.items() if t['family'] == family)
                task = session.capability_tasks[key]
                call = lambda: session.capability_selection(key, side='context', binding=task['binding'])
                need(call()['fields'] and not set(f['id'] for f in call()['fields']) &
                     {f['id'] for f in task['selections']['targets']['fields']}, 'FSV_FUTURE_LEAKAGE')
                refuse(family + ':stale_binding', 'MEMORY_STALE_BINDING', lambda:
                       session.capability_selection(key, side='context', binding='0'*64))
                field = task['selections']['targets']['fields'][0]['id']
                ep = session.capability_fields[task['population'] + ':' + field]['episode']
                affected = session.memory.descendants(ep)
                before = session.head
                session.invalidate(ep, reason='FSV source eligibility withdrawal; original bytes retained unchanged')
                need(all(session.memory.state[k]['grounded_scope'] is None for k in affected), 'FSV_DESCENDANTS')
                refuse(family + ':withdrawn_target_blocks_context', 'TOOL_EVIDENCE_UNAVAILABLE', call)
                refuse(family + ':task_review_before_source', 'TOOL_EVIDENCE_UNAVAILABLE', lambda:
                       session.review_capability(task['episode'], reason='FSV premature dependent review'))
                reason = 'FSV independent retained-source readback; restores source attestation only'
                session.review_capability(ep, reason=reason)
                for descendant in affected:
                    if descendant != ep: session.review_capability(descendant, reason=reason)
                need(call() == task['selections']['context'], 'FSV_REVIEW_SELECTION')
                boundaries.append(dict(family=family, before=before, after=session.head,
                    action='withdraw_then_recheck_retained_source', affected=affected,
                    selected_task=key, source_episode=ep))
            write_committed(root / 'snapshot-final.json', session.snapshot())
            write_committed(root / 'boundaries.json', boundaries)
        else:
            need(session.snapshot() == read(root / 'snapshot-final.json'), 'FSV_COLD_SNAPSHOT')
        final = session.snapshot()
        for k, value in old.items():
            if k not in ('head', 'events', 'memory'):
                need(final[k] == value, 'FSV_OLD_STATE:' + k)
        need(all(final['memory']['states'][k] == v for k, v in old['memory']['states'].items()),
             'FSV_OLD_EVIDENCE')
        need(retained(session, session.model) == legacy == read(root / 'legacy.json'), 'FSV_LEGACY_MODEL')
        need(sha((parent / 'experience.sqlite3').read_bytes()) == config['source_pin'], 'FSV_PARENT_CHANGED')
        surfaces = None
        if a.surfaces is not None:
            from audit.capability_surface_fsv import run
            surfaces = run(session, a.surfaces)
            need(session.snapshot() == final, 'FSV_SURFACE_STATE_CHANGED')
        result = dict(stage=a.stage, events=final['events'], head=final['head'],
            tasks=len(session.capability_tasks), fields=len(session.capability_fields),
            relations=len(session.capability_relations), controls=controls, code=code_pins(),
            snapshot=digest(final), journal=sha((root / 'experience.sqlite3').read_bytes()),
            latest_snapshot_required_for_restore=False, legacy_retained=True,
            surfaces=surfaces,
            kernel_ready=False, learned_capability=False, scientific_admission=False, serving=False,
            number_status='measured', acceptance='author-checked',
            phases=dict(restore=restored-start, operations=time.perf_counter()-restored))
        pin = write_committed(root / (a.stage + '-stage.json'), result)
        print(json.dumps(dict(stage=a.stage, pin=pin, events=final['events'],
                              seconds=time.perf_counter()-start)), flush=True)
    finally:
        for item in sessions: item.close()


if __name__ == '__main__':
    try: run_cli(main)
    except BaseException:
        if '--config' in sys.argv:
            config = read(sys.argv[sys.argv.index('--config') + 1])
            root = Path(config['root'])
            if root.is_dir():
                stage = sys.argv[sys.argv.index('--stage') + 1]
                write_committed(root / ('failure-' + stage + '.json'), dict(traceback=traceback.format_exc()))
        raise
