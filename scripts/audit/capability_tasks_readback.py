"""Persist three-boundary task FSV with role, future, source and corruption controls."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.capability_source_readback import SourceReadback
from claims.capability_population import CapabilityPopulation, need
from claims.episodic_memory import canonical, digest
from claims.process_conditions import write_committed
from text.evaluation_multiscale import file_sha


def rebound(task):
    task['id'] = digest({k: v for k, v in task.items() if k != 'id'})
    return task


def run(root, raw, output):
    tick = time.perf_counter()
    output.mkdir(parents=True, exist_ok=False)
    pin = (root / 'COMMITTED').read_text()
    pop = CapabilityPopulation(root, pin)
    before = {n: file_sha(root / n) for n in [*pop.report['files'], 'report.json', 'COMMITTED']}
    write_committed(output / 'before.json', dict(population=pin, files=before))
    original = SourceReadback(pop, raw)
    source = original.run()
    write_committed(output / 'source-readback.json', source)
    selections = {}
    for key, task in pop.tasks.items():
        context = pop.selection(key, side='context')
        target = pop.selection(key, side='targets')
        need([f['id'] for f in context['fields']] == task['context']
             and [f['id'] for f in target['fields']] == task['targets'], 'FSV_SELECTION')
        need(not set(context) & {'checks', 'provenance', 'consequences', 'target_origin'}, 'FSV_METADATA_LEAK')
        selections[key] = dict(context=digest(context), target=digest(target),
                               context_fields=len(context['fields']), target_fields=len(target['fields']))
    write_committed(output / 'selections.json', selections)
    controls = []

    def reject(name, action):
        before_seal = pop.seal
        try:
            action()
        except (ValueError, KeyError) as exc:
            code = str(exc)
            need(code.startswith('MNLM_CAPABILITY_POPULATION_'), 'FSV_UNEXPECTED_ERROR')
        else:
            raise ValueError('MNLM_CAPABILITY_POPULATION_FSV_CONTROL_PASSED:' + name)
        controls.append(dict(name=name, rejected=True, error=code, before=before_seal, after=pop.seal))

    example = next(iter(pop.tasks.values()))
    missing = copy.deepcopy(example)
    missing['targets'] = []
    reject('missing_target', lambda: pop.task(rebound(missing)))
    swapped = copy.deepcopy(example)
    swapped['context'], swapped['targets'] = swapped['targets'], swapped['context']
    reject('swapped_input_target_roles', lambda: pop.task(rebound(swapped)))
    future = copy.deepcopy(example)
    future['context'].append(future['targets'][0])
    reject('future_target_in_context', lambda: pop.task(rebound(future)))
    old_links = pop.links
    pop.links = set()
    reject('dropped_relationship', lambda: pop.task(example))
    pop.links = old_links
    altered = copy.deepcopy(next(iter(pop.fields.values())))
    altered['text'] += ' changed'
    reject('source_text_rewritten', lambda: original.field(altered))
    if pop.family == 'reasoning':
        wrong = copy.deepcopy(next(t for t in pop.tasks.values() if t['checks']))
        wrong['checks'][0]['left_exact'] = '1000001'
        reject('arithmetic_receipt_value_changed', lambda: original.task(wrong))
    # Real persisted damage in a disposable evidence copy, leaving all authoritative bytes untouched.
    damaged = output / 'damaged-copy'
    shutil.copytree(root, damaged)
    file_before = file_sha(damaged / 'tasks.jsonl')
    with (damaged / 'tasks.jsonl').open('ab') as stream: stream.write(b'\n')
    reject('persisted_task_file_corruption', lambda: CapabilityPopulation(damaged, pin))
    damage = dict(before=file_before, action='append byte to copied tasks.jsonl',
                  after=file_sha(damaged / 'tasks.jsonl'), authoritative_before=before['tasks.jsonl'])
    manifest = json.loads((damaged / 'report.json').read_bytes())
    manifest['complete'] = False
    with (damaged / 'report.json').open('wb') as stream:
        stream.write(canonical(manifest))
    stale = copy.copy(pop)
    stale.root = damaged
    reject('stale_manifest', stale.fresh)
    pop.fresh()
    after = {n: file_sha(root / n) for n in before}
    need(before == after, 'FSV_AUTHORITATIVE_BYTES_CHANGED')
    # New process invocation is used below by callers for the final independent disk read.
    reloaded = CapabilityPopulation(root, pin)
    need(reloaded.seal == pop.seal, 'FSV_RELOAD')
    write_committed(output / 'controls.json', controls)
    write_committed(output / 'after.json', dict(files=after, damage=damage, cold_seal=reloaded.seal))
    files = ['before.json', 'source-readback.json', 'selections.json', 'controls.json', 'after.json']
    report = dict(schema='capability-task-source-fsv-v1', complete=True, population=str(root), population_pin=pin,
        family=pop.family, source_readback=source, selections=len(selections), controls=controls,
        boundaries=['original_bytes_to_fields', 'relationships_and_cutoffs_to_tasks',
                    'task_context_vs_private_target_selection', 'persisted_reload_and_corruption'],
        files={n: dict(bytes=(output / n).stat().st_size, sha256=file_sha(output / n)) for n in files},
        author_checked=True, independent_session_verified=False, learned_capability=False,
        native_projection_connected=False, kernel_connected=False, training_ready=False,
        scientific_admission=False, serving=False, code_sha256=file_sha(Path(__file__)),
        wall_seconds=time.perf_counter() - tick)
    report_pin = write_committed(output / 'report.json', report)
    print(json.dumps(dict(family=pop.family, tasks=len(selections), controls=len(controls),
                          complete=True, report_sha256=report_pin, wall_seconds=report['wall_seconds'])))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('population', 'raw', 'output'): p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    run(a.population, a.raw, a.output)
