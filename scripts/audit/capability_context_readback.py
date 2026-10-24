"""Read retained Git bytes, task boundaries and import witnesses through a separate audit path."""
import argparse
import ast
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path, PurePosixPath
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.capability_revision_fsv import rejected
from claims.capability_population import need, sha
from claims.capability_context_population import ContextPopulation
from claims.episodic_memory import digest
from claims.process_conditions import write_committed
from oracle.prepare_swe import source_files
from senses.trajectory_population import read_pinned
from text.evaluation_multiscale import file_sha


def imports(raw, path):
    """Reconstruct declarations independently of the producer's adjacency traversal."""
    parts = list(PurePosixPath(path).with_suffix('').parts)
    if parts[0] in ('src', 'lib'): parts = parts[1:]
    package = parts[:-1]
    result = set()
    for node in ast.walk(ast.parse(raw)):
        if isinstance(node, ast.Import):
            result.update((node.lineno, a.name) for a in node.names)
        if isinstance(node, ast.ImportFrom):
            prefix = package[:len(package)+1-node.level] if node.level else []
            base = '.'.join(prefix + ([node.module] if node.module else []))
            result.add((node.lineno, base))
            result.update((node.lineno, '.'.join(filter(None, (base, a.name))))
                          for a in node.names if a.name != '*')
    return result


def run(a):
    start = tick = time.perf_counter()
    report = read_pinned(a.population / 'report.json', a.pin)
    before = {str(a.population / n):file_sha(a.population / n) for n in report['files']}
    rows = {n:[json.loads(v) for v in (a.population / (n+'.jsonl')).read_bytes().splitlines()]
            for n in ('tasks', 'fields', 'edges')}
    fields = {f['id']:f for f in rows['fields']}
    source = report['source']; c = source['census']
    census = read_pinned(Path(c['path']) / 'report.json', c['pin'])
    repos = read_pinned(Path(c['path']) / 'repositories.json', census['files']['repositories.json']['sha256'])
    snapshots, import_count = {}, 0
    for repo in repos:
        manifest = read_pinned(Path(repo['source']['path']), repo['source']['sha256'])
        files = source_files(Path(source['source_root']), manifest['src'])
        snapshots[repo['task']] = files
        selection = report['coverage']['selections'][repo['task']]
        witnessed = set()
        parsed = {}
        for edge in selection['imports']:
            origin, target = edge['origin'], edge['target']
            need(origin in files and target in files, 'CONTEXT_READBACK_IMPORT_FILES')
            if origin not in parsed: parsed[origin] = imports(files[origin], origin)
            target_parts = list(PurePosixPath(target).with_suffix('').parts)
            if target_parts[0] in ('src', 'lib'): target_parts = target_parts[1:]
            if target_parts[-1] == '__init__': target_parts = target_parts[:-1]
            module = '.'.join(target_parts)
            for declaration in edge['declarations']:
                requested = declaration['requested']
                need((declaration['line'], requested) in parsed[origin]
                     and (requested == module or requested.startswith(module+'.')),
                     'CONTEXT_READBACK_IMPORT_DECLARATION')
            witnessed.add((origin,target)); import_count += 1
        for path, reasons in selection['selected'].items():
            for reason in reasons:
                route = reason['path']
                if reason['kind'] in ('direct_local_import', 'base_test_import_path'):
                    need(all(pair in witnessed for pair in zip(route, route[1:])), 'CONTEXT_READBACK_ROUTE')
                    need((reason['kind'] == 'direct_local_import' and route[0] in selection['roots']
                          and route[-1] == path) or (reason['kind'] == 'base_test_import_path'
                          and route[0] == path and route[-1] in selection['roots']), 'CONTEXT_READBACK_ROUTE_ENDS')
                else:
                    need(reason['kind'] in ('same_commit_test_suite', 'root_project_readme')
                         and route == [path], 'CONTEXT_READBACK_CONTEXT_REASON')
    new_fields = [f for f in fields.values() if 'context_census' in f['source']]
    for field in new_fields:
        ref = field['source']; raw = snapshots[ref['task']][ref['path']]
        need(raw == field['text'].encode() and sha(raw) == ref['sha256'] == field['text_sha256'],
             'CONTEXT_READBACK_SOURCE_BYTES')
    phases = dict(original_git_bytes_and_import_witnesses=time.perf_counter()-tick); tick = time.perf_counter()
    p = source['parent']; parent_report = read_pinned(Path(p['path']) / 'report.json', p['pin'])
    old_tasks = read_pinned_rows(Path(p['path']) / 'tasks.jsonl', parent_report['files']['tasks.jsonl']['sha256'])
    old_fields = read_pinned_rows(Path(p['path']) / 'fields.jsonl', parent_report['files']['fields.jsonl']['sha256'])
    edges = {(e['origin'],e['target'],e['kind']) for e in rows['edges']}
    for task in rows['tasks']:
        old = old_tasks[task['provenance']['parent_task']]
        need(all(task[k] == old[k] for k in ('targets','consequences','split_group','checks','target_origin')),
             'CONTEXT_READBACK_PRIVATE_TARGETS')
        for key in old['context']+old['targets']+old['consequences']:
            need({k:v for k,v in fields[key].items() if k != 'parts'} ==
                 {k:v for k,v in old_fields[key].items() if k != 'parts'}, 'CONTEXT_READBACK_OLD_FIELD')
        need(task['context'][:len(old['context'])] == old['context']
             and not set(task['context']) & set(task['targets']+task['consequences']), 'CONTEXT_READBACK_MASK')
        for key in task['context'][len(old['context']):]:
            need((key,task['targets'][0],'base_context_available_for_patch') in edges,
                 'CONTEXT_READBACK_MISSING_EDGE')
    phases['independent_parent_and_mask_joins'] = time.perf_counter()-tick; tick = time.perf_counter()
    consumer = ContextPopulation(a.population, a.pin)
    task = deepcopy(next(iter(consumer.tasks.values())))
    def changed(**kwargs):
        value = dict(task, **kwargs); value['id'] = digest({k:v for k,v in value.items() if k != 'id'})
        consumer.task(value)
    controls = dict(missing_target=rejected(lambda:changed(targets=[])),
        future_target_in_context=rejected(lambda:changed(context=task['context']+task['targets'])))
    field = consumer.fields[task['context'][-1]]
    for attr, replacement in (('role','recorded_patch'), ('text',field['text']+' corrupted'),
                               ('clock',dict(id='future-execution', wall_time=None))):
        original = field[attr]
        try:
            field[attr] = replacement
            controls['changed_'+attr] = rejected(lambda:consumer.task(task))
        finally: field[attr] = original
    links = consumer.links
    try:
        consumer.links = set()
        controls['dropped_relationship'] = rejected(lambda:consumer.task(task))
    finally: consumer.links = links
    consumer.fresh()
    a.negative.mkdir(parents=True, exist_ok=False)
    altered = a.negative / 'corrupted-publication'
    shutil.copytree(a.population, altered)
    with (altered / 'fields.jsonl').open('ab') as stream: stream.write(b'corrupted\n')
    controls['persisted_corruption'] = rejected(lambda:ContextPopulation(altered,a.pin))
    after = {p:file_sha(Path(p)) for p in before}
    need(before == after, 'CONTEXT_READBACK_ORIGINAL_CHANGED')
    phases['consumer_negative_controls_and_unchanged_sources'] = time.perf_counter()-tick
    wall = time.perf_counter()-start
    result = dict(schema='capability-context-readback-v1', population=a.pin, complete=True,
        author_checked=True, independent_session=False, shared_low_level_git_blob_reader=True,
        snapshots=len(snapshots), tasks=len(rows['tasks']), new_fields=len(new_fields),
        new_roles=dict(Counter(f['role'] for f in new_fields)), witnessed_import_edges=import_count,
        controls=controls, before=before, after=after, phases_seconds=phases, wall_seconds=wall,
        phase_coverage=sum(phases.values())/wall, producer_sha256=file_sha(Path(__file__)),
        scientific_admission=False, learned_capability=False, public_test_execution_in_this_stage=False)
    need(result['phase_coverage'] >= .9, 'CONTEXT_READBACK_TIMING')
    pin = write_committed(a.output,result)
    print(json.dumps(dict(readback_sha256=pin, tasks=result['tasks'], new_fields=len(new_fields), controls=controls)))


def read_pinned_rows(path, pin):
    raw = path.read_bytes(); need(sha(raw) == pin,'CONTEXT_READBACK_PARENT_PIN')
    return {r['id']:r for r in map(json.loads,raw.splitlines())}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('population','negative','output'): p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--pin',required=True)
    run(p.parse_args())
