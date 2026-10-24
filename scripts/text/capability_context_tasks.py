"""Attach retained repository API/public-test context to existing real coding episodes."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from claims.capability_population import CapabilityPopulation, need
from claims.capability_context_population import ContextPopulation, ROLES
from claims.repository_context import POLICY, RepositoryContext
from claims.process_conditions import write_committed
from text.capability_tasks import TaskBuilder
from text.evaluation_multiscale import file_sha


def build(args):
    start = tick = time.perf_counter()
    parent = CapabilityPopulation(args.parent, args.parent_pin)
    context = RepositoryContext(args.census, args.census_pin, args.source_root, parent)
    phases = dict(source_binding_and_static_graph=time.perf_counter()-tick); tick = time.perf_counter()
    source = dict(adapter='repository-context-population-v1', policy=POLICY,
        parent=dict(path=str(parent.root), pin=parent.pin),
        census=dict(path=str(args.census), pin=args.census_pin), source_root=str(args.source_root),
        licence='retained source-specific repository and dataset terms; no relicensing',
        code_pins={n:file_sha(Path(__file__).resolve().parents[1] / n) for n in (
            'claims/repository_context.py', 'claims/capability_context_population.py',
            'text/capability_context_tasks.py')})
    builder = TaskBuilder('coding', source)
    for key, f in parent.fields.items():
        need(builder.field(f['text'], f['role'], f['source'], clock=f['clock']) == key,
             'CONTEXT_ORIGINAL_ID')
    for edge in parent.edges: builder.edge(**edge)
    builder.gaps.extend(json.loads(x) for x in parent.files['gaps.jsonl'].splitlines())
    additions, selected_bytes, selected_files = {}, 0, 0
    for name, selection in context.selections.items():
        ids = []
        for path in sorted(selection['selected']):
            raw = context.files[name][path]
            ids.append(builder.field(raw.decode(), ROLES[context.catalogs[name][path]['kind']],
                       context.address(name, path), clock=context.clock(name)))
            selected_bytes += len(raw); selected_files += 1
        additions[name] = ids
        builder.gaps.extend(dict(task=name, **gap) for gap in selection['gaps'])
    for old in parent.tasks.values():
        name = old['split_group'].removeprefix('repository-task:')
        extra = additions[name]
        for key in extra:
            path = builder.fields[key]['source']['path']
            builder.edge(key, old['targets'][0], 'base_context_available_for_patch',
                         dict(parent_task=old['id'], base_commit=context.rows[name]['base_commit'],
                              reasons=context.selections[name]['selected'][path],
                              available_context_is_not_a_correctness_label=True))
        builder.task(context=old['context']+extra, targets=old['targets'], consequences=old['consequences'],
            group=old['split_group'], kind='repository_context_to_recorded_patch',
            provenance=dict(parent_population=parent.pin, parent_task=old['id'], parent_kind=old['kind'],
                context_policy=POLICY, localization='inherited patch-touched original paths; supplied, not autonomous',
                benchmark_test_patch_read=False, current_outcome_used_for_selection=False),
            checks=old['checks'], target_origin=old['target_origin'])
    phases['field_sharing_and_task_links'] = time.perf_counter()-tick; tick = time.perf_counter()
    coverage = dict(selected_base_files=selected_files, selected_base_bytes=selected_bytes,
        parent_tasks=len(parent.tasks), original_targets_and_consequences_preserved=True,
        roles=dict(Counter(builder.fields[k]['role'] for values in additions.values() for k in values)),
        selections=context.selections, public_tests_executed_by_this_stage=False,
        inherited_observed_execution=True, autonomous_code_generation=False,
        source_context_selection='supplied original localization plus static imports and base suite/documentation',
        groups_with_public_test_context=sum(any(builder.fields[k]['role']=='public_base_test' for k in keys)
                                            for keys in additions.values()))
    result = builder.publish(args.output, scope='base-commit context attached to attested patch/execution tasks',
                             coverage=coverage)
    phases['publication'] = time.perf_counter()-tick; tick = time.perf_counter()
    consumer = ContextPopulation(args.output, result['report_sha256'])
    need(len(consumer.tasks) == len(parent.tasks), 'CONTEXT_CONSUMER_ROSTER')
    phases['strict_consumer_readback'] = time.perf_counter()-tick
    wall = time.perf_counter()-start
    receipt = dict(**result, selected_base_files=selected_files, selected_base_bytes=selected_bytes,
        groups_with_public_test_context=coverage['groups_with_public_test_context'],
        number_status='measured', acceptance='author-checked', phases_seconds=phases, wall_seconds=wall,
        phase_coverage=sum(phases.values())/wall, native_windows='unmeasured',
        metric_mapping=['T1/spec27 source provenance', 'T3/spec27 capability context coverage'])
    need(receipt['phase_coverage'] >= .9, 'CONTEXT_TIMING')
    write_committed(args.receipt, receipt)
    print(json.dumps(receipt))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('parent', 'census', 'source-root', 'output', 'receipt'):
        parser.add_argument('--'+name, type=Path, required=True)
    for name in ('parent-pin', 'census-pin'): parser.add_argument('--'+name, required=True)
    build(parser.parse_args())
