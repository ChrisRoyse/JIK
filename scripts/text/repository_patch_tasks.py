"""Expose retained repository patches as targets, with observed failure/repair contrasts."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from claims.repository_cohort_population import CohortPopulation
from claims.repository_depth_population import DepthPopulation
from claims.repository_demonstration_population import DemonstrationPopulation
from text.capability_tasks import TaskBuilder, check, text_sha
from text.evaluation_multiscale import file_sha

CONFIG_PIN = '19834383f447d460d74fdebbdf99fcef9bd63e3a4fe7a6504cb89bf8d0c9997c'
POPULATIONS = {'repository_cohort_intake': CohortPopulation,
               'repository_depth_intake': DepthPopulation,
               'repository_demonstration_intake': DemonstrationPopulation}


def build(config_path, output):
    tick = time.perf_counter()
    check(file_sha(config_path) == CONFIG_PIN, 'PATCH_CONFIG_PIN')
    config = json.loads(config_path.read_bytes())
    source = dict(config=str(config_path), config_sha256=CONFIG_PIN,
                  populations=config['populations'], encoder_exposure_certified=False,
                  producer_sha256=file_sha(Path(__file__)),
                  licence='inherits source-specific retained repository and dataset licences',
                  use_scope='local development; original source terms remain binding')
    out = TaskBuilder('coding', source)
    examples, per_task, source_count = {}, defaultdict(list), 0
    for record in config['populations']:
        native = record['native']
        pop = POPULATIONS[record['kind']](native['population'], native['population_pin'])
        source_count += len(pop.fields)
        for key, spec in pop.fields.items():
            check(key not in examples, 'PATCH_DUPLICATE_CONDITION')
            roles, consequences = defaultdict(list), []
            for side in ('input', 'echo'):
                for ordinal, field in enumerate(spec[side]):
                    if field['empty']:
                        continue
                    raw = b''.join(pop.rows[p['unit']]['text'].encode() for p in field['parts'])
                    check(len(raw) == field['bytes'] and text_sha(raw.decode()) == field['sha256'],
                          'PATCH_FIELD_CONSERVATION')
                    address = dict(population=native['population'], population_pin=native['population_pin'],
                        condition=key, side=side, field_index=ordinal, original_source=field['source'],
                        original_span=field['source_span'], native_parts=field['parts'],
                        native_path=native['native'], native_pin=native['native_pin'])
                    if not raw.decode().strip():
                        out.gaps.append(dict(condition=key, reason='unmeasurable_whitespace_field',
                                             role=field['role'], bytes=len(raw), sha256=field['sha256'],
                                             source=address, original_retained=True))
                        continue
                    node = out.field(raw.decode(), field['role'], address,
                        clock=dict(id='recorded-process:' + key,
                                   available=spec['input_available_by' if side == 'input'
                                                  else 'echo_available_after'],
                                   input_clock_includes_patch=True))
                    roles[field['role']].append(node)
                    if side == 'echo': consequences.append(node)
            entry = pop.observations.entries[key]
            origin = entry.get('action_origin', 'recorded_source_attempt')
            context = roles['task_report'] + roles['original_source']
            targets = roles['recorded_patch']
            after = roles['candidate_source'] + consequences
            provenance = dict(condition=key, task=spec['task'], population=native['population_pin'],
                action_origin=origin, observed_outcome=entry['outcome'],
                execution_evidence=entry, source_gaps=spec['missing'],
                source_context_selection='patch-touched files; localization supplied, not autonomous search',
                excluded_oracle_paths=spec['excluded_oracle_paths'],
                conditioning_excludes_current_candidate_patch_and_echo=True)
            if not targets or not roles['original_source']:
                out.gaps.append(dict(condition=key, reason='missing_patch_or_original_source',
                                     source=provenance))
                continue
            check(len(targets) == 1 and roles['task_report'], 'PATCH_ROLES')
            for node in context:
                out.edge(node, targets[0], 'recorded_task_to_patch', provenance)
            for node in after:
                out.edge(targets[0], node, 'recorded_patch_consequence', provenance)
            out.task(context=context, targets=targets, consequences=after,
                group='repository-task:' + spec['task'], kind='reproduce_recorded_patch',
                provenance=provenance, checks=[dict(scope='retained_execution', outcome=entry['outcome'],
                                                  receipt_population=native['population_pin'])],
                target_origin=origin)
            examples[key] = dict(roles=roles, context=context, targets=targets, after=after,
                                 provenance=provenance, origin=origin, outcome=entry['outcome'],
                                 task=spec['task'], reference=record['kind']=='repository_demonstration_intake')
            per_task[spec['task']].append(key)
    contrasts = 0
    # A contrast is a documented teaching relation, not an invented chronological repair event.
    for task, keys in per_task.items():
        failures = [examples[k] for k in keys if examples[k]['outcome'] == 0]
        references = [examples[k] for k in keys if examples[k]['reference'] and examples[k]['outcome'] == 1]
        for repair in references:
            for failure in failures:
                context = list(dict.fromkeys(failure['context'] + failure['targets'] + failure['after']))
                if set(context) & set(repair['targets']):
                    out.gaps.append(dict(task=task, reason='repair_already_present_in_failure'))
                    continue
                provenance = dict(failure=failure['provenance'], repair=repair['provenance'],
                    relation='same repository task, observed failed attempt and supplied passing repair',
                    autonomous_correction=False, historical_repair_sequence_established=False)
                for a in failure['targets']:
                    for b in repair['targets']:
                        out.edge(a, b, 'supplied_passing_repair_contrast', provenance)
                out.task(context=context, targets=repair['targets'], consequences=repair['after'],
                    group='repository-task:' + task, kind='contrast_failure_with_supplied_repair',
                    provenance=provenance, target_origin=repair['origin'],
                    checks=[dict(scope='paired_retained_execution', failed_outcome=0, repair_outcome=1)])
                contrasts += 1
    result = out.publish(output, scope='retained patches and real execution observations; supplied repair contrasts',
        coverage=dict(source_conditions=source_count, target_tasks=len(examples),
            outcomes=dict(Counter(e['outcome'] for e in examples.values())),
            supplied_repairs=sum(e['reference'] for e in examples.values()), failure_repair_contrasts=contrasts,
            independently_observed_execution=True, autonomous_code_generation=False,
            test_source_in_conditioning=False, api_and_test_context_coverage='existing source gaps retained',
            split_assignment='development exposed; all attempts of one task must stay grouped'))
    print(json.dumps(dict(result, wall_seconds=time.perf_counter() - tick)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build(args.config, args.output)
