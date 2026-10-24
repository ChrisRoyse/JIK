"""Persist the real experience delta and test source, task and publication boundaries."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.capability_revision_readback import RevisionReadback
from claims.capability_population import need
from claims.capability_revision_population import RevisionPopulation
from claims.episodic_memory import digest
from claims.process_conditions import write_committed
from text.ami_continuation_tasks import build as dialogue
from text.condaqa_revision_tasks import build as reading
from text.evaluation_multiscale import file_sha


def rejected(action):
    try:
        action()
    except ValueError as exc:
        need(str(exc).startswith('MNLM_'), 'CONTROL_ERROR_CLASS')
        return dict(rejected=True, code=str(exc))
    raise ValueError('MNLM_CAPABILITY_REVISION_CONTROL_ACCEPTED')


def task_controls(pop, reader):
    task = deepcopy(next(iter(pop.tasks.values())))
    def changed(**kwargs):
        value = dict(task, **kwargs)
        value['id'] = digest({k: v for k, v in value.items() if k != 'id'})
        return pop.task(value)
    controls = dict(missing_target=rejected(lambda: changed(targets=[])),
        future_target_in_context=rejected(lambda: changed(context=task['context'] + task['targets'])))
    target = pop.fields[task['targets'][0]]
    original = target['role']
    try:
        target['role'] = 'wrong_role'
        controls['swapped_role'] = rejected(lambda: pop.task(task))
    finally:
        target['role'] = original
    original_links = pop.links
    try:
        pop.links = set()
        controls['dropped_relationship'] = rejected(lambda: pop.task(task))
    finally:
        pop.links = original_links
    bad_field = dict(target, text=target['text'] + ' corrupted')
    controls['source_text_mismatch'] = rejected(lambda: reader.field(bad_field))
    if pop.family == 'reasoning':
        controls['reversed_evidence_direction'] = rejected(lambda: changed(
            context=[task['context'][3], *task['context'][1:3], task['context'][0]]))
    else:
        p = deepcopy(task['provenance'])
        if task['kind'] == 'continue_attested_dialogue':
            p['meeting'] = 'wrong_meeting'
        else:
            p['question_id'] = 'wrong_question'
        controls['cross_episode_join'] = rejected(lambda: changed(provenance=p))
    pop.fresh()
    return controls


def read_delta(config):
    results = {}
    for name, item in config.items():
        pop = RevisionPopulation(item['population'], item['population_pin'])
        reader = RevisionReadback(pop)
        results[name] = dict(source=reader.run(), counts=pop.report['counts'],
                            controls=task_controls(pop, reader))
    return results


def build(args):
    start = tick = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=False)
    before = {str(p): file_sha(p) for p in args.preserve}
    phases = dict(source_state_before=time.perf_counter() - tick)
    tick = time.perf_counter()
    ami = dialogue(args.raw_root / 'ami-event-v1/ami_public_manual_1.6.2.zip', args.output / 'dialogue')
    conda = reading(args.raw_root / 'condaqa-training-v1', args.output / 'condaqa')
    write_committed(args.output / 'source-builds.json', dict(ami=ami, condaqa=conda))
    config = dict(dialogue=dict(population=str(args.output / 'dialogue'),
        population_pin=ami['population']['report_sha256']))
    for family, result in conda['populations'].items():
        config['condaqa_' + family] = dict(population=str(args.output / 'condaqa' / family),
                                         population_pin=result['report_sha256'])
    write_committed(args.output / 'config.json', config)
    phases['source_adapters_and_publication'] = time.perf_counter() - tick
    tick = time.perf_counter()
    readback = read_delta(config)
    phases['original_source_readback_and_task_controls'] = time.perf_counter() - tick
    tick = time.perf_counter()
    controls = {}
    for name, item in config.items():
        altered = args.output / 'negative-controls' / name
        shutil.copytree(item['population'], altered)
        with (altered / 'fields.jsonl').open('ab') as stream:
            stream.write(b'corrupted\n')
        controls[name] = rejected(lambda: RevisionPopulation(altered, item['population_pin']))
    phases['publication_corruption_controls'] = time.perf_counter() - tick
    tick = time.perf_counter()
    after = {str(p): file_sha(p) for p in args.preserve}
    need(before == after, 'REVISION_EXISTING_STATE_CHANGED')
    files = {str(p.relative_to(args.output)): dict(bytes=p.stat().st_size, sha256=file_sha(p))
             for p in sorted(args.output.rglob('*')) if p.is_file()}
    phases['preserved_state_and_file_census'] = time.perf_counter() - tick
    wall = time.perf_counter() - start
    report = dict(schema='capability-experience-delta-fsv-v1', complete=True, author_checked=True,
        config=config, readback=readback, corruption_controls=controls, before=before, after=after,
        files=files, phases_seconds=phases, wall_seconds=wall, phase_coverage=sum(phases.values()) / wall,
        producer_sha256=file_sha(Path(__file__)), scientific_admission=False, training_ready=False,
        learned_capability=False, sealed_evaluation=False, native_measurement_complete=False,
        global_split_registration_complete=False, live_kernel_integration_complete=False,
        source_relative_judgments_independently_adjudicated=False)
    need(report['phase_coverage'] >= .9, 'REVISION_TIMING_COVERAGE')
    pin = write_committed(args.output / 'report.json', report)
    print(json.dumps(dict(report_sha256=pin, counts={k: v['counts'] for k, v in readback.items()},
                          wall_seconds=wall, phase_coverage=report['phase_coverage'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--raw-root', type=Path)
    parser.add_argument('--preserve', nargs='*', type=Path, default=[])
    parser.add_argument('--read-config', type=Path)
    parser.add_argument('--readback-output', type=Path)
    args = parser.parse_args()
    if args.read_config:
        result = read_delta(json.loads(args.read_config.read_bytes()))
        pin = write_committed(args.readback_output, dict(config_sha256=file_sha(args.read_config),
            results=result, author_checked=True, independent_session=False))
        print(json.dumps(dict(readback_sha256=pin, populations=len(result))))
    else:
        need(args.output is not None and args.raw_root is not None, 'REVISION_ARGUMENTS')
        build(args)
